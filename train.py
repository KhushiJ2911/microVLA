"""Tiny pi0.5: joint VLM/action attention and time-conditioned flow matching."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32
    horizon: int = 10
    action_dim: int = 3
    state_dim: int = 3
    width: int = 256
    depth: int = 6
    heads: int = 4
    head_dim: int = 64
    mlp: int = 1024
    expert_width: int = 128
    expert_mlp: int = 512
    vit_width: int = 192
    vit_depth: int = 4
    vit_heads: int = 3


class RMSNorm(nn.Module):
    def __init__(self, width: int, adaptive: bool = False):
        super().__init__()
        self.modulation = nn.Linear(width, 3 * width) if adaptive else None
        if adaptive:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        if self.modulation is None:
            return (y * (1 + self.scale)).to(x.dtype), None
        scale, shift, gate = self.modulation(cond)[:, None].chunk(3, -1)
        return (y * (1 + scale) + shift).to(x.dtype), gate


class GeGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up(x).chunk(2, -1)
        return self.down(F.gelu(gate, approximate="tanh") * value)


class Vision(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.patch = nn.Conv2d(3, c.vit_width, 8, stride=8)
        self.position = nn.Parameter(torch.randn(1, 64, c.vit_width) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(c.vit_width, c.vit_heads, 4 * c.vit_width,
                                       dropout=0, activation="gelu", batch_first=True,
                                       norm_first=True) for _ in range(c.vit_depth)
        ])
        self.norm = nn.LayerNorm(c.vit_width)
        self.project = nn.Linear(c.vit_width, c.width)

    def forward(self, images: Tensor) -> Tensor:
        x = images.float().div(127.5).sub(1).contiguous(memory_format=torch.channels_last)
        x = self.patch(x).flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            x = block(x)
        return self.project(self.norm(x))


def attention_mask(prefix_valid: Tensor, horizon: int) -> Tensor:
    """True means allowed; action tokens form one bidirectional suffix block."""
    b, p = prefix_valid.shape
    valid = F.pad(prefix_valid, (0, horizon), value=True)
    group = torch.arange(p + horizon, device=valid.device) >= p
    allowed = group[:, None] >= group[None, :]
    return (allowed[None] & valid[:, None, :])[:, None]  # mask keys only: an all-False query row gives NaN


def rope(x: Tensor, positions: Tensor) -> Tensor:
    freq = 10000 ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    angle = positions[:, None, :, None].float() * freq
    left, right = x.float().chunk(2, -1)
    return torch.cat((left * angle.cos() - right * angle.sin(),
                      right * angle.cos() + left * angle.sin()), -1).to(x.dtype)


class Expert(nn.Module):
    def __init__(self, width: int, hidden: int, c: Config, adaptive: bool):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.head_dim
        self.attn_norm = RMSNorm(width, adaptive)
        self.ffn_norm = RMSNorm(width, adaptive)
        self.qkv = nn.Linear(width, (c.heads + 2) * c.head_dim, bias=False)
        self.out = nn.Linear(c.heads * c.head_dim, width, bias=False)
        self.ffn = GeGLU(width, hidden)

    def prepare(self, x: Tensor, pos: Tensor, cond: Tensor | None) -> tuple:
        normed, gate = self.attn_norm(x, cond)
        q, k, v = self.qkv(normed).split([self.heads * self.head_dim, self.head_dim, self.head_dim], -1)
        q = q.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        k, v = k[:, None], v[:, None]
        return rope(q, pos), rope(k, pos), v, gate

    def finish(self, x: Tensor, attended: Tensor, gate: Tensor | None, cond: Tensor | None) -> Tensor:
        update = self.out(attended.transpose(1, 2).flatten(2))
        x = x + (update if gate is None else gate * update)
        normed, gate = self.ffn_norm(x, cond)
        update = self.ffn(normed)
        return x + (update if gate is None else gate * update)


class JointBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.prefix = Expert(c.width, c.mlp, c, False)
        self.action = Expert(c.expert_width, c.expert_mlp, c, True)

    def forward(self, p: Tensor | None, a: Tensor | None, mask: Tensor,
                ppos: Tensor, apos: Tensor, cond: Tensor | None, cache: tuple | None = None) -> tuple:
        parts = []
        if p is not None:
            parts.append(self.prefix.prepare(p, ppos, None))
        if a is not None:
            parts.append(self.action.prepare(a, apos, cond))
        q, k, v = (torch.cat([part[i] for part in parts], dim=2) for i in range(3))
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        attended = F.scaled_dot_product_attention(q, k.expand(-1, q.shape[1], -1, -1),
                                                  v.expand(-1, q.shape[1], -1, -1), attn_mask=mask)
        offset = 0
        if p is not None:
            offset = p.shape[1]
            p = self.prefix.finish(p, attended[:, :, :offset], parts[0][3], None)
        if a is not None:
            a = self.action.finish(a, attended[:, :, offset:], parts[-1][3], cond)
        return p, a, (k, v)


class MicroVLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vision = Vision(config)
        self.tokens = nn.Embedding(config.vocab_size + 256 * config.state_dim, config.width)
        nn.init.normal_(self.tokens.weight, std=0.02)
        self.layers = nn.ModuleList([JointBlock(config) for _ in range(config.depth)])
        w = config.expert_width
        self.time_mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU())
        self.action_in = nn.Linear(config.action_dim, w)
        self.final_norm = RMSNorm(w, True)
        self.action_out = nn.Linear(w, config.action_dim)

    def embed_prefix(self, images: Tensor, state: Tensor, instr: Tensor) -> tuple[Tensor, Tensor]:
        c = self.config
        bins = ((state.clamp(-1, 1) + 1) * 128).long().clamp(max=255)
        ids = bins + c.vocab_size + 256 * torch.arange(c.state_dim, device=state.device)
        words = torch.cat((instr, ids), 1)
        p = torch.cat((self.vision(images), self.tokens(words) * math.sqrt(c.width)), 1)
        valid = torch.cat((torch.ones_like(p[:, :64, 0], dtype=torch.bool), instr != 0,
                           torch.ones_like(ids, dtype=torch.bool)), 1)
        return p, valid

    def time_condition(self, t: Tensor) -> Tensor:
        fraction = torch.linspace(0, 1, self.config.expert_width // 2, device=t.device)
        angle = t[:, None] * (2 * math.pi / (0.004 * 1000 ** fraction))
        return self.time_mlp(torch.cat((angle.sin(), angle.cos()), -1))

    def forward(self, images: Tensor, state: Tensor, instr: Tensor, noisy: Tensor, t: Tensor) -> Tensor:
        p, valid = self.embed_prefix(images, state, instr)
        a, cond = self.action_in(noisy), self.time_condition(t)
        pos = F.pad(valid, (0, noisy.shape[1]), value=True).long().cumsum(1) - 1
        mask = attention_mask(valid, noisy.shape[1])
        for layer in self.layers:
            p, a, _ = layer(p, a, mask, pos[:, :p.shape[1]], pos[:, p.shape[1]:], cond)
        return self.action_out(self.final_norm(a, cond)[0])

    def prefix_cache(self, images: Tensor, state: Tensor, instr: Tensor) -> tuple:
        p, valid = self.embed_prefix(images, state, instr)
        pos, cache = valid.long().cumsum(1) - 1, []
        mask = attention_mask(valid, 0)
        for layer in self.layers:
            p, _, kv = layer(p, None, mask, pos, pos[:, :0], None)
            cache.append(kv)
        return cache, valid

    def cached_velocity(self, noisy: Tensor, t: Tensor, cache: list, valid: Tensor) -> Tensor:
        a, cond = self.action_in(noisy), self.time_condition(t)
        pos = valid.sum(1)[:, None] + torch.arange(noisy.shape[1], device=noisy.device)
        mask = F.pad(valid, (0, noisy.shape[1]), value=True)[:, None, None, :]
        for layer, kv in zip(self.layers, cache):
            _, a, _ = layer(None, a, mask, pos[:, :0], pos, cond, kv)
        return self.action_out(self.final_norm(a, cond)[0])

    @torch.no_grad()
    def sample(self, images: Tensor, state: Tensor, instr: Tensor, num_steps: int = 10,
               noise: Tensor | None = None, cached: bool = True) -> Tensor:
        x = torch.randn(state.shape[0], self.config.horizon, self.config.action_dim,
                        device=state.device) if noise is None else noise.clone()
        if cached:
            cache, valid = self.prefix_cache(images, state, instr)
        for step in range(num_steps):
            t = torch.full((x.shape[0],), 1 - step / num_steps, device=x.device)
            v = self.cached_velocity(x, t, cache, valid) if cached else self(images, state, instr, x, t)
            x = x - v / num_steps
        return x

    def loss(self, images: Tensor, state: Tensor, instr: Tensor, actions: Tensor,
             noise: Tensor | None = None, t: Tensor | None = None) -> Tensor:
        noise = torch.randn_like(actions) if noise is None else noise
        # Inverse CDF of Beta(1.5, 1) avoids a CPU distribution sampler.
        t = torch.rand(actions.shape[0], device=actions.device).pow(2 / 3) * 0.999 + 0.001 if t is None else t
        noisy = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        return F.mse_loss(self(images, state, instr, noisy, t).float(), noise - actions)


def parameter_counts(model: MicroVLA) -> dict[str, int]:
    counts = {"vision": sum(p.numel() for p in model.vision.parameters()),
              "backbone": sum(p.numel() for layer in model.layers for p in layer.prefix.parameters()),
              "action_expert": sum(p.numel() for layer in model.layers for p in layer.action.parameters())}
    counts["embeddings_and_heads"] = sum(p.numel() for p in model.parameters()) - sum(counts.values())
    return {**counts, "total": sum(counts.values())}


def autocast(device: torch.device) -> torch.autocast | nullcontext:
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def benchmark(device: torch.device, steps: int) -> None:
    for batch in (64, 128, 256):
        torch.manual_seed(0)
        model = MicroVLA(Config()).to(device).to(memory_format=torch.channels_last)
        if batch == 64:
            print(json.dumps(parameter_counts(model)))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        images = torch.randint(256, (batch, 3, 64, 64), dtype=torch.uint8, device=device)
        state, instr = torch.zeros(batch, 3, device=device), torch.ones(batch, 8, dtype=torch.long, device=device)
        actions = torch.randn(batch, 10, 3, device=device)
        def step():
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                loss = model.loss(images, state, instr, actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        for _ in range(2):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(json.dumps({"device": str(device), "batch": batch, "steps/s": steps / elapsed,
                          "samples/s": steps * batch / elapsed,
                          "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}))
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()


def demo(sim, args, device: torch.device) -> None:
    paths = sorted(Path(args.runs).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    selected = [paths[0], paths[len(paths) // 2], paths[-1]]
    clips, labels = [], []
    for path in selected:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = MicroVLA(Config(**ckpt["config"])).to(device).eval()
        model.load_state_dict(ckpt["model"])
        def policy(images, state, instr):
            with autocast(device):
                return model.sample(images, state, instr, args.flow_steps)
        torch.manual_seed(args.seed)
        result = sim.evaluate(policy, 3, args.seed, device, record=3)
        clips.append(result["frames"])
        labels.extend([f"step {ckpt['step']} / episode {i + 1}" for i in range(3)])
    Path("assets").mkdir(exist_ok=True)
    length = max(clip.shape[0] for clip in clips)
    clips = [torch.cat((clip, clip[-1:].expand(length - len(clip), -1, -1, -1, -1))) for clip in clips]
    sim.save_gif(torch.cat(clips, 1), "assets/training.gif", labels=labels)
    clips = []
    for swap in (False, True):
        torch.manual_seed(args.seed)
        clips.append(sim.evaluate(policy, 1, args.seed, device, swap=swap, record=1)["frames"])
    length = max(len(clip) for clip in clips)
    clips = [torch.cat((clip, clip[-1:].expand(length - len(clip), -1, -1, -1, -1))) for clip in clips]
    sim.save_gif(torch.cat(clips, 1), "assets/language_swap.gif", labels=["original instruction", "swapped instruction"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.3, help="DART noise std on executed expert actions")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--logdir", default="runs/tb", help="TensorBoard log dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.benchmark:
        benchmark(device, args.benchmark_steps)
        return
    import sim
    if args.demo:
        demo(sim, args, device)
        return
    c = Config(vocab_size=len(sim.VOCAB), action_dim=sim.ACTION_DIM, state_dim=sim.STATE_DIM)
    model = MicroVLA(c).to(device).to(memory_format=torch.channels_last)
    print(json.dumps(parameter_counts(model)))
    data = sim.collect(args.episodes, c.horizon, args.noise, args.seed, device)
    # Validation: unseen episodes (different seed), fixed noise and t so the curve is low-variance.
    val = sim.collect(max(1, args.episodes // 10), c.horizon, args.noise, args.seed + 1, device)
    vidx = torch.randperm(len(val["scene"]), device=device)[:1024]
    val = {k: v[vidx] for k, v in val.items()}
    val["noise"] = torch.randn_like(val["actions"])
    val["t"] = torch.rand(len(vidx), device=device).pow(2 / 3) * 0.999 + 0.001
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.logdir, flush_secs=10)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        with autocast(device):
            s = val["scene"]
            loss = model.loss(sim.render(s), sim.robot_state(s), sim.encode_instruction(val["task"]),
                              val["actions"], val["noise"], val["t"])
        model.train()
        return loss.item()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.compile:
        model.forward = torch.compile(model.forward)
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    metrics = []
    def checkpoint(step: int) -> None:
        model.eval()
        def policy(images, state, instr):
            with autocast(device):
                return model.sample(images, state, instr, args.flow_steps)
        row = {"step": step}
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            for name, split, swap in (("train", "train", False), ("heldout", "heldout", False), ("swap", "train", True)):
                torch.manual_seed(args.seed + 1)
                row[name] = sim.evaluate(policy, args.eval_episodes, args.seed + 1, device, split=split, swap=swap)["success"]
        metrics.append(row)
        print(json.dumps(row))
        for name in ("train", "heldout", "swap"):
            writer.add_scalar(f"success/{name}", row[name], step)
        torch.save({"config": asdict(c), "model": model.state_dict(), "step": step}, runs / f"ckpt_{step}.pt")
        (runs / "metrics.json").write_text(json.dumps(metrics, indent=2))
        model.train()
    fixed = torch.randint(len(data["scene"]), (args.batch_size,), device=device)
    fixed_noise = torch.randn(args.batch_size, c.horizon, c.action_dim, device=device)
    fixed_t = torch.rand(args.batch_size, device=device) * 0.998 + 0.001
    checkpoint(0)
    start, accumulated = time.perf_counter(), torch.zeros((), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        idx = fixed if args.overfit else torch.randint(len(data["scene"]), (args.batch_size,), device=device)
        scene = data["scene"][idx]
        images, state, instr = sim.render(scene), sim.robot_state(scene), sim.encode_instruction(data["task"][idx])
        actions = data["actions"][idx]
        opt.zero_grad(set_to_none=True)
        with autocast(device):
            loss = model.loss(images, state, instr, actions, *((fixed_noise, fixed_t) if args.overfit else ()))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        warmup = min(100, max(1, args.steps // 10))
        lr = args.lr * min(step / warmup, 1) * (0.1 + 0.9 * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, args.steps - warmup))) / 2)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        accumulated += loss.detach()
        if step % 100 == 0 or step == args.steps:
            mean_loss = accumulated.item() / (100 if step % 100 == 0 else step % 100)
            elapsed = time.perf_counter() - start
            count = 100 if step % 100 == 0 else step % 100
            row = {"step": step, "loss": mean_loss, "val_loss": val_loss(), "lr": lr, "steps/s": count / elapsed,
                   "samples/s": count * args.batch_size / elapsed,
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0}
            print(json.dumps(row))
            for key, tag in (("loss", "loss/train"), ("val_loss", "loss/val"), ("lr", "opt/lr"),
                             ("steps/s", "perf/steps_per_s"), ("samples/s", "perf/samples_per_s"),
                             ("peak_vram_mb", "perf/peak_vram_mb")):
                writer.add_scalar(tag, row[key], step)
            metrics.append(row)
            accumulated.zero_()
            start = time.perf_counter()
        if step % args.eval_every == 0 or step in (args.steps // 2, args.steps):
            eval_start = time.perf_counter()
            checkpoint(step)
            start += time.perf_counter() - eval_start


if __name__ == "__main__":
    main()
