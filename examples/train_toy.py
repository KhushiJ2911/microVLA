"""Tiny pi0.5: joint VLM/action attention and time-conditioned flow matching."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from train import Config, MicroVLA, autocast, make_logger, parameter_counts


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
        tasks = sim.reset(3, args.seed + sim.EVAL_SEED_OFFSET, device)[:, 17:19]
        labels.extend([f"step {ckpt['step']}: {sim.instruction_text(t)}" for t in tasks])
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
    scene = sim.reset(1, args.seed + sim.EVAL_SEED_OFFSET, device)
    swapped = scene[:, 17:19].clone()
    swapped[:, 0] = torch.where(scene[:, 7] == scene[:, 17], scene[:, 8], scene[:, 7])
    sim.save_gif(torch.cat(clips, 1), "assets/language_swap.gif",
                 labels=[sim.instruction_text(scene[0, 17:19]), sim.instruction_text(swapped[0])])


def distill_vision(model: MicroVLA, sim, device: torch.device, steps: int, path: Path, writer) -> None:
    """Bootstrap the ViT from frozen SigLIP-B/16, as pi0.5 starts from a pretrained SigLIP.

    A from-scratch ViT fails to localize the small objects; matching SigLIP's patch features fixes that.
    The teacher sees the render upsampled to 128 px, which gives the same 8x8 patch grid as the student.
    """
    if path.exists():
        model.vision.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return
    from transformers import SiglipVisionModel
    teacher = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.bfloat16)
    teacher = teacher.to(device).eval().requires_grad_(False)
    head = nn.Linear(model.config.width, teacher.config.hidden_size).to(device)
    params = [*model.vision.parameters(), *head.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scenes = sim.collect(4000, 1, 0.3, 1234, device)["scene"]
    for step in range(1, steps + 1):
        images = sim.render(scenes[torch.randint(len(scenes), (128,), device=device)])
        with torch.no_grad():
            x = F.interpolate(images.float() / 127.5 - 1, size=128, mode="nearest").bfloat16()
            target = teacher(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state.float()
            target = F.layer_norm(target, target.shape[-1:])
        for group in opt.param_groups:
            group["lr"] = 1e-3 * min(step / 200, 1) * 0.5 * (1 + math.cos(math.pi * step / steps))
        with autocast(device):
            pred = head(model.vision(images)).float()
        loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            writer.add_scalar("distill/loss", loss.item(), step)
        if step % 250 == 0:
            print(json.dumps({"distill_step": step, "distill_loss": loss.item()}), flush=True)
    torch.save(model.vision.state_dict(), path)
    del teacher, head, opt
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.3, help="DART noise std on executed expert actions")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distill-steps", type=int, default=3000, help="SigLIP->ViT distillation; 0 disables")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --runs")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--logdir", default="runs/tb", help="TensorBoard log dir")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "none"), default="tensorboard")
    parser.add_argument("--wandb-project", default="microvla")
    parser.add_argument("--wandb-run-name")
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
    from examples import sim
    if args.demo:
        demo(sim, args, device)
        return
    c = Config(vocab_size=len(sim.VOCAB), action_dim=sim.ACTION_DIM, state_dim=sim.STATE_DIM)
    model = MicroVLA(c).to(device).to(memory_format=torch.channels_last)
    print(json.dumps(parameter_counts(model)))
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    writer = make_logger(args, settings, c, args.logdir)
    try:
        if args.distill_steps:
            distill_vision(model, sim, device, args.distill_steps, runs / "vit_distilled.pt", writer)
        data = sim.collect(args.episodes, c.horizon, args.noise, args.seed, device)
        # Validation: unseen episodes (different seed), fixed noise and t so the curve is low-variance.
        val = sim.collect(max(1, args.episodes // 10), c.horizon, args.noise, args.seed + 1, device)
        vidx = torch.randperm(len(val["scene"]), device=device)[:1024]
        val = {k: v[vidx] for k, v in val.items()}
        val["noise"] = torch.randn_like(val["actions"])
        val["t"] = torch.rand(len(vidx), device=device).pow(2 / 3) * 0.999 + 0.001

        @torch.no_grad()
        def validate() -> tuple[float, float]:
            """Val flow loss, and cosine between sampled and expert first moves (learns long before success)."""
            model.eval()
            s, obs = val["scene"], (sim.render(val["scene"]), sim.robot_state(val["scene"]), sim.encode_instruction(val["task"]))
            with autocast(device), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                loss = model.loss(*obs, val["actions"], val["noise"], val["t"])
                torch.manual_seed(0)
                moves = model.sample(*obs, args.flow_steps)[:, 0, :2].float()
            model.train()
            return loss.item(), F.cosine_similarity(moves, val["actions"][:, 0, :2]).mean().item()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        metrics, first_step = [], 1
        if args.resume and (runs / "latest.pt").exists():
            ckpt = torch.load(runs / "latest.pt", map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            first_step = ckpt["step"] + 1
            metrics = [m for m in json.loads((runs / "metrics.json").read_text()) if m["step"] <= ckpt["step"]]
            print(f"resumed from step {ckpt['step']}")

        def save(step: int, path: Path) -> None:
            state = {"config": asdict(c), "settings": settings, "model": model.state_dict(), "opt": opt.state_dict(), "step": step}
            torch.save(state, runs / "latest.tmp")
            (runs / "latest.tmp").replace(runs / "latest.pt")  # atomic: a crash mid-write never corrupts latest.pt
            if path.name != "latest.pt":
                torch.save(state, path)
            (runs / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if args.compile:
            model.forward = torch.compile(model.forward)
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
            save(step, runs / f"ckpt_{step}.pt")
            model.train()
        fixed = torch.randint(len(data["scene"]), (args.batch_size,), device=device)
        fixed_noise = torch.randn(args.batch_size, c.horizon, c.action_dim, device=device)
        fixed_t = torch.rand(args.batch_size, device=device) * 0.998 + 0.001
        if first_step == 1:
            checkpoint(0)
        start, accumulated = time.perf_counter(), torch.zeros((), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(first_step, args.steps + 1):
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
                vloss, vcos = validate()
                row = {"step": step, "loss": mean_loss, "val_loss": vloss, "val_dir_cos": vcos, "lr": lr,
                       "steps/s": count / elapsed, "samples/s": count * args.batch_size / elapsed,
                       "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0}
                print(json.dumps(row), flush=True)
                for key, tag in (("loss", "loss/train"), ("val_loss", "loss/val"), ("val_dir_cos", "val/direction_cosine"), ("lr", "opt/lr"),
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
            elif step % 250 == 0:
                save(step, runs / "latest.pt")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
