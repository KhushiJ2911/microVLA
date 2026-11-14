"""Disk-bounded LIBERO-Spatial preprocessing, SigLIP distillation and MicroVLA training."""
import argparse
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import mmap
import os
from pathlib import Path
import resource
import shutil
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def make_logger(args, settings, model_config, logdir):
    """Keep scalar logging identical across backends; persist W&B identity beside checkpoints."""
    if args.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(logdir), flush_secs=10)
    if args.logger == "none":
        return SimpleNamespace(add_scalar=lambda tag, value, step: None, close=lambda: None)
    import wandb
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / "wandb_run_id"
    run_id = path.read_text().strip() if args.resume and path.exists() else None
    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                     config={**settings, "model_config": asdict(model_config)}, dir=str(runs),
                     id=run_id, resume="allow" if args.resume else None)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(run.id)
    tmp.replace(path)
    return SimpleNamespace(add_scalar=lambda tag, value, step: run.log({tag: value}, step=step),
                           close=run.finish)


@dataclass
class Config:
    image_size: int = 64
    num_cameras: int = 1
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
        self.position = nn.Parameter(torch.randn(1, (c.image_size // 8) ** 2, c.vit_width) * 0.02)
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
        self.camera_embedding = nn.Parameter(torch.randn(config.num_cameras, 1, config.width) * 0.02) if config.num_cameras > 1 else None
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
        if self.camera_embedding is None:
            visual = self.vision(images)
        else:
            visual = self.vision(images.flatten(0, 1)).unflatten(0, (images.shape[0], c.num_cameras))
            visual = (visual + self.camera_embedding).flatten(1, 2)
        p = torch.cat((visual, self.tokens(words) * math.sqrt(c.width)), 1)
        valid = torch.cat((torch.ones_like(visual[:, :, 0], dtype=torch.bool), instr != 0,
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


REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
REPO = "HuggingFaceVLA/libero"
DATASET = Path.home() / f".cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/{REVISION}"
CAMERAS = ("observation.images.image", "observation.images.image2")


def words(text):
    return text.lower().split()


def tokenizer(texts):
    return {"vocab": ["<pad>", "<unk>", *sorted({w for t in texts for w in words(t)})],
            "length": max(map(lambda t: len(words(t)), texts))}


def tokenize(texts, tok, *, truncate=False):
    """Keep training length checks; inference may clip to the saved context length."""
    lookup = {w: i for i, w in enumerate(tok["vocab"])}
    # Reuse the checkpoint's reserved embedding; legacy vocabularies without <unk>
    # use padding (ID 0, masked by MicroVLA), without adding/resizing embeddings.
    unk = lookup.get("<unk>", 0)
    out = np.zeros((len(texts), tok["length"]), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [lookup.get(w, unk) for w in words(text)]
        if len(ids) > tok["length"]:
            if not truncate:
                raise ValueError("Instruction exceeds saved tokenizer length")
            ids = ids[:tok["length"]]
        out[i, :len(ids)] = ids
    return out


def spatial_tasks():
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    return [suite.get_task(i).language for i in range(suite.n_tasks)]


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def decode_image(value, size):
    # Dataset frames have already undergone LeRobot's 180 degree rotation. Do not flip again.
    if value["bytes"] is None:
        raise ValueError("Expected embedded image bytes")
    with Image.open(io.BytesIO(value["bytes"])) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).transpose(2, 0, 1)


def array_specs(n, size):
    return {"images": (np.uint8, (n, 2, 3, size, size)), "state": (np.float32, (n, 8)),
            "actions": (np.float32, (n, 7)), "episode": (np.int64, (n,)),
            "task": (np.int64, (n,)), "frame": (np.int64, (n,))}


def open_arrays(cache, manifest, mode="r"):
    return {k: np.memmap(Path(cache) / f"{k}.bin", mode=mode, dtype=dtype, shape=shape)
            for k, (dtype, shape) in array_specs(manifest["frames"], manifest["image_size"]).items()}


def release_pages(array):
    # Random memmap access otherwise grows RSS toward the full 2.93 GB cache.
    if hasattr(array._mmap, "madvise"):
        array._mmap.madvise(mmap.MADV_DONTNEED)


def fit_norm(x):
    low, high = np.quantile(x, [.01, .99], axis=0).astype(np.float32)
    return {"center": ((high + low) / 2).tolist(), "scale": np.maximum((high - low) / 2, 1e-6).tolist()}


def normalize(x, stats):
    return (x - x.new_tensor(stats["center"])) / x.new_tensor(stats["scale"])


def unnormalize(x, stats):
    return x * x.new_tensor(stats["scale"]) + x.new_tensor(stats["center"])


class StreamingSource:
    """Range-read scalar columns; keep at most one full parquet on disk."""
    def __init__(self, cache):
        from huggingface_hub import HfFileSystem
        self.fs = HfFileSystem()
        self.root = f"datasets/{REPO}@{REVISION}/"
        self.scratch = Path(cache) / ".stream-source"
        # Includes partial downloads left by a killed process. Never use the shared HF cache.
        if self.scratch.exists():
            shutil.rmtree(self.scratch)

    def glob(self, pattern):
        return sorted(p.removeprefix(self.root) for p in self.fs.glob(self.root + pattern))

    @contextmanager
    def open(self, file, download=False):
        if not download:
            # No fsspec read-ahead or Arrow pre-buffering of image columns.
            with self.fs.open(self.root + file, "rb", cache_type="none", block_size=64 * 1024) as f:
                yield f
            return
        from huggingface_hub import constants, hf_hub_download
        # Use HTTP streaming: Xet can add persistent chunk caches and large buffers.
        previous_xet = constants.HF_HUB_DISABLE_XET
        constants.HF_HUB_DISABLE_XET = True
        try:
            path = hf_hub_download(REPO, file, repo_type="dataset", revision=REVISION,
                                   local_dir=self.scratch)
            with Path(path).open("rb") as f:
                yield f
        finally:
            constants.HF_HUB_DISABLE_XET = previous_xet
            if self.scratch.exists():
                shutil.rmtree(self.scratch)


def preprocess(source=DATASET, cache="runs/libero_cache", image_size=96, task_texts=None, seed=0, stream=False):
    """Commit progress per source parquet. Interrupted files are safely replayed in place."""
    import pyarrow.parquet as pq
    source, cache = Path(source), Path(cache)
    remote = StreamingSource(cache) if stream else None
    def paths(pattern):
        return remote.glob(pattern) if remote else sorted(p.relative_to(source).as_posix() for p in source.glob(pattern))

    def batches(file, columns=None, download=False):
        handle = remote.open(file, download) if remote else (source / file).open("rb")
        with handle as f, pq.ParquetFile(f, pre_buffer=False) as parquet:
            yield from parquet.iter_batches(batch_size=32, columns=columns, use_threads=False)

    def metadata(file, columns=None):
        handle = remote.open(file) if remote else (source / file).open("rb")
        with handle as f:
            return pq.read_table(f, columns=columns, use_threads=False, pre_buffer=False).to_pylist()

    texts = spatial_tasks() if task_texts is None else task_texts
    tasks = {r["task_index"]: r["__index_level_0__"] for r in metadata("meta/tasks.parquet")}
    selected = {i: texts.index(t) for i, t in tasks.items() if t in texts}
    if set(tasks[i] for i in selected) != set(texts):
        raise ValueError("Dataset is missing Spatial task languages")
    episodes = []
    columns = ["episode_index", "tasks", "length"]
    for file in paths("meta/episodes/**/*.parquet"):
        episodes.extend(r for r in metadata(file, columns) if len(r["tasks"]) == 1 and r["tasks"][0] in texts)
    episodes.sort(key=lambda r: r["episode_index"])
    if not episodes or len({r["episode_index"] for r in episodes}) != len(episodes):
        raise ValueError("Missing or duplicate episodes")
    offset, by_id = 0, {}
    for r in episodes:
        r["offset"] = offset
        by_id[r["episode_index"]] = r
        offset += r["length"]
    if image_size < 8 or image_size % 8:
        raise ValueError("image_size must be a positive multiple of 8")
    def validate_counts(counts, stage, expected=None):
        expected = {i: ep["length"] for i, ep in by_id.items()} if expected is None else expected
        bad = []
        for i, length in expected.items():
            count = counts.get(i, 0)
            if count != length:
                status = "missing" if count == 0 else "short" if count < length else "excess"
                bad.append(f"{i}: {count}/{length} ({status})")
        if bad:
            raise ValueError(f"Spatial episode frame count mismatch {stage} (episode: found/expected): " + "; ".join(bad))

    # Episode metadata has unreliable data/file_index values. Discover source files
    # using scalar column projection only; unrelated suites need no image reads.
    files, counts = {}, {}
    for file in paths("data/**/*.parquet"):
        found = {}
        for batch in batches(file, ["episode_index", "task_index"]):
            for r in batch.to_pylist():
                i = r["episode_index"]
                if i not in by_id:
                    continue
                if tasks.get(r["task_index"]) != by_id[i]["tasks"][0]:
                    raise ValueError(f"Task mismatch for Spatial episode {i} in {file}")
                found[i] = found.get(i, 0) + 1
                counts[i] = counts.get(i, 0) + 1
        if found:
            files[file] = found
    validate_counts(counts, "before decoding")
    identity = dict(version=2, source=remote.root if remote else str(source.resolve()), image_size=image_size, frames=offset,
                    tasks=texts, episodes=episodes, seed=seed,
                    files={p: {"bytes": remote.fs.size(remote.root + p)} if remote else
                           {"bytes": (source / p).stat().st_size, "mtime_ns": (source / p).stat().st_mtime_ns} for p in files})
    needed = sum(np.dtype(d).itemsize * math.prod(s) for d, s in array_specs(offset, image_size).values())
    path = cache / "manifest.json"
    manifest = None
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Cache configuration/source changed; choose another --cache")
        for name, (dtype, shape) in array_specs(offset, image_size).items():
            if (cache / f"{name}.bin").stat().st_size != np.dtype(dtype).itemsize * math.prod(shape):
                raise ValueError(f"Incomplete cache allocation: {name}")
        if manifest.get("complete"):
            return manifest
    # Account for physical blocks already allocated by an interrupted build, not
    # sparse file lengths. A resume must not require room for a second cache.
    allocated = sum((cache / f"{k}.bin").stat().st_blocks * 512 for k in array_specs(offset, image_size)) if manifest else 0
    required = max(0, needed - allocated) + 2**30
    parent = cache
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required:
        raise ValueError(f"Insufficient free disk for cache: need {required:,} bytes "
                         f"(cache {needed:,}, already allocated {allocated:,}, plus 1 GiB); have {free:,}")
    cache.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        for name, (dtype, shape) in array_specs(offset, image_size).items():
            with (cache / f"{name}.bin").open("wb") as f:
                f.truncate(np.dtype(dtype).itemsize * math.prod(shape))
        manifest = dict(identity=identity, frames=offset, image_size=image_size, completed_files=[], complete=False)
        atomic_json(path, manifest)
    arrays = open_arrays(cache, manifest, "r+")
    seen, decoded = np.zeros(offset, dtype=bool), {}
    for file in files:
        completed = file in manifest["completed_files"]
        file_decoded = {}
        columns = ["episode_index", "frame_index", "task_index"]
        if not completed:
            columns += [*CAMERAS, "observation.state", "action"]
        # Reconstruct seen frames for committed files without decoding them again.
        for batch in batches(file, columns, download=not completed):
            for r in batch.to_pylist():
                ep = by_id.get(r["episode_index"])
                if ep is None:
                    continue
                frame = r["frame_index"]
                if not 0 <= frame < ep["length"] or r["task_index"] not in selected or tasks[r["task_index"]] != ep["tasks"][0]:
                    raise ValueError(f"Invalid frame/task for Spatial episode {r['episode_index']} in {file}")
                index = ep["offset"] + frame
                if seen[index]:
                    raise ValueError(f"Duplicate frame {frame} for Spatial episode {r['episode_index']} in {file}")
                if not completed:
                    arrays["images"][index] = np.stack([decode_image(r[c], image_size) for c in CAMERAS])
                    arrays["state"][index], arrays["actions"][index] = r["observation.state"], r["action"]
                    arrays["episode"][index], arrays["task"][index], arrays["frame"][index] = r["episode_index"], selected[r["task_index"]], frame
                seen[index] = True
                decoded[r["episode_index"]] = decoded.get(r["episode_index"], 0) + 1
                file_decoded[r["episode_index"]] = file_decoded.get(r["episode_index"], 0) + 1
            if not completed:
                # Bound resident image pages even when a source file is large.
                arrays["images"].flush()
                release_pages(arrays["images"])
        validate_counts(file_decoded, f"after decoding {file}", files[file])
        if completed:
            continue
        for a in arrays.values():
            a.flush()
            release_pages(a)
        manifest["completed_files"].append(file)
        atomic_json(path, manifest)
        print(json.dumps(dict(cached_file=file, completed=len(manifest["completed_files"]), total=len(files))), flush=True)
    validate_counts(decoded, "after decoding")
    # Deterministic, task-stratified episode split; no action chunk ever crosses an episode.
    rng, val_episodes = np.random.default_rng(seed), []
    for text in texts:
        ids = [r["episode_index"] for r in episodes if r["tasks"] == [text]]
        if len(ids) < 2:
            raise ValueError("Need >=2 episodes per task for held-out validation")
        val_episodes.extend(rng.permutation(ids)[:max(1, round(.05 * len(ids)))].tolist())
    train = ~np.isin(arrays["episode"], val_episodes)
    stats = {key: fit_norm(arrays[array][train]) for key, array in (("state", "state"), ("action", "actions"))}
    if not all(np.isfinite(arrays[k]).all() for k in ("state", "actions")):
        raise ValueError("Nonfinite state/action data")
    manifest.update(complete=True, val_episodes=val_episodes, norm=stats, tokenizer=tokenizer(texts),
                    cache_bytes=needed, normalization="train-episode q01/q99; no gripper inversion", orientation="LeRobot canonical; no additional flip")
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(path, manifest)
    return manifest


class Batches:
    """Vectorized NumPy gathers, pinned transfers, no worker copies or decoded-image queue."""
    def __init__(self, cache, horizon, seed=0):
        self.meta = json.loads((Path(cache) / "manifest.json").read_text())
        if not self.meta["complete"]:
            raise ValueError("Run --preprocess first")
        self.arrays = open_arrays(cache, self.meta)
        self.horizon, self.rng = horizon, np.random.default_rng(seed)
        val = np.isin(self.arrays["episode"], self.meta["val_episodes"])
        self.indices = {"train": np.flatnonzero(~val), "val": np.flatnonzero(val)}
        self.ends = np.empty(self.meta["frames"], dtype=np.int64)
        for ep in self.meta["identity"]["episodes"]:
            self.ends[ep["offset"]:ep["offset"] + ep["length"]] = ep["offset"] + ep["length"] - 1
        self.instructions = tokenize(self.meta["identity"]["tasks"], self.meta["tokenizer"])

    def batch(self, size, device, split="train", rng=None):
        rng = self.rng if rng is None else rng
        idx = rng.choice(self.indices[split], size)
        future = np.minimum(idx[:, None] + np.arange(self.horizon), self.ends[idx, None])
        a = self.arrays
        values = [a["images"][idx], a["state"][idx], self.instructions[a["task"][idx]], a["actions"][future]]
        release_pages(a["images"])
        tensors = []
        for value in values:
            t = torch.from_numpy(np.ascontiguousarray(value))
            if device.type == "cuda":
                t = t.pin_memory()
            tensors.append(t.to(device, non_blocking=True))
        images, state, instr, actions = tensors
        return images, normalize(state, self.meta["norm"]["state"]), instr, normalize(actions, self.meta["norm"]["action"])


def augment(images, pad=4):
    b, cameras, channels, h, w = images.shape
    x = F.pad(images.flatten(0, 1).float(), (pad,) * 4, mode="replicate")
    shifts = torch.randint(2 * pad + 1, (b * cameras, 2), device=images.device)
    y = shifts[:, 0, None, None] + torch.arange(h, device=x.device)[None, :, None]
    z = shifts[:, 1, None, None] + torch.arange(w, device=x.device)[None, None, :]
    return x.permute(0, 2, 3, 1)[torch.arange(len(x), device=x.device)[:, None, None], y, z].permute(0, 3, 1, 2).to(torch.uint8).reshape_as(images)


def color_jitter(images, strength=.1):
    # Share small brightness/contrast changes across a sample's cameras.
    x = images.float()
    shape = (len(x), 1, 1, 1, 1)
    brightness, contrast = [torch.empty(shape, device=x.device).uniform_(1 - strength, 1 + strength)
                            for _ in range(2)]
    mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    return ((x - mean) * contrast + mean).mul_(brightness).round_().clamp_(0, 255).to(images.dtype)


@torch.no_grad()
def update_ema(ema, model, decay):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema[name].lerp_(param.detach(), 1 - decay)


def save_checkpoint(path, model, opt, phase, step, meta, settings, rng, head=None, ema=None, muon=None):
    state = dict(config=asdict(model.config), model=model.state_dict(), opt=opt.state_dict(), phase=phase,
                 step=step, norm=meta["norm"], tokenizer=meta["tokenizer"], cache_fingerprint=meta["fingerprint"],
                 settings=settings, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
                 head=head.state_dict() if head is not None else None)
    if ema is not None:
        state["ema"] = ema
    if muon is not None:
        state["muon"] = muon.state_dict()
    tmp = Path(path).with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def restore_checkpoint(ckpt, model, opt, rng, head=None, ema=None, muon=None):
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    if muon is not None:
        muon.load_state_dict(ckpt["muon"])
    rng.bit_generator.state = ckpt["numpy_rng"]
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt["cuda_rng"] is not None:
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
    if head is not None:
        head.load_state_dict(ckpt["head"])
    if ema is not None:
        for name, value in ckpt["ema"].items():
            ema[name].copy_(value)
    return ckpt["step"] + 1


class IdentityProcessor:
    def __call__(self, x):
        return x

    def reset(self):
        pass


class TinyPolicy(nn.Module):
    """LeRobot rollout adapter: canonical RGB/state in, raw relative 7-D commands out."""
    def __init__(self, ckpt, device="cuda", ablation=None):
        super().__init__()
        if ablation not in (None, "blind", "no-wrist", "wrong-text", "no-state"):
            raise ValueError(f"Unknown tiny input ablation: {ablation}")
        self.ablation = ablation
        if ablation == "wrong-text":
            texts = spatial_tasks()  # Benchmark order, independent of --tasks selection.
            self.wrong_text = dict(zip(texts, texts[1:] + texts[:1]))
        self.model = MicroVLA(Config(**ckpt["config"])).to(device).eval()
        self.model.load_state_dict({**ckpt["model"], **(ckpt.get("ema") or {})})
        self.norm, self.tok = ckpt["norm"], ckpt["tokenizer"]
        self.config = SimpleNamespace(chunk_size=self.model.config.horizon, max_action_dim=7,
                                      n_action_steps=self.model.config.horizon,
                                      use_visual_memory=False, use_proprioceptive_memory=False)
        self.reset()

    def reset(self):
        self.queue = deque()

    def instruction_tokenization(self, text):
        """Describe the policy's instruction, counting OOVs before truncation."""
        if self.ablation == "wrong-text":
            text = self.wrong_text[text]
        tokens = words(text)
        return dict(oov_fraction=sum(w not in self.tok["vocab"] for w in tokens) / max(1, len(tokens)),
                    truncated=len(tokens) > self.tok["length"])

    def prepare(self, batch):
        device = next(self.parameters()).device
        images = []
        for key in CAMERAS:
            value = batch[key]
            x = torch.as_tensor(value.copy() if isinstance(value, np.ndarray) else value)
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            if x.dtype != torch.uint8:  # LeRobot canonical preparation scales RGB into [0,1].
                x = x.float().mul(255).round().clamp(0, 255).to(torch.uint8)
            # PIL bilinear, identical to cache preprocessing (two small images per chunk).
            images.append(torch.stack([torch.from_numpy(np.array(Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).resize(
                (self.model.config.image_size,) * 2, Image.Resampling.BILINEAR))).permute(2, 0, 1) for im in x]))
        value = batch["observation.state"]
        state = torch.as_tensor(value.copy() if isinstance(value, np.ndarray) else value, device=device, dtype=torch.float32)
        texts = [self.wrong_text[t] for t in batch["task"]] if self.ablation == "wrong-text" else batch["task"]
        instr = torch.from_numpy(tokenize(texts, self.tok, truncate=True)).to(device)
        images, state = torch.stack(images, 1).to(device), normalize(state, self.norm["state"])
        if self.ablation == "blind":
            images.zero_()
        elif self.ablation == "no-wrist":
            images[:, 1].zero_()  # CAMERAS[1] is observation.images.image2.
        elif self.ablation == "no-state":
            state.zero_()  # Saved training normalization center, not raw zero state.
        return images, state, instr

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, noise, num_steps):
        # episode() prepares these before starting its synchronized model-only latency timer.
        obs = batch["tiny_inputs"] if "tiny_inputs" in batch else self.prepare(batch)
        actions = self.model.sample(*obs, num_steps=num_steps, noise=noise)
        return actions

    def postprocess(self, actions):
        return unnormalize(actions.float(), self.norm["action"]).clamp(-1, 1)

    @torch.no_grad()
    def select_action(self, batch):
        if not self.queue:
            batch = {**batch, "tiny_inputs": self.prepare(batch)}
            actions = self.postprocess(self.predict_action_chunk(batch))
            self.queue.extend(actions[:, :self.config.n_action_steps].transpose(0, 1))
        return self.queue.popleft()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess", action="store_true", help="only build/resume the disk cache")
    parser.add_argument("--stream", action="store_true", help="preprocess pinned HF dataset one parquet at a time")
    parser.add_argument("--source", type=Path, default=DATASET, help="local source (without --stream)")
    parser.add_argument("--cache", type=Path, help="default: runs/libero_cache; streaming: runs/libero_cache_<image-size>")
    parser.add_argument("--runs", type=Path, default=Path("runs/libero_tiny"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--distill-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--distill-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--ema", type=float, help="EMA decay in [0, 1); disabled by default")
    parser.add_argument("--save-every", type=int, default=0, help="numbered action-training snapshots; 0 disables")
    parser.add_argument("--color-jitter", action="store_true", help="small per-sample brightness/contrast augmentation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "none"), default="tensorboard")
    parser.add_argument("--wandb-project", default="microvla")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    if args.stream and not args.preprocess:
        parser.error("--stream requires --preprocess")
    if args.cache is None:
        args.cache = Path(f"runs/libero_cache_{args.image_size}" if args.stream else "runs/libero_cache")
    if min(args.batch_size, args.distill_batch_size, args.horizon, args.lr) <= 0 or min(args.steps, args.distill_steps) < 0:
        parser.error("batch sizes, horizon and lr must be positive; steps must be nonnegative")
    if args.ema is not None and not 0 <= args.ema < 1:
        parser.error("--ema must be in [0, 1)")
    if args.save_every < 0:
        parser.error("--save-every must be nonnegative")
    if args.preprocess:
        meta = preprocess(args.source, args.cache, args.image_size, seed=args.seed, stream=args.stream)
        print(json.dumps({k: meta[k] for k in ("frames", "cache_bytes", "image_size", "val_episodes")}))
        return
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = Batches(args.cache, args.horizon, args.seed)
    c = Config(image_size=data.meta["image_size"], num_cameras=2, vocab_size=len(data.meta["tokenizer"]["vocab"]), horizon=args.horizon, state_dim=8, action_dim=7)
    settings = {k: getattr(args, k) for k in ("steps", "distill_steps", "batch_size", "distill_batch_size", "lr", "seed", "ema", "color_jitter", "optimizer")}
    path = args.runs / "latest.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True) if args.resume and path.exists() else None
    if args.resume and ckpt is None:
        raise FileNotFoundError(f"No checkpoint to resume: {path}")
    if path.exists() and not args.resume:
        raise ValueError("Checkpoint exists; use --resume or another --runs")
    if ckpt and (ckpt["config"] != asdict(c) or {"ema": None, "color_jitter": False, "optimizer": "adamw", **ckpt["settings"]} != settings or ckpt["cache_fingerprint"] != data.meta["fingerprint"]):
        raise ValueError("Resume configuration/cache differs from checkpoint")
    args.runs.mkdir(parents=True, exist_ok=True)
    model = MicroVLA(c).to(device)
    # Only block-internal matrices: attention, MLPs and adaptive norm modulation.
    # Embeddings and boundary projections (including vision.project) stay AdamW.
    muon_params = [p for name, p in model.named_parameters() if args.optimizer == "muon"
                   and p.requires_grad and p.ndim == 2 and name.startswith(("layers.", "vision.blocks."))]
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema is not None else None
    print(json.dumps(dict(parameters=parameter_counts(model), cache_bytes=data.meta["cache_bytes"],
                          optimizer_parameters=dict(adamw=sum(p.numel() for p in adamw_params),
                                                    muon=sum(p.numel() for p in muon_params)),
                          estimated_hours=(args.steps / 1.2 + args.distill_steps / 1.0) / 3600,
                          estimate="unmeasured 3050 heuristic; measured ETA follows after 50 steps")), flush=True)
    writer = make_logger(args, settings, c, args.runs / "tb")
    try:
        for phase, total in (("distill", args.distill_steps), ("train", args.steps)):
            if not total or (ckpt and ckpt["phase"] == "train" and phase == "distill"):
                continue
            teacher = head = None
            if phase == "distill":
                from transformers import SiglipVisionModel
                teacher = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", local_files_only=True,
                    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval().requires_grad_(False)
                head = nn.Linear(c.width, teacher.config.hidden_size).to(device)
                params = [*model.vision.parameters(), *head.parameters()]
            else:
                params = list(model.parameters())
            opt = torch.optim.AdamW(adamw_params if phase == "train" else params,
                                    lr=args.lr if phase == "train" else 1e-3, weight_decay=.01)
            muon = torch.optim.Muon(muon_params, lr=args.lr, momentum=.95, nesterov=True, ns_steps=5,
                                    weight_decay=.01, adjust_lr_fn="match_rms_adamw") if phase == "train" and args.optimizer == "muon" else None
            optimizers = (opt,) if muon is None else (opt, muon)
            first = 1
            if ckpt and ckpt["phase"] == phase:
                first = restore_checkpoint(ckpt, model, opt, data.rng, head, ema, muon)
                ckpt = None
            if phase == "train" and args.compile:
                from torch._inductor import config as inductor_config
                inductor_config.compile_threads = 1
                model.forward = torch.compile(model.forward)
            start = time.perf_counter()
            for step in range(first, total + 1):
                model.train()
                size = args.batch_size if phase == "train" else args.distill_batch_size
                images, state, instr, actions = data.batch(size, device)
                images = augment(images)
                if args.color_jitter:
                    images = color_jitter(images)
                if phase == "distill":
                    # One random camera per frame; both cameras share the ViT.
                    images = images[torch.arange(size, device=device), torch.randint(2, (size,), device=device)]
                    with torch.no_grad(), autocast(device):
                        x = F.interpolate(images.float().div(127.5).sub(1), size=c.image_size * 2, mode="bilinear", align_corners=False)
                        target = teacher(pixel_values=x.to(next(teacher.parameters()).dtype), interpolate_pos_encoding=True).last_hidden_state.float()
                        target = F.layer_norm(target, target.shape[-1:])
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    if phase == "train":
                        loss = model.loss(images, state, instr, actions)
                    else:
                        pred = head(model.vision(images)).float()
                        loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean() + F.mse_loss(pred, target)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.)
                lr = (args.lr if phase == "train" else 1e-3) * min(step / 100, 1) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total)))
                for optimizer in optimizers:
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                if ema is not None:
                    update_ema(ema, model, args.ema)
                if step % 50 == 0 or step == total:
                    value = loss.item()
                    writer.add_scalar(f"{phase}/loss", value, step)
                    rate = (step - first + 1) / (time.perf_counter() - start)
                    print(json.dumps(dict(phase=phase, step=step, loss=value, steps_per_s=rate, eta_minutes=(total-step)/rate/60,
                        peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None)), flush=True)
                snapshot = phase == "train" and args.save_every > 0 and step % args.save_every == 0
                if step % 250 == 0 or step == total or snapshot:
                    if phase == "train" and (step % 250 == 0 or step == total):
                        model.eval()
                        with torch.no_grad(), autocast(device), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                            torch.manual_seed(12345)
                            rng = np.random.default_rng(12345)
                            val = sum(model.loss(*data.batch(args.batch_size, device, "val", rng)).item() for _ in range(4)) / 4
                        writer.add_scalar("val/loss", val, step)
                        print(json.dumps(dict(step=step, val_loss=val)), flush=True)
                    if snapshot:
                        save_checkpoint(args.runs / f"step_{step}.pt", model, opt, phase, step, data.meta, settings, data.rng, head, ema, muon)
                    save_checkpoint(path, model, opt, phase, step, data.meta, settings, data.rng, head, ema, muon)
            del teacher, head, opt, muon, optimizers, params
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
