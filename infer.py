"""Real pi0.5 on 4 GiB: NF4 prefix or pinned-RAM prefix streaming via Accelerate.
Offload retains action path/embeddings, reserves 700 MiB, fits SigLIP before Gemma.
Construct on meta, omit vocabulary heads, then load one weight at a time.

Trim removes only masked camera/text slots. SigLIP encodes cameras independently;
prefix attention masks invalid keys and queries, and action attention masks
the same prefix keys. Positions are cumsum(pad_mask)-1; action positions start at
sum(prefix_pad_mask). Removing zero-mask slots preserves valid embeddings,
positions and attention, hence actions mathematically (GPU rounding may differ).
"""
import argparse
from collections.abc import Callable
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
from torch import Tensor, nn

POLICY = "lerobot/pi05_libero_finetuned_v044"
REVISION = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
PREFIX = "model.paligemma_with_expert.paligemma.model"
DEFAULT_CONFIGS = ["nf4", "nf4+trim", "offload-3.5", "offload-2.5", "offload-2.5+trim",
                   "steps-10", "steps-5", "steps-5+trim", "steps-2"]


@dataclass
class Frame:
    observation: dict
    action: Tensor
    episode: int
    index: int


def checkpoint() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(POLICY, "model.safetensors", revision=REVISION,
                               local_files_only=True)).parent


def load_frames(count: int = 8) -> list[Frame]:
    """Sample across episodes; decode only selected PNGs, using no pandas."""
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import pyarrow.parquet as pq
    def cached(name):
        return hf_hub_download("HuggingFaceVLA/libero", name, repo_type="dataset", local_files_only=True)
    tasks = {r["task_index"]: r["__index_level_0__"] for r in pq.read_table(cached("meta/tasks.parquet")).to_pylist()}
    path = cached("data/chunk-000/file-000.parquet")
    episodes = pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()
    groups = [np.flatnonzero(np.array(episodes) == e) for e in sorted(set(episodes))]
    if not 1 <= count <= len(episodes):
        raise ValueError("frame count must be between 1 and the number of cached frames")
    # Round-robin episodes, evenly spaced within each episode.
    picks = [g[np.linspace(0, len(g) - 1, (count + len(groups) - 1 - i) // len(groups), dtype=int)]
             for i, g in enumerate(groups)]
    indices = [int(picks[i % len(groups)][i // len(groups)]) for i in range(count)]
    frames = []
    for row in pq.read_table(path).take(indices).to_pylist():
        obs = {"task": tasks[row["task_index"]], "observation.state": torch.tensor(row["observation.state"])}
        for key in ("observation.images.image", "observation.images.image2"):
            pixels = np.array(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"), copy=True)
            obs[key] = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        frames.append(Frame(obs, torch.tensor(row["action"]), row["episode_index"], row["frame_index"]))
    return frames


def empty_policy(path: Path, trim: bool = False) -> nn.Module:
    """Meta initialization also avoids LeRobot's eager fp32 vision allocation."""
    from accelerate import init_empty_weights
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    config = PI05Config.from_pretrained(path)
    if trim:
        config.empty_cameras, config.tokenizer_max_length = 0, 64
        config.input_features.pop("observation.images.empty_camera_0")
    config.device, config.compile_model, config.gradient_checkpointing = "meta", False, False
    with init_empty_weights():
        policy = PI05Policy(config)
    config.device = "cuda"
    joint = policy.model.paligemma_with_expert
    joint.paligemma.lm_head = joint.gemma_expert.lm_head = nn.Identity()
    # Nonpersistent buffers are absent from safetensors: reconstruct their formulas.
    for module in policy.modules():
        if hasattr(module, "rope_type"):
            fresh = type(module)(module.config)
            for name, value in fresh.named_buffers():
                module._buffers[name] = value
        elif "position_ids" in module._buffers:
            module.position_ids = torch.arange(module.position_ids.numel()).expand(1, -1)
        elif "embed_scale" in module._buffers:
            module.embed_scale = torch.tensor(module.scalar_embed_scale)
    return policy.eval().requires_grad_(False)


def placement(policy: nn.Module, budget: float) -> tuple[list, set]:
    """Partition prefix into whole transformer blocks and remaining parameter units."""
    units = []
    for name, module in policy.named_modules():
        if not name.startswith(PREFIX) or any(name.startswith(n + ".") for n, _ in units):
            continue
        if re.search(r"\.layers\.\d+$", name) or list(module.parameters(recurse=False)):
            units.append((name, module))
    mandatory = sum(p.numel() * 2 for n, p in policy.named_parameters()
                    if not n.startswith(PREFIX) or ".embed_tokens." in n)
    available = int(budget * 2**30) - 700 * 2**20 - mandatory
    if available < 0:
        raise RuntimeError("budget cannot hold action path, embeddings and 700 MiB workspace")
    resident = {n for n, _ in units if ".embed_tokens" in n}
    for name, module in sorted(units, key=lambda item: "vision_tower" not in item[0]):
        size = sum(p.numel() * 2 for p in module.parameters())
        if name not in resident and size <= available:
            resident.add(name)
            available -= size
    return units, resident


def weight_sources(policy: nn.Module, keys: list[str]) -> dict[str, str]:
    """Transformers 5 flattened SigLIP; the checkpoint also stores tied embeddings as lm_head."""
    sources = {}
    for name in dict(policy.named_parameters()):
        source = name.replace(".vision_tower.", ".vision_tower.vision_model.")
        if source not in keys and ".language_model.embed_tokens." in name:
            source = name.replace(".model.language_model.embed_tokens", ".lm_head")
        sources[name] = source if source in keys else name
    return sources


def load_policy(config: str = "nf4", trim: bool = False) -> tuple[nn.Module, Callable, Callable, dict]:
    from accelerate import cpu_offload
    from safetensors import safe_open
    trim = trim or config.endswith("+trim")
    config = config.removesuffix("+trim")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU inference was not run")
    budget = float(config.removeprefix("offload-")) if config.startswith("offload-") else None
    if config != "nf4" and budget is None:
        raise ValueError(f"unknown placement: {config}")
    torch.cuda.set_per_process_memory_fraction(min(1., budget * 2**30 / torch.cuda.get_device_properties(0).total_memory)
                                               if budget else 1.)
    torch.cuda.reset_peak_memory_stats()
    path = checkpoint()
    policy = empty_policy(path, trim)
    units, resident = placement(policy, budget) if budget else ([], set())
    offloaded = [n for n, _ in units if n not in resident]
    if config == "nf4":
        import bitsandbytes as bnb
        for name, module in list(policy.named_modules()):
            if name.startswith(PREFIX) and ("language_model" in name or "vision_tower" in name) and isinstance(module, nn.Linear):
                with torch.device("meta"):
                    quant = bnb.nn.Linear4bit(module.in_features, module.out_features, module.bias is not None,
                                             compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4")
                parent, attr = name.rsplit(".", 1)
                setattr(policy.get_submodule(parent), attr, quant)
    with safe_open(path / "model.safetensors", framework="pt") as weights:
        sources = weight_sources(policy, list(weights.keys()))
        unused = set(weights.keys()) - set(sources.values())
        if any(not n.endswith("lm_head.weight") for n in unused):
            raise ValueError(f"unexpected checkpoint keys: {unused}")
        for name, param in policy.named_parameters():
            value = weights.get_tensor(sources[name])
            if value.shape != param.shape:
                raise ValueError(f"checkpoint shape mismatch: {name}")
            parent, attr = name.rsplit(".", 1)
            module = policy.get_submodule(parent)
            cpu = any(name.startswith(n + ".") for n in offloaded)
            if config == "nf4" and isinstance(param, bnb.nn.Params4bit):
                module._parameters[attr] = bnb.nn.Params4bit(value.to(torch.bfloat16), requires_grad=False,
                                                            compress_statistics=True, quant_type="nf4").to("cuda")
            else:
                value = value.to(dtype=torch.bfloat16)
                value = value.pin_memory() if cpu else value.to("cuda")
                module._parameters[attr] = nn.Parameter(value, requires_grad=False)
    for module in policy.modules():
        for name, value in module.named_buffers(recurse=False):
            module._buffers[name] = value.to("cuda")
    for name, module in units:
        if name in offloaded:
            cpu_offload(module, execution_device=torch.device("cuda"), state_dict=module.state_dict(),
                        preload_module_classes=[type(module).__name__])
    pre, post = processors(policy, path)
    vision = policy.model.paligemma_with_expert.paligemma.config.vision_config
    info = {"trim": trim, "prefix_tokens": {"images": len(policy.config.image_features) * (vision.image_size // vision.patch_size)**2,
                                          "text": policy.config.tokenizer_max_length},
            "offloaded_units": offloaded, "resident_prefix_units": sorted(resident),
            "resident_weight_gib": sum(p.numel() * p.element_size() for p in policy.parameters()
                                       if p.device.type == "cuda") / 2**30, "budget_gib": budget}
    return policy, pre, post, info


def processors(policy: nn.Module, path: Path, device: str = "cuda") -> tuple[Callable, Callable]:
    from huggingface_hub import hf_hub_download
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import TokenizerProcessorStep
    # A local tokenizer directory avoids AutoTokenizer fetching an uncached model config.
    tokenizer = Path(hf_hub_download("google/paligemma-3b-pt-224", "tokenizer_config.json", local_files_only=True)).parent
    trimmed = policy.config.tokenizer_max_length == 64
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path, preprocessor_overrides={
        "device_processor": {"device": device}, "tokenizer_processor": {"tokenizer_name": str(tokenizer),
        "max_length": policy.config.tokenizer_max_length, "padding": "max_length", "truncation": not trimmed}})
    if trimmed:
        def check_prompt(index, transition):
            step = pre.steps[index]
            if isinstance(step, TokenizerProcessorStep):
                for prompt in step.get_task(transition):
                    length = len(step.input_tokenizer(prompt, truncation=False, padding=False)["input_ids"])
                    if length > 64:
                        raise ValueError(f"Prompt for {prompt!r} has {length} tokens; trim limit is 64")
        pre.before_step_hooks.append(check_prompt)
    return pre, post


@torch.inference_mode()
def predict(policy: nn.Module, pre: Callable, post: Callable, frame: Frame, steps: int = 10, seed: int = 0) -> tuple[Tensor, dict]:
    """Time prefix (including images) and the complete Euler loop with CUDA events."""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    original = policy.model.denoise_step
    first = True
    def denoise(*args, **kwargs):
        nonlocal first
        if first:
            events[1].record()
            first = False
        return original(*args, **kwargs)
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                        generator=torch.Generator().manual_seed(seed)).to("cuda")
    policy.model.denoise_step = denoise
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        batch = pre(dict(frame.observation))
        events[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = policy.predict_action_chunk(batch, noise=noise, num_steps=steps)
        events[2].record()
        actions = post(actions)[0].float().cpu()
        torch.cuda.synchronize()
    finally:
        policy.model.denoise_step = original
    latency = time.perf_counter() - start
    if not torch.isfinite(actions).all():
        raise RuntimeError("predicted actions contain NaN/Inf")
    return actions, {"latency_ms": latency * 1000, "prefix_ms": events[0].elapsed_time(events[1]),
                     "denoise_ms": events[1].elapsed_time(events[2]),
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def run_config(config: str, steps: int, count: int, timed: bool = True, trim: bool = False) -> dict:
    frames = load_frames(count)
    policy, pre, post, info = load_policy(config, trim)
    for _ in range(2 if timed else 0):
        predict(policy, pre, post, frames[0], steps)
    actions, samples = [], []
    for i in range(max(5, count) if timed else 1):
        action, metrics = predict(policy, pre, post, frames[i % count], steps, seed=i % count)
        actions.append(action.tolist())
        samples.append(metrics)
    latency = [s["latency_ms"] for s in samples]
    return {"status": "ok", "placement": config, "steps": steps, **info, "samples": samples,
            "latency_p50_ms": float(np.percentile(latency, 50)), "latency_p95_ms": float(np.percentile(latency, 95)),
            "actions_per_s": len(actions[0]) / (float(np.median(latency)) / 1000),
            "peak_allocated_gib": max(s["peak_allocated_gib"] for s in samples),
            "peak_reserved_gib": max(s["peak_reserved_gib"] for s in samples),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # Linux reports KiB.
            "prefix_ms": float(np.median([s["prefix_ms"] for s in samples])),
            "denoise_ms": float(np.median([s["denoise_ms"] for s in samples])), "actions": actions,
            "first_action_mae": float(np.mean([np.abs(np.array(a)[0] - frames[i % count].action.numpy()).mean()
                                               for i, a in enumerate(actions)]))}


def deviation(actions: list, reference: list) -> dict[str, float]:
    delta = np.array(actions) - np.array(reference)
    return {"mean_l2": float(np.linalg.norm(delta, axis=-1).mean()), "max_abs": float(np.abs(delta).max())}


def benchmark(configs: list[str], count: int = 8) -> dict:
    """Isolate allocator budgets and failed CUDA contexts in fresh processes."""
    results = {}
    configs = list(dict.fromkeys(c for label in configs for c in
                                 ([label.removesuffix("+trim"), label] if label.endswith("+trim") else [label])))
    bases = list(dict.fromkeys([c for c in configs if not c.startswith("steps-")]
                              + (["nf4", "offload-3.5", "offload-2.5"] if any(c.startswith("steps-") for c in configs) else [])))
    def execute(label, base, steps):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--config", base,
                                  "--steps", str(steps), "--frames", str(count)], capture_output=True, text=True,
                                 env={**os.environ, "PYTHONPATH": "", "HF_HUB_OFFLINE": "1"})
        lines = [line[7:] for line in process.stdout.splitlines() if line.startswith("RESULT=")]
        results[label] = json.loads(lines[-1]) if lines else {"status": "failed", "error": process.stderr[-4000:] or process.stdout[-4000:]}
        print(label, results[label].get("latency_p50_ms", results[label].get("error")), flush=True)
    for base in bases:
        execute(base, base, 10)
    successes = [c for c in bases if results[c]["status"] == "ok" and not c.endswith("+trim")]
    best = min(successes, key=lambda c: results[c]["latency_p50_ms"]) if successes else None
    reference = max(successes, key=lambda c: (c.startswith("offload-"), results[c]["resident_weight_gib"])) if successes else None
    for label in configs:
        if label.startswith("steps-"):
            k = int(label.removeprefix("steps-").removesuffix("+trim"))
            base = best + ("+trim" if label.endswith("+trim") else "") if best else None
            if base in results and k == 10:
                results[label] = dict(results[base])
            elif best:
                execute(label, base, k)
            else:
                results[label] = {"status": "failed", "error": "no successful placement for flow-step sweep"}
    for label, row in results.items():
        if row["status"] == "ok":
            if reference:
                row["vs_reference"] = deviation(row["actions"], results[reference]["actions"])
            if label.startswith("steps-"):
                row["vs_10_steps"] = deviation(row["actions"], results[best]["actions"])
            if label.endswith("+trim"):
                counterpart = results[label.removesuffix("+trim")]
                row["vs_untrimmed"] = (deviation(row["actions"], counterpart["actions"]) if counterpart["status"] == "ok"
                                       else {"error": "untrimmed counterpart failed"})
    for row in results.values():
        row.pop("actions", None)
    report = {"policy": POLICY, "revision": REVISION, "units": "GiB, ms; actions in dataset units",
              "reference": reference, "best_placement": best, "warmups": 2, "seed": "frame ordinal",
              "frames": [{"episode": f.episode, "frame": f.index, "task": f.observation["task"]} for f in load_frames(count)],
              "results": results}
    Path("runs").mkdir(exist_ok=True)
    Path("runs/inference_bench.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, row in results.items():
        if row["status"] == "ok":
            x, y = row["peak_allocated_gib"], row["latency_p50_ms"]
            ax.scatter(x, y, marker="X" if row["trim"] else "o", s=65, label=label)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Trim: X; original: circle")
    failures = [c for c, r in results.items() if r["status"] != "ok"]
    ax.set(xlabel="Peak allocated VRAM (GiB)", ylabel="Chunk latency p50 (ms)", title=f"pi0.5 LIBERO — reference: {reference}")
    fig.text(.02, .01, "Failed: " + ", ".join(failures) if failures else "Two warmups; ≥5 timed calls per configuration", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/inference_bench.png", dpi=150)
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["predict", "benchmark", "worker"], default="predict")
    parser.add_argument("--config", default="nf4")
    parser.add_argument("--trim", action="store_true", help="drop masked camera and use 64 text slots; reject longer prompts")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--steps", type=int, choices=[10, 5, 2], default=10)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark([c.removesuffix("+trim") + "+trim" for c in args.configs] if args.trim else args.configs, args.frames)
        return
    try:
        result = run_config(args.config, args.steps, args.frames if args.mode == "worker" else 1, args.mode == "worker", args.trim)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    if args.mode == "worker":
        print("RESULT=" + json.dumps(result))
    else:
        if result["status"] == "ok":
            frame = load_frames(1)[0]
            print("Task:", frame.observation["task"], "\nChunk shape:", np.array(result["actions"][0]).shape)
            print("First predicted:", result.pop("actions")[0][0], "\nRecorded:", frame.action.tolist())
        print(json.dumps(result, indent=2))
        if result["status"] != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
