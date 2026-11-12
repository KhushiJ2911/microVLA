"""CPU data/checkpoint checks plus one real, optional GPU prediction."""
import io
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn

import infer


@pytest.fixture(scope="module")
def cached_policy():
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = infer.checkpoint()
    except LocalEntryNotFoundError:
        pytest.skip("pinned pi0.5 checkpoint is not cached")
    return path, infer.empty_policy(path)


def test_load_frames_cpu(tmp_path, monkeypatch):
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    import huggingface_hub
    png = io.BytesIO()
    Image.new("RGB", (256, 256), (255, 128, 0)).save(png, format="PNG")
    rows = [{"episode_index": episode, "frame_index": frame, "task_index": episode,
             "observation.state": [float(frame)] * 8, "action": [float(episode)] * 7,
             "observation.images.image": {"bytes": png.getvalue(), "path": None},
             "observation.images.image2": {"bytes": png.getvalue(), "path": None}}
            for episode in range(3) for frame in range(4)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "frames.parquet")
    pq.write_table(pa.Table.from_pylist([{"task_index": i, "__index_level_0__": f"task {i}"}
                                       for i in range(3)]), tmp_path / "tasks.parquet")
    def cached(repo, name, **kwargs):
        assert repo == "HuggingFaceVLA/libero"
        assert kwargs == {"repo_type": "dataset", "local_files_only": True}
        return str(tmp_path / ("tasks.parquet" if name.startswith("meta/") else "frames.parquet"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached)
    frames = infer.load_frames(8)
    assert [(f.episode, f.index) for f in frames] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 3), (0, 3), (1, 3)]
    for frame in frames:
        assert frame.observation["task"] == f"task {frame.episode}"
        assert frame.observation["observation.state"].shape == (8,)
        torch.testing.assert_close(frame.action, torch.full((7,), float(frame.episode)))
        image = frame.observation["observation.images.image"]
        assert image.shape == (3, 256, 256) and image.dtype == torch.float32
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([1., 128 / 255, 0.]))
    assert len(infer.load_frames(1)) == 1
    with pytest.raises(ValueError):
        infer.load_frames(0)


def test_cached_checkpoint_shapes_and_buffers(cached_policy):
    from safetensors import safe_open
    path, policy = cached_policy
    with safe_open(path / "model.safetensors", framework="pt") as weights:
        sources = infer.weight_sources(policy, list(weights.keys()))
        for name, param in policy.named_parameters():
            assert param.device.type == "meta"
            assert list(param.shape) == weights.get_slice(sources[name]).get_shape()
        assert all(name.endswith("lm_head.weight") for name in set(weights.keys()) - set(sources.values()))
    assert all(b.device.type == "cpu" and torch.isfinite(b).all() for b in policy.buffers())
    assert policy.config.chunk_size == 50 and policy.config.max_action_dim == 32
    assert not policy.config.compile_model and not policy.config.gradient_checkpointing
    assert isinstance(policy.model.paligemma_with_expert.gemma_expert.lm_head, nn.Identity)


@pytest.mark.parametrize("budget", [2.5, 3.5])
def test_placement_preserves_hot_path_and_workspace(cached_policy, budget):
    _, policy = cached_policy
    units, resident = infer.placement(policy, budget)
    offloaded = [n for n, _ in units if n not in resident]
    assert offloaded and all(n.startswith(infer.PREFIX) for n in offloaded)
    assert all("gemma_expert" not in n and "embed_tokens" not in n for n in offloaded)
    parameters = dict(policy.named_parameters())
    # Units partition the prefix: no dropped or duplicated parameter ownership.
    owned = [n + "." + key for n, module in units for key, _ in module.named_parameters()]
    assert len(owned) == len(set(owned))
    assert set(owned) == {n for n in parameters if n.startswith(infer.PREFIX)}
    resident_bytes = sum(p.numel() * 2 for n, p in parameters.items()
                         if not any(n.startswith(unit + ".") for unit in offloaded))
    assert resident_bytes + 700 * 2**20 <= budget * 2**30
    assert infer.pinned_bytes(units, resident) == sum(p.numel() * 2 for p in parameters.values()) - resident_bytes
    largest_stream = max(sum(p.numel() * 2 for p in module.parameters())
                         for name, module in units if name in offloaded)
    assert largest_stream < 700 * 2**20


def test_saved_processors_cpu(cached_policy):
    path, policy = cached_policy
    pre, post = infer.processors(policy, path, device="cpu")
    frame = infer.Frame({"task": "pick up the mug", "observation.state": torch.zeros(8),
                         "observation.images.image": torch.ones(3, 256, 256),
                         "observation.images.image2": torch.zeros(3, 256, 256)}, torch.zeros(7), 0, 0)
    batch = pre(frame.observation)
    assert batch["observation.language.tokens"].shape == (1, 200)
    assert batch["observation.language.attention_mask"].dtype == torch.bool
    # This LeRobot version serializes the normalized 8D state into the prompt.
    assert batch["observation.state"].shape == (1, 8)
    assert batch["observation.images.image"].shape == (1, 3, 256, 256)
    output = post(torch.zeros(1, 50, 7))
    assert output.shape == (1, 50, 7) and torch.isfinite(output).all()


def test_trimmed_processors_cpu(cached_policy, monkeypatch):
    path, original = cached_policy
    policy = infer.empty_policy(path, trim=True)
    pre, _ = infer.processors(policy, path, device="cpu")
    untrimmed, _ = infer.processors(original, path, device="cpu")
    observation = {"task": "pick up the mug", "observation.state": torch.zeros(8),
                   "observation.images.image": torch.ones(3, 256, 256),
                   "observation.images.image2": torch.zeros(3, 256, 256)}
    batch, baseline = pre(dict(observation)), untrimmed(dict(observation))
    assert policy.config.empty_cameras == 0 and policy.config.tokenizer_max_length == 64
    assert list(policy.config.image_features) == ["observation.images.image", "observation.images.image2"]
    assert batch["observation.language.tokens"].shape == (1, 64)
    for key in ("observation.language.tokens", "observation.language.attention_mask"):
        torch.testing.assert_close(batch[key], baseline[key][:, :64])
    assert not baseline["observation.language.attention_mask"][:, 64:].any()
    # Exercise real image preprocessing without allocating the meta model's weights.
    monkeypatch.setattr(policy, "parameters", lambda: iter([torch.empty(0)]))
    images, masks = policy._preprocess_images(batch)
    assert len(images) == len(masks) == 2 and all(mask.all() for mask in masks)
    assert all(image.shape == (1, 3, 224, 224) for image in images)
    with pytest.raises(ValueError, match=r"pick up the mug.*has \d+ tokens; trim limit is 64"):
        pre({**observation, "task": "pick up the mug " * 80})


@pytest.mark.parametrize("configs", [infer.DEFAULT_CONFIGS, ["steps-5+trim"], ["nf4+trim"]])
def test_benchmark_trim_pairing_cpu(tmp_path, monkeypatch, configs):
    monkeypatch.setattr(infer, "check_pinned", lambda needed: None)
    monkeypatch.setattr(infer, "checkpoint", lambda: tmp_path)
    monkeypatch.setattr(infer, "empty_policy", lambda *args: nn.Linear(1, 1, device="meta"))
    monkeypatch.setattr(infer, "placement", lambda *args: ([], set()))
    calls = []
    def worker(command, **kwargs):
        placement = command[command.index("--config") + 1]
        steps = int(command[command.index("--steps") + 1])
        calls.append((placement, steps))
        assert kwargs["env"]["PYTHONPATH"] == "" and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        trimmed = placement.endswith("+trim")
        action = steps + (.01 if trimmed else 0)
        row = {"status": "ok", "placement": placement, "steps": steps, "trim": trimmed,
               "latency_p50_ms": 1 if trimmed else (2 if placement == "nf4" else 3),
               "resident_weight_gib": 2, "peak_allocated_gib": 2,
               "prefix_tokens": {"images": 512 if trimmed else 768, "text": 64 if trimmed else 200},
               "actions": [[[action, action]]], "samples": []}
        return SimpleNamespace(stdout="RESULT=" + json.dumps(row), stderr="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(infer, "subprocess", SimpleNamespace(run=worker))
    monkeypatch.setattr(infer, "load_frames", lambda count: [infer.Frame({"task": "mug"}, torch.zeros(7), 0, 0)])
    report = infer.benchmark(configs, count=1)
    assert report["best_placement"] == ("nf4+trim" if "nf4+trim" in report["results"] else "nf4")
    for label, row in report["results"].items():
        assert row["actions"]
        if label.endswith("+trim"):
            assert row["vs_untrimmed"]["max_abs"] == pytest.approx(.01)
            assert row["vs_untrimmed"]["mean_l2"] == pytest.approx(2**.5 * .01)
            assert row["prefix_tokens"] == {"images": 512, "text": 64}
            if label.startswith("steps-"):
                assert row["placement"] == "nf4+trim" and ("nf4+trim", row["steps"]) in calls
                assert row["vs_10_steps"]["max_abs"] == pytest.approx(10 - row["steps"] - .01)
    assert (tmp_path / "runs/inference_bench.json").is_file()
    if configs == infer.DEFAULT_CONFIGS:
        first_offload = next(i for i, (base, _) in enumerate(calls) if base.startswith("offload-"))
        assert all(steps == 10 for _, steps in calls[first_offload:])
        assert ("nf4+trim", 2) in calls
    else:
        assert not any(base.startswith("offload-") for base, _ in calls)
    assert (tmp_path / "assets/inference_bench.png").is_file()


@pytest.fixture
def cpu_bench(tmp_path, monkeypatch):
    calls = []
    def worker(command, **kwargs):
        base = command[command.index("--config") + 1]
        steps = int(command[command.index("--steps") + 1])
        calls.append((base, steps))
        row = {"status": "ok", "placement": base, "steps": steps, "trim": base.endswith("+trim"),
               "latency_p50_ms": 1 if base.startswith("offload-") else 2,
               "resident_weight_gib": 2, "peak_allocated_gib": 2, "actions": [[[float(steps) + int(base.startswith("offload-"))]]]}
        return SimpleNamespace(stdout="RESULT=" + json.dumps(row), stderr="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(infer, "subprocess", SimpleNamespace(run=worker))
    monkeypatch.setattr(infer, "load_frames", lambda count: [infer.Frame({"task": "mug"}, torch.zeros(7), 0, i) for i in range(count)])
    monkeypatch.setattr(infer, "checkpoint", lambda: tmp_path)
    monkeypatch.setattr(infer, "empty_policy", lambda *args: nn.Linear(4, 4, bias=False, device="meta"))
    monkeypatch.setattr(infer, "placement", lambda policy, budget: ([("prefix", policy)], set()))
    return calls, tmp_path / "runs/inference_bench.json"


def mock_meminfo(monkeypatch, available_gib):
    original = infer.Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemTotal: 16777216 kB\nMemFree: 0 kB\nMemAvailable: {int(available_gib * 2**20)} kB\n"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(infer.Path, "read_text", read)


def test_benchmark_resume_and_fresh(cpu_bench, monkeypatch):
    calls, path = cpu_bench
    infer.benchmark(["nf4+trim", "steps-5+trim", "steps-2+trim"], count=1)
    first = json.loads(path.read_text())
    assert all(base.startswith("nf4") for base, _ in calls)
    assert all(row["actions"] for row in first["results"].values())
    calls.clear()
    assert infer.benchmark(["nf4+trim", "steps-5+trim", "steps-2+trim"], count=1) == first
    assert calls == []
    infer.benchmark(["nf4+trim"], count=1, fresh=True)
    assert calls == [("nf4", 10), ("nf4+trim", 10)]
    # CLI passes --fresh through to the coordinator.
    monkeypatch.setattr(infer.sys, "argv", ["infer.py", "--mode", "benchmark", "--configs", "nf4", "--fresh"])
    monkeypatch.setattr(infer, "benchmark", lambda configs, count, fresh: calls.append((configs, fresh)))
    infer.main()
    assert calls[-1] == (["nf4"], True)


@pytest.mark.parametrize("change", ["policy", "revision", "frame_count"])
def test_benchmark_invalidates_incompatible_cache(cpu_bench, change):
    calls, path = cpu_bench
    infer.benchmark(["nf4"], count=1)
    cached = json.loads(path.read_text())
    cached[change] = "different"
    path.write_text(json.dumps(cached))
    calls.clear()
    infer.benchmark(["nf4"], count=1)
    assert calls == [("nf4", 10)]


@pytest.mark.parametrize("status", ["ok", "failed", "skipped"])
def test_benchmark_saves_before_next_worker_and_resumes(cpu_bench, monkeypatch, status):
    calls, path = cpu_bench
    worker = infer.subprocess.run
    replace = infer.Path.replace
    replacements = []
    def atomic(source, target):
        assert source != target and source.parent == target.parent
        json.loads(source.read_text())
        replacements.append(target)
        return replace(source, target)
    def interrupted(command, **kwargs):
        if len(calls):
            cached = json.loads(path.read_text())
            assert cached["results"]["nf4"]["status"] == status
            assert replacements == [path.relative_to(path.parents[1])]
            if status == "ok":
                assert cached["results"]["nf4"]["vs_reference"]["max_abs"] == 0
            raise KeyboardInterrupt
        result = worker(command, **kwargs)
        return result if status == "ok" else SimpleNamespace(stdout="RESULT=" + json.dumps({"status": status}), stderr="")
    monkeypatch.setattr(infer.Path, "replace", atomic)
    monkeypatch.setattr(infer.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        infer.benchmark(["nf4", "offload-3.5"], count=1)
    monkeypatch.setattr(infer.subprocess, "run", worker)
    calls.clear()
    report = infer.benchmark(["nf4"], count=1)
    assert calls == [] and report["results"]["nf4"]["status"] == status


def test_benchmark_reuses_legacy_rows_without_actions(cpu_bench):
    calls, path = cpu_bench
    infer.benchmark(["nf4"], count=1)
    cached = json.loads(path.read_text())
    cached["results"]["nf4"].pop("actions")
    path.write_text(json.dumps(cached))
    calls.clear()
    report = infer.benchmark(["nf4", "steps-5"], count=1)
    assert calls == [("nf4", 5)]
    assert report["results"]["nf4"] == cached["results"]["nf4"]
    assert "actions unavailable" in report["results"]["steps-5"]["vs_reference"]["error"]


def test_offload_preflight_skips_and_continues(cpu_bench, monkeypatch):
    calls, path = cpu_bench
    mock_meminfo(monkeypatch, 2)  # All MemAvailable is reserved; even 32 pinned bytes must be refused.
    report = infer.benchmark(["offload-2.5", "nf4", "steps-2+trim"], count=1)
    skipped = report["results"]["offload-2.5"]
    assert skipped["status"] == "skipped" and "needs" in skipped["reason"]
    assert "GiB pinned, 0.00 GiB available" in skipped["reason"]
    assert all(base.startswith("nf4") for base, _ in calls)
    assert json.loads(path.read_text())["results"]["offload-2.5"] == skipped
    calls.clear()
    mock_meminfo(monkeypatch, 16)
    infer.benchmark(["offload-2.5"], count=1)
    assert calls == []  # Skips are cached, too; --fresh is the explicit retry.
    infer.benchmark(["offload-2.5"], count=1, fresh=True)
    assert calls == [("offload-2.5", 10)]


def test_pinned_estimate_and_memavailable(monkeypatch):
    modules = [("offloaded", nn.Linear(1024, 1024, bias=False, device="meta")),
               ("resident", nn.Linear(32, 32, device="meta"))]
    needed = infer.pinned_bytes(modules, {"resident"})
    assert needed == 2 * 1024**2
    mock_meminfo(monkeypatch, 2 + needed / 2**30)
    infer.check_pinned(needed)  # Exact fit after the reserve; MemFree is deliberately zero.
    mock_meminfo(monkeypatch, 2 + needed / 2**31)
    with pytest.raises(infer.OffloadRAMError, match="GiB pinned"):
        infer.check_pinned(needed)


def test_worker_rechecks_before_pinning(cpu_bench, monkeypatch, capsys):
    import safetensors
    mock_meminfo(monkeypatch, 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=4 * 2**30))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    model = nn.Module()
    model.prefix = nn.Linear(4, 4, bias=False, device="meta")
    monkeypatch.setattr(infer, "empty_policy", lambda *args: model)
    monkeypatch.setattr(infer, "placement", lambda *args: ([("prefix", model.prefix)], set()))
    class Weights:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return ["prefix.weight"]
        def get_tensor(self, name): return torch.zeros(4, 4)
    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: Weights())
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: pytest.fail("must check before pinning"))
    with pytest.raises(infer.OffloadRAMError):
        infer.load_policy("offload-2.5")
    monkeypatch.setattr(infer.sys, "argv", ["infer.py", "--mode", "worker", "--config", "offload-2.5"])
    monkeypatch.setattr(infer, "load_frames", lambda count: [])
    infer.main()
    result = json.loads(capsys.readouterr().out.removeprefix("RESULT="))
    assert result["status"] == "skipped" and "GiB pinned" in result["reason"]


def test_steps_use_only_completed_or_cached_placements(cpu_bench, monkeypatch):
    calls, _ = cpu_bench
    mock_meminfo(monkeypatch, 16)
    infer.benchmark(["steps-5", "offload-3.5"], count=1)
    assert calls == [("nf4", 10), ("nf4", 5), ("offload-3.5", 10)]
    calls.clear()
    report = infer.benchmark(["steps-2+trim"], count=1)
    assert calls == [("offload-3.5", 2), ("offload-3.5+trim", 2)]
    assert report["results"]["steps-5"]["placement"] == "nf4"
    assert report["results"]["steps-5"]["vs_10_steps"]["max_abs"] == 5
    assert report["results"]["steps-2"]["placement"] == "offload-3.5"
    assert report["results"]["nf4"]["vs_reference"]["max_abs"] == 1
    assert report["results"]["steps-5"]["vs_reference"]["max_abs"] == 6


def test_deviation():
    assert infer.deviation([[[3., 4.], [0., 0.]]], [[[0., 0.], [0., 0.]]]) == {"mean_l2": 2.5, "max_abs": 4.}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nf4_predict_smoke(cached_policy):
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        frame = infer.load_frames(1)[0]
    except LocalEntryNotFoundError:
        pytest.skip("LIBERO observations are not cached")
    policy, pre, post, _ = infer.load_policy("nf4")
    actions, metrics = infer.predict(policy, pre, post, frame)
    assert actions.shape == (50, 7) and torch.isfinite(actions).all()
    assert metrics["peak_allocated_gib"] < 4
    assert metrics["peak_reserved_gib"] < 4
    assert metrics["prefix_ms"] > 0 and metrics["denoise_ms"] > 0
