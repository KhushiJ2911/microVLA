"""CPU data/checkpoint checks plus one real, optional GPU prediction."""
import io
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
