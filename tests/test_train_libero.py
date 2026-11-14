import io
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import eval_libero as ev
import train_libero as tl
from train import Config, MicroVLA


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    return Config(image_size=16, num_cameras=2, vocab_size=8, horizon=4, action_dim=7, state_dim=8,
                  width=32, depth=2, heads=2, head_dim=8, mlp=64, expert_width=16, expert_mlp=32,
                  vit_width=16, vit_depth=1, vit_heads=2, **kwargs)


def png(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return dict(bytes=buf.getvalue(), path=None)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    texts = ["move mug left", "move mug right"]
    pq.write_table(pa.Table.from_pylist([dict(task_index=i + 10, __index_level_0__=s) for i, s in enumerate(texts)] +
                                      [dict(task_index=21, __index_level_0__="another suite")]), root / "meta/tasks.parquet")
    episodes = []
    for file in range(2):
        rows = []
        for ep in range(file * 4, file * 4 + 4):
            # Deliberately wrong metadata, as in HuggingFaceVLA/libero @ 86958911.
            episodes.append({"episode_index": ep, "data/chunk_index": 0, "data/file_index": 55 + file,
                             "length": 3, "tasks": [texts[ep % 2]]})
            for frame in range(3):
                im = np.zeros((16, 16, 3), np.uint8)
                im[:8, :8] = [200, ep, frame]
                rows.append({tl.CAMERAS[0]: png(im), tl.CAMERAS[1]: png(im + 3),
                             "observation.state": [ep + frame / 10] * 8,
                             "action": [ep + frame / 10] * 6 + [float(2 * (frame % 2) - 1)],
                             "episode_index": ep, "frame_index": frame, "task_index": 10 + ep % 2})
        if file == 0:
            unrelated = {**rows[0], "episode_index": 999, "task_index": 21}
            unrelated.update({c: dict(bytes=b"not an image", path=None) for c in tl.CAMERAS})
            rows.insert(0, unrelated)  # Must be skipped even though the file contains Spatial rows.
        pq.write_table(pa.Table.from_pylist(rows), root / f"data/chunk-000/file-{308 + file:03}.parquet")
    episodes.append({"episode_index": 999, "data/chunk_index": 0, "data/file_index": 55,
                     "length": 1, "tasks": ["another suite"]})
    # Unrelated files need only the discovery columns; attempting full reads fails.
    pq.write_table(pa.Table.from_pylist([dict(episode_index=1000, task_index=21)]),
                   root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    return root, tmp_path / "cache", texts


def build(source):
    root, cache, texts = source
    return tl.preprocess(root, cache, 16, texts)


def test_cache_decode_orientation_stats_and_episode_split(source, monkeypatch):
    projections, original = [], pq.ParquetFile.iter_batches
    def tracked(self, *args, **kwargs):
        projections.append(kwargs.get("columns"))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", tracked)
    source_files = sorted(p.relative_to(source[0]) for p in source[0].rglob("*"))
    meta = build(source)
    assert projections[:3] == [["episode_index", "task_index"]] * 3
    assert set(meta["identity"]["files"]) == {f"data/chunk-000/file-{i}.parquet" for i in (308, 309)}
    assert all("data/file_index" not in ep for ep in meta["identity"]["episodes"])
    assert sorted(p.relative_to(source[0]) for p in source[0].rglob("*")) == source_files
    assert {p.name for p in source[1].iterdir()} == {"manifest.json", *(f"{k}.bin" for k in tl.array_specs(24, 16))}
    arrays = tl.open_arrays(source[1], meta)
    assert meta["frames"] == 24 and len(meta["val_episodes"]) == 2
    assert meta["cache_bytes"] == sum(p.stat().st_size for p in source[1].glob("*.bin"))
    assert arrays["images"].dtype == np.uint8 and arrays["images"].shape == (24, 2, 3, 16, 16)
    np.testing.assert_array_equal(arrays["images"][0, 0, :, 0, 0], [200, 0, 0])
    np.testing.assert_array_equal(arrays["images"][0, 0, :, -1, -1], 0)
    assert arrays["actions"][0, -1] == -1  # No gripper inversion.
    assert arrays["actions"][1, -1] == 1
    train = ~np.isin(arrays["episode"], meta["val_episodes"])
    assert meta["norm"]["state"] == tl.fit_norm(arrays["state"][train])
    data = tl.Batches(source[1], 6)
    assert not set(arrays["episode"][data.indices["train"]]) & set(arrays["episode"][data.indices["val"]])
    images, state, instr, actions = data.batch(100, torch.device("cpu"))
    assert images.shape == (100, 2, 3, 16, 16) and instr.shape == (100, 3)
    raw = tl.unnormalize(actions, meta["norm"]["action"])
    # Padding repeats the final action rather than taking actions from the next episode.
    assert torch.allclose(raw[:, 2:, :6], raw[:, 2:3, :6].expand(-1, 4, -1))
    assert torch.isfinite(state).all()
    before = (source[1] / "manifest.json").read_bytes()
    monkeypatch.setattr(tl, "decode_image", lambda *a: pytest.fail("idempotent cache decoded again"))
    assert build(source) == meta
    assert (source[1] / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("split_episode", [False, True])
def test_cache_replays_only_interrupted_file(source, monkeypatch, split_episode):
    first = source[0] / "data/chunk-000/file-308.parquet"
    second = source[0] / "data/chunk-000/file-309.parquet"
    if split_episode:
        rows = pq.read_table(first).to_pylist()
        moved = rows.pop()
        pq.write_table(pa.Table.from_pylist(rows), first)
        pq.write_table(pa.Table.from_pylist([moved, *pq.read_table(second).to_pylist()]), second)
    first_calls = (12 - int(split_episode)) * 2
    original, calls = tl.decode_image, []
    def interrupted(*args):
        calls.append(1)
        if len(calls) == first_calls + 2:  # First file has committed.
            raise KeyboardInterrupt
        return original(*args)
    monkeypatch.setattr(tl, "decode_image", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build(source)
    assert json.loads((source[1] / "manifest.json").read_text())["completed_files"] == ["data/chunk-000/file-308.parquet"]
    calls.clear()
    monkeypatch.setattr(tl, "decode_image", lambda *a: calls.append(1) or original(*a))
    assert build(source)["complete"]
    assert len(calls) == (12 + int(split_episode)) * 2
    arrays = tl.open_arrays(source[1], json.loads((source[1] / "manifest.json").read_text()))
    np.testing.assert_array_equal(arrays["episode"], np.repeat(np.arange(8), 3))
    np.testing.assert_array_equal(arrays["frame"], np.tile(np.arange(3), 8))


def test_cache_lists_missing_and_short_episodes_before_allocating(source):
    root, cache, texts = source
    file = root / "data/chunk-000/file-308.parquet"
    rows = pq.read_table(file).to_pylist()
    rows = [r for r in rows if r["episode_index"] != 0 and
            not (r["episode_index"] == 1 and r["frame_index"] == 0)]
    pq.write_table(pa.Table.from_pylist(rows), file)
    with pytest.raises(ValueError, match=r"before decoding.*0: 0/3 \(missing\); 1: 2/3 \(short\)"):
        build(source)
    assert not cache.exists()


def test_cache_validates_counts_after_decoding_before_committing(source, monkeypatch):
    original = pq.ParquetFile.iter_batches
    def dropped(self, *args, **kwargs):
        for batch in original(self, *args, **kwargs):
            if tl.CAMERAS[0] in kwargs["columns"]:
                keep = [i for i, r in enumerate(batch.to_pylist()) if r["episode_index"] != 0 and
                        not (r["episode_index"] == 1 and r["frame_index"] == 0)]
                batch = batch.take(pa.array(keep, type=pa.int64()))
            yield batch
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", dropped)
    with pytest.raises(ValueError, match=r"after decoding.*0: 0/3 \(missing\); 1: 2/3 \(short\)"):
        build(source)
    manifest = json.loads((source[1] / "manifest.json").read_text())
    assert not manifest["complete"] and not manifest["completed_files"]


@pytest.mark.parametrize("problem", ["task", "duplicate", "range", "cross_file_duplicate"])
def test_cache_rejects_invalid_spatial_rows(source, problem):
    file = source[0] / "data/chunk-000/file-308.parquet"
    rows = pq.read_table(file).to_pylist()
    row = next(r for r in rows if r["episode_index"] == 0 and r["frame_index"] == 1)
    if problem == "task":
        row["task_index"] = 21
    elif problem == "range":
        row["frame_index"] = 3
    else:
        row["frame_index"] = 0
        if problem == "cross_file_duplicate":
            rows.remove(row)
            other = source[0] / "data/chunk-000/file-309.parquet"
            pq.write_table(pa.Table.from_pylist([row, *pq.read_table(other).to_pylist()]), other)
    pq.write_table(pa.Table.from_pylist(rows), file)
    with pytest.raises(ValueError, match="Spatial episode 0"):
        build(source)
    if problem == "task":
        assert not source[1].exists()
    else:
        assert not json.loads((source[1] / "manifest.json").read_text())["complete"]


def test_cache_rejects_changed_configuration(source):
    build(source)
    with pytest.raises(ValueError, match="changed"):
        tl.preprocess(source[0], source[1], 24, source[2])


@pytest.fixture
def remote_source(source, monkeypatch):
    """Exercise the real streaming adapter without any network or shared HF cache."""
    import huggingface_hub as hub
    root, cache, _ = source
    prefix = f"datasets/{tl.REPO}@{tl.REVISION}/"
    downloads, projections = [], []
    class FileSystem:
        def glob(self, pattern):
            assert pattern.startswith(prefix)
            return [prefix + p.relative_to(root).as_posix() for p in root.glob(pattern[len(prefix):])]
        def size(self, path):
            return (root / path.removeprefix(prefix)).stat().st_size
        def open(self, path, mode, **kwargs):
            assert kwargs == dict(cache_type="none", block_size=64 * 1024)
            return (root / path.removeprefix(prefix)).open(mode)
    def download(repo, file, *, repo_type, revision, local_dir):
        assert (repo, repo_type, revision) == (tl.REPO, "dataset", tl.REVISION)
        assert hub.constants.HF_HUB_DISABLE_XET
        assert local_dir == cache / ".stream-source" and not local_dir.exists()
        downloads.append(file)
        target = local_dir / file
        target.parent.mkdir(parents=True)
        shutil.copyfile(root / file, target)
        return str(target)
    original = pq.ParquetFile.iter_batches
    def tracked(self, *args, **kwargs):
        projections.append(kwargs["columns"])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(hub, "HfFileSystem", FileSystem)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", tracked)
    return downloads, projections


def test_stream_matches_local_arrays_and_metadata_at_128(source, remote_source, monkeypatch):
    root, cache, texts = source
    # An episode can span files, and physical row order need not be frame order.
    first, second = [root / f"data/chunk-000/file-{i}.parquet" for i in (308, 309)]
    rows = pq.read_table(first).to_pylist()
    moved = rows.pop()
    pq.write_table(pa.Table.from_pylist(rows[::-1]), first)
    pq.write_table(pa.Table.from_pylist([moved, *pq.read_table(second).to_pylist()][::-1]), second)
    local_cache = cache.with_name("local-cache")
    local = tl.preprocess(root, local_cache, 128, texts)
    downloads, projections = remote_source
    projections.clear()
    streamed = tl.preprocess(cache=cache, image_size=128, task_texts=texts, stream=True)
    assert projections[:3] == [["episode_index", "task_index"]] * 3
    assert downloads == [f"data/chunk-000/file-{i}.parquet" for i in (308, 309)]
    assert not (cache / ".stream-source").exists()
    for key in tl.array_specs(24, 128):
        assert (cache / f"{key}.bin").read_bytes() == (local_cache / f"{key}.bin").read_bytes()
    for key in ("frames", "image_size", "completed_files", "complete", "val_episodes", "norm", "tokenizer", "cache_bytes"):
        assert streamed[key] == local[key]
    assert streamed["identity"]["episodes"] == local["identity"]["episodes"]
    assert streamed["identity"]["source"] == f"datasets/{tl.REPO}@{tl.REVISION}/"
    data = tl.Batches(cache, 4)
    assert data.batch(2, torch.device("cpu"))[0].shape == (2, 2, 3, 128, 128)
    downloads.clear()
    monkeypatch.setattr(tl, "decode_image", lambda *a: pytest.fail("complete cache decoded again"))
    assert tl.preprocess(cache=cache, image_size=128, task_texts=texts, stream=True) == streamed
    assert not downloads


@pytest.mark.parametrize("stage", ["download", "decode", "commit"])
def test_stream_resume_skips_finished_downloads_and_cleans_partial(source, remote_source, monkeypatch, stage):
    import huggingface_hub as hub
    _, cache, texts = source
    original_decode, original_download, original_json = tl.decode_image, hub.hf_hub_download, tl.atomic_json
    downloads, projections = remote_source
    calls = []
    def decode(*args):
        calls.append(1)
        if stage == "decode" and len(calls) == 26:
            raise KeyboardInterrupt
        return original_decode(*args)
    def download(*args, **kwargs):
        path = original_download(*args, **kwargs)
        if stage == "download" and len(downloads) == 2:
            Path(path).with_suffix(".incomplete").write_bytes(b"partial")
            raise KeyboardInterrupt
        return path
    def commit(path, manifest):
        if stage == "commit" and len(manifest["completed_files"]) == 2:
            raise KeyboardInterrupt
        return original_json(path, manifest)
    monkeypatch.setattr(tl, "decode_image", decode)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(tl, "atomic_json", commit)
    with pytest.raises(KeyboardInterrupt):
        tl.preprocess(cache=cache, image_size=128, task_texts=texts, stream=True)
    meta = json.loads((cache / "manifest.json").read_text())
    assert meta["completed_files"] == ["data/chunk-000/file-308.parquet"] and not meta["complete"]
    assert "fingerprint" not in meta and not (cache / ".stream-source").exists()
    with pytest.raises(ValueError, match="preprocess"):
        tl.Batches(cache, 4)
    # Simulate SIGKILL leaving a stale download and only enough free space for
    # remaining physical blocks plus the reserve (less than a fresh build).
    stale = cache / ".stream-source/stale.incomplete"
    stale.parent.mkdir()
    stale.write_bytes(b"partial")
    allocated = sum(p.stat().st_blocks * 512 for p in cache.glob("*.bin"))
    needed = sum(np.dtype(d).itemsize * np.prod(s) for d, s in tl.array_specs(24, 128).values())
    monkeypatch.setattr(tl.shutil, "disk_usage", lambda p: SimpleNamespace(free=max(0, needed - allocated) + 2**30))
    monkeypatch.setattr(tl, "decode_image", original_decode)
    monkeypatch.setattr(hub, "hf_hub_download", original_download)
    monkeypatch.setattr(tl, "atomic_json", original_json)
    downloads.clear()
    projections.clear()
    meta = tl.preprocess(cache=cache, image_size=128, task_texts=texts, stream=True)
    assert meta["complete"] and meta["fingerprint"]
    assert downloads == ["data/chunk-000/file-309.parquet"]
    assert ["episode_index", "frame_index", "task_index"] in projections
    assert not stale.parent.exists()
    arrays = tl.open_arrays(cache, meta)
    np.testing.assert_array_equal(arrays["episode"], np.repeat(np.arange(8), 3))
    np.testing.assert_array_equal(arrays["frame"], np.tile(np.arange(3), 8))


@pytest.mark.parametrize("stream", [False, True])
def test_disk_reserve_refuses_before_allocation_or_download(source, remote_source, monkeypatch, stream):
    root, cache, texts = source
    needed = sum(np.dtype(d).itemsize * np.prod(s) for d, s in tl.array_specs(24, 128).values())
    monkeypatch.setattr(tl.shutil, "disk_usage", lambda p: SimpleNamespace(free=needed + 2**30 - 1))
    with pytest.raises(ValueError, match="Insufficient free disk.*plus 1 GiB"):
        tl.preprocess(root, cache, 128, texts, stream=stream)
    assert not cache.exists() and not remote_source[0]


def test_stream_cli_defaults_to_separate_cache(monkeypatch):
    calls = []
    def preprocess(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(frames=24, cache_bytes=1, image_size=128, val_episodes=[])
    monkeypatch.setattr(tl, "preprocess", preprocess)
    monkeypatch.setattr("sys.argv", ["train_libero.py", "--preprocess", "--stream", "--image-size", "128"])
    tl.main()
    assert calls == [((tl.DATASET, Path("runs/libero_cache_128"), 128), dict(seed=0, stream=True))]


def test_tokenizer_padding_unknown_and_round_trip():
    tok = tl.tokenizer(["Move the mug left", "move right"])
    assert tok["vocab"][:2] == ["<pad>", "<unk>"]
    assert tok == tl.tokenizer(["move right", "Move the mug left"])
    ids = tl.tokenize(["MOVE right", "alien"], tok)
    assert ids.shape == (2, 4) and ids[0, 2:].tolist() == [0, 0] and ids[1, 0] == 1
    with pytest.raises(ValueError, match="exceeds"):
        tl.tokenize(["one two three four five"], tok)
    x = torch.randn(20, 7)
    norm = tl.fit_norm(x.numpy())
    torch.testing.assert_close(tl.unnormalize(tl.normalize(x, norm), norm), x)
    assert np.isfinite(tl.fit_norm(np.zeros((3, 8)))["scale"]).all()


@pytest.mark.parametrize("vocab,unk", [(["<pad>", "<unk>", "move"], 1),
                                      (["<pad>", "move", "<unk>"], 2),
                                      (["<pad>", "move"], 0)])
def test_inference_oov_reuses_reserved_id(vocab, unk):
    tok = dict(vocab=vocab, length=3)
    ids = tl.tokenize(["MOVE novel", "alien alien alien alien"], tok, truncate=True)
    np.testing.assert_array_equal(ids, [[vocab.index("move"), unk, 0], [unk] * 3])
    assert ids.dtype == np.int64 and tok == dict(vocab=vocab, length=3)


def test_inference_spatial_tokens_are_bit_identical():
    texts = [f"pick up the black bowl {position} the {object} and place it on the plate"
             for position in ("between", "next to")
             for object in ("plate", "ramekin", "cookie box", "stove", "wooden cabinet")]
    tok = tl.tokenizer(texts)
    # Original encoding: exact word IDs, zero padding, saved length/dtype.
    lookup = {w: i for i, w in enumerate(tok["vocab"])}
    expected = np.zeros((len(texts), tok["length"]), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [lookup.get(w, 1) for w in tl.words(text)]
        expected[i, :len(ids)] = ids
    for truncate in (False, True):
        actual = tl.tokenize(texts, tok, truncate=truncate)
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def test_inference_truncation_keeps_prefix_and_training_length_check():
    tok = tl.tokenizer(["move mug left"])
    text = "MOVE mug left past a plate"
    np.testing.assert_array_equal(tl.tokenize([text], tok, truncate=True), tl.tokenize(["move mug left"], tok))
    with pytest.raises(ValueError, match="exceeds saved tokenizer length"):
        tl.tokenize([text], tok)


def test_two_camera_forward_loss_and_cached_sampling():
    c = config()
    model = MicroVLA(c)
    images = torch.randint(256, (2, 2, 3, 16, 16), dtype=torch.uint8)
    state, instr, actions = torch.randn(2, 8), torch.tensor([[1, 2, 0], [1, 3, 4]]), torch.randn(2, 4, 7)
    prefix, valid = model.embed_prefix(images, state, instr)
    assert prefix.shape == (2, 8 + 3 + 8, 32) and valid.shape == prefix.shape[:2]
    assert not valid[0, 10] and valid[0, :8].all()
    assert model(images, state, instr, actions, torch.ones(2)).shape == (2, 4, 7)
    loss = model.loss(images, state, instr, actions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.camera_embedding.grad is not None
    # Activate adaptive gates to exercise joint attention, not only the initial heads.
    for layer in model.layers:
        for norm in (layer.action.attn_norm, layer.action.ffn_norm):
            nn = norm.modulation
            torch.nn.init.normal_(nn.weight, std=.1)
            torch.nn.init.constant_(nn.bias, .2)
    cached = model.sample(images, state, instr, 2, actions)
    full = model.sample(images, state, instr, 2, actions, cached=False)
    torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-5)
    shifted = tl.augment(images)
    assert shifted.dtype == images.dtype and shifted.shape == images.shape


def test_ema_math_independent_copy_and_frozen_weights():
    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(2)
    model.bias.data.fill_(7)
    model.bias.requires_grad_(False)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.weight.data.fill_(6)
    model.bias.data.fill_(9)
    tl.update_ema(ema, model, .75)
    torch.testing.assert_close(ema["weight"], torch.full_like(model.weight, 3))
    tl.update_ema(ema, model, .75)
    torch.testing.assert_close(ema["weight"], torch.full_like(model.weight, 3.75))
    assert ema["bias"].item() == 7 and not ema["weight"].requires_grad
    assert model.weight[0, 0].item() == 6


@pytest.mark.parametrize("use_ema", [False, True])
def test_tiny_policy_loads_ema_or_legacy_raw_weights(use_ema):
    ckpt = fake_checkpoint(dict(tokenizer=tl.tokenizer(["move mug"]), norm={}))
    if use_ema:
        ckpt["ema"] = {k: v.clone().add_(.125) for k, v in ckpt["model"].items()}
    policy = tl.TinyPolicy(ckpt, "cpu")
    expected = ckpt["ema"] if use_ema else ckpt["model"]
    for k, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, expected[k], atol=0, rtol=0)


def test_color_jitter_is_small_per_sample_and_preserves_input():
    images = torch.arange(48, 208, dtype=torch.uint8).reshape(1, 1, 1, 10, 16).expand(12, 2, 3, -1, -1).clone()
    before = images.clone()
    jittered = tl.color_jitter(images)
    assert jittered.dtype == images.dtype and jittered.shape == images.shape
    assert torch.equal(images, before) and not torch.equal(images, jittered)
    assert torch.equal(jittered[:, 0], jittered[:, 1])
    assert not torch.equal(jittered[0], jittered[1])
    assert (jittered.float() - images.float()).abs().max() < 40


@pytest.mark.parametrize("phase", ["distill", "train"])
def test_checkpoint_resume_restores_optimizer_rng_and_progress(source, phase):
    meta = build(source)
    model = MicroVLA(config())
    head = torch.nn.Linear(32, 4) if phase == "distill" else None
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.AdamW(params, lr=.001)
    opt.zero_grad()
    sum(p.square().sum() for p in params).backward()
    opt.step()
    rng = np.random.default_rng(42)
    path = source[1] / "latest.pt"
    tl.save_checkpoint(path, model, opt, phase, 250, meta, {"steps": 300}, rng, head)
    saved = torch.load(path, weights_only=True)
    expected_np, expected_torch = rng.random(8), torch.rand(8)
    expected_params = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p in params:
            p.zero_()
    rng.random(20)
    torch.rand(20)
    assert tl.restore_checkpoint(saved, model, opt, rng, head) == 251
    np.testing.assert_array_equal(rng.random(8), expected_np)
    torch.testing.assert_close(torch.rand(8), expected_torch)
    for actual, expected in zip(params, expected_params):
        torch.testing.assert_close(actual, expected)
    assert opt.state_dict()["state"] and saved["norm"] == meta["norm"] and saved["tokenizer"] == meta["tokenizer"]
    assert not path.with_suffix(".tmp").exists()


def fake_checkpoint(meta):
    c = config()
    c.vocab_size = len(meta["tokenizer"]["vocab"])
    return dict(config=asdict(c), model=MicroVLA(c).state_dict(), norm=meta["norm"], tokenizer=meta["tokenizer"], phase="train")


def test_adapter_handles_zero_shot_text_and_reports_full_instruction_oov():
    tok = tl.tokenizer(["move mug left"])
    meta = dict(tokenizer=tok, norm={k: dict(center=[0.] * n, scale=[1.] * n)
                                   for k, n in (("state", 8), ("action", 7))})
    texts = ["move mug left", "MOVE alien mug beyond limit", "", "alien"]
    raw = {key: np.zeros((4, 16, 16, 3), dtype=np.uint8) for key in tl.CAMERAS}
    raw.update({"observation.state": np.zeros((4, 8), dtype=np.float32), "task": texts})
    policy = tl.TinyPolicy(fake_checkpoint(meta), "cpu")
    instr = policy.prepare(raw)[2]
    lookup = {w: i for i, w in enumerate(tok["vocab"])}
    expected = [[lookup["move"], lookup["mug"], lookup["left"]],
                [lookup["move"], lookup["<unk>"], lookup["mug"]], [0, 0, 0], [lookup["<unk>"], 0, 0]]
    torch.testing.assert_close(instr, torch.tensor(expected), atol=0, rtol=0)
    assert [policy.instruction_tokenization(t) for t in texts] == [
        dict(oov_fraction=0., truncated=False), dict(oov_fraction=.6, truncated=True),
        dict(oov_fraction=0., truncated=False), dict(oov_fraction=1., truncated=False)]
    assert tok == tl.tokenizer(["move mug left"])


@pytest.mark.parametrize("ablation", ["blind", "no-wrist", "wrong-text", "no-state"])
@pytest.mark.parametrize("canonical", [False, True])
def test_input_ablation_changes_only_requested_tensor(monkeypatch, ablation, canonical):
    texts = [f"move mug {i}" for i in range(10)]
    monkeypatch.setattr(tl, "spatial_tasks", lambda: texts)
    meta = dict(tokenizer=tl.tokenizer(texts), norm={
        k: dict(center=[3.] * n, scale=[2.] * n) for k, n in (("state", 8), ("action", 7))})
    ckpt = fake_checkpoint(meta)
    raw = {key: np.full((10, 24, 24, 3), 60 + i * 90, dtype=np.uint8)
           for i, key in enumerate(tl.CAMERAS)}
    raw.update({"observation.state": np.arange(80, dtype=np.float32).reshape(10, 8), "task": texts.copy()})
    if canonical:
        raw = {k: torch.from_numpy(v).permute(0, 3, 1, 2).float() / 255 if k in tl.CAMERAS
               else torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in raw.items()}
    before = {k: v.copy() if isinstance(v, (np.ndarray, list)) else v.clone() for k, v in raw.items()}
    expected = list(tl.TinyPolicy(ckpt, "cpu").prepare(raw))
    changed = {"blind": 0, "no-wrist": 0, "wrong-text": 2, "no-state": 1}[ablation]
    if ablation == "no-wrist":
        expected[0][:, 1].zero_()
    elif ablation == "wrong-text":
        expected[2] = torch.from_numpy(tl.tokenize(texts[1:] + texts[:1], meta["tokenizer"]))
    else:
        expected[changed].zero_()
    policy = tl.TinyPolicy(ckpt, "cpu", ablation=ablation)
    actual = policy.prepare(raw)
    baseline = tl.TinyPolicy(ckpt, "cpu").prepare(raw)
    assert not torch.equal(actual[changed], baseline[changed])
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
        assert a.dtype == e.dtype and a.device == e.device
    for key in raw:
        if key == "task":
            assert raw[key] == before[key]
        else:
            np.testing.assert_array_equal(raw[key], before[key])
    # The same prepared tensors must reach inference, including through the queue.
    seen = []
    def sample(*obs, **kwargs):
        seen.append(obs)
        return torch.zeros(10, 4, 7)
    monkeypatch.setattr(policy.model, "sample", sample)
    policy.predict_action_chunk(raw, noise=torch.zeros(10, 4, 7), num_steps=2)
    policy.predict_action_chunk = lambda batch: policy.model.sample(*batch["tiny_inputs"])
    policy.select_action(raw)
    for obs in seen:
        for a, e in zip(obs, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


def test_adapter_input_resize_orientation_action_shape_and_gripper(source):
    meta = build(source)
    policy = tl.TinyPolicy(fake_checkpoint(meta), "cpu")
    a = tl.open_arrays(source[1], meta)
    raw = {key: np.asarray(a["images"][:1, i]).transpose(0, 2, 3, 1) for i, key in enumerate(tl.CAMERAS)}
    raw.update({"observation.state": np.asarray(a["state"][:1]), "task": [source[2][0]]})
    obs = policy.prepare(raw)
    torch.testing.assert_close(obs[0], torch.from_numpy(a["images"][:1].copy()))
    canonical = {key: torch.as_tensor(raw[key].copy()).permute(0, 3, 1, 2).float() / 255 for key in tl.CAMERAS}
    canonical.update({k: raw[k] for k in ("observation.state", "task")})
    torch.testing.assert_close(policy.prepare(canonical)[0], obs[0])
    out = policy.postprocess(policy.predict_action_chunk(raw, noise=torch.zeros(1, 4, 7), num_steps=2))
    assert out.shape == (1, 4, 7) and torch.isfinite(out).all() and out.abs().max() <= 1
    normalized = tl.normalize(torch.tensor([[0., 0., 0., 0., 0., 0., -1.], [0., 0., 0., 0., 0., 0., 1.]]), meta["norm"]["action"])
    torch.testing.assert_close(policy.postprocess(normalized)[:, -1], torch.tensor([-1., 1.]))
    # Exercise adapter queue independently of the evaluator's inference wrapper.
    policy.config.n_action_steps = 2
    calls = []
    policy.predict_action_chunk = lambda batch: calls.append(1) or normalized[:1, None].expand(1, 4, 7)
    for _ in range(5):
        action = policy.select_action(raw)
        assert action.shape == (1, 7) and torch.isfinite(action).all()
    assert len(calls) == 3
    policy.reset()
    assert not policy.queue


def protocol():
    return ev.protocol(SimpleNamespace(suite="libero_spatial", replan=5, seed=7, max_steps=None, num_steps_wait=10,
                                       observation_height=256, observation_width=256))


def result(arm="nf4+trim:10"):
    return dict(arm=arm, steps=10, task_id=0, task_text="move mug", init_index=0, success=False,
                policy_calls=1, mean_policy_latency_s=.1, n_action_steps=5, checkpoint_n_action_steps=16,
                chunk_size=16, max_episode_steps=220, inits_requested=10, protocol=protocol())


@pytest.mark.parametrize("version", [2, 3])
def test_protocol_model_split_old_rows_unchanged_on_disk(tmp_path, version):
    path = tmp_path / "episodes.jsonl"
    old = result()
    old["protocol"].update(version=version, policy="pi05", revision="abc")
    if version == 2:
        old["protocol"].update(placement="nf4", trim=True)
        old.pop("arm")
    ev.append_row(path, old)
    before = path.read_bytes()
    tiny = result("tiny:10")
    tiny["model_identity"] = {"checkpoint_path": "tiny.pt", "checkpoint_sha256": "hash"}
    ev.append_row(path, tiny)
    for ablation in ("blind", "no-wrist", "wrong-text", "no-state"):
        ev.append_row(path, {**tiny, "arm": f"tiny-{ablation}:10",
                             "model_identity": {**tiny["model_identity"], "ablation": ablation}})
    rows = ev.read_rows(path)
    assert path.read_bytes().startswith(before)
    assert rows["nf4+trim:10", 0, 0]["protocol"] == rows["tiny:10", 0, 0]["protocol"] == protocol()
    assert rows["nf4+trim:10", 0, 0]["model_identity"] == dict(policy="pi05", revision="abc")
    summary = ev.summarize(rows.values())
    assert summary["paired"]["nf4+trim:10 vs tiny:10"]["complete_pairs"] == 1
    assert len(summary["arms"]) == 6 and len(summary["paired"]) == 15
    assert all(p["complete_pairs"] == 1 and p["incomplete_pairs"] == 0 for p in summary["paired"].values())
    conflict = {**tiny, "init_index": 1, "model_identity": {"checkpoint_sha256": "other"}}
    with pytest.raises(ValueError, match="identities"):
        ev.summarize([*rows.values(), conflict])
    changed = {**tiny, "protocol": {**protocol(), "env_seed": 8}}
    with pytest.raises(ValueError, match="protocols"):
        ev.summarize([*rows.values(), changed])


def test_live_log_compatibility_on_copy_only(tmp_path):
    live = Path(__file__).resolve().parents[1] / "runs/libero_eval_libero_spatial.jsonl"
    if not live.exists():
        pytest.skip("No coordinator log available")
    # Snapshot bytes only; repair/resume must never touch the actively appended source.
    data = live.read_bytes()
    path = tmp_path / "snapshot.jsonl"
    path.write_bytes(data[:data.rfind(b"\n") + 1])
    rows = ev.read_rows(path)
    assert rows and ev.summarize(rows.values())["protocol"] == protocol()
    assert all("policy" not in r["protocol"] and "model_identity" in r for r in rows.values())


@pytest.mark.parametrize("extras", [[], ["--ema", ".9", "--save-every", "1", "--color-jitter"]])
@pytest.mark.parametrize("logger", [None, "wandb", "none"])
def test_cli_cpu_training_and_resume(source, tmp_path, monkeypatch, extras, logger, fake_wandb):
    build(source)
    monkeypatch.chdir(tmp_path)
    baseline = tmp_path / "runs/libero_tiny/latest.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"existing baseline checkpoint")
    runs = tmp_path / "training"
    original = tl.Config
    monkeypatch.setattr(tl, "Config", lambda **kw: original(**{**asdict(config()), **kw}))
    command = ["train_libero.py", "--cache", str(source[1]), "--runs", str(runs), "--device", "cpu",
               "--steps", "2", "--distill-steps", "0", "--batch-size", "2", "--horizon", "4"]
    command += extras
    if logger:
        command += ["--logger", logger]
    monkeypatch.setattr("sys.argv", command)
    tl.main()
    saved = torch.load(runs / "latest.pt", weights_only=True)
    assert saved["step"] == 2 and saved["phase"] == "train"
    if logger == "wandb":
        assert fake_wandb.runs[0].kwargs["config"] == {**saved["settings"], "model_config": saved["config"]}
    assert (runs / "tb").exists() == (logger is None)
    assert (runs / "wandb_run_id").exists() == (logger == "wandb")
    assert baseline.read_bytes() == b"existing baseline checkpoint"
    if extras:
        for step in (1, 2):
            snapshot = torch.load(runs / f"step_{step}.pt", weights_only=True)
            assert snapshot["step"] == step and snapshot["phase"] == "train" and snapshot["ema"]
        for key in saved["model"]:
            torch.testing.assert_close(snapshot["model"][key], saved["model"][key], atol=0, rtol=0)
            torch.testing.assert_close(snapshot["ema"][key], saved["ema"][key], atol=0, rtol=0)
        assert not list(runs.glob("*.tmp"))
    else:
        assert "ema" not in saved and not list(runs.glob("step_*.pt"))
        # Legacy settings (before these flags existed) still resume with defaults.
        for key in ("ema", "color_jitter"):
            saved["settings"].pop(key)
        torch.save(saved, runs / "latest.pt")
    before = (runs / "latest.pt").read_bytes()
    monkeypatch.setattr("sys.argv", [*command, "--resume"])
    tl.main()
    assert (runs / "latest.pt").read_bytes() == before
    if logger == "wandb":
        first, resumed = fake_wandb.runs
        assert first.id == resumed.id == (runs / "wandb_run_id").read_text()
        assert resumed.kwargs["id"] == first.id and resumed.kwargs["resume"] == "allow"
        assert {next(iter(data)) for data, step in first.logs if step == 2} == {"train/loss", "val/loss"}
        assert all(run.finished for run in fake_wandb.runs)
    else:
        assert not fake_wandb.runs


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("interrupted_phase", ["distill", "train"])
def test_offline_distillation_resumes_into_training_exactly(source, tmp_path, monkeypatch, ema, interrupted_phase):
    import transformers
    build(source)
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(hidden_size=12)
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            assert name == "google/siglip-base-patch16-224" and kwargs["local_files_only"] is True
            return cls()
        def forward(self, pixel_values, interpolate_pos_encoding):
            assert interpolate_pos_encoding
            x = torch.nn.functional.adaptive_avg_pool2d(pixel_values, 2).flatten(2).transpose(1, 2)
            return SimpleNamespace(last_hidden_state=x.repeat(1, 1, 4))
    monkeypatch.setattr(transformers, "SiglipVisionModel", Teacher)
    original_config = tl.Config
    monkeypatch.setattr(tl, "Config", lambda **kw: original_config(**{**asdict(config()), **kw}))
    base = ["train_libero.py", "--cache", str(source[1]), "--device", "cpu", "--steps", "3", "--save-every", "2",
            "--distill-steps", "2", "--batch-size", "2", "--distill-batch-size", "2", "--horizon", "4"]
    if ema:
        base += ["--ema", ".9", "--color-jitter"]
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    monkeypatch.setattr("sys.argv", [*base, "--runs", str(uninterrupted)])
    tl.main()
    original_save = tl.save_checkpoint
    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[3] == interrupted_phase and Path(args[0]).name == "latest.pt":
            raise KeyboardInterrupt
    monkeypatch.setattr(tl, "save_checkpoint", interrupt)
    monkeypatch.setattr("sys.argv", [*base, "--runs", str(resumed)])
    with pytest.raises(KeyboardInterrupt):
        tl.main()
    checkpoint = torch.load(resumed / "latest.pt", weights_only=True)
    assert checkpoint["phase"] == interrupted_phase and checkpoint["step"] == 2
    assert bool(checkpoint["head"]) == (interrupted_phase == "distill")
    monkeypatch.setattr(tl, "save_checkpoint", original_save)
    monkeypatch.setattr("sys.argv", [*base, "--runs", str(resumed), "--resume"])
    tl.main()
    a = torch.load(uninterrupted / "latest.pt", weights_only=True)
    b = torch.load(resumed / "latest.pt", weights_only=True)
    assert a["phase"] == b["phase"] == "train" and a["step"] == b["step"] == 3
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
        if ema:
            torch.testing.assert_close(a["ema"][key], b["ema"][key], atol=0, rtol=0)
