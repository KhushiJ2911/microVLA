import pytest
import torch

from examples import train_toy as train
from train import Config, MicroVLA, attention_mask, parameter_counts


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(threads)


def tiny_config() -> Config:
    return Config(vocab_size=12, horizon=3, width=32, depth=2, heads=2,
                  head_dim=8, mlp=64, expert_width=16, expert_mlp=32,
                  vit_width=16, vit_depth=1, vit_heads=2)


def batch(c: Config, size: int = 2) -> tuple:
    images = torch.randint(256, (size, 3, 64, 64), dtype=torch.uint8)
    state = torch.rand(size, c.state_dim) * 2 - 1
    instr = torch.randint(1, c.vocab_size, (size, 8))
    instr[0, -2:] = 0
    actions = torch.randn(size, c.horizon, c.action_dim)
    return images, state, instr, actions


def activate_gates(model: MicroVLA) -> None:
    # Cache equivalence must exercise attention, not only zero-initialized residuals.
    for layer in model.layers:
        for norm in (layer.action.attn_norm, layer.action.ffn_norm):
            torch.nn.init.normal_(norm.modulation.weight, std=0.1)
            torch.nn.init.constant_(norm.modulation.bias, 0.2)


def test_output_shapes_and_discrete_state():
    c = tiny_config()
    model = MicroVLA(c)
    images, state, instr, actions = batch(c)
    t = torch.tensor([0.2, 0.9])
    assert model(images, state, instr, actions, t).shape == actions.shape
    assert model.sample(images, state, instr, num_steps=2).shape == actions.shape
    p, valid = model.embed_prefix(images, state, instr)
    assert p.shape == (2, 64 + 8 + c.state_dim, c.width)
    assert valid.shape == p.shape[:2]
    assert model.loss(images, state, instr, actions).ndim == 0
    low = torch.full_like(state, -1)
    high = torch.full_like(state, 1)
    assert torch.isfinite(model.embed_prefix(images, low, instr)[0]).all()
    assert torch.isfinite(model.embed_prefix(images, high, instr)[0]).all()
    p1, _ = model.embed_prefix(images, torch.zeros_like(state), instr)
    p2, _ = model.embed_prefix(images, torch.full_like(state, 0.001), instr)
    torch.testing.assert_close(p1, p2, rtol=0, atol=0)


def test_block_causal_mask_and_padding():
    valid = torch.tensor([[True, True, True], [True, False, True]])
    mask = attention_mask(valid, 2)[:, 0]
    assert mask.shape == (2, 5, 5)
    assert mask[0, :3, :3].all()
    assert not mask[:, :3, 3:].any()
    assert mask[0, 3:, :].all()
    assert not mask[1, :, 1].any()
    assert mask[1, 3:, [0, 2, 3, 4]].all()


def test_padded_prefix_stays_finite():
    c = tiny_config()
    images, state, instr, actions = batch(c)  # row 0 has trailing <pad> tokens
    out = MicroVLA(c)(images, state, instr, actions, torch.tensor([0.3, 0.7]))
    assert torch.isfinite(out).all()


def test_prefix_cannot_see_actions():
    c = tiny_config()
    model = MicroVLA(c)
    activate_gates(model)
    images, state, instr, actions = batch(c)
    p, valid = model.embed_prefix(images, state, instr)
    pos = torch.nn.functional.pad(valid, (0, c.horizon), value=True).long().cumsum(1) - 1
    mask = attention_mask(valid, c.horizon)
    t = model.time_condition(torch.tensor([0.4, 0.8]))
    def prefix_after(noisy):
        prefix, action = p, model.action_in(noisy)
        for layer in model.layers:
            prefix, action, _ = layer(prefix, action, mask, pos[:, :p.shape[1]], pos[:, p.shape[1]:], t)
        return prefix
    torch.testing.assert_close(prefix_after(actions), prefix_after(actions * 100), rtol=0, atol=0)


def test_cached_sampling_matches_uncached_and_encodes_prefix_once():
    c = tiny_config()
    model = MicroVLA(c).eval()
    activate_gates(model)
    images, state, instr, noise = batch(c)
    calls = []
    hook = model.vision.register_forward_hook(lambda *args: calls.append(1))
    cached = model.sample(images, state, instr, num_steps=4, noise=noise)
    hook.remove()
    assert len(calls) == 1
    uncached = model.sample(images, state, instr, num_steps=4, noise=noise, cached=False)
    torch.testing.assert_close(cached, uncached, rtol=2e-5, atol=2e-5)


def test_flow_loss_matches_convention():
    c = tiny_config()
    model = MicroVLA(c)
    images, state, instr, actions = batch(c)
    noise, t = torch.randn_like(actions), torch.tensor([0.25, 0.75])
    noisy = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    expected = (model(images, state, instr, noisy, t) - (noise - actions)).square().mean()
    torch.testing.assert_close(model.loss(images, state, instr, actions, noise, t), expected)


def test_overfits_tiny_random_batch():
    c = tiny_config()
    model = MicroVLA(c)
    images, state, instr, actions = batch(c)
    noise, t = torch.randn_like(actions), torch.tensor([0.2, 0.8])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0)
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(images, state, instr, actions, noise, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    print(f"overfit MSE: {losses[0]:.6f} -> {losses[-1]:.6f}")
    assert losses[-1] < 0.02
    assert losses[-1] < losses[0] * 0.02


def test_default_parameter_count():
    counts = parameter_counts(MicroVLA(Config()))
    print(f"default parameter counts: {counts}")
    assert counts["total"] == sum(counts[name] for name in counts if name != "total")
    assert 10_000_000 <= counts["total"] <= 20_000_000


@pytest.mark.parametrize("logger", [None, "wandb", "none"])
def test_cli_cpu_training_and_resume(tmp_path, monkeypatch, logger, fake_wandb):
    from examples import sim
    monkeypatch.chdir(tmp_path)
    original = train.Config
    monkeypatch.setattr(train, "Config", lambda **kwargs: original(**{**vars(tiny_config()), **kwargs}))
    monkeypatch.setattr(sim, "evaluate", lambda *args, **kwargs: {"success": .5})
    runs, logdir = tmp_path / "training", tmp_path / "tb"
    command = ["examples.train_toy", "--runs", str(runs), "--logdir", str(logdir), "--device", "cpu",
               "--steps", "2", "--episodes", "2", "--distill-steps", "0", "--batch-size", "2",
               "--flow-steps", "1"]
    if logger:
        command += ["--logger", logger]
    monkeypatch.setattr("sys.argv", command)
    train.main()
    saved = torch.load(runs / "latest.pt", weights_only=True)
    assert saved["step"] == 2
    assert saved["settings"]["logger"] == (logger or "tensorboard")
    assert logdir.exists() == (logger is None)
    assert (runs / "wandb_run_id").exists() == (logger == "wandb")
    before = (runs / "latest.pt").read_bytes()
    monkeypatch.setattr("sys.argv", [*command, "--resume"])
    train.main()
    assert (runs / "latest.pt").read_bytes() == before
    if logger == "wandb":
        first, resumed = fake_wandb.runs
        assert first.id == resumed.id == (runs / "wandb_run_id").read_text()
        assert resumed.kwargs["id"] == first.id and resumed.kwargs["resume"] == "allow"
        assert first.kwargs["config"] == {**saved["settings"], "model_config": saved["config"]}
        assert [step for data, step in first.logs if "loss/train" in data] == [2]
        assert all(run.finished for run in fake_wandb.runs)
    else:
        assert not fake_wandb.runs
