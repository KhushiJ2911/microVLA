import torch
import pytest
from PIL import Image

import sim


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def oracle(seed, split="train", swap=False, n=32, horizon=5):
    scene = sim.reset(n, seed + sim.EVAL_SEED_OFFSET, "cpu", split)
    if swap:
        scene[:, 17] = torch.where(scene[:, 7] == scene[:, 17], scene[:, 8], scene[:, 7])

    def policy(images, state, instr):
        nonlocal scene
        assert not torch.is_grad_enabled()
        assert torch.equal(images, sim.render(scene))
        assert torch.equal(state, sim.robot_state(scene))
        assert torch.equal(instr, sim.encode_instruction(scene[:, 17:19].long()))
        actions = []
        for _ in range(horizon):
            action = sim.expert(scene)
            actions.append(action)
            scene = sim.step(scene, action)
        return torch.stack(actions, 1)

    return policy


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_expert_solves_both_splits(split):
    scene = sim.reset(512, 42, "cpu", split)
    for _ in range(sim.MAX_STEPS):
        action = sim.expert(scene)
        assert action.abs().amax() <= 1
        scene = sim.step(scene, action)
    assert sim.success(scene).float().mean() >= 0.98
    result = sim.evaluate(oracle(91, split, horizon=3), 32, 91, "cpu", split=split, replan=5)
    assert result["success"] >= 0.98
    assert result["per_episode"].shape == (32,)
    assert result["per_episode"].dtype == torch.bool


def test_reset_render_and_instructions_are_deterministic():
    before = torch.random.get_rng_state()
    scene = sim.reset(64, 123, "cpu")
    assert torch.equal(scene, sim.reset(64, 123, "cpu"))
    assert not torch.equal(scene, sim.reset(64, 124, "cpu"))
    image = sim.render(scene)
    assert image.shape == (64, 3, sim.IMG_SIZE, sim.IMG_SIZE)
    assert image.dtype == torch.uint8
    assert torch.equal(image, sim.render(scene))
    assert torch.equal(before, torch.random.get_rng_state())
    colors = scene[:, [7, 8, 13, 14]].sort(1).values
    assert torch.equal(colors, torch.arange(4).expand(64, -1))
    centers = torch.cat((scene[:, :2, None].transpose(1, 2), scene[:, 3:7].reshape(-1, 2, 2),
                         scene[:, 9:13].reshape(-1, 2, 2)), 1)
    distances = (centers[:, :, None] - centers[:, None]).abs().amax(-1)
    distances[:, torch.arange(5), torch.arange(5)] = 2
    assert distances.amin() > 2 * sim.BIN_HALF
    tokens = sim.encode_instruction(scene[:, 17:19].long())
    assert tokens.shape == (64, sim.INSTR_LEN) and tokens.dtype == torch.long
    assert sim.VOCAB[0] == "<pad>"
    for task, encoded in zip(scene[:4, 17:19], tokens[:4]):
        assert " ".join(sim.VOCAB[i] for i in encoded.tolist() if i) == sim.instruction_text(task)
    assert torch.equal(sim.robot_state(scene), scene[:, :3])


def test_grasp_carry_drop_and_completed_states_are_pure():
    scene = sim.reset(1, 3, "cpu")
    scene[:, :3] = torch.tensor([0.8, 0.8, -1.])
    scene[:, 3:9] = torch.tensor([0.82, 0.8, 0.85, 0.8, 0., 1.])
    scene[:, 9:15] = torch.tensor([-0.7, -0.7, -0.7, 0.7, 2., 3.])
    scene[:, 17:19] = torch.tensor([0., 2.])
    original = scene.clone()
    attached = sim.step(scene, torch.tensor([[0., 0., 1.]]))
    assert torch.equal(scene, original)
    assert attached[0, 15] == 0
    carried = sim.step(attached, torch.tensor([[1., -1., 1.]]))
    torch.testing.assert_close(carried[:, 3:5], carried[:, :2])
    assert not sim.success(carried).item()
    carried[:, 9:11] = carried[:, :2]
    assert not sim.success(carried).item()
    dropped = sim.step(carried, torch.tensor([[0., 0., -1.]]))
    assert dropped[0, 15] == -1 and sim.success(dropped).item()
    assert torch.equal(dropped, sim.step(dropped, torch.ones(1, 3)))
    scene[:, :2], scene[:, 16] = 0.99, sim.MAX_STEPS - 1
    clamped = sim.step(scene, torch.tensor([[10., 10., -1.]]))
    assert torch.equal(clamped[:, :2], torch.ones(1, 2))
    assert torch.equal(clamped, sim.step(clamped, -torch.ones(1, 3)))


def test_collect_split_noise_and_clean_chunk_labels():
    data = sim.collect(12, horizon=24, noise=0.25, seed=19, device="cpu")
    repeated = sim.collect(12, horizon=24, noise=0.25, seed=19, device="cpu")
    for key in data:
        assert torch.equal(data[key], repeated[key])
    scene, task, labels = data["scene"], data["task"], data["actions"]
    assert scene.shape[1] == sim.SCENE_DIM and labels.shape == (len(scene), 24, 3)
    assert task.dtype == torch.long and labels.dtype == scene.dtype == torch.float32
    assert not sim.success(scene).any()
    for combo in sim.HELDOUT_COMBOS:
        assert not (task == torch.tensor(combo)).all(1).any()
    clean = sim.collect(12, horizon=1, noise=0, seed=19, device="cpu")
    assert torch.equal(scene[:12], clean["scene"][:12])
    assert not torch.equal(scene[12:24], clean["scene"][12:24])
    rollout, final_action = scene.clone(), None
    saw_padding = False
    for i in range(labels.shape[1]):
        done = sim.success(rollout) | (rollout[:, 16] >= sim.MAX_STEPS)
        expected = sim.expert(rollout)
        if final_action is not None:
            expected = torch.where(done[:, None], final_action, expected)
        torch.testing.assert_close(labels[:, i], expected, rtol=0, atol=0)
        rollout = sim.step(rollout, expected)
        final_action = expected
        saw_padding |= bool(done.any())
    assert saw_padding


def test_swap_preserves_scenes_and_retargets_other_block():
    observations = []

    def policy(images, state, instr):
        if not observations or len(observations) == 1 and instr[0, 2] != observations[0][2][0, 2]:
            observations.append((images.clone(), state.clone(), instr.clone()))
        return torch.zeros(len(images), 5, 3)

    sim.evaluate(policy, 16, 11, "cpu")
    sim.evaluate(policy, 16, 11, "cpu", swap=True)
    first, swapped = observations
    assert torch.equal(first[0], swapped[0]) and torch.equal(first[1], swapped[1])
    assert torch.equal(first[2][:, 6], swapped[2][:, 6])
    scene = sim.reset(16, 11 + sim.EVAL_SEED_OFFSET, "cpu")
    other = torch.where(scene[:, 7] == scene[:, 17], scene[:, 8], scene[:, 7]).long()
    assert torch.equal(swapped[2][:, 2] - 3, other)
    assert (first[2][:, 2] != swapped[2][:, 2]).all()
    assert sim.evaluate(oracle(11, swap=True, n=16), 16, 11, "cpu", swap=True)["success"] >= 0.98


def test_random_policy_has_low_success():
    rng = torch.Generator().manual_seed(8)

    def policy(images, state, instr):
        return torch.rand(len(images), 5, 3, generator=rng) * 2 - 1

    assert sim.evaluate(policy, 128, 4, "cpu")["success"] < 0.05


def test_record_and_gif(tmp_path):
    result = sim.evaluate(oracle(7, n=4), 4, 7, "cpu", record=4)
    frames = result["frames"]
    assert frames.shape == (sim.MAX_STEPS + 1, 4, 3, sim.IMG_SIZE, sim.IMG_SIZE)
    assert frames.dtype == torch.uint8 and result["success"] == 1
    path = tmp_path / "preview.gif"
    sim.save_gif(frames[::5], str(path), labels=["red", "green", "blue", "yellow"])
    with Image.open(path) as image:
        assert image.format == "GIF" and image.n_frames > 1
        assert image.width == sim.IMG_SIZE * 3 * 4
