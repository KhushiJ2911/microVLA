from collections import deque
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eval_libero as ev


def protocol(**overrides):
    args = SimpleNamespace(suite="libero_spatial", replan=5, seed=7, max_steps=None,
                           num_steps_wait=10, observation_height=256, observation_width=256)
    return {**ev.protocol(args), **overrides}


def row(task=0, init=0, steps=10, success=False, config="nf4+trim"):
    arm = f"{config}:{steps}"
    return dict(arm=arm, placement=config.removesuffix("+trim"), trim=config.endswith("+trim"),
                policy_calls=2, mean_policy_latency_s=.5, task_id=task, task_text=f"task {task}", init_index=init, steps=steps,
                success=success, n_action_steps=5, checkpoint_n_action_steps=50, chunk_size=50,
                max_episode_steps=220, protocol=protocol(), inits_requested=10)


def test_resume_repairs_only_torn_tail_and_fsyncs(tmp_path, monkeypatch):
    path = tmp_path / "runs" / "episodes.jsonl"
    synced = []
    monkeypatch.setattr(ev.os, "fsync", lambda fd: synced.append(fd))
    ev.append_row(path, row())
    ev.append_row(path, row(steps=5, success=True))
    good = path.read_bytes()
    path.write_bytes(good + b'{"task_id": 1')
    rows = ev.read_rows(path)
    assert set(rows) == {("nf4+trim:10", 0, 0), ("nf4+trim:5", 0, 0)}
    assert path.read_bytes() == good and len(synced) == 3
    ev.append_row(path, row(task=1))
    assert len(ev.read_rows(path)) == 3
    path.write_bytes(path.read_bytes() + b"broken but newline-terminated\n")
    with pytest.raises(json.JSONDecodeError):
        ev.read_rows(path)


def test_missing_log_duplicate_and_conflicting_rows(tmp_path):
    path = tmp_path / "episodes.jsonl"
    assert ev.read_rows(path) == {}
    ev.append_row(path, row())
    ev.append_row(path, row())
    assert len(ev.read_rows(path)) == 1
    ev.append_row(path, row(success=True))
    with pytest.raises(ValueError, match="Conflicting"):
        ev.read_rows(path)


def test_seed_is_stable_independent_of_setting_and_global_rng():
    import hashlib
    expected = int.from_bytes(hashlib.sha256(b"libero_spatial:2:3:4").digest()[:4], "little")
    assert ev.seed("libero_spatial", 2, 3, 4) == expected
    torch.manual_seed(12)
    before = torch.random.get_rng_state()
    noise = [torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(ev.seed("libero_spatial", 2, 3, 4)))
             for arm in ("nf4+trim:10", "nf4+trim:5", "offload-3.5:10")]
    assert all(torch.equal(noise[0], n) for n in noise) and torch.equal(before, torch.random.get_rng_state())
    seeds = {ev.seed(s, t, i, c) for s in ("libero_spatial", "libero_goal")
             for t in range(2) for i in range(2) for c in (-1, 0, 1)}
    assert len(seeds) == 24


@pytest.mark.parametrize("wins,n,expected", [(0, 0, [None, None]), (0, 10, [0, .2775327998628892]),
                                           (10, 10, [.7224672001371107, 1]),
                                           (5, 10, [.236593090512564, .763406909487436])])
def test_wilson(wins, n, expected):
    if n:
        assert ev.wilson(wins, n) == pytest.approx(expected)
    else:
        assert ev.wilson(wins, n) == expected


@pytest.mark.parametrize("a,b,p", [(0, 0, 1), (1, 0, 1), (0, 5, .0625), (1, 9, .021484375), (5, 5, 1)])
def test_exact_mcnemar(a, b, p):
    assert ev.mcnemar(a, b) == p
    assert ev.mcnemar(b, a) == p


def test_summary_pairs_by_task_and_init_and_excludes_incomplete_pairs():
    rows = [row(task, init, steps, success) for task, init, a, b in
            [(0, 0, True, True), (0, 1, True, False), (1, 0, False, True), (1, 1, False, False)]
            for steps, success in [(10, a), (5, b)]]
    rows.append(row(task=2, steps=10, success=True))
    summary = ev.summarize(reversed(rows))
    pair = summary["paired"]["nf4+trim:10 vs nf4+trim:5"]
    assert pair["table"] == dict(both_succeed=1, only_a=1, only_b=1, both_fail=1)
    assert pair["complete_pairs"] == 4 and pair["incomplete_pairs"] == 1
    assert pair["mcnemar_exact_p"] == 1
    assert summary["arms"]["nf4+trim:10"]["rate"] == .6 and summary["arms"]["nf4+trim:5"]["rate"] == .5
    assert summary["arms"]["nf4+trim:10"]["wilson95"] == pytest.approx(ev.wilson(3, 5))
    assert summary["per_task"]["0"]["nf4+trim:10"] == [2, 2]
    assert summary["per_task"]["2"]["nf4+trim:5"] == [0, 0]
    assert summary["n_action_steps"] == [5] and summary["max_episode_steps"] == [220]
    assert summary["checkpoint_n_action_steps"] == summary["chunk_size"] == [50]
    assert summary["protocol"] == protocol() and summary["inits_requested"] == [10]
    assert ev.summarize([])["paired"] == {}


def test_summary_cli_without_importing_infer_or_simulator(tmp_path, monkeypatch, capsys):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "infer" and not name.startswith(("lerobot", "libero", "mujoco"))
        return original(name, *args, **kwargs)
    monkeypatch.chdir(tmp_path)
    ev.append_row(tmp_path / "runs/libero_eval_libero_spatial.jsonl", row())
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(ev.argparse._sys, "argv", ["eval_libero.py", "--summary"])
    ev.main()
    summary = json.loads((tmp_path / "runs/libero_eval_libero_spatial_summary.json").read_text())
    output = capsys.readouterr().out
    assert json.loads(output.split("\nTask |", 1)[0]) == summary
    assert "0 | 0/1 | task 0" in output
    assert "nf4+trim:10: 0.0% | openpi 98.8%" in output
    assert "different checkpoint/framework; not paired" in output
    assert summary["paired"] == {}


def test_summary_rejects_mixed_or_missing_protocols():
    a, b = row(), row(steps=5)
    b["protocol"]["replan"] = 50
    with pytest.raises(ValueError, match="protocols"):
        ev.summarize([a, b])
    del a["protocol"]
    with pytest.raises(ValueError, match="missing"):
        ev.summarize([a])


@pytest.fixture
def cpu_rollout(monkeypatch):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    class Policy(torch.nn.Module):
        select_action = PI05Policy.select_action  # Exercise the installed policy's actual queue.
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_action_steps=50, chunk_size=50, max_action_dim=32,
                                          use_visual_memory=False, use_proprioceptive_memory=False)
            self.calls = []
        def _rtc_enabled(self):
            return False
        def reset(self):
            self._action_queue = deque(maxlen=self.config.n_action_steps)
        def predict_action_chunk(self, batch, *, noise, num_steps):
            assert not torch.is_grad_enabled()
            assert batch["task"] == ["move the mug"]
            assert batch["observation.images.image"].shape == (1, 3, 4, 4)
            assert batch["observation.images.image"].max() == 1
            self.calls.append((num_steps, noise.clone()))
            assert noise.shape == (1, 50, 32)
            return torch.arange(50.).reshape(1, 50, 1).expand(1, 50, 7) + num_steps
    class Env:
        num_envs, use_async_envs = 1, False
        metadata = {"canonical_observations": True}
        def __init__(self):
            self._envs = [SimpleNamespace(init_state_id=99, _init_states=np.zeros((3, 4)))]
            self.resets, self.actions, self.waits = [], [], []
            self.limit, self.success_at = 12, None
        def call(self, name):
            return {"_max_episode_steps": (self.limit,), "task_description": ("move the mug",), "task": ("mug",)}[name]
        get_attr = call
        def obs(self):
            return {"observation.images.image": np.full((1, 4, 4, 3), 255, dtype=np.uint8),
                    "observation.state": np.zeros((1, 8), dtype=np.float32)}
        def reset(self, seed):
            self.resets.append((self._envs[0].init_state_id, seed))
            self.waits.append(self._envs[0].num_steps_wait)
            self._envs[0].init_state_id += 1
            self.actions = []
            return self.obs(), {}
        def step(self, action):
            self.actions.append(action.copy())
            success = len(self.actions) == self.success_at
            return self.obs(), np.zeros(1), np.array([success]), np.array([False]), {"is_success": np.array([success])}
        def render(self):
            return (self.obs()["observation.images.image"][0],)
    class Processor:
        def __init__(self, transform=lambda x: x):
            self.transform, self.resets, self.calls = transform, 0, 0
        def __call__(self, x):
            self.calls += 1
            return self.transform(x)
        def reset(self):
            self.resets += 1
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self if a == ("cuda",) else original_to(self, *a, **kw))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    env, policy = Env(), Policy()
    pre, post = Processor(), Processor(lambda x: x * 2)
    env_pre, env_post = Processor(), Processor(lambda x: {"action": x["action"] + 3})
    return env, policy, pre, post, env_pre, env_post


def test_actual_lerobot_rollout_queue_processors_and_paired_noise(cpu_rollout):
    env, policy, pre, post, env_pre, env_post = cpu_rollout
    results = []
    for steps in (10, 5, 10):  # Third episode represents a freshly loaded reference pass.
        original = policy.predict_action_chunk
        result, frames = ev.episode(*cpu_rollout, "libero_spatial", 2, 1, steps, True, protocol())
        results.append(result)
        assert policy.predict_action_chunk == original
        assert policy.config.n_action_steps == 50
        assert result["episode_length"] == result["max_episode_steps"] == 12
        assert result["policy_calls"] == 3 and result["n_action_steps"] == 5
        assert result["checkpoint_n_action_steps"] == result["chunk_size"] == 50
        assert result["mean_policy_latency_s"] >= 0 and result["wall_time_s"] > 0
        assert not result["success"] and len(frames) == 13
        expected = [(steps + i) * 2 + 3 for i in range(5)]
        assert [a[0, 0] for a in env.actions] == expected * 2 + expected[:2]
    assert env.resets == [(1, [7])] * 3
    assert env.waits == [10] * 3  # Settling belongs to reset; no extra rollout steps.
    for i in range(3):
        assert policy.calls[i][0] == 10 and policy.calls[i + 3][0] == 5
        assert torch.equal(policy.calls[i][1], policy.calls[i + 3][1])
        assert torch.equal(policy.calls[i][1], policy.calls[i + 6][1])
    assert not torch.equal(policy.calls[0][1], policy.calls[1][1])
    assert all(p.calls == 36 and p.resets == 3 for p in (pre, post, env_pre, env_post))
    env.success_at = 7
    result, frames = ev.episode(*cpu_rollout, "libero_spatial", 2, 1, 5, False, protocol())
    assert result["success"] and result["episode_length"] == 7 and result["policy_calls"] == 2 and not frames


def test_rollout_restores_chunk_method_after_error(cpu_rollout):
    env, policy, *_ = cpu_rollout
    original = policy.predict_action_chunk
    env.step = lambda _: (_ for _ in ()).throw(RuntimeError("sim failed"))
    with pytest.raises(RuntimeError, match="sim failed"):
        ev.episode(*cpu_rollout, "libero_spatial", 0, 0, 10, False, protocol(replan=2))
    assert policy.predict_action_chunk == original
    assert policy.config.n_action_steps == 50
    with pytest.raises(ValueError, match="unavailable"):
        ev.episode(*cpu_rollout, "libero_spatial", 0, 3, 10, False, protocol())
    with pytest.raises(ValueError, match="chunk_size"):
        ev.episode(*cpu_rollout, "libero_spatial", 0, 0, 10, False, protocol(replan=51))


def test_rollout_applies_wait_seed_and_replan_overrides(cpu_rollout):
    env, policy, *_ = cpu_rollout
    result, _ = ev.episode(*cpu_rollout, "libero_goal", 0, 0, 5, False,
                           protocol(num_steps_wait=3, env_seed=11, replan=4))
    assert env.waits == [3] and env.resets == [(0, [11])]
    assert result["n_action_steps"] == 4 and result["policy_calls"] == 3
    assert [a[0, 0] for a in env.actions] == [13, 15, 17, 19] * 3
    assert policy.config.n_action_steps == 50


def test_main_interleaves_resumes_half_pair_and_loads_once(tmp_path, monkeypatch):
    import infer
    import lerobot.envs.factory as factory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev.argparse._sys, "argv", ["eval_libero.py", "--inits", "2", "--tasks", "0", "1", "--arms", "nf4+trim:10", "nf4+trim:5"])
    loaded, calls = [], []
    envs = {t: SimpleNamespace(close=lambda: None) for t in (0, 1)}
    monkeypatch.setattr(factory, "make_env", lambda cfg, n_envs: {cfg.task: envs})
    monkeypatch.setattr(factory, "make_env_pre_post_processors", lambda *a: (None, None))
    def load(config, trim):
        loaded.append((config, trim))
        return SimpleNamespace(config=None), None, None, {}
    monkeypatch.setattr(infer, "load_policy", load)
    path = tmp_path / "runs/libero_eval_libero_spatial.jsonl"
    def episode(*args):
        task, init, steps = args[7:10]
        if calls and len(calls) != 2:  # The interrupted second call never finished.
            assert (f"nf4+trim:{calls[-1][2]}", calls[-1][0], calls[-1][1]) in ev.read_rows(path)
        calls.append((task, init, steps))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return row(task, init, steps), []
    monkeypatch.setattr(ev, "episode", episode)
    with pytest.raises(KeyboardInterrupt):
        ev.main()
    assert set(ev.read_rows(path)) == {("nf4+trim:10", 0, 0)}
    ev.main()
    assert calls[2:] == [(0, 0, 5), (0, 1, 10), (0, 1, 5), (1, 0, 10), (1, 0, 5), (1, 1, 10), (1, 1, 5)]
    assert loaded == [("nf4", True), ("nf4", True)]
    ev.main()
    assert len(loaded) == 2 and len(ev.read_rows(path)) == 8


@pytest.mark.parametrize("suite,limit", list(ev.MAX_STEPS.items()))
@pytest.mark.parametrize("override", [False, True])
def test_main_protocol_defaults_and_overrides_and_extend_to_50(tmp_path, monkeypatch, suite, limit, override):
    import infer
    import lerobot.envs.factory as factory
    monkeypatch.chdir(tmp_path)
    configs, specs, loads = [], [], []
    def make_env(cfg, n_envs):
        assert n_envs == 1
        configs.append(cfg)
        return {cfg.task: {0: SimpleNamespace(close=lambda: None)}}
    monkeypatch.setattr(factory, "make_env", make_env)
    monkeypatch.setattr(factory, "make_env_pre_post_processors", lambda *a: (None, None))
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return SimpleNamespace(config=None), None, None, {}
    monkeypatch.setattr(infer, "load_policy", load)
    def episode(*args):
        task, init, steps = args[7:10]
        specs.append(args[-1])
        return row(task, init, steps), []
    monkeypatch.setattr(ev, "episode", episode)
    command = ["eval_libero.py", "--suite", suite, "--tasks", "0"]
    if override:
        command += ["--replan", "8", "--episode-length", "99", "--num-steps-wait", "0",
                    "--seed", "42", "--observation-height", "128", "--observation-width", "192"]
    monkeypatch.setattr(ev.argparse._sys, "argv", command)
    ev.main()  # Default ten init states.
    path = tmp_path / f"runs/libero_eval_{suite}.jsonl"
    initial = path.read_bytes()
    assert len(ev.read_rows(path)) == 20 and len(loads) == 1
    cfg, spec = configs[0], specs[0]
    assert cfg.episode_length == spec["max_episode_steps"] == (99 if override else limit)
    assert cfg.observation_height == spec["observation_height"] == (128 if override else 256)
    assert cfg.observation_width == spec["observation_width"] == (192 if override else 256)
    assert spec["replan"] == (8 if override else 5) and spec["env_seed"] == (42 if override else 7)
    assert spec["num_steps_wait"] == (0 if override else 10)
    assert spec["no_op_action"] == [0] * 6 + [-1]
    assert cfg.init_states and cfg.hard_reset and spec["official_inits_per_task"] == 50
    assert spec["policy"] == infer.POLICY and spec["revision"] == infer.REVISION
    monkeypatch.setattr(ev.argparse._sys, "argv", command + ["--inits", "50"])
    ev.main()
    saved = ev.read_rows(path)
    assert len(saved) == 100 and path.read_bytes().startswith(initial)
    assert saved["nf4+trim:5", 0, 9]["inits_requested"] == 10 and saved["nf4+trim:5", 0, 49]["inits_requested"] == 50
    summary = json.loads(path.with_name(path.stem + "_summary.json").read_text())
    assert summary["protocol"] == spec and summary["inits_requested"] == [10, 50]
    assert summary["paired"]["nf4+trim:10 vs nf4+trim:5"]["complete_pairs"] == 50 and len(loads) == 2


@pytest.mark.parametrize("flag,value", [("--replan", "0"), ("--max-steps", "0"), ("--max-steps", "-1"),
                                      ("--num-steps-wait", "-1"), ("--seed", "-1"), ("--seed", str(2**32)),
                                      ("--observation-height", "0"), ("--observation-width", "-1")])
def test_invalid_protocol_flags(tmp_path, monkeypatch, flag, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev.argparse._sys, "argv", ["eval_libero.py", flag, value])
    with pytest.raises(SystemExit) as exc:
        ev.main()
    assert exc.value.code == 2 and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("field", list(protocol()) + ["policy", "revision"])
def test_main_rejects_incompatible_resume_metadata(tmp_path, monkeypatch, field):
    import infer
    monkeypatch.chdir(tmp_path)
    saved = row()
    saved["protocol"].update(policy=infer.POLICY, revision=infer.REVISION)
    saved["protocol"][field] = "different"
    ev.append_row(tmp_path / "runs/libero_eval_libero_spatial.jsonl", saved)
    monkeypatch.setattr(ev.argparse._sys, "argv", ["eval_libero.py"])
    monkeypatch.setattr(infer, "load_policy", lambda *a, **kw: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="incompatible"):
        ev.main()


def test_video_encoding_pairs_frames_and_holds_shorter_ending(tmp_path):
    import av
    from lerobot.utils.io_utils import write_video
    paths = [tmp_path / f"{s}.mp4" for s in (10, 5)]
    write_video(paths[0], [np.full((256, 256, 3), x, dtype=np.uint8) for x in (30, 60, 90)], 20)
    write_video(paths[1], [np.full((256, 256, 3), 180, dtype=np.uint8)], 20)
    target = tmp_path / "paired.mp4"
    ev.paired_video(paths, [row(success=True), row(steps=5)], target, 20)
    with av.open(str(target)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        assert container.streams.video[0].average_rate == 20
    assert len(frames) == 3 and frames[0].shape[1] == 512 and frames[0].shape[0] > 256
    assert frames[-1][-100, 100, 0] == pytest.approx(90, abs=3)
    assert frames[-1][-100, 400, 0] == pytest.approx(180, abs=3)


@pytest.mark.parametrize("label", ["nf4+trim:10", "nf4:5", "offload-3.5:10", "offload-2.5+trim:5"])
def test_parse_arm(label):
    assert ev.parse_arm(label) == label


@pytest.mark.parametrize("label", ["bf16:10", "nf4:0", "nf4:-1", "nf4:1.5", "nf4:10:5",
                                   "offload-0:10", "offload-nan:10", "offload--3.5:10", "nf4+trim"])
def test_invalid_arm(label):
    with pytest.raises(ev.argparse.ArgumentTypeError):
        ev.parse_arm(label)


def test_duplicate_arms_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev.sys, "argv", ["eval_libero.py", "--arms", "nf4:10", "nf4:10"])
    with pytest.raises(SystemExit) as exc:
        ev.main()
    assert exc.value.code == 2 and not (tmp_path / "runs").exists()


def test_resume_legacy_rows_and_new_reference_arm(tmp_path):
    path = tmp_path / "episodes.jsonl"
    old = row()
    for key in ("arm", "placement", "trim"):
        old.pop(key)
    old["protocol"].update(version=2, placement="nf4", trim=True)
    ev.append_row(path, old)
    original = path.read_bytes()
    saved = ev.read_rows(path)
    assert saved["nf4+trim:10", 0, 0] == row()
    ev.append_row(path, row(config="offload-3.5"))
    assert set(ev.read_rows(path)) == {("nf4+trim:10", 0, 0), ("offload-3.5:10", 0, 0)}
    assert path.read_bytes().startswith(original)
    summary = ev.summarize(ev.read_rows(path).values())
    assert summary["paired"]["nf4+trim:10 vs offload-3.5:10"]["complete_pairs"] == 1


def test_summary_all_arm_pairs_and_call_weighted_latency():
    rows = [row(init=init, steps=steps, success=success, config=config)
            for init, outcomes in enumerate([(True, True, True), (True, False, False),
                                              (False, True, False), (False, False, False)])
            for (config, steps), success in zip([("nf4+trim", 10), ("nf4+trim", 5), ("offload-3.5", 10)], outcomes)]
    rows += [row(task=1, success=True), row(task=2, config="offload-3.5", success=True)]
    rows[0].update(policy_calls=10, mean_policy_latency_s=2.)
    summary = ev.summarize(reversed(rows))
    assert len(summary["paired"]) == 3
    ab = summary["paired"]["nf4+trim:10 vs nf4+trim:5"]
    ac = summary["paired"]["nf4+trim:10 vs offload-3.5:10"]
    bc = summary["paired"]["nf4+trim:5 vs offload-3.5:10"]
    assert ab["table"] == dict(both_succeed=1, only_a=1, only_b=1, both_fail=1)
    assert ac["table"] == bc["table"] == dict(both_succeed=1, only_a=1, only_b=0, both_fail=2)
    assert [p["complete_pairs"] for p in (ab, ac, bc)] == [4, 4, 4]
    assert [p["incomplete_pairs"] for p in (ab, ac, bc)] == [1, 2, 1]
    assert ac["mcnemar_exact_p"] == ev.mcnemar(1, 0)
    assert summary["arms"]["nf4+trim:10"]["mean_policy_latency_s"] == pytest.approx(24 / 18)
    assert summary["per_task"]["2"]["nf4+trim:10"] == [0, 0]
    assert summary["per_task"]["2"]["offload-3.5:10"] == [1, 1]
    assert summary["external_sanity_anchor"]["success_percent"] == dict(
        libero_spatial=98.8, libero_object=98.2, libero_goal=98.0, libero_10=92.4)


def test_arm_metadata_must_match_label():
    for field, value in [("steps", 5), ("placement", "offload-3.5"), ("trim", False)]:
        with pytest.raises(ValueError, match="arm metadata"):
            ev.normalize({**row(), field: value})


def mock_offload_meta(monkeypatch, needed):
    import infer
    events = []
    monkeypatch.setattr(infer, "checkpoint", lambda: events.append("checkpoint") or "cached")
    def empty(path, trim):
        assert path == "cached" and trim is False
        events.append("meta")
        return "meta"
    monkeypatch.setattr(infer, "empty_policy", empty)
    def placement(policy, budget):
        assert policy == "meta" and budget == 3.5
        events.append("placement")
        return [], set()
    monkeypatch.setattr(infer, "placement", placement)
    monkeypatch.setattr(infer, "pinned_bytes", lambda *a: needed)
    original = ev.Path.read_text
    monkeypatch.setattr(ev.Path, "read_text", lambda path, *a, **kw:
                        "MemAvailable: 5242880 kB\n" if str(path) == "/proc/meminfo" else original(path, *a, **kw))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("preflight touched CUDA"))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: pytest.fail("preflight touched CUDA"))
    return events


def test_preflight_uses_infer_two_gib_margin(monkeypatch):
    import infer
    events = mock_offload_meta(monkeypatch, 3 * 2**30)
    ev.preflight("offload-3.5")  # Exactly 5 GiB MemAvailable minus 2 GiB reserve.
    assert events == ["checkpoint", "meta", "placement"]
    monkeypatch.setattr(infer, "pinned_bytes", lambda *a: 3 * 2**30 + 1)
    with pytest.raises(infer.OffloadRAMError, match="2 GiB reserve"):
        ev.preflight("offload-3.5")
    events.clear()
    ev.preflight("nf4+trim")
    assert events == []


def test_preflight_refuses_before_loader_or_simulator_preserving_jsonl(tmp_path, monkeypatch, capsys):
    import builtins
    import infer
    mock_offload_meta(monkeypatch, 4 * 2**30)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "runs/libero_eval_libero_spatial.jsonl"
    saved = row()
    saved["protocol"].update(policy=infer.POLICY, revision=infer.REVISION)
    ev.append_row(path, saved)
    original = path.read_bytes()
    guarded_import = builtins.__import__
    def guarded(name, *a, **kw):
        assert not name.startswith(("lerobot.envs", "libero", "mujoco"))
        return guarded_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(infer, "load_policy", lambda *a, **kw: pytest.fail("must refuse before loader"))
    monkeypatch.setattr(ev.sys, "argv", ["eval_libero.py", "--arms", "offload-3.5:10"])
    with pytest.raises(SystemExit) as exc:
        ev.main()
    assert exc.value.code == 1 and path.read_bytes() == original
    output = capsys.readouterr().out
    assert "before GPU/env initialization" in output and "2 GiB reserve" in output and "preserved" in output


def test_multiple_placements_dispatch_sequential_workers_and_propagate_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = ["eval_libero.py", "--tasks", "2", "--inits", "3", "--arms",
               "nf4+trim:10", "offload-3.5:10", "nf4+trim:5", "offload-2.5+trim:5"]
    monkeypatch.setattr(ev.sys, "argv", command)
    monkeypatch.setattr(ev, "preflight", lambda *a: pytest.fail("parent must not load a placement"))
    calls = []
    def run(args):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ev.subprocess, "run", run)
    ev.main()
    assert len(calls) == 3
    assert calls[0] == [ev.sys.executable, str(ev.Path(ev.__file__).resolve()), *command[1:],
                        "--arms", "nf4+trim:10", "nf4+trim:5"]
    assert calls[1][-2:] == ["--arms", "offload-3.5:10"]
    assert calls[2][-2:] == ["--arms", "offload-2.5+trim:5"]
    calls.clear()
    monkeypatch.setattr(ev.subprocess, "run", lambda args: calls.append(args) or SimpleNamespace(returncode=9))
    with pytest.raises(SystemExit) as exc:
        ev.main()
    assert exc.value.code == 9 and len(calls) == 1


def test_reference_pass_resumes_per_arm_and_combines_summary(tmp_path, monkeypatch):
    import infer
    import lerobot.envs.factory as factory
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "runs/libero_eval_libero_spatial.jsonl"
    for steps in (10, 5):
        for init in (0, 1):
            saved = row(init=init, steps=steps)
            saved["protocol"].update(policy=infer.POLICY, revision=infer.REVISION)
            ev.append_row(path, saved)
    first_pass = path.read_bytes()
    events, calls = [], []
    monkeypatch.setattr(ev, "preflight", lambda config: events.append(("preflight", config)))
    def make_env(cfg, n_envs):
        events.append(("env",))
        return {cfg.task: {0: SimpleNamespace(close=lambda: None)}}
    monkeypatch.setattr(factory, "make_env", make_env)
    monkeypatch.setattr(factory, "make_env_pre_post_processors", lambda *a: (None, None))
    def load(config, trim):
        events.append(("load", config, trim))
        return SimpleNamespace(config=None), None, None, {}
    monkeypatch.setattr(infer, "load_policy", load)
    def episode(*args):
        task, init, steps = args[7:10]
        calls.append((task, init, steps))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return row(task, init, steps, True, config="offload-3.5"), []
    monkeypatch.setattr(ev, "episode", episode)
    monkeypatch.setattr(ev.sys, "argv", ["eval_libero.py", "--tasks", "0", "--inits", "2", "--arms", "offload-3.5:10"])
    with pytest.raises(KeyboardInterrupt):
        ev.main()
    assert len(ev.read_rows(path)) == 5
    ev.main()
    assert calls == [(0, 0, 10), (0, 1, 10), (0, 1, 10)]
    assert path.read_bytes().startswith(first_pass)
    assert events[:3] == [("preflight", "offload-3.5"), ("env",), ("load", "offload-3.5", False)]
    assert events.count(("load", "offload-3.5", False)) == 2
    ev.main()
    assert len(calls) == 3 and len(ev.read_rows(path)) == 6
    summary = json.loads(path.with_name(path.stem + "_summary.json").read_text())
    assert len(summary["arms"]) == len(summary["paired"]) == 3
    assert all(p["complete_pairs"] == 2 for p in summary["paired"].values())
