"""Paired closed-loop pi0.5 evaluation; --summary needs neither CUDA nor LIBERO."""
import argparse, hashlib, itertools, json, math, os, re, subprocess, sys, textwrap, time
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"  # Must precede every simulator import.
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520, libero_90=400)
SEED_SCHEME = "sha256(suite:task:init:chunk), first 4 bytes little-endian; CPU torch.Generator"
def protocol(args):
    return dict(version=3, suite=args.suite, replan=args.replan, env_seed=args.seed,
                max_episode_steps=args.max_steps or MAX_STEPS[args.suite], num_steps_wait=args.num_steps_wait,
                no_op_action=[0, 0, 0, 0, 0, 0, -1], observation_height=args.observation_height,
                observation_width=args.observation_width, fps=20, init_state_selection="episode index (0-based)",
                official_inits_per_task=50, seed_scheme=SEED_SCHEME)
def seed(suite, task, init, chunk):
    return int.from_bytes(hashlib.sha256(f"{suite}:{task}:{init}:{chunk}".encode()).digest()[:4], "little")

def parse_arm(label):
    if not re.fullmatch(r"(?:nf4|offload-\d+(?:\.\d+)?)(?:\+trim)?:[1-9]\d*", label):
        raise argparse.ArgumentTypeError("arm must be nf4[+trim]:steps or offload-GiB[+trim]:steps")
    placement, steps = label.split(":")
    if placement.startswith("offload-") and float(placement.removeprefix("offload-").removesuffix("+trim")) <= 0:
        raise argparse.ArgumentTypeError("offload budget must be positive")
    return f"{placement}:{int(steps)}"
def normalize(row):
    row = dict(row)
    spec = dict(row.get("protocol") or {})
    if spec.get("version") == 2 and (spec.get("placement"), spec.get("trim")) == ("nf4", True):
        spec.pop("placement"); spec.pop("trim"); spec["version"] = 3
    label = parse_arm(row.get("arm", f"nf4+trim:{row['steps']}"))
    config = label.split(":")[0]
    placement, trim = config.removesuffix("+trim"), config.endswith("+trim")
    if any(row.get(k, v) != v for k, v in dict(steps=int(label.split(":")[1]), placement=placement, trim=trim).items()):
        raise ValueError("Inconsistent arm metadata")
    return {**row, "arm": label, "placement": placement, "trim": trim, "protocol": spec or None}
def row_key(row):
    return row["arm"], row["task_id"], row["init_index"]
def preflight(config):
    import infer
    if config.startswith("offload-"):
        policy = infer.empty_policy(infer.checkpoint(), trim=config.endswith("+trim"))
        units, resident = infer.placement(policy, float(config.removeprefix("offload-").removesuffix("+trim")))
        infer.check_pinned(infer.pinned_bytes(units, resident))  # Same 2 GiB reserve; meta model, no CUDA.

def read_rows(path):
    """Repair only an uncommitted tail; never hide corruption in completed lines."""
    rows = {}
    if path.exists():
        with path.open("r+b") as f:
            while line := f.readline():
                if not line.endswith(b"\n"):
                    f.truncate(f.tell() - len(line))
                    os.fsync(f.fileno())
                    break
                row = normalize(json.loads(line))
                key = row_key(row)
                if key in rows and rows[key] != row: raise ValueError(f"Conflicting completed episode: {key}")
                rows[key] = row
    return rows
def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def wilson(successes, n):
    if not n: return [None, None]
    z = 1.959963984540054
    p, d = successes / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0., center - half), min(1., center + half)]
def mcnemar(a, b):
    n = a + b
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)

def summarize(rows):
    rows = list(map(normalize, rows))
    protocols = {json.dumps(r.get("protocol"), sort_keys=True) for r in rows}
    if len(protocols) > 1 or "null" in protocols: raise ValueError("Summary has incompatible or missing protocols")
    arms, tasks, episodes, comparisons = {}, {}, {}, {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm]
        n, wins, calls = len(selected), sum(r["success"] for r in selected), sum(r["policy_calls"] for r in selected)
        latency = sum(r["mean_policy_latency_s"] * r["policy_calls"] for r in selected)
        arms[arm] = dict(episodes=n, successes=wins, rate=wins / n, wilson95=wilson(wins, n),
                         mean_policy_latency_s=latency / calls if calls else None)
    for r in rows:
        task = tasks.setdefault(str(r["task_id"]), {"task_text": r["task_text"], **{a: [0, 0] for a in arms}})
        task[r["arm"]][0] += int(r["success"])
        task[r["arm"]][1] += 1
        episodes.setdefault((r["protocol"]["suite"], r["task_id"], r["init_index"]), {})[r["arm"]] = r["success"]
    for a, b in itertools.combinations(arms, 2):
        table = dict(both_succeed=0, only_a=0, only_b=0, both_fail=0)
        eligible = [e for e in episodes.values() if a in e or b in e]
        for e in eligible:
            if a in e and b in e:
                table[("both_succeed" if e[b] else "only_a") if e[a] else ("only_b" if e[b] else "both_fail")] += 1
        comparisons[f"{a} vs {b}"] = dict(arm_a=a, arm_b=b, table=table, complete_pairs=sum(table.values()),
            incomplete_pairs=len(eligible) - sum(table.values()), mcnemar_exact_p=mcnemar(table["only_a"], table["only_b"]))
    return dict(arms=arms, per_task=tasks, paired=comparisons, protocol=rows[0]["protocol"] if rows else None,
                external_sanity_anchor=dict(label="openpi published; different checkpoint/framework; not paired",
                    success_percent=dict(libero_spatial=98.8, libero_object=98.2, libero_goal=98.0, libero_10=92.4)),
                **{k: sorted({r[k] for r in rows}) for k in ("n_action_steps", "checkpoint_n_action_steps",
                                                           "chunk_size", "max_episode_steps", "inits_requested")})
def report(path, rows):
    summary = summarize(rows.values())
    target = path.with_name(path.stem + "_summary.json")
    temporary = target.with_suffix(".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(summary, indent=2), flush=True)
    arms = list(summary["arms"])
    print("Task | " + " | ".join(arms) + " | successes/episodes", flush=True)
    for task, values in sorted(summary["per_task"].items(), key=lambda item: int(item[0])):
        print(" | ".join([task, *[f"{values[a][0]}/{values[a][1]}" for a in arms], values["task_text"]]), flush=True)
    anchor = summary["external_sanity_anchor"]
    for arm, values in summary["arms"].items():
        reference = anchor["success_percent"].get(summary["protocol"]["suite"], "unavailable")
        print(f"{arm}: {100 * values['rate']:.1f}% | openpi {reference}% ({anchor['label']})", flush=True)

def episode(env, policy, pre, post, env_pre, env_post, suite, task, init, steps, video, protocol):
    import torch
    from lerobot.scripts.lerobot_eval import rollout
    env.call("_max_episode_steps")  # Build lazy envs; public reset has no init-index selector.
    raw = env._envs[0]
    if env.num_envs != 1 or env.use_async_envs or not hasattr(raw, "init_state_id"):
        raise RuntimeError("Exact init selection requires this LeRobot's synchronous local LIBERO env")
    if not 0 <= init < len(raw._init_states): raise ValueError(f"Init {init} unavailable for task {task}")
    raw.init_state_id, raw.num_steps_wait = init, protocol["num_steps_wait"]  # reset owns settling.
    checkpoint_replan = policy.config.n_action_steps
    if not 1 <= protocol["replan"] <= policy.config.chunk_size: raise ValueError("replan must be between 1 and the checkpoint chunk_size")
    original, latencies, frames = policy.predict_action_chunk, [], []
    def chunk(batch, **kwargs):
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                            generator=torch.Generator().manual_seed(seed(suite, task, init, len(latencies)))).to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = original(batch, **{**kwargs, "noise": noise, "num_steps": steps})
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        if not torch.isfinite(actions).all(): raise RuntimeError("Predicted actions contain NaN/Inf")
        return actions
    policy.predict_action_chunk, policy.config.n_action_steps = chunk, protocol["replan"]
    start = time.perf_counter()
    try:
        env_pre.reset(); env_post.reset()
        data = rollout(env, policy, env_pre, env_post, pre, post, seeds=[protocol["env_seed"]],
                       render_callback=(lambda e: frames.append(e.render()[0].copy())) if video else None)
    finally:
        policy.predict_action_chunk, policy.config.n_action_steps = original, checkpoint_replan
    return dict(task_id=task, task_text=env.call("task_description")[0], init_index=init, steps=steps,
                success=bool(data["success"].any()), episode_length=data["done"].shape[1],
                policy_calls=len(latencies), mean_policy_latency_s=sum(latencies) / len(latencies),
                wall_time_s=time.perf_counter() - start, n_action_steps=protocol["replan"],
                checkpoint_n_action_steps=checkpoint_replan, chunk_size=policy.config.chunk_size,
                max_episode_steps=env.call("_max_episode_steps")[0]), frames

def paired_video(paths, rows, target, fps):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    scenes, frames = [], []
    for path in paths:
        with av.open(str(path)) as container:
            scenes.append([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    captions = [f'{r.get("arm", str(r["steps"]) + " steps")} | success={r["success"]}\n' + textwrap.fill(r["task_text"], 32) for r in rows]
    banner = max(c.count("\n") + 1 for c in captions) * 18 + 12
    for i in range(max(map(len, scenes))):
        scene = np.concatenate([s[min(i, len(s) - 1)] for s in scenes], axis=1)  # Hold shorter ending.
        frame = Image.new("RGB", (scene.shape[1], scene.shape[0] + banner))
        frame.paste(Image.fromarray(scene), (0, banner))
        for j, caption in enumerate(captions):
            ImageDraw.Draw(frame).multiline_text((j * scene.shape[1] // len(scenes) + 6, 4), caption,
                                               font=ImageFont.load_default(size=14), fill="white")
        frames.append(np.asarray(frame))
    save_video(target, frames, fps)
def save_video(target, frames, fps):
    from lerobot.utils.io_utils import write_video
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.mp4")
    write_video(temporary, frames, fps)
    temporary.replace(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=list(MAX_STEPS), default="libero_spatial")
    for flag, default in (("inits", 10), ("videos", 0), ("replan", 5), ("num-steps-wait", 10),
                          ("observation-height", 256), ("observation-width", 256), ("seed", 7)):
        parser.add_argument(f"--{flag}", type=int, default=default)
    parser.add_argument("--arms", nargs="+", type=parse_arm, default=["nf4+trim:10", "nf4+trim:5"])
    parser.add_argument("--tasks", nargs="+", type=int)
    parser.add_argument("--summary", action="store_true", help="summarize all persisted episodes without loading policy/env")
    parser.add_argument("--max-steps", "--episode-length", type=int, help="policy actions, excluding settling")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms) or args.inits < 1 or args.videos < 0:
        parser.error("use unique arms, --inits >= 1, --videos >= 0")
    if min(args.replan, args.observation_height, args.observation_width, args.max_steps or 1) < 1 or args.max_steps == 0 or args.num_steps_wait < 0 or not 0 <= args.seed < 2**32:
        parser.error("replan/dimensions/max-steps must be positive, wait >= 0, seed in [0, 2**32)")
    path = Path(f"runs/libero_eval_{args.suite}.jsonl")
    rows = read_rows(path)
    if args.summary: return report(path, rows)
    groups = {a.split(":")[0]: [] for a in args.arms}
    for arm in args.arms: groups[arm.split(":")[0]].append(arm)
    if len(groups) > 1:
        for arms in groups.values():
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--arms", *arms])
            if result.returncode: raise SystemExit(result.returncode)
        return
    import infer
    spec = {**protocol(args), "policy": infer.POLICY, "revision": infer.REVISION}
    if any(r.get("protocol") != spec for r in rows.values()):
        raise ValueError("Resume log has incompatible protocol/checkpoint metadata; archive it before changing settings")
    config = next(iter(groups))
    try: preflight(config)
    except infer.OffloadRAMError as exc:
        print(f"Offload RAM preflight refused before GPU/env initialization: {exc}. Completed JSONL episodes are preserved.", flush=True)
        raise SystemExit(1) from exc
    from lerobot.envs.configs import LiberoEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    cfg = LiberoEnvConfig(task=args.suite, task_ids=args.tasks, episode_length=spec["max_episode_steps"],
                          observation_height=args.observation_height, observation_width=args.observation_width)
    envs = make_env(cfg, n_envs=1)[args.suite]
    policy = None
    try:
        for ordinal, (task, init) in enumerate((t, i) for t in sorted(envs) for i in range(args.inits)):
            stem = Path(f"assets/libero/{args.suite}_task{task}_init{init}")
            scenes = [stem.with_name(stem.name + f"_{a.replace(':', '_')}.mp4") for a in args.arms]
            for arm, scene in zip(args.arms, scenes, strict=True):
                steps = int(arm.split(":")[1])
                if (arm, task, init) in rows: continue
                if policy is None:
                    policy, pre, post, _ = infer.load_policy(config.removesuffix("+trim"), trim=config.endswith("+trim"))
                    env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
                row, frames = episode(envs[task], policy, pre, post, env_pre, env_post, args.suite, task, init, steps, ordinal < args.videos, spec)
                row.update(arm=arm, placement=config.removesuffix("+trim"), trim=config.endswith("+trim"), protocol=spec, inits_requested=args.inits)
                append_row(path, row)  # Durable before video encoding or the next episode.
                rows[arm, task, init] = row
                print(json.dumps(row), flush=True)
                if frames: save_video(scene, frames, cfg.fps)
            if ordinal < args.videos and len(scenes) > 1:
                if all(p.exists() for p in scenes):
                    paired_video(scenes, [rows[a, task, init] for a in args.arms], stem.with_name(stem.name + f"_{config}_paired.mp4"), cfg.fps)
                else:
                    print(f"Missing scene cache for {stem}; completed episodes remain skipped", flush=True)
            if init + 1 == args.inits: envs[task].close()  # Release EGL before the next task.
    finally:
        for env in envs.values(): env.close()
    report(path, rows)
if __name__ == "__main__": main()
