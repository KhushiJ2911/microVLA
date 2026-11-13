# microVLA

A pi0.5-style vision-language-action model trained from scratch on an RTX 3050 Laptop GPU (4 GB).

![training progression](assets/training.gif)

Same scene, two instructions: the policy follows the language, not a memorized trajectory.

![language swap](assets/language_swap.gif)

## Results (RTX 3050 4 GB, 10.1M params)

| Metric | Value |
|---|---|
| Success, seen colour pairs (512 episodes) | **100.0%** |
| Success, held-out colour pairs never targeted in training | **91.8%** |
| Success, same scene with the instruction swapped to the other block | **97.1%** |
| Training time | 13 min SigLIP distillation + 31 min training (8,000 steps, batch 256) |
| Peak training VRAM | 1.9 GB |
| Action-chunk latency (10 actions, 10 flow steps, batch 1) | 64 ms |

## Run

```bash
python train.py --compile          # distill SigLIP -> ViT, then train; TensorBoard logs in runs/tb
python train.py --demo             # write assets/training.gif and assets/language_swap.gif
tensorboard --logdir runs          # http://localhost:6006
python train.py --compile --resume # continue after an interruption (checkpoint every 250 steps)
```

## Pretrained pi0.5: paired LIBERO evaluation

```bash
PYTHONPATH= /home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python eval_libero.py --suite libero_spatial --inits 10 --arms nf4+trim:10 nf4+trim:5 --videos 4
PYTHONPATH= /home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python eval_libero.py --suite libero_spatial --inits 10 --arms offload-3.5:10
PYTHONPATH= /home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python eval_libero.py --suite libero_spatial --summary
```

Add `--tasks 0 1` to select tasks. The runner sets `MUJOCO_GL=egl`, uses cached weights,
and defaults to `nf4+trim:10 nf4+trim:5`, loading the trimmed NF4 policy once.
Each task/init pair runs the arms in argument order. All resets select the same LIBERO init state and reset seed; each chunk
gets CPU-generated noise seeded by SHA-256 of suite, task, init and chunk index,
independent of the arm, pass, and earlier episodes. The second command adds the bf16,
untrimmed 10-step reference to the same JSONL. Use identical task/init selection and
protocol flags in both passes. Multiple placements in one `--arms` invocation run in
sequential worker processes; arms sharing a placement and trim configuration interleave
on each episode. NF4 and bf16 are never resident together. Before an offload worker
initializes the simulator or GPU, it estimates pinned RAM with the meta model and
uses `infer.check_pinned` with its 2 GiB reserve. Refusal exits with completed rows preserved.

Defaults follow the coordinator-verified openpi LIBERO protocol: execute the first
5 actions of each 50-action chunk, render both cameras at 256x256, seed every
environment reset with 7, and let LeRobot's reset perform 10 no-op settle steps
with `[0,0,0,0,0,0,-1]`. The policy-action limits exclude settling: spatial 220,
object 280, goal 300, libero_10 520, libero_90 400. Override these with `--replan`,
`--max-steps` (alias `--episode-length`), `--observation-height`,
`--observation-width`, `--seed`, and `--num-steps-wait`. Init indices are 0 through
`--inits - 1`; the default is 10. Set `--inits 50` for the official count per task;
finished episodes from a shorter run are reused.

LeRobot's `rollout()` prepares canonical images/state, applies the environment and
checkpoint preprocessors, calls `select_action()`, then applies the checkpoint and
environment postprocessors before stepping the simulator. The policy queue's
`n_action_steps` is temporarily set to `--replan` for each episode, keeping only
the first requested actions of each full prediction. Every row records the effective
`n_action_steps`, checkpoint `checkpoint_n_action_steps`, `chunk_size`, episode
limit, requested init count, `arm`, `placement`, `trim`, and a `protocol` object containing
the shared evaluation settings. Model identity is stored separately in each row. `policy_calls` counts chunk predictions;
`mean_policy_latency_s` times synchronized chunk inference, excluding processors and
noise generation/transfer. `wall_time_s` includes reset and the rollout, excluding
model loading and video encoding.

Each finished episode is appended and fsynced to `runs/libero_eval_<suite>.jsonl`.
Restarting skips completed arm/task/init keys and repairs an unfinished final line.
Existing version-2/3 rows are normalized to version 4 in memory without rewriting completed lines.
Resume rejects changed protocol settings or changed model identity within a requested arm; archive the log
or run in another output directory when changing them. Increasing `--inits`, changing
task selection, or requesting videos does not change the protocol.
`--summary` summarizes all saved episodes without loading the policy or simulator:
success rates with Wilson 95% intervals per arm, a per-task table with one column per arm,
and a paired 2x2 table plus two-sided exact McNemar p-value for every pair of arms present.
Mean policy latency per arm is weighted by chunk call count. Incomplete pairs are counted separately and
excluded from the paired test. The summary is also written to
`runs/libero_eval_<suite>_summary.json`, including the saved protocol. Per-arm
rates include all finished episodes, including incomplete pairs. Published openpi
rates (Spatial 98.8%, Object 98.2%, Goal 98.0%, LIBERO-10 92.4%) appear as an external
sanity anchor from a different checkpoint/framework, never as a paired comparison.

Run `PYTHONPATH= /home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python plot_libero.py --suite libero_spatial`
to write the three-panel results figure to `assets/libero_results.png`. On this laptop,
prefix evaluation commands with `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=0,1`
to render the simulator on the Intel iGPU without taking memory from the 4 GB NVIDIA
card; NVIDIA EGL with the NF4 policy went out of memory.

Set `--videos` on the initial run (default 0). The first requested pairs get captioned
`*_paired.mp4` clips under `assets/libero/`, with arms in argument order within each placement
at the simulator's 20 Hz; the shorter episode holds its final frame. Scene clips are
cached for resumed pairing. Results are saved before encoding: a crash during encoding
can leave a completed episode without a scene clip, which is reported without rerunning
that episode. Exact init selection uses this installed LeRobot's private local
`_envs[0].init_state_id`; its public reset API has no init-state selector.

## Tiny model on LIBERO-Spatial

`train_libero.py` imports the model, flow loss, cached sampler and bf16 autocast from
`train.py`. The LIBERO configuration has **10,480,711 parameters**: two 96x96 cameras
with a shared ViT and learned camera embeddings, 8 state values discretized into
256 bins each, word tokens, and 16 relative 7-D actions per chunk. Evaluation executes
the first 5 actions. The deterministic word vocabulary comes from all ten task strings;
ID 0 is padding, ID 1 is unknown, and instructions are padded to the longest task.

The cache selects episodes by exact language match with LIBERO's Spatial benchmark,
then discovers local files by projecting only `episode_index`/`task_index` with PyArrow;
unrelated episodes in mixed files are skipped. Expected source: `HuggingFaceVLA/libero` revision
`86958911c0f959db2bbbdb107eb3e17c5f9c798e`, 432 episodes / 52,970 frames.
Both embedded cameras are decoded with PIL bilinear resize into uint8 CHW memmaps.
No additional flip is applied: the dataset and LeRobot's canonical observations already
have the same orientation (LeRobot rotates simulator cameras by 180 degrees).
Cache and inference use the same PIL resize. The state order is end-effector xyz,
axis-angle xyz, and the two gripper positions. Commands stay in LIBERO's relative
xyz/rotation/gripper convention; there is **no gripper sign inversion**. The adapter
unnormalizes and clips commands to [-1,1], feeding the same LeRobot rollout and identity
environment postprocessor used by the pretrained policy.

The 96 px cache occupies **2,933,478,600 bytes** (2.93 GB / 2.73 GiB), plus a small JSON
manifest. Preprocessing checks per-episode counts and task IDs before allocation, then
validates frame IDs and decoded counts against metadata lengths. Missing or short
episodes are listed in the error. It commits one parquet file at a time and safely
replays an interrupted file, including episodes spanning files. Completed caches
are reused; changed source/configuration is rejected. The split is by episode,
approximately 5% held out per task. Training-only q01/q99 statistics map state/actions
to roughly [-1,1]; statistics, tokenizer, configuration and cache fingerprint are saved
in every checkpoint. Future chunks repeat the last action at the episode boundary.

```bash
# Use the existing venv; no downloads or installs.
export PYTHONPATH=
PY=/home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python

# One-time preprocessing (restart the same command after interruption).
"$PY" train_libero.py --preprocess

# Cached SigLIP-B/16 -> shared ViT, then flow matching on Spatial actions.
"$PY" train_libero.py
# Resume either phase, including optimizer and RNG state.
"$PY" train_libero.py --resume

# After training, append tiny results to the existing paired evaluation log.
"$PY" eval_libero.py --suite libero_spatial --inits 10 --arms tiny:10 \
  --tiny-ckpt runs/libero_tiny/latest.pt --videos 4
"$PY" plot_libero.py --suite libero_spatial
```

Input ablations use the same paired runner and checkpoint:

```bash
PYTHONPATH= /home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python eval_libero.py \
  --suite libero_spatial --inits 10 --tiny-ckpt runs/libero_tiny/latest.pt \
  --arms tiny:10 tiny-blind:10 tiny-no-wrist:10 tiny-wrong-text:10 tiny-no-state:10
```

`blind` zeros both camera images; `no-wrist` zeros only `observation.images.image2`;
`wrong-text` uses Spatial task `(i + 1) % 10`'s instruction in benchmark order,
independent of task selection; `no-state` zeros the normalized state (the saved
training normalization center). Each ablation is recorded in its model identity
and appears as a separate arm in the shared JSONL, `--summary`, and results plot.
Existing `tiny:10` episodes are reused when the checkpoint and protocol match.

Defaults: 1,000 distillation steps at batch 16 (randomly selecting either camera),
then 8,000 action-training steps at batch 32, horizon 16, AdamW at 3e-4, four-pixel
random shifts on GPU, bf16 autocast, validation and atomic `latest.pt` every 250 steps.
SigLIP is frozen, local-cache-only, and sees 192 px images to match the student's 12x12
patch grid. Only the vision path receives pretrained-teacher supervision; language,
backbone and action expert start from scratch. TensorBoard logs are in
`runs/libero_tiny/tb`. Checkpoints preserve both training phases, the distillation head,
optimizer, NumPy sampling RNG, and CPU/CUDA torch RNGs. Resume requires the same run
settings; use another `--runs` for a different training schedule.

The target runtime is **about 1.5–2.5 hours on the RTX 3050**, excluding preprocessing;
the initial 2.13-hour estimate assumes 1.2 training steps/s and 1 distillation step/s.
These are unmeasured estimates. Progress prints measured phase ETA, peak process RAM
and peak allocated VRAM every 50 steps. Disk images stay mapped, copied batches are
pinned for GPU transfer, and mapped image pages are released after each gather to keep
host residency bounded. The RAM/VRAM targets still require a coordinator GPU run.
`--compile` is optional and uses one compiler worker to limit host memory; eager mode
is the default. The cache plus atomic checkpoint pair needs about 3.2 GB disk, excluding
TensorBoard and videos. No intermediate model snapshots or teacher-feature cache accumulate.

`tiny:<flow-steps>` uses the same environment creation, task/init selection, seed 7,
10 wait steps, 220-step Spatial limit, replan 5, chunk seed function, video path,
JSONL schema and synchronized model-only latency measurement as the pretrained arms.
Noise shape follows the checkpoint (16x7 by default). Tiny preprocessing and action
unnormalization sit outside the inference timer. Version 4 protocols contain only
environment/eval settings; each row records either pretrained policy/revision or tiny
checkpoint path/SHA-256. A changed checkpoint cannot silently extend an existing tiny
arm. Summaries and plots accept the different model identities while requiring the
same protocol, and old version-2/3 rows remain readable unchanged.

The Spatial-trained checkpoint also accepts `--suite libero_object`, `libero_goal`,
`libero_10` (or any supported suite) with `--arms tiny:10` for zero-shot evaluation;
results go to `runs/libero_eval_<suite>.jsonl`. Only `tiny-wrong-text` requires Spatial.
Inference maps unseen words to the saved `<unk>` ID (ID 0/padding if absent) and keeps
only the first saved tokenizer-length words, without changing the vocabulary or weights.
Training retains its length check, and in-vocabulary Spatial tokens are unchanged.
Every new tiny JSONL row includes `instruction_tokenization.oov_fraction` (over all
words before truncation; 0 for empty text) and `instruction_tokenization.truncated`.
For `tiny-wrong-text`, these describe the substituted instruction.

This revision's episode metadata has incorrect `data/file_index` values; preprocessing
does not use them. Spatial episodes 1261–1692 are in `file-308.parquet` through
`file-376.parquet`, and file 308 also contains unrelated episodes. Those files must be
present locally (use `--source PATH` for another snapshot). Missing episodes stop
preprocessing before cache allocation.
Real embedded camera bytes were successfully decoded; GPU training and a simulator
rollout with trained weights have not been run here.

Run all CPU checks with:

```bash
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/keyu/Documents/ChatGPT/Mujuco/.venv/bin/python -m pytest -q tests/
```

## Architecture (pi0.5, scaled down)

- **Vision**: SigLIP-style ViT, 64x64 image, 8x8 patches -> 64 tokens.
- **Prefix**: image tokens + instruction tokens + robot state discretized into 256-bin tokens (pi0.5 feeds state as text tokens, not a continuous input).
- **Backbone + action expert**: Gemma-style blocks (RMSNorm, GeGLU, RoPE, multi-query attention). Two sets of weights, one joint attention per layer. The prefix attends bidirectionally to itself; action tokens attend to the prefix and each other; the prefix never sees the actions.
- **Flow matching** with adaRMSNorm timestep conditioning in the action expert (pi0.5's change over pi0), Beta(1.5, 1) time sampling, 10 Euler steps at inference with the prefix KV cache computed once.

## What made it work

A ViT trained from scratch on this data could not localize the small objects: after 2,000 steps the predicted move direction had cosine 0.10 with the expert's and task success was 0%. Replacing the image with ground-truth object positions trained fine (cosine 0.86), which isolated the problem to vision.

Real pi0.5 starts from a pretrained SigLIP, so the fix bootstraps the same way. A frozen SigLIP-B/16 sees each render upsampled to 128 px, which gives the same 8x8 patch grid as the student. The student ViT is trained for 3,000 steps to match its patch features. With that initialization, the direction cosine reached 0.90 and success 86% after only 1,000 training steps.

Training data comes from a scripted expert with DART noise injection: the executed action is perturbed but the label is the clean expert chunk, so the policy learns to recover from its own mistakes.
