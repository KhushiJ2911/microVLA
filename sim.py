"""Batched tabletop: gripper xy/grip, block xy/colors, bin xy/colors, held, time, task."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

IMG_SIZE, ACTION_DIM, STATE_DIM, SCENE_DIM = 64, 3, 3, 19
MAX_STEPS, MAX_STEP = 100, 0.055
BLOCK_HALF, BIN_HALF, GRASP_RADIUS = 0.055, 0.16, 0.075
COLORS = ("red", "green", "blue", "yellow")
VOCAB = ["<pad>", "put", "the", *COLORS, "block", "in", "bin"]
INSTR_LEN = 9
HELDOUT_COMBOS = ((0, 1), (1, 2), (2, 3), (3, 0))
EVAL_SEED_OFFSET = 10_000_000
RGB = ((235, 65, 65), (65, 215, 105), (65, 125, 245), (245, 210, 55))
# Layout: [gripper x,y,grip; blocks xy (4), colors (2); bins xy (4),
# colors (2); held block (-1/0/1); elapsed steps; target block/bin colors].


def instruction_text(task: Tensor) -> str:
    block, bin_color = task.tolist()
    return f"put the {COLORS[int(block)]} block in the {COLORS[int(bin_color)]} bin"


def encode_instruction(task: Tensor) -> Tensor:
    tokens = torch.tensor([1, 2, 0, 7, 8, 2, 0, 9, 0], device=task.device)
    tokens = tokens.expand(task.shape[0], -1).clone()
    tokens[:, 2], tokens[:, 6] = task[:, 0] + 3, task[:, 1] + 3
    return tokens


def reset(n: int, seed: int, device: str | torch.device, split: str = "train") -> Tensor:
    """Jitter distinct grid cells to avoid rejection sampling and initial overlaps."""
    if split not in ("train", "heldout"):
        raise ValueError("split must be 'train' or 'heldout'")
    rng = torch.Generator(device=device).manual_seed(seed)
    combos = [(a, b) for a in range(4) for b in range(4) if a != b
              and (((a, b) in HELDOUT_COMBOS) == (split == "heldout"))]
    tasks = torch.tensor(combos, device=device)[torch.randint(len(combos), (n,), generator=rng, device=device)]
    grid = torch.tensor([(x, y) for x in (-0.65, 0., 0.65) for y in (-0.65, 0., 0.65)], device=device)
    cells = torch.rand(n, 9, generator=rng, device=device).argsort(dim=1)[:, :5]
    xy = grid[cells] + 0.2 * torch.rand(n, 5, 2, generator=rng, device=device) - 0.1
    remaining = torch.arange(4, device=device).expand(n, -1)
    excluded = (remaining == tasks[:, :1]) | (remaining == tasks[:, 1:])
    other = (torch.rand(n, 4, generator=rng, device=device) + excluded * 2).argsort(dim=1)[:, :2]
    block_colors, bin_colors = torch.stack((tasks[:, 0], other[:, 0]), 1), torch.stack((tasks[:, 1], other[:, 1]), 1)
    flips = torch.randint(2, (n, 2), generator=rng, device=device).bool()
    scene = torch.zeros(n, SCENE_DIM, device=device)
    scene[:, :2], scene[:, 2] = xy[:, 0], -1
    scene[:, 3:7], scene[:, 9:13] = xy[:, 1:3].flatten(1), xy[:, 3:5].flatten(1)
    scene[:, 7:9] = torch.where(flips[:, :1], block_colors.flip(1), block_colors)
    scene[:, 13:15] = torch.where(flips[:, 1:], bin_colors.flip(1), bin_colors)
    scene[:, 15], scene[:, 17:19] = -1, tasks
    return scene


def _targets(scene: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    block = (scene[:, 7:9] == scene[:, 17:18]).long().argmax(1)
    bin_index = (scene[:, 13:15] == scene[:, 18:19]).long().argmax(1)
    block_xy = scene[:, 3:7].reshape(-1, 2, 2).gather(1, block[:, None, None].expand(-1, 1, 2))[:, 0]
    bin_xy = scene[:, 9:13].reshape(-1, 2, 2).gather(1, bin_index[:, None, None].expand(-1, 1, 2))[:, 0]
    return block, block_xy, bin_xy


def success(scene: Tensor) -> Tensor:
    block, block_xy, bin_xy = _targets(scene)
    return ((block_xy - bin_xy).abs().amax(1) <= BIN_HALF) & (scene[:, 15] != block)


def step(scene: Tensor, action: Tensor) -> Tensor:
    """Move, carry, then grasp/release; completed episodes remain frozen."""
    result = scene.clone()
    xy = (scene[:, :2] + action[:, :2].clamp(-1, 1) * MAX_STEP).clamp(-1, 1)
    closed = action[:, 2] > 0
    blocks = scene[:, 3:7].reshape(-1, 2, 2)
    carried = scene[:, 15:16] == torch.arange(2, device=scene.device)
    blocks = torch.where(carried[:, :, None], xy[:, None], blocks)
    distance = (blocks - xy[:, None]).square().sum(2)
    nearest = distance.argmin(1)
    grasp = closed & (distance.amin(1) <= GRASP_RADIUS ** 2)
    held = torch.where(scene[:, 15] >= 0, scene[:, 15], torch.where(grasp, nearest, -1))
    held = torch.where(closed, held, -1)
    blocks = torch.where((held[:, None] == torch.arange(2, device=scene.device))[:, :, None], xy[:, None], blocks)
    result[:, :2], result[:, 2] = xy, torch.where(closed, 1., -1.)
    result[:, 3:7], result[:, 15], result[:, 16] = blocks.flatten(1), held, scene[:, 16] + 1
    done = success(scene) | (scene[:, 16] >= MAX_STEPS)
    return torch.where(done[:, None], scene, result)


def expert(scene: Tensor) -> Tensor:
    block, block_xy, bin_xy = _targets(scene)
    carrying = scene[:, 15] == block
    delta = torch.where(carrying[:, None], bin_xy, block_xy) - scene[:, :2]
    arriving = delta.abs().amax(1) <= MAX_STEP
    close = torch.where(carrying, ~arriving, arriving & (scene[:, 15] < 0))
    return torch.cat(((delta / MAX_STEP).clamp(-1, 1), torch.where(close, 1., -1.)[:, None]), 1)


def robot_state(scene: Tensor) -> Tensor:
    return scene[:, :3]


def render(scene: Tensor) -> Tensor:
    axis = torch.linspace(-1, 1, IMG_SIZE, device=scene.device)
    y, x = torch.meshgrid(axis.flip(0), axis, indexing="ij")
    grid = torch.stack((x, y), -1)[None, None]
    palette = torch.tensor(RGB, dtype=scene.dtype, device=scene.device)
    bins = (grid - scene[:, 9:13].reshape(-1, 2, 1, 1, 2)).abs().amax(-1)
    fill, outline = bins <= BIN_HALF, (bins <= BIN_HALF) & (bins >= BIN_HALF - 0.025)
    bin_rgb = palette[scene[:, 13:15].long()].transpose(1, 2)
    tint = fill.to(scene.dtype) * 0.16 + outline.to(scene.dtype) * 0.44
    pixels = 18 + torch.bmm(bin_rgb, tint.flatten(2)).reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    blocks = (grid - scene[:, 3:7].reshape(-1, 2, 1, 1, 2)).abs().amax(-1) <= BLOCK_HALF
    block_rgb = palette[scene[:, 7:9].long()].transpose(1, 2)
    color = torch.bmm(block_rgb, blocks.to(scene.dtype).flatten(2)).reshape_as(pixels)
    pixels = torch.where(blocks.any(1)[:, None], color, pixels)
    delta = (grid[:, 0] - scene[:, None, None, :2]).abs()
    width = torch.where(scene[:, 2] > 0, 0.065, 0.10)[:, None, None]
    vertical = ((delta[..., 0] - width).abs() <= 0.016) & (delta[..., 1] <= 0.085)
    horizontal = ((delta[..., 1] - 0.085).abs() <= 0.016) & (delta[..., 0] <= width)
    horizontal &= (scene[:, 2] > 0)[:, None, None] | (delta[..., 0] >= width - 0.035)
    return torch.where((vertical | horizontal)[:, None], 245, pixels).clamp(0, 255).to(torch.uint8)


def _chunk(scene: Tensor, horizon: int) -> Tensor:
    actions, previous = [], expert(scene)
    for _ in range(horizon):
        done = success(scene) | (scene[:, 16] >= MAX_STEPS)
        action = torch.where(done[:, None], previous, expert(scene))
        actions.append(action)
        scene, previous = step(scene, action), action
    return torch.stack(actions, 1)


@torch.no_grad()
def collect(n_episodes: int, horizon: int, noise: float, seed: int, device: str | torch.device) -> dict[str, Tensor]:
    if n_episodes < 1 or horizon < 1 or noise < 0:
        raise ValueError("n_episodes/horizon must be positive and noise nonnegative")
    scene = reset(n_episodes, seed, device)
    rng = torch.Generator(device=device).manual_seed(seed)
    states, valid = [], []
    for _ in range(MAX_STEPS):
        states.append(scene)
        valid.append(~success(scene))
        action = expert(scene)
        action[:, :2] += noise * torch.randn(n_episodes, 2, generator=rng, device=device)
        scene = step(scene, action)
    samples = torch.stack(states)[torch.stack(valid)]
    return {"scene": samples, "task": samples[:, 17:19].long(), "actions": _chunk(samples, horizon)}


@torch.no_grad()
def evaluate(policy: Callable[[Tensor, Tensor, Tensor], Tensor], n_episodes: int, seed: int,
             device: str | torch.device, replan: int = 5,
             split: str = "train", swap: bool = False, record: int = 0) -> dict:
    if n_episodes < 1 or replan < 1 or record < 0:
        raise ValueError("n_episodes/replan must be positive and record nonnegative")
    scene = reset(n_episodes, seed + EVAL_SEED_OFFSET, device, split)
    if swap:
        scene[:, 17] = torch.where(scene[:, 7] == scene[:, 17], scene[:, 8], scene[:, 7])
    instr, frames, elapsed = encode_instruction(scene[:, 17:19].long()), [], 0
    if record:
        frames.append(render(scene[:record]))
    while elapsed < MAX_STEPS:
        actions = policy(render(scene), robot_state(scene), instr)
        if actions.ndim != 3 or actions.shape[0] != n_episodes or actions.shape[1] < 1 or actions.shape[2] != ACTION_DIM:
            raise ValueError("policy must return actions [B, H, 3] with H >= 1")
        for i in range(min(replan, actions.shape[1], MAX_STEPS - elapsed)):
            scene = step(scene, actions[:, i])
            elapsed += 1
            if record:
                frames.append(render(scene[:record]))
    per_episode = success(scene)
    result = {"success": per_episode.float().mean().item(), "per_episode": per_episode}
    if record:
        result["frames"] = torch.stack(frames)
    return result


def save_gif(frames: Tensor, path: str, fps: int = 10, labels: list[str] | None = None) -> None:
    from textwrap import fill

    from PIL import Image, ImageDraw

    frames = frames.detach().cpu().permute(0, 1, 3, 4, 2).numpy()
    images = []
    for frame in frames:
        canvas = Image.new("RGB", (frame.shape[0] * IMG_SIZE * 3, IMG_SIZE * 3 + (34 if labels else 0)))
        for i, tile in enumerate(frame):
            canvas.paste(Image.fromarray(tile).resize((IMG_SIZE * 3, IMG_SIZE * 3), Image.Resampling.NEAREST), (i * IMG_SIZE * 3, 0))
            if labels:
                ImageDraw.Draw(canvas).multiline_text((i * IMG_SIZE * 3 + 3, IMG_SIZE * 3 + 4), fill(labels[i], 28), fill="white")
        images.append(canvas)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)


if __name__ == "__main__":
    scene, frames = reset(4, 7, "cpu"), []
    for _ in range(MAX_STEPS):
        frames.append(render(scene))
        scene = step(scene, expert(scene))
    save_gif(torch.stack(frames), "assets/sim_preview.gif", labels=[instruction_text(t) for t in scene[:, 17:19]])
