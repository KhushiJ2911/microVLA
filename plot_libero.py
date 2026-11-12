"""Plot completed LIBERO episodes without loading the policy or simulator."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_libero import MAX_STEPS, summarize, wilson

COLORS = {"nf4+trim:10": "#4477AA", "nf4+trim:5": "#EE6677", "offload-3.5:10": "#228833"}


def plot(suite="libero_spatial"):
    path = Path(f"runs/libero_eval_{suite}.jsonl")
    # Read only: the coordinator may be appending an episode while we plot.
    rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
    if not rows:
        raise ValueError(f"No completed episodes in {path}")
    summary = summarize(rows)
    if summary["protocol"]["suite"] != suite:
        raise ValueError(f"Saved protocol does not match {suite}")
    arms = list(summary["arms"])
    values = list(summary["arms"].values())
    colors = [COLORS.get(a, plt.get_cmap("tab10")(i)) for i, a in enumerate(arms)]
    title = [f"LIBERO results — {suite}", "Episodes per arm: " + ", ".join(
        f"{a} n={v['episodes']}" for a, v in zip(arms, values))]
    for pair, comparison in summary["paired"].items():
        n = comparison["complete_pairs"]
        p = f"{comparison['mcnemar_exact_p']:.4g}" if n else "n/a"
        title.append(f"Paired McNemar {pair}: p={p} ({n} complete, "
                     f"{comparison['incomplete_pairs']} incomplete)")
    fig, (success, tasks, latency) = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    fig.suptitle("\n".join(title), fontsize=11)
    rates = [100 * v["rate"] for v in values]
    intervals = [wilson(v["successes"], v["episodes"]) for v in values]
    errors = [[max(0, rates[i] - 100 * low) for i, (low, high) in enumerate(intervals)],
              [max(0, 100 * high - rates[i]) for i, (low, high) in enumerate(intervals)]]
    bars = success.bar(arms, rates, color=colors, yerr=errors, capsize=5)
    for bar, v in zip(bars, values):
        success.annotate(f"n={v['episodes']}", (bar.get_x() + bar.get_width() / 2, 3),
                         ha="center", va="bottom", fontsize=9)
    anchor = summary["external_sanity_anchor"]["success_percent"].get(suite)
    if anchor is not None:
        success.axhline(anchor, color="0.3", ls="--", label="openpi published (not paired)")
        success.legend(loc="upper center", fontsize=8)
    success.set(title="Success rate (95% Wilson CI)", ylabel="Success (%)", ylim=(0, 110))
    width = .8 / len(arms)
    for i, (arm, color) in enumerate(zip(arms, colors)):
        counts = [summary["per_task"].get(str(t), {}).get(arm, [0, 0]) for t in range(10)]
        task_rates = [100 * wins / n if n else float("nan") for wins, n in counts]
        tasks.bar([t + (i - (len(arms) - 1) / 2) * width for t in range(10)],
                  task_rates, width, color=color, label=arm)
    tasks.set(title="Per-task success (gaps = no episodes)", xlabel="Task ID", ylabel="Success (%)",
              xticks=range(10), ylim=(0, 110))
    tasks.legend(fontsize=8)
    milliseconds = [1000 * v["mean_policy_latency_s"] if v["mean_policy_latency_s"] is not None
                    else float("nan") for v in values]
    bars = latency.bar(arms, milliseconds, color=colors)
    for bar, arm, ms in zip(bars, arms, milliseconds):
        selected = [r["episode_length"] for r in rows if r["arm"] == arm]
        latency.annotate(f"{sum(selected) / len(selected):.1f} steps/episode",
                         (bar.get_x() + bar.get_width() / 2, ms if ms == ms else 0),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    latency.set(title="Mean policy latency per chunk", ylabel="Latency (ms)")
    latency.margins(y=.2)
    for ax in (success, latency):
        ax.tick_params(axis="x", labelrotation=15)
    target = Path("assets/libero_results.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=list(MAX_STEPS), default="libero_spatial")
    print(plot(parser.parse_args().suite))


if __name__ == "__main__":
    main()
