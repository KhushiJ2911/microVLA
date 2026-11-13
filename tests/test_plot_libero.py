import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest

import plot_libero as plotter
from eval_libero import normalize

ROOT = Path(__file__).resolve().parents[1]
ARMS = [*plotter.COLORS, "tiny:10", "tiny-blind:10", "tiny-no-wrist:10", "tiny-wrong-text:10", "tiny-no-state:10"]


@pytest.fixture
def sample(tmp_path):
    target = tmp_path / "runs/libero_eval_libero_spatial.jsonl"
    target.parent.mkdir()
    shutil.copyfile(ROOT / "runs/libero_eval_libero_spatial.jsonl", target)
    return target


@pytest.mark.parametrize("n_arms", [1, 2, 3, 4, 8])
def test_render_synthetic_arms(sample, tmp_path, monkeypatch, n_arms):
    template = json.loads(sample.read_text().splitlines()[0])
    rows = []
    for i, arm in enumerate(ARMS[:n_arms]):
        for init, calls, seconds in [(0, 1, .1), (1, 3, .3)]:
            row = copy.deepcopy(template)
            row.update(arm=arm, placement=arm.split(":")[0].removesuffix("+trim"),
                       trim="+trim" in arm, steps=int(arm.split(":")[1]),
                       init_index=init, task_id=0, success=(init == i), episode_length=20 + 10 * init,
                       policy_calls=calls, mean_policy_latency_s=seconds)
            if arm.startswith("tiny"):
                row = normalize(row)
                row["model_identity"] = dict(checkpoint_path="tiny.pt", checkpoint_sha256="synthetic")
                if arm != "tiny:10":
                    row["model_identity"]["ablation"] = arm.split(":")[0].removeprefix("tiny-")
            rows.append(row)
    # Only the first arm has task 9: every comparison involving it is incomplete.
    rows.append({**rows[0], "task_id": 9, "episode_length": 40})
    sample.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"arm":')
    original = sample.read_bytes()
    monkeypatch.chdir(tmp_path)
    figures = []
    close = plotter.plt.close
    monkeypatch.setattr(plotter.plt, "close", figures.append)
    try:
        target = plotter.plot()
        assert target == Path("assets/libero_results.png")
        with Image.open(target) as image:
            assert image.format == "PNG" and image.width == 2560 and image.height >= 960
            assert image.convert("L").getextrema()[0] < 100
        fig = figures[-1]
        success, tasks, latency = fig.axes
        title = fig._suptitle.get_text()
        assert "libero_spatial" in title and "nf4+trim:10 n=3" in title
        assert title.count("Paired McNemar") == n_arms * (n_arms - 1) // 2
        summary = plotter.summarize(rows)
        for pair, comparison in summary["paired"].items():
            assert f"{pair}: p={comparison['mcnemar_exact_p']:.4g}" in title
        if n_arms > 1:
            assert "2 complete, 1 incomplete" in title
        assert success.lines[-1].get_label() == "openpi published (not paired)"
        assert list(tasks.get_xticks()) == list(range(10))
        assert len(tasks.patches) == 10 * n_arms
        for i, arm in enumerate(ARMS[:n_arms]):
            assert success.patches[i].get_height() == pytest.approx(100 * summary["arms"][arm]["rate"])
            assert latency.patches[i].get_height() == pytest.approx(
                1000 * summary["arms"][arm]["mean_policy_latency_s"])
            assert success.patches[i].get_facecolor() == tasks.patches[i * 10].get_facecolor()
            assert success.patches[i].get_facecolor() == latency.patches[i].get_facecolor()
        assert "30.0 steps/episode" in [t.get_text() for t in latency.texts]
        assert sample.read_bytes() == original
    finally:
        for fig in figures:
            if hasattr(fig, "savefig"):
                close(fig)


def test_cli_copied_sample(sample, tmp_path):
    original = sample.read_bytes()
    result = subprocess.run([sys.executable, str(ROOT / "plot_libero.py"), "--suite", "libero_spatial"],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""},
                            capture_output=True, text=True, check=True)
    assert "assets/libero_results.png" in result.stdout
    assert (tmp_path / "assets/libero_results.png").is_file()
    assert sample.read_bytes() == original


def test_no_complete_pairs(sample, tmp_path, monkeypatch):
    rows = [json.loads(line) for line in sample.read_text().splitlines()[:2]]
    rows[1]["init_index"] = 1
    sample.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.chdir(tmp_path)
    titles = []
    original = plotter.plt.Figure.savefig
    def savefig(fig, *args, **kwargs):
        titles.append(fig._suptitle.get_text())
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(plotter.plt.Figure, "savefig", savefig)
    plotter.plot()
    assert "p=n/a (0 complete, 2 incomplete)" in titles[0]


@pytest.fixture
def all_suites(sample):
    paths = {"libero_spatial": sample}
    for suite in list(plotter.SUITES)[1:]:
        paths[suite] = sample.with_name(f"libero_eval_{suite}.jsonl")
        shutil.copyfile(ROOT / "runs" / paths[suite].name, paths[suite])
    return paths


@pytest.mark.parametrize("suite", list(plotter.SUITES)[1:])
def test_suite_output_preserves_spatial(all_suites, tmp_path, monkeypatch, suite):
    monkeypatch.chdir(tmp_path)
    spatial = Path("assets/libero_results.png")
    spatial.parent.mkdir()
    spatial.write_bytes(b"existing Spatial figure")
    original = all_suites[suite].read_bytes()
    target = plotter.plot(suite)
    assert target == Path(f"assets/libero_results_{suite}.png")
    with Image.open(target) as image:
        assert image.format == "PNG" and image.width == 2560
    assert spatial.read_bytes() == b"existing Spatial figure"
    assert all_suites[suite].read_bytes() == original


@pytest.mark.parametrize("missing", [None, "libero_goal"])
def test_all_suites(all_suites, tmp_path, monkeypatch, missing):
    if missing:
        all_suites.pop(missing).unlink()
    # Ignore an unfinished append, without repairing the source file.
    with all_suites["libero_spatial"].open("a") as stream:
        stream.write('{"arm":')
    originals = {suite: path.read_bytes() for suite, path in all_suites.items()}
    monkeypatch.chdir(tmp_path)
    figures = []
    close = plotter.plt.close
    monkeypatch.setattr(plotter.plt, "close", figures.append)
    try:
        target = plotter.plot_all()
        assert target == Path("assets/libero_suites.png")
        with Image.open(target) as image:
            assert image.format == "PNG" and image.size == (1600, 960)
        fig = figures[-1]
        ax, = fig.axes
        assert [t.get_text() for t in ax.get_xticklabels()] == [plotter.SUITES[s] for s in all_suites]
        summaries = [plotter.read_suite(s)[1] for s in all_suites]
        bars = [container for container in ax.containers if hasattr(container, "patches")]
        assert len(bars) == 2
        for arm, container in zip(("nf4+trim:5", "tiny:10"), bars):
            for summary, bar, segment in zip(summaries, container.patches,
                                              container.errorbar.lines[2][0].get_segments()):
                values = summary["arms"][arm]
                assert bar.get_height() == pytest.approx(100 * values["rate"])
                assert segment[:, 1] == pytest.approx([100 * v for v in values["wilson95"]])
        anchor = next(line for line in ax.lines if line.get_label() == "openpi published (external anchor)")
        expected = dict(libero_spatial=98.8, libero_object=98.2, libero_goal=98.0, libero_10=92.4)
        assert list(anchor.get_ydata()) == [expected[s] for s in all_suites]
        assert anchor.get_marker() == "D" and anchor.get_linestyle() == "None"
        assert "trained on Spatial only" in fig._supxlabel.get_text()
        assert "Object / Goal / LIBERO-10 are zero-shot" in fig._supxlabel.get_text()
        assert {suite: path.read_bytes() for suite, path in all_suites.items()} == originals
    finally:
        for fig in figures:
            if hasattr(fig, "savefig"):
                close(fig)


def test_all_cli(all_suites, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "plot_libero.py"), "--all"],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""},
                            capture_output=True, text=True, check=True)
    assert "assets/libero_suites.png" in result.stdout
    assert (tmp_path / "assets/libero_suites.png").is_file()
    assert not (tmp_path / "assets/libero_results.png").exists()


def test_all_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="No LIBERO suite JSONL files found"):
        plotter.plot_all()
