from dataclasses import asdict
import sys
from types import SimpleNamespace

import pytest

from train import Config, make_logger


def args(tmp_path, **kwargs):
    return SimpleNamespace(**{**dict(logger="wandb", runs=tmp_path / "training", resume=False,
                                    wandb_project="microvla", wandb_run_name=None), **kwargs})


def test_wandb_scalars_config_and_resume(tmp_path, fake_wandb):
    options = args(tmp_path, wandb_project="cpu-tests", wandb_run_name="tiny")
    settings, model_config = dict(steps=10, lr=.001), Config()
    writer = make_logger(options, settings, model_config, tmp_path / "tb")
    writer.add_scalar("train/loss", .25, 7)
    writer.add_scalar("val/loss", .5, 7)
    writer.close()
    run = fake_wandb.runs[0]
    assert run.kwargs == dict(project="cpu-tests", name="tiny", config={**settings, "model_config": asdict(model_config)},
                              dir=str(options.runs), id=None, resume=None)
    assert run.logs == [({"train/loss": .25}, 7), ({"val/loss": .5}, 7)]
    assert run.finished and (options.runs / "wandb_run_id").read_text() == run.id
    assert not (tmp_path / "tb").exists()
    options.resume = True
    writer = make_logger(options, settings, model_config, tmp_path / "tb")
    writer.close()
    assert fake_wandb.runs[1].kwargs["id"] == run.id
    assert fake_wandb.runs[1].kwargs["resume"] == "allow"
    # A fresh run in the same directory must not reuse the previous run's ID.
    options.resume = False
    writer = make_logger(options, settings, model_config, tmp_path / "tb")
    writer.close()
    assert fake_wandb.runs[2].id != run.id


def test_wandb_legacy_resume_without_id(tmp_path, fake_wandb):
    options = args(tmp_path, resume=True)
    writer = make_logger(options, {}, Config(), tmp_path / "tb")
    writer.close()
    assert fake_wandb.runs[0].kwargs["id"] is None
    assert fake_wandb.runs[0].kwargs["project"] == "microvla"
    assert fake_wandb.runs[0].kwargs["resume"] == "allow"
    assert (options.runs / "wandb_run_id").read_text() == fake_wandb.runs[0].id


@pytest.mark.parametrize("logger", ["none", "tensorboard"])
def test_backends_do_not_import_wandb(tmp_path, monkeypatch, logger):
    monkeypatch.setitem(sys.modules, "wandb", None)
    if logger == "none":
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    writer = make_logger(args(tmp_path, logger=logger), {}, Config(), tmp_path / "tb")
    writer.add_scalar("train/loss", .25, 7)
    writer.close()
    if logger == "none":
        assert list(tmp_path.iterdir()) == []
    else:
        assert list((tmp_path / "tb").glob("events.out.tfevents.*"))
        assert not (tmp_path / "training").exists()
