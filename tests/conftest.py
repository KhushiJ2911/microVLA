import os
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def fake_wandb(monkeypatch):
    module = ModuleType("wandb")
    module.runs = []

    def init(**kwargs):
        assert os.environ["WANDB_MODE"] == "offline"
        run = SimpleNamespace(id=kwargs["id"] or f"run-{len(module.runs)}", kwargs=kwargs,
                              logs=[], finished=False)
        run.log = lambda data, *, step: run.logs.append((data, step))

        def finish():
            run.finished = True

        run.finish = finish
        module.runs.append(run)
        return run

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_MODE", "offline")
    return module
