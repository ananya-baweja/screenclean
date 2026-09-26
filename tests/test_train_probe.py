"""The trainer's speed probe: torch.compile "auto" and a schedule length set from the time budget."""

import torch
from test_train import make_split, tiny_cfg
from torch import nn

from screenclean.data.pairs_dataset import PairsDataset
from screenclean.train import trainer as trainer_mod
from screenclean.train.trainer import Trainer


class FakeCompiled(nn.Module):
    """Stands in for torch.compile's result (real compilation needs a C++ compiler and takes long)."""

    def __init__(self, model, fail=False):
        super().__init__()
        self.inner, self.fail, self.calls = model, fail, 0

    def forward(self, x):
        self.calls += 1
        if self.fail:
            raise RuntimeError("inductor exploded")
        return self.inner(x)


def _data(tmp_path):
    train = make_split(tmp_path / "train", 6, seed=1)
    val = make_split(tmp_path / "val", 2, seed=2, prefix="v")
    return PairsDataset(sorted(train.glob("*.tar"))), PairsDataset(sorted(val.glob("*.tar")))


def probe_cfg(**kw):
    return tiny_cfg(
        iters="auto",
        budget_minutes=0.05,
        compile="auto",
        probe_iters=4,
        iters_round=1,
        val_every=1000,
        val_seconds=0.0,
        optim={"warmup": 20},
        **kw,
    )


def test_compile_auto_and_iterations_from_budget(tmp_path, monkeypatch):
    made = []

    def fake_compile(model, **kw):
        made.append(FakeCompiled(model))
        return made[-1]

    monkeypatch.setattr(trainer_mod.torch, "compile", fake_compile)
    train, val = _data(tmp_path)
    summary = Trainer(probe_cfg(), tmp_path / "run", [train], val).fit()
    probe = summary["speed_probe"]
    assert probe["eager_it_per_s"] > 0 and probe["compiled_it_per_s"] > 0 and probe["error"] is None
    assert probe["choice"] in ("eager", "compiled") and made and made[0].calls >= 4
    assert summary["state"] == "done" and summary["iteration"] == summary["iters"] >= 21  # warmup + 1

    # a resumed run keeps the decided schedule and doesn't probe again
    again = Trainer(probe_cfg(), tmp_path / "run", [train], val).fit()
    assert (
        again["iters"] == summary["iters"] and again["speed_probe"] == probe and again["iters_this_run"] == 0
    )


def test_failed_compile_falls_back_to_eager(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_mod.torch, "compile", lambda model, **kw: FakeCompiled(model, fail=True))
    train, val = _data(tmp_path)
    summary = Trainer(probe_cfg(), tmp_path / "run", [train], val).fit()
    probe = summary["speed_probe"]
    assert probe["choice"] == "eager" and "inductor exploded" in probe["error"]
    assert summary["state"] == "done" and summary["iteration"] == summary["iters"]


def test_plan_iters_counts_validation_time(tmp_path):
    cfg = tiny_cfg(iters="auto", budget_minutes=100, val_every=2000, val_seconds=100, optim={"warmup": 1000})
    t = Trainer(cfg, tmp_path / "run", [])
    t.cfg["iters_round"] = 1000
    # 100 min at 2 it/s = 12,000 iterations; with 100 s of validation per 2,000 iterations: 10,000
    assert t._plan_iters(2.0) == 10000
    assert t._plan_iters(0.01) == 2000  # never shorter than warmup + one round


def test_forced_compile_uses_the_compiled_module(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(
        trainer_mod.torch, "compile", lambda model, **kw: made.append(FakeCompiled(model)) or made[-1]
    )
    train, val = _data(tmp_path)
    summary = Trainer(tiny_cfg(compile=True), tmp_path / "run", [train], val).fit()
    assert summary["speed_probe"]["choice"] == "compiled" and made[0].calls == 6
    assert isinstance(torch.load(tmp_path / "run" / "last.pt", weights_only=False)["model"], dict)
