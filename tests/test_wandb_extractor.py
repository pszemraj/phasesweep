"""W&B extractor test with a stubbed API.

We don't want a real W&B account in CI, so we install a fake `wandb.apis.public`
module before importing the extractor, then drive different scenarios.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from phasesweep.config import WandbExtractor
from phasesweep.extractors import ExtractorError, TrialContext, run_extractor


class _FakeRun:
    def __init__(self, name: str, state: str, summary: dict):
        self.display_name = name
        self.state = state
        self.summary = summary


class _FakeApi:
    def __init__(self, runs_for_filter):
        self._runs_for_filter = runs_for_filter

    def runs(self, path, filters):
        name = filters["display_name"]
        return self._runs_for_filter(path, name)


def _install_fake_wandb(api_factory):
    # `wandb` top-level
    wandb_mod = types.ModuleType("wandb")
    sys.modules["wandb"] = wandb_mod
    # `wandb.apis`
    apis_mod = types.ModuleType("wandb.apis")
    sys.modules["wandb.apis"] = apis_mod
    wandb_mod.apis = apis_mod  # type: ignore[attr-defined]
    # `wandb.apis.public`
    public_mod = types.ModuleType("wandb.apis.public")

    class Api:
        def __init__(self):
            self._inner = api_factory()

        def runs(self, path, filters):
            return self._inner.runs(path, filters)

    public_mod.Api = Api
    sys.modules["wandb.apis.public"] = public_mod
    apis_mod.public = public_mod  # type: ignore[attr-defined]


@pytest.fixture
def fake_wandb_finished(monkeypatch):
    def factory():
        return _FakeApi(
            runs_for_filter=lambda path, name: [
                _FakeRun(name=name, state="finished", summary={"eval/loss": 0.123})
            ]
        )

    _install_fake_wandb(factory)
    yield
    for k in ("wandb", "wandb.apis", "wandb.apis.public"):
        sys.modules.pop(k, None)


@pytest.fixture
def fake_wandb_missing(monkeypatch):
    def factory():
        return _FakeApi(runs_for_filter=lambda path, name: [])

    _install_fake_wandb(factory)
    yield
    for k in ("wandb", "wandb.apis", "wandb.apis.public"):
        sys.modules.pop(k, None)


def _ctx(tmp_path: Path) -> TrialContext:
    return TrialContext(
        experiment="exp",
        phase="ph",
        trial_id=7,
        trial_dir=tmp_path,
        run_name="exp-ph-7",
        return_code=0,
        duration_seconds=0.0,
    )


def test_wandb_extractor_finds_metric(fake_wandb_finished, tmp_path):
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        run_name_template="{experiment}-{phase}-{trial_id}",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    assert run_extractor(_ctx(tmp_path), cfg) == pytest.approx(0.123)


def test_wandb_extractor_timeout(fake_wandb_missing, tmp_path):
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        run_name_template="{experiment}-{phase}-{trial_id}",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=0.1,  # bail fast
    )
    with pytest.raises(ExtractorError, match="not found or metric"):
        run_extractor(_ctx(tmp_path), cfg)
