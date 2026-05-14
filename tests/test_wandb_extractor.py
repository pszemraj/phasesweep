"""W&B extractor tests with a stubbed API."""

from __future__ import annotations

import sys
import types

import pytest

from phasesweep.config import WandbExtractor
from phasesweep.evidence import ExtractorError, run_extractor
from tests.conftest import make_trial_context


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


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``wandb.apis.public.Api`` backed by a supplied runs callable."""

    def install(runs_for_filter):
        wandb_mod = types.ModuleType("wandb")
        apis_mod = types.ModuleType("wandb.apis")
        public_mod = types.ModuleType("wandb.apis.public")

        class Api:
            def __init__(self):
                self._inner = _FakeApi(runs_for_filter)

            def runs(self, path, filters):
                return self._inner.runs(path, filters)

        public_mod.Api = Api
        wandb_mod.apis = apis_mod  # type: ignore[attr-defined]
        apis_mod.public = public_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis", apis_mod)
        monkeypatch.setitem(sys.modules, "wandb.apis.public", public_mod)

    return install


def test_wandb_extractor_finds_metric(fake_wandb, tmp_path):
    fake_wandb(
        lambda path, name: [_FakeRun(name=name, state="finished", summary={"eval/loss": 0.123})]
    )
    cfg = WandbExtractor(
        type="wandb",
        entity="me",
        project="proj",
        run_name_template="{experiment}-{phase}-{trial_id}",
        metric_key="eval/loss",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )
    ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
    assert run_extractor(ctx, cfg) == pytest.approx(0.123)


def test_wandb_extractor_timeout(fake_wandb, tmp_path):
    fake_wandb(lambda path, name: [])
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
        ctx = make_trial_context(tmp_path, experiment="exp", phase="ph", trial_id=7)
        run_extractor(ctx, cfg)
