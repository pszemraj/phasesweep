"""An MLflow rerun in its trial's directory is scored from the evidence any of the trial's runs hold."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep.evidence.baseline import EvidenceBaseline
from phasesweep.evidence.evaluation import (
    ExtractorError,
    NothingToScoreError,
    TrialContext,
    run_extractor,
)
from phasesweep.evidence.mlflow import observation_id
from phasesweep.evidence.models import MlflowExtractor, MlflowRequiredGate
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import session
from tests.tracking_helpers import bind_mlflow_attempt, mlflow_run, run_mlflow_worker_in_process

_OBJECTIVE = MlflowExtractor(type="mlflow", metric_key="eval/loss", select="step", step=100)


def _rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    earlier_steps: list[tuple[int, float]],
    run_mode: str = "managed",
    gate: MlflowRequiredGate | None = None,
    own_metrics: dict[str, float] | None = None,
) -> tuple[dict[str, str | int], TrialContext]:
    """Bind a rerun in a reused directory whose own run logged ``own_metrics``, over one earlier run.

    The earlier run logged ``eval/loss`` at ``earlier_steps``, ``n_params``, the param ``lr``
    and the tag ``dataset``; the fake MLflow serves both runs to an in-process worker.

    :param Path tmp_path: The attempt directory.
    :param pytest.MonkeyPatch monkeypatch: Replaces the MLflow client and worker launch.
    :param list[tuple[int, float]] earlier_steps: The earlier run's objective history.
    :param str run_mode: Ownership of both runs, defaults to ``managed``.
    :param MlflowRequiredGate | None gate: Optional gate in the query, defaults to None.
    :param dict[str, float] | None own_metrics: What the rerun's own run logged, defaults to
        nothing.
    :return tuple[dict[str, str | int], TrialContext]: The earlier binding and the rerun's context.
    """
    import mlflow

    binding, ctx = bind_mlflow_attempt(tmp_path)
    binding = {**binding, "run_mode": run_mode}
    directory = session.attempt_tracking_dir(tmp_path, ctx.generation_id, ctx.attempt_id)
    session.write_record(
        directory / session.EXPECTED_BINDING_FILE,
        {k: v for k, v in binding.items() if k != "run_id"},
    )
    session.write_record(directory / session.BINDING_FILE, binding)
    prior = {**binding, "run_id": "stopped", "generation_id": "d" * 32, "attempt_id": "e" * 32}
    session.write_record(
        session.attempt_tracking_dir(tmp_path, "d" * 32, "e" * 32) / session.BINDING_FILE, prior
    )
    selections = [_OBJECTIVE]
    if gate is not None:
        selections += [
            MlflowExtractor(type="mlflow", metric_key=k, select="last") for k in gate.metrics
        ]
    ctx = dataclasses.replace(
        ctx,
        baseline=EvidenceBaseline(True, {}, 1.0),
        earlier_attempts=(("d" * 32, "e" * 32),),
        mlflow_query={
            "revision": 1,
            "objective": _OBJECTIVE.model_dump(),
            "constraints": {},
            "gates": [] if gate is None else [gate.model_dump()],
            "selections": {observation_id(s): s.model_dump() for s in selections},
        },
    )
    own = mlflow_run(binding)
    own.data.metrics = dict(own_metrics or {})
    earlier = mlflow_run(prior, metrics={"eval/loss": earlier_steps[-1][1], "n_params": 42.0})
    earlier.data.params = {"lr": "0.1"}
    earlier.data.tags = {**earlier.data.tags, "dataset": "v2"}
    histories = {
        ("native-id", "eval/loss"): [],
        ("stopped", "eval/loss"): [
            SimpleNamespace(value=v, step=s, timestamp=s) for s, v in earlier_steps
        ],
        ("stopped", "n_params"): [SimpleNamespace(value=42.0, step=0, timestamp=1)],
    }
    client = SimpleNamespace(
        get_run={"native-id": own, "stopped": earlier}.__getitem__,
        get_metric_history=lambda run_id, key: histories[run_id, key],
    )
    monkeypatch.setattr(mlflow, "MlflowClient", lambda tracking_uri: client)
    run_mlflow_worker_in_process(monkeypatch)
    return prior, ctx


@pytest.mark.parametrize("earlier_reaches_the_step", [True, False], ids=["held", "absent"])
def test_an_empty_managed_rerun_scores_the_selected_point_its_earlier_run_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, earlier_reaches_the_step: bool
) -> None:
    """A managed rerun in its trial's directory whose own run logged nothing is scored from the selected point in its earlier attempt's run, names that run as the source, marks the result as an earlier attempt's and keeps its own run as the capture's binding; when the merged history lacks the point, the attempt leaves nothing to score."""
    steps = [(50, 0.5), (100, 0.25)] if earlier_reaches_the_step else [(50, 0.5)]
    prior, ctx = _rerun(tmp_path, monkeypatch, earlier_steps=steps)
    provenance: JsonObject = {}
    if not earlier_reaches_the_step:
        with pytest.raises(NothingToScoreError, match="step 100"):
            run_extractor(ctx, _OBJECTIVE, provenance=provenance)
        assert ctx.earlier_files == []
        return
    assert run_extractor(ctx, _OBJECTIVE, provenance=provenance) == 0.25
    source = provenance["source"]
    assert isinstance(source, dict) and source["run_id"] == "stopped"
    assert source["evaluation"] == {"progress": {"axis": "step", "value": 100}, "timestamp": 100}
    assert ctx.earlier_files == ["mlflow:stopped"]
    capture = provenance["remote_capture"]
    assert isinstance(capture, dict) and capture["binding"]["run_id"] == "native-id"
    assert capture["earlier_bindings"] == [prior]
    assert "objective_run_id" not in capture


@pytest.mark.parametrize("run_mode", ["managed", "trainer"])
def test_a_rerun_that_logged_other_metrics_but_not_the_objective_is_not_scored_from_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_mode: str
) -> None:
    """A rerun whose own run logged a metric but not the objective ran without reporting it, so the point its earlier attempt's run holds is not its result: it leaves nothing to score, and a metric PhaseSweep delivered to the run does not count as the trainer's."""
    logged = {"train/loss": 0.4, "phasesweep.objective": 0.9}
    _, ctx = _rerun(
        tmp_path, monkeypatch, earlier_steps=[(100, 0.25)], run_mode=run_mode, own_metrics=logged
    )
    with pytest.raises(NothingToScoreError, match="though its trainer logged train/loss,"):
        run_extractor(ctx, _OBJECTIVE)
    assert ctx.earlier_files == []


def test_a_rerun_holding_only_delivered_metrics_is_scored_from_its_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun's run that holds only the ``phasesweep.*`` metrics an earlier reading delivered logged nothing of its trainer's, so a later reading still scores it from its earlier attempt's run."""
    _, ctx = _rerun(
        tmp_path,
        monkeypatch,
        earlier_steps=[(100, 0.25)],
        own_metrics={"phasesweep.objective": 0.25},
    )
    assert run_extractor(ctx, _OBJECTIVE) == 0.25
    assert ctx.earlier_files == ["mlflow:stopped"]


def test_an_empty_rerun_meets_its_gate_from_the_earlier_runs_metric_param_and_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun whose own run logged nothing passes an ``mlflow_required`` gate on the metric, param and tag its trial's earlier run logged, and the capture keeps those values."""
    gate = MlflowRequiredGate(
        type="mlflow_required", metrics=["n_params"], params=["lr"], tags=["dataset"]
    )
    _, ctx = _rerun(tmp_path, monkeypatch, earlier_steps=[(100, 0.25)], gate=gate)
    provenance: JsonObject = {}
    assert run_extractor(ctx, _OBJECTIVE, provenance=provenance) == 0.25
    capture = provenance["remote_capture"]
    assert isinstance(capture, dict)
    assert (capture["params"], capture["tags"]) == ({"lr": "0.1"}, {"dataset": "v2"})
    n_params = MlflowExtractor(type="mlflow", metric_key="n_params", select="last")
    assert capture["observations"][observation_id(n_params)]["value"] == 42.0


def test_a_trainer_owned_rerun_without_the_point_anywhere_leaves_nothing_to_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trainer-owned rerun in its trial's directory whose run and earlier run both lack the selected point leaves nothing to score; the same read outside a reused directory fails as an ordinary extractor error."""
    _, ctx = _rerun(tmp_path, monkeypatch, earlier_steps=[(50, 0.5)], run_mode="trainer")
    with pytest.raises(NothingToScoreError, match="step 100"):
        run_extractor(ctx, _OBJECTIVE)
    fresh = dataclasses.replace(ctx, baseline=EvidenceBaseline(False, {}), earlier_files=[])
    with pytest.raises(ExtractorError, match="step 100") as raised:
        run_extractor(fresh, _OBJECTIVE)
    assert type(raised.value) is ExtractorError


def test_a_trainer_owned_rerun_that_never_bound_a_run_leaves_nothing_to_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trainer-owned rerun in its trial's directory that exits without binding a run leaves nothing to score instead of failing the trial; outside a reused directory it fails."""
    _, ctx = _rerun(tmp_path, monkeypatch, earlier_steps=[(100, 0.25)], run_mode="trainer")
    directory = session.attempt_tracking_dir(tmp_path, ctx.generation_id, ctx.attempt_id)
    (directory / session.BINDING_FILE).unlink()
    with pytest.raises(NothingToScoreError, match="never bound"):
        run_extractor(ctx, _OBJECTIVE)
    fresh = dataclasses.replace(ctx, baseline=EvidenceBaseline(False, {}))
    with pytest.raises(ExtractorError, match="never bound") as raised:
        run_extractor(fresh, _OBJECTIVE)
    assert type(raised.value) is ExtractorError
