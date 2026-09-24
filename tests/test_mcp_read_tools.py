"""MCP server status, winners, and snapshot reads through tools. Logic that does not need a real detached runner."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import optuna
import pytest

from phasesweep.config import (
    Experiment,
    JsonEnvelopeExtractor,
    load_config,
)
from phasesweep.engine import (
    read_winners,
    run_experiment,
)
from phasesweep.engine.generation import (
    _write_generation_state,
)
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
)
from phasesweep.engine.state import (
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
)
from phasesweep.evidence.models import objective_evidence_assurance
from phasesweep.mcp.errors import (
    ConfigChangedError,
    McpToolError,
    RunPersistentStateUnavailableError,
    RunSnapshotUnavailableError,
    UnknownRunError,
)
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.server import (
    AwaitRunResult,
    GetRunResultsResult,
    GetRunStatusResult,
)
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.mcp.tools import PhaseSweepMCP
from tests.conftest import (
    write_constant_trainer,
)
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    _catalog,
    _config,
    _drift_experiment,
    _write_experiment_config,
    make_mcp_app,
    make_run_handle,
    stage_dead_run,
    write_mcp_catalog,
    write_run_status,
)


def _record_published_run_snapshot(
    tmp_path: Path,
    *,
    extractor: object | None = None,
) -> tuple[str, Path, Path, Path]:
    """Publish one generation whose id also has a completed MCP run snapshot."""
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = _drift_experiment(tmp_path, trainer, extractor=extractor)
    _write_experiment_config(config, experiment)
    catalog = _catalog(tmp_path, config)
    _app, registry, store = make_mcp_app(catalog)
    reg = registry.get("srv")
    run_id = "srv-frozen-result"
    stage_dead_run(store, run_id, config, reg.id, cleanup_uncertain=False)
    run_experiment(experiment, generation_id=run_id)
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="complete",
        result_snapshot=capture_result_snapshot(experiment, generation_id=run_id),
    )
    return run_id, trainer, config, catalog


@pytest.mark.parametrize(
    "method_name",
    # Only the winners read needs a published run, and a real run is slow.
    ["status", pytest.param("winners", marks=pytest.mark.integration)],
)
def test_run_tools_read_launched_config_snapshot_after_catalog_edit(
    tmp_path: Path,
    method_name: str,
) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = _drift_experiment(tmp_path, trainer)
    config = tmp_path / "srv.yaml"
    _write_experiment_config(config, experiment)
    catalog = _catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    _app, registry, store = make_mcp_app(catalog)
    reg = registry.get("srv")
    run_id = "srv-launched"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
        )
    )
    if method_name == "winners":
        run_experiment(experiment, generation_id=run_id)
    _write_experiment_config(config, _drift_experiment(tmp_path, trainer, phase_name="edited"))

    restarted_registry = Registry.load(catalog)
    restarted_app = PhaseSweepMCP(restarted_registry, store)

    by_run = getattr(restarted_app, method_name)(run_id=run_id)
    assert [phase["phase"] for phase in by_run["phases"]] == ["p"]
    if method_name == "status":
        assert by_run["run"]["state"] == "running"
    else:
        assert by_run["phases"][0]["metric"] == 0.5


@pytest.mark.integration
def test_winners_by_run_id_defaults_to_redacted_params_after_decatalog(
    tmp_path: Path,
) -> None:
    trainer = write_constant_trainer(tmp_path)
    old_exp = _drift_experiment(tmp_path, trainer, name="old")
    old_config = tmp_path / "old.yaml"
    _write_experiment_config(old_config, old_exp)
    run_id = "old-launched"
    run_experiment(old_exp, generation_id=run_id)
    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(
        write_mcp_catalog(tmp_path, {"other": other_config}, visible_params={"other": "all"})
    )
    snapshot = old_config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id="old",
            config_sha256=hashlib.sha256(snapshot).hexdigest(),
        )
    )

    result = app.winners(run_id=run_id)

    assert result["experiment_id"] == "old"
    assert result["phases"][0]["params"] == {"lr": "<redacted>"}


@pytest.mark.integration
def test_decataloged_live_run_leaves_config_drift_unknown(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    old_exp = _drift_experiment(tmp_path, trainer, name="old")
    old_config = tmp_path / "old.yaml"
    _write_experiment_config(old_config, old_exp)
    run_id = "old-live-published"
    # The published summary carries the launched config's own fingerprint, so
    # only the missing catalog entry can leave the drift unknown.
    run_experiment(old_exp, generation_id=run_id)

    other_config = _config(tmp_path, name="other")
    app, _registry, store = make_mcp_app(write_mcp_catalog(tmp_path, {"other": other_config}))
    snapshot = old_config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id="old",
            config_sha256=hashlib.sha256(snapshot).hexdigest(),
        )
    )

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.result_source == "current_shared_study"
    assert results.result_source == "current_shared_study"
    assert status.represented_generation_id == run_id
    assert results.represented_generation_id == run_id
    assert status.published_config_matches_current is None
    assert results.published_config_matches_current is None


@pytest.mark.parametrize("method_name", ["status", "winners", "await_run"])
def test_run_tools_reject_config_snapshot_hash_mismatch(
    tmp_path: Path,
    method_name: str,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    run_id = "srv-mismatch"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=registry.get("srv").id,
            config_sha256=hashlib.sha256(snapshot + b"\n").hexdigest(),
        )
    )

    with pytest.raises(RunSnapshotUnavailableError, match="saved config snapshot"):
        if method_name == "await_run":
            asyncio.run(app.await_run(run_id))
        else:
            getattr(app, method_name)(run_id=run_id)


@pytest.mark.parametrize("method_name", ["status", "winners"])
def test_read_tools_require_a_known_run_id(tmp_path: Path, method_name: str) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))
    method = getattr(app, method_name)

    with pytest.raises(UnknownRunError, match="unknown run id"):
        method("nope-123")


@pytest.mark.parametrize("read_method", ["status", "winners"])
@pytest.mark.integration
def test_mcp_run_reads_redact_downgraded_persistent_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_method: str,
) -> None:
    """Live status and result reads never expose a downgraded ledger's details."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, registry, store = make_mcp_app(catalog)
    handle = store.get(run_id)
    assert handle is not None
    complete = store.recorded_terminal_status(handle)
    assert complete is not None
    write_run_status(store, **{**complete, "result_snapshot_state": "pending"})
    monkeypatch.setattr(store, "_runner_is_live", lambda _handle: True)

    experiment = registry.get("srv").experiment
    study = optuna.load_study(
        study_name=f"{experiment.experiment}::p",
        storage=experiment.storage,
    )
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION - 1)

    with pytest.raises(RunPersistentStateUnavailableError) as excinfo:
        getattr(app, read_method)(run_id)

    message = str(excinfo.value)
    assert run_id in message
    assert "persistent study state" in message
    assert str(tmp_path) not in message


@pytest.mark.parametrize("integrity", ["failed", "permission_denied"])
@pytest.mark.integration
def test_mcp_winners_hide_unusable_frozen_publications_before_and_after_live_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    integrity: str,
) -> None:
    """A frozen publication verdict suppresses winners across both snapshot paths."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    handle = store.get(run_id)
    assert handle is not None
    complete = store.recorded_terminal_status(handle)
    assert complete is not None
    snapshot = json.loads(json.dumps(complete["result_snapshot"]))
    snapshot["status"]["publication_integrity"] = integrity
    unusable = {**complete, "result_snapshot": snapshot}

    write_run_status(store, **unusable)
    frozen = app.winners(run_id=run_id)

    assert frozen["publication_integrity"] == integrity
    assert frozen["winner_count"] == 0
    assert frozen["phases"] == []

    # The runner can finish its snapshot while a live storage read is in
    # progress. The second branch must apply the same winner suppression to
    # the snapshot that replaced that temporary live view.
    write_run_status(store, **{**unusable, "result_snapshot_state": "pending"})
    monkeypatch.setattr(store, "_runner_is_live", lambda _handle: True)
    original_live_status = app._live_status_payload
    finalized = False

    def finalize_during_live_read(
        experiment_id: str,
        experiment: Experiment,
        saved: RunHandle,
    ) -> dict[str, Any]:
        nonlocal finalized
        status = original_live_status(experiment_id, experiment, saved)
        assert not finalized
        finalized = True
        write_run_status(store, **unusable)
        return status

    monkeypatch.setattr(app, "_live_status_payload", finalize_during_live_read)
    completed_during_read = app.winners(run_id=run_id)

    assert finalized
    assert completed_during_read["publication_integrity"] == integrity
    assert completed_during_read["winner_count"] == 0
    assert completed_during_read["phases"] == []


@pytest.mark.parametrize("integrity", ["failed", "permission_denied"])
@pytest.mark.integration
def test_mcp_winners_hide_unusable_live_publications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    integrity: str,
) -> None:
    """A live publication failure or access denial cannot expose stale winners."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    experiment = app._registry.get("srv").experiment
    store.status_path(run_id).unlink()
    monkeypatch.setattr(store, "_runner_is_live", lambda _handle: True)
    if integrity == "failed":
        _generation_summary_path(experiment, run_id).unlink()
    else:
        import phasesweep.engine.read as engine_read
        from phasesweep.engine.publication import PublicationPointer

        monkeypatch.setattr(
            engine_read,
            "_resolve_publication_pointer",
            lambda _experiment: PublicationPointer(
                state="permission_denied",
                generation_id=run_id,
                error="injected publication access denial",
            ),
        )

    results = app.winners(run_id=run_id)

    assert results["result_source"] == "current_shared_study"
    assert results["publication_integrity"] == integrity
    assert results["winner_count"] == 0
    assert results["phases"] == []


@pytest.mark.parametrize("kind", ["wandb", "json"])
@pytest.mark.integration
def test_restored_reader_frozen_mcp_results_need_no_remote_access(
    tmp_path, monkeypatch, wandb_worker_sdk, kind
):
    from phasesweep.config import JsonExtractor, WandbExtractor

    wandb_worker_sdk("""
        class Api:
            def __init__(self, **kwargs): pass
            def run(self, path):
                return type("Run", (), {"state": "finished", "summary_metrics": {"eval/loss": 0.25}})()
    """)
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(
        tmp_path,
        extractor=WandbExtractor(
            type="wandb", entity="e", project="p", metric_key="eval/loss", timeout_seconds=5
        )
        if kind == "wandb"
        else JsonExtractor(type="json", path="r.json", key="x"),
    )
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    wandb_worker_sdk("raise AssertionError('deleted remote run')")
    app, _registry, _store = make_mcp_app(catalog)
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))
    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1
    assert results.metric.objective_evidence.kind == kind
    assert results.metric.objective_evidence.source_identity_keyed == (kind == "wandb")
    assert not results.metric.objective_evidence.evaluation_policy_bound
    assert status.metric == results.metric


@pytest.mark.integration
def test_published_results_keep_their_own_metric_after_a_catalog_metric_edit(
    tmp_path: Path,
) -> None:
    """A run snapshot keeps x/minimize after the catalog changes to y/maximize."""
    run_id, trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, _store = make_mcp_app(catalog)
    baseline_status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    baseline_results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))
    assert baseline_results.published_config_matches_current is True
    assert baseline_status.published_config_matches_current is True

    _write_experiment_config(
        config,
        _drift_experiment(tmp_path, trainer, metric_name="y", goal="maximize"),
    )
    restarted, _restarted_registry, _restarted_store = make_mcp_app(catalog)
    status = GetRunStatusResult.model_validate(restarted.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(restarted.winners(run_id=run_id))

    assert (results.metric.name, results.metric.goal) == ("x", "minimize")
    assert (status.metric.name, status.metric.goal) == (results.metric.name, results.metric.goal)
    assert results.result_context == "represented_generation"
    assert results.published_config_matches_current is False
    assert status.published_config_matches_current is False
    assert results.winner_count == 1


@pytest.mark.integration
def test_run_scoped_snapshot_recomputes_config_drift_against_current_catalog(
    tmp_path: Path,
) -> None:
    run_id, trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    _write_experiment_config(
        config,
        _drift_experiment(tmp_path, trainer, metric_name="y", goal="maximize"),
    )
    app, _registry, _store = make_mcp_app(catalog)

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.result_source == "frozen_run_snapshot"
    assert results.result_source == "frozen_run_snapshot"
    assert status.published_config_matches_current is False
    assert results.published_config_matches_current is False
    assert (results.metric.name, results.metric.goal) == ("x", "minimize")


@pytest.mark.integration
def test_run_scoped_snapshot_survives_later_artifact_corruption(tmp_path: Path) -> None:
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, _store = make_mcp_app(catalog)
    experiment = app._registry.get("srv").experiment
    winner_path = _generation_winner_path(experiment, run_id, "p")
    winner_path.write_text("broken: true\n")

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.publication_integrity == "ok"
    assert status.is_published is True
    assert status.phases[0].winner_present is True
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1


@pytest.mark.integration
def test_run_scoped_snapshot_keeps_captured_generation_pointers(tmp_path: Path) -> None:
    run_id, _trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    later_failed_id = "srv-later-failed"
    _write_generation_state(
        experiment,
        generation_id=later_failed_id,
        state="failed",
        from_phase=None,
        publish_current=True,
        error_class="RuntimeError",
    )
    app, _registry, _store = make_mcp_app(catalog)

    after_failure = GetRunStatusResult.model_validate(app.status(run_id=run_id))

    assert after_failure.current_generation_id == run_id
    assert after_failure.published_generation_id == run_id
    assert after_failure.represented_generation_id == run_id
    assert after_failure.is_published is True

    later_published_id = "srv-later-published"
    run_experiment(experiment, generation_id=later_published_id)

    after_publication = GetRunStatusResult.model_validate(app.status(run_id=run_id))

    assert after_publication.current_generation_id == run_id
    assert after_publication.published_generation_id == run_id
    assert after_publication.represented_generation_id == run_id
    assert after_publication.is_published is True


@pytest.mark.integration
def test_run_scoped_snapshot_survives_publication_pointer_removal(
    tmp_path: Path,
) -> None:
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, _store = make_mcp_app(catalog)
    experiment = app._registry.get("srv").experiment
    _last_successful_generation_path(experiment).unlink()

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert status.publication_integrity == "ok"
    assert status.is_published is True
    assert status.phases[0].winner_present is True
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1


@pytest.mark.parametrize("config_snapshot_damage", ["missing", "corrupt"])
@pytest.mark.integration
def test_run_scoped_snapshot_survives_artifact_tree_relocation(
    tmp_path: Path,
    config_snapshot_damage: str,
) -> None:
    """Completed run reads use frozen evidence without mutable sibling artifacts."""
    run_id, _trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    config_snapshot = store.config_snapshot_path(run_id)
    if config_snapshot_damage == "missing":
        config_snapshot.unlink()
    else:
        config_snapshot.write_text("not: a valid experiment snapshot\n")
    source = _experiment_dir(experiment)
    relocated = tmp_path / "relocated" / source.name
    relocated.parent.mkdir()
    shutil.move(source, relocated)

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))
    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))
    awaited = AwaitRunResult.model_validate(asyncio.run(app.await_run(run_id)))

    for payload in (status, awaited):
        assert payload.result_source == "frozen_run_snapshot"
        assert payload.publication_integrity == "ok"
        assert payload.phases[0].winner_present is True
    assert results.result_source == "frozen_run_snapshot"
    assert results.publication_integrity == "ok"
    assert results.winner_count == 1


@pytest.mark.parametrize("read_tool", ["status", "winners", "await_run"])
@pytest.mark.integration
def test_run_scoped_read_keeps_complete_snapshot_under_cleanup_reservation(
    tmp_path: Path,
    read_tool: str,
) -> None:
    """Cleanup uncertainty reserves the run without erasing its frozen result."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    store.config_snapshot_path(run_id).unlink()
    handle = store.get(run_id)
    assert handle is not None
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    write_run_status(store, **{**terminal, "cleanup_confirmed": False})
    store.mark_cleanup_uncertain(handle)

    payload: GetRunResultsResult | GetRunStatusResult | AwaitRunResult
    if read_tool == "winners":
        results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))
        assert results.winner_count == 1
        payload = results
    else:
        payload = (
            AwaitRunResult.model_validate(asyncio.run(app.await_run(run_id)))
            if read_tool == "await_run"
            else GetRunStatusResult.model_validate(app.status(run_id=run_id))
        )
        assert payload.phases[0].winner_present is True
        assert payload.run is not None
        assert payload.run.state == "running"
        assert payload.run.recovery_required is True
    assert payload.result_source == "frozen_run_snapshot"
    assert payload.publication_integrity == "ok"
    assert payload.represented_generation_id == run_id


@pytest.mark.parametrize("read_tool", ["status", "await_run"])
@pytest.mark.integration
def test_run_scoped_status_refreshes_state_when_snapshot_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_tool: str
) -> None:
    """A pending-to-complete write cannot pair terminal facts with running state."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    handle = store.get(run_id)
    assert handle is not None
    complete = store.recorded_terminal_status(handle)
    assert complete is not None
    write_run_status(store, **{**complete, "result_snapshot_state": "pending"})
    monkeypatch.setattr(store, "_runner_is_live", lambda _handle: True)
    original_run_payload = app._run_payload
    finalized = False

    def finish_after_run_payload(saved: RunHandle) -> dict[str, Any]:
        nonlocal finalized
        run = original_run_payload(saved)
        if not finalized:
            assert run["state"] == "running"
            finalized = True
            write_run_status(store, **complete)
        return run

    monkeypatch.setattr(app, "_run_payload", finish_after_run_payload)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)

    payload = (
        asyncio.run(app.await_run(run_id, timeout_seconds=0))
        if read_tool == "await_run"
        else app.status(run_id=run_id)
    )

    assert finalized
    assert payload["result_source"] == "frozen_run_snapshot"
    assert payload["run"]["state"] == "succeeded"
    assert payload["run"]["recovery_required"] is False
    if read_tool != "status":
        assert payload["reason"] == "terminal"


@pytest.mark.parametrize("read_tool", ["status", "winners", "await_run"])
@pytest.mark.integration
def test_run_scoped_read_rereads_pending_snapshot_after_runner_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_tool: str
) -> None:
    """A runner's final write during its exit probe remains readable."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    handle = store.get(run_id)
    assert handle is not None
    complete = store.recorded_terminal_status(handle)
    assert complete is not None
    write_run_status(store, **{**complete, "result_snapshot_state": "pending"})
    store.config_snapshot_path(run_id).unlink()
    finalized = False

    def finish_before_reporting_exit(_handle: RunHandle) -> bool:
        nonlocal finalized
        assert not finalized
        finalized = True
        write_run_status(store, **complete)
        return False

    monkeypatch.setattr(store, "_runner_is_live", finish_before_reporting_exit)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)

    payload = (
        app.winners(run_id=run_id)
        if read_tool == "winners"
        else (
            asyncio.run(app.await_run(run_id, timeout_seconds=0))
            if read_tool == "await_run"
            else app.status(run_id=run_id)
        )
    )

    assert finalized
    assert payload["result_source"] == "frozen_run_snapshot"
    assert payload["represented_generation_id"] == run_id
    if read_tool == "winners":
        assert payload["winner_count"] == 1
    else:
        assert payload["run"]["state"] == "succeeded"
        if read_tool == "await_run":
            assert payload["reason"] == "terminal"


@pytest.mark.parametrize(
    ("read_tool", "transition"),
    [
        ("status", "complete"),
        ("winners", "complete"),
        ("await_run", "complete"),
        ("status", "pending_runner_exit"),
        ("winners", "pending_runner_exit"),
        ("await_run", "pending_runner_exit"),
        ("status", "no_status_runner_exit"),
        ("winners", "no_status_runner_exit"),
        ("status", "no_status_runner_exit_with_recovery_lock"),
        ("winners", "no_status_runner_exit_with_recovery_lock"),
        ("await_run", "no_status_runner_exit_with_recovery_lock"),
        ("await_run", "no_status_runner_exit"),
    ],
)
@pytest.mark.integration
def test_run_scoped_live_read_uses_snapshot_completed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_tool: str, transition: str
) -> None:
    """A live read follows final snapshot or recovery state recorded during it."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    handle = store.get(run_id)
    assert handle is not None
    complete = store.recorded_terminal_status(handle)
    assert complete is not None
    if transition.startswith("no_status_runner_exit"):
        store.status_path(run_id).unlink()
    else:
        write_run_status(store, **{**complete, "result_snapshot_state": "pending"})
    runner_live = True
    monkeypatch.setattr(store, "_runner_is_live", lambda _handle: runner_live)
    monkeypatch.setattr(
        "phasesweep.mcp.runs.is_same_live_process", lambda _pid, _starttime: runner_live
    )
    original_live = app._live_status_payload
    finalized = False

    def finish_during_live_read(
        experiment_id: str, experiment: Experiment, saved: RunHandle | None
    ) -> dict[str, Any]:
        nonlocal finalized, runner_live
        status = original_live(experiment_id, experiment, saved)
        if not finalized:
            finalized = True
            if transition == "complete":
                write_run_status(store, **complete)
            else:
                runner_live = False
        return status

    monkeypatch.setattr(app, "_live_status_payload", finish_during_live_read)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)

    lock = (
        store.transition_lock(handle)
        if transition == "no_status_runner_exit_with_recovery_lock"
        else contextlib.nullcontext()
    )
    with lock:
        payload = (
            app.winners(run_id=run_id)
            if read_tool == "winners"
            else (
                asyncio.run(app.await_run(run_id, timeout_seconds=0))
                if read_tool == "await_run"
                else app.status(run_id=run_id)
            )
        )
    if transition == "no_status_runner_exit_with_recovery_lock":
        assert not store.cleanup_uncertain(handle)
        assert store.state(handle) == "running"

    assert finalized
    if transition == "complete":
        assert payload["result_source"] == "frozen_run_snapshot"
        assert payload["represented_generation_id"] == run_id
        if read_tool == "winners":
            assert payload["winner_count"] == 1
        else:
            assert payload["run"]["state"] == "succeeded"
            if read_tool == "await_run":
                assert payload["reason"] == "terminal"
    else:
        assert payload["result_source"] == "terminal_snapshot_unavailable"
        assert payload["publication_integrity"] == "unknown"
        assert payload["represented_generation_id"] is None
        if read_tool == "winners":
            assert payload["winner_count"] == 0
            assert payload["failure"]["code"] == "result_snapshot_unavailable"
        else:
            assert payload["phases"][0]["trial_data_available"] is False
            assert payload["phases"][0]["winner_present"] is False
            assert payload["run"]["recovery_required"] is True
            assert payload["run"]["failure"]["code"] == "result_snapshot_unavailable"
        assert store.recovery_required(handle)
        if read_tool == "await_run":
            assert payload["reason"] == "recovery_required"


@pytest.mark.parametrize("read_tool", ["status", "await_run"])
@pytest.mark.integration
def test_run_scoped_status_refreshes_cleanup_added_after_frozen_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_tool: str
) -> None:
    """A frozen result can coexist with cleanup uncertainty added during the read."""
    run_id, _trainer, _config, catalog = _record_published_run_snapshot(tmp_path)
    app, _registry, store = make_mcp_app(catalog)
    handle = store.get(run_id)
    assert handle is not None
    complete = store.recorded_terminal_status(handle)
    assert complete is not None
    original_run_payload = app._run_payload
    marked = False

    def reserve_after_run_payload(saved: RunHandle) -> dict[str, Any]:
        nonlocal marked
        run = original_run_payload(saved)
        if not marked:
            assert run["state"] == "succeeded"
            marked = True
            write_run_status(store, **{**complete, "cleanup_confirmed": False})
            store.mark_cleanup_uncertain(saved)
        return run

    monkeypatch.setattr(app, "_run_payload", reserve_after_run_payload)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)

    payload = (
        asyncio.run(app.await_run(run_id, timeout_seconds=0))
        if read_tool == "await_run"
        else app.status(run_id=run_id)
    )

    assert marked
    assert payload["result_source"] == "frozen_run_snapshot"
    assert payload["run"]["state"] == "running"
    assert payload["run"]["recovery_required"] is True
    if read_tool == "await_run":
        assert payload["reason"] == "recovery_required"


@pytest.mark.integration
def test_published_results_keep_their_objective_evidence_after_an_extractor_swap(
    tmp_path: Path,
) -> None:
    """A log-scraped number must not inherit a structured extractor's guarantees.

    Reporting the current extractor's assurance beside a historical winner
    claims evidence properties that run never had - the one field an agent is
    told to use when deciding how far to trust a metric.
    """
    run_id, trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    published_assurance = objective_evidence_assurance(
        _drift_experiment(tmp_path, trainer).metric.extractor
    )

    _write_experiment_config(
        config,
        _drift_experiment(
            tmp_path,
            trainer,
            extractor=JsonEnvelopeExtractor(
                type="json_envelope",
                path="r.json",
                objective_name="x",
                split="test",
                policy="test",
            ),
        ),
    )
    app, _registry, _store = make_mcp_app(catalog)

    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))
    evidence = results.metric.objective_evidence.model_dump()

    assert evidence == published_assurance
    assert evidence["kind"] == "log_regex"
    # The four guarantees the swapped-in extractor would have asserted.
    assert evidence["objective_name_bound"] is False
    assert evidence["split_bound"] is False
    assert evidence["evaluation_policy_bound"] is False
    assert evidence["source_identity_keyed"] is False
    assert results.published_config_matches_current is False


@pytest.mark.integration
def test_published_winner_survives_a_catalog_phase_rename(tmp_path: Path) -> None:
    """Renaming a phase must not delete the published result from the payload.

    Enumerating winners under the *current* phase names returned an "ok"
    publication with zero winners and the new name listed as missing: an
    agent's cue to launch a run over evidence that was there all along.
    """
    run_id, trainer, config, catalog = _record_published_run_snapshot(tmp_path)
    _write_experiment_config(config, _drift_experiment(tmp_path, trainer, phase_name="q"))
    app, _registry, _store = make_mcp_app(catalog)

    results = GetRunResultsResult.model_validate(app.winners(run_id=run_id))

    assert results.publication_integrity == "ok"
    assert [phase.phase for phase in results.phases] == ["p"]
    assert results.winner_count == 1
    assert results.declared_phase_count == 1
    assert results.missing_phases == []
    assert results.all_phases_have_winners is True
    assert results.published_config_matches_current is False

    status = GetRunStatusResult.model_validate(app.status(run_id=run_id))

    # Run-specific status uses the phase plan captured when the run published.
    assert status.is_published is True
    assert [phase.phase for phase in status.phases] == ["p"]
    assert status.phases[0].winner_present is True
    assert status.result_phase_plan == ["p"]
    assert status.published_config_matches_current is False


@pytest.mark.parametrize(
    ("launch_policy", "current_policy", "visible"),
    [
        pytest.param("none", "all", False, id="later-loosening-cannot-reveal"),
        pytest.param("all", "none", False, id="later-tightening-redacts"),
        pytest.param(["lr"], "all", True, id="launch-allowlist-remains-visible"),
        pytest.param("all", ["lr"], True, id="current-allowlist-restricts"),
    ],
)
@pytest.mark.integration
def test_run_winner_visibility_intersects_launch_and_restarted_catalog_policy(
    tmp_path: Path,
    launch_policy: object,
    current_policy: object,
    visible: bool,
) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = _drift_experiment(tmp_path, trainer)
    config = tmp_path / "srv.yaml"
    _write_experiment_config(config, experiment)
    catalog = _catalog(tmp_path, config, visible_params=launch_policy)
    launch_registry = Registry.load(catalog)
    store = RunStore(launch_registry.state_dir)
    reg = launch_registry.get("srv")
    run_id = "srv-historical"
    snapshot = config.read_bytes()
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    store.create(
        make_run_handle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            visible_params_at_launch=reg.visible_params,
        )
    )
    run_experiment(experiment, generation_id=run_id)
    (published,) = read_winners(experiment)
    expected = published.params["lr"] if visible else "<redacted>"

    _catalog(tmp_path, config, visible_params=current_policy)
    restarted = PhaseSweepMCP(Registry.load(catalog), store)

    assert restarted.winners(run_id=run_id)["phases"][0]["params"]["lr"] == expected


def test_list_experiments_pages_catalog(tmp_path: Path) -> None:
    configs = {f"srv{i}": _config(tmp_path, name=f"srv{i}") for i in range(3)}
    registry = Registry.load(write_mcp_catalog(tmp_path, configs))
    store = RunStore(registry.state_dir)
    app = PhaseSweepMCP(registry, store)

    first = app.list_experiments(limit=2)
    assert [item["id"] for item in first["experiments"]] == ["srv0", "srv1"]
    assert first["total_count"] == 3
    assert first["next_cursor"] == "2"

    second = app.list_experiments(limit=2, cursor=first["next_cursor"])
    assert [item["id"] for item in second["experiments"]] == ["srv2"]
    assert second["total_count"] == 3
    assert second["next_cursor"] is None

    with pytest.raises(McpToolError, match="invalid cursor"):
        app.list_experiments(cursor="not-a-cursor")
    with pytest.raises(McpToolError, match="limit must be between"):
        app.list_experiments(limit=0)


def test_validate_rejects_config_changed_after_startup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))
    config.write_text(config.read_text() + "\n# changed after server startup\n")

    with pytest.raises(ConfigChangedError, match="restart the MCP server"):
        app.validate("srv")


def test_latest_run_returns_one_computed_reattachment_handle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config))

    assert app.latest_run("srv") == {
        "experiment_id": "srv",
        "found": False,
        "run": None,
    }

    handle = make_run_handle(
        run_id="srv-current",
        experiment_id=registry.get("srv").id,
        config_sha256=registry.get("srv").config_sha256,
    )
    store.create(handle)

    result = app.latest_run("srv")
    assert result["found"] is True
    assert result["run"] == {
        "run_id": handle.run_id,
        "state": "running",
        "started_at": handle.started_at,
        "recovery_required": False,
        "failure": None,
    }


def test_read_tools_use_live_view_while_result_snapshot_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    run_id = "srv-live-pending-finalization"
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config.read_bytes())
    write_run_status(
        store,
        run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="pending",
    )

    status = app.status(run_id=run_id)
    winners = app.winners(run_id=run_id)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)
    awaited = asyncio.run(app.await_run(run_id, timeout_seconds=0))

    for payload in (status, awaited):
        assert payload["result_source"] == "current_shared_study"
        assert payload["run"]["state"] == "running"
        assert payload["run"]["recovery_required"] is False
    assert winners["result_source"] == "current_shared_study"
    assert winners["represented_generation_id"] == run_id
    assert awaited["reason"] == "timeout"
