"""recover-run and MLflow delivery: the outcomes of the trials it settles, sent once or left pending with their step, and the tracking workers whose cleanup it confirms."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import Result

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _retire_active_attempt
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import CLEANUP_CONFIRMED_ATTR, TRIAL_DIR_ATTR
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.json import JsonObject
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import OperationRejectedError, OperationUnavailableError
from phasesweep.tracking.session import read_record, tracking_root
from tests.conftest import write_experiment_config
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    make_mcp_app,
    run_claimed_runner,
    stage_dead_run,
    write_srv_catalog,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import recover_run_cli
from tests.tracking_helpers import (
    RunnerKilled,
    killed_sweep,
    leak_tracking_worker,
    leave_cleanup_uncertain_attempt,
    remote_mlflow_example,
)

_RUN_ID = "mcp-recovered-run"
_SENDING_RUN_ID = "mcp-sending-run"
_PENDING_STEP = "`phasesweep sync-tracking"


class _RecoveredRun:
    """An MCP run whose runner died leaving one managed attempt for recover-run to settle."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        experiment: Experiment,
        uploads: list[dict[str, object]],
        attempt: str,
        *,
        wrote_status: bool = True,
        marked: bool = True,
        snapshot: bool = False,
    ) -> None:
        """Stage the run, with the unsafe-cleanup status its runner wrote unless ``wrote_status`` is False, holding an unfinalized result snapshot when ``snapshot`` is True, and the cleanup marker a server writes unless ``marked`` is False."""
        config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
        app, registry, store = make_mcp_app(
            write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
        )
        self.experiment = experiment
        self.uploads = uploads
        self.attempt = attempt
        self.app: PhaseSweepMCP = app
        self.store: RunStore = store
        self.state_dir = registry.state_dir
        self.handle: RunHandle = stage_dead_run(
            store, _RUN_ID, config, registry.get("srv").id, cleanup_uncertain=marked
        )
        if wrote_status:
            frozen = (
                capture_result_snapshot(experiment, generation_id=_RUN_ID) if snapshot else None
            )
            write_unsafe_cleanup_status(
                store, _RUN_ID, **({} if frozen is None else {"result_snapshot": frozen})
            )
        monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *a, **k: True)

    def recover(self, *, confirm: bool = True) -> Result:
        """Run recover-run on the run, confirmed unless ``confirm`` is False."""
        return recover_run_cli(self.state_dir, _RUN_ID, confirm=confirm)

    @property
    def record(self) -> Path:
        """Return the attempt's delivery record."""
        return (
            tracking_root(self.experiment) / _RUN_ID / "delivery" / f"attempt-{self.attempt}.json"
        )

    def sent(self) -> list[object]:
        """Return the finish status of each upload of the attempt so far."""
        return [r["finish"] for r in self.uploads if r["name"] == f"attempt-{self.attempt}"]

    def summaries(self) -> list[object]:
        """Return the finish status of each upload of the run's summary run so far."""
        return [r["finish"] for r in self.uploads if r["name"] == "generation"]


def _killed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    in_flight: bool = True,
    wrote_status: bool | None = None,
    marked: bool = True,
    snapshot: bool = False,
) -> _RecoveredRun:
    """Stage an MCP run killed with its one managed attempt still RUNNING.

    :param bool in_flight: ``False`` lets the attempt finish and kills the runner before it
        publishes, writing no status, as a hard kill between trials leaves it.
    :param bool | None wrote_status: Whether the runner wrote its unsafe-cleanup status;
        ``None`` writes it only for a runner killed in flight.
    :param bool marked: Whether a server watching the run wrote its cleanup marker.
    :param bool snapshot: Whether that status holds a result snapshot recovery must finalize.
    :return _RecoveredRun: The staged run.
    """
    experiment, uploads, [attempt] = killed_sweep(
        tmp_path, monkeypatch, n_jobs=1, generation_id=_RUN_ID, in_flight=in_flight
    )
    return _RecoveredRun(
        tmp_path,
        monkeypatch,
        experiment,
        uploads,
        attempt,
        wrote_status=in_flight if wrote_status is None else wrote_status,
        marked=marked,
        snapshot=snapshot,
    )


# How a killed runner can leave its run for recover-run: what it wrote and whether a server
# watching it wrote the cleanup marker.
_STAGINGS = {
    "unsafe-status": {"in_flight": True, "wrote_status": True, "marked": True},
    "unsafe-status-no-marker": {"in_flight": True, "wrote_status": True, "marked": False},
    "unsafe-status-with-snapshot": {
        "in_flight": True,
        "wrote_status": True,
        "marked": False,
        "snapshot": True,
    },
    "no-status-no-marker": {"in_flight": True, "wrote_status": False, "marked": False},
    "between-trials": {"in_flight": False, "wrote_status": False, "marked": True},
}


def _uncertain_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, generation_id: str | None = None
) -> tuple[Experiment, list[dict[str, object]], str]:
    """Fail the one managed attempt of a one-trial sweep with its cleanup unconfirmed.

    :param str | None generation_id: The failed run's generation, as an MCP runner names it.
    :return tuple[Experiment, list[dict[str, object]], str]: The experiment, the upload log,
        and the attempt id.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, sampler={"type": "random", "seed": 123})
    experiment = Experiment.model_validate(config)
    attempt = leave_cleanup_uncertain_attempt(experiment, monkeypatch, generation_id=generation_id)
    return experiment, uploads, attempt


def _uncertain_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, retired: bool
) -> _RecoveredRun:
    """Stage an MCP run whose one managed attempt FAILED with its cleanup unconfirmed.

    :param bool retired: Drop its registry entry, so only the trial records the uncertainty.
    :return _RecoveredRun: The staged run.
    """
    experiment, uploads, attempt = _uncertain_attempt(tmp_path, monkeypatch, generation_id=_RUN_ID)
    if retired:
        _retire_active_attempt(experiment, attempt)
    return _RecoveredRun(tmp_path, monkeypatch, experiment, uploads, attempt)


def _logged_requests(monkeypatch: pytest.MonkeyPatch, *, down: bool) -> list[str]:
    """Record each MLflow request's action; a tracker that is down refuses every one."""
    respond = session.request_operation
    requests: list[str] = []

    def logged(request: dict[str, object], **kwargs: object) -> object:
        requests.append(str(request["action"]))
        if down:
            raise OperationUnavailableError("tracking server unreachable")
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", logged)
    monkeypatch.setattr(worker, "request_operation", logged)
    return requests


def _recover_after_a_silent_preview(
    run: _RecoveredRun, monkeypatch: pytest.MonkeyPatch, *, delivered: str
) -> list[str]:
    """Preview recover-run, then confirm it, and check what the confirmation leaves behind.

    A preview writes and sends nothing. The confirmed run delivers ``delivered`` and leaves
    nothing pending, no recovery required and no delivery remedy.

    :param str delivered: The record count the confirmed run reports, such as ``1 record``.
    :return list[str]: The MLflow request actions of the confirmed run, in order.
    """
    requests = _logged_requests(monkeypatch, down=False)
    root = tracking_root(run.experiment)
    before = sorted(root.rglob("*"))

    preview = run.recover(confirm=False)

    assert preview.exit_code == 0, preview.output
    assert (requests, sorted(root.rglob("*"))) == ([], before)

    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    assert f"MLflow: delivered {delivered}; none stay pending." in confirmed.output
    assert _PENDING_STEP not in confirmed.output
    assert not run.store.observe(run.handle).recovery_required
    delivery = run.app.status(_RUN_ID)["tracking_delivery"]
    assert (delivery["pending"], delivery["remedy"]) == (0, None)
    return requests


@pytest.mark.integration
@pytest.mark.parametrize("source", ["running", "uncertain-registered", "uncertain-retired"])
def test_recover_run_finishes_the_mlflow_run_of_each_trial_it_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A confirmed recover-run sends each settled attempt's outcome once, after one target check, and counts it.

    A RUNNING attempt ends KILLED, and so does the summary run of the run killed with it. A
    failed attempt whose cleanup recovery confirms ends FAILED, whether its registry entry or
    only its trial is left to confirm it. That confirmation lives in the study's cleanup ledger,
    never in a rewritten trial, and nothing of the attempt is sent before it; sync-tracking
    can finish the failed run's summary run FAILED before. A preview writes and sends nothing.
    """
    if source == "running":
        run = _killed_run(tmp_path, monkeypatch)
        finish = "KILLED"
        run.uploads.clear()
    else:
        run = _uncertain_run(tmp_path, monkeypatch, retired=source == "uncertain-retired")
        finish = "FAILED"
        run.uploads.clear()
        assert publication.sync_tracking(run.experiment, _RUN_ID) == {"pending": 0, "delivered": 1}
        assert (run.sent(), run.summaries()) == ([], [finish])
    sends = 2 if source == "running" else 1

    requests = _recover_after_a_silent_preview(
        run, monkeypatch, delivered="1 record" if sends == 1 else f"{sends} records"
    )

    assert requests == ["resolve"] + ["publish"] * sends
    assert (run.sent(), run.summaries()) == ([finish], [finish])
    read = read_phase_trials(validate_ledger(run.experiment), "depth")
    [trial] = read.trials
    assert read.cleanup_recovered == {trial.number}
    if source != "running":
        assert trial.user_attrs[CLEANUP_CONFIRMED_ATTR] is False
    journal = tmp_path / "study.journal"
    before = journal.read_bytes()
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == ([finish], [finish])
    assert journal.read_bytes() == before


@pytest.mark.integration
def test_recover_run_ends_the_summary_run_of_a_run_killed_between_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmed recover-run ends the summary run of a run killed with no trial left to settle.

    The runner is proven dead, so its summary run ends KILLED once and is counted, while the
    attempt's outcome, sent before the kill, is not sent again. A preview writes and sends
    nothing, and a later sync-tracking sends nothing more.
    """
    run = _killed_run(tmp_path, monkeypatch, in_flight=False)
    run.uploads.clear()

    requests = _recover_after_a_silent_preview(run, monkeypatch, delivered="1 record")

    assert requests == ["resolve", "publish"]
    assert (run.sent(), run.summaries()) == ([], ["KILLED"])
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == ([], ["KILLED"])


@pytest.mark.integration
def test_recover_run_distinguishes_a_creation_miss_from_an_upload_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answering tracker can have no run for a recently interrupted creation."""
    run = _killed_run(tmp_path, monkeypatch)
    [trial] = read_phase_trials(validate_ledger(run.experiment), "depth").trials
    bound = Path(trial.user_attrs[TRIAL_DIR_ATTR]) / session.BINDING_FILE
    identity = session.read_binding(bound.parent)
    pending = bound.with_suffix(".pending.json")
    session.write_record(
        pending,
        {
            "schema_version": 1,
            "identity": {k: v for k, v in identity.items() if k != "run_id"},
            "run_name": "trial",
        },
    )
    bound.unlink()
    respond = session.request_operation

    def lookup(request: dict[str, object], **kwargs: object) -> object:
        if request["action"] == "find":
            return {"found": False}
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", lookup)
    result = run.recover()
    assert result.exit_code == 0, result.output
    # The summary record went out; the attempt's creation, inside its window, stays unconfirmed.
    assert (
        "MLflow: delivered 1 record. MLflow: 1 unconfirmed run creation pending. The next launch"
        in result.output
    )
    assert _PENDING_STEP in result.output
    assert pending.is_file() and not bound.exists()
    assert not run.store.observe(run.handle).recovery_required


@pytest.mark.integration
def test_recover_run_leaves_outcomes_pending_while_mlflow_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracker that is down costs one target check and never fails the recovery.

    The outcome stays pending, recover-run counts it and names the sync-tracking step, the
    run's MCP status carries the same step, and sync-tracking sends it once.
    """
    run = _killed_run(tmp_path, monkeypatch)
    respond = session.request_operation
    requests = _logged_requests(monkeypatch, down=True)
    run.uploads.clear()

    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    assert requests == ["resolve"]
    assert run.sent() == []
    assert read_record(run.record)["state"] == "pending"
    assert "MLflow: 2 records pending. The next launch" in confirmed.output
    assert _PENDING_STEP in confirmed.output
    assert not run.store.observe(run.handle).recovery_required
    delivery = run.app.status(_RUN_ID)["tracking_delivery"]
    assert delivery["pending"] == 2
    assert f"`phasesweep sync-tracking CONFIG --generation {_RUN_ID}`" in delivery["remedy"]

    monkeypatch.setattr(session, "request_operation", respond)
    monkeypatch.setattr(worker, "request_operation", respond)
    publication.sync_tracking(run.experiment)
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == (["KILLED"], ["KILLED"])


@pytest.mark.integration
def test_recover_run_and_mcp_status_say_what_keeps_a_refused_record_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record MLflow refuses during recover-run stays pending, and recover-run and MCP status say what clears it.

    MLflow refuses the attempt's outcome, as for a run deleted in MLflow, and accepts the
    summary run's end. recover-run's line counts both and names the refusal; the run's MCP
    remedy places the refusal's warning, as an MCP reader never sees the console.
    """
    run = _killed_run(tmp_path, monkeypatch)
    respond = session.request_operation

    def deleted(request: dict[str, object], **kwargs: object) -> JsonObject:
        if request["action"] == "publish" and request["name"] == f"attempt-{run.attempt}":
            raise OperationRejectedError("MLflow run is deleted; restore it, or its experiment.")
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", deleted)
    monkeypatch.setattr(worker, "request_operation", deleted)
    run.uploads.clear()

    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    assert (run.sent(), run.summaries()) == ([], ["KILLED"])
    assert read_record(run.record)["state"] == "pending"
    assert (
        "MLflow: delivered 1 record. MLflow: 1 record pending. The next launch with this config, "
        "or `phasesweep sync-tracking` with this run's config file and `--set` overrides, sends "
        "them; delivery never retrains. MLflow refused 1 record; a refused record stays pending "
        "until the cause given in its warning is fixed in MLflow, for example by restoring a "
        "deleted run or its experiment." in confirmed.output.splitlines()
    )
    assert not run.store.observe(run.handle).recovery_required
    delivery = run.app.status(_RUN_ID)["tracking_delivery"]
    assert delivery["remedy"] == (
        "1 MLflow record(s) of this run are pending. Its scores and winners are unaffected. The "
        "next launch with this config sends them, or ask the operator to retry them with "
        f"`phasesweep sync-tracking CONFIG --generation {_RUN_ID}` using the config this run "
        "launched with; delivery never retrains. If MLflow refused a record, such as for a run "
        "deleted in MLflow, that record stays pending until the cause given in its warning in "
        "the engine's `run.log` in the experiment workdir, or on the console of the operator's "
        "retry, is fixed in MLflow, for example by restoring a deleted run or its experiment."
    )


@pytest.mark.integration
@pytest.mark.parametrize("staging", list(_STAGINGS))
def test_an_upload_worker_recover_run_cannot_confirm_leaves_the_run_for_another_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staging: str
) -> None:
    """An upload worker whose cleanup is unconfirmed fails recover-run as an evidence worker's does.

    However the runner left the run, with its unsafe-cleanup status or none, marked by a server
    or not, the run stays recovery-required and no status reports its cleanup confirmed; the
    next confirmed pass resolves the worker through its registry entry, clears the run, and
    sends the outcomes the failed pass left pending, each once.
    """
    run = _killed_run(tmp_path, monkeypatch, **_STAGINGS[staging])
    in_flight, wrote_status = _STAGINGS[staging]["in_flight"], _STAGINGS[staging]["wrote_status"]
    # Killed between trials, the attempt's outcome went out before the kill.
    record = run.record if in_flight else run.record.with_name("generation.json")
    leaked = leak_tracking_worker(monkeypatch, name=record.stem)

    failed = run.recover()

    [operation] = leaked
    assert failed.exit_code != 0
    assert "worker cleanup is uncertain" in failed.output
    assert run.store.observe(run.handle).recovery_required
    status = run.store.recorded_terminal_status(run.handle)
    # recover-run writes the status of a runner that left none only after its uploads.
    assert status["cleanup_confirmed"] is False if wrote_status else status is None
    if _STAGINGS[staging].get("snapshot"):
        # Its snapshot was finalized before the uploads, keeping the runner's unconfirmed cleanup.
        assert status["result_snapshot_state"] == "complete"
    assert read_record(record)["state"] == "pending"
    entry = _attempts_dir(run.experiment) / f"{operation}.json"
    assert entry.is_file()
    run.uploads.clear()

    retried = run.recover()

    assert retried.exit_code == 0, retried.output
    assert not entry.exists()
    assert not run.store.observe(run.handle).recovery_required
    assert not run.store.cleanup_uncertain(run.handle)
    sent = (["KILLED"] if in_flight else [], ["KILLED"])
    assert (run.sent(), run.summaries()) == sent
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == sent


@pytest.mark.integration
@pytest.mark.parametrize(
    ("staging", "killed_at"),
    [
        ("no-status-no-marker", "after-cleanup-record"),
        ("no-status-no-marker", "between-uploads"),
        ("no-status-no-marker", "before-status"),
        ("unsafe-status-no-marker", "after-cleanup-record"),
        ("unsafe-status-no-marker", "between-uploads"),
    ],
)
def test_a_recover_run_killed_before_its_uploads_end_is_settled_by_one_more_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staging: str, killed_at: str
) -> None:
    """recover-run killed after recording its cleanup and before its uploads end leaves the run recovery-required.

    It dies after its cleanup record, after sending the attempt's outcome and before the summary
    run's, or before writing the status of a runner that left none. One more confirmed pass
    clears the run, and each outcome reaches its MLflow run once.
    """
    run = _killed_run(tmp_path, monkeypatch, **_STAGINGS[staging])
    run.uploads.clear()
    deliver = publication._deliver

    def dies(*args: object, **kwargs: object) -> None:
        raise RunnerKilled()

    def dies_before_summary(
        experiment: Experiment, path: Path, environment: dict[str, str], **kwargs: object
    ) -> None:
        if path.stem == "generation":
            raise RunnerKilled()
        deliver(experiment, path, environment, **kwargs)

    with monkeypatch.context() as patch:
        if killed_at == "after-cleanup-record":
            patch.setattr("phasesweep.mcp.recovery._finish_result_recovery", dies)
        elif killed_at == "between-uploads":
            patch.setattr(publication, "_deliver", dies_before_summary)
        else:
            patch.setattr("phasesweep.mcp.recovery.write_status_file", dies)
        with pytest.raises(RunnerKilled):
            run.recover()

    assert run.store.observe(run.handle).recovery_required
    status = run.store.recorded_terminal_status(run.handle)
    assert status is None or status["cleanup_confirmed"] is False
    sent_before = {
        "after-cleanup-record": ([], []),
        "between-uploads": (["KILLED"], []),
        "before-status": (["KILLED"], ["KILLED"]),
    }
    assert (run.sent(), run.summaries()) == sent_before[killed_at]

    retried = run.recover()

    assert retried.exit_code == 0, retried.output
    assert not run.store.observe(run.handle).recovery_required
    assert not run.store.cleanup_uncertain(run.handle)
    assert (run.sent(), run.summaries()) == (["KILLED"], ["KILLED"])
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == (["KILLED"], ["KILLED"])


@pytest.mark.integration
def test_the_printed_steps_clear_a_run_whose_worker_sent_an_earlier_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracking worker a run starts for an earlier generation's recovered outcome is that run's to recover.

    The run recovers an earlier sweep's attempt and, right after its target check, sends that
    outcome; the worker uploading it is left with its cleanup unconfirmed, so the run stops
    before its first trial. Following the steps the run's MCP status prints, sync-tracking where
    it names one and then a confirmed recover-run, clears the run, and the outcome reaches its
    MLflow run once. A run starts no such worker before its own target check.
    """
    experiment, uploads, [attempt] = killed_sweep(tmp_path, monkeypatch, n_jobs=1)
    # The runner resolves the config's relative paths against its directory, and the earlier
    # sweep ran from the working directory.
    config = write_experiment_config(Path.cwd() / "experiment.yaml", experiment)
    app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    leaked = leak_tracking_worker(monkeypatch, name=f"attempt-{attempt}")
    with pytest.raises(UnsafeProcessCleanupError):
        run_claimed_runner(store, _SENDING_RUN_ID, config, cwd=Path.cwd())
    assert leaked
    failure = app.status(_SENDING_RUN_ID)["run"]["failure"]
    assert failure["code"] == "cleanup_uncertain"
    # The in-process runner recorded this test's own PID, and a launch saves the snapshot the
    # helper does not.
    store.config_snapshot_path(_SENDING_RUN_ID).write_bytes(config.read_bytes())
    monkeypatch.setattr("phasesweep.mcp.recovery.is_same_live_process", lambda *a, **k: False)
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *a, **k: True)

    if _PENDING_STEP in failure["remediation"]:
        publication.sync_tracking(experiment)
    recovered = recover_run_cli(registry.state_dir, _SENDING_RUN_ID, confirm=True)

    assert recovered.exit_code == 0, recovered.output
    handle = store.get(_SENDING_RUN_ID)
    assert handle is not None and not store.observe(handle).recovery_required
    publication.sync_tracking(experiment)
    publication.sync_tracking(experiment)
    assert [r["finish"] for r in uploads if r["name"] == f"attempt-{attempt}"] == ["KILLED"]


@pytest.mark.integration
def test_the_printed_steps_clear_a_recovery_whose_target_check_worker_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker recover-run starts to resolve the target before it sends is that run's to recover.

    recover-run resolves the target itself, as no check in its process covered it, and the
    worker doing so is left with its cleanup unconfirmed. The run stays recovery-required with
    a cleanup_uncertain failure; following the steps recover-run and MCP status print clears
    it, and the outcomes reach their MLflow runs once.
    """
    run = _killed_run(tmp_path, monkeypatch)
    leaked = leak_tracking_worker(monkeypatch, action="resolve")
    run.uploads.clear()

    failed = run.recover()

    [operation] = leaked
    assert failed.exit_code != 0
    assert "fix: run `phasesweep run` again with the same config for a CLI run, or " in (
        failed.output
    )
    assert (run.sent(), run.summaries()) == ([], [])
    assert run.store.observe(run.handle).recovery_required
    failure = run.app.status(_RUN_ID)["run"]["failure"]
    assert failure["code"] == "cleanup_uncertain"
    assert (
        f"`phasesweep mcp recover-run --state-dir STATE_DIR --run-id {_RUN_ID}`"
        in (failure["remediation"])
    )
    entry = _attempts_dir(run.experiment) / f"{operation}.json"
    assert entry.is_file()

    # The MCP step: read the preview, then repeat it with --confirm.
    preview = run.recover(confirm=False)
    assert preview.exit_code == 0, preview.output
    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    assert not entry.exists()
    assert not run.store.observe(run.handle).recovery_required
    assert not run.store.cleanup_uncertain(run.handle)
    assert (run.sent(), run.summaries()) == (["KILLED"], ["KILLED"])
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == (["KILLED"], ["KILLED"])


@pytest.mark.integration
@pytest.mark.parametrize("target", ["published-winner", "recovered-attempt"])
def test_recover_run_cleans_worker_before_finalizing_pending_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A runner killed during mirroring holds its slot until its registered worker is reaped.

    A recovered earlier outcome is still pending at the mirror when MLflow gave no answer to
    the pass after the run's target check.
    """
    experiment, uploads, [attempt] = killed_sweep(
        tmp_path, monkeypatch, n_jobs=1, generation_id=_RUN_ID
    )
    config = write_experiment_config(Path.cwd() / "experiment.yaml", experiment)
    app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    store.config_snapshot_path(_SENDING_RUN_ID).write_bytes(config.read_bytes())
    name = "winner-depth" if target == "published-winner" else f"attempt-{attempt}"
    target_generation = _SENDING_RUN_ID if target == "published-winner" else _RUN_ID
    pending = tracking_root(experiment) / target_generation / "delivery" / f"{name}.json"
    crash_code = 74

    with monkeypatch.context() as crash:
        leak_tracking_worker(crash, name=name)
        request_operation = session.request_operation

        def die_during_upload(
            request: Mapping[str, object],
            *,
            operation_id: str,
            directory: Path,
            environment: Mapping[str, str],
            deadline: float,
        ) -> JsonObject:
            try:
                return request_operation(
                    request,
                    operation_id=operation_id,
                    directory=directory,
                    environment=environment,
                    deadline=deadline,
                )
            except UnsafeProcessCleanupError:
                os._exit(crash_code)

        crash.setattr(session, "request_operation", die_during_upload)
        crash.setattr(worker, "request_operation", die_during_upload)
        if target == "recovered-attempt":
            silent: list[str] = []

            def silent_before_training(
                request: Mapping[str, object], **kwargs: object
            ) -> JsonObject:
                if request["action"] == "publish" and request.get("name") == name and not silent:
                    silent.append(name)
                    raise OperationUnavailableError("tracking server unreachable")
                return die_during_upload(request, **kwargs)

            crash.setattr(session, "request_operation", silent_before_training)
            crash.setattr(worker, "request_operation", silent_before_training)
        child = os.fork()
        if child == 0:
            os._exit(run_claimed_runner(store, _SENDING_RUN_ID, config, cwd=Path.cwd()))
        _pid, wait_status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(wait_status) == crash_code

    store = RunStore.open_existing(registry.state_dir)
    handle = store.get(_SENDING_RUN_ID)
    assert handle is not None
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is True
    assert terminal["result_snapshot_state"] == "pending"
    assert terminal["result_publication_state"] == "committed"
    assert store.observe(handle).recovery_required
    before = store.status_path(_SENDING_RUN_ID).read_bytes()
    journal = tmp_path / "study.journal"
    journal_before = journal.read_bytes()
    worker_generation = tracking_root(experiment) / _SENDING_RUN_ID
    [entry] = _attempts_dir(experiment).glob("*.json")
    assert (worker_generation / "operations" / entry.stem).is_dir()
    receipt = worker_generation / "cleanup-receipts" / entry.name
    assert read_record(pending)["state"] == "pending"

    preview = recover_run_cli(registry.state_dir, _SENDING_RUN_ID)
    assert preview.exit_code == 0, preview.output
    assert "reconcile 1 registered MLflow tracking operation" in preview.output
    assert entry.is_file() and not receipt.exists()
    assert store.status_path(_SENDING_RUN_ID).read_bytes() == before

    with monkeypatch.context() as refusal:
        refusal.setattr(
            "phasesweep.engine.attempts.cleanup_stale_trial_process", lambda *a, **k: False
        )
        failed = recover_run_cli(registry.state_dir, _SENDING_RUN_ID, confirm=True)
    assert failed.exit_code != 0
    assert "process cleanup could not be confirmed" in failed.output
    assert entry.is_file() and not receipt.exists()
    assert store.status_path(_SENDING_RUN_ID).read_bytes() == before
    assert store.observe(handle).recovery_required

    recovered = recover_run_cli(registry.state_dir, _SENDING_RUN_ID, confirm=True)
    assert recovered.exit_code == 0, recovered.output
    assert not entry.exists()
    assert read_record(receipt)["generation_id"] == _SENDING_RUN_ID
    assert store.observe(handle).state == "succeeded"
    assert not store.observe(handle).recovery_required
    assert app.winners(_SENDING_RUN_ID)["winner_count"] == 1
    assert journal.read_bytes() == journal_before
    assert read_record(pending)["state"] == "pending"
    publication.sync_tracking(experiment)
    publication.sync_tracking(experiment)
    assert read_record(pending)["state"] == "delivered"
    assert sum(request["name"] == name for request in uploads) == 1
