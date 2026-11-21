"""recover-run and MLflow delivery: the outcomes of the trials it settles, sent once or left pending with their step, and the tracking workers whose cleanup it confirms."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _retire_active_attempt
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import CLEANUP_CONFIRMED_ATTR
from phasesweep.errors import ProcessCleanupUncertainError
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import OperationUnavailableError
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
_PENDING_STEP = "`phasesweep sync-tracking CONFIG`"


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
    requests = _logged_requests(monkeypatch, down=False)
    root = tracking_root(run.experiment)
    before = sorted(root.rglob("*"))

    preview = run.recover(confirm=False)

    assert preview.exit_code == 0, preview.output
    assert (requests, sorted(root.rglob("*"))) == ([], before)

    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    sends = 2 if source == "running" else 1
    assert requests == ["resolve"] + ["publish"] * sends
    assert (run.sent(), run.summaries()) == ([finish], [finish])
    assert (
        f"Delivered {sends} MLflow outcome(s) of the recovered trials and generations; none stay "
        "pending." in confirmed.output
    )
    assert _PENDING_STEP not in confirmed.output
    assert not run.store.observe(run.handle).recovery_required
    delivery = run.app.status(_RUN_ID)["tracking_delivery"]
    assert (delivery["pending"], delivery["remedy"]) == (0, None)
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
    requests = _logged_requests(monkeypatch, down=False)
    root = tracking_root(run.experiment)
    before = sorted(root.rglob("*"))

    preview = run.recover(confirm=False)

    assert preview.exit_code == 0, preview.output
    assert (requests, sorted(root.rglob("*"))) == ([], before)

    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    assert requests == ["resolve", "publish"]
    assert (run.sent(), run.summaries()) == ([], ["KILLED"])
    assert (
        "Delivered 1 MLflow outcome(s) of the recovered trials and generations; none stay "
        "pending." in confirmed.output
    )
    assert not run.store.observe(run.handle).recovery_required
    delivery = run.app.status(_RUN_ID)["tracking_delivery"]
    assert (delivery["pending"], delivery["remedy"]) == (0, None)
    publication.sync_tracking(run.experiment)
    assert (run.sent(), run.summaries()) == ([], ["KILLED"])


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
    assert (
        "Delivered 0 MLflow outcome(s) of the recovered trials and generations; 2 stay pending."
        in confirmed.output
    )
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
    ) -> bool:
        if path.stem == "generation":
            raise RunnerKilled()
        return deliver(experiment, path, environment, **kwargs)

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
@pytest.mark.parametrize(
    ("earlier", "action"),
    [("killed", "publish"), ("uncertain", "publish"), ("uncertain", "resolve")],
    ids=["upload-after-success", "upload-after-failure", "resolve-after-failure"],
)
def test_the_printed_steps_clear_a_run_whose_worker_sent_an_earlier_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, earlier: str, action: str
) -> None:
    """A tracking worker a run starts for an earlier generation's recovered outcome is that run's to recover.

    The run recovers an earlier sweep's attempt, then succeeds or fails, leaving the worker that
    resolves the target for that outcome, or uploads it, with its cleanup unconfirmed. Following
    the steps the run's MCP status prints, sync-tracking where it names one and then a confirmed
    recover-run, clears the run, and the outcome reaches its MLflow run once.
    """
    if earlier == "killed":
        experiment, uploads, [attempt] = killed_sweep(tmp_path, monkeypatch, n_jobs=1)
        finish = "KILLED"
    else:
        experiment, uploads, attempt = _uncertain_attempt(tmp_path, monkeypatch)
        finish = "FAILED"
    # The runner resolves the config's relative paths against its directory, and the earlier
    # sweep ran from the working directory.
    config = write_experiment_config(Path.cwd() / "experiment.yaml", experiment)
    app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    leaked = leak_tracking_worker(
        monkeypatch, action=action, name=f"attempt-{attempt}" if action == "publish" else None
    )
    if earlier == "killed":
        assert run_claimed_runner(store, _SENDING_RUN_ID, config, cwd=Path.cwd()) == 0
    else:
        # The earlier sweep's unsafe-cleanup abort stays recorded, so this run fails.
        with pytest.raises(ProcessCleanupUncertainError, match="worker cleanup is uncertain"):
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
    assert [r["finish"] for r in uploads if r["name"] == f"attempt-{attempt}"] == [finish]
