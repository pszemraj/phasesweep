"""recover-run's MLflow delivery: the outcomes of the trials it settles, sent once or left pending with their step."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _retire_active_attempt
from phasesweep.engine.errors import TrackingSyncRefusedError
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import CLEANUP_CONFIRMED_ATTR
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import OperationUnavailableError
from phasesweep.tracking.session import read_record, tracking_root
from tests.conftest import write_experiment_config
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    make_mcp_app,
    stage_dead_run,
    write_srv_catalog,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import recover_run_cli
from tests.tracking_helpers import (
    killed_sweep,
    leak_tracking_worker,
    leave_cleanup_uncertain_attempt,
    remote_mlflow_example,
)

_RUN_ID = "mcp-recovered-run"
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
    ) -> None:
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
            store, _RUN_ID, config, registry.get("srv").id, cleanup_uncertain=True
        )
        write_unsafe_cleanup_status(store, _RUN_ID)
        monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *a, **k: True)

    def recover(self) -> Result:
        """Run a confirmed recover-run on the run."""
        return recover_run_cli(self.state_dir, _RUN_ID, confirm=True)

    @property
    def record(self) -> Path:
        """Return the attempt's delivery record."""
        return (
            tracking_root(self.experiment) / _RUN_ID / "delivery" / f"attempt-{self.attempt}.json"
        )

    def sent(self) -> list[object]:
        """Return the finish status of each upload of the attempt so far."""
        return [r["finish"] for r in self.uploads if r["name"] == f"attempt-{self.attempt}"]


def _killed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _RecoveredRun:
    """Stage an MCP run killed with its one managed attempt still RUNNING."""
    experiment, uploads, [attempt] = killed_sweep(
        tmp_path, monkeypatch, n_jobs=1, generation_id=_RUN_ID
    )
    return _RecoveredRun(tmp_path, monkeypatch, experiment, uploads, attempt)


def _uncertain_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, retired: bool
) -> _RecoveredRun:
    """Stage an MCP run whose one managed attempt FAILED with its cleanup unconfirmed.

    :param bool retired: Drop its registry entry, so only the trial records the uncertainty.
    :return _RecoveredRun: The staged run.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    config["phases"][0].update(n_trials=1, sampler={"type": "random", "seed": 123})
    experiment = Experiment.model_validate(config)
    attempt = leave_cleanup_uncertain_attempt(experiment, monkeypatch, generation_id=_RUN_ID)
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

    A RUNNING attempt ends KILLED. A failed attempt whose cleanup recovery confirms ends FAILED,
    whether its registry entry or only its trial is left to confirm it. That confirmation lives
    in the study's cleanup ledger, never in a rewritten trial, and nothing is sent before it.
    """
    if source == "running":
        run = _killed_run(tmp_path, monkeypatch)
        finish = "KILLED"
    else:
        run = _uncertain_run(tmp_path, monkeypatch, retired=source == "uncertain-retired")
        finish = "FAILED"
        run.uploads.clear()
        with pytest.raises(TrackingSyncRefusedError, match="No tracking deliveries exist"):
            publication.sync_tracking(run.experiment, _RUN_ID)
        assert run.sent() == []
    requests = _logged_requests(monkeypatch, down=False)
    run.uploads.clear()

    confirmed = run.recover()

    assert confirmed.exit_code == 0, confirmed.output
    assert requests == ["resolve", "publish"]
    assert run.sent() == [finish]
    assert "Delivered 1 MLflow outcome(s) of the recovered trials; none stay pending." in (
        confirmed.output
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
    assert run.sent() == [finish]
    assert journal.read_bytes() == before


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
    assert "Delivered 0 MLflow outcome(s) of the recovered trials; 1 stay pending." in (
        confirmed.output
    )
    assert _PENDING_STEP in confirmed.output
    assert not run.store.observe(run.handle).recovery_required
    delivery = run.app.status(_RUN_ID)["tracking_delivery"]
    assert delivery["pending"] == 1
    assert f"`phasesweep sync-tracking CONFIG --generation {_RUN_ID}`" in delivery["remedy"]

    monkeypatch.setattr(session, "request_operation", respond)
    monkeypatch.setattr(worker, "request_operation", respond)
    publication.sync_tracking(run.experiment)
    publication.sync_tracking(run.experiment)
    assert run.sent() == ["KILLED"]


@pytest.mark.integration
def test_an_upload_worker_recover_run_cannot_confirm_leaves_the_run_for_another_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upload worker whose cleanup is unconfirmed fails recover-run as an evidence worker's does.

    The run stays recovery-required; the next confirmed pass resolves the worker through its
    registry entry and clears the run without sending again, and the outcome waits for
    sync-tracking.
    """
    run = _killed_run(tmp_path, monkeypatch)
    leaked = leak_tracking_worker(monkeypatch, name=f"attempt-{run.attempt}")

    failed = run.recover()

    [operation] = leaked
    assert failed.exit_code != 0
    assert "worker cleanup is uncertain" in failed.output
    assert run.store.observe(run.handle).recovery_required
    assert read_record(run.record)["state"] == "pending"
    entry = _attempts_dir(run.experiment) / f"{operation}.json"
    assert entry.is_file()
    run.uploads.clear()

    retried = run.recover()

    assert retried.exit_code == 0, retried.output
    assert not entry.exists()
    assert not run.store.observe(run.handle).recovery_required
    assert run.sent() == []
    publication.sync_tracking(run.experiment)
    assert run.sent() == ["KILLED"]
