"""Run closeout distinguishes accepted incomplete results and pending MLflow delivery, and delivers what earlier runs left pending."""

from __future__ import annotations

import logging
import signal
from pathlib import Path
from typing import NamedTuple

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine import run as run_module
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    ShutdownCleanupReport,
    ShutdownLaunchRefusedError,
)
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportBlock
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import OperationRejectedError, OperationUnavailableError
from phasesweep.tracking.session import read_record, tracking_root
from tests.tracking_helpers import RunnerKilled, remote_mlflow_example

_PENDING_STEP = (
    "results are saved locally. The next run or `phasesweep sync-tracking CONFIG` delivers them."
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "failure",
    [OperationRejectedError("RESOURCE_DOES_NOT_EXIST"), OperationUnavailableError("unreachable")],
    ids=["rejected", "no-answer"],
)
def test_pending_mlflow_uploads_get_final_closeout_without_failing_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    """The final warning names pending delivery while the committed result remains successful.

    Each trial's upload fails as it ends, whether MLflow rejects it or gives no answer, and
    neither failure reaches the trial or the run.
    """
    config, _uploads = remote_mlflow_example(tmp_path, monkeypatch)
    experiment = Experiment.model_validate(config)

    def outage(*_args: object, **_kwargs: object) -> JsonObject:
        raise failure

    monkeypatch.setattr(publication, "call_tracking", outage)
    caplog.set_level(logging.INFO)
    reports: list[TerminalReport] = []
    winners = run_experiment(experiment, terminal_callback=reports.append)
    [report] = reports
    assert set(winners) == {"depth", "rate"}
    assert report.primary_error is None
    assert read_status(experiment)["published_generation_id"] == report.generation_id
    assert publication.delivery_status(experiment, report.generation_id) == {
        "pending": 7,
        "delivered": 0,
    }

    records = [record for record in caplog.records if record.name == publication.__name__]
    final = records[-1]
    assert final.levelno == logging.WARNING
    message = final.getMessage()
    assert message == f"MLflow: 7 upload(s) pending; {_PENDING_STEP}"
    assert message in (_experiment_dir(experiment) / "run.log").read_text()


class _Interrupted(NamedTuple):
    """A sweep stopped by a shutdown that left its MLflow outcomes pending."""

    experiment: Experiment
    uploads: list[JsonObject]
    generation: str

    @property
    def summary(self) -> Path:
        """Return the interrupted generation's summary-run delivery record."""
        return tracking_root(self.experiment) / self.generation / "delivery" / "generation.json"

    def summaries(self) -> list[object]:
        """Return the finish status of each upload of the interrupted generation's summary run."""
        finishes: list[object] = []
        for request in self.uploads:
            projection = request["projection"]
            summary = projection.get("summary") if isinstance(projection, dict) else None
            if (
                request["name"] == "generation"
                and isinstance(summary, dict)
                and summary.get("generation_id") == self.generation
            ):
                finishes.append(request["finish"])
        return finishes


def _interrupted_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, persistent: bool
) -> _Interrupted:
    """Stop the native example the way a Ctrl-C during its first upload does.

    The shutdown arrives while trial 0's accepted outcome is being sent, and every upload after
    it is refused as a worker launch is while a shutdown is serviced. The run saves its records,
    the trial's accepted outcome and its summary run's KILLED end, and sends neither, so its
    summary run stays RUNNING.

    :param bool persistent: Record the trials in a journal; ``False`` keeps them in memory.
    :return _Interrupted: The experiment, the upload log and the interrupted generation.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    if persistent:
        config["storage"] = f"journal:///{tmp_path}/study.journal"
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    interrupted: list[int] = []

    def ctrl_c_at_first_upload(request: dict[str, object], **kwargs: object) -> JsonObject:
        if request["action"] == "publish":
            if interrupted:
                raise ShutdownLaunchRefusedError(interrupted[0])
            interrupted.append(signal.SIGINT)
            raise PhaseSweepShutdown(signal.SIGINT, ShutdownCleanupReport(signal.SIGINT, True, ()))
        return respond(request, **kwargs)

    reports: list[TerminalReport] = []
    with monkeypatch.context() as patch:
        patch.setattr(session, "request_operation", ctrl_c_at_first_upload)
        patch.setattr(worker, "request_operation", ctrl_c_at_first_upload)
        with pytest.raises(PhaseSweepShutdown):
            run_experiment(experiment, terminal_callback=reports.append)
    assert uploads == []
    return _Interrupted(experiment, uploads, reports[0].generation_id)


def _listing(root: Path) -> dict[str, tuple[int, int]]:
    """Map every entry below ``root`` to its size and modification time."""
    return {
        path.relative_to(root).as_posix(): (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.integration
@pytest.mark.parametrize("persistent", [True, False], ids=["journal", "in-memory"])
def test_the_next_run_delivers_an_interrupted_runs_pending_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    persistent: bool,
) -> None:
    """A run that reaches closeout sends what an interrupted earlier run saved, each record once.

    A dry run writes and sends nothing; the next real run ends the earlier summary run KILLED
    once, says so, and leaves nothing for sync-tracking.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=persistent)
    experiment = sweep.experiment
    summary = read_record(sweep.summary)
    assert (summary["state"], summary["finish"]) == ("pending", "KILLED")
    assert publication.delivery_status(experiment) == {"pending": 2, "delivered": 0}
    tree = _listing(tracking_root(experiment))

    run_experiment(experiment, dry_run=True)
    assert (_listing(tracking_root(experiment)), sweep.uploads) == (tree, [])

    caplog.set_level(logging.INFO)
    reports: list[TerminalReport] = []
    winners = run_experiment(experiment, terminal_callback=reports.append)

    assert set(winners) == {"depth", "rate"}
    assert reports[0].primary_error is None
    assert sweep.summaries() == ["KILLED"]
    assert read_record(sweep.summary)["state"] == "delivered"
    assert "MLflow: delivered 2 record(s) left pending by earlier runs." in caplog.text
    # A report block, so the console wraps it to the terminal.
    [delivered] = [r for r in caplog.records if "left pending by earlier runs." in r.getMessage()]
    assert delivered.name == REPORT_LOGGER and isinstance(delivered.msg, ReportBlock)
    assert publication.delivery_status(experiment)["pending"] == 0
    sweep.uploads.clear()
    publication.sync_tracking(experiment)
    assert sweep.uploads == []


@pytest.mark.integration
@pytest.mark.parametrize("creation", [False, True], ids=["uploads", "creation-and-uploads"])
@pytest.mark.parametrize("failure", ["outage", "rejection"])
@pytest.mark.parametrize("ending", ["normal", "failed"])
def test_closeout_sends_earlier_records_first_and_stops_at_the_first_unanswered_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ending: str,
    failure: str,
    creation: bool,
) -> None:
    """Closeout sends what earlier runs left before the run's own records, and one unanswered request ends it.

    The earlier run's records go first, its summary record last of them, so MLflow never shows
    this run's result before the outcomes an earlier run left. An outage at closeout costs one
    request, the search for an earlier creation intent when one is pending and otherwise the
    earlier run's first record, and leaves every record pending, the run's own too, under one
    warning. The run keeps its outcome, its winners or its own error, whether it ended normally
    or failed after checking its target. A rejection of one earlier record, after an answered
    search, is a verdict on that record alone, so every other record is still sent.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    respond = session.request_operation
    intents: list[Path] = []
    if creation:
        # A run killed outright while creating its summary run leaves that creation unconfirmed
        # and runs no closeout, as with `killed_sweep`.
        def killed_at_summary_creation(request: dict[str, object], **kwargs: object) -> JsonObject:
            if request["action"] == "create":
                raise RunnerKilled()
            return respond(request, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(session, "request_operation", killed_at_summary_creation)
            patch.setattr(publication, "deliver_closeout", lambda *args, **kwargs: None)
            with pytest.raises(RunnerKilled):
                run_experiment(sweep.experiment)
        assert sweep.uploads == []
        intents = session.pending_creation_records(sweep.experiment)
        assert len(intents) == 1
    closing: list[bool] = []
    requested: list[str] = []
    rejected: list[str] = []
    sent: list[tuple[str, str]] = []

    def failing(request: dict[str, object], **kwargs: object) -> JsonObject:
        if closing:
            name = str(request.get("name", request["action"]))
            requested.append(name)
            if failure == "outage":
                raise OperationUnavailableError("tracking server unreachable")
            if name == "find":
                return {"found": False}
            if name.startswith("attempt-") and not rejected:
                rejected.append(name)
                raise OperationRejectedError("RESOURCE_DOES_NOT_EXIST: run was deleted")
        return respond(request, **kwargs)

    close = publication.deliver_closeout
    deliver = publication._deliver

    def closeout(*args: object, **kwargs: object) -> None:
        closing.append(True)
        close(*args, **kwargs)

    def recording(experiment: Experiment, path: Path, *args: object, **kwargs: object) -> None:
        if closing:
            sent.append((path.parent.parent.name, path.stem))
        deliver(experiment, path, *args, **kwargs)

    monkeypatch.setattr(session, "request_operation", failing)
    monkeypatch.setattr(worker, "request_operation", failing)
    monkeypatch.setattr(publication, "deliver_closeout", closeout)
    monkeypatch.setattr(publication, "_deliver", recording)
    lost = RuntimeError("trainer host lost")
    if ending == "failed":
        run_phase = run_module._run_phase
        phases: list[str] = []

        def second_phase_fails(*args: object, **kwargs: object) -> object:
            # The first phase checked the target and bound the run's summary run.
            if phases:
                raise lost
            phases.append("depth")
            return run_phase(*args, **kwargs)

        monkeypatch.setattr(run_module, "_run_phase", second_phase_fails)
    caplog.set_level(logging.INFO)
    caplog.clear()
    reports: list[TerminalReport] = []

    if ending == "failed":
        with pytest.raises(RuntimeError) as caught:
            run_experiment(sweep.experiment, terminal_callback=reports.append)
        assert caught.value is lost
        assert reports[0].primary_error is lost
    else:
        winners = run_experiment(sweep.experiment, terminal_callback=reports.append)
        assert set(winners) == {"depth", "rate"}
        assert reports[0].primary_error is None
        assert read_status(sweep.experiment)["published_generation_id"] == reports[0].generation_id
    own = reports[0].generation_id
    own_pending = publication.delivery_status(sweep.experiment, own)["pending"]
    assert closing == [True]
    assert all(intent.exists() for intent in intents)
    searched = ["find"] if creation else []
    if failure == "outage":
        [request] = requested
        assert request == "find" if creation else request.startswith("attempt-")
        assert sent == ([] if creation else [(sweep.generation, request)])
        assert read_record(sweep.summary)["state"] == "pending"
        assert own_pending > 0
        assert caplog.text.count("MLflow did not answer") == 1
        earlier = 2 + len(intents)
        assert (
            f"MLflow: {earlier + own_pending} upload(s) pending ({earlier} from earlier runs); "
            f"{_PENDING_STEP}" in caplog.text
        )
        assert "left pending by earlier runs." not in caplog.text
    else:
        assert requested[: len(searched)] == searched
        [attempt] = rejected
        assert sent[:2] == [(sweep.generation, attempt), (sweep.generation, "generation")]
        assert sent[2:] and {generation for generation, _ in sent[2:]} == {own}
        assert sent[-1] == (own, "generation")
        assert sweep.summaries() == ["KILLED"]
        assert own_pending == 0
        assert "MLflow: delivered 1 record(s) left pending by earlier runs." in caplog.text
        pending = 1 + len(intents)
        assert (
            f"MLflow: {pending} upload(s) pending ({pending} from earlier runs); {_PENDING_STEP}"
            in caplog.text
        )


@pytest.mark.integration
def test_a_different_target_sends_nothing_and_the_original_config_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A config naming another MLflow target leaves an earlier run's records as they are, under one warning.

    The original config's sync-tracking then delivers them.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    experiment = sweep.experiment
    moved = experiment.model_copy(
        update={
            "mlflow": experiment.mlflow.model_copy(update={"tracking_uri": "http://moved:5000"})
        }
    )
    before = _listing(tracking_root(experiment))

    # The tracking URI is part of the trainer environment, so the kept trial's cohort refuses
    # the moved config before it allocates anything; its closeout still runs.
    with pytest.raises(StudyFingerprintMismatchError, match="http://moved:5000"):
        run_experiment(moved)

    assert sweep.uploads == []
    assert _listing(tracking_root(experiment)) == before
    assert caplog.text.count("records of earlier runs were neither saved nor sent") == 1
    assert publication.sync_tracking(experiment) == {"pending": 0, "delivered": 2}
    assert sweep.summaries() == ["KILLED"]
    # Once nothing is left to send, the moved target costs no warning.
    caplog.clear()
    publication.deliver_ended_generations(
        moved, validate_ledger(experiment), "f" * 32, session.DeliveryTally()
    )
    assert "neither saved nor sent" not in caplog.text
