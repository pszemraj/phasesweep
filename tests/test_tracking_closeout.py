"""Run closeout distinguishes accepted incomplete results and pending MLflow delivery, and delivers what earlier runs left pending."""

from __future__ import annotations

import logging
import os
import re
import signal
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine import run as run_module
from phasesweep.engine.errors import StudyFingerprintMismatchError, TrackingSyncRefusedError
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.runtime import process, shutdown
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    ShutdownCleanupReport,
    ShutdownLaunchRefusedError,
)
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportBlock
from phasesweep.tracking import mlflow as worker
from phasesweep.tracking import publication, session
from phasesweep.tracking.mlflow import (
    OperationOutcomeUnknownError,
    OperationRejectedError,
    OperationUnavailableError,
)
from phasesweep.tracking.session import read_record, tracking_root
from tests.conftest import invoke_cli_boundary, write_experiment_config
from tests.tracking_helpers import (
    RunnerKilled,
    attempt_identity,
    native_experiment,
    remote_mlflow_example,
)

_PENDING_STEP = (
    "results are saved locally. The next run or `phasesweep sync-tracking CONFIG` sends them."
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
    neither failure reaches the trial or the run. Records MLflow refused at closeout are counted
    in the final warning, which says what keeps them pending.
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
    refused = (
        " MLflow refused 7 of them; a refused record stays pending until the cause its warning "
        "gives is fixed in MLflow."
        if isinstance(failure, OperationRejectedError)
        and not isinstance(failure, OperationUnavailableError)
        else ""
    )
    assert message == f"MLflow: 7 record(s) pending; {_PENDING_STEP}{refused}"
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

    A dry run writes and sends nothing; the next real run says on the console how many records
    it is sending before its first request, ends the earlier summary run KILLED once, says so,
    and leaves nothing for sync-tracking, whose pass then sends and says nothing.
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
    report = [r.getMessage() for r in caplog.records if r.name == REPORT_LOGGER]
    [sending] = [line for line in report if line.startswith("MLflow: sending")]
    assert re.fullmatch(
        r"MLflow: sending \d+ record\(s\), 2 of them left pending by earlier runs\.", sending
    )
    assert report.index(sending) < report.index(
        "MLflow: delivered 2 record(s) left pending by earlier runs."
    )
    # Report blocks, so the console wraps them to the terminal.
    blocks = [
        r.msg
        for r in caplog.records
        if r.name == REPORT_LOGGER and r.getMessage().startswith("MLflow: ")
    ]
    assert len(blocks) == 2 and all(isinstance(block, ReportBlock) for block in blocks)
    assert publication.delivery_status(experiment)["pending"] == 0
    sweep.uploads.clear()
    caplog.clear()
    publication.sync_tracking(experiment)
    assert sweep.uploads == []
    assert "MLflow: sending" not in caplog.text


@pytest.mark.integration
@pytest.mark.parametrize("moment", ["trial", "target-check", "closeout"])
def test_a_shutdown_requests_nothing_more_and_still_counts_what_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    moment: str,
) -> None:
    """A Ctrl-C stops MLflow delivery without blaming MLflow, and the closing line counts what stays pending.

    The shutdown arrives during a trial's upload, during the next run's target check while an
    earlier run's records are pending, or during that run's closeout delivery, after its first
    record. No request follows it, no line says MLflow did not answer or was not ready, the
    record the interrupted closeout delivered is still reported, and the closing line counts
    every pending record, the earlier run's in parentheses.
    """
    caplog.set_level(logging.INFO)
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    pending, earlier = 2, 0
    if moment != "trial":
        caplog.clear()
        respond = session.request_operation
        closing: list[bool] = []
        closeout_requests: list[str] = []
        after: list[str] = []

        def ctrl_c(request: dict[str, object], **kwargs: object) -> JsonObject:
            if shutdown.servicing_shutdown_signal() is not None:
                # What the supervisor does to every launch while the shutdown is serviced.
                after.append(str(request["action"]))
                raise ShutdownLaunchRefusedError(signal.SIGINT)
            if closing:
                closeout_requests.append(str(request["action"]))
            if (moment == "target-check" and request["action"] == "resolve") or (
                moment == "closeout" and len(closeout_requests) == 2
            ):
                monkeypatch.setattr(shutdown, "_servicing_signum", signal.SIGINT)
                report = ShutdownCleanupReport(signal.SIGINT, True, ())
                raise PhaseSweepShutdown(signal.SIGINT, report)
            return respond(request, **kwargs)

        close = publication.deliver_closeout

        def closeout(*args: object, **kwargs: object) -> None:
            closing.append(True)
            close(*args, **kwargs)

        monkeypatch.setattr(session, "request_operation", ctrl_c)
        monkeypatch.setattr(worker, "request_operation", ctrl_c)
        monkeypatch.setattr(publication, "deliver_closeout", closeout)
        reports: list[TerminalReport] = []
        with pytest.raises(PhaseSweepShutdown):
            run_experiment(sweep.experiment, terminal_callback=reports.append)
        assert (closing, after) == ([True], [])
        earlier = 1 if moment == "closeout" else 2
        pending = publication.delivery_status(sweep.experiment)["pending"]
        own = publication.delivery_status(sweep.experiment, reports[0].generation_id)
        assert pending == earlier + own["pending"]
        assert (own["pending"] > 0) is (moment == "closeout")
        delivered = "MLflow: delivered 1 record(s) left pending by earlier runs."
        assert (delivered in caplog.text) is (moment == "closeout")
    assert "did not answer" not in caplog.text
    assert "is not ready" not in caplog.text
    assert "launch refused" not in caplog.text
    in_parentheses = f" ({earlier} from earlier runs)" if earlier else ""
    assert f"MLflow: {pending} record(s) pending{in_parentheses}; {_PENDING_STEP}" in caplog.text


@pytest.mark.integration
@pytest.mark.signals_own_pid
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_ctrl_c_while_trials_upload_stops_delivery_without_blaming_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, n_jobs: int
) -> None:
    """A real SIGINT while each trial's outcome is uploading stops MLflow delivery quietly.

    The shutdown handler kills every upload worker in flight. On the main thread the shutdown
    raises out of the upload; on each parallel trial's worker thread the killed upload ends its
    pass. No request follows, no line blames MLflow, and the closing line counts every record
    left pending: each trial's accepted outcome and the summary run's KILLED end.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = f"journal:///{tmp_path}/study.journal"
    config["phases"] = config["phases"][:1]
    # With one job the signal arrives during the first trial's upload, so the second never runs.
    config["phases"][0].update(n_jobs=n_jobs, allow_no_gpu_isolation=True)
    experiment = Experiment.model_validate(config)
    respond = session.request_operation
    requests: list[str] = []

    def upload_until_killed(
        request: Mapping[str, object],
        *,
        operation_id: str,
        directory: Path,
        environment: Mapping[str, str],
        deadline: float,
    ) -> JsonObject:
        if request["action"] != "publish":
            return respond(
                request,
                operation_id=operation_id,
                directory=directory,
                environment=environment,
                deadline=deadline,
            )
        requests.append(str(request["name"]))
        # A real supervised worker, so the handler sees and kills it as an MLflow upload's.
        with Path(os.devnull).open("w") as sink:
            result = process.run_supervised(
                "sleep 30",
                env=dict(os.environ),
                stdout=sink,
                stderr=sink,
                timeout=60,
                trial_dir=directory,
                attempt_id=operation_id,
            )
        raise OperationOutcomeUnknownError(f"MLflow worker failed (exit {result.return_code}).")

    def interrupt_once_every_upload_runs() -> None:
        # A trial uploads once its trainer has exited, so then the registered children are the
        # upload workers alone.
        deadline = time.monotonic() + 30.0
        while (
            len(requests) < n_jobs or len(shutdown._active_children) < n_jobs
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(session, "request_operation", upload_until_killed)
    monkeypatch.setattr(worker, "request_operation", upload_until_killed)
    sender = threading.Thread(target=interrupt_once_every_upload_runs, daemon=True)
    caplog.set_level(logging.INFO)
    sender.start()
    with pytest.raises(PhaseSweepShutdown):
        run_experiment(experiment)
    sender.join(timeout=5.0)

    assert sorted(requests) == sorted({name for name in requests if name.startswith("attempt-")})
    assert (len(requests), uploads) == (n_jobs, [])
    assert publication.delivery_status(experiment) == {"pending": n_jobs + 1, "delivered": 0}
    for blame in ("did not answer", "is not ready", "launch refused", "is pending ("):
        assert blame not in caplog.text
    assert f"MLflow: {n_jobs + 1} record(s) pending; {_PENDING_STEP}" in caplog.text


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
            f"MLflow: {earlier + own_pending} record(s) pending ({earlier} from earlier runs); "
            f"{_PENDING_STEP}" in caplog.text
        )
        assert "MLflow: delivered" not in caplog.text
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
            f"MLflow: {pending} record(s) pending ({pending} from earlier runs); {_PENDING_STEP} "
            "MLflow refused 1 of them; a refused record stays pending until the cause its warning "
            "gives is fixed in MLflow." in caplog.text
        )
        assert (
            f"MLflow: attempt record {attempt} of generation {sweep.generation} is pending "
            "(RESOURCE_DOES_NOT_EXIST: run was deleted)." in caplog.text
        )


@pytest.mark.integration
def test_a_run_refused_before_its_target_check_never_waits_on_the_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A run that fails before checking its target makes no MLflow request for earlier records.

    Earlier records stay pending and are counted, so a refusal is reported at once rather
    than after a down tracker's timeout; the next run that checks its target sends them.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    experiment = sweep.experiment
    changed = experiment.model_copy(update={"env": {**experiment.env, "EXTRA_FLAG": "1"}})

    requests: list[object] = []

    def no_request(request: dict[str, object], **kwargs: object) -> JsonObject:
        # A raise from here is swallowed by the closeout, so the request is recorded instead.
        requests.append(request["action"])
        raise OperationUnavailableError("tracking server unreachable")

    caplog.set_level(logging.INFO)
    with monkeypatch.context() as patch:
        patch.setattr(session, "request_operation", no_request)
        patch.setattr(worker, "request_operation", no_request)
        # The trainer environment is part of the kept trial's cohort, so the changed config is
        # refused before it allocates anything or checks its target.
        with pytest.raises(StudyFingerprintMismatchError):
            run_experiment(changed)

    assert requests == []
    assert read_record(sweep.summary)["state"] == "pending"
    assert "MLflow: sending" not in caplog.text
    assert f"MLflow: 2 record(s) pending (2 from earlier runs); {_PENDING_STEP}" in caplog.text
    run_experiment(experiment)
    assert sweep.summaries() == ["KILLED"]


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


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["no-answer", "refusal"])
def test_sync_tracking_names_what_keeps_its_records_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    """sync-tracking ends with the step that clears what is left: a retry or a fix in MLflow.

    A tracker that gives no answer is named with its tracking URI, and the step is a retry. A
    record MLflow refuses, as for a run deleted in MLflow, is named in its warning, and the
    step is to fix that cause in MLflow, since a retry alone meets the same refusal.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    experiment = sweep.experiment
    assert experiment.mlflow is not None
    [stem] = [path.stem for path in sweep.summary.parent.glob("attempt-*.json")]
    respond = session.request_operation

    def failing(request: dict[str, object], **kwargs: object) -> JsonObject:
        if failure == "no-answer":
            raise TimeoutError("MLflow operation exceeded its deadline.")
        if request["action"] == "publish" and request["name"] == stem:
            raise OperationRejectedError("MLflow run is deleted; restore it, or its experiment.")
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", failing)
    monkeypatch.setattr(worker, "request_operation", failing)
    config = write_experiment_config(Path.cwd() / "experiment.yaml", experiment)
    capsys.readouterr()

    assert invoke_cli_boundary(["sync-tracking", str(config)], monkeypatch) == 1

    error = capsys.readouterr().err
    if failure == "no-answer":
        assert error == (
            "phasesweep: MLflow delivery remains pending: MLflow tracking target "
            f"experiment_name={experiment.mlflow.experiment_name!r} at "
            f"{experiment.mlflow.tracking_uri!r} is not ready: it did not answer within "
            "timeout_seconds; check that the server is reachable.\n"
            "fix: Wait briefly, then retry the same operation once the reported blocker is "
            "resolved.\n"
        )
        assert sweep.summaries() == []
    else:
        assert error == (
            "phasesweep: MLflow delivery remains pending: MLflow refused 1 record(s), each for "
            "the cause its warning gives.\n"
            "fix: Fix the cause MLflow gave for each refused record, for example by restoring a "
            "deleted run or its experiment in MLflow, then run `phasesweep sync-tracking` again.\n"
        )
        assert (
            f"MLflow: attempt record {stem} of generation {sweep.generation} is pending (MLflow "
            "run is deleted; restore it, or its experiment.)." in caplog.text
        )
        assert sweep.summaries() == ["KILLED"]


def test_a_run_whose_target_gives_no_answer_ends_with_the_retry_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`phasesweep run` against a tracker that gives no answer to its target check is told to retry.

    The refusal names the target and starts no trial, and the step is the retry `sync-tracking`
    prints for the same tracker, not a configuration fix.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    experiment = Experiment.model_validate(config)
    assert experiment.mlflow is not None

    def silent(request: dict[str, object], **kwargs: object) -> JsonObject:
        raise TimeoutError("MLflow operation exceeded its deadline.")

    monkeypatch.setattr(session, "request_operation", silent)
    monkeypatch.setattr(worker, "request_operation", silent)
    path = write_experiment_config(Path.cwd() / "experiment.yaml", experiment)
    capsys.readouterr()

    assert invoke_cli_boundary(["run", str(path)], monkeypatch) == 1

    assert uploads == []
    assert (
        f"phasesweep: MLflow tracking target experiment_name={experiment.mlflow.experiment_name!r} "
        f"at {experiment.mlflow.tracking_uri!r} is not ready: it did not answer within "
        "timeout_seconds; check that the server is reachable. No trial was started.\n"
        "fix: Wait briefly, then retry the same operation once the reported blocker is resolved.\n"
    ) in capsys.readouterr().err


def test_a_silent_search_names_a_creation_record_whatever_its_generation_id_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A search MLflow does not answer stops the pass at the first creation record and names it.

    An MCP run's generation id is named after its catalog id, which may hold ``__``. The
    record's warning names the attempt and the generation read between the fixed markers of its
    trial directory, no local error is reported, and the second record is not searched.
    """
    experiment = native_experiment(tmp_path)
    generation = "lr__sweep-" + "a" * 8
    attempts = ["b" * 32, "c" * 32]
    for number, attempt in enumerate(attempts):
        trial = f"trial_{number:05d}__generation_{generation}__attempt_{attempt}"
        path = _experiment_dir(experiment) / "depth" / trial / "mlflow-binding.pending.json"
        path.parent.mkdir(parents=True)
        intent = {"identity": attempt_identity(generation, attempt), "run_name": "r"}
        session.write_record(path, {"schema_version": 1, **intent})
    searches: list[str] = []

    def silent(*args: object, **kwargs: object) -> JsonObject:
        searches.append("find")
        raise TimeoutError("MLflow operation exceeded its deadline.")

    monkeypatch.setattr(session, "call_tracking", silent)
    errors: list[TrackingSyncRefusedError] = []
    tally = session.DeliveryTally()
    session.resolve_pending_creations(experiment, {}, None, errors=errors, tally=tally)

    assert (searches, errors, tally.sending) == (["find"], [], False)
    assert (
        f"MLflow: the run-creation record of attempt {attempts[0]} of generation {generation} is "
        "still unresolved (MLflow operation exceeded its deadline.)." in caplog.text
    )
