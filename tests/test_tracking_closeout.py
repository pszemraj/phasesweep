"""Run closeout distinguishes accepted incomplete results and pending MLflow delivery, and delivers what earlier runs left pending."""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

from phasesweep.config.models import Experiment
from phasesweep.engine import run as run_module
from phasesweep.engine.errors import (
    PhaseTimeoutError,
    StudyFingerprintMismatchError,
    TrackingSyncRefusedError,
)
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.read import read_status
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.errors import ProcessCleanupUncertainError
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
    leak_tracking_worker,
    native_experiment,
    remote_mlflow_example,
)

_PENDING_STEP = (
    "results are saved locally. The next run or `phasesweep sync-tracking` with this run's "
    "config file and `--set` overrides sends them."
)
_REFUSAL = (
    "a refused record stays pending until the cause given in its warning is fixed in MLflow, for "
    "example by restoring a deleted run or its experiment."
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
        f" MLflow refused 7 records; {_REFUSAL}"
        if isinstance(failure, OperationRejectedError)
        and not isinstance(failure, OperationUnavailableError)
        else ""
    )
    assert message == f"MLflow: 7 records pending; {_PENDING_STEP}{refused}"
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


def _hang_uploads_until_killed(monkeypatch: pytest.MonkeyPatch, started: list[str]) -> None:
    """Make every MLflow upload run a real supervised worker that only a signal ends.

    The worker registers as ``runtime.remote_worker.run_worker`` registers a real one, so the
    shutdown handler names and kills it as an MLflow request. Every other request is answered
    as before.

    :param pytest.MonkeyPatch monkeypatch: Replaces the request function for the test.
    :param list[str] started: Receives the record name of each upload as it starts.
    """
    respond = session.request_operation

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
        started.append(str(request["name"]))
        with Path(os.devnull).open("w") as sink:
            result = process.run_supervised(
                "sleep 30",
                env=dict(os.environ),
                stdout=sink,
                stderr=sink,
                timeout=60,
                trial_dir=directory,
                attempt_id=operation_id,
                label="MLflow request",
            )
        raise OperationOutcomeUnknownError(f"MLflow worker failed (exit {result.return_code}).")

    monkeypatch.setattr(session, "request_operation", upload_until_killed)
    monkeypatch.setattr(worker, "request_operation", upload_until_killed)


def _kill_at_summary_creation(monkeypatch: pytest.MonkeyPatch, experiment: Experiment) -> Path:
    """Run ``experiment`` as a run killed outright while it creates its summary run.

    The creation is left unconfirmed and no closeout runs, as with ``killed_sweep``.

    :param pytest.MonkeyPatch monkeypatch: Replaces the request function for this run only.
    :param Experiment experiment: The experiment to start and kill.
    :return Path: The creation record the killed run left pending.
    """
    respond = session.request_operation

    def killed_at_summary_creation(request: dict[str, object], **kwargs: object) -> JsonObject:
        if request["action"] == "create":
            raise RunnerKilled()
        return respond(request, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(session, "request_operation", killed_at_summary_creation)
        patch.setattr(publication, "deliver_closeout", lambda *args, **kwargs: None)
        with pytest.raises(RunnerKilled):
            run_experiment(experiment)
    [intent] = session.pending_creation_records(experiment)
    return intent


@pytest.mark.integration
@pytest.mark.parametrize("persistent", [True, False], ids=["journal", "in-memory"])
def test_the_next_run_delivers_an_interrupted_runs_pending_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    persistent: bool,
) -> None:
    """The next run sends what an interrupted earlier run saved before its own first trial, each record once.

    A dry run writes and sends nothing. The next real run, right after its target check, says
    on the console how many records it is sending and ends the earlier summary run KILLED once,
    before its first trial's outcome goes out; its closeout sends only its own records, and its
    closing line counts the earlier ones. Nothing is left for sync-tracking, whose pass then
    sends and says nothing.
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
    sent = [(request["binding"]["generation_id"], request["name"]) for request in sweep.uploads]
    # The earlier run's trial outcome and its summary run's end, before any trial of this run.
    assert sent[:2] == [(sweep.generation, sent[0][1]), (sweep.generation, "generation")]
    assert sent[0][1].startswith("attempt-")
    assert (reports[0].generation_id, sent[2][1]) == sent[2]
    report = [r.getMessage() for r in caplog.records if r.name == REPORT_LOGGER]
    sending = [line for line in report if line.startswith("MLflow: sending")]
    assert sending[0] == "MLflow: sending 2 records left pending by earlier runs."
    assert len(sending) == 2 and re.fullmatch(r"MLflow: sending \d+ records?\.", sending[1])
    assert report.index(sending[1]) < report.index(
        "MLflow: delivered 2 records left pending by earlier runs."
    )
    # Report blocks, so the console wraps them to the terminal.
    blocks = [
        r.msg
        for r in caplog.records
        if r.name == REPORT_LOGGER and r.getMessage().startswith("MLflow: ")
    ]
    assert len(blocks) == 3 and all(isinstance(block, ReportBlock) for block in blocks)
    assert publication.delivery_status(experiment)["pending"] == 0
    sweep.uploads.clear()
    caplog.clear()
    publication.sync_tracking(experiment)
    assert sweep.uploads == []
    assert "MLflow: sending" not in caplog.text


@pytest.mark.integration
def test_the_pass_before_the_first_trial_leaves_the_phase_its_whole_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow pass over an earlier run's records spends none of the phase's timeout.

    The two records the interrupted run left take longer to send than the depth phase's
    ``timeout_seconds_per_phase``; the phase's budget starts once the pass ends, so its trial
    runs and the sweep publishes.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    phases = [
        phase.model_copy(update={"timeout_seconds_per_phase": 4.0})
        if phase.name == "depth"
        else phase
        for phase in sweep.experiment.phases
    ]
    experiment = sweep.experiment.model_copy(update={"phases": phases})
    respond = session.request_operation

    def slow_for_earlier_records(request: dict[str, object], **kwargs: object) -> JsonObject:
        binding = request.get("binding")
        if (
            request["action"] == "publish"
            and isinstance(binding, dict)
            and binding["generation_id"] == sweep.generation
        ):
            time.sleep(2.5)
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", slow_for_earlier_records)
    monkeypatch.setattr(worker, "request_operation", slow_for_earlier_records)

    assert set(run_experiment(experiment)) == {"depth", "rate"}
    assert sweep.summaries() == ["KILLED"]


@pytest.mark.integration
def test_the_pass_before_the_first_trial_never_outlasts_the_run_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracker silent during the pass before the first trial is given up on at the run deadline.

    The upload gets ``timeout_seconds_per_run``'s deadline rather than ``mlflow.timeout_seconds``,
    so the run stops on its own guard before any trial, and its closeout sends the record once
    the tracker answers.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    assert sweep.experiment.mlflow is not None
    experiment = sweep.experiment.model_copy(
        update={
            "timeout_seconds_per_run": 3.0,
            "mlflow": sweep.experiment.mlflow.model_copy(update={"timeout_seconds": 60.0}),
        }
    )
    respond = session.request_operation
    deadlines: list[float] = []

    def silent_once(request: dict[str, object], *, deadline: float, **kwargs: object) -> JsonObject:
        if request["action"] == "publish" and not deadlines:
            deadlines.append(deadline)
            time.sleep(max(0.0, deadline - time.monotonic()))
            raise TimeoutError("MLflow operation exceeded its deadline.")
        return respond(request, deadline=deadline, **kwargs)

    monkeypatch.setattr(session, "request_operation", silent_once)
    monkeypatch.setattr(worker, "request_operation", silent_once)
    started = time.monotonic()

    with pytest.raises(PhaseTimeoutError, match="Run wallclock deadline reached before phase"):
        run_experiment(experiment)

    # The run's deadline, set as it started, rather than one 60 s after the upload began.
    assert len(deadlines) == 1 and deadlines[0] - started < 5.0
    assert sweep.summaries() == ["KILLED"]


@pytest.mark.integration
@pytest.mark.parametrize("moment", ["trial", "target-check", "early-pass", "closeout"])
def test_a_shutdown_requests_nothing_more_and_still_counts_what_stays_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    moment: str,
) -> None:
    """A Ctrl-C stops MLflow delivery without blaming MLflow, and the closing line counts what stays pending.

    The shutdown arrives during a trial's upload; during the next run's target check while an
    earlier run's records are pending; after the first record of the pass that follows that
    check, before any trial; or after the first record of that run's closeout. No request
    follows it, no line says MLflow did not answer or was not ready, the closing lines count
    the earlier records the run delivered, and the closing line counts every pending record,
    the earlier run's in parentheses.
    """
    caplog.set_level(logging.INFO)
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    pending, earlier = 2, 0
    if moment != "trial":
        caplog.clear()
        respond = session.request_operation
        closing: list[bool] = []
        publishes: list[str] = []
        closeout_requests: list[str] = []
        after: list[str] = []

        def ctrl_c(request: dict[str, object], **kwargs: object) -> JsonObject:
            if shutdown.servicing_shutdown_signal() is not None:
                # What the supervisor does to every launch while the shutdown is serviced.
                after.append(str(request["action"]))
                raise ShutdownLaunchRefusedError(signal.SIGINT)
            if request["action"] == "publish":
                publishes.append(str(request["name"]))
            if closing:
                closeout_requests.append(str(request["action"]))
            if (
                (moment == "target-check" and request["action"] == "resolve")
                or (moment == "early-pass" and len(publishes) == 2)
                or (moment == "closeout" and len(closeout_requests) == 2)
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
        earlier = {"target-check": 2, "early-pass": 1, "closeout": 0}[moment]
        pending = publication.delivery_status(sweep.experiment)["pending"]
        own = publication.delivery_status(sweep.experiment, reports[0].generation_id)
        assert pending == earlier + own["pending"]
        # Only a run whose target check bound its summary run saves its own records.
        assert (own["pending"] > 0) is (moment != "target-check")
        delivered = {
            "target-check": None,
            "early-pass": "MLflow: delivered 1 record left pending by earlier runs.",
            "closeout": "MLflow: delivered 2 records left pending by earlier runs.",
        }[moment]
        lines = [r.getMessage() for r in caplog.records if r.name == REPORT_LOGGER]
        assert [line for line in lines if line.startswith("MLflow: delivered")] == (
            [delivered] if delivered else []
        )
    assert "did not answer" not in caplog.text
    assert "is not ready" not in caplog.text
    assert "launch refused" not in caplog.text
    in_parentheses = f" ({earlier} from earlier runs)" if earlier else ""
    assert pending > 1
    assert f"MLflow: {pending} records pending{in_parentheses}; {_PENDING_STEP}" in caplog.text


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
    requests: list[str] = []

    def interrupt_once_every_upload_runs() -> None:
        # A trial uploads once its trainer has exited, so then the registered children are the
        # upload workers alone.
        deadline = time.monotonic() + 30.0
        while (
            len(requests) < n_jobs or len(shutdown._active_children) < n_jobs
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)

    _hang_uploads_until_killed(monkeypatch, requests)
    sender = threading.Thread(target=interrupt_once_every_upload_runs, daemon=True)
    caplog.set_level(logging.INFO)
    sender.start()
    with pytest.raises(PhaseSweepShutdown):
        run_experiment(experiment)
    sender.join(timeout=5.0)

    stopping = "1 MLflow request" if n_jobs == 1 else f"{n_jobs} MLflow requests"
    assert f"Interrupted (SIGINT): no trial is running; stopping {stopping}." in caplog.text
    assert sorted(requests) == sorted({name for name in requests if name.startswith("attempt-")})
    assert (len(requests), uploads) == (n_jobs, [])
    assert publication.delivery_status(experiment) == {"pending": n_jobs + 1, "delivered": 0}
    for blame in ("did not answer", "is not ready", "launch refused", "is pending ("):
        assert blame not in caplog.text
    assert f"MLflow: {n_jobs + 1} records pending; {_PENDING_STEP}" in caplog.text


@pytest.mark.integration
@pytest.mark.parametrize("creation", [False, True], ids=["uploads", "creation-and-uploads"])
@pytest.mark.parametrize("failure", ["outage", "rejection"])
@pytest.mark.parametrize("ending", ["normal", "failed"])
def test_each_delivery_pass_sends_earlier_records_first_and_stops_at_the_first_unanswered_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ending: str,
    failure: str,
    creation: bool,
) -> None:
    """A run sends what earlier runs left after its target check and at closeout, and one unanswered request ends a pass.

    The pass after the target check sends the earlier run's records before any trial, its
    summary record last of them, so MLflow never shows this run's result before the outcomes an
    earlier run left; the closeout sends what is still pending before the run's own records. An
    outage costs each pass one request, the search for an earlier creation intent when one is
    pending and otherwise the earlier run's first record, and one warning; the run trains on
    between them, and every record stays pending, the run's own too. The run keeps its
    outcome, its winners or its own error, whether it ended normally or failed after checking
    its target. A rejection of one earlier record, after an answered search, is a verdict on
    that record alone, so every other record is still sent, and the closing line counts the
    record once though both passes met the refusal.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    respond = session.request_operation
    intents: list[Path] = []
    if creation:
        intents = [_kill_at_summary_creation(monkeypatch, sweep.experiment)]
        assert sweep.uploads == []
    closing: list[bool] = []
    delivering: list[bool] = []
    requested: list[list[str]] = []
    rejected: list[str] = []
    sent: list[list[tuple[str, str]]] = []

    def failing(request: dict[str, object], **kwargs: object) -> JsonObject:
        if delivering:
            name = str(request.get("name", request["action"]))
            requested[-1].append(name)
            if failure == "outage":
                raise OperationUnavailableError("tracking server unreachable")
            if name == "find":
                return {"found": False}
            if name.startswith("attempt-") and name == (rejected or [name])[0]:
                rejected.append(name)
                raise OperationRejectedError("RESOURCE_DOES_NOT_EXIST: run was deleted")
        return respond(request, **kwargs)

    close = publication.deliver_closeout
    deliver_pass = publication.deliver_ended_generations
    deliver = publication._deliver

    def closeout(*args: object, **kwargs: object) -> None:
        closing.append(True)
        close(*args, **kwargs)

    def delivery_pass(*args: object, **kwargs: object) -> None:
        delivering.append(True)
        requested.append([])
        sent.append([])
        try:
            deliver_pass(*args, **kwargs)
        finally:
            delivering.clear()

    def recording(experiment: Experiment, path: Path, *args: object, **kwargs: object) -> None:
        if delivering:
            sent[-1].append((path.parent.parent.name, path.stem))
        deliver(experiment, path, *args, **kwargs)

    monkeypatch.setattr(session, "request_operation", failing)
    monkeypatch.setattr(worker, "request_operation", failing)
    monkeypatch.setattr(publication, "deliver_closeout", closeout)
    monkeypatch.setattr(publication, "deliver_ended_generations", delivery_pass)
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
    assert len(intents) == creation
    assert all(intent.exists() for intent in intents)
    searched = ["find"] if creation else []
    # The pass after the target check, then the closeout's.
    assert len(requested) == len(sent) == 2
    if failure == "outage":
        for requests, sends in zip(requested, sent, strict=True):
            [request] = requests
            assert request == "find" if creation else request.startswith("attempt-")
            assert sends == ([] if creation else [(sweep.generation, request)])
        assert read_record(sweep.summary)["state"] == "pending"
        assert own_pending > 0
        assert caplog.text.count("MLflow did not answer") == 2
        earlier = 2 + len(intents)
        creations = " and 1 unconfirmed run creation" if creation else ""
        assert (
            f"MLflow: {2 + own_pending} records{creations} pending ({earlier} from earlier runs); "
            f"{_PENDING_STEP}" in caplog.text
        )
        assert "MLflow: delivered" not in caplog.text
    else:
        early, closeout_pass = sent
        assert [requests[: len(searched)] for requests in requested] == [searched, searched]
        attempt = rejected[0]
        assert rejected == [attempt, attempt]
        assert early == [(sweep.generation, attempt), (sweep.generation, "generation")]
        assert closeout_pass[0] == (sweep.generation, attempt)
        assert closeout_pass[1:] and {generation for generation, _ in closeout_pass[1:]} == {own}
        assert closeout_pass[-1] == (own, "generation")
        assert sweep.summaries() == ["KILLED"]
        assert own_pending == 0
        assert "MLflow: delivered 1 record left pending by earlier runs." in caplog.text
        pending = 1 + len(intents)
        # The search answered that no run carries the identity yet, inside the creation window.
        creations = " and 1 unconfirmed run creation" if creation else ""
        assert (
            f"MLflow: 1 record{creations} pending ({pending} from earlier runs); {_PENDING_STEP} "
            f"MLflow refused 1 record; {_REFUSAL}" in caplog.text
        )
        assert (
            f"MLflow: attempt record {attempt} of generation {sweep.generation} is pending "
            "(RESOURCE_DOES_NOT_EXIST: run was deleted)." in caplog.text
        )


@pytest.mark.integration
def test_the_closing_lines_account_for_every_creation_and_record_the_sending_lines_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A closeout counts run creations apart from records, and its closing line adds them up.

    An interrupted run left its trial's outcome and its summary run's end pending, and a run
    killed while creating its summary run left that creation unconfirmed. The closeout first
    says it is checking that creation; the search binds the run, so the killed run's summary
    record is rebuilt and sent in the same pass, counted in the sending line that follows the
    search. The closing line names the creation settled and every record delivered.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    respond = session.request_operation
    intent = _kill_at_summary_creation(monkeypatch, sweep.experiment)
    killed = intent.parent.name

    def late_run(request: dict[str, object], **kwargs: object) -> JsonObject:
        if request["action"] == "find":
            binding = request["binding"]
            assert isinstance(binding, dict)
            return {"found": True, "binding": {**binding, "run_id": "late-run"}}
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", late_run)
    monkeypatch.setattr(worker, "request_operation", late_run)
    caplog.set_level(logging.INFO)
    caplog.clear()
    publication.deliver_closeout(sweep.experiment, validate_ledger(sweep.experiment), "f" * 32)

    assert [r.getMessage() for r in caplog.records if r.name == REPORT_LOGGER] == [
        "MLflow: checking 1 unconfirmed run creation left by earlier runs.",
        "MLflow: sending 3 records left pending by earlier runs.",
        "MLflow: delivered 3 records and settled 1 unconfirmed run creation left pending by "
        "earlier runs.",
    ]
    assert not intent.exists()
    killed_summaries = [
        request["finish"]
        for request in sweep.uploads
        if request["name"] == "generation" and request["binding"]["generation_id"] == killed
    ]
    assert killed_summaries == ["KILLED"]
    assert sweep.summaries() == ["KILLED"]
    assert publication.delivery_status(sweep.experiment)["pending"] == 0
    assert "pending;" not in caplog.text


@pytest.mark.integration
def test_an_unconfirmed_worker_of_the_early_pass_still_ends_the_run_with_its_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A worker the pass after the target check leaves unconfirmed stops the run, which still counts.

    Nothing more is requested once cleanup is uncertain, but the closing count is a local read,
    so it names what stays pending, the earlier run's records included.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    leaked = leak_tracking_worker(monkeypatch, action="publish")
    caplog.set_level(logging.INFO)

    with pytest.raises(ProcessCleanupUncertainError):
        run_experiment(sweep.experiment)

    assert leaked and sweep.uploads == []
    assert f"MLflow: 2 records pending (2 from earlier runs); {_PENDING_STEP}" in caplog.text


@pytest.mark.integration
@pytest.mark.parametrize("damage", ["no-source", "unknown-state"])
def test_a_malformed_earlier_record_costs_a_warning_naming_it_and_the_run_trains_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    damage: str,
) -> None:
    """An earlier run's record that lost its target, or holds an unknown state, is refused by name.

    Both delivery passes warn naming the record's file and send nothing of the earlier run, the
    record is left as it is rather than sent and rewritten, and the run trains and publishes.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    [attempt] = sweep.summary.parent.glob("attempt-*.json")
    record = read_record(attempt)
    damaged = (
        {key: value for key, value in record.items() if key != "source"}
        if damage == "no-source"
        else {**record, "state": "refused"}
    )
    session.write_record(attempt, damaged)
    caplog.set_level(logging.INFO)
    reports: list[TerminalReport] = []

    winners = run_experiment(sweep.experiment, terminal_callback=reports.append)

    assert set(winners) == {"depth", "rate"}
    assert reports[0].primary_error is None
    assert read_record(attempt) == damaged
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "were not all sent" in r.getMessage()
    ]
    assert len(warnings) == 2 and all(str(attempt) in warning for warning in warnings)
    assert sweep.generation not in {
        request["binding"]["generation_id"] for request in sweep.uploads
    }


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
    assert f"MLflow: 2 records pending (2 from earlier runs); {_PENDING_STEP}" in caplog.text
    run_experiment(experiment)
    assert sweep.summaries() == ["KILLED"]


@pytest.mark.integration
def test_a_pass_whose_target_resolve_gets_no_answer_never_says_it_is_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A pass that resolves the target first, as a replay's closeout does, reports sending only once it answered.

    With the tracker silent, the resolve's warning is the pass's only line and nothing is sent.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)

    def silent(request: dict[str, object], **kwargs: object) -> JsonObject:
        raise TimeoutError("MLflow operation exceeded its deadline.")

    monkeypatch.setattr(session, "request_operation", silent)
    monkeypatch.setattr(worker, "request_operation", silent)
    caplog.set_level(logging.INFO)
    caplog.clear()
    tally = session.DeliveryTally()

    # A generation that bound no summary run made no target check of its own.
    publication.deliver_ended_generations(
        sweep.experiment, validate_ledger(sweep.experiment), "f" * 32, tally
    )

    assert not tally.sending
    assert [r.getMessage() for r in caplog.records if r.name == REPORT_LOGGER] == []
    assert "is not ready: it did not answer within timeout_seconds" in caplog.text
    assert publication.delivery_status(sweep.experiment)["pending"] == 2


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
    record MLflow refuses, as for a run deleted in MLflow, is named in its warning, printed on
    the console under a ``warning:`` label, and the step is to fix that cause in MLflow, since
    a retry alone meets the same refusal. Either step ends with the command as typed, its
    ``--set`` and ``--generation`` flags included, on a line of its own.
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
    arguments = [str(config), "--set", "phases.0.n_trials=2", "--generation", sweep.generation]

    assert invoke_cli_boundary(["sync-tracking", *arguments], monkeypatch) == 1

    error = capsys.readouterr().err
    command = shlex.join(["phasesweep", "sync-tracking", *arguments]) + "\n"
    if failure == "no-answer":
        assert error.endswith(
            "phasesweep: MLflow delivery remains pending: MLflow tracking target "
            f"experiment_name={experiment.mlflow.experiment_name!r} at "
            f"{experiment.mlflow.tracking_uri!r} is not ready: it did not answer within "
            "timeout_seconds; check that the server is reachable.\n"
            f"fix: Once the server answers, run this command again:\n{command}"
        )
        assert sweep.summaries() == []
    else:
        assert error.endswith(
            f"phasesweep: MLflow delivery remains pending: MLflow refused 1 record; {_REFUSAL}\n"
            f"fix: Once each such cause is fixed, run this command again:\n{command}"
        )
        assert (
            f"MLflow: attempt record {stem} of generation {sweep.generation} is pending (MLflow "
            "run is deleted; restore it, or its experiment.)." in caplog.text
        )
        [warning] = [line for line in error.splitlines() if "MLflow: attempt record" in line]
        assert re.fullmatch(r"\d\d:\d\d:\d\d  warning: MLflow: attempt record\b.*", warning)
        assert sweep.summaries() == ["KILLED"]


@pytest.mark.integration
@pytest.mark.parametrize("ending", ["finished", "stopped"])
def test_a_run_started_with_overrides_prints_the_sync_command_that_sends_what_it_left(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ending: str,
) -> None:
    """A run that leaves MLflow work pending prints the sync-tracking command as typed, and it delivers.

    Whether the run finishes with its uploads unanswered or a shutdown stops it, the last lines
    before the log path give the command with the run's ``--set`` overrides, whole on a line of
    its own; the closing warning says in words what that command needs. Run as printed, the
    command finds the study the overrides named and sends everything.
    """
    config, uploads = remote_mlflow_example(tmp_path, monkeypatch)
    config["storage"] = "auto"
    path = write_experiment_config(
        Path.cwd() / "experiment.yaml", Experiment.model_validate(config)
    )
    respond = session.request_operation
    stopped: list[int] = []

    def unsent(request: dict[str, object], **kwargs: object) -> JsonObject:
        if request["action"] != "publish":
            return respond(request, **kwargs)
        if ending == "finished":
            raise OperationUnavailableError("tracking server unreachable")
        if stopped:
            raise ShutdownLaunchRefusedError(stopped[0])
        stopped.append(signal.SIGINT)
        raise PhaseSweepShutdown(signal.SIGINT, ShutdownCleanupReport(signal.SIGINT, True, ()))

    capsys.readouterr()
    with monkeypatch.context() as patch:
        patch.setattr(session, "request_operation", unsent)
        patch.setattr(worker, "request_operation", unsent)
        status = invoke_cli_boundary(["run", str(path), "--set", "experiment=tuned"], monkeypatch)

    assert status == (0 if ending == "finished" else 130)
    lines = capsys.readouterr().err.splitlines()
    printed = shlex.join(["phasesweep", "sync-tracking", str(path), "--set", "experiment=tuned"])
    step = lines.index("MLflow: to send what is pending without training, run:")
    # Only the log path, which may break after its slashes, follows the command.
    assert (lines[step + 1], lines[step + 2][:5]) == (printed, "log: ")
    assert lines.count(printed) == 1
    # The closing warning wraps to the terminal.
    assert _PENDING_STEP in " ".join(" ".join(lines).split())
    assert uploads == []

    assert invoke_cli_boundary(shlex.split(printed)[1:], monkeypatch) == 0
    assert capsys.readouterr().out.splitlines()[0] == "pending: 0"
    assert uploads and all(request["action"] == "publish" for request in uploads)


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_a_ctrl_c_during_sync_tracking_says_what_stays_pending_and_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real SIGINT while sync-tracking uploads prints the interrupt, what stays pending and the command.

    The shutdown handler kills the upload worker in flight and the command exits 130, after
    the interrupt line and a line counting what stays pending, followed by the command as
    typed on a line of its own. Nothing is delivered.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    uploading: list[str] = []

    def interrupt_once_the_upload_runs() -> None:
        deadline = time.monotonic() + 30.0
        while (not uploading or not shutdown._active_children) and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)

    _hang_uploads_until_killed(monkeypatch, uploading)
    config = write_experiment_config(Path.cwd() / "experiment.yaml", sweep.experiment)
    capsys.readouterr()
    sender = threading.Thread(target=interrupt_once_the_upload_runs, daemon=True)
    sender.start()

    status = invoke_cli_boundary(["sync-tracking", str(config)], monkeypatch)
    sender.join(timeout=5)

    error = capsys.readouterr().err
    assert status == 130
    assert len(uploading) == 1
    assert re.search(
        r"^\d\d:\d\d:\d\d  Interrupted \(SIGINT\): no trial is running; "
        r"stopping 1 MLflow request\.$",
        error,
        re.MULTILINE,
    )
    lines = error.splitlines()
    # The sentence wraps to the terminal; the command stays whole on the last line.
    assert lines[-1] == f"phasesweep sync-tracking {config}"
    stopped = next(i for i, line in enumerate(lines) if line.startswith("phasesweep: stopped"))
    assert " ".join(lines[stopped:-1]) == (
        "phasesweep: stopped with 2 records pending. Rerun the same command to send what is left:"
    )
    assert publication.delivery_status(sweep.experiment) == {"pending": 2, "delivered": 0}
    for blame in ("did not answer", "is not ready", "is pending ("):
        assert blame not in error


@pytest.mark.integration
def test_sync_tracking_says_when_a_creation_inside_its_window_can_be_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A creation MLflow has no run for yet, inside its window, gets its cause and when to retry.

    The search answers that no run carries the identity, so the lost request may still create
    one until timeout_seconds after it was sent; the step names that time and the command.
    """
    sweep = _interrupted_sweep(tmp_path, monkeypatch, persistent=True)
    experiment = sweep.experiment.model_copy(
        update={"mlflow": sweep.experiment.mlflow.model_copy(update={"timeout_seconds": 600.0})}
    )
    respond = session.request_operation
    intent = _kill_at_summary_creation(monkeypatch, experiment)

    def no_run_yet(request: dict[str, object], **kwargs: object) -> JsonObject:
        if request["action"] == "find":
            return {"found": False}
        return respond(request, **kwargs)

    monkeypatch.setattr(session, "request_operation", no_run_yet)
    monkeypatch.setattr(worker, "request_operation", no_run_yet)
    config = write_experiment_config(Path.cwd() / "experiment.yaml", experiment)
    capsys.readouterr()

    assert invoke_cli_boundary(["sync-tracking", str(config)], monkeypatch) == 1

    error = capsys.readouterr().err
    until = time.strftime("%H:%M:%S", time.localtime(intent.stat().st_mtime + 600.0))
    match = re.search(
        r"phasesweep: MLflow delivery remains pending: MLflow has no run yet for 1 unconfirmed "
        r"run creation, and the request whose answer was lost may still create one until "
        r"timeout_seconds \(600 s\) after it was sent\.\n"
        rf"fix: Wait until {until}, about (\d+) s from now, then run this command again:\n"
        rf"phasesweep sync-tracking {re.escape(str(config))}\n$",
        error,
    )
    assert match is not None, error
    assert 0 < int(match.group(1)) <= 600
    assert intent.exists()


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
