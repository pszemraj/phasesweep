"""A rerun continues in its trial's directory and is scored only from files its own attempt wrote."""

from __future__ import annotations

import dataclasses
import hashlib
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Observation
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.evidence import _verify_winner_evidence
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.paths import _attempt_dir, _phase_dir, _trial_dir_for
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
)
from phasesweep.engine.trial_dirs import (
    TrialDirectory,
    _expected_trial_dir_name,
    _trial_dir_for_attempt,
)
from phasesweep.errors import OperatorAction, PhaseSweepError, ProcessCleanupUncertainError
from phasesweep.evidence.baseline import (
    EVIDENCE_PINS_DIR,
    read_evidence_baseline,
    release_evidence_pins,
    write_evidence_baseline,
)
from phasesweep.evidence.evaluation import ExtractorError, run_extractor
from phasesweep.evidence.models import JsonExtractor, LogRegexExtractor
from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle
from phasesweep.tracking import publication, session
from tests.conftest import (
    make_experiment,
    make_trial_context,
    phase_trials,
    reaped_pid,
    trial_outcome,
    write_experiment_config,
    write_flaky_trainer,
    write_trainer,
)
from tests.recovery_helpers import runner_once, write_trial_identity
from tests.tracking_helpers import killed_sweep

_JSON_METRIC = Metric(extractor=JsonExtractor(type="json", path="r.json", key="loss"))


def test_a_value_file_left_before_launch_fails_as_not_written_by_this_attempt(
    tmp_path: Path,
) -> None:
    """Metadata changes leave stale evidence, while changed contents and replacement qualify."""
    experiment = make_experiment(persistent=tmp_path, metric=_JSON_METRIC)
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    trial_dir.mkdir()
    attempt_dir.mkdir()
    (trial_dir / "r.json").write_text('{"loss": 0.1}', encoding="utf-8")

    baseline = write_evidence_baseline(
        experiment, experiment.phases[0], trial_dir, attempt_dir, reused=True
    )
    assert read_evidence_baseline(attempt_dir) == baseline and set(baseline.files) == {"r.json"}
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=baseline,
    )
    (trial_dir / "r.json").chmod(0o600)
    (trial_dir / "r.json").touch()
    with pytest.raises(ExtractorError, match="was not written by this attempt"):
        run_extractor(ctx, _JSON_METRIC.extractor)

    (trial_dir / "r.json").write_text('{"loss": 0.25}', encoding="utf-8")
    assert run_extractor(ctx, _JSON_METRIC.extractor) == 0.25

    atomic_write_text(trial_dir / "r.json", '{"loss": 0.5}')
    atomic_write_text(trial_dir / "r.json", '{"loss": 0.1}')
    assert run_extractor(ctx, _JSON_METRIC.extractor) == 0.1


@pytest.mark.parametrize(
    ("prior", "content", "expected_line"),
    [
        (b"x=0.1\r\nx=0.", b"x=0.1\r\nx=0.25\r\nx=0.8\n", 3),
        (b"x=0.1\r\nunfinished ", b"x=0.1\r\nunfinished x=0.8\n", 2),
        (b"x=0.1\r\nx=0.", b"x=0.8\n", 1),
        (b"x=0.1\r\nx=0.", b"x=0.8\nnew log contents follow\n", 1),
    ],
    ids=["append", "partial-line", "truncate", "rewrite"],
)
def test_resumed_log_selects_fresh_lines_and_digests_the_complete_file(
    tmp_path: Path, prior: bytes, content: bytes, expected_line: int
) -> None:
    """Regex selection reads the new bytes, with provenance covering the complete log."""
    extractor = LogRegexExtractor(
        type="log_regex", file="train.log", pattern=r"x=(?P<value>[0-9.]+)", select="first"
    )
    experiment = make_experiment(persistent=tmp_path, metric=Metric(extractor=extractor))
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    trial_dir.mkdir()
    attempt_dir.mkdir()
    log_file = trial_dir / "train.log"
    log_file.write_bytes(prior)
    write_evidence_baseline(experiment, experiment.phases[0], trial_dir, attempt_dir, reused=True)
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=read_evidence_baseline(attempt_dir),
    )
    log_file.write_bytes(content)
    provenance: JsonObject = {}
    assert run_extractor(ctx, extractor, provenance=provenance) == 0.8
    assert provenance["source"] == {
        "kind": "file",
        "path": "train.log",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "matched_line": expected_line,
        "match_count": 1,
    }


def test_resumed_replacement_does_not_skip_the_old_prefix(tmp_path: Path) -> None:
    """A replaced log can score its first line, even if it repeats the prior log's contents."""
    extractor = LogRegexExtractor(
        type="log_regex", file="train.log", pattern=r"x=(?P<value>[0-9.]+)", select="first"
    )
    experiment = make_experiment(persistent=tmp_path, metric=Metric(extractor=extractor))
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    trial_dir.mkdir()
    attempt_dir.mkdir()
    log = trial_dir / "train.log"
    log.write_text("x=0.1\n", encoding="utf-8")
    original = log.stat()
    write_evidence_baseline(experiment, experiment.phases[0], trial_dir, attempt_dir, reused=True)
    (pin,) = (attempt_dir / EVIDENCE_PINS_DIR).iterdir()
    atomic_write_text(log, "x=0.25\n")
    atomic_write_text(log, "x=0.1\nx=0.8\n")

    assert pin.stat().st_ino == original.st_ino != log.stat().st_ino
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=read_evidence_baseline(attempt_dir),
    )
    assert run_extractor(ctx, extractor) == 0.1


def test_resumed_append_without_links_uses_separate_filesystem_creation_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier creation tick establishes continuity when hard links are unavailable."""
    extractor = LogRegexExtractor(
        type="log_regex", file="train.log", pattern=r"x=(?P<value>[0-9.]+)", select="first"
    )
    experiment = make_experiment(persistent=tmp_path, metric=Metric(extractor=extractor))
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    trial_dir.mkdir()
    attempt_dir.mkdir()
    log = trial_dir / "train.log"
    log.write_text("x=0.1\n", encoding="utf-8")
    births = {"train.log": 10, "evidence_baseline.json": 11}
    monkeypatch.setattr(Path, "hardlink_to", Mock(side_effect=OSError("no hard links")))
    monkeypatch.setattr(
        "phasesweep.evidence.baseline.file_birth_ns", lambda path, _: births[path.name]
    )
    monkeypatch.setattr(
        "phasesweep.evidence.evaluation.file_birth_ns", lambda path, _: births[path.name]
    )
    write_evidence_baseline(experiment, experiment.phases[0], trial_dir, attempt_dir, reused=True)
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=read_evidence_baseline(attempt_dir),
    )
    with log.open("a", encoding="utf-8") as stream:
        stream.write("x=0.8\n")
    assert run_extractor(ctx, extractor) == 0.8
    births["evidence_baseline.json"] = 10
    with pytest.raises(ExtractorError, match="could not establish the prior file's identity"):
        run_extractor(ctx, extractor)
    births["train.log"] = 12
    assert run_extractor(ctx, extractor) == 0.1


def test_releasing_evidence_pins_keeps_unrelated_paths(tmp_path: Path) -> None:
    """Retirement removes only pins and tolerates missing or symlinked pin directories."""
    attempt = tmp_path / "attempt"
    pins = attempt / EVIDENCE_PINS_DIR
    pins.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    evidence.touch()
    (pins / "0").hardlink_to(evidence)
    (pins / "1").mkdir()
    (pins / "notes").write_text("keep", encoding="utf-8")
    release_evidence_pins(attempt)
    assert {path.name for path in pins.iterdir()} == {"1", "notes"}
    assert evidence.is_file()

    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "0").touch()
    missing = tmp_path / "missing"
    release_evidence_pins(missing)
    missing.mkdir()
    (missing / EVIDENCE_PINS_DIR).symlink_to(other, target_is_directory=True)
    release_evidence_pins(missing)
    assert (other / "0").is_file()


@pytest.mark.parametrize("reused", [False, True], ids=["first-attempt", "resumed"])
def test_local_result_retarget_uses_the_final_files_creation_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reused: bool
) -> None:
    """Only a new result can supply evidence at an unfamiliar target in a reused directory."""
    extractor = JsonExtractor(type="json", path="latest/r.json", key="loss")
    experiment = make_experiment(persistent=tmp_path, metric=Metric(extractor=extractor))
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    attempt_dir.mkdir()
    for checkpoint in ("checkpoint-400", "checkpoint-500"):
        parent = trial_dir / checkpoint
        parent.mkdir(parents=True)
        (parent / "r.json").write_text('{"loss": 0.25}', encoding="utf-8")
    latest = trial_dir / "latest"
    if reused:
        latest.symlink_to("checkpoint-500")

    baseline = write_evidence_baseline(
        experiment, experiment.phases[0], trial_dir, attempt_dir, reused=reused
    )
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=read_evidence_baseline(attempt_dir),
    )
    latest.unlink(missing_ok=True)
    latest.symlink_to("checkpoint-400")
    result = latest / "r.json"
    if not reused:
        result.unlink()
        result.write_text('{"loss": 0.75}', encoding="utf-8")
    result.touch()
    monkeypatch.setattr("phasesweep.evidence.evaluation.file_birth_ns", lambda *_: None)
    if not reused:
        assert run_extractor(ctx, extractor) == 0.75
        return
    with pytest.raises(ExtractorError, match="creation during this attempt cannot be established"):
        run_extractor(ctx, extractor)
    atomic_write_text(result, '{"loss": 0.75}')
    monkeypatch.setattr(
        "phasesweep.evidence.evaluation.file_birth_ns", lambda *_: baseline.captured_at_ns + 1
    )
    assert run_extractor(ctx, extractor) == 0.75


def _interrupt_first_attempt(
    tmp_path: Path, body: str, metric: Metric, *, observe: list[Observation] | None = None
) -> Experiment:
    """Run the trainer under a runner process, SIGKILL the runner once the first attempt is ready.

    The trainer touches ``tmp_path / "first"`` and then sleeps, so the next run's recovery
    stops it and its trial reruns.

    :param Path tmp_path: Scratch directory.
    :param str body: Trainer body; the first attempt touches the marker and sleeps.
    :param Metric metric: The experiment's metric.
    :param list[Observation] | None observe: Optional observations, defaults to None.
    :return Experiment: The persistent one-trial experiment, its first trial left RUNNING.
    """
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_trainer(tmp_path / "trainer.py", body),
        metric=metric,
        observe=observe,
        n_trials=1,
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with runner_once(config, tmp_path / "first"):
        pass
    return experiment


_FIRST_THEN = """
import json, os, sys, time
from pathlib import Path
trial = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
first = Path({first!r})
if not first.exists():
{first_body}
    first.touch()
    time.sleep(60)
{rerun_body}
"""


def _trainer(tmp_path: Path, first_body: str, rerun_body: str) -> str:
    """Return a trainer whose first attempt runs ``first_body`` and waits to be stopped.

    :param Path tmp_path: Scratch directory holding the marker.
    :param str first_body: Statements of the first attempt, indented by four spaces.
    :param str rerun_body: Statements of every later attempt, unindented.
    :return str: The trainer body.
    """
    return _FIRST_THEN.format(
        first=str(tmp_path / "first"), first_body=first_body, rerun_body=rerun_body
    )


@pytest.mark.integration
def test_a_rerun_that_rewrites_no_result_fails_as_not_written_by_this_attempt(
    tmp_path: Path,
) -> None:
    """A rerun in its trial's directory that exits 0 without writing a result fails instead of scoring the earlier attempt's."""
    body = _trainer(
        tmp_path,
        '    (trial / "r.json").write_text(json.dumps({"loss": 0.1}))',
        "sys.exit(0)",
    )
    experiment = _interrupt_first_attempt(tmp_path, body, _JSON_METRIC)

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    first, rerun = phase_trials(experiment)
    assert rerun.user_attrs[RERUN_OF_ATTR] == 0
    assert rerun.user_attrs[TRIAL_DIR_ATTR] == first.user_attrs[TRIAL_DIR_ATTR]
    outcome = _parsed_trial_outcome(rerun.user_attrs[TRIAL_OUTCOME_ATTR])
    assert outcome is not None and outcome.outcome == "failure"
    assert f"{first.user_attrs[TRIAL_DIR_ATTR]}/r.json" in str(outcome.cause)
    assert "was not written by this attempt" in str(outcome.cause)


@pytest.mark.integration
def test_rerun_can_replace_an_unresolvable_observation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unresolved observation is reported before the trainer repairs it on the scheduled rerun."""
    body = _trainer(
        tmp_path,
        '    (trial / "observe.json").symlink_to("observe.json")',
        """
(trial / "observe.json").unlink()
(trial / "observe.json").write_text(json.dumps({"value": 8.0}))
(trial / "r.json").write_text(json.dumps({"loss": 1.0}))
""",
    )
    experiment = _interrupt_first_attempt(
        tmp_path,
        body,
        _JSON_METRIC,
        observe=[
            Observation(
                name="extra", extractor=JsonExtractor(type="json", path="observe.json", key="value")
            )
        ],
    )
    winner = run_experiment(experiment)["p"]
    assert winner.metric == 1.0
    assert winner.observed == {"extra": 8.0}
    assert "Could not read observe.json before launch because symlink" in caplog.text
    assert "observe.json -> observe.json could not be followed" in caplog.text


@pytest.mark.integration
def test_log_regex_reads_only_the_reruns_own_stdout(tmp_path: Path) -> None:
    """A rerun's ``log_regex`` metric reads its own captured stdout, never what the interrupted attempt printed."""
    body = _trainer(tmp_path, '    print("x=0.1", flush=True)', 'print("x=0.7")')
    metric = Metric(
        extractor=LogRegexExtractor(
            type="log_regex", pattern=r"x=(?P<value>[0-9.]+)", select="first"
        )
    )
    experiment = _interrupt_first_attempt(tmp_path, body, metric)

    assert run_experiment(experiment)["p"].metric == 0.7
    first, rerun = phase_trials(experiment)
    phase_dir = _phase_dir(experiment, "p")
    for trial, printed in ((first, "x=0.1\n"), (rerun, "x=0.7\n")):
        attempt_dir = _attempt_dir(phase_dir, trial.user_attrs[ATTEMPT_ID_ATTR])
        assert (attempt_dir / "stdout.log").read_text() == printed


_CHECKPOINTING = """
    for step in range(5):
        (trial / "ckpt.txt").write_text(str(step + 1))
"""
_RESUMING = """
start = int((trial / "ckpt.txt").read_text()) if (trial / "ckpt.txt").exists() else 0
for step in range(start, 8):
    (trial / "ckpt.txt").write_text(str(step + 1))
(trial / "r.json").write_text(json.dumps({"loss": float(start)}))
"""


@pytest.mark.integration
def test_a_killed_runs_trial_resumes_its_checkpoint_in_the_next_run(tmp_path: Path) -> None:
    """After a runner SIGKILL, the next run reruns the trial in its own directory, the trainer resumes its checkpoint, and each attempt keeps its own records."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )

    winner = run_experiment(experiment)["p"]
    first, rerun = phase_trials(experiment)
    # The rerun started from step 5, the checkpoint the stopped attempt left.
    assert winner.metric == 5.0 and winner.trial_number == rerun.number == 1
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number == 0
    trial_dir = Path(first.user_attrs[TRIAL_DIR_ATTR])
    assert Path(rerun.user_attrs[TRIAL_DIR_ATTR]) == trial_dir
    assert winner.trial_dir == trial_dir.name
    assert (trial_dir / "ckpt.txt").read_text() == "8"
    first_attempt = _attempt_dir(trial_dir.parent, first.user_attrs[ATTEMPT_ID_ATTR])
    rerun_attempt = _attempt_dir(trial_dir.parent, rerun.user_attrs[ATTEMPT_ID_ATTR])
    assert first_attempt != rerun_attempt
    stopped = read_attempt_lifecycle(first_attempt, expected_attempt_id=first_attempt.name)
    assert stopped is not None and stopped.ended_by == "cancelled"
    finished = read_attempt_lifecycle(rerun_attempt, expected_attempt_id=rerun_attempt.name)
    assert finished is not None and (finished.ended_by, finished.return_code) == ("trainer", 0)
    # The trainer sees one command across the chain: the directory's opener's names.
    assert (first_attempt / "command.txt").read_text() == (
        rerun_attempt / "command.txt"
    ).read_text()


@pytest.mark.integration
def test_an_interruption_before_a_reruns_launch_keeps_the_earlier_checkpoint(
    tmp_path: Path,
) -> None:
    """Recovery crosses a rerun interrupted before directory allocation to resume the original checkpoint."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    # Killing a separate controller leaves the claimed trial RUNNING without a directory.
    program = """
import os, signal, sys
from phasesweep import run_experiment
from phasesweep.config.io import load_experiment
from phasesweep.engine import phase

def interrupt_launch(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)

phase._trial_dir_for_attempt = interrupt_launch
run_experiment(load_experiment(sys.argv[1]))
"""
    runner = subprocess.run(
        [sys.executable, "-c", program, str(config)], capture_output=True, timeout=15, check=False
    )
    assert runner.returncode == -signal.SIGKILL, runner.stderr.decode()
    first, unlaunched = phase_trials(experiment)
    assert unlaunched.state == optuna.trial.TrialState.RUNNING
    assert unlaunched.user_attrs[RERUN_OF_ATTR] == first.number
    assert TRIAL_DIR_ATTR not in unlaunched.user_attrs

    winner = run_experiment(experiment)["p"]
    _, settled, rerun = phase_trials(experiment)
    outcome = _parsed_trial_outcome(settled.user_attrs[TRIAL_OUTCOME_ATTR])
    assert outcome is not None and outcome.outcome == "interrupted"
    assert rerun.user_attrs[RERUN_OF_ATTR] == unlaunched.number
    assert rerun.user_attrs[TRIAL_DIR_ATTR] == first.user_attrs[TRIAL_DIR_ATTR]
    assert winner.metric == 5.0 and winner.trial_number == rerun.number
    _verify_winner_evidence(experiment, experiment.phases[0], winner)


@pytest.mark.integration
def test_a_reruns_winner_verifies_in_its_owners_directory(tmp_path: Path) -> None:
    """A winning rerun's evidence is verified in the directory it continued, and a changed result there refuses it."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )
    winner = run_experiment(experiment)["p"]
    phase = experiment.phases[0]
    _verify_winner_evidence(experiment, phase, winner)

    owner_dir = _phase_dir(experiment, "p") / str(winner.trial_dir)
    (owner_dir / "r.json").write_text('{"loss": 5.0} ', encoding="utf-8")
    with pytest.raises(
        TrialEvidenceMissingError, match="recorded its objective evidence in 'r.json'"
    ):
        _verify_winner_evidence(experiment, phase, winner)


def test_each_attempt_in_a_directory_delivers_its_own_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed run's attempt and the next run's in-place rerun each keep their own MLflow binding and run: the stopped one ends KILLED, the accepted rerun's is left to its trainer, and nothing stays pending."""
    experiment, uploads, [stopped] = killed_sweep(tmp_path, monkeypatch, n_jobs=1)

    run_experiment(experiment)

    first, rerun = read_phase_trials(validate_ledger(experiment), "depth").trials
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number
    assert rerun.user_attrs[TRIAL_DIR_ATTR] == first.user_attrs[TRIAL_DIR_ATTR]
    resumed = rerun.user_attrs[ATTEMPT_ID_ATTR]
    bindings = {
        path.parent.name: session.read_binding(path.parent)
        for path in session.attempt_tracking_records(experiment, session.BINDING_FILE, None)
    }
    assert set(bindings) == {stopped, resumed}
    assert not (Path(first.user_attrs[TRIAL_DIR_ATTR]) / session.BINDING_FILE).exists()
    # The rerun runs in a later generation than the attempt it replaces, under its own run.
    assert bindings[stopped]["generation_id"] != bindings[resumed]["generation_id"]
    assert bindings[stopped]["run_id"] != bindings[resumed]["run_id"]
    # PhaseSweep finishes the stopped attempt's run; an accepted attempt's run is the
    # trainer's to end, so its record carries the result and no status.
    finishes = {
        record["binding"]["attempt_id"]: record["finish"]
        for record in uploads
        if str(record["name"]).startswith("attempt-")
    }
    assert finishes == {stopped: "KILLED", resumed: None}
    assert publication.sync_tracking(experiment)["pending"] == 0


def _interrupted(
    number: int, trial_dir: Path, attempt_id: str, **attrs: object
) -> optuna.trial.FrozenTrial:
    """Return a finished interrupted trial that recorded its directory and attempt."""
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.FAIL,
        user_attrs={
            TRIAL_OUTCOME_ATTR: trial_outcome(number + 1, "interrupted"),
            TRIAL_DIR_ATTR: str(trial_dir),
            ATTEMPT_ID_ATTR: attempt_id,
            **attrs,
        },
    )


def _trials(*trials: optuna.trial.FrozenTrial) -> dict[int, optuna.trial.FrozenTrial]:
    """Number hand-built trials as a study would, in the order given."""
    for number, trial in enumerate(trials):
        trial.number = number
    return {trial.number: trial for trial in trials}


def _waiting_rerun(source: int) -> optuna.trial.FrozenTrial:
    """Return the WAITING rerun that recovery schedules for ``source``."""
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.WAITING, user_attrs={RERUN_OF_ATTR: source}
    )


def test_a_rerun_refuses_a_directory_whose_source_cleanup_is_unconfirmed(tmp_path: Path) -> None:
    """A rerun may enter its source's directory only once that attempt's cleanup is confirmed, so two processes never write one directory."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    source_dir = _trial_dir_for(experiment, "p", 0, attempt_id="a0")
    source_attempt = _attempt_dir(source_dir.parent, "a0")
    source_attempt.mkdir(parents=True)
    write_attempt_lifecycle(source_attempt, attempt_id="a0", state="launching")
    write_trial_identity(source_attempt, attempt_id="a0", pid=reaped_pid(), starttime=1)
    trials = _trials(_interrupted(0, source_dir, "a0"), _waiting_rerun(0))

    with pytest.raises(ProcessCleanupUncertainError, match="may still be running"):
        _trial_dir_for_attempt(experiment, "p", trials, trials[1], attempt_id="a1")

    write_attempt_lifecycle(
        source_attempt,
        attempt_id="a0",
        state="exited",
        ended_by="cancelled",
        cleanup_confirmed=True,
        duration_seconds=1.0,
    )
    assert _trial_dir_for_attempt(
        experiment, "p", trials, trials[1], attempt_id="a1"
    ) == TrialDirectory(source_dir, 0, "a0")


def test_parallel_reruns_each_use_their_own_sources_directory(tmp_path: Path) -> None:
    """Reruns of different sources each continue their own source's directory, and an interrupted rerun's rerun keeps the original and is bound to the trial that opened it."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    dirs = {
        name: _trial_dir_for(experiment, "p", number, attempt_id=name)
        for number, name in ((0, "a0"), (1, "a1"))
    }
    for attempt_id in ("a0", "a1", "a2"):
        attempt_dir = _attempt_dir(dirs["a0"].parent, attempt_id)
        attempt_dir.mkdir(parents=True)
        write_attempt_lifecycle(attempt_dir, attempt_id=attempt_id, state="allocated")
    trials = _trials(
        _interrupted(0, dirs["a0"], "a0"),
        _interrupted(1, dirs["a1"], "a1"),
        _interrupted(2, dirs["a0"], "a2", **{RERUN_OF_ATTR: 0}),
        _waiting_rerun(1),
        _waiting_rerun(2),
    )

    third = _trial_dir_for_attempt(experiment, "p", trials, trials[3], attempt_id="a3")
    fourth = _trial_dir_for_attempt(experiment, "p", trials, trials[4], attempt_id="a4")
    assert third == TrialDirectory(dirs["a1"], 1, "a1")
    assert fourth == TrialDirectory(dirs["a0"], 0, "a0")
    # Winner verification walks the whole chain to the trial that opened the directory.
    assert _expected_trial_dir_name(trials, trials[4]) == dirs["a0"].name


_PRINTS_JSON = """
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--out")
parser.add_argument("--x", type=int)
print(json.dumps({"loss": parser.parse_args().x / 10}))
"""


@pytest.mark.integration
def test_a_json_objective_over_stdout_reads_and_verifies_the_captured_stream(
    tmp_path: Path,
) -> None:
    """A ``json`` objective at ``stdout.log`` scores the attempt's captured stdout, and its winner verifies against the same file in later runs."""
    metric = Metric(extractor=JsonExtractor(type="json", path="stdout.log", key="loss"))
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_trainer(tmp_path / "trainer.py", _PRINTS_JSON),
        metric=metric,
    )

    winner = run_experiment(experiment)["p"]
    _verify_winner_evidence(experiment, experiment.phases[0], winner)
    assert run_experiment(experiment)["p"].metric == winner.metric
    trial = next(t for t in phase_trials(experiment) if t.number == winner.trial_number)
    attempt_dir = _attempt_dir(_phase_dir(experiment, "p"), trial.user_attrs[ATTEMPT_ID_ATTR])
    assert (attempt_dir / "stdout.log").read_text() == f'{{"loss": {winner.metric}}}\n'


def test_a_dotted_captured_stream_spelling_scores_the_captured_stream(tmp_path: Path) -> None:
    """``log_regex`` with ``file: ./stdout.log`` reads the attempt's captured stdout."""
    metric = Metric(
        extractor=LogRegexExtractor(
            type="log_regex", file="./stdout.log", pattern=r"--x[= ](?P<value>[0-9]+)"
        )
    )
    experiment = make_experiment(workdir=tmp_path / "runs", metric=metric, n_trials=1)

    winner = run_experiment(experiment)["p"]
    assert winner.metric == float(winner.params["x"])


def test_in_memory_runs_never_reuse_a_directory(tmp_path: Path) -> None:
    """Two ``storage: null`` runs restart trial numbers but open distinct directories, named by their attempts."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=0)
    experiment = make_experiment(workdir=tmp_path / "runs", trainer=trainer, n_trials=1)

    run_experiment(experiment)
    run_experiment(experiment)

    opened = sorted(_phase_dir(experiment, "p").glob("trial_*"))
    assert len(opened) == 2
    assert all(directory.name.startswith("trial_00000__") for directory in opened)


@pytest.mark.integration
@pytest.mark.parametrize("directory_state", ["missing", "working_link"])
def test_rerun_keeps_the_original_directory_identity_when_its_contents_move(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, directory_state: str
) -> None:
    """A missing directory restarts the same draw at its path, and a working directory link still resumes."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )
    (first,) = phase_trials(experiment)
    directory = Path(first.user_attrs[TRIAL_DIR_ATTR])
    saved = directory.with_name("saved-directory")
    directory.rename(saved)
    if directory_state == "working_link":
        directory.symlink_to(saved.name)

    winner = run_experiment(experiment)["p"]
    first, rerun = phase_trials(experiment)
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number
    assert rerun.params == first.params
    assert rerun.user_attrs[ATTEMPT_ID_ATTR] != first.user_attrs[ATTEMPT_ID_ATTR]
    assert Path(rerun.user_attrs[TRIAL_DIR_ATTR]) == directory
    assert winner.trial_dir == directory.name
    assert winner.trial_number == rerun.number
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "without its previous local checkpoints" in record.getMessage()
    ]
    if directory_state == "working_link":
        assert winner.metric == 5.0
        assert directory.readlink() == Path(saved.name)
        assert warnings == []
    else:
        assert winner.metric == 0.0
        assert directory.is_dir() and not directory.is_symlink()
        assert (saved / "ckpt.txt").read_text() == "5"
        (warning,) = warnings
        assert str(directory) in warning
        assert (
            f"Trial {rerun.number} reruns trial {first.number} with the same parameters" in warning
        )
        assert "is missing" in warning

    assert run_experiment(experiment)["p"].metric == winner.metric


@pytest.mark.integration
@pytest.mark.parametrize("target", ["missing-target", "checkpoint-file"])
def test_a_rerun_refuses_a_trial_directory_link_it_cannot_follow(
    tmp_path: Path, target: str
) -> None:
    """A dangling or non-directory link at a rerun's trial path stops the run before launch, naming the link and its target, and leaves the link alone; deleting it reruns from step 0."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )
    (first,) = phase_trials(experiment)
    directory = Path(first.user_attrs[TRIAL_DIR_ATTR])
    directory.rename(directory.with_name("saved-directory"))
    if target == "checkpoint-file":
        directory.with_name(target).write_text("keep")
    directory.symlink_to(target)

    with pytest.raises(PhaseSweepError) as refused:
        run_experiment(experiment)

    assert f"symlink {directory} -> {target}" in str(refused.value)
    assert refused.value.action is OperatorAction.RESTORE_TREE
    assert refused.value.remediation == (
        f"Restore the link's target, or delete the link {directory} to rerun this trial from "
        "step 0."
    )
    assert directory.readlink() == Path(target)
    assert not list(directory.parent.glob(f".{directory.name}*"))

    directory.unlink()
    assert run_experiment(experiment)["p"].metric == 0.0
