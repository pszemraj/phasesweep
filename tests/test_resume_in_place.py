"""A rerun continues in its trial's directory and is scored only from files its own attempt wrote."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.evidence import _verify_winner_evidence
from phasesweep.engine.paths import _attempt_dir, _phase_dir, _trial_dir_for
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
)
from phasesweep.engine.trial_dirs import TrialDirectory, _trial_dir_for_attempt
from phasesweep.errors import ProcessCleanupUncertainError
from phasesweep.evidence.baseline import read_evidence_baseline, write_evidence_baseline
from phasesweep.evidence.evaluation import ExtractorError, run_extractor
from phasesweep.evidence.models import JsonExtractor, LogRegexExtractor
from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle
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
from tests.recovery_helpers import write_trial_identity

_JSON_METRIC = Metric(extractor=JsonExtractor(type="json", path="r.json", key="loss"))
_RUNNER = (
    "import sys\n"
    "from phasesweep import run_experiment\n"
    "from phasesweep.config.io import load_experiment\n"
    "run_experiment(load_experiment(sys.argv[1]))\n"
)


def test_a_value_file_left_before_launch_fails_as_not_written_by_this_attempt(
    tmp_path: Path,
) -> None:
    """A result file unchanged since before the attempt ran is refused; the attempt's own rewrite of it reads."""
    experiment = make_experiment(persistent=tmp_path, metric=_JSON_METRIC)
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    trial_dir.mkdir()
    attempt_dir.mkdir()
    (trial_dir / "r.json").write_text('{"loss": 0.1}', encoding="utf-8")

    baseline = write_evidence_baseline(experiment, experiment.phases[0], trial_dir, attempt_dir)
    assert read_evidence_baseline(attempt_dir) == baseline and set(baseline) == {"r.json"}
    ctx = dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=baseline,
    )
    with pytest.raises(ExtractorError, match="was not written by this attempt"):
        run_extractor(ctx, _JSON_METRIC.extractor)

    (trial_dir / "r.json").write_text('{"loss": 0.25}', encoding="utf-8")
    assert run_extractor(ctx, _JSON_METRIC.extractor) == 0.25


def _interrupt_first_attempt(tmp_path: Path, body: str, metric: Metric) -> Experiment:
    """Run the trainer under a runner process, SIGKILL the runner once the first attempt is ready.

    The trainer touches ``tmp_path / "first"`` and then sleeps, so the next run's recovery
    stops it and its trial reruns.

    :param Path tmp_path: Scratch directory.
    :param str body: Trainer body; the first attempt touches the marker and sleeps.
    :param Metric metric: The experiment's metric.
    :return Experiment: The persistent one-trial experiment, its first trial left RUNNING.
    """
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_trainer(tmp_path / "trainer.py", body),
        metric=metric,
        n_trials=1,
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with (tmp_path / "runner.log").open("w") as output:
        runner = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, str(config)], stdout=output, stderr=output
        )
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "first").exists():
                assert runner.poll() is None and time.monotonic() < deadline
                time.sleep(0.02)
        finally:
            runner.kill()
            runner.wait(timeout=5)
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
    assert "r.json` in trial directory" in str(outcome.cause)
    assert "was not written by this attempt" in str(outcome.cause)


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


def test_a_retried_grid_cell_gets_a_fresh_directory(tmp_path: Path) -> None:
    """A failed fixed configuration retried by a raised target reruns from scratch in a directory of its own."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=1)

    def attempt(n_trials: int) -> Experiment:
        phase = Phase(name="p", n_trials=n_trials, fixed_overrides={"x": 1})
        return make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(attempt(1))
    assert run_experiment(attempt(2))["p"].metric == 1.0

    failed, retry = phase_trials(attempt(2))
    assert retry.user_attrs[TRIAL_DIR_ATTR] != failed.user_attrs[TRIAL_DIR_ATTR]
    assert Path(retry.user_attrs[TRIAL_DIR_ATTR]).name.startswith("trial_00001__")


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
    """Reruns of different sources each continue their own source's directory, and an interrupted rerun's rerun keeps the original."""
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


def test_in_memory_runs_never_reuse_a_directory(tmp_path: Path) -> None:
    """Two ``storage: null`` runs restart trial numbers but open distinct directories, named by their attempts."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=0)
    experiment = make_experiment(workdir=tmp_path / "runs", trainer=trainer, n_trials=1)

    run_experiment(experiment)
    run_experiment(experiment)

    opened = sorted(_phase_dir(experiment, "p").glob("trial_*"))
    assert len(opened) == 2
    assert all(directory.name.startswith("trial_00000__") for directory in opened)
