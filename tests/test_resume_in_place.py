"""A rerun continues in its trial's directory and is scored only from files its own attempt wrote."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Observation
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.evidence import _earlier_attempts, _verify_winner_evidence
from phasesweep.engine.ledger import read_phase_trials, validate_ledger
from phasesweep.engine.paths import _attempt_dir, _phase_dir, _trial_dir_for
from phasesweep.engine.selection import NoFeasibleTrialError, _ranked_revision_sources
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
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
from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
from phasesweep.evidence.baseline import (
    EVIDENCE_BASELINE_FILE,
    read_evidence_baseline,
    write_evidence_baseline,
)
from phasesweep.evidence.evaluation import ExtractorError, TrialContext, run_extractor
from phasesweep.evidence.mlflow import observation_id
from phasesweep.evidence.models import JsonExtractor, LogRegexExtractor, MlflowExtractor
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import read_attempt_lifecycle, write_attempt_lifecycle
from phasesweep.tracking import mlflow as mlflow_worker
from phasesweep.tracking import publication, session
from tests.conftest import (
    make_experiment,
    make_trial_context,
    phase_trials,
    reaped_pid,
    study_name,
    trial_outcome,
    write_experiment_config,
    write_flaky_trainer,
    write_trainer,
)
from tests.recovery_helpers import runner_once, write_trial_identity
from tests.tracking_helpers import killed_sweep, mlflow_binding, mlflow_run

_JSON_METRIC = Metric(extractor=JsonExtractor(type="json", path="r.json", key="loss"))


def _age(path: Path) -> None:
    """Date a file an hour back, as an earlier attempt's write would be.

    :param Path path: File to backdate.
    """
    earlier = path.stat().st_mtime_ns - 3_600_000_000_000
    os.utime(path, ns=(earlier, earlier))


def _resumed_context(
    tmp_path: Path, extractor: JsonExtractor | LogRegexExtractor
) -> tuple[Path, TrialContext]:
    """Record a reused directory's baseline over files an earlier attempt left.

    :param Path tmp_path: Scratch directory holding ``trial`` and ``attempt``; ``trial`` is
        already populated.
    :param JsonExtractor | LogRegexExtractor extractor: The metric extractor.
    :return tuple[Path, TrialContext]: The trial directory and the rerun's context.
    """
    experiment = make_experiment(persistent=tmp_path, metric=Metric(extractor=extractor))
    trial_dir, attempt_dir = tmp_path / "trial", tmp_path / "attempt"
    attempt_dir.mkdir()
    for path in trial_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            _age(path)
    write_evidence_baseline(experiment, experiment.phases[0], trial_dir, attempt_dir, reused=True)
    return trial_dir, dataclasses.replace(
        make_trial_context(tmp_path),
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        baseline=read_evidence_baseline(attempt_dir),
    )


@pytest.mark.parametrize("touch_only", [False, True], ids=["rewrite", "touch"])
def test_a_rewritten_value_file_is_fresh_and_an_untouched_one_is_stale(
    tmp_path: Path, touch_only: bool
) -> None:
    """Score an identical rewrite or touch while refusing untouched and relinked results.

    :param Path tmp_path: Scratch directory for the trial and attempt.
    :param bool touch_only: Update the file timestamps without writing its bytes.
    """
    (tmp_path / "trial").mkdir()
    result = tmp_path / "trial" / "r.json"
    result.write_text('{"loss": 0.1}', encoding="utf-8")
    trial_dir, ctx = _resumed_context(tmp_path, _JSON_METRIC.extractor)
    stale = "was not written by this attempt: its metadata shows no fresh write"
    with pytest.raises(ExtractorError, match=stale):
        run_extractor(ctx, _JSON_METRIC.extractor)

    os.link(result, trial_dir / "r.bak")
    result.unlink()
    os.link(trial_dir / "r.bak", result)
    with pytest.raises(ExtractorError, match=stale):
        run_extractor(ctx, _JSON_METRIC.extractor)

    if touch_only:
        result.touch()
    else:
        result.write_text('{"loss": 0.1}', encoding="utf-8")
    assert run_extractor(ctx, _JSON_METRIC.extractor) == 0.1


def test_a_same_size_copy_preserving_mtime_is_fresh(tmp_path: Path) -> None:
    """Score a copy that changes only the recorded file's change time.

    :param Path tmp_path: Scratch directory for the trial and attempt.
    """
    result = tmp_path / "trial" / "r.json"
    result.parent.mkdir()
    result.write_text('{"loss": 0.1}', encoding="utf-8")
    _, ctx = _resumed_context(tmp_path, _JSON_METRIC.extractor)
    prior = result.stat()

    source = tmp_path / "source.json"
    source.write_text('{"loss": 0.2}', encoding="utf-8")
    os.utime(source, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    shutil.copy2(source, result)
    copied = result.stat()
    assert (copied.st_size, copied.st_mtime_ns) == (prior.st_size, prior.st_mtime_ns)
    assert run_extractor(ctx, _JSON_METRIC.extractor) == 0.2


def test_a_resumed_log_is_scored_from_every_line(tmp_path: Path) -> None:
    """A trainer log kept across attempts selects over the earlier attempt's lines too once the rerun appends, and fails as not written by this attempt while untouched."""
    extractor = LogRegexExtractor(
        type="log_regex", file="train.log", pattern=r"x=(?P<value>[0-9.]+)", select="min"
    )
    (tmp_path / "trial").mkdir()
    log = tmp_path / "trial" / "train.log"
    log.write_text("x=0.5\nx=0.1\n", encoding="utf-8")
    _, ctx = _resumed_context(tmp_path, extractor)
    with pytest.raises(ExtractorError, match="was not written by this attempt"):
        run_extractor(ctx, extractor)

    with log.open("a", encoding="utf-8") as stream:
        stream.write("x=0.7\n")
    provenance: JsonObject = {}
    assert run_extractor(ctx, extractor, provenance=provenance) == 0.1
    assert provenance["source"] == {
        "kind": "file",
        "path": "train.log",
        "sha256": hashlib.sha256(b"x=0.5\nx=0.1\nx=0.7\n").hexdigest(),
        "size_bytes": 18,
        "matched_line": 2,
        "match_count": 3,
    }


def test_a_retargeted_link_uses_change_time_when_modification_time_is_inconclusive(
    tmp_path: Path,
) -> None:
    """An unfamiliar target can share the launch tick if its change time reaches that tick.

    :param Path tmp_path: Scratch directory for the trial and attempt.
    """
    extractor = JsonExtractor(type="json", path="latest/r.json", key="loss")
    for checkpoint in ("checkpoint-400", "checkpoint-500"):
        parent = tmp_path / "trial" / checkpoint
        parent.mkdir(parents=True)
        (parent / "r.json").write_text('{"loss": 0.25}', encoding="utf-8")
    (tmp_path / "trial" / "latest").symlink_to("checkpoint-500")
    trial_dir, ctx = _resumed_context(tmp_path, extractor)
    latest = trial_dir / "latest"
    latest.unlink()
    latest.symlink_to("checkpoint-400")
    result = latest / "r.json"
    prior = result.stat()
    baseline_path = ctx.attempt_dir / EVIDENCE_BASELINE_FILE
    launch_stamp = time.time_ns() + 3_600_000_000_000
    os.utime(baseline_path, ns=(launch_stamp, launch_stamp))
    os.utime(result, ns=(launch_stamp, launch_stamp))
    old_target = result.stat()
    assert old_target.st_ctime_ns < old_target.st_mtime_ns == baseline_path.stat().st_mtime_ns
    with pytest.raises(ExtractorError, match="last changed before this attempt started"):
        run_extractor(ctx, extractor)

    source = tmp_path / "source.json"
    source.write_text('{"loss": 0.75}', encoding="utf-8")
    os.utime(source, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    shutil.copy2(source, result)
    copied = result.stat()
    os.utime(baseline_path, ns=(copied.st_ctime_ns, copied.st_ctime_ns))
    assert copied.st_mtime_ns < baseline_path.stat().st_mtime_ns == copied.st_ctime_ns
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
def test_rerun_can_replace_an_unresolvable_observation(tmp_path: Path) -> None:
    """An observation path left as a link loop reads normally once the rerun replaces it."""
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
@pytest.mark.parametrize("directory_recorded", [False, True])
def test_recovery_of_an_interrupted_directory_reset(
    tmp_path: Path, directory_recorded: bool
) -> None:
    """Recovery uses a restored checkpoint until the reset's own directory reaches the ledger."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )
    (first,) = phase_trials(experiment)
    directory = Path(first.user_attrs[TRIAL_DIR_ATTR])
    saved = directory.rename(directory.with_name("saved-directory"))
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    program = """
import os, signal, sys
from phasesweep import run_experiment
from phasesweep.config.io import load_experiment
from phasesweep.engine import phase

def interrupt_launch(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)

if sys.argv[2] == "True":
    phase.prepare_trainer_input = interrupt_launch
else:
    register = phase._register_active_attempt
    def interrupt_registration(*args, **kwargs):
        register(*args, **kwargs)
        interrupt_launch()
    phase._register_active_attempt = interrupt_registration
run_experiment(load_experiment(sys.argv[1]))
"""
    runner = subprocess.run(
        [sys.executable, "-c", program, str(config), str(directory_recorded)],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert runner.returncode == -signal.SIGKILL, runner.stderr.decode()
    first, unlaunched = phase_trials(experiment)
    assert unlaunched.state == optuna.trial.TrialState.RUNNING
    assert unlaunched.user_attrs[RERUN_OF_ATTR] == first.number
    assert (TRIAL_DIR_ATTR in unlaunched.user_attrs) is directory_recorded
    saved.rename(directory)

    winner = run_experiment(experiment)["p"]
    _, settled, rerun = phase_trials(experiment)
    outcome = _parsed_trial_outcome(settled.user_attrs[TRIAL_OUTCOME_ATTR])
    assert outcome is not None and outcome.outcome == "interrupted"
    assert rerun.user_attrs[RERUN_OF_ATTR] == unlaunched.number
    opener = settled if directory_recorded else first
    assert rerun.user_attrs[TRIAL_DIR_ATTR] == opener.user_attrs[TRIAL_DIR_ATTR]
    assert winner.metric == (0.0 if directory_recorded else 5.0)
    assert winner.trial_number == rerun.number
    assert (directory / "ckpt.txt").read_text() == ("5" if directory_recorded else "8")
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


@pytest.mark.parametrize("directory_missing", [False, True])
def test_each_attempt_delivers_its_own_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_missing: bool
) -> None:
    """Each attempt delivers its own run, and extraction includes only histories from the same directory."""
    experiment, uploads, [stopped] = killed_sweep(tmp_path, monkeypatch, n_jobs=1)
    (first,) = phase_trials(experiment, "depth")
    if directory_missing:
        directory = Path(first.user_attrs[TRIAL_DIR_ATTR])
        directory.rename(directory.with_name("saved-directory"))

    run_experiment(experiment)

    first, rerun = read_phase_trials(
        validate_ledger(experiment), study_name(experiment, "depth")
    ).trials
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number
    assert (
        rerun.user_attrs[TRIAL_DIR_ATTR] != first.user_attrs[TRIAL_DIR_ATTR]
    ) is directory_missing
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
    capture = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])["remote_capture"]
    assert capture["earlier_bindings"] == ([] if directory_missing else [bindings[stopped]])
    # PhaseSweep finishes the stopped attempt's run; an accepted attempt's run is the
    # trainer's to end, so its record carries the result and no status.
    finishes = {
        record["binding"]["attempt_id"]: record["finish"]
        for record in uploads
        if str(record["name"]).startswith("attempt-")
    }
    assert finishes == {stopped: "KILLED", resumed: None}
    assert publication.sync_tracking(experiment)["pending"] == 0


def test_a_resumed_trials_mlflow_objective_reads_its_attempts_merged_history() -> None:
    """Each selector keeps the previous attempt's prefix and discards its superseded steps, including steps past the retry's final one."""
    current = mlflow_binding()
    earlier = {**current, "run_id": "stopped", "attempt_id": "d" * 32}
    loss = {step: 1.0 + 0.1 * (step - 5) ** 2 for step in range(1, 11)}
    logged = {
        "stopped": [*((step, loss[step]) for step in range(1, 8)), (12, 0.1)],
        "native-id": [(7, 9.0), *((step, loss[step]) for step in range(8, 11))],
    }
    client = SimpleNamespace(
        get_run=lambda run_id: mlflow_run(earlier if run_id == "stopped" else current),
        get_metric_history=lambda run_id, _key: [
            SimpleNamespace(value=value, step=step, timestamp=step)
            for step, value in logged[run_id]
        ],
    )
    selectors = {
        "min": MlflowExtractor(type="mlflow", metric_key="loss", select="min"),
        "last": MlflowExtractor(type="mlflow", metric_key="loss", select="last"),
        "step": MlflowExtractor(type="mlflow", metric_key="loss", select="step", step=7),
    }
    query = {
        "revision": 1,
        "objective": selectors["min"].model_dump(),
        "constraints": {},
        "selections": {observation_id(s): s.model_dump() for s in selectors.values()},
        "gates": [],
    }
    capture = mlflow_worker._capture(
        client,
        {
            "binding": current,
            "earlier": [earlier],
            "query": query,
            "settings": {"timeout_seconds": 1, "poll_seconds": 0.01},
            "timeout_at": time.monotonic() + 1,
        },
    )
    chosen = {
        name: capture["observations"][observation_id(spec)] for name, spec in selectors.items()
    }
    assert (chosen["min"]["value"], chosen["min"]["step"]) == (1.0, 5)
    assert (chosen["last"]["value"], chosen["last"]["step"]) == (loss[10], 10)
    assert chosen["step"]["value"] == 9.0
    assert capture["earlier_bindings"] == [earlier]


@pytest.mark.parametrize("gate_requires_metric", [False, True], ids=["constraint", "gate-too"])
def test_a_rerun_can_use_a_constraint_logged_once_on_the_first_attempt(
    gate_requires_metric: bool,
) -> None:
    """A step-zero constraint survives a retry while a gate still requires current evidence."""
    binding = mlflow_binding()
    prior = {**binding, "attempt_id": "d" * 32, "run_id": "earlier"}
    objective = MlflowExtractor(type="mlflow", metric_key="loss", select="min")
    count = MlflowExtractor(type="mlflow", metric_key="n_params", select="last")
    runs = {
        binding["run_id"]: mlflow_run(binding, metrics={"loss": 0.4}),
        prior["run_id"]: mlflow_run(prior, status="FAILED", metrics={"loss": 0.05, "n_params": 42}),
    }
    histories = {
        (binding["run_id"], "loss"): [SimpleNamespace(value=0.4, step=7, timestamp=3)],
        (prior["run_id"], "loss"): [
            SimpleNamespace(value=0.2, step=2, timestamp=1),
            SimpleNamespace(value=0.05, step=8, timestamp=2),
        ],
        (prior["run_id"], "n_params"): [SimpleNamespace(value=42, step=0, timestamp=1)],
    }
    client = SimpleNamespace(
        get_run=runs.__getitem__, get_metric_history=lambda run, key: histories[run, key]
    )
    request = {
        "binding": binding,
        "earlier": [prior],
        "query": {
            "revision": 1,
            "objective": objective.model_dump(),
            "constraints": {"size": count.model_dump()},
            "selections": {observation_id(s): s.model_dump() for s in (objective, count)},
            "gates": [{"type": "mlflow_required", "metrics": ["n_params"]}]
            if gate_requires_metric
            else [],
        },
        "settings": {"timeout_seconds": 1, "poll_seconds": 0.01},
        "timeout_at": time.monotonic() + 1,
    }
    if gate_requires_metric:
        with pytest.raises(ValueError, match="without metric 'n_params'"):
            mlflow_worker._capture(client, request)
    else:
        capture = mlflow_worker._capture(client, request)
        assert capture["observations"][observation_id(count)] == {
            "value": 42.0,
            "step": 0,
            "timestamp": 1,
        }
        assert capture["observations"][observation_id(objective)]["value"] == 0.2
        assert capture["earlier_bindings"] == [prior]


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


@pytest.mark.parametrize("ancestor", [0, 1, 2])
def test_rerun_directory_rejects_a_source_that_is_not_earlier(ancestor: int) -> None:
    """Every directory lineage edge must point to an existing, strictly earlier trial."""
    trials = _trials(_waiting_rerun(ancestor), _waiting_rerun(0))

    with pytest.raises(ProcessCleanupUncertainError, match="not an earlier trial") as refused:
        _expected_trial_dir_name(trials, trials[1])

    assert refused.value.action is OperatorAction.RESTORE_LEDGER


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
    source_dir.mkdir()
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
    for directory in dirs.values():
        directory.mkdir()
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
    assert _expected_trial_dir_name(trials, trials[4]) == dirs["a0"].name


def test_tracking_and_code_revision_ancestry_stop_at_the_directory_opener(tmp_path: Path) -> None:
    """A fresh directory cuts off earlier attempts in live and recorded continuations."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    original = _trial_dir_for(experiment, "p", 0, attempt_id="a0")
    fresh = _trial_dir_for(experiment, "p", 1, attempt_id="a1")
    fresh.mkdir(parents=True)
    trials = _trials(
        _interrupted(0, original, "a0", **{GENERATION_ID_ATTR: "g0"}),
        _interrupted(1, fresh, "a1", **{RERUN_OF_ATTR: 0, GENERATION_ID_ATTR: "g1"}),
        _interrupted(2, fresh, "a2", **{RERUN_OF_ATTR: 1, GENERATION_ID_ATTR: "g2"}),
        _waiting_rerun(2),
        _waiting_rerun(3),
    )

    assert _earlier_attempts(trials, trials[1]) == ()
    assert _earlier_attempts(trials, trials[2]) == (("g1", "a1"),)
    assert _earlier_attempts(trials, trials[4]) == (("g1", "a1"), ("g2", "a2"))
    study = optuna.create_study()
    study.add_trials(trials.values())
    assert _ranked_revision_sources(study, list(trials.values()), [trials[1], trials[2]]) == {
        2: ["g1"]
    }
    assert _trial_dir_for_attempt(
        experiment, "p", trials, trials[4], attempt_id="a4", recovered=frozenset({2})
    ) == TrialDirectory(fresh, 1, "a1")


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
@pytest.mark.parametrize(
    "target", [None, "saved-directory", "missing-target", "checkpoint-file", "loop"]
)
def test_rerun_uses_the_previous_directory_when_it_is_available(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, target: str | None
) -> None:
    """A usable directory link keeps its checkpoint, and a missing directory or unusable link starts a fresh attempt."""
    experiment = _interrupt_first_attempt(
        tmp_path, _trainer(tmp_path, _CHECKPOINTING, _RESUMING), _JSON_METRIC
    )
    (first,) = phase_trials(experiment)
    directory = Path(first.user_attrs[TRIAL_DIR_ATTR])
    saved = directory.rename(directory.with_name("saved-directory"))
    if target == "checkpoint-file":
        directory.with_name(target).write_text("keep")
    if target == "loop":
        target = directory.name
    if target is not None:
        directory.symlink_to(target)

    winner = run_experiment(experiment)["p"]
    first, rerun = phase_trials(experiment)
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number
    assert rerun.params == first.params
    assert rerun.user_attrs[ATTEMPT_ID_ATTR] != first.user_attrs[ATTEMPT_ID_ATTR]
    rerun_dir = Path(rerun.user_attrs[TRIAL_DIR_ATTR])
    assert winner.trial_dir == rerun_dir.name
    assert winner.trial_number == rerun.number
    warnings = [
        record.getMessage() for record in caplog.records if "from step 0" in record.getMessage()
    ]
    if target == saved.name:
        assert rerun_dir == directory
        assert winner.metric == 5.0
        assert warnings == []
    else:
        assert rerun_dir == _trial_dir_for(
            experiment, "p", rerun.number, attempt_id=rerun.user_attrs[ATTEMPT_ID_ATTR]
        )
        assert winner.metric == 0.0
        assert (saved / "ckpt.txt").read_text() == "5"
        (warning,) = warnings
        assert f"Trial {rerun.number} reruns trial {first.number} from step 0" in warning
        if target is None:
            assert f"{directory} is missing" in warning
            assert not directory.exists()
        else:
            assert f"symlink {directory} -> {target}" in warning
    if target is not None:
        assert directory.readlink() == Path(target)
    if target == "checkpoint-file":
        assert directory.read_text() == "keep"

    assert run_experiment(experiment)["p"].metric == winner.metric
