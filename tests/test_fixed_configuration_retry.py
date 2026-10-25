"""A fixed configuration is retried by raising ``n_trials``, one attempt per failed one.

A fixed configuration draws nothing, so every attempt runs the same trial. ``n_trials`` counts
terminal attempts, failed ones included, which would leave a failed fixed configuration stuck
unless raising the target may replace its failures. It may, by one attempt per failure while
nothing completed, and never once something did. An interrupted attempt never counts: it reruns
at the same target.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.cli import cli as cli_main
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Sampler, fixed_configuration_target_refusal
from phasesweep.engine.errors import TrialTargetRefusedError
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.selection import (
    NoFeasibleTrialError,
    PhaseAbortRefusedError,
    PhaseTrialsFailedError,
    PhaseTrialsInfeasibleError,
    ReplayWithoutFeasibleTrialError,
)
from phasesweep.engine.state import PHASE_FINGERPRINT_ATTR, TRIAL_TARGET_ATTR
from phasesweep.engine.study_policy import _validate_trial_target
from phasesweep.evidence.models import LogRegexExtractor
from phasesweep.mcp import runner as mcp_runner
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import (
    assert_advises_target,
    invoke_cli_boundary,
    make_experiment,
    phase_trials,
    write_flaky_trainer,
)
from tests.experiments import write_experiment
from tests.recovery_helpers import fabricate_stale_trial

_FRESH_THREE = "Phase 'p': n_trials=3 with an empty search_space would run 3 identical trials."
_FRESH_THREE_REMEDY = "Use n_trials: 1 for a fixed configuration, or repeats to measure seed noise."
_WARNING = (
    "phase 'p' is a fixed configuration: n_trials above 1 only retries failed attempts, and a "
    "launch refuses more than one past the failed ones."
)


def _call_count(calls: Path) -> int:
    """Return how many times the trainer ran."""
    return len(calls.read_text().splitlines()) if calls.exists() else 0


def _fixed(
    tmp_path: Path,
    trainer: Path,
    n_trials: int,
    *,
    max_consecutive_failures: int = 5,
    sampler: Sampler | None = None,
    metric: Metric | None = None,
) -> Experiment:
    """Build a persistent experiment of one fixed configuration, default tpe sampler unless given."""
    fixed = Phase(
        name="p",
        n_trials=n_trials,
        max_consecutive_failures=max_consecutive_failures,
        sampler=Sampler() if sampler is None else sampler,
        fixed_overrides={"x": 1},
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=[fixed], metric=metric)


def _states(exp: Experiment) -> list[str]:
    """Return the state name of every trial in the experiment's only phase."""
    return [trial.state.name for trial in phase_trials(exp)]


@pytest.mark.parametrize(
    ("sampler", "max_failures", "repeat_error"),
    [
        pytest.param(None, 5, NoFeasibleTrialError, id="default_tpe"),
        pytest.param(None, 1, PhaseAbortRefusedError, id="default_tpe_abort_recorded"),
        pytest.param(Sampler(type="random", seed=0), 5, NoFeasibleTrialError, id="random"),
        pytest.param(Sampler(type="grid"), 5, NoFeasibleTrialError, id="grid"),
    ],
)
def test_failed_fixed_configuration_is_retried_by_raising_n_trials(
    tmp_path: Path,
    sampler: Sampler | None,
    max_failures: int,
    repeat_error: type[Exception],
) -> None:
    """One failed attempt buys one retry under any sampler, and a result is final."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=1)

    def attempt(n_trials: int) -> Experiment:
        return _fixed(
            tmp_path, trainer, n_trials, max_consecutive_failures=max_failures, sampler=sampler
        )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(attempt(1))
    assert _call_count(calls) == 1

    # The failed attempt used the target up, so repeating it launches nothing: it replays the
    # failure, or the recorded abort refuses it.
    with pytest.raises(repeat_error):
        run_experiment(attempt(1))
    assert _call_count(calls) == 1

    with pytest.raises(TrialTargetRefusedError) as refused:
        run_experiment(attempt(3))
    assert str(refused.value) == (
        "Phase 'p' is a fixed configuration with 1 failed attempt and no completed one; "
        "raising n_trials retries it one attempt at a time."
    )
    assert refused.value.remediation == "Use n_trials at most 2, not 3."
    assert _call_count(calls) == 1

    winners = run_experiment(attempt(2))
    assert _call_count(calls) == 2
    assert winners["p"].metric == 1.0
    assert _states(attempt(2)) == ["FAIL", "COMPLETE"]

    with pytest.raises(TrialTargetRefusedError) as repeated:
        run_experiment(attempt(3))
    assert str(repeated.value) == (
        "Phase 'p' is a fixed configuration that already has a COMPLETE or INFEASIBLE trial; "
        "n_trials=3 would run the same configuration again."
    )
    assert repeated.value.remediation == "Keep n_trials at 2, or use repeats to measure seed noise."
    assert _call_count(calls) == 2

    # The finished target still replays its result without launching.
    assert run_experiment(attempt(2))["p"].metric == 1.0
    assert _call_count(calls) == 2


def test_fixed_configuration_retry_counts_every_failed_attempt(tmp_path: Path) -> None:
    """After F failures the target may reach F + 1, and a retry that fails again counts too."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_fixed(tmp_path, trainer, 1))
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_fixed(tmp_path, trainer, 2))
    assert _call_count(calls) == 2

    with pytest.raises(TrialTargetRefusedError, match="2 failed attempts") as refused:
        run_experiment(_fixed(tmp_path, trainer, 4))
    assert refused.value.remediation == "Use n_trials at most 3, not 4."
    assert _call_count(calls) == 2

    assert run_experiment(_fixed(tmp_path, trainer, 3))["p"].metric == 1.0
    assert _call_count(calls) == 3


def _assert_advises_next_target(message: str, from_target: int) -> None:
    """Assert the message tells the operator to raise phase ``p``'s target by one in the config."""
    assert_advises_target(
        message, from_target, from_target + 1, "retry its failed attempt once more"
    )
    # The one attempt the target allowed is spent, so a higher limit would schedule nothing.
    assert "max_consecutive_failures above" not in message


def test_aborted_fixed_configuration_names_the_next_target(tmp_path: Path) -> None:
    """Each abort and the rerun refusing it quote the one-more-attempt target."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)

    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(_fixed(tmp_path, trainer, 1, max_consecutive_failures=1))
    _assert_advises_next_target(_operational_error_text(aborted.value), 1)
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(_fixed(tmp_path, trainer, 1, max_consecutive_failures=1))
    _assert_advises_next_target(_operational_error_text(refused.value), 1)

    # The retry fails again, so the next abort names the next target.
    with pytest.raises(NoFeasibleTrialError) as aborted_again:
        run_experiment(_fixed(tmp_path, trainer, 2, max_consecutive_failures=1))
    _assert_advises_next_target(_operational_error_text(aborted_again.value), 2)
    assert _call_count(calls) == 2

    assert (
        run_experiment(_fixed(tmp_path, trainer, 3, max_consecutive_failures=1))["p"].metric == 1.0
    )
    assert _states(_fixed(tmp_path, trainer, 3)) == ["FAIL", "FAIL", "COMPLETE"]


def _bounded() -> Metric:
    """Return a ``x=<value>`` metric whose upper bound a logged 5.0 exceeds."""
    return Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )


def test_infeasible_completed_attempt_closes_a_fixed_configuration(tmp_path: Path) -> None:
    """A COMPLETE trial out of the metric bounds still measured the configuration."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    bounded = _bounded()

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_fixed(tmp_path, trainer, 1, metric=bounded))
    assert _states(_fixed(tmp_path, trainer, 1)) == ["COMPLETE"]

    with pytest.raises(TrialTargetRefusedError, match="already has a COMPLETE or INFEASIBLE trial"):
        run_experiment(_fixed(tmp_path, trainer, 2, metric=bounded))
    assert _call_count(calls) == 1


def test_measured_infeasible_fixed_configuration_says_nothing_can_continue(tmp_path: Path) -> None:
    """An abort over a completed infeasible attempt offers no setting that retries it."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    experiment = _fixed(tmp_path, trainer, 1, max_consecutive_failures=1, metric=_bounded())

    with pytest.raises(NoFeasibleTrialError, match="1 consecutive failed/infeasible") as aborted:
        run_experiment(experiment)
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(experiment)
    for message in (_operational_error_text(aborted.value), _operational_error_text(refused.value)):
        assert "Its configuration was measured and is not feasible" in message
        assert "phases[p]" not in message
        assert "new `experiment` name" in message
        assert "with `storage: auto`, a new `workdir` also works" in message
    assert _call_count(calls) == 1


def _failure_code(error: BaseException) -> object:
    """Return the failure code the MCP runner records for ``error``."""
    return mcp_runner._terminal_failure_payload(error, stage="execution", cleanup_confirmed=True)[
        "code"
    ]


@pytest.mark.parametrize(
    ("failures", "value", "error_type", "code"),
    [
        pytest.param(1, 1.0, PhaseTrialsFailedError, "trials_failed", id="trials-failed"),
        pytest.param(0, 5.0, PhaseTrialsInfeasibleError, "trials_infeasible", id="out-of-bounds"),
    ],
)
def test_no_feasible_failure_code_follows_the_raised_subclass(
    tmp_path: Path,
    failures: int,
    value: float,
    error_type: type[NoFeasibleTrialError],
    code: str,
) -> None:
    """A failed or out-of-bounds trial and a replay of it each report their own failure code."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=failures, value=value)
    experiment = _fixed(tmp_path, trainer, 1, metric=_bounded())

    with pytest.raises(NoFeasibleTrialError) as measured:
        run_experiment(experiment)
    with pytest.raises(NoFeasibleTrialError, match="launched no trial") as replayed:
        run_experiment(experiment)

    assert _call_count(calls) == 1
    assert (type(measured.value), _failure_code(measured.value)) == (error_type, code)
    assert (type(replayed.value), _failure_code(replayed.value)) == (
        ReplayWithoutFeasibleTrialError,
        "replay_without_feasible_trial",
    )


@pytest.mark.parametrize(
    ("failures", "value", "error_type", "code"),
    [
        pytest.param(1, 1.0, PhaseTrialsFailedError, "trials_failed", id="failed-streak"),
        pytest.param(
            0, 5.0, PhaseTrialsInfeasibleError, "trials_infeasible", id="infeasible-streak"
        ),
    ],
)
def test_streak_abort_failure_code_says_whether_a_trial_failed(
    tmp_path: Path,
    failures: int,
    value: float,
    error_type: type[NoFeasibleTrialError],
    code: str,
) -> None:
    """An abort over infeasible trials is not reported as a trainer failure."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=failures, value=value)
    experiment = _fixed(tmp_path, trainer, 1, max_consecutive_failures=1, metric=_bounded())

    with pytest.raises(NoFeasibleTrialError, match="aborted") as aborted:
        run_experiment(experiment)

    assert (type(aborted.value), _failure_code(aborted.value)) == (error_type, code)


def test_stale_recovered_fixed_configuration_attempt_reruns_at_the_same_target(
    tmp_path: Path,
) -> None:
    """An attempt a kill or reboot left RUNNING recovers as interrupted and reruns without a raised target."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0)
    exp = _fixed(tmp_path, trainer, 1)
    study, trial_dir, _number = fabricate_stale_trial(exp, "p", attempt_id="killed-attempt")
    study.set_user_attr(PHASE_FINGERPRINT_ATTR, _phase_fingerprint(exp, exp.phases[0], {}))
    write_attempt_lifecycle(trial_dir, attempt_id="killed-attempt", state="allocated")

    assert run_experiment(exp)["p"].metric == 1.0
    assert _call_count(calls) == 1
    assert _states(exp) == ["FAIL", "COMPLETE"]

    with pytest.raises(TrialTargetRefusedError, match="already has a COMPLETE or INFEASIBLE trial"):
        run_experiment(_fixed(tmp_path, trainer, 2))
    assert _call_count(calls) == 1


def test_fresh_persistent_fixed_configuration_refuses_a_larger_target_before_any_launch(
    tmp_path: Path,
) -> None:
    """Run and the dry run refuse it up front, so an earlier phase never launches first."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0)
    searching = Phase(
        name="a",
        n_trials=1,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    fixed = Phase(name="p", n_trials=3, inherits=["a"])
    exp = make_experiment(persistent=tmp_path, trainer=trainer, phases=[searching, fixed])

    with pytest.raises(TrialTargetRefusedError) as dry:
        run_experiment(exp, dry_run=True)
    assert str(dry.value) == _FRESH_THREE
    assert dry.value.remediation == _FRESH_THREE_REMEDY
    assert not (tmp_path / "studies.journal").exists()

    with pytest.raises(TrialTargetRefusedError) as launched:
        run_experiment(exp)
    assert str(launched.value) == _FRESH_THREE
    assert launched.value.remediation == _FRESH_THREE_REMEDY
    assert _call_count(calls) == 0

    # The MCP runner calls run_experiment, so this is the failure it records for a launch.
    failure = mcp_runner._terminal_failure_payload(
        launched.value, stage="preflight", cleanup_confirmed=True
    )
    assert failure["code"] == "trial_target_refused"
    assert failure["actor"] == "operator"

    # The same tree runs once the fixed configuration asks for its one trial.
    ok = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[searching, fixed.model_copy(update={"n_trials": 1})],
    )
    assert set(run_experiment(ok)) == {"a", "p"}
    assert _call_count(calls) == 2


def test_dry_run_reads_the_recorded_attempts_of_a_persistent_fixed_configuration(
    tmp_path: Path,
) -> None:
    """The preview agrees with a launch: a raised target passes after a failure, not after a result."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=1)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_fixed(tmp_path, trainer, 1))

    run_experiment(_fixed(tmp_path, trainer, 2), dry_run=True)
    with pytest.raises(TrialTargetRefusedError, match="1 failed attempt") as refused:
        run_experiment(_fixed(tmp_path, trainer, 3), dry_run=True)
    assert refused.value.remediation == "Use n_trials at most 2, not 3."
    assert _call_count(calls) == 1

    run_experiment(_fixed(tmp_path, trainer, 2))
    run_experiment(_fixed(tmp_path, trainer, 2), dry_run=True)
    with pytest.raises(TrialTargetRefusedError, match="already has a COMPLETE or INFEASIBLE trial"):
        run_experiment(_fixed(tmp_path, trainer, 3), dry_run=True)
    assert _call_count(calls) == 2


def _fixed_config(tmp_path: Path, n_trials: int) -> Path:
    """Write a fixed configuration without an explicit sampler or search space."""
    return write_experiment(
        tmp_path / "exp.yaml",
        storage="auto",
        workdir=str(tmp_path / "runs"),
        provenance={"revision": "test"},
        phases=[{"name": "p", "n_trials": n_trials, "fixed_overrides": {"x": 1}}],
    )


def test_validate_warns_and_cli_dry_run_refuses_a_persistent_fixed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Load accepts it with one warning line, and the CLI dry run then refuses a fresh tree."""
    config = _fixed_config(tmp_path, 3)

    validated = CliRunner().invoke(cli_main, ["validate", str(config)])
    assert validated.exit_code == 0, validated.output
    warnings = [line.strip() for line in validated.output.splitlines() if "warning:" in line]
    assert warnings == [f"warning: {_WARNING}"]

    assert invoke_cli_boundary(["run", str(config), "--dry-run"], monkeypatch) != 0
    error = capsys.readouterr().err
    assert error.splitlines() == [f"phasesweep: {_FRESH_THREE}", f"fix: {_FRESH_THREE_REMEDY}"]

    one = _fixed_config(tmp_path, 1)
    clean = CliRunner().invoke(cli_main, ["validate", str(one)])
    assert clean.exit_code == 0, clean.output
    assert "warning:" not in clean.output


def test_dry_run_logs_the_fixed_configuration_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The dry run prints the advisory validate does, once a retry target is plausible."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=1)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_fixed(tmp_path, trainer, 1))
    caplog.set_level(logging.INFO)

    run_experiment(_fixed(tmp_path, trainer, 2), dry_run=True)

    warnings = [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("DRY RUN warning:")
    ]
    assert warnings == [(logging.WARNING, f"DRY RUN warning: {_WARNING}")]


@pytest.mark.parametrize(
    ("finished", "measured", "n_trials", "refusal"),
    [
        (0, 0, 1, None),
        (0, 0, 2, "would run 2 identical trials"),
        (1, 0, 1, None),
        (1, 0, 2, None),
        (1, 0, 3, "1 failed attempt and no completed one"),
        (3, 0, 4, None),
        (3, 0, 5, "3 failed attempts and no completed one"),
        (1, 1, 1, None),
        (1, 1, 2, "already has a COMPLETE or INFEASIBLE trial; n_trials=2"),
        (2, 1, 2, None),
        (2, 1, 3, "Keep n_trials at 2"),
    ],
)
def test_fixed_configuration_target_rule(
    finished: int, measured: int, n_trials: int, refusal: str | None
) -> None:
    """The whole rule in one table: F failures allow F + 1, and a COMPLETE trial closes it."""
    phase = Phase(name="p", n_trials=n_trials, fixed_overrides={"x": 1})
    message = fixed_configuration_target_refusal(phase, finished=finished, measured=measured)
    if refusal is None:
        assert message is None
    else:
        assert message is not None
        assert refusal in " ".join(message)


def test_a_phase_that_draws_is_not_limited_to_retries() -> None:
    """Only a fixed configuration is held to one attempt per failure; sampling phases take any target."""
    study = optuna.create_study()
    study.set_user_attr(TRIAL_TARGET_ATTR, 1)
    study.add_trial(optuna.trial.create_trial(value=1.0))
    drawing = Phase(
        name="p",
        n_trials=9,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )

    _validate_trial_target(study, drawing)
