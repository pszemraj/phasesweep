"""Reruns against a phase's recorded failure policy: a durable abort, a raised n_trials that recovers from it, and a changed max_consecutive_failures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import StudySchemaMismatchError, TrialTargetRefusedError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _last_successful_generation_path, _run_log_path
from phasesweep.engine.preview import _recorded_phase
from phasesweep.engine.selection import NoFeasibleTrialError, PhaseAbortRefusedError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    PHASE_RECOVERY_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.study_policy import (
    _ALLOCATION_CONTEXT_ATTR,
    _load_phase_policy_state,
    _PhasePolicyState,
)
from phasesweep.errors import OperatorAction, PhaseSweepError, operator_step
from phasesweep.mcp import runner as mcp_runner
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    phase_study,
    trial_outcome,
    write_flag_gated_trainer,
    write_flaky_trainer,
    write_trainer,
    write_trial_zero_trainer,
)


def test_cli_abort_advice_is_logged_once_per_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A live abort and its refused restart each print and durably log one concrete fix."""
    experiment = make_experiment(
        persistent=tmp_path,
        trial_command="false {overrides}",
        n_trials=1,
        max_consecutive_failures=1,
        sampler=Sampler(type="random", seed=0),
        gpu_policy="none",
    )
    config = tmp_path / "abort.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")), encoding="utf-8")
    run_log = _run_log_path(experiment)
    for invocation in range(2):
        caplog.clear()
        assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 1
        output = capsys.readouterr()
        fixes = [
            line.removeprefix("fix: ")
            for line in output.err.splitlines()
            if line.startswith("fix: ")
        ]
        assert len(fixes) == 1
        advice = fixes[0]
        assert "phases[p].n_trials above 1 in the config" in advice
        assert (output.err + output.out + caplog.text).count(advice) == 1
        assert run_log.read_text(encoding="utf-8").count(advice) == invocation + 1
    assert len(phase_study(experiment).trials) == 1


def test_evidence_pending_preview_routes_to_resume(tmp_path: Path) -> None:
    """Completed trainer evidence makes dry-run request recovery rather than a config edit."""
    directory = tmp_path / "trial"
    directory.mkdir()
    write_attempt_lifecycle(
        directory, attempt_id="attempt", state="exited", return_code=0, duration_seconds=1.0
    )
    study = optuna.create_study()
    running = study.ask()
    running.set_user_attr(ATTEMPT_ID_ATTR, "attempt")
    running.set_user_attr(TRIAL_DIR_ATTR, str(directory))
    with pytest.raises(
        PhaseSweepError, match="completed trainer evidence awaiting recovery"
    ) as caught:
        _recorded_phase(Phase(name="p", n_trials=1, gpu_policy="none"), study)

    error = caught.value
    assert error.action is OperatorAction.RUN_RECOVER_RUN
    advice = operator_step(error)
    assert "phasesweep run with the same config, without --dry-run" in advice
    assert "correct the experiment configuration" not in _operational_error_text(error)
    assert _operational_error_text(error).count(advice) == 1
    failure = mcp_runner._terminal_failure_payload(error, stage="preflight", cleanup_confirmed=True)
    assert failure["remediation"] == f"Ask the operator: {advice}"


@pytest.mark.integration
def test_aborted_phase_is_not_published_by_identical_noop_retry(tmp_path: Path) -> None:
    """A durable abort survives restart: the identical no-op re-run stays failed.

    First invocation: one success, then failures reach the threshold with every
    trial terminal. A second identical invocation sees ``remaining == 0`` but
    must not publish the lone COMPLETE trial as a completed phase, which would
    convert a recorded failure into a success with no new work.
    """
    marker = tmp_path / "succeeded_once"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import pathlib, sys
        marker = pathlib.Path({str(marker)!r})
        if marker.exists():
            sys.exit(1)
        marker.touch()
        print("x=0.25")
        """,
    )
    db = tmp_path / "abort.journal"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{db}",
        trainer=trainer,
        n_trials=3,
        max_consecutive_failures=2,
        sampler={"type": "random", "seed": 7},
    )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(exp)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    record = _load_phase_policy_state(study).abort
    assert record is not None
    assert record["policy"] == "max_consecutive_failures"
    assert record["consecutive_failures"] == 2

    # Identical retry: no new work is schedulable (3/3 terminal trials), so the
    # durable abort must keep the phase failed instead of publishing.
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


@pytest.mark.integration
def test_abort_recovery_target_with_no_remaining_slots_is_schema_mismatch(
    tmp_path: Path,
) -> None:
    """A raised target the recorded trials already fill leaves no slot to recover into, so it is refused."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    storage = f"journal:///{tmp_path / 'abort.journal'}"

    def experiment(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=storage,
            trainer=trainer,
            n_trials=n_trials,
            max_consecutive_failures=2,
            sampler={"type": "random", "seed": 7},
        )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(experiment(3))

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(storage))
    cohort_digest = study.trials[0].user_attrs[TRAINER_ENV_DIGEST_ATTR]
    for sequence in (3, 4):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                user_attrs={
                    TRAINER_ENV_DIGEST_ATTR: cohort_digest,
                    TRIAL_OUTCOME_ATTR: trial_outcome(
                        sequence, "failure", cause="simulated external terminal row"
                    ),
                },
            )
        )
    study.set_user_attr(TRIAL_TARGET_ATTR, 4)

    with pytest.raises(StudySchemaMismatchError, match="no remaining trial slots"):
        run_experiment(experiment(4))


@pytest.mark.integration
@pytest.mark.parametrize("interrupt_selection", [False, True], ids=["normal", "selection-crash"])
def test_topup_after_abort_runs_new_work_and_clears_durable_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_selection: bool
) -> None:
    """A raised target retires the durable abort before selection and survives replay."""
    flag = tmp_path / "resume_enabled"
    trainer = write_flag_gated_trainer(tmp_path, flag)
    db = tmp_path / "abort.journal"

    def _exp(n_trials: int):
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            trainer=trainer,
            n_trials=n_trials,
            max_consecutive_failures=2,
            sampler={"type": "random", "seed": 7},
        )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(_exp(3))

    flag.touch()
    if interrupt_selection:
        import phasesweep.engine.phase as phase_module

        def crash_in_selection(*args: object, **kwargs: object) -> None:
            study = optuna.load_study(
                study_name="t::p", storage=_resolve_storage(f"journal:///{db}")
            )
            assert _load_phase_policy_state(study).abort is None
            raise RuntimeError("simulated crash during winner selection")

        with monkeypatch.context() as patch:
            patch.setattr(phase_module, "_select_phase_winner", crash_in_selection)
            with pytest.raises(
                RuntimeError, match="simulated crash during winner selection"
            ) as caught:
                run_experiment(_exp(6))
            assert type(caught.value) is RuntimeError
        study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
        assert _load_phase_policy_state(study).abort is None

    winners = run_experiment(_exp(6))

    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert _load_phase_policy_state(study).abort is None


@pytest.mark.integration
def test_supported_topup_preserves_consecutive_failure_streak(tmp_path: Path) -> None:
    """A random-sampler top-up continues the durable failure streak."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os, sys
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            print("x=0.5")
        else:
            sys.exit(1)
        """,
    )
    db = tmp_path / "topup.journal"

    def _exp(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            trainer=trainer,
            n_trials=n_trials,
            max_consecutive_failures=3,
            sampler={"type": "random", "seed": 7},
        )

    first = run_experiment(_exp(3))
    assert first["p"].metric == pytest.approx(0.5)
    published_before = _last_successful_generation_path(_exp(3)).read_text()

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(_exp(4))

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(f"journal:///{db}"))
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "FAIL", "FAIL", "FAIL"]
    assert [trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in study.trials] == [
        1,
        2,
        3,
        4,
    ]
    abort = _load_phase_policy_state(study).abort
    assert abort is not None
    assert abort["consecutive_failures"] == 3
    assert _last_successful_generation_path(_exp(4)).read_text() == published_before


@pytest.mark.integration
def test_retry_that_fails_again_aborts_after_a_fresh_failure_count(tmp_path: Path) -> None:
    """Recovering from an abort restarts the failure count at zero.

    The two cells that tripped a limit of 2 would trip it again after one more failure if they
    still counted, so the retry that fails again must run both cells before it aborts, and a
    later retry that succeeds must publish.
    """
    trainer, calls = write_flaky_trainer(tmp_path, failures=4)

    def grid(n_trials: int) -> Experiment:
        return make_experiment(
            persistent=tmp_path,
            trainer=trainer,
            n_trials=n_trials,
            max_consecutive_failures=2,
            gpu_policy="none",
            sampler=Sampler(type="grid", seed=0),
            search_space={"x": IntParam(type="int", low=1, high=2)},
        )

    def policy_state() -> tuple[list[str], _PhasePolicyState]:
        study = phase_study(grid(2))
        return [trial.state.name for trial in study.trials], _load_phase_policy_state(study)

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(grid(2))
    assert len(calls.read_text().splitlines()) == 2

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(grid(4))
    assert len(calls.read_text().splitlines()) == 4
    states, state = policy_state()
    assert states == ["FAIL"] * 4
    assert state.consecutive_failures == 2
    assert state.abort is not None
    assert state.abort["trial_target"] == 4
    assert state.abort["consecutive_failures"] == 2

    assert run_experiment(grid(6))["p"].metric == pytest.approx(1.0)
    states, state = policy_state()
    assert states == ["FAIL"] * 4 + ["COMPLETE"] * 2
    assert state.abort is None


@pytest.mark.integration
def test_changing_the_failure_limit_never_reinterprets_recorded_outcomes(tmp_path: Path) -> None:
    """``max_consecutive_failures`` governs new work, not outcomes already recorded.

    The limit is a run control outside the phase fingerprint. Lowering it must
    neither abort a completed phase nor leave state behind that restoring the
    limit cannot undo.
    """
    trainer = write_trial_zero_trainer(tmp_path, otherwise="raise SystemExit(1)")
    original = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=2,
        max_consecutive_failures=2,
        gpu_policy="none",
    )
    lowered = original.model_copy(
        update={"phases": [original.phases[0].model_copy(update={"max_consecutive_failures": 1})]}
    )

    first = run_experiment(original)
    assert run_experiment(lowered)["p"].attempt_id == first["p"].attempt_id
    study = phase_study(original)
    assert _load_phase_policy_state(study).consecutive_failures == 1
    assert _load_phase_policy_state(study).abort is None
    assert run_experiment(original)["p"].attempt_id == first["p"].attempt_id


def _with_limit(experiment: Experiment, limit: int) -> Experiment:
    """Return ``experiment`` with phase ``p``'s ``max_consecutive_failures`` set to ``limit``."""
    phase = experiment.phases[0].model_copy(update={"max_consecutive_failures": limit})
    return experiment.model_copy(update={"phases": [phase]})


def test_raised_failure_limit_retires_a_streak_abort_while_trials_remain(tmp_path: Path) -> None:
    """A limit above the recorded streak recovers the abort and runs the trials the target left."""
    flag = tmp_path / "resume_enabled"
    initial = make_experiment(
        persistent=tmp_path,
        trainer=write_flag_gated_trainer(tmp_path, flag),
        n_trials=4,
        max_consecutive_failures=2,
        gpu_policy="none",
        sampler={"type": "random", "seed": 7},
    )
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(initial)
    assert [trial.state.name for trial in phase_study(initial).trials] == ["FAIL", "FAIL"]
    message = _operational_error_text(aborted.value)
    assert "phases[p].max_consecutive_failures above 2 in the config" in message
    assert "run the 2 trials still left at n_trials=4" in message
    assert "phases[p].n_trials above 4 in the config" in message

    # A limit that is not above the recorded streak leaves the abort in place, preview and run.
    flag.touch()
    for limit in (1, 2):
        for dry_run in (True, False):
            with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
                run_experiment(_with_limit(initial, limit), dry_run=dry_run)
    assert len(phase_study(initial).trials) == 2

    raised = _with_limit(initial, 3)
    run_experiment(raised, dry_run=True)
    assert len(phase_study(initial).trials) == 2
    assert run_experiment(raised)["p"].metric == pytest.approx(0.5)
    study = phase_study(initial)
    assert [trial.state.name for trial in study.trials] == ["FAIL", "FAIL", "COMPLETE", "COMPLETE"]
    state = _load_phase_policy_state(study)
    assert (state.abort, state.consecutive_failures) == (None, 0)
    # The recovery keeps the accepted target and spends it: nothing else was raised.
    assert study.user_attrs[PHASE_RECOVERY_ATTR]["trial_target"] == 4
    assert study.user_attrs[TRIAL_TARGET_ATTR] == 4


def test_raised_failure_limit_leaves_an_abort_with_no_trials_left(tmp_path: Path) -> None:
    """With the target spent, only a raised target recovers, and a limit alone is no recovery."""
    initial = make_experiment(
        persistent=tmp_path,
        trainer=write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)"),
        n_trials=2,
        max_consecutive_failures=2,
        gpu_policy="none",
        sampler={"type": "random", "seed": 7},
    )
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(initial)

    refusals = [_operational_error_text(aborted.value)]
    for dry_run in (True, False):
        with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
            run_experiment(_with_limit(initial, 5), dry_run=dry_run)
        refusals.append(_operational_error_text(refused.value))
    for message in refusals:
        assert "phases[p].n_trials above 2 in the config" in message
        assert "max_consecutive_failures above" not in message
    assert len(phase_study(initial).trials) == 2


def _abort_fatally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **phase: object
) -> tuple[Experiment, Path]:
    """Run a three-trial phase whose first trial hits an unexpected objective error.

    :param object phase: Phase fields that replace the seeded random sampler default.
    :return tuple[Experiment, Path]: The aborted experiment, and the log that gains a line
        for every trainer launch, which the abort stopped before the first.
    """
    import phasesweep.engine.phase as phase_module

    calls = tmp_path / "calls.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        from pathlib import Path
        with Path({str(calls)!r}).open("a") as log:
            log.write("run\\n")
        print("x=0.5")
        """,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=3,
        max_consecutive_failures=2,
        gpu_policy="none",
        **{"sampler": {"type": "random", "seed": 7}, **phase},
    )

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated objective defect")

    with monkeypatch.context() as patch:
        patch.setattr(phase_module, "prepare_trainer_input", explode)
        with pytest.raises(RuntimeError, match="simulated objective defect"):
            run_experiment(experiment)
    abort = _load_phase_policy_state(phase_study(experiment)).abort
    assert abort is not None
    assert abort["policy"] == "unexpected_objective_exception"
    return experiment, calls


def test_aborted_phase_headline_states_the_recorded_cause_of_a_fatal_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline quotes the ledger's cause, not a failure count, and advises the target alone."""
    import phasesweep.engine.phase as phase_module

    experiment, _calls = _abort_fatally(tmp_path, monkeypatch)
    # A live run raises a fatal error itself before the phase is finished, so this is the one
    # way to reach the headline with a fatal abort on record.
    run = SimpleNamespace(
        experiment=experiment,
        phase=experiment.phases[0],
        study=phase_study(experiment),
        optimize_deadline=None,
        deadline_exhausted=False,
        aborted=True,
    )

    with pytest.raises(NoFeasibleTrialError) as aborted:
        phase_module._finish_phase(run, phase_fingerprint="unused")

    message = _operational_error_text(aborted.value)
    assert "Phase 'p' aborted: RuntimeError: simulated objective defect" in message
    assert "consecutive" not in message
    assert "phases[p].n_trials above 3 in the config" in message
    assert "max_consecutive_failures above" not in message
    # The abort recorded no streak, so there is no failure count to restart.
    assert "failure count" not in message


def test_fatal_abort_of_a_grid_advises_the_retry_target_without_a_failure_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fatal abort recorded no streak, so a grid's target advice does not promise a restart."""
    experiment, _calls = _abort_fatally(
        tmp_path,
        monkeypatch,
        sampler={"type": "grid", "seed": 0},
        search_space={"x": IntParam(type="int", low=1, high=3)},
    )

    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(experiment)

    message = _operational_error_text(refused.value)
    assert "phases[p].n_trials from 3 to 4 in the config" in message
    assert "retry its 1 failed cell and run its 2 remaining cells" in message
    assert "failure count" not in message


def test_raised_failure_limit_does_not_retire_a_fatal_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fatal abort records no streak, so only a raised target recovers it."""
    initial, calls = _abort_fatally(tmp_path, monkeypatch)

    # The limit now exceeds the one failure counted, with two trials left to run, and still
    # retires nothing: the preview and the run both refuse, and nothing launched.
    roomy = _with_limit(initial, 5)
    refusals = []
    for dry_run in (True, False):
        with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
            run_experiment(roomy, dry_run=dry_run)
        refusals.append(_operational_error_text(refused.value))
    for message in refusals:
        assert "simulated objective defect" in message
        assert "phases[p].n_trials above 3 in the config" in message
        assert "max_consecutive_failures above" not in message
        assert "failure count" not in message
    assert not calls.exists()
    assert len(phase_study(initial).trials) == 1

    raised_target = roomy.phases[0].model_copy(update={"n_trials": 4})
    recovered = roomy.model_copy(update={"phases": [raised_target]})
    assert run_experiment(recovered)["p"].metric == pytest.approx(0.5)
    assert len(calls.read_text().splitlines()) == 3


@pytest.mark.parametrize(
    ("failing_attr", "old_target_refusal"),
    [
        (PHASE_RECOVERY_ATTR, PhaseAbortRefusedError),
        (TRIAL_TARGET_ATTR, TrialTargetRefusedError),
        (_ALLOCATION_CONTEXT_ATTR, TrialTargetRefusedError),
    ],
)
@pytest.mark.integration
def test_interrupted_recovery_keeps_the_raised_target_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_attr: str,
    old_target_refusal: type[Exception],
) -> None:
    """A failed write while accepting a recovery target neither spends nor transfers it.

    Raising ``n_trials`` authorizes recovering from a streak that meets a
    lowered limit. Whichever write of that transition fails, the old target
    still cannot run the recovery, and an identical retry runs it without
    demanding a further raise.
    """
    import phasesweep.engine.phase as phase_module

    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            raise SystemExit(1)
        print("x=0.5")
        """,
    )
    initial = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=2,
        max_consecutive_failures=2,
        gpu_policy="none",
    )

    def with_target(n_trials: int) -> Experiment:
        phase = initial.phases[0].model_copy(
            update={"n_trials": n_trials, "max_consecutive_failures": 1}
        )
        return initial.model_copy(update={"phases": [phase]})

    real_after_trial = phase_module._after_trial

    def stop_after_first_outcome(*args: Any, **kwargs: Any) -> None:
        real_after_trial(*args, **kwargs)
        raise RuntimeError("simulated orchestrator stop")

    with monkeypatch.context() as ctx:
        ctx.setattr(phase_module, "_after_trial", stop_after_first_outcome)
        with pytest.raises(RuntimeError, match="simulated orchestrator stop"):
            run_experiment(initial)

    real_set_user_attr = optuna.Study.set_user_attr

    def fail_write(study: optuna.Study, key: str, value: Any) -> None:
        if key == failing_attr:
            raise RuntimeError("simulated storage failure")
        real_set_user_attr(study, key, value)

    with monkeypatch.context() as ctx:
        ctx.setattr(optuna.Study, "set_user_attr", fail_write)
        with pytest.raises(RuntimeError, match="simulated storage failure"):
            run_experiment(with_target(3))

    def trial_states() -> list[str]:
        study = phase_study(initial)
        return [trial.state.name for trial in study.trials]

    assert trial_states() == ["FAIL"]
    with pytest.raises(old_target_refusal):
        run_experiment(with_target(2))
    assert trial_states() == ["FAIL"]

    assert run_experiment(with_target(3))["p"].metric == pytest.approx(0.5)
    assert trial_states() == ["FAIL", "COMPLETE", "COMPLETE"]
