"""Reruns against a phase's recorded failure policy: a durable abort, a raised n_trials that recovers from it, and a changed max_consecutive_failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import Experiment
from phasesweep.engine import StudySchemaMismatchError, TrialTargetRegressionError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _last_successful_generation_path
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    PHASE_RECOVERY_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.study_policy import _ALLOCATION_CONTEXT_ATTR, _load_phase_policy_state
from tests.conftest import (
    make_experiment,
    write_flag_gated_trainer,
    write_trainer,
    write_trial_zero_trainer,
)


@pytest.mark.integration
def test_aborted_phase_is_not_published_by_identical_noop_retry(tmp_path: Path) -> None:
    """A durable abort survives restart: the identical no-op re-run stays failed.

    First invocation: one success, then failures reach the threshold with every
    trial terminal. Before review v0.5.17 / blocker 1, a second identical
    invocation saw ``remaining == 0`` and published the lone COMPLETE trial as
    a completed phase — converting a recorded failure into a success with no
    new work.
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
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
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
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(exp)
    assert not _last_successful_generation_path(exp).exists()


@pytest.mark.integration
def test_abort_recovery_target_with_no_remaining_slots_is_schema_mismatch(
    tmp_path: Path,
) -> None:
    """Contradictory durable abort/trial counts are operator-visible study state."""
    trainer = write_trainer(tmp_path / "trainer.py", "raise SystemExit(1)")
    storage = f"journal:///{tmp_path / 'abort.journal'}"

    def experiment(n_trials: int) -> Experiment:
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=storage,
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
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
                    TRIAL_OUTCOME_ATTR: {
                        "schema_version": 1,
                        "sequence": sequence,
                        "outcome": "failure",
                        "cause": "simulated external terminal row",
                    },
                },
            )
        )
    study.set_user_attr(TRIAL_TARGET_ATTR, 4)

    with pytest.raises(StudySchemaMismatchError, match="no remaining trial slots"):
        run_experiment(experiment(4))


@pytest.mark.integration
def test_topup_after_abort_runs_new_work_and_clears_durable_abort(tmp_path: Path) -> None:
    """Raising n_trials after an abort is the explicit resume path.

    The top-up schedules genuinely new attempts; reaching a successful winner
    selection consumes the durable abort record (review v0.5.17 / blocker 1).
    """
    flag = tmp_path / "resume_enabled"
    trainer = write_flag_gated_trainer(tmp_path, flag)
    db = tmp_path / "abort.journal"

    def _exp(n_trials: int):
        return make_experiment(
            workdir=tmp_path / "runs",
            storage=f"journal:///{db}",
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
            n_trials=n_trials,
            max_consecutive_failures=2,
            sampler={"type": "random", "seed": 7},
        )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(_exp(3))

    flag.touch()
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
            trial_command=f"python {trainer} {{overrides}}",
            override_format="argparse",
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
    study = optuna.load_study(
        study_name="t::p", storage=_resolve_storage(original.resolved_storage)
    )
    assert _load_phase_policy_state(study).consecutive_failures == 1
    assert _load_phase_policy_state(study).abort is None
    assert run_experiment(original)["p"].attempt_id == first["p"].attempt_id


@pytest.mark.parametrize(
    ("failing_attr", "old_target_refusal"),
    [
        (PHASE_RECOVERY_ATTR, NoFeasibleTrialError),
        (TRIAL_TARGET_ATTR, TrialTargetRegressionError),
        (_ALLOCATION_CONTEXT_ATTR, TrialTargetRegressionError),
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
        study = optuna.load_study(
            study_name="t::p", storage=_resolve_storage(initial.resolved_storage)
        )
        return [trial.state.name for trial in study.trials]

    assert trial_states() == ["FAIL"]
    with pytest.raises(old_target_refusal):
        run_experiment(with_target(2))
    assert trial_states() == ["FAIL"]

    assert run_experiment(with_target(3))["p"].metric == pytest.approx(0.5)
    assert trial_states() == ["FAIL", "COMPLETE", "COMPLETE"]
