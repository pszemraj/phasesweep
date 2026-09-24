"""Phase/run wallclock timeouts: deadline attribution, partial-winner-versus-abort precedence, and the extraction/launch deadline boundary."""

from __future__ import annotations

import signal
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import Experiment, IntParam, LogRegexExtractor, Metric, Phase, Sampler
from phasesweep.engine import PhaseSweepError
from phasesweep.engine.artifacts import _load_winner
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _last_successful_generation_path
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import PHASE_ABORT_ATTR, PHASE_DECISION_ATTR, TRIAL_OUTCOME_ATTR
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.engine.trial import ExecutedTrial, extract_trial_result
from phasesweep.evidence import TrialContext
from phasesweep.runtime.process import ProcessResult
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from tests.conftest import (
    make_experiment,
    write_constant_trainer,
    write_trainer,
    write_trial_zero_trainer,
)


def _sleeping_score_experiment(
    tmp_path: Path,
    *,
    experiment: str,
    n_trials: int = 3,
    timeout_seconds_per_phase: float | None = None,
    timeout_seconds_per_run: float | None = None,
    allow_incomplete_on_timeout: bool = False,
    sleep_seconds: float = 0.5,
) -> Experiment:
    trainer = write_trainer(
        tmp_path,
        f"""
        import argparse, json, time
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        time.sleep({sleep_seconds})
        with open(args.out, "w") as f:
            json.dump({{"x": 1.0}}, f)
        print("x=1.0")
        """,
    )
    return Experiment(
        experiment=experiment,
        workdir=str(tmp_path / "runs"),
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        metric=Metric(
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
        ),
        timeout_seconds_per_run=timeout_seconds_per_run,
        phases=[
            Phase(
                name="p",
                n_trials=n_trials,
                timeout_seconds_per_phase=timeout_seconds_per_phase,
                allow_incomplete_on_timeout=allow_incomplete_on_timeout,
                search_space={},
            )
        ],
    )


@pytest.mark.integration
def test_runtime_rejects_unexplained_trial_budget_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampler that stops early cannot publish a falsely complete winner."""
    exp = _sleeping_score_experiment(tmp_path, experiment="budget_shortfall", n_trials=1)
    monkeypatch.setattr(optuna.Study, "optimize", lambda self, objective, **kwargs: None)

    with pytest.raises(RuntimeError, match="stopped after 0/1 terminal trials") as exc_info:
        run_experiment(exp)
    assert not isinstance(exc_info.value, PhaseSweepError)


@pytest.mark.integration
def test_phase_timeout_refuses_incomplete_winner(tmp_path: Path) -> None:
    """A phase wallclock timeout must not bless the best partial trial by default."""
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout",
        timeout_seconds_per_phase=0.2,
    )

    with pytest.raises(
        TimeoutError,
        match=r"timed out via phase guard .*Refusing to select a winner",
    ):
        run_experiment(exp)


@pytest.mark.integration
def test_phase_timeout_preempts_active_trial(tmp_path: Path) -> None:
    """A phase wallclock timeout is a hard subprocess deadline, not only an Optuna scheduler timeout."""
    # The trainer would sleep 30s per trial; timing margins are deliberately
    # enormous on both sides so a loaded CI host cannot flip the verdict: a
    # preempted run finishes in well under 10s, a non-preempted one needs 30+.
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_hard",
        timeout_seconds_per_phase=0.05,
        sleep_seconds=30.0,
    )

    started = time.monotonic()
    with pytest.raises(
        TimeoutError,
        match=r"timed out via phase guard .*Refusing to select a winner",
    ):
        run_experiment(exp)
    elapsed = time.monotonic() - started

    assert elapsed < 10.0
    # Causal marker: had the trial run to completion, it would have written
    # its result file.
    phase_dir = tmp_path / "runs" / "phase_timeout_hard" / "p"
    assert list(phase_dir.glob("trial_00000__*/r.json")) == []


@pytest.mark.integration
def test_incomplete_timeout_winner_is_accepted_only_under_current_opt_in(tmp_path: Path) -> None:
    """An opted-in timeout publishes an incomplete winner; resume without the opt-in refuses it."""
    # Two-sided timing margin: one 1s trial must finish well within the 6s
    # budget even on a loaded host, while all ten (>= 10s of sleeping alone)
    # can never finish inside it.
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_allowed",
        n_trials=10,
        timeout_seconds_per_phase=6.0,
        allow_incomplete_on_timeout=True,
        sleep_seconds=1.0,
    )

    winners = run_experiment(exp)

    completion = winners["p"].completion
    assert completion["requested_trials"] == 10
    assert 1 <= completion["completed_trials"] < completion["requested_trials"]
    assert completion["completed_trials"] <= completion["finished_trials"]
    assert completion["finished_trials"] <= completion["requested_trials"]
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"

    current = _sleeping_score_experiment(
        tmp_path,
        experiment="phase_timeout_allowed",
        n_trials=10,
        timeout_seconds_per_phase=6.0,
        sleep_seconds=1.0,
    )
    with pytest.raises(RuntimeError, match="incomplete phase result"):
        _load_winner(current, current.phases[0], {})


@pytest.mark.parametrize("allow_incomplete_on_timeout", [False, True])
@pytest.mark.integration
def test_timeout_after_all_terminal_trials_is_complete_enough(
    tmp_path: Path,
    allow_incomplete_on_timeout: bool,
) -> None:
    """A timeout guard should not reject a phase once every requested trial is terminal."""
    trainer = write_trial_zero_trainer(tmp_path, otherwise="time.sleep(30.0)")
    exp = make_experiment(
        experiment="phase_timeout_all_terminal",
        workdir=tmp_path / "runs",
        trainer=trainer,
        n_trials=2,
        timeout_seconds_per_phase=8.0,
        allow_incomplete_on_timeout=allow_incomplete_on_timeout,
        search_space={},
    )

    winners = run_experiment(exp)

    completion = winners["p"].completion
    assert completion["requested_trials"] == 2
    assert completion["completed_trials"] == 1
    assert completion["finished_trials"] == 2
    assert completion["incomplete"] is False

    current = exp.model_copy(
        update={
            "phases": [
                exp.phases[0].model_copy(update={"allow_incomplete_on_timeout": False}),
            ],
        }
    )
    loaded = _load_winner(current, current.phases[0], {})
    assert loaded.completion["incomplete"] is False


@pytest.mark.integration
def test_timeout_winner_is_not_masked_by_consecutive_failure_abort(tmp_path: Path) -> None:
    trainer = write_trial_zero_trainer(tmp_path, otherwise="time.sleep(30.0)")
    exp = make_experiment(
        experiment="phase_timeout_allowed_abort_counter",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'timeout-counter.journal'}",
        trainer=trainer,
        n_trials=3,
        max_consecutive_failures=1,
        timeout_seconds_per_phase=8.0,
        allow_incomplete_on_timeout=True,
        sampler=Sampler(type="random", seed=7),
        search_space={},
    )

    winners = run_experiment(exp)

    assert winners["p"].trial_number == 0
    completion = winners["p"].completion
    assert completion["requested_trials"] == 3
    assert 1 <= completion["completed_trials"] < completion["requested_trials"]
    assert (
        completion["completed_trials"]
        <= completion["finished_trials"]
        < completion["requested_trials"]
    )
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"
    study = optuna.load_study(
        study_name="phase_timeout_allowed_abort_counter::p",
        storage=_resolve_storage(exp.storage),
    )
    deadline_trials = [
        trial
        for trial in study.trials
        if trial.user_attrs.get(TRIAL_OUTCOME_ATTR, {}).get("outcome") == "cancelled"
    ]
    assert len(deadline_trials) == 1
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert _load_phase_policy_state(study).consecutive_failures == 0


@pytest.mark.parametrize("clock_elapses", [True, False])
@pytest.mark.integration
def test_scheduler_deadline_decides_partial_winner_versus_failure_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock_elapses: bool
) -> None:
    """Only an elapsed clock lets a failure-aborted phase publish a partial winner.

    The failing trial exits on its own before its wallclock-capped subprocess
    timeout fires, so no per-trial cause sets ``deadline_exhausted``, and the
    same trial trips ``max_consecutive_failures``. When the phase clock also
    elapses while that trial is in flight, only the scheduler-level check can
    observe the deadline, and timeout precedence must still publish the earlier
    successful trial. With budget left, nothing is relabelled: the streak abort
    stands and the phase fails.
    """
    import types

    import phasesweep.engine.phase as phase_mod

    trainer = write_trial_zero_trainer(tmp_path, otherwise="sys.exit(1)")
    exp = make_experiment(
        experiment="phase_scheduler_deadline_with_abort",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'decision.journal'}",
        trainer=trainer,
        phases=[
            Phase(
                name="p",
                n_trials=3,
                max_consecutive_failures=1,
                timeout_seconds_per_phase=600.0,
                allow_incomplete_on_timeout=True,
                gpu_policy="none",
                allow_no_gpu_isolation=True,
                sampler=Sampler(
                    type="tpe",
                    seed=7,
                    n_startup_trials=10,
                    acknowledge_nonresumable=True,
                ),
                search_space={"x": IntParam(type="int", low=1, high=3)},
            ),
            Phase(
                name="child",
                inherits=["p"],
                n_trials=1,
                gpu_policy="none",
                allow_no_gpu_isolation=True,
                sampler=Sampler(type="random", seed=7),
                search_space={},
            ),
        ],
    )

    # Only the phase module's clock is virtualised: the real clock still bounds
    # Optuna's own timeout, GPU leases, and extraction, so no other code path
    # can attribute a deadline here.
    offset = {"seconds": 0.0}
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        phase_mod,
        "time",
        types.SimpleNamespace(
            monotonic=lambda: real_monotonic() + offset["seconds"],
            sleep=lambda _seconds: None,
        ),
    )
    real_launch_trial = phase_mod.launch_trial
    launched_trials: list[int] = []

    def launch_and_burn_the_budget(**kwargs: object) -> object:
        launched_trials.append(int(kwargs["trial_id"]))
        executed = real_launch_trial(**kwargs)
        if clock_elapses and kwargs["trial_id"] == 1:
            offset["seconds"] = 10_000.0
        return executed

    monkeypatch.setattr(phase_mod, "launch_trial", launch_and_burn_the_budget)

    if not clock_elapses:
        with pytest.raises(NoFeasibleTrialError, match="aborted after 1 consecutive failures"):
            run_experiment(exp)
        return

    real_select = phase_mod._select_phase_winner
    crashed = {"once": False}

    def crash_first_selection(*args: object, **kwargs: object):
        if not crashed["once"]:
            crashed["once"] = True
            raise RuntimeError("simulated crash after partial-timeout decision")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(phase_mod, "_select_phase_winner", crash_first_selection)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_experiment(exp)

    study = optuna.load_study(
        study_name="phase_scheduler_deadline_with_abort::p",
        storage=_resolve_storage(exp.storage),
    )
    assert study.user_attrs[PHASE_DECISION_ATTR]["decision"] == "accepted_partial_timeout"
    trial_count = len(study.trials)

    winners = run_experiment(exp)

    study = optuna.load_study(
        study_name="phase_scheduler_deadline_with_abort::p",
        storage=_resolve_storage(exp.storage),
    )
    assert len(study.trials) == trial_count

    assert winners["p"].trial_number == 0
    completion = winners["p"].completion
    assert completion["requested_trials"] == 3
    assert completion["finished_trials"] == 2
    assert completion["completed_trials"] == 1
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"
    assert completion["timeout_scope"] == "phase"

    # The child study now binds the parent's published winner. A third
    # identical run must still replay both studies without treating the
    # parent's accepted partial target as a top-up, and without asking TPE to
    # reconstruct continuation state for suggestions that will never launch.
    launch_count = len(launched_trials)
    replayed = run_experiment(exp)
    assert replayed["p"].trial_number == winners["p"].trial_number
    assert replayed["child"].trial_number == winners["child"].trial_number
    assert len(launched_trials) == launch_count


@pytest.mark.integration
def test_refused_partial_timeout_consumes_simultaneous_failure_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raising the timeout must remain a valid retry after timeout wins a race."""
    import types

    import phasesweep.engine.phase as phase_mod

    trainer = write_trial_zero_trainer(tmp_path, otherwise="sys.exit(1)")
    exp = make_experiment(
        experiment="refused_partial_timeout_abort",
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'refused.journal'}",
        trainer=trainer,
        n_trials=3,
        max_consecutive_failures=1,
        timeout_seconds_per_phase=600.0,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        sampler=Sampler(type="random", seed=7),
        search_space={},
    )

    offset = {"seconds": 0.0}
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        phase_mod,
        "time",
        types.SimpleNamespace(
            monotonic=lambda: real_monotonic() + offset["seconds"],
            sleep=lambda _seconds: None,
        ),
    )
    real_launch_trial = phase_mod.launch_trial

    def launch_and_expire(**kwargs: object) -> object:
        executed = real_launch_trial(**kwargs)
        if kwargs["trial_id"] == 1:
            offset["seconds"] = 10_000.0
        return executed

    monkeypatch.setattr(phase_mod, "launch_trial", launch_and_expire)
    with pytest.raises(TimeoutError, match="Refusing to select a winner"):
        run_experiment(exp)

    study = optuna.load_study(
        study_name="refused_partial_timeout_abort::p",
        storage=_resolve_storage(exp.storage),
    )
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert _load_phase_policy_state(study).consecutive_failures == 0

    # The operator's documented remedy is now viable: with a fresh/larger
    # budget, the remaining attempt can run instead of the stale failure abort
    # rejecting the invocation during phase startup.
    offset["seconds"] = 0.0
    write_constant_trainer(tmp_path)
    monkeypatch.setattr(phase_mod, "launch_trial", real_launch_trial)
    winners = run_experiment(exp)
    assert winners["p"].metric == pytest.approx(0.5)


@pytest.mark.integration
def test_shutdown_during_objective_does_not_persist_fatal_phase_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An orchestrator signal is cancellation, not an objective implementation bug."""
    import phasesweep.engine.phase as phase_mod

    trainer = write_constant_trainer(tmp_path)
    exp = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'shutdown.journal'}",
        trainer=trainer,
        n_trials=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        max_consecutive_failures=1,
        sampler={"type": "random", "seed": 7},
    )
    real_launch_trial = phase_mod.launch_trial
    interrupted = {"once": False}

    def interrupt_first_launch(**kwargs: object) -> object:
        if not interrupted["once"]:
            interrupted["once"] = True
            report = ShutdownCleanupReport(
                signum=signal.SIGTERM,
                cleanup_confirmed=True,
                child_pgids=(),
            )
            raise PhaseSweepShutdown(signal.SIGTERM, report)
        return real_launch_trial(**kwargs)

    monkeypatch.setattr(phase_mod, "launch_trial", interrupt_first_launch)
    with pytest.raises(PhaseSweepShutdown) as exc_info:
        run_experiment(exp)

    assert exc_info.value.published_result_committed is False

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(exp.storage))
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None
    assert study.trials[0].user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "cancelled"

    winners = run_experiment(exp)
    assert winners["p"].metric == pytest.approx(0.5)
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(exp.storage))
    assert study.user_attrs.get(PHASE_ABORT_ATTR) is None


@pytest.mark.parametrize("lease_timeout", [True, False])
def test_gpu_lease_timeout_type_decides_partial_winner_versus_fatal_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_timeout: bool
) -> None:
    """Only ``GpuLeaseTimeoutError`` may relabel a lease wait as wallclock exhaustion.

    Phase attribution narrows on the dedicated subclass. A plain
    ``TimeoutError`` escaping the lock layer is an infrastructure failure: it
    must surface as a fatal abort, never let ``allow_incomplete_on_timeout``
    publish a partial winner off broken infrastructure. This pins the
    ``except GpuLeaseTimeoutError`` contract in ``engine/phase.py`` — widening
    it to ``TimeoutError`` or raising the plain superclass from the pool fails
    one side of this parametrization.
    """
    import contextlib

    import phasesweep.engine.phase as phase_mod
    from phasesweep.runtime.gpu import GpuAssignment, GpuLeaseTimeoutError

    trainer = write_constant_trainer(tmp_path, value=1.0)
    exp = make_experiment(
        experiment="gpu_lease_timeout_attribution",
        workdir=tmp_path / "runs",
        trainer=trainer,
        n_trials=3,
        max_consecutive_failures=1,
        timeout_seconds_per_phase=600.0,
        allow_incomplete_on_timeout=True,
        search_space={},
    )

    exc_type = GpuLeaseTimeoutError if lease_timeout else TimeoutError

    class _FailingSecondLeasePool:
        def __init__(self) -> None:
            self.calls = 0

        def cancel_waiters(self) -> None:
            pass

        @contextlib.contextmanager
        def acquire(self, *, deadline: float | None = None):
            self.calls += 1
            if self.calls > 1:
                raise exc_type("GPU lease wait exhausted")
            yield GpuAssignment(visible_devices=None, lease_fds=())

    monkeypatch.setattr(
        phase_mod.GpuPool, "create", classmethod(lambda cls, **kwargs: _FailingSecondLeasePool())
    )

    if not lease_timeout:
        with pytest.raises(TimeoutError, match="GPU lease wait exhausted"):
            run_experiment(exp)
        return

    winners = run_experiment(exp)

    assert winners["p"].trial_number == 0
    completion = winners["p"].completion
    assert completion["incomplete"] is True
    assert completion["reason"] == "timeout"


@pytest.mark.integration
def test_run_timeout_refuses_incomplete_winner(tmp_path: Path) -> None:
    exp = _sleeping_score_experiment(
        tmp_path,
        experiment="run_timeout",
        timeout_seconds_per_run=0.2,
    )

    with pytest.raises(TimeoutError, match="run guard"):
        run_experiment(exp)


def test_extraction_past_deadline_fails_the_trial(tmp_path: Path) -> None:
    """A successful trainer whose evidence lands past the deadline cannot count.

    ``timeout_seconds_per_phase``/``timeout_seconds_per_run`` bound the whole
    trial; before review v0.5.17 / blocker 8 extraction ran with no deadline
    at all, so a run could exceed both configured limits and still publish a
    complete winner.
    """
    experiment = make_experiment(workdir=tmp_path)
    (tmp_path / "r.json").write_text('{"x": 1.0}')
    executed = ExecutedTrial(
        ctx=TrialContext(
            experiment="t",
            phase="p",
            trial_id=0,
            generation_id="generation-test",
            attempt_id="attempt-test",
            overrides_sha256="0" * 64,
            trial_dir=tmp_path,
            run_name="t-p-0-attempt-test",
            return_code=0,
            duration_seconds=0.1,
        ),
        process=ProcessResult(
            return_code=0,
            timed_out=False,
            pid=123,
            duration_seconds=0.1,
        ),
    )

    result = extract_trial_result(
        experiment=experiment,
        executed=executed,
        deadline=time.monotonic() - 1.0,
    )

    assert result.metric is None
    assert result.feasible is False
    assert result.failure_reason is not None
    assert "wallclock deadline exceeded" in result.failure_reason
    assert result.deadline_exhausted is True


def test_process_failure_after_deadline_is_not_relabelled(tmp_path: Path) -> None:
    """An elapsed clock is not causal when the trainer already failed."""
    experiment = make_experiment(workdir=tmp_path)
    executed = ExecutedTrial(
        ctx=TrialContext(
            experiment="t",
            phase="p",
            trial_id=0,
            generation_id="generation-test",
            attempt_id="attempt-test",
            overrides_sha256="0" * 64,
            trial_dir=tmp_path,
            run_name="t-p-0-attempt-test",
            return_code=1,
            duration_seconds=0.1,
        ),
        process=ProcessResult(
            return_code=1,
            timed_out=False,
            pid=123,
            duration_seconds=0.1,
            failure_reason="trainer failed",
        ),
    )

    result = extract_trial_result(
        experiment=experiment,
        executed=executed,
        deadline=time.monotonic() - 1.0,
    )

    assert result.failure_reason == "trainer failed"
    assert result.deadline_exhausted is False


@pytest.mark.parametrize(
    ("capped_by_deadline", "timed_out", "expected"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_trainer_timeout_attribution_follows_wallclock_cap(
    tmp_path: Path, capped_by_deadline: bool, timed_out: bool, expected: bool
) -> None:
    """A killed trainer is deadline-attributed only when its cap was the deadline."""
    experiment = make_experiment(workdir=tmp_path)
    failure = "timed out after 1.0s" if timed_out else "trainer failed"
    executed = ExecutedTrial(
        ctx=TrialContext(
            experiment="t",
            phase="p",
            trial_id=0,
            generation_id="generation-test",
            attempt_id="attempt-test",
            overrides_sha256="0" * 64,
            trial_dir=tmp_path,
            run_name="t-p-0-attempt-test",
            return_code=1,
            duration_seconds=0.1,
        ),
        process=ProcessResult(
            return_code=1,
            timed_out=timed_out,
            pid=123,
            duration_seconds=0.1,
            failure_reason=failure,
            timeout_capped_by_wallclock=capped_by_deadline,
        ),
    )

    result = extract_trial_result(experiment=experiment, executed=executed)

    assert result.failure_reason == failure
    assert result.deadline_exhausted is expected


def test_elapsed_phase_clock_does_not_relabel_completed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal ordinary failure stays NoFeasible after the phase clock advances."""
    import types

    import phasesweep.engine.phase as phase_mod

    real_now = time.monotonic()
    # Deadline creation and both prelaunch checks stay in budget. The next
    # observation occurs after the ordinary trainer failure is terminal.
    calls = iter([real_now, real_now, real_now, real_now + 1_000.0])
    monkeypatch.setattr(
        phase_mod,
        "time",
        types.SimpleNamespace(
            monotonic=lambda: next(calls, real_now + 1_000.0),
            sleep=lambda _seconds: None,
        ),
    )
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command="false {overrides}",
        override_format="argparse",
        n_trials=1,
        max_consecutive_failures=10,
        timeout_seconds_per_phase=100.0,
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)


@pytest.mark.integration
def test_unrelated_launch_timeout_is_not_relabelled_as_phase_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only GPU-lease timeout is deadline policy; launch timeouts stay fatal."""
    storage = f"journal:///{tmp_path / 'timeout-cause.journal'}"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        n_trials=1,
        gpu_policy="none",
        timeout_seconds_per_phase=100.0,
    )

    def unrelated_timeout(**_kwargs: object) -> None:
        raise TimeoutError("injected non-deadline launch timeout")

    monkeypatch.setattr("phasesweep.engine.phase.launch_trial", unrelated_timeout)

    with pytest.raises(TimeoutError, match="injected non-deadline launch timeout"):
        run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(storage))
    assert study.user_attrs[PHASE_ABORT_ATTR]["policy"] == "unexpected_objective_exception"


@pytest.mark.integration
def test_slow_extraction_cannot_publish_complete_past_phase_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer repro (review v0.5.17 / blocker 8): quick trainer, slow extraction.

    The trainer finishes inside the 0.5s budget but extraction takes 3s. The
    run used to publish ``incomplete: false`` more than 2x past both limits;
    it must now report a timeout and publish nothing.
    """
    trainer = write_trainer(tmp_path, "print('x=1.0')")
    exp = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        n_trials=1,
        timeout_seconds_per_phase=0.5,
    )

    import phasesweep.engine.trial as trial_mod

    real_extractor = trial_mod.run_extractor

    def slow_extractor(*args, **kwargs):
        time.sleep(3.0)
        return real_extractor(*args, **kwargs)

    monkeypatch.setattr("phasesweep.engine.trial.run_extractor", slow_extractor)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_experiment(exp)
    elapsed = time.monotonic() - started

    assert elapsed < 20.0
    assert not _last_successful_generation_path(exp).exists()
