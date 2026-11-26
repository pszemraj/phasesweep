"""Reuse sibling training at the target's own draws, budgets and failure boundaries."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Literal
from unittest.mock import DEFAULT, patch

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, FloatParam, IntParam, Sampler
from phasesweep.engine import phase as phase_ops
from phasesweep.engine import reread as reread_ops
from phasesweep.engine.errors import (
    LaunchEnvironmentError,
    PhaseTimeoutError,
    TrialTargetRefusedError,
)
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.optuna import _draw_positions
from phasesweep.engine.read import inspect_status, read_status
from phasesweep.engine.selection import PhaseTrialsFailedError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.study_policy import _allocation_contexts
from phasesweep.engine.trial_dirs import producing_attempts
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import (
    addressed_phase_study,
    make_experiment,
    record_accepted_partial_timeout,
    write_trainer,
)


def _trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Write two equivalent readings, and optionally fail x=2 after counting its launch."""
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, sys
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, default=1)
        ap.add_argument("--y", type=float, default=0)
        args = ap.parse_args()
        with Path({str(launches)!r}).open("a") as handle:
            print(args.x, file=handle)
        if args.x == 2 and Path({str(tmp_path / "fail")!r}).exists():
            sys.exit(7)
        loss = (args.x - 3) ** 2 + args.y
        Path(args.out).write_text(json.dumps({{"loss": loss, "acc": 100 + loss}}))
        """,
    )
    return trainer, launches


def _phase(sampler: Literal["grid", "random", "tpe"], n_trials: int) -> Phase:
    """Build a partial grid or seeded stochastic search with the same training contract."""
    space = (
        {"x": CategoricalParam(type="categorical", choices=[1, 2, 3])}
        if sampler == "grid"
        else {"x": FloatParam(type="float", low=0.1, high=3)}
    )
    settings = Sampler(type="grid") if sampler == "grid" else Sampler(type=sampler, seed=11)
    if sampler == "tpe":
        settings = settings.model_copy(update={"n_startup_trials": 1})
    return Phase(
        name="p",
        n_trials=n_trials,
        gpu_policy="none",
        allow_partial_grid=sampler == "grid",
        sampler=settings,
        search_space=space,
    )


def _config(
    tmp_path: Path,
    trainer: Path,
    phase: Phase,
    key: str = "loss",
    goal: str = "minimize",
) -> Experiment:
    """Read a trainer's selected JSON key without changing its launch command."""
    return make_experiment(
        persistent=tmp_path,
        trial_command=f"{shlex.quote(str(trainer))} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(
            name="score", goal=goal, extractor=JsonExtractor(type="json", path="r.json", key=key)
        ),
        phases=[phase],
    )


def _launches(path: Path) -> int:
    """Count the actual trainer executions."""
    return len(path.read_text().splitlines())


def test_recovered_abort_accepts_sibling_draw_under_raised_target(tmp_path: Path) -> None:
    """A recovery target covers an adopted draw on both the recovering run and its replay."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path)
        args, _ = parser.parse_known_args()
        count = Path(__file__).with_name("count")
        n = int(count.read_text()) + 1 if count.exists() else 1
        count.write_text(str(n))
        args.out.write_text(json.dumps({"alt": -n} if n == 2 else {"alt": -n, "loss": -n}))
        """,
    )
    phase = _phase("random", 2).model_copy(update={"max_consecutive_failures": 1})
    with pytest.raises(PhaseTrialsFailedError):
        run_experiment(_config(tmp_path, trainer, phase))
    raised = phase.model_copy(update={"n_trials": 3})
    donor = run_experiment(_config(tmp_path, trainer, raised, "alt"))["p"]
    target = _config(tmp_path, trainer, raised)
    for _ in range(2):
        winner = run_experiment(target)["p"]
        assert (winner.metric, winner.attempt_id) == (-3, donor.attempt_id)
    assert trainer.with_name("count").read_text() == "3"


def test_returning_grid_reuses_finished_draws_without_launch_resources(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preview and replay preserve sibling physical attempts even when the trainer is gone."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    raised = phase.model_copy(update={"n_trials": 3})
    wider = phase.model_copy(
        update={
            "n_trials": 4,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
        }
    )
    sibling = _config(tmp_path, trainer, wider, "acc")
    run_experiment(sibling)
    donors = addressed_phase_study(sibling).trials
    target = _config(tmp_path, trainer, raised)
    trainer.unlink()
    caplog.set_level(logging.INFO)

    run_experiment(target, dry_run=True)
    assert "would re-read 2 kept trials from other reading studies" in caplog.text
    assert "target met (3 of 3 trials recorded): would select from the re-read" in caplog.text
    run_experiment(target)
    rows = addressed_phase_study(target).trials

    assert _launches(launches) == 4
    assert sorted(row.params["x"] for row in rows) == [1, 2, 3]
    by_value = {row.params["x"]: row for row in donors}
    assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows] == [
        by_value[row.params["x"]].user_attrs[ATTEMPT_ID_ATTR] for row in rows
    ]
    assert all(RERUN_OF_ATTR not in row.user_attrs for row in rows)
    assert [row.value for row in rows] == [(row.params["x"] - 3) ** 2 for row in rows]
    assert read_status(target)["publication_integrity"] == "ok"
    run_experiment(target)
    assert _launches(launches) == 4


@pytest.mark.parametrize("sampler", ["random", "tpe"])
def test_repeated_configs_keep_independent_draw_slots(
    tmp_path: Path, sampler: Literal["random", "tpe"]
) -> None:
    """Independent equal-parameter draws take separate slots in every reading."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase(sampler, 1).model_copy(
        update={"search_space": {"x": IntParam(type="int", low=1, high=1)}}
    )
    run_experiment(_config(tmp_path, trainer, phase))
    sibling = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 3}), "acc")
    run_experiment(sibling)
    donors = addressed_phase_study(sibling).trials
    for target in (2, 3, 4):
        config = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": target}))
        run_experiment(config)
        rows = addressed_phase_study(config).trials
        assert len(rows) == target
        assert len({row.user_attrs[ATTEMPT_ID_ATTR] for row in rows}) == target
        assert _launches(launches) == max(3, target)
        assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows[:3]] == [
            row.user_attrs[ATTEMPT_ID_ATTR] for row in donors[:target]
        ]


def test_ordinary_trainer_failure_is_spent_without_another_launch(tmp_path: Path) -> None:
    """A sibling's nonzero trainer exit spends the same grid slot in the target."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    (tmp_path / "fail").touch()
    raised = phase.model_copy(update={"n_trials": 2})
    sibling = _config(tmp_path, trainer, raised, "acc")
    run_experiment(sibling)
    failed = addressed_phase_study(sibling).trials[-1]
    assert failed.state == optuna.trial.TrialState.FAIL
    trainer.unlink()
    target = _config(tmp_path, trainer, raised)

    run_experiment(target)

    adopted = addressed_phase_study(target).trials[-1]
    assert _launches(launches) == 2
    assert adopted.state == optuna.trial.TrialState.FAIL
    assert adopted.user_attrs[RETURN_CODE_ATTR] == 7
    assert adopted.user_attrs[ATTEMPT_ID_ATTR] == failed.user_attrs[ATTEMPT_ID_ATTR]


def test_grown_grid_reuses_a_later_donor_between_parallel_launches(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmatched next cell neither hides a later donor nor inflates the preview's count."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    sibling = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc")
    run_experiment(sibling)
    donor = addressed_phase_study(sibling).trials[-1]
    grown = phase.model_copy(
        update={
            "n_trials": 4,
            "n_jobs": 2,
            "allow_no_gpu_isolation": True,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
        }
    )
    target = _config(tmp_path, trainer, grown)
    caplog.set_level(logging.INFO)

    run_experiment(target, dry_run=True)
    assert "DRY RUN phase=p would launch 2 trials" in caplog.text
    assert _launches(launches) == 2
    run_experiment(target)

    rows = addressed_phase_study(target).trials
    assert _launches(launches) == 4
    assert sorted(row.params["x"] for row in rows) == [1, 2, 3, 4]
    assert donor.user_attrs[ATTEMPT_ID_ATTR] in [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows]
    assert read_status(target)["publication_integrity"] == "ok"


def test_returning_tpe_study_adopts_every_sibling_draw(tmp_path: Path) -> None:
    """A returning TPE study adopts every draw a sibling trained, whatever order it ranks them in."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("tpe", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    raised = phase.model_copy(update={"n_trials": 3})
    sibling = _config(tmp_path, trainer, raised, "acc", "maximize")
    run_experiment(sibling)
    donors = addressed_phase_study(sibling).trials
    target = _config(tmp_path, trainer, raised)

    run_experiment(target)

    rows = addressed_phase_study(target).trials
    assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows] == [
        row.user_attrs[ATTEMPT_ID_ATTR] for row in donors
    ]
    assert _launches(launches) == 3


def test_stochastic_preview_stops_after_the_final_possible_donor(tmp_path: Path) -> None:
    """A large top-up with one early donor forecasts only the positions it could supply."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("random", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc"))
    target = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 100000}))
    with patch.object(reread_ops, "_suggest_all", wraps=reread_ops._suggest_all) as suggest:
        run_experiment(target, dry_run=True)
    assert suggest.call_count < 10
    assert _launches(launches) == 2


def test_kept_grid_retry_and_new_cell_need_no_launch_preflight(tmp_path: Path) -> None:
    """A successful sibling retry and the next grid cell can cover one top-up together."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 2)
    (tmp_path / "fail").touch()
    run_experiment(_config(tmp_path, trainer, phase))
    (tmp_path / "fail").unlink()
    raised = phase.model_copy(update={"n_trials": 4})
    sibling = _config(tmp_path, trainer, raised, "acc")
    run_experiment(sibling)
    target = _config(tmp_path, trainer, raised)
    trainer.unlink()

    run_experiment(target, dry_run=True)
    assert len(addressed_phase_study(target).trials) == 2
    run_experiment(target)

    rows = addressed_phase_study(target).trials
    assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows] == [
        row.user_attrs[ATTEMPT_ID_ATTR] for row in addressed_phase_study(sibling).trials
    ]
    assert rows[2].user_attrs[RERUN_OF_ATTR] == 1
    assert RERUN_OF_ATTR not in rows[3].user_attrs
    assert _launches(launches) == 4
    assert read_status(target)["publication_integrity"] == "ok"


def test_adaptive_donor_positions_bound_coverage_before_adoption(tmp_path: Path) -> None:
    """A definite launch beyond the last donor is checked before the target accepts work."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("tpe", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc"))
    target = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 4}))
    trainer.unlink()

    for dry_run in (True, False):
        with pytest.raises(LaunchEnvironmentError, match="not an executable file"):
            run_experiment(target, dry_run=dry_run)
        study = addressed_phase_study(target)
        assert len(study.trials) == study.user_attrs[TRIAL_TARGET_ATTR] == 1
    assert _launches(launches) == 2


@pytest.mark.parametrize("interrupted_fill", [False, True])
def test_fill_preview_counts_donors_outside_its_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, interrupted_fill: bool
) -> None:
    """Bound new launches after a fresh or interrupted fill without writing its ledger."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase, "acc"))
    run_experiment(_config(tmp_path, trainer, phase))
    two = phase.model_copy(update={"n_trials": 2})
    sibling = _config(tmp_path, trainer, two, "acc")
    run_experiment(sibling)
    grown = two.model_copy(
        update={"search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])}}
    )
    run_experiment(_config(tmp_path, trainer, grown))
    target = _config(
        tmp_path, trainer, grown.model_copy(update={"n_trials": 3}), "loss", "maximize"
    )
    if interrupted_fill:
        with patch.object(reread_ops, "_reread_trial", wraps=reread_ops._reread_trial) as read:
            read.side_effect = [DEFAULT, reread_ops._reread_timeout(target, target.phases[0])]
            with pytest.raises(PhaseTimeoutError):
                run_experiment(target)
        assert len(addressed_phase_study(target).trials) == 1
    trainer.unlink()
    journal = tmp_path / "studies.journal"
    before = journal.read_bytes()
    caplog.set_level(logging.INFO)

    run_experiment(target, dry_run=True)

    assert "target met (3 of 3 trials recorded): would select from the re-read" in caplog.text
    sparse = _config(
        tmp_path, trainer, grown.model_copy(update={"n_trials": 4}), "loss", "maximize"
    )
    with pytest.raises(LaunchEnvironmentError, match="not an executable file"):
        run_experiment(sparse, dry_run=True)
    assert journal.read_bytes() == before
    run_experiment(target)
    assert _launches(launches) == 3
    rows = addressed_phase_study(target).trials
    assert [row.params["x"] for row in rows] == [3, 4, 2]
    assert (
        rows[-1].user_attrs[ATTEMPT_ID_ATTR]
        == addressed_phase_study(sibling).trials[-1].user_attrs[ATTEMPT_ID_ATTR]
    )
    assert read_status(target)["publication_integrity"] == "ok"


@pytest.mark.parametrize("sampler", ["random", "tpe"])
def test_stopped_adoption_takes_the_siblings_continuation_of_an_interrupted_draw(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, sampler: Literal["random", "tpe"]
) -> None:
    """An adoption stopped part-way keeps its rows; the next run takes the continuation, not a rerun."""
    stage = tmp_path / "stage"
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys, time
        from pathlib import Path
        with Path({str(launches)!r}).open("a") as log:
            print("launch", file=log)
        stage = Path({str(stage)!r}).read_text()
        if stage == "interrupt":
            time.sleep(30)
        if stage == "fail":
            raise SystemExit(9)
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text('{{"loss": 1, "acc": 2}}')
        """,
    )
    phase = _phase(sampler, 1).model_copy(
        update={
            "max_consecutive_failures": 10,
            "search_space": {"x": IntParam(type="int", low=1, high=1)},
        }
    )
    two = phase.model_copy(update={"n_trials": 2})
    stage.write_text("measure")
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase, "acc"))
    stage.write_text("interrupt")
    timed = two.model_copy(update={"timeout_seconds_per_phase": 3})
    with pytest.raises(PhaseTimeoutError, match="timed out via phase guard"):
        run_experiment(_config(tmp_path, trainer, timed))
    stage.write_text("measure")
    run_experiment(_config(tmp_path, trainer, two, "acc"))
    stage.write_text("fail")
    sibling = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 5}), "acc")
    run_experiment(sibling)
    stage.write_text("measure")
    target = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 4}), goal="maximize")
    caplog.set_level(logging.INFO)
    run_experiment(target, dry_run=True)
    assert "keep 1 interrupted trial as recorded" in caplog.text
    assert "would launch 0 trials" in caplog.text
    caplog.clear()
    with (
        patch.object(
            reread_ops,
            "_write_adopted",
            wraps=reread_ops._write_adopted,
            side_effect=[DEFAULT, DEFAULT, RuntimeError("stop after two rows")],
        ),
        pytest.raises(RuntimeError, match="stop after two rows"),
    ):
        run_experiment(target)
    assert len(addressed_phase_study(target).trials) == 2
    before = _launches(launches)
    caplog.set_level(logging.INFO)

    run_experiment(target, dry_run=True)
    assert "target met (4 of 4 trials recorded): would select from the re-read" in caplog.text
    assert _launches(launches) == before
    run_experiment(target)

    rows = addressed_phase_study(target).trials
    assert _launches(launches) == before
    assert RERUN_OF_ATTR not in rows[1].user_attrs
    assert rows[2].user_attrs[RERUN_OF_ATTR] == 1
    assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows[2:]] == [
        row.user_attrs[ATTEMPT_ID_ATTR] for row in addressed_phase_study(sibling).trials[2:5]
    ]


def test_fill_preview_refuses_an_unavoidable_launch_outside_the_donor_domain(
    tmp_path: Path,
) -> None:
    """A wider sibling's out-of-range proposal cannot cover a narrow adaptive top-up."""
    trainer, launches = _trainer(tmp_path)
    narrow = _phase("tpe", 1).model_copy(
        update={"search_space": {"x": IntParam(type="int", low=1, high=2)}}
    )
    run_experiment(_config(tmp_path, trainer, narrow))
    wider = narrow.model_copy(
        update={"n_trials": 3, "search_space": {"x": IntParam(type="int", low=1, high=10000)}}
    )
    sibling = _config(tmp_path, trainer, wider, "acc")
    run_experiment(sibling)
    assert addressed_phase_study(sibling).trials[1].params["x"] > 2
    target = _config(tmp_path, trainer, narrow.model_copy(update={"n_trials": 2}), goal="maximize")
    journal = tmp_path / "studies.journal"
    before = journal.read_bytes()
    offline = trainer.rename(tmp_path / "offline")

    with pytest.raises(LaunchEnvironmentError, match="not an executable file"):
        run_experiment(target, dry_run=True)

    assert journal.read_bytes() == before
    assert _launches(launches) == 3
    offline.rename(trainer)
    run_experiment(target)
    assert _launches(launches) == 4
    assert addressed_phase_study(target).trials[-1].params["x"] in (1, 2)


def test_adoption_stopped_before_launch_accepts_only_the_rows_it_holds(tmp_path: Path) -> None:
    """A Ctrl-C after adopting kept attempts leaves a floor of the adopted rows, not the requested target."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("random", 2)
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 4}), "acc"))
    target = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 5}))
    with (
        patch.object(phase_ops, "_launch_preflight", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        run_experiment(target)
    assert len(addressed_phase_study(target).trials) == 4

    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 4})))
    with pytest.raises(TrialTargetRefusedError, match="accepted a target of 4 finished trials"):
        run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 3})))
    assert _launches(launches) == 4


def test_mid_phase_adoption_deadline_preserves_new_training_on_restart(tmp_path: Path) -> None:
    """A stopped sibling read between batches keeps the just-finished new grid cell."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc"))
    grown = phase.model_copy(
        update={
            "n_trials": 4,
            "timeout_seconds_per_phase": 30,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
        }
    )
    target = _config(tmp_path, trainer, grown)
    with (
        patch.object(
            reread_ops, "_reread_trial", side_effect=reread_ops._reread_timeout(target, grown)
        ),
        pytest.raises(PhaseTimeoutError, match="phase guard after 2/4"),
    ):
        run_experiment(target)
    assert len(addressed_phase_study(target).trials) == 2
    run_experiment(target)
    assert sorted(row.params["x"] for row in addressed_phase_study(target).trials) == [1, 2, 3, 4]
    assert _launches(launches) == 4


def test_returning_study_continues_a_siblings_interrupted_attempt_in_place(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A draw a sibling's deadline interrupted reruns from its checkpoint, unwarned, in its directory."""
    launches = tmp_path / "launches"
    slow = tmp_path / "slow"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, time
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, default=1)
        args = ap.parse_args()
        step = Path(args.out).with_name("step")
        start = int(step.read_text()) if step.exists() else 0
        with Path({str(launches)!r}).open("a") as handle:
            print(args.x, start, file=handle)
        for done in range(start, 4):
            if Path({str(slow)!r}).exists():
                time.sleep(1)
            step.write_text(str(done + 1))
        Path(args.out).write_text(json.dumps({{"loss": args.x, "acc": 100 + args.x}}))
        """,
    )
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    slow.touch()
    timed = phase.model_copy(update={"n_trials": 2, "timeout_seconds_per_phase": 2})
    with pytest.raises(PhaseTimeoutError):
        run_experiment(_config(tmp_path, trainer, timed, "acc"))
    slow.unlink()
    interrupted = addressed_phase_study(_config(tmp_path, trainer, timed, "acc")).trials[1]
    target = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}))
    caplog.set_level(logging.INFO)

    run_experiment(target)

    assert "model can be overwritten" not in caplog.text
    rows = addressed_phase_study(target).trials
    resumed = launches.read_text().splitlines()[-1].split()
    assert len(launches.read_text().splitlines()) == 3
    assert int(resumed[1]) > 0
    assert rows[1].user_attrs[ATTEMPT_ID_ATTR] == interrupted.user_attrs[ATTEMPT_ID_ATTR]
    assert rows[2].user_attrs[RERUN_OF_ATTR] == 1
    assert rows[2].user_attrs[TRIAL_DIR_ATTR] == interrupted.user_attrs[TRIAL_DIR_ATTR]


def _failed_retry_sibling(tmp_path: Path) -> tuple[Path, Path, Phase]:
    """Run a grid cell, then a sibling whose x=2 trainer exits 7 and whose retry succeeds."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    (tmp_path / "fail").touch()
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 3}), "acc"))
    (tmp_path / "fail").unlink()
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 4}), "acc"))
    return trainer, launches, phase


def test_returning_and_fresh_readings_reach_one_verdict_over_a_kept_retry(tmp_path: Path) -> None:
    """A returning study and a new reading both adopt a failed cell and its retry, launching nothing."""
    trainer, launches, phase = _failed_retry_sibling(tmp_path)
    four = phase.model_copy(update={"n_trials": 4})
    for config in (
        _config(tmp_path, trainer, four),
        _config(tmp_path, trainer, four, "loss", "maximize"),
    ):
        run_experiment(config)
        rows = addressed_phase_study(config).trials
        assert sorted(row.params["x"] for row in rows) == [1, 2, 2, 3]
        assert sum(RERUN_OF_ATTR in row.user_attrs for row in rows) == 1
    assert _launches(launches) == 4


def test_the_run_names_a_kept_trainer_failure_as_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A copied nonzero trainer exit is reported as a trainer failure, apart from the re-reads."""
    trainer, launches, phase = _failed_retry_sibling(tmp_path)
    caplog.set_level(logging.INFO)

    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 4})))

    assert (
        'p  reading settings changed (metric.extractor.key: "acc" -> "loss"): re-read 2 kept '
        "trials from other reading studies (2 changed outcome), kept 1 trainer failure as "
        "recorded; nothing was retrained." in caplog.text
    )
    assert "unfinished" not in caplog.text
    assert _launches(launches) == 4


def test_a_child_never_adopts_a_draw_trained_under_another_parent_value(tmp_path: Path) -> None:
    """Only studies whose inherited values match the child's own give it kept attempts."""
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, default=1)
        ap.add_argument("--y", type=float, default=0)
        args = ap.parse_args()
        with Path({str(launches)!r}).open("a") as handle:
            print(args.x, args.y, file=handle)
        Path(args.out).write_text(json.dumps({{"loss": args.x + args.y, "alt": args.y - args.x}}))
        """,
    )
    parent = Phase(
        name="parent",
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )
    child = Phase(
        name="child",
        inherits=["parent"],
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="random", seed=3),
        search_space={"y": FloatParam(type="float", low=0, high=1)},
    )

    def config(key: str, goal: str) -> Experiment:
        experiment = _config(tmp_path, trainer, parent, key, goal)
        return experiment.model_copy(update={"phases": [parent, child]})

    run_experiment(config("loss", "minimize"))
    winners = run_experiment(config("alt", "minimize"))

    assert winners["parent"].params == {"x": 2}
    assert winners["child"].metric == pytest.approx(
        min(y - 2 for x, y in _child_launches(launches) if x == 2)
    )
    assert sorted({x for x, _ in _child_launches(launches)}) == [1, 2]
    assert len(_child_launches(launches)) == 4


def _child_launches(launches: Path) -> list[tuple[float, float]]:
    """Read the (x, y) of every child launch, which is any launch with a nonzero y."""
    rows = [tuple(map(float, line.split())) for line in launches.read_text().splitlines()]
    return [(x, y) for x, y in rows if y != 0]


def test_a_partly_adopted_study_reports_its_rows_not_a_pending_re_read(tmp_path: Path) -> None:
    """Status counts the rows an interrupted adoption wrote and names no pending re-read."""
    trainer, _ = _trainer(tmp_path)
    phase = _phase("grid", 3)
    run_experiment(_config(tmp_path, trainer, phase))
    target = _config(tmp_path, trainer, phase, "loss", "maximize")
    with (
        patch.object(
            reread_ops,
            "_write_adopted",
            wraps=reread_ops._write_adopted,
            side_effect=[DEFAULT, RuntimeError("stop after one row")],
        ),
        pytest.raises(RuntimeError, match="stop after one row"),
    ):
        run_experiment(target)

    status = read_status(target)["phases"][0]

    assert status["reread_pending"] is None
    assert status["trials"]["COMPLETE"] == 1


def test_an_adopted_attempt_belongs_to_the_study_that_launched_it(tmp_path: Path) -> None:
    """Adoption copies no allocation context, so each attempt resolves to its launching study."""
    trainer, _ = _trainer(tmp_path)
    phase = _phase("grid", 1)
    first = _config(tmp_path, trainer, phase)
    run_experiment(first)
    second = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc")
    run_experiment(second)
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2})))
    launched_by = {
        study.study_name: {
            row.user_attrs[ATTEMPT_ID_ATTR]
            for row in study.trials
            if row.user_attrs[GENERATION_ID_ATTR]
            in {record["generation_id"] for record in _allocation_contexts(study)}
        }
        for study in (addressed_phase_study(first), addressed_phase_study(second))
    }

    owners = producing_attempts((addressed_phase_study(first), addressed_phase_study(second)))

    assert all(len(attempts) == 1 for attempts in launched_by.values())
    assert {attempt_id: owner.study.study_name for attempt_id, owner in owners.items()} == {
        attempt_id: name for name, ids in launched_by.items() for attempt_id in ids
    }


def test_an_accepted_partial_result_replays_until_its_target_rises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A study at its accepted partial target adopts nothing; a raised target adopts the rest."""
    trainer, launches = _trainer(tmp_path)
    partial = _phase("random", 6).model_copy(
        update={"timeout_seconds_per_phase": 30, "allow_incomplete_on_timeout": True}
    )
    accepted = _config(tmp_path, trainer, partial)
    run_experiment(_config(tmp_path, trainer, partial.model_copy(update={"n_trials": 2})))
    record_accepted_partial_timeout(addressed_phase_study(accepted), 6)
    run_experiment(_config(tmp_path, trainer, _phase("random", 8), "acc"))
    assert _launches(launches) == 8
    caplog.set_level(logging.INFO)

    run_experiment(accepted, dry_run=True)
    replayed = run_experiment(accepted)["p"]

    assert "accepted a partial result at 2 of 6 trials" in caplog.text
    assert replayed.completion["incomplete"]
    assert len(addressed_phase_study(accepted).trials) == 2
    raised = _config(tmp_path, trainer, _phase("random", 8))
    run_experiment(raised)
    assert len(addressed_phase_study(raised).trials) == 8
    assert _launches(launches) == 8


def test_a_grown_grid_is_judged_with_the_cells_its_growth_adds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Growth and adoption cover a grown grid, so neither the preview nor preflight needs a trainer."""
    trainer, launches = _trainer(tmp_path)

    def grid(n_trials: int, top: int) -> Phase:
        space = {"x": CategoricalParam(type="categorical", choices=list(range(1, top + 1)))}
        return _phase("grid", n_trials).model_copy(update={"search_space": space})

    run_experiment(_config(tmp_path, trainer, grid(3, 3)))
    run_experiment(_config(tmp_path, trainer, grid(6, 6), "acc"))
    trainer.unlink()
    grown = _config(tmp_path, trainer, grid(6, 6))
    caplog.set_level(logging.INFO)

    run_experiment(grown, dry_run=True)
    run_experiment(grown)

    assert "target met (6 of 6 trials recorded)" in caplog.text
    assert sorted(row.params["x"] for row in addressed_phase_study(grown).trials) == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert _launches(launches) == 6


def test_a_preview_counts_launches_when_only_failing_re_reads_would_abort(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A streak only all-failing re-reads reach is one conditional line beside the launch count."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("random", 6).model_copy(update={"max_consecutive_failures": 5})
    run_experiment(_config(tmp_path, trainer, phase, "acc"))
    raised = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 8}))
    raised = raised.model_copy(update={"metric": raised.metric.model_copy(update={"max": 10})})
    caplog.set_level(logging.INFO)

    run_experiment(raised, dry_run=True)
    run_experiment(raised)

    assert "phase=p aborts if every re-read fails (max_consecutive_failures=5)" in caplog.text
    assert "DRY RUN phase=p would launch 2 trials" in caplog.text
    assert "previously aborted" not in caplog.text
    assert _launches(launches) == 8


def test_status_judges_a_grown_grid_target_over_the_attempts_it_adopts(tmp_path: Path) -> None:
    """A target only another reading's new cells and retry can fill reads as runnable, and runs."""
    trainer, launches = _trainer(tmp_path)

    def grid(n_trials: int, choices: list[int]) -> Phase:
        space = {"x": CategoricalParam(type="categorical", choices=choices)}
        return _phase("grid", n_trials).model_copy(update={"search_space": space})

    run_experiment(_config(tmp_path, trainer, grid(2, [1, 3])))
    run_experiment(_config(tmp_path, trainer, grid(2, [1, 3]), "acc"))
    (tmp_path / "fail").touch()
    run_experiment(_config(tmp_path, trainer, grid(3, [1, 3, 2])))
    (tmp_path / "fail").unlink()
    run_experiment(_config(tmp_path, trainer, grid(4, [1, 3, 2])))
    grown = _config(tmp_path, trainer, grid(4, [1, 3, 2]), "acc")

    assert "p" not in inspect_status(grown, validate_ledger(grown)).refusals
    run_experiment(grown)
    assert len(addressed_phase_study(grown).trials) == 4
    assert _launches(launches) == 4


def test_projecting_adoption_prints_no_optuna_study_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status of a phase other readings hold kept attempts for stays free of Optuna's notices."""
    trainer, _ = _trainer(tmp_path)
    run_experiment(_config(tmp_path, trainer, _phase("random", 3)))
    changed = _config(tmp_path, trainer, _phase("random", 4), "acc")
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    notices: list[str] = []
    handler = logging.Handler()
    monkeypatch.setattr(handler, "emit", lambda record: notices.append(record.getMessage()))
    logging.getLogger("optuna").addHandler(handler)
    try:
        inspect_status(changed, validate_ledger(changed))
    finally:
        logging.getLogger("optuna").removeHandler(handler)
        optuna.logging.set_verbosity(previous)

    assert not [notice for notice in notices if "A new study created in memory" in notice]


def test_adopted_interrupted_trials_rerun_together(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every interrupted attempt another reading kept is adopted, and their reruns share a batch."""
    stage = tmp_path / "stage"
    events = tmp_path / "events"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys, time
        from pathlib import Path
        events = Path({str(events)!r})
        with events.open("a") as log:
            print("start", file=log)
        if Path({str(stage)!r}).read_text() == "interrupt":
            time.sleep(30)
        deadline = time.monotonic() + 5
        while events.read_text().count("start") < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
        with events.open("a") as log:
            print("end", file=log)
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text('{{"loss": 1, "acc": 2}}')
        """,
    )
    phase = _phase("random", 2).model_copy(
        update={"n_jobs": 2, "allow_no_gpu_isolation": True, "timeout_seconds_per_phase": 3}
    )
    stage.write_text("interrupt")
    with pytest.raises(PhaseTimeoutError, match="timed out via phase guard"):
        run_experiment(_config(tmp_path, trainer, phase))
    stage.write_text("measure")
    changed = _config(
        tmp_path, trainer, phase.model_copy(update={"timeout_seconds_per_phase": None}), "acc"
    )
    caplog.set_level(logging.INFO)

    run_experiment(changed, dry_run=True)
    run_experiment(changed)

    assert "would launch 2 trials, 2 of them reruns" in caplog.text
    assert events.read_text().split() == ["start", "start", "start", "start", "end", "end"]
    assert len(addressed_phase_study(changed).trials) == 4


def test_an_accepted_partial_result_keeps_an_adopted_interrupted_trial_as_recorded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An interrupted attempt an accepted partial result holds is not said to continue."""
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys, time
        from pathlib import Path
        with Path({str(launches)!r}).open("a") as log:
            print("launch", file=log)
        if len(Path({str(launches)!r}).read_text().split()) == 3:
            time.sleep(30)
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text('{{"loss": 1, "acc": 2}}')
        """,
    )
    phase = _phase("random", 3).model_copy(
        update={"timeout_seconds_per_phase": 3, "allow_incomplete_on_timeout": True}
    )
    run_experiment(_config(tmp_path, trainer, phase))
    changed = _config(tmp_path, trainer, phase, "acc")
    caplog.set_level(logging.INFO)

    run_experiment(changed, dry_run=True)
    run_experiment(changed)

    assert "keep 1 interrupted trial as recorded" in caplog.text
    assert "kept 1 interrupted trial as recorded" in caplog.text
    assert "continue from its checkpoint" not in caplog.text
    assert _launches(launches) == 3


@pytest.mark.integration
@pytest.mark.parametrize("case", ["restart", "full-union", "interrupted-retry"])
def test_partial_boundary_replays_past_an_unrelated_grid_retry(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    case: Literal["restart", "full-union", "interrupted-retry"],
) -> None:
    """Keep the accepted cohort and retry ancestry across a stopped re-read and a later top-up."""
    (tmp_path / "case").write_text(case)
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json, time
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path)
        parser.add_argument("--x", type=int)
        args = parser.parse_args()
        count = Path(__file__).with_name("count")
        launch = int(count.read_text()) + 1 if count.exists() else 1
        count.write_text(str(launch))
        checkpoint = args.out.with_name("checkpoint")
        previous = checkpoint.read_text() if checkpoint.exists() else "fresh"
        with Path(__file__).with_name("starts").open("a") as starts:
            print(args.out.parent, previous, file=starts)
        checkpoint.write_text(str(launch))
        interrupted_retry = Path(__file__).with_name("case").read_text() == "interrupted-retry"
        if launch == (3 if interrupted_retry else 4):
            time.sleep(60)
        data = {"loss": args.x, "acc": args.x}
        if launch == 1:
            del data["loss"]
        if launch == (1 if interrupted_retry else 2):
            del data["acc"]
        args.out.write_text(json.dumps(data))
        """,
    )
    original = _phase("grid", 2).model_copy(
        update={
            "sampler": Sampler(type="grid", seed=19),
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
            "allow_incomplete_on_timeout": True,
        },
    )
    target = original.model_copy(update={"n_trials": 5})
    returning = _config(tmp_path, trainer, target)
    donor = _config(
        tmp_path, trainer, target.model_copy(update={"timeout_seconds_per_phase": 3}), "acc"
    )
    run_experiment(_config(tmp_path, trainer, original))
    run_experiment(_config(tmp_path, trainer, original, "acc"))
    donor_finished = 2 if case == "interrupted-retry" else 3
    assert run_experiment(donor)["p"].completion["finished_trials"] == donor_finished
    donor_rows = addressed_phase_study(donor).trials
    caplog.set_level(logging.INFO)

    if case == "full-union":
        earlier = _config(tmp_path, trainer, target.model_copy(update={"n_trials": 4}))
        run_experiment(earlier)
        original_rows = addressed_phase_study(earlier).trials
        run_experiment(returning, dry_run=True)
        assert "target met (5 of 5 trials recorded)" in caplog.text
        assert "would launch 0 trials" in caplog.text
        assert not run_experiment(returning)["p"].completion["incomplete"]
        assert trainer.with_name("count").read_text() == "6"
        rows = addressed_phase_study(returning).trials
        assert rows[:-1] == original_rows
        assert rows[-1].user_attrs[ATTEMPT_ID_ATTR] == donor_rows[2].user_attrs[ATTEMPT_ID_ATTR]
        assert rows[-1].user_attrs[RERUN_OF_ATTR] == 1
        return

    run_experiment(returning, dry_run=True)
    assert f"accepted a partial result at {donor_finished} of 5 trials" in caplog.text
    assert "would launch 0 trials" in caplog.text
    if case == "interrupted-retry":
        assert run_experiment(returning)["p"].completion["incomplete"]
        rows = addressed_phase_study(returning).trials
        assert len(rows) == 2
        assert donor_rows[2].user_attrs[RERUN_OF_ATTR] == 0
        assert all(
            row.user_attrs[ATTEMPT_ID_ATTR] != donor_rows[2].user_attrs[ATTEMPT_ID_ATTR]
            for row in rows
        )
        widened = target.model_copy(
            update={
                "n_trials": 6,
                "search_space": {
                    "x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4, 5])
                },
            }
        )
        run_experiment(_config(tmp_path, trainer, widened))
        starts = trainer.with_name("starts").read_text().splitlines()
        interrupted_dir, previous = starts[2].rsplit(" ", 1)
        assert previous == "1"
        assert starts[3] == f"{interrupted_dir} 3"
        assert len(starts) == 7
        return

    assert _draw_positions(donor_rows) == {0: 0, 1: 1, 2: 1, 3: 2}
    with (
        patch.object(
            reread_ops,
            "_write_adopted",
            wraps=reread_ops._write_adopted,
            side_effect=[DEFAULT, RuntimeError("stop after the first accepted row")],
        ),
        pytest.raises(RuntimeError, match="stop after the first accepted row"),
    ):
        run_experiment(returning)
    rows = addressed_phase_study(returning).trials
    assert len(rows) == 3
    assert rows[-1].user_attrs[ATTEMPT_ID_ATTR] == donor_rows[2].user_attrs[ATTEMPT_ID_ATTR]

    with patch.object(
        reread_ops, "_reread_trial", side_effect=AssertionError("row is already read")
    ):
        winner = run_experiment(returning)["p"]
    assert winner.completion["incomplete"]
    assert (winner.completion["finished_trials"], winner.completion["measured_trials"]) == (3, 2)
    assert trainer.with_name("count").read_text() == "4"
    rows = addressed_phase_study(returning).trials
    assert _draw_positions(rows) == _draw_positions(donor_rows)
    assert [trial.user_attrs[ATTEMPT_ID_ATTR] for trial in rows] == [
        trial.user_attrs[ATTEMPT_ID_ATTR] for trial in donor_rows
    ]
    assert rows[2].user_attrs[RERUN_OF_ATTR] == 1

    raised = _config(tmp_path, trainer, target.model_copy(update={"n_trials": 6}))
    run_experiment(raised)
    rows = addressed_phase_study(raised).trials
    assert trainer.with_name("count").read_text() == "7"
    assert _draw_positions(rows) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 0, 6: 3}
    assert rows[4].user_attrs[TRIAL_DIR_ATTR] == rows[3].user_attrs[TRIAL_DIR_ATTR]


@pytest.mark.integration
def test_a_raised_target_replaces_an_older_partial_decision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Replay the accepted sibling target after the original study accepted a smaller one."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json, time
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path)
        parser.add_argument("--x", type=float)
        args = parser.parse_args()
        count = Path(__file__).with_name("count")
        launch = int(count.read_text()) + 1 if count.exists() else 1
        count.write_text(str(launch))
        if launch in (2, 4):
            time.sleep(60)
        args.out.write_text(json.dumps({"loss": args.x, "acc": 100 + args.x}))
        """,
    )
    original = _phase("random", 3).model_copy(
        update={"allow_incomplete_on_timeout": True, "timeout_seconds_per_phase": 3}
    )
    target = original.model_copy(update={"n_trials": 4})
    donor = _config(tmp_path, trainer, target, "acc")
    returning = _config(
        tmp_path, trainer, target.model_copy(update={"timeout_seconds_per_phase": None})
    )
    assert (
        run_experiment(_config(tmp_path, trainer, original))["p"].completion["finished_trials"] == 1
    )
    assert run_experiment(donor)["p"].completion["finished_trials"] == 2
    caplog.set_level(logging.INFO)

    run_experiment(returning, dry_run=True)
    assert "accepted a partial result at 2 of 4 trials" in caplog.text
    assert "would launch 0 trials" in caplog.text
    for _ in range(2):
        winner = run_experiment(returning)["p"]
        assert winner.completion["incomplete"]
        assert winner.completion["finished_trials"] == 2
    assert trainer.with_name("count").read_text() == "4"
    rows = addressed_phase_study(returning).trials
    donor_rows = addressed_phase_study(donor).trials
    assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows] == [
        row.user_attrs[ATTEMPT_ID_ATTR] for row in donor_rows
    ]
