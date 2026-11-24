"""A changed reading setting re-reads the kept evidence of finished trials and retrains nothing."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Constraint, Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Sampler
from phasesweep.engine.errors import (
    PhaseTimeoutError,
    StudyFingerprintMismatchError,
    TrialTargetRefusedError,
    WinnerIntegrityError,
)
from phasesweep.engine.ledger import _resolve_storage, validate_ledger
from phasesweep.engine.preview import validate_trial_targets
from phasesweep.engine.read import inspect_status
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FAILURE_REASON_ATTR,
    PHASE_DECISION_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.evidence.baseline import EVIDENCE_BASELINE_FILE
from phasesweep.evidence.models import JsonExtractor, LogRegexExtractor
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    record_accepted_partial_timeout,
    study_name,
    write_experiment_config,
    write_trainer,
)
from tests.ledger_fixtures import tree_changes, tree_snapshot


def _counting_trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Drop a trainer that appends one line per launch and writes loss, acc and mem for ``--x``."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, default=1.0)
        ap.add_argument("--y", type=float, default=0.0)
        args, _ = ap.parse_known_args()
        with open({str(launches)!r}, "a") as handle:
            handle.write("launch\\n")
        loss = (args.x - 3) ** 2 + args.y
        Path(args.out).write_text(json.dumps({{"loss": loss, "acc": args.x, "mem": 10 * args.x}}))
        print(f"loss={{loss}}")
        """,
    )
    return trainer, launches


def _launches(path: Path) -> int:
    """Count the trainer's launches so far."""
    return len(path.read_text().splitlines()) if path.exists() else 0


def _metric(key: str, goal: str = "minimize") -> Metric:
    """Read ``key`` from the trainer's ``r.json``."""
    return Metric(
        name="loss", goal=goal, extractor=JsonExtractor(type="json", path="r.json", key=key)
    )


def _phases(*, child: bool = False) -> list[Phase]:
    """A three-cell grid over x, and optionally a child that inherits it and searches y."""
    phases = [
        Phase(
            name="p",
            n_trials=3,
            gpu_policy="none",
            max_consecutive_failures=5,
            sampler=Sampler(type="grid"),
            search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 4])},
        )
    ]
    if child:
        phases.append(
            Phase(
                name="c",
                inherits=["p"],
                n_trials=2,
                gpu_policy="none",
                max_consecutive_failures=5,
                sampler=Sampler(type="grid"),
                search_space={"y": CategoricalParam(type="categorical", choices=[0, 1])},
            )
        )
    return phases


def _experiment(tmp_path: Path, trainer: Path, metric: Metric, **extra: object) -> Experiment:
    """The experiment under ``tmp_path`` with the given reading settings."""
    phases = extra.pop("phases", None) or _phases()
    return make_experiment(
        persistent=tmp_path, trainer=trainer, metric=metric, phases=phases, **extra
    )


def phase_study_of(experiment: Experiment, phase: str = "p") -> optuna.Study:
    """The study the current config addresses for a phase that inherits nothing."""
    return optuna.load_study(
        study_name=study_name(experiment, phase),
        storage=_resolve_storage(experiment.resolved_storage),
    )


def test_a_misspelled_key_is_corrected_by_re_reading_every_finished_trial(tmp_path: Path) -> None:
    """Every trial failed at extraction under a typo; the corrected key completes them all unlaunched."""
    trainer, launches = _counting_trainer(tmp_path)
    typo = _experiment(tmp_path, trainer, _metric("los"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(typo)
    assert _launches(launches) == 3

    fixed = _experiment(tmp_path, trainer, _metric("loss"))
    winners = run_experiment(fixed)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 4}
    trials = phase_study_of(fixed).get_trials()
    assert [trial.state for trial in trials] == [optuna.trial.TrialState.COMPLETE] * 3
    assert sorted(trial.value for trial in trials) == [1.0, 1.0, 4.0]
    assert [t.number for t in trials] == [t.number for t in phase_study_of(typo).get_trials()]


def test_a_flipped_goal_re_reads_into_a_study_of_the_other_direction(tmp_path: Path) -> None:
    """Maximizing the same evidence flips the winner; the trials keep their numbers and directories."""
    trainer, launches = _counting_trainer(tmp_path)
    minimize = _experiment(tmp_path, trainer, _metric("loss"))
    assert run_experiment(minimize)["p"].effective_overrides == {"x": 4}
    before = {t.number: t.user_attrs[TRIAL_DIR_ATTR] for t in phase_study_of(minimize).get_trials()}

    maximize = _experiment(tmp_path, trainer, _metric("loss", goal="maximize"))
    winners = run_experiment(maximize)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 1}
    study = phase_study_of(maximize)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert {t.number: t.user_attrs[TRIAL_DIR_ATTR] for t in study.get_trials()} == before


@pytest.mark.parametrize("fixed", [False, True], ids=["grid", "fixed"])
@pytest.mark.parametrize("private", [False, True], ids=["shared-json", "private-stdout"])
def test_reread_attributes_results_to_their_attempt_and_keeps_the_retry_budget(
    tmp_path: Path, fixed: bool, private: bool
) -> None:
    """A directory retry preserves its own result and rereading never schedules a new one.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    :param bool fixed: Retry a fixed configuration instead of a grid cell.
    :param bool private: Read the attempt's private captured stream instead of shared JSON.
    """
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none") if fixed else _phases()[0]
    target = phase.n_trials + 1
    retried = phase.model_copy(update={"n_trials": target})
    for planned in (phase, retried):
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[planned]))

    metric = (
        Metric(extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)"))
        if private
        else _metric("loss")
    )
    corrected = _experiment(tmp_path, trainer, metric, phases=[retried])
    run_experiment(corrected)
    trials = phase_study_of(corrected).get_trials()
    first, last = trials[0], trials[-1]
    assert last.user_attrs[RERUN_OF_ATTR] == first.number
    assert last.user_attrs[TRIAL_DIR_ATTR] == first.user_attrs[TRIAL_DIR_ATTR]
    assert last.state == optuna.trial.TrialState.COMPLETE
    if private:
        assert first.state == optuna.trial.TrialState.COMPLETE
    else:
        assert first.state == optuna.trial.TrialState.FAIL
        assert "changed after this attempt finished" in first.user_attrs[FAILURE_REASON_ATTR]

    validate_trial_targets(corrected)
    run_experiment(corrected, dry_run=True)
    run_experiment(corrected)
    too_many = retried.model_copy(update={"n_trials": target + 1})
    with pytest.raises(TrialTargetRefusedError):
        validate_trial_targets(_experiment(tmp_path, trainer, metric, phases=[too_many]))
    assert _launches(launches) == target


def test_reread_does_not_score_a_file_overwritten_from_another_reading_study(
    tmp_path: Path,
) -> None:
    """A later retry's shared file is unavailable to a reread from a sibling study.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    """
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        args = argparse.ArgumentParser()
        args.add_argument("--out", type=Path, required=True)
        out = args.parse_known_args()[0].out
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            handle.write("launch\\n")
        count = len(launches.read_text().splitlines())
        out.write_text(json.dumps({{"loss": count}}))
        if count == 4:
            raise SystemExit(3)
        """,
    )
    phase = _phases()[0]
    older = _experiment(tmp_path, trainer, _metric("typo"))
    sibling = _experiment(tmp_path, trainer, _metric("another-typo"))
    retried = _experiment(
        tmp_path, trainer, _metric("typo"), phases=[phase.model_copy(update={"n_trials": 4})]
    )
    for planned in (older, sibling, retried):
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(planned)

    corrected = _experiment(tmp_path, trainer, _metric("loss"))
    assert run_experiment(corrected)["p"].metric == 2
    trials = phase_study_of(corrected).get_trials()
    assert len(trials) == len(phase_study_of(sibling).get_trials()) == 3
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert "changed after this attempt finished" in trials[0].user_attrs[FAILURE_REASON_ATTR]
    assert _launches(launches) == 4


def test_reread_refuses_an_older_file_reached_through_a_retargeted_link(tmp_path: Path) -> None:
    """A redirected result path gives no score even when its new target predates the retry.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import os
        from pathlib import Path
        root = Path(os.environ["PHASESWEEP_TRIAL_DIR"])
        latest = root / "latest"
        if latest.is_symlink():
            latest.unlink()
            latest.symlink_to("next")
        else:
            for checkpoint, value in (("prior", 0.1), ("next", 0.2)):
                directory = root / checkpoint
                directory.mkdir()
                (directory / "loss.json").write_text(f'{{"loss": {value}}}')
            latest.symlink_to("prior")
            (root / "r.json").symlink_to("latest/loss.json")
        """,
    )
    first = Phase(name="p", n_trials=1, gpu_policy="none")
    retried = first.model_copy(update={"n_trials": 2})
    for phase in (first, retried):
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[phase]))

    corrected = _experiment(tmp_path, trainer, _metric("loss"), phases=[retried])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(corrected)
    older, retry = phase_study_of(corrected).get_trials()
    assert "changed after this attempt finished" in older.user_attrs[FAILURE_REASON_ATTR]
    assert "last changed before this attempt started" in retry.user_attrs[FAILURE_REASON_ATTR]


@pytest.mark.integration
@pytest.mark.skipif(not hasattr(time, "tzset"), reason="the host cannot change process time zones")
def test_reread_uses_the_next_launch_instead_of_local_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The middle of three attempts owns the shared file it rewrote, across time zones.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    :param pytest.MonkeyPatch monkeypatch: Controller's local time zone for each run.
    """
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        args = argparse.ArgumentParser()
        args.add_argument("--out", type=Path, required=True)
        out = args.parse_known_args()[0].out
        launches = Path({str(launches)!r})
        with launches.open("a") as handle:
            handle.write("launch\\n")
        count = len(launches.read_text().splitlines())
        if count < 3:
            out.write_text(json.dumps({{"loss": count}}))
        """,
    )
    phase = Phase(name="p", n_trials=1, gpu_policy="none")
    env = {"TZ": "UTC0"}
    try:
        with monkeypatch.context() as controller:
            for target, zone in enumerate(("UTC-10", "UTC0", "UTC+10"), 1):
                controller.setenv("TZ", zone)
                time.tzset()
                phase = phase.model_copy(update={"n_trials": target})
                with pytest.raises(NoFeasibleTrialError):
                    run_experiment(
                        _experiment(tmp_path, trainer, _metric("typo"), phases=[phase], env=env)
                    )

            corrected = _experiment(tmp_path, trainer, _metric("loss"), phases=[phase], env=env)
            assert run_experiment(corrected)["p"].metric == 2
    finally:
        time.tzset()

    first, middle, last = phase_study_of(corrected).get_trials()
    assert len({trial.user_attrs[TRIAL_DIR_ATTR] for trial in (first, middle, last)}) == 1
    assert [trial.value for trial in (first, middle, last)] == [None, 2, None]
    assert "changed after this attempt finished" in first.user_attrs[FAILURE_REASON_ATTR]
    assert "no fresh write since launch" in last.user_attrs[FAILURE_REASON_ATTR]
    assert _launches(launches) == 3


@pytest.mark.integration
def test_reread_refuses_undated_reuse_only_for_shared_evidence(tmp_path: Path) -> None:
    """A retry's missing baseline prevents dating its shared JSON but leaves private logs usable.

    :param Path tmp_path: Scratch directory for the experiment and trainer.
    """
    trainer, launches = _counting_trainer(tmp_path)
    phase = Phase(name="p", n_trials=1, gpu_policy="none")
    retried = phase.model_copy(update={"n_trials": 2})
    original = _experiment(tmp_path, trainer, _metric("typo"), phases=[retried])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_experiment(tmp_path, trainer, _metric("typo"), phases=[phase]))
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    _, retry = phase_study_of(original).get_trials()
    records = Path(retry.user_attrs[TRIAL_DIR_ATTR]).parent / "attempts"
    (records / retry.user_attrs[ATTEMPT_ID_ATTR] / EVIDENCE_BASELINE_FILE).unlink()

    shared = _experiment(tmp_path, trainer, _metric("loss"), phases=[retried])
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(shared)
    opener, _ = phase_study_of(shared).get_trials()
    reason = opener.user_attrs[FAILURE_REASON_ATTR]
    assert "cannot be attributed to this attempt" in reason
    assert f"launch time of attempt {retry.user_attrs[ATTEMPT_ID_ATTR]!r}" in reason

    log = Metric(extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)"))
    private = _experiment(tmp_path, trainer, log, phases=[retried])
    assert run_experiment(private)["p"].metric == 4
    assert [trial.value for trial in phase_study_of(private).get_trials()] == [4, None]
    assert _launches(launches) == 2


def test_an_added_constraint_is_measured_for_every_finished_trial(tmp_path: Path) -> None:
    """The new constraint reads each kept result, so the trial it rules out loses the phase."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))
    mem = Constraint(
        name="mem", extractor=JsonExtractor(type="json", path="r.json", key="mem"), max=25
    )

    constrained = _experiment(tmp_path, trainer, _metric("loss"), constraints=[mem])
    winners = run_experiment(constrained)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 2}
    measured = {
        trial.params["x"]: trial.user_attrs["constraint:mem"]
        for trial in phase_study_of(constrained).get_trials()
    }
    assert measured == {1: 10.0, 2: 20.0, 4: 40.0}


def test_evidence_deleted_since_fails_only_its_trial_with_the_readers_reason(
    tmp_path: Path,
) -> None:
    """A trial whose result file is gone fails with the extractor's reason; the others re-read."""
    trainer, launches = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("loss"))
    run_experiment(original)
    lost = next(t for t in phase_study_of(original).get_trials() if t.params["x"] == 2)
    (Path(lost.user_attrs[TRIAL_DIR_ATTR]) / "r.json").unlink()

    changed = _experiment(tmp_path, trainer, _metric("acc"))
    winners = run_experiment(changed)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 1}
    by_x = {t.params["x"]: t for t in phase_study_of(changed).get_trials()}
    assert by_x[2].state == optuna.trial.TrialState.FAIL
    assert "JSON file not found" in by_x[2].user_attrs[FAILURE_REASON_ATTR]
    assert (by_x[1].value, by_x[4].value) == (1.0, 4.0)


def test_a_child_follows_its_re_read_parent_from_its_own_study(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With the parent's winner unchanged, the child re-reads its own trials: nothing launches."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss"), phases=_phases(child=True)))
    assert _launches(launches) == 5
    regex = Metric(
        name="loss",
        extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.eE+-]+)"),
    )

    changed = _experiment(tmp_path, trainer, regex, phases=_phases(child=True))
    before = tree_snapshot(tmp_path)
    caplog.set_level(logging.INFO)
    run_experiment(changed, dry_run=True)
    assert "phase=c study and launch count depend on the inherited values selected by 'p' (x)" in (
        caplog.text
    )
    assert "phase=c would launch" not in caplog.text
    assert "placeholders until 'p' publishes a winner: x" in caplog.text
    assert (
        "DRY RUN total: no launches for phases with known counts. "
        "Launch counts depend on inherited values for 1 phase"
    ) in caplog.text

    caplog.clear()
    # A child with no studies will need its full budget under every parent winner.
    new_child = changed.phases[1].model_copy(update={"name": "new_child"})
    run_experiment(
        changed.model_copy(update={"phases": [*changed.phases, new_child]}), dry_run=True
    )
    assert "DRY RUN phase=new_child would launch 2 trials" in caplog.text
    assert (
        "DRY RUN total: would launch 2 trials across 1 phase. "
        "Launch counts depend on inherited values for 1 phase"
    ) in caplog.text
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 5

    winners = run_experiment(changed)

    assert _launches(launches) == 5
    assert winners["p"].effective_overrides == {"x": 4}
    assert winners["c"].effective_overrides == {"x": 4, "y": 0}


@pytest.mark.parametrize("measured", [0, 1])
def test_reread_keeps_the_partial_timeout_boundary_with_the_new_measurement_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, measured: int
) -> None:
    """A timed-out fill replays its boundary and selects only measurements the new reader has."""
    trainer, launches = _counting_trainer(tmp_path)
    short = _phases()[0].model_copy(update={"n_trials": 2, "allow_partial_grid": True})
    original = _experiment(tmp_path, trainer, _metric("loss"), phases=[short])
    run_experiment(original)
    source = phase_study_of(original)
    for trial in source.get_trials()[measured:]:
        (Path(trial.user_attrs[TRIAL_DIR_ATTR]) / "r.json").write_text('{"loss": 1}')
    record_accepted_partial_timeout(source, 3)
    decision = source.user_attrs[PHASE_DECISION_ATTR]
    accepting = short.model_copy(update={"n_trials": 3, "allow_incomplete_on_timeout": True})
    changed = _experiment(tmp_path, trainer, _metric("acc"), phases=[accepting])

    caplog.set_level(logging.INFO)
    before = tree_snapshot(tmp_path)
    run_experiment(changed, dry_run=True)
    assert (
        "accepted a partial result at 2 of 3 trials: would select from the re-read evidence "
        "and launch no trials"
    ) in caplog.text
    refusing = accepting.model_copy(update={"allow_incomplete_on_timeout": False})
    with pytest.raises(WinnerIntegrityError, match="accepted an incomplete result at 2/3"):
        run_experiment(
            _experiment(tmp_path, trainer, _metric("acc"), phases=[refusing]), dry_run=True
        )
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}

    for _ in range(2):
        if measured:
            winner = run_experiment(changed)["p"]
            assert winner.completion["incomplete"]
            assert winner.completion["measured_trials"] == measured
        else:
            with pytest.raises(PhaseTimeoutError, match="no winner can be selected"):
                run_experiment(changed)

    assert _launches(launches) == 2
    assert phase_study_of(changed).user_attrs[PHASE_DECISION_ATTR] == {
        **decision,
        "measured_trials": measured,
    }
    assert source.user_attrs[PHASE_DECISION_ATTR]["measured_trials"] == 2


def test_a_changed_trainer_beside_a_changed_reader_is_refused_not_retrained(
    tmp_path: Path,
) -> None:
    """A study with the same inherited values but another trainer config is refused by name."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))

    both = _experiment(tmp_path, trainer, _metric("acc"), trainer_config={"y": 5})
    with pytest.raises(StudyFingerprintMismatchError, match="trainer_config"):
        run_experiment(both)
    assert _launches(launches) == 3


def test_status_names_the_run_that_fills_a_phase_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before the fill, status says the phase has not started under the current settings."""
    trainer, _launches_path = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("loss"))
    run_experiment(original)
    changed = _experiment(tmp_path, trainer, _metric("acc"))

    refusals = inspect_status(changed, validate_ledger(changed)).refusals

    notice = refusals["p"]
    assert str(notice) == (
        "Phase 'p' has not started under the current reading settings "
        '(metric.extractor.key: "loss" -> "acc").'
    )
    assert notice.remediation == (
        "Run the same configuration to re-read 3 finished trials from kept evidence without "
        "retraining those trials."
    )
    config = write_experiment_config(tmp_path / "sweep config.yaml", original)
    overrides = ["--set", 'metric.extractor.key="acc"']
    for options in ([], ["--trials"]):
        assert invoke_cli_boundary(["status", str(config), *overrides, *options], monkeypatch) == 0
        status = capsys.readouterr().out
        (command,) = [line.lstrip("# ") for line in status.splitlines() if "phasesweep run" in line]
        assert "fix: Run the same configuration to re-read" in status
        assert shlex.split(command) == ["phasesweep", "run", str(config), *overrides]


def test_a_dry_run_says_it_would_re_read_and_writes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The preview names the re-read and launches no trainer; the ledger and runs stay as they were."""
    trainer, launches = _counting_trainer(tmp_path)
    run_experiment(_experiment(tmp_path, trainer, _metric("loss")))
    changed = _experiment(tmp_path, trainer, _metric("acc"))
    before = tree_snapshot(tmp_path)
    caplog.set_level(logging.INFO)

    run_experiment(changed, dry_run=True)

    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 3
    assert (
        'DRY RUN phase=p reading settings changed (metric.extractor.key: "loss" -> "acc"): '
        "would re-read 3 finished trials from kept evidence; nothing would retrain"
    ) in [record.getMessage() for record in caplog.records]
    assert "would select from the re-read evidence and launch no trials" in caplog.text
    assert "DRY RUN total: would launch 0 trials across 0 phases" in caplog.text
    assert "phase replays" not in caplog.text


def test_validate_counts_kept_grid_trials_when_the_reading_settings_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The read-only ceiling includes available attempts without reading their evidence."""
    trainer, launches = _counting_trainer(tmp_path)
    original = _experiment(tmp_path, trainer, _metric("missing"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(original)
    phase = _phases()[0]
    retry = _experiment(
        tmp_path,
        trainer,
        _metric("still_missing"),
        phases=[phase.model_copy(update={"n_trials": 4})],
    )
    retry_path = write_experiment_config(tmp_path / "retry.yaml", retry)
    oversized = _experiment(
        tmp_path,
        trainer,
        _metric("still_missing"),
        phases=[phase.model_copy(update={"n_trials": 7})],
    )
    oversized_path = write_experiment_config(tmp_path / "oversized.yaml", oversized)
    before = tree_snapshot(tmp_path)

    assert invoke_cli_boundary(["validate", str(retry_path)], monkeypatch) == 0
    assert invoke_cli_boundary(["validate", str(oversized_path)], monkeypatch) == 1
    assert "allow at most 6" in capsys.readouterr().err
    assert tree_changes(before, tree_snapshot(tmp_path)) == {}
    assert _launches(launches) == 3


@pytest.mark.integration
def test_a_fill_killed_part_way_is_finished_by_the_next_run(tmp_path: Path) -> None:
    """A runner killed after one copied trial leaves a fill the next run completes in place."""
    trainer, launches = _counting_trainer(tmp_path)
    typo = _experiment(tmp_path, trainer, _metric("los"))
    with pytest.raises(NoFeasibleTrialError, match="No completed trials"):
        run_experiment(typo)
    fixed = _experiment(tmp_path, trainer, _metric("loss"))
    config = write_experiment_config(tmp_path / "fixed.yaml", fixed)
    killer = textwrap.dedent(
        """
        import os, sys
        import optuna
        from phasesweep.cli import main
        real = optuna.Study.add_trial
        def add_then_die(self, trial):
            real(self, trial)
            os._exit(9)
        optuna.Study.add_trial = add_then_die
        sys.argv = ["phasesweep", "run", sys.argv[1]]
        main()
        """
    )
    killed = subprocess.run(
        [sys.executable, "-c", killer, str(config)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert killed.returncode == 9, killed.stderr
    assert len(phase_study_of(fixed).get_trials()) == 1

    winners = run_experiment(fixed)

    assert _launches(launches) == 3
    assert winners["p"].effective_overrides == {"x": 4}
    filled = phase_study_of(fixed).get_trials()
    source = phase_study_of(typo).get_trials()
    assert [(t.number, t.params, t.user_attrs[TRIAL_DIR_ATTR]) for t in filled] == [
        (t.number, t.params, t.user_attrs[TRIAL_DIR_ATTR]) for t in source
    ]
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in filled)


@pytest.mark.integration
def test_an_interrupted_fill_keeps_its_source_trials_when_other_studies_grow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Growth of the source and a competing study keeps the interrupted fill's original cutoff."""
    trainer, launches = _counting_trainer(tmp_path)

    def config(key: str, target: int) -> Experiment:
        phase = Phase(
            name="p",
            n_trials=target,
            gpu_policy="none",
            sampler=Sampler(type="random", seed=0),
            search_space={"x": IntParam(type="int", low=1, high=100_000)},
        )
        return _experiment(tmp_path, trainer, _metric(key), phases=[phase])

    run_experiment(config("loss", 1))
    run_experiment(config("acc", 2))
    original = config("loss", 3)
    run_experiment(original)
    source_trials = phase_study_of(original).get_trials()
    interrupted = config("mem", 3)
    add_trial = optuna.Study.add_trial

    def stop_after_two(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        add_trial(study, trial)
        if len(study.get_trials(deepcopy=False)) == 2:
            raise RuntimeError("stop after the second copied trial")

    with (
        patch.object(optuna.Study, "add_trial", stop_after_two),
        pytest.raises(RuntimeError, match="stop after the second copied trial"),
    ):
        run_experiment(interrupted)
    run_experiment(config("loss", 4))
    run_experiment(config("acc", 5))
    before = _launches(launches)

    notice = inspect_status(interrupted, validate_ledger(interrupted)).refusals["p"]
    assert "re-reads 1 finished trial" in notice.remediation
    caplog.set_level(logging.INFO)
    run_experiment(interrupted, dry_run=True)
    assert "would re-read 1 finished trial from kept evidence" in caplog.text
    assert "target met (3 of 3 trials recorded)" in caplog.text

    run_experiment(interrupted)
    assert _launches(launches) == before
    filled = phase_study_of(interrupted).get_trials()
    assert [t.user_attrs[ATTEMPT_ID_ATTR] for t in filled] == [
        t.user_attrs[ATTEMPT_ID_ATTR] for t in source_trials
    ]
