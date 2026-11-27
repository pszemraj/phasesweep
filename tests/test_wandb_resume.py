"""Every attempt of a trial logs to the trial's one W&B run, and a rerun is scored only once it logged."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import phasesweep.runtime.reaper as reaper
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric
from phasesweep.engine.errors import TrackingTargetUnavailableError
from phasesweep.engine.evidence import _verify_winner_evidence
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    EARLIER_ATTEMPT_RESULT_ATTR,
    LEFT_NOTHING_TO_SCORE_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _trial_attempt_dir,
)
from phasesweep.evidence.baseline import read_evidence_baseline
from phasesweep.evidence.models import WandbExtractor, WandbSummaryRequiredGate
from tests.conftest import make_experiment, phase_trials, write_experiment_config, write_trainer
from tests.recovery_helpers import runner_once


def test_env_wandb_resume_is_refused_whatever_its_value() -> None:
    """A configured WANDB_RESUME matches neither per-attempt policy, so config load refuses it by name."""
    extractor = WandbExtractor(type="wandb", entity="e", project="p", metric_key="x")
    with pytest.raises(
        ValueError,
        match=(
            r"env\.WANDB_RESUME conflicts with the resume policy PhaseSweep sets per attempt; "
            r"remove it\."
        ),
    ):
        make_experiment(metric=Metric(extractor=extractor), env={"WANDB_RESUME": "never"})


def _fake_wandb(install_sdk: Callable[[str], None], root: Path) -> SimpleNamespace:
    """Serve the real reader a run whose answer the ``mode`` file picks.

    ``logged`` is a finished run whose summary was just logged, ``left`` the crashed run an
    interrupted attempt leaves with an old ``_timestamp``, and ``hang`` touches ``hung``
    and blocks. ``idle`` is that crashed run, and ``settled`` that run ``killed`` at its rerun's
    launch and ``finished`` after it; ``absent`` has no run at the launch. Those three answer
    for the first trial's run, and any other run is ``logged``. Every run holds the row count
    in ``rows``, 5 without it. The launch read of a rerun in the first trial's directory, its
    run's first lookup, never hangs. Every lookup appends the path it asked for to ``lookups``.

    :param Callable[[str], None] install_sdk: The ``wandb_worker_sdk`` fixture's installer.
    :param Path root: Directory holding the mode, lookup, hang, row and launch files.
    :return SimpleNamespace: ``mode``, ``lookups``, ``hung`` and ``rows`` paths.
    """
    files = SimpleNamespace(
        mode=root / "mode", lookups=root / "lookups", hung=root / "hung", rows=root / "rows"
    )
    files.mode.write_text("logged")
    install_sdk(f"""
        import json, time
        from pathlib import Path

        class Api:
            def __init__(self, **kwargs):
                pass

            def run(self, path):
                mode = Path({str(files.mode)!r}).read_text()
                lookups = Path({str(files.lookups)!r})
                with lookups.open("a") as out:
                    out.write(path + "\\n")
                envs, rows = Path({str(root / "envs")!r}), Path({str(files.rows)!r})
                opener = json.loads(envs.read_text().splitlines()[0])[0] if envs.exists() else None
                first_run = path == f"e/p/{{opener}}"
                launch = first_run and lookups.read_text().split().count(path) == 1
                count = int(rows.read_text()) if rows.exists() else 5
                if mode == "hang" and not launch:
                    Path({str(files.hung)!r}).touch()
                    time.sleep(60)
                if mode in ("idle", "settled", "absent") and not first_run:
                    mode = "logged"
                if mode == "absent" and launch:
                    raise ValueError("Could not find run " + path)
                if mode in ("left", "idle", "settled"):
                    state = "crashed" if mode != "settled" else "killed" if launch else "finished"
                    summary = {{"x": 0.5, "_timestamp": 1.0}}
                else:
                    state, summary = "finished", {{"x": 1.0, "_timestamp": time.time()}}
                run = {{"state": state, "summary_metrics": summary, "historyLineCount": count}}
                return type("Run", (), run)()
    """)
    return files


def _interrupted_wandb_trial(tmp_path: Path, *, timeout_seconds: float) -> Experiment:
    """Leave a persistent W&B trial RUNNING after a runner SIGKILL, its trainer still running.

    Every launch appends its ``WANDB_RUN_ID``, ``WANDB_RESUME`` and attempt ID to ``envs``.
    The first touches ``first`` and sleeps until the next run's recovery stops it; later
    attempts log 3 rows to the fake run's count in ``rows`` when that file exists, and exit 0.

    :param Path tmp_path: Scratch directory.
    :param float timeout_seconds: The W&B reader's polling budget.
    :return Experiment: The one-trial experiment.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import json, os, time
        from pathlib import Path
        names = ["WANDB_RUN_ID", "WANDB_RESUME", "PHASESWEEP_ATTEMPT_ID"]
        with Path({str(tmp_path / "envs")!r}).open("a") as envs:
            envs.write(json.dumps([os.environ.get(name) for name in names]) + "\\n")
        first = Path({str(tmp_path / "first")!r})
        if not first.exists():
            first.touch()
            time.sleep(60)
        rows = Path({str(tmp_path / "rows")!r})
        if rows.exists():
            rows.write_text(str(int(rows.read_text()) + 3))
        """,
    )
    extractor = WandbExtractor(
        type="wandb",
        base_url="https://example.test",
        entity="e",
        project="p",
        metric_key="x",
        poll_seconds=0.05,
        timeout_seconds=timeout_seconds,
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, metric=Metric(extractor=extractor), n_trials=1
    )
    with runner_once(
        write_experiment_config(tmp_path / "experiment.yaml", experiment), tmp_path / "first"
    ):
        pass
    return experiment


def _launch_envs(tmp_path: Path) -> list[list[str | None]]:
    """Read what each launch recorded.

    :param Path tmp_path: Scratch directory holding ``envs``.
    :return list[list[str | None]]: ``[WANDB_RUN_ID, WANDB_RESUME, attempt ID]`` per launch.
    """
    return [json.loads(line) for line in (tmp_path / "envs").read_text().splitlines()]


@pytest.mark.integration
def test_a_rerun_resumes_its_trials_wandb_run_and_is_scored_from_it(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """The opener starts the trial's run with resume never; the rerun resumes that run with allow, is scored from it, and verifies as a winner."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=10)

    winner = run_experiment(experiment)["p"]
    first, rerun = phase_trials(experiment)
    opener = first.user_attrs[ATTEMPT_ID_ATTR]
    assert rerun.user_attrs[RERUN_OF_ATTR] == first.number
    assert winner.trial_number == rerun.number and winner.metric == 1.0
    assert _launch_envs(tmp_path) == [
        [opener, "never", opener],
        [opener, "allow", rerun.user_attrs[ATTEMPT_ID_ATTR]],
    ]
    assert set(tracker.lookups.read_text().splitlines()) == {f"e/p/{opener}"}
    capture = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])["remote_capture"]
    assert capture["run_id"] == opener
    _verify_winner_evidence(experiment, experiment.phases[0], winner)


@pytest.mark.integration
def test_a_rerun_whose_run_still_shows_the_earlier_summary_stays_pending_until_it_is_served(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun that exits 0 after logging rows while its run still serves the earlier attempt's crashed summary is neither scored from it nor failed: the trial stays pending without a second launch, saying W&B holds rows it has not summarized, and the next run scores it."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=2)
    tracker.mode.write_text("left")
    tracker.rows.write_text("5")

    with pytest.raises(
        TrackingTargetUnavailableError, match="W&B holds 3 rows logged since the attempt launched"
    ):
        run_experiment(experiment)
    first, rerun = phase_trials(experiment)
    assert rerun.state.name == "RUNNING" and TRIAL_OUTCOME_ATTR not in rerun.user_attrs
    assert len(_launch_envs(tmp_path)) == 2

    tracker.mode.write_text("logged")
    assert run_experiment(experiment)["p"].metric == 1.0
    _first, scored = phase_trials(experiment)
    assert scored.state.name == "COMPLETE"
    assert scored.user_attrs[ATTEMPT_ID_ATTR] == rerun.user_attrs[ATTEMPT_ID_ATTR]
    assert len(_launch_envs(tmp_path)) == 2


@pytest.mark.integration
def test_a_rerun_that_adds_no_row_to_a_closed_run_is_scored_from_its_summary(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun that exits 0 without logging into a run its earlier session left ``killed`` is waited out to the deadline; the run is then finished with the row count it had at the launch, so its summary is the trial's result, disclosed as an earlier attempt's, with no second launch."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=2)
    tracker.mode.write_text("settled")

    assert run_experiment(experiment)["p"].metric == 0.5
    _first, scored = phase_trials(experiment)
    assert scored.state.name == "COMPLETE"
    assert scored.user_attrs[EARLIER_ATTEMPT_RESULT_ATTR] is True
    capture = json.loads(scored.user_attrs[OBJECTIVE_PROVENANCE_ATTR])["remote_capture"]
    assert "earlier_attempt" not in capture
    assert len(_launch_envs(tmp_path)) == 2


@pytest.mark.integration
@pytest.mark.parametrize("readable", [True, False], ids=["read", "run-missing"])
def test_a_rerun_launch_records_its_runs_row_count_and_state(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None], readable: bool
) -> None:
    """A rerun in its trial's directory records the run's row count and state before its trainer starts, or ``None`` for both when the read finds no run and the trainer launches anyway; the attempt that opened the directory records neither.

    :param Path tmp_path: Scratch directory.
    :param Callable[[str], None] wandb_worker_sdk: The fake SDK installer.
    :param bool readable: Whether the run exists when the rerun launches.
    """
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=10)
    tracker.mode.write_text("logged" if readable else "absent")

    assert run_experiment(experiment)["p"].metric == 1.0
    first, rerun = phase_trials(experiment)
    stamps = []
    for trial in (first, rerun):
        attempt_dir = _trial_attempt_dir(trial)
        assert attempt_dir is not None
        baseline = read_evidence_baseline(attempt_dir)
        stamps.append((baseline.wandb_rows, baseline.wandb_state))
    assert stamps == [(None, None), (5, "finished") if readable else (None, None)]
    assert len(_launch_envs(tmp_path)) == 2


@pytest.mark.integration
def test_a_rerun_whose_idle_run_never_finished_reruns_once_in_a_fresh_run(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun that exits 0 while its resumed run shows no row since its launch and its earlier session never closed leaves nothing to score at the deadline: it is interrupted, and its trial reruns once in a fresh directory that opens a new W&B run, which scores it."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=2)
    tracker.mode.write_text("idle")

    winner = run_experiment(experiment)["p"]
    first, in_place, fresh = phase_trials(experiment)
    opener, fresh_id = first.user_attrs[ATTEMPT_ID_ATTR], fresh.user_attrs[ATTEMPT_ID_ATTR]
    assert in_place.user_attrs[LEFT_NOTHING_TO_SCORE_ATTR] is True
    assert in_place.user_attrs[TRIAL_DIR_ATTR] == first.user_attrs[TRIAL_DIR_ATTR]
    assert fresh.user_attrs[TRIAL_DIR_ATTR] != first.user_attrs[TRIAL_DIR_ATTR]
    assert winner.trial_number == fresh.number and winner.metric == 1.0
    assert EARLIER_ATTEMPT_RESULT_ATTR not in fresh.user_attrs
    assert _launch_envs(tmp_path) == [
        [opener, "never", opener],
        [opener, "allow", in_place.user_attrs[ATTEMPT_ID_ATTR]],
        [fresh_id, "never", fresh_id],
    ]
    assert set(tracker.lookups.read_text().split()) == {f"e/p/{opener}", f"e/p/{fresh_id}"}


@pytest.mark.integration
@pytest.mark.parametrize("directory_missing", [False, True])
def test_recovery_after_a_runner_kill_reads_the_reruns_trial_run(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None], directory_missing: bool
) -> None:
    """Recovery reads the run that the recorded directory's opener logged to."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=30)
    (first,) = phase_trials(experiment)
    if directory_missing:
        directory = Path(first.user_attrs[TRIAL_DIR_ATTR])
        directory.rename(directory.with_name("saved-directory"))
    tracker.mode.write_text("hang")
    reader = None
    try:
        config = tmp_path / "experiment.yaml"
        with runner_once(config, tracker.hung):
            _first, rerun = phase_trials(experiment)
            attempt_dir = _trial_attempt_dir(rerun)
            assert attempt_dir is not None
            reader = reaper.read_stale_process_identity(
                attempt_dir, expected_attempt_id=rerun.user_attrs[ATTEMPT_ID_ATTR]
            )
        tracker.mode.write_text("logged")
        tracker.lookups.write_text("")

        winner = run_experiment(experiment)["p"]
        first, rerun = phase_trials(experiment)
        opener = (rerun if directory_missing else first).user_attrs[ATTEMPT_ID_ATTR]
        assert winner.trial_number == rerun.number and winner.metric == 1.0
        assert _launch_envs(tmp_path) == [
            [first.user_attrs[ATTEMPT_ID_ATTR], "never", first.user_attrs[ATTEMPT_ID_ATTR]],
            [opener, "never" if directory_missing else "allow", rerun.user_attrs[ATTEMPT_ID_ATTR]],
        ]
        assert set(tracker.lookups.read_text().splitlines()) == {f"e/p/{opener}"}
        capture = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])["remote_capture"]
        assert capture["run_id"] == opener
    finally:
        if reader is not None:
            reaper.cleanup_stale_trial_process(reader, grace_seconds=0.2)


@pytest.mark.integration
def test_recovery_stops_the_reader_of_a_killed_reread(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """The next run stops the registered reader and resumes the fill without training."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        from pathlib import Path
        with Path({str(tmp_path / "launches")!r}).open("a") as out:
            print("launch", file=out)
        """,
    )
    extractor = WandbExtractor(
        type="wandb",
        base_url="https://example.test",
        entity="e",
        project="p",
        metric_key="x",
        poll_seconds=0.05,
        timeout_seconds=30,
    )
    original = make_experiment(
        persistent=tmp_path, trainer=trainer, metric=Metric(extractor=extractor), n_trials=1
    )
    run_experiment(original)
    (trial,) = phase_trials(original)
    source_slot = _trial_attempt_dir(trial)
    assert source_slot is not None
    source_identity = reaper.read_stale_process_identity(
        source_slot, expected_attempt_id=trial.user_attrs[ATTEMPT_ID_ATTR]
    )
    # A new gate changes what the kept trial's evidence reads as, so the next run re-reads it.
    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required",
        base_url="https://example.test",
        entity="e",
        project="p",
        keys=["x"],
        poll_seconds=0.05,
        timeout_seconds=30,
    )
    gated = original.phases[0].model_copy(update={"gates": [gate]})
    changed = original.model_copy(update={"phases": [gated]})
    config = write_experiment_config(tmp_path / "changed.yaml", changed)
    tracker.mode.write_text("hang")
    reader = None
    try:
        with runner_once(config, tracker.hung):
            registry = list(_attempts_dir(original).glob("*.json"))
            if registry:
                (entry_path,) = registry
                entry = json.loads(entry_path.read_text())
                worker_path = Path(entry["attempt_dir"])
                worker_id = entry["attempt_id"]
            else:
                worker_path = source_slot
                worker_id = trial.user_attrs[ATTEMPT_ID_ATTR]
            reader = reaper.read_stale_process_identity(worker_path, expected_attempt_id=worker_id)
        tracker.mode.write_text("logged")
        assert run_experiment(changed)["p"].metric == 1
        assert reaper.stale_trial_process_decision(reader).action == "gone"
        assert (
            reaper.read_stale_process_identity(
                source_slot, expected_attempt_id=trial.user_attrs[ATTEMPT_ID_ATTR]
            )
            == source_identity
        )
        assert (tmp_path / "launches").read_text().splitlines() == ["launch"]
    finally:
        if reader is not None and reaper.stale_trial_process_decision(reader).action != "gone":
            reaper.cleanup_stale_trial_process(reader, grace_seconds=0.2)
