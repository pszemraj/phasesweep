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
from phasesweep.engine.evidence import _verify_winner_evidence
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
    _trial_attempt_dir,
)
from phasesweep.evidence.models import WandbExtractor
from tests.conftest import make_experiment, phase_trials, write_experiment_config, write_trainer
from tests.recovery_helpers import runner_once
from tests.tracking_helpers import ManualClock

_LAUNCHED_AT = 1_000.0


def _poll() -> dict[str, object]:
    """Run the in-process summary poll for a rerun launched at :data:`_LAUNCHED_AT`.

    :return dict[str, object]: The capture.
    """
    from phasesweep.evidence.wandb import _poll_wandb_summary

    return _poll_wandb_summary(
        base_url="https://example.test",
        entity="e",
        project="p",
        run_id="opener",
        poll_seconds=5,
        timeout_seconds=240,
        required_keys=["x"],
        logged_after=_LAUNCHED_AT,
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch, answers: Callable[[int], tuple[str, dict[str, float]]]
) -> tuple[ManualClock, list[str]]:
    """Answer each zero-based run lookup with ``answers(read)`` on a manual clock.

    :param pytest.MonkeyPatch monkeypatch: Fixture that replaces the SDK client and clock.
    :param Callable[[int], tuple[str, dict[str, float]]] answers: State and summary per read.
    :return tuple[ManualClock, list[str]]: The clock and the paths looked up.
    """
    public = pytest.importorskip("wandb.apis.public")
    clock = ManualClock()
    monkeypatch.setattr("phasesweep.evidence.wandb.time.monotonic", clock.monotonic)
    monkeypatch.setattr("phasesweep.evidence.wandb.time.sleep", clock.sleep)
    paths: list[str] = []

    def lookup(path: str) -> SimpleNamespace:
        paths.append(path)
        state, summary = answers(len(paths) - 1)
        return SimpleNamespace(state=state, summary_metrics=summary)

    monkeypatch.setattr(public, "Api", lambda **kwargs: SimpleNamespace(run=lookup))
    return clock, paths


_LEFT = {"x": 0.5, "_timestamp": _LAUNCHED_AT - 60}
_LOGGED = {"x": 0.25, "_timestamp": _LAUNCHED_AT + 5}


def test_a_reruns_read_waits_out_the_state_its_earlier_attempt_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run still crashed, then finished, as the earlier attempt left it is neither failed nor scored; the rerun's own finished summary is."""
    clock, paths = _serve(
        monkeypatch,
        lambda read: [("crashed", _LEFT), ("finished", _LEFT), ("finished", _LOGGED)][min(read, 2)],
    )
    assert _poll()["values"] == {"x": 0.25}
    assert paths == ["e/p/opener"] * 3
    assert clock.now == 10


def test_a_reruns_run_that_never_moves_is_refused_after_the_settle_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose summary never passes the rerun's launch fails as not written by this attempt once the settle window ends."""
    from phasesweep.evidence.wandb import _FINISH_SETTLE_SECONDS, WandbRunStaleError

    clock, _ = _serve(monkeypatch, lambda read: ("finished", _LEFT))
    with pytest.raises(WandbRunStaleError):
        _poll()
    assert clock.now == _FINISH_SETTLE_SECONDS


def test_a_reruns_own_crash_fails_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that crashed after the rerun logged to it is the rerun's own failure."""
    from phasesweep.evidence.wandb import WandbRunTerminalError

    _serve(monkeypatch, lambda read: ("crashed", _LOGGED))
    with pytest.raises(WandbRunTerminalError):
        _poll()


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
    and blocks. Every lookup appends the path it asked for to ``lookups``.

    :param Callable[[str], None] install_sdk: The ``wandb_worker_sdk`` fixture's installer.
    :param Path root: Directory holding the mode, lookup and hang files.
    :return SimpleNamespace: ``mode``, ``lookups`` and ``hung`` paths.
    """
    files = SimpleNamespace(mode=root / "mode", lookups=root / "lookups", hung=root / "hung")
    files.mode.write_text("logged")
    install_sdk(f"""
        import time
        from pathlib import Path

        class Api:
            def __init__(self, **kwargs):
                pass

            def run(self, path):
                mode = Path({str(files.mode)!r}).read_text()
                with Path({str(files.lookups)!r}).open("a") as lookups:
                    lookups.write(path + "\\n")
                if mode == "hang":
                    Path({str(files.hung)!r}).touch()
                    time.sleep(60)
                if mode == "left":
                    summary = {{"x": 0.5, "_timestamp": 1.0}}
                    return type("Run", (), {{"state": "crashed", "summary_metrics": summary}})()
                summary = {{"x": 1.0, "_timestamp": time.time()}}
                return type("Run", (), {{"state": "finished", "summary_metrics": summary}})()
    """)
    return files


def _interrupted_wandb_trial(tmp_path: Path, *, timeout_seconds: float) -> Experiment:
    """Leave a persistent W&B trial RUNNING after a runner SIGKILL, its trainer still running.

    Every launch appends its ``WANDB_RUN_ID``, ``WANDB_RESUME`` and attempt ID to ``envs``.
    The first touches ``first`` and sleeps until the next run's recovery stops it; later
    attempts exit 0.

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
def test_a_rerun_that_logs_nothing_fails_as_not_written_by_this_attempt(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun that exits 0 leaving the trial's run crashed with the earlier attempt's summary fails instead of scoring that summary."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=2)
    tracker.mode.write_text("left")

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)
    first, rerun = phase_trials(experiment)
    outcome = _parsed_trial_outcome(rerun.user_attrs[TRIAL_OUTCOME_ATTR])
    assert outcome is not None and outcome.outcome == "failure"
    assert outcome.cause == (
        f"metric extractor: W&B run https://example.test/e/p/{first.user_attrs[ATTEMPT_ID_ATTR]} "
        "was not written by this attempt: its summary records no wandb.log() call after the "
        "attempt launched."
    )


@pytest.mark.integration
def test_recovery_after_a_runner_kill_reads_the_reruns_trial_run(
    tmp_path: Path, wandb_worker_sdk: Callable[[str], None]
) -> None:
    """A rerun whose runner died while reading W&B is recovered from the trial's run, not one named by its own attempt."""
    tracker = _fake_wandb(wandb_worker_sdk, tmp_path)
    experiment = _interrupted_wandb_trial(tmp_path, timeout_seconds=30)
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
        opener = first.user_attrs[ATTEMPT_ID_ATTR]
        assert winner.trial_number == rerun.number and winner.metric == 1.0
        assert len(_launch_envs(tmp_path)) == 2
        assert set(tracker.lookups.read_text().splitlines()) == {f"e/p/{opener}"}
        capture = json.loads(rerun.user_attrs[OBJECTIVE_PROVENANCE_ATTR])["remote_capture"]
        assert capture["run_id"] == opener
    finally:
        if reader is not None:
            reaper.cleanup_stale_trial_process(reader, grace_seconds=0.2)
