"""Durable state that stale-trial recovery acts on, the operator command that acts on it, and the runner process a test stops to leave it."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import optuna
from click.testing import CliRunner, Result

from phasesweep.cli import cli as cli_main
from phasesweep.config.io import load_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _attempt_dir, _experiment_dir, _trial_dir_for
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    TRAINER_ENV_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.trial import _environment_identity
from phasesweep.mcp.recovery import _RecoveryNeeds
from phasesweep.runtime.process import _write_process_identity
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    kill_stale_group,
    read_boot_id,
    read_pid_namespace,
)
from tests.conftest import (
    FAKE_TRAIN_METRIC,
    console_script,
    copy_fake_train,
    make_experiment,
    mark_current_format,
    reaped_pid,
    write_experiment_config,
    write_trainer,
)

#: How long a run may take to reach a held trial, or to finish, before the test fails.
_WAIT_SECONDS = 120.0


def write_trial_identity(
    attempt_dir: Path,
    *,
    attempt_id: str,
    pid: int,
    starttime: int | None,
    pgid: int | None = None,
    boot_id: str | None = None,
    pid_namespace: str | None = None,
) -> None:
    """Leave the process identity file for a trainer whose guardian is already gone.

    ``pgid`` defaults to ``pid``, ``boot_id`` to this boot, and ``pid_namespace`` to this process's.
    """
    _write_process_identity(
        attempt_dir / PROCESS_IDENTITY_FILE,
        StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id=attempt_id,
            pid=pid,
            pgid=pid if pgid is None else pgid,
            proc_starttime=starttime,
            boot_id=read_boot_id() if boot_id is None else boot_id,
            pid_namespace=read_pid_namespace() if pid_namespace is None else pid_namespace,
            guardian_pid=reaped_pid(),
            guardian_starttime=1,
        ),
    )


def new_phase_study(experiment: Experiment, phase_name: str) -> optuna.Study:
    """Create one phase's minimizing study."""
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase_name}",
        storage=_resolve_storage(experiment.resolved_storage),
        direction="minimize",
    )
    mark_current_format(experiment, study)
    return study


class StaleTrial(NamedTuple):
    """A RUNNING trial an orchestrator abandoned: its study, directories and number.

    ``attempt_dir`` holds the attempt's lifecycle record and process identity.
    """

    study: optuna.Study
    trial_dir: Path
    number: int
    attempt_dir: Path


def stamp_artifact_root(study: optuna.Study, experiment: Experiment) -> None:
    """Bind a hand-built study to the artifact root a prior run would have claimed."""
    # A study an earlier invocation left behind always carries this binding.
    # Without it, preflight refuses the run as unmarked state, which
    # preempts the current-format recovery behavior under test.
    study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment)))


def fabricate_stale_trial(
    experiment: Experiment,
    phase_name: str,
    *,
    attempt_id: str,
    generation_id: str = "old-generation",
    persist_attempt_id: bool = True,
    persist_generation_id: bool = True,
) -> StaleTrial:
    """Create the phase study, as a crashed engine run leaves it, holding one attribute-complete RUNNING trial."""
    study = new_phase_study(experiment, phase_name)
    phase = next(phase for phase in experiment.phases if phase.name == phase_name)
    study.set_user_attr(TRIAL_TARGET_ATTR, phase.n_trials)
    stamp_artifact_root(study, experiment)
    trial = study.ask()
    identity = _environment_identity(experiment, phase_name)
    trial.set_user_attr(TRAINER_ENV_DIGEST_ATTR, identity.digest)
    trial.set_user_attr(TRAINER_ENV_ATTR, dict(identity.tracked))
    trial_dir = _trial_dir_for(experiment, phase_name, trial.number, attempt_id=attempt_id)
    trial_dir.mkdir(parents=True)
    attempt_dir = _attempt_dir(trial_dir.parent, attempt_id)
    attempt_dir.mkdir(parents=True)
    if persist_generation_id:
        trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    if persist_attempt_id:
        trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    return StaleTrial(study, trial_dir, trial.number, attempt_dir)


def fabricate_registered_attempt(
    experiment: Experiment,
    phase_name: str,
    *,
    attempt_id: str,
    persist_trial_attempt_id: bool = True,
    persist_trial_generation_id: bool = True,
) -> StaleTrial:
    """Leave the stale trial :func:`fabricate_stale_trial` does, plus its attempt-registry entry."""
    stale = fabricate_stale_trial(
        experiment,
        phase_name,
        attempt_id=attempt_id,
        persist_attempt_id=persist_trial_attempt_id,
        persist_generation_id=persist_trial_generation_id,
    )
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase_name,
        study_name=stale.study.study_name,
        trial_number=stale.number,
        trial_dir=stale.trial_dir,
        attempt_dir=stale.attempt_dir,
        generation_id="old-generation",
    )
    return stale


def write_launched_stale_trial(
    config: Path,
    *,
    cleanup_confirmed: bool | None = None,
    generation_id: str = "stale-generation",
    persist_trial_attrs: bool = True,
    boot_id: str | None = None,
    pid: int | None = None,
) -> int:
    """Leave the config's first phase holding a RUNNING trial whose trainer was launched and never reaped."""
    exp = load_experiment(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = new_phase_study(exp, phase.name)
    trial = study.ask()
    attempt_id = f"stale-attempt-{trial.number}"
    trial_dir = _trial_dir_for(exp, phase.name, trial.number, attempt_id=attempt_id)
    trial_dir.mkdir(parents=True)
    attempt_dir = _attempt_dir(trial_dir.parent, attempt_id)
    attempt_dir.mkdir(parents=True)
    write_trial_identity(
        attempt_dir,
        attempt_id=attempt_id,
        pid=pid or reaped_pid(),
        starttime=222,
        boot_id=boot_id,
    )
    if persist_trial_attrs:
        trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
        trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
        trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
        if cleanup_confirmed is not None:
            trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, cleanup_confirmed)
    return trial.number


def write_uncertain_failed_trial(
    config: Path, *, generation_id: str = "stale-generation", pid: int | None = None
) -> int:
    """Leave the config's first phase holding a FAIL trial whose process cleanup was never confirmed."""
    exp = load_experiment(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = new_phase_study(exp, phase.name)
    trial = study.ask()
    attempt_id = f"stale-attempt-{trial.number}"
    trial_dir = _trial_dir_for(exp, phase.name, trial.number, attempt_id=attempt_id)
    trial_dir.mkdir(parents=True)
    attempt_dir = _attempt_dir(trial_dir.parent, attempt_id)
    attempt_dir.mkdir(parents=True)
    write_trial_identity(
        attempt_dir,
        attempt_id=attempt_id,
        pid=pid or reaped_pid(),
        starttime=111,
    )
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return trial.number


def load_only_recovery_needs(*, ownership_storage_unavailable: bool = False) -> _RecoveryNeeds:
    """Return recovery decisions that load the phase studies and nothing else."""
    return _RecoveryNeeds(
        terminal_status=None,
        stored_snapshot=None,
        prepared_publication_generation=None,
        cleanup_needed=False,
        terminal_cleanup_uncertain=False,
        ownership_storage_unavailable=ownership_storage_unavailable,
        snapshot_recovery_required=False,
        snapshot_unavailable=False,
        snapshot_finalize_needed=False,
    )


def recover_run_cli(state_dir: Path | str, run_id: str, *, confirm: bool = False) -> Result:
    """Return the outcome of ``phasesweep mcp recover-run`` run in-process as the operator types it."""
    argv = ["mcp", "recover-run", "--state-dir", str(state_dir), "--run-id", run_id]
    return CliRunner().invoke(cli_main, [*argv, "--confirm"] if confirm else argv)


_RUNNER = (
    "import sys\n"
    "from phasesweep import run_experiment\n"
    "from phasesweep.config.io import load_experiment\n"
    "run_experiment(load_experiment(sys.argv[1]))\n"
)


@contextlib.contextmanager
def runner_once(
    config: Path, marker: Path, *, timeout: float = 30
) -> Iterator[subprocess.Popen[bytes]]:
    """Run ``config`` in a runner process of its own, and hand it over once ``marker`` exists.

    The runner's output goes to ``runner.log`` beside ``config``. On exit a runner still
    running is SIGKILLed, so the run it leaves is the one a killed runner leaves.

    :param Path config: Experiment config to run.
    :param Path marker: File the trainer creates when the runner is to be stopped.
    :param float timeout: Seconds to wait for ``marker``, defaults to 30.
    :return Iterator[subprocess.Popen[bytes]]: The runner, while ``marker`` exists.
    """
    with (config.parent / "runner.log").open("w") as output:
        runner = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, str(config)], stdout=output, stderr=output
        )
        try:
            deadline = time.monotonic() + timeout
            while not marker.exists():
                assert runner.poll() is None and time.monotonic() < deadline
                time.sleep(0.02)
            yield runner
        finally:
            if runner.poll() is None:
                runner.kill()
            runner.wait(timeout=5)


class RerunHarness:
    """One test's workdirs, gated trainer and spawned runs."""

    def __init__(self, root: Path) -> None:
        """Write the gated trainer under ``root``.

        :param Path root: The test's scratch directory.
        """
        self.root = root
        self.gate = root / "gate"
        self.gate.mkdir()
        fake_train = copy_fake_train(root)
        self.trainer = write_trainer(
            root / "gated_train.py",
            f"""
            import hashlib, os, sys, time
            from pathlib import Path
            gate = Path({str(self.gate)!r})
            # A rerun sees the trial id of the trial whose directory it continues, so the
            # gate counts each phase directory's launches instead: every trial number
            # launches once, in order.
            phase_dir = str(Path(os.environ["PHASESWEEP_TRIAL_DIR"]).parent)
            launches = gate / "launches" / hashlib.sha256(phase_dir.encode()).hexdigest()
            launches.mkdir(parents=True, exist_ok=True)
            trial = 0
            while True:
                try:
                    os.close(os.open(launches / str(trial), os.O_CREAT | os.O_EXCL))
                    break
                except FileExistsError:
                    trial += 1
            if (gate / f"hold-{{trial}}").exists():
                (gate / f"reached-{{trial}}").touch()
                while (gate / f"hold-{{trial}}").exists():
                    time.sleep(0.05)
            os.execv(sys.executable, [sys.executable, {str(fake_train)!r}, *sys.argv[1:]])
            """,
        )
        self.spawned: list[subprocess.Popen[bytes]] = []
        self.log = root / "runs.log"

    def experiment(self, name: str, phase: Phase) -> Experiment:
        """Build a persistent experiment ``t`` under ``root/name`` running the gated trainer.

        :param str name: Directory holding this copy's ledger and workdir.
        :param Phase phase: The experiment's one phase.
        :return Experiment: The experiment.
        """
        return make_experiment(
            persistent=self.root / name,
            trial_command=f"python {self.trainer} {{overrides}}",
            metric=FAKE_TRAIN_METRIC,
            phases=[phase],
        )

    def config(self, experiment: Experiment) -> Path:
        """Write ``experiment`` as the YAML file a ``phasesweep run`` subprocess loads.

        :param Experiment experiment: The experiment to write.
        :return Path: The config path.
        """
        assert experiment.workdir is not None
        path = Path(experiment.workdir).parent / "experiment.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_experiment_config(path, experiment)
        return path

    def _spawn(self, config: Path) -> subprocess.Popen[bytes]:
        """Start ``phasesweep run`` on ``config`` as its own session.

        :param Path config: Config to run.
        :return subprocess.Popen[bytes]: The run.
        """
        with self.log.open("a") as handle:
            proc = subprocess.Popen(
                [str(console_script("phasesweep")), "run", str(config)],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                # A wide terminal keeps the closing ``log:`` path on one line.
                env={**os.environ, "COLUMNS": "500"},
            )
        self.spawned.append(proc)
        return proc

    def interrupt(self, config: Path, trial: int, sig: signal.Signals) -> None:
        """Run ``config`` until ``trial`` is in flight, then stop the run with ``sig``.

        SIGINT, SIGHUP and SIGTERM go to the run's process group, as a terminal's Ctrl-C and
        hangup do; SIGKILL goes to the orchestrator alone, so its trainer survives it.

        :param Path config: Config to run.
        :param int trial: Trial number the trainer holds in flight.
        :param signal.Signals sig: ``SIGINT``, ``SIGHUP``, ``SIGTERM`` or ``SIGKILL``.
        """
        (self.gate / f"hold-{trial}").touch()
        proc = self._spawn(config)
        reached = self.gate / f"reached-{trial}"
        deadline = time.monotonic() + _WAIT_SECONDS
        while not reached.exists():
            if proc.poll() is not None:
                raise AssertionError(
                    f"run exited ({proc.returncode}) before trial {trial} started:\n"
                    f"{self.log.read_text()}"
                )
            if time.monotonic() > deadline:
                raise AssertionError(f"trial {trial} never started:\n{self.log.read_text()}")
            time.sleep(0.05)
        if sig == signal.SIGKILL:
            os.kill(proc.pid, signal.SIGKILL)
        else:
            os.killpg(proc.pid, sig)
        assert proc.wait(timeout=_WAIT_SECONDS) != 0, self.log.read_text()

    def complete(self, config: Path) -> None:
        """Run ``config`` to completion and require success.

        :param Path config: Config to run.
        """
        proc = self._spawn(config)
        assert proc.wait(timeout=_WAIT_SECONDS) == 0, self.log.read_text()

    def release(self) -> None:
        """Remove every hold, so a trainer still waiting on one finishes."""
        for hold in self.gate.glob("hold-*"):
            hold.unlink()

    def kill_survivors(self) -> None:
        """Kill every spawned run still alive and every trainer group its trials recorded."""
        for proc in self.spawned:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        for path in self.root.rglob(PROCESS_IDENTITY_FILE):
            identity = json.loads(path.read_text())
            kill_stale_group(
                identity["pid"],
                identity["proc_starttime"],
                pgid=identity["pgid"],
                grace_seconds=1.0,
            )
