"""Which trial directory an attempt runs in: its own, or the one an interrupted trial left.

An interrupted trial's rerun runs the same command in the same directory, so a trainer that
resumes from the latest checkpoint in its output directory continues where it was stopped.
PhaseSweep hands the trainer no pointer and copies nothing. A rerun of a failed grid cell or
fixed configuration runs from scratch in a directory of its own: its failure may have left
corrupt state, and the retry exists to run it again.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from optuna.trial import FrozenTrial

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _attempt_process_resolution
from phasesweep.engine.paths import (
    _attempt_dir,
    _phase_dir,
    _trial_dir_for,
    _trial_dir_name,
    _valid_trial_dir_name,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    _recorded_trial_dir,
    _required_trial_dir,
    _trial_interrupted,
)
from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
from phasesweep.runtime.process import read_attempt_lifecycle
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE


def _directory_owner(trials: Mapping[int, FrozenTrial], trial: FrozenTrial) -> FrozenTrial:
    """Return the trial whose directory ``trial`` runs in.

    Follow the rerun chain back while each source was interrupted and recorded a
    directory: an interrupted rerun's own rerun keeps the directory too.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: The trial about to launch, or one already launched.
    :raises ProcessCleanupUncertainError: An interrupted source's recorded directory is malformed.
    :return FrozenTrial: The earliest such trial, ``trial`` itself when it is no such rerun.
    """
    owner = trial
    while True:
        source_number = owner.user_attrs.get(RERUN_OF_ATTR)
        source = trials.get(source_number) if type(source_number) is int else None
        if source is None or not _trial_interrupted(source):
            return owner
        try:
            recorded = _recorded_trial_dir(source)
        except ValueError as exc:
            raise _unusable_directory_error(source, exc) from exc
        if recorded is None:
            return owner
        owner = source


def _unusable_directory_error(owner: FrozenTrial, exc: ValueError) -> ProcessCleanupUncertainError:
    """Refuse a rerun whose source recorded a directory it cannot be found by.

    :param FrozenTrial owner: The trial whose recorded directory is unusable.
    :param ValueError exc: Why it is unusable.
    :return ProcessCleanupUncertainError: Error routed to restoring the ledger.
    """
    return ProcessCleanupUncertainError(
        f"Trial {owner.number} records an unusable {TRIAL_DIR_ATTR!r}, so its rerun "
        f"cannot find the directory it continues in: {exc}.",
        action=OperatorAction.RESTORE_LEDGER,
    )


class TrialDirectory(NamedTuple):
    """A trial directory and the attempt that opened it, whose names the trainer sees."""

    path: Path
    opener_number: int
    opener_attempt_id: str


def _owner_directory(phase_dir: Path, owner: FrozenTrial) -> TrialDirectory:
    """Return the directory a trial recorded when it opened it, under the current phase directory.

    :param Path phase_dir: The phase directory, which may have moved since.
    :param FrozenTrial owner: A trial that recorded its directory.
    :raises ProcessCleanupUncertainError: The recorded directory is malformed or not a name
        PhaseSweep builds.
    :return TrialDirectory: The directory, its opener's number and attempt.
    """
    try:
        # The name carries the opener's number and attempt, which the trainer sees.
        name = _valid_trial_dir_name(_required_trial_dir(owner).name, label="trial directory")
        number, _, attempt_id = name.removeprefix("trial_").partition("__")
        return TrialDirectory(phase_dir / name, int(number), attempt_id)
    except ValueError as exc:
        raise _unusable_directory_error(owner, exc) from exc


def _expected_trial_dir_name(trials: Mapping[int, FrozenTrial], trial: FrozenTrial) -> str | None:
    """Return the directory name a launched trial must have recorded.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: A launched trial.
    :return str | None: The name :func:`_trial_dir_for` builds for the trial that opened its
        directory, or ``None`` when that trial records no attempt.
    """
    owner = _directory_owner(trials, trial)
    attempt_id = owner.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    return _trial_dir_name(owner.number, attempt_id)


def _trial_dir_for_attempt(
    experiment: Experiment,
    phase_name: str,
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
    *,
    attempt_id: str,
    recovered: frozenset[int] = frozenset(),
) -> TrialDirectory:
    """Return the trial directory a new attempt runs in, and the attempt that opened it.

    A trial that continues another's directory may start only once that directory's last
    attempt is provably gone, so two live processes never write one directory: its
    lifecycle says ``exited`` with cleanup confirmed, or ``allocated`` with no process
    identity beside it, or recovery recorded its cleanup. The first two are recovery's own
    rule, :func:`phasesweep.engine.attempts._attempt_process_resolution`.

    :param Experiment experiment: Configuration naming the phase directory.
    :param str phase_name: Phase the attempt runs in.
    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: The trial about to launch.
    :param str attempt_id: The new attempt, which names a directory it opens itself.
    :param frozenset[int] recovered: Trial numbers whose cleanup recovery recorded, from
        :func:`phasesweep.engine.attempts._cleanup_recovered_trial_numbers`.
    :raises ProcessCleanupUncertainError: The directory's last attempt may still be running,
        or the recorded directory is unusable.
    :return TrialDirectory: The directory, rebuilt under the current phase directory, with
        the trial number and attempt that opened it: ``trial`` and ``attempt_id`` themselves
        for a directory of its own.
    """
    owner = _directory_owner(trials, trial)
    if owner is trial:
        path = _trial_dir_for(experiment, phase_name, trial.number, attempt_id=attempt_id)
        return TrialDirectory(path, trial.number, attempt_id)
    phase_dir = _phase_dir(experiment, phase_name)
    source = trials[int(trial.user_attrs[RERUN_OF_ATTR])]
    source_attempt = source.user_attrs.get(ATTEMPT_ID_ATTR)
    if source.number not in recovered:
        lifecycle = None
        if isinstance(source_attempt, str) and source_attempt:
            source_dir = _attempt_dir(phase_dir, source_attempt)
            try:
                lifecycle = read_attempt_lifecycle(source_dir, expected_attempt_id=source_attempt)
            except ValueError:
                lifecycle = None
            # Recovery's own rule for an attempt that needs no signal: anything else it
            # would have to inspect or clean first.
            settled = lifecycle is not None and _attempt_process_resolution(
                lifecycle, identity_exists=(source_dir / PROCESS_IDENTITY_FILE).exists()
            ) in {"exited", "allocated"}
        else:
            settled = False
        if not settled:
            raise ProcessCleanupUncertainError(
                f"Trial {trial.number} reruns interrupted trial {source.number} in its directory, "
                f"but attempt {source_attempt!r} of trial {source.number} may still be running "
                f"(lifecycle {None if lifecycle is None else lifecycle.state!r}). Recover the run "
                "first, so two processes never write one trial directory.",
                action=OperatorAction.INSPECT_LOGS,
            )
    return _owner_directory(phase_dir, owner)
