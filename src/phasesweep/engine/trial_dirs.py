"""Choose an attempt's training directory and track which attempts trained there."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from stat import S_ISLNK
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
)
from phasesweep.errors import OperatorAction, PhaseSweepError, ProcessCleanupUncertainError
from phasesweep.runtime.files import broken_symlink_detail
from phasesweep.runtime.process import read_attempt_lifecycle
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE


def _rerun_directory_sources(
    trials: Mapping[int, FrozenTrial], trial: FrozenTrial
) -> Iterator[FrozenTrial]:
    """Yield the earlier attempts that recorded the same directory, newest first.

    Walk through attempts that stopped before recording a directory. Stop at the attempt
    whose recorded directory carries its own name, even when it was itself a rerun.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: A trial about to launch or one that recorded its directory.
    :raises ProcessCleanupUncertainError: A source is invalid or its recorded directory is malformed.
    :return Iterator[FrozenTrial]: Directory sources, from the most recent to the opener.
    """
    current = trial
    while True:
        source_number = current.user_attrs.get(RERUN_OF_ATTR)
        source = trials.get(source_number) if type(source_number) is int else None
        if source_number is not None and (source is None or source.number >= current.number):
            raise ProcessCleanupUncertainError(
                f"Trial {current.number} names {source_number!r} under {RERUN_OF_ATTR!r}, "
                "which is not an earlier trial of its study.",
                action=OperatorAction.RESTORE_LEDGER,
            )
        try:
            recorded = _recorded_trial_dir(current)
        except ValueError as exc:
            raise _unusable_directory_error(current, exc) from exc
        if recorded is not None:
            if current is not trial:
                yield current
            if recorded.name == _own_directory_name(current):
                return
        if source is None:
            return
        current = source


def _own_directory_name(trial: FrozenTrial) -> str | None:
    """Return the name a trial would record if it opened its own directory.

    :param FrozenTrial trial: The trial whose name to derive.
    :return str | None: The name for the trial's recorded attempt, if any.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    return _trial_dir_name(trial.number, attempt_id)


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
    restart_reason: str | None = None


def _owner_directory(phase_dir: Path, owner: FrozenTrial) -> TrialDirectory:
    """Return the directory a trial recorded when it opened it, under the current phase directory.

    :param Path phase_dir: The phase directory, which may have moved since.
    :param FrozenTrial owner: A trial that recorded its directory.
    :raises ProcessCleanupUncertainError: The recorded directory does not name its opener.
    :return TrialDirectory: The directory, its opener's number and attempt.
    """
    try:
        name = _valid_trial_dir_name(_required_trial_dir(owner).name, label="trial directory")
        attempt_id = owner.user_attrs.get(ATTEMPT_ID_ATTR)
        if not isinstance(attempt_id, str) or name != _trial_dir_name(owner.number, attempt_id):
            raise ValueError(f"{name!r} does not name the trial and attempt that opened it")
        return TrialDirectory(phase_dir / name, owner.number, attempt_id)
    except ValueError as exc:
        raise _unusable_directory_error(owner, exc) from exc


def _expected_trial_dir_name(trials: Mapping[int, FrozenTrial], trial: FrozenTrial) -> str | None:
    """Return the directory name a launched trial must have recorded.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: A launched trial.
    :return str | None: The name :func:`_trial_dir_for` builds for the trial that opened its
        directory, or ``None`` when that trial records no attempt.
    """
    sources = list(_rerun_directory_sources(trials, trial))
    owner = sources[-1] if sources else trial
    return _own_directory_name(owner)


def _trial_dir_for_attempt(
    experiment: Experiment,
    phase_name: str,
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
    *,
    attempt_id: str,
    recovered: frozenset[int] = frozenset(),
) -> TrialDirectory:
    """Reuse an available training directory after cleanup, or open a fresh one.

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
    :raises PhaseSweepError: The directory path could not be inspected.
    :return TrialDirectory: The directory, rebuilt under the current phase directory, with
        the trial number and attempt that opened it and why it started fresh, if applicable.
    """
    sources = list(_rerun_directory_sources(trials, trial))
    if not sources:
        path = _trial_dir_for(experiment, phase_name, trial.number, attempt_id=attempt_id)
        return TrialDirectory(path, trial.number, attempt_id)
    phase_dir = _phase_dir(experiment, phase_name)
    source = sources[0]
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
                f"Trial {trial.number} reruns trial {source.number} in its directory, "
                f"but attempt {source_attempt!r} of trial {source.number} may still be running "
                f"(lifecycle {None if lifecycle is None else lifecycle.state!r}). Recover the run "
                "first, so two processes never write one trial directory.",
                action=OperatorAction.INSPECT_LOGS,
            )
    directory = _owner_directory(phase_dir, sources[-1])
    unavailable = _unavailable_trial_directory(directory.path)
    if unavailable is None:
        return directory
    path = _trial_dir_for(experiment, phase_name, trial.number, attempt_id=attempt_id)
    return TrialDirectory(path, trial.number, attempt_id, unavailable)


def _unavailable_trial_directory(trial_dir: Path) -> str | None:
    """Describe why a missing directory or unusable link would require a fresh directory.

    :param Path trial_dir: Directory to continue if available.
    :raises PhaseSweepError: The directory path could not be inspected.
    :return str | None: The missing directory or the unusable symlink and its target, if any.
    """
    try:
        is_link = S_ISLNK(trial_dir.lstat().st_mode)
    except FileNotFoundError:
        return f"the previous trial directory {trial_dir} is missing"
    except OSError as exc:
        raise PhaseSweepError(
            f"The previous trial directory {trial_dir} could not be inspected: {exc}.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    if is_link:
        return broken_symlink_detail(trial_dir.parent, trial_dir.name, require_directory=True)
    return None
