"""Which trial directory an attempt runs in: its own, or the one the trial it reruns opened.

Every rerun of a recorded trial, interrupted or failed, runs the same command in the
directory that trial's chain opened, so a trainer that resumes from the latest checkpoint in
its output directory continues where it stopped. PhaseSweep hands the trainer no pointer and
copies nothing. Whether a checkpoint a failure left is usable is the trainer's own resume
logic to decide, as it is after a killed runner; the evidence baseline makes the rerun count
only the files it writes itself.
"""

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
    """Yield the ancestors in a trial's rerun chain that recorded their directory, newest first.

    The chain follows :data:`~phasesweep.engine.state.RERUN_OF_ATTR` whether each source was
    interrupted or failed. A rerun can stop before recording its directory; the walk continues
    through it to find the last attempt that could have used the directory, and its opener.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: The trial about to launch, or one already launched.
    :raises ProcessCleanupUncertainError: A source's recorded directory is malformed.
    :return Iterator[FrozenTrial]: Directory sources, from the most recent to the opener.
    """
    current = trial
    while True:
        source_number = current.user_attrs.get(RERUN_OF_ATTR)
        source = trials.get(source_number) if type(source_number) is int else None
        if source is None:
            return
        try:
            recorded = _recorded_trial_dir(source)
        except ValueError as exc:
            raise _unusable_directory_error(source, exc) from exc
        if recorded is not None:
            yield source
        current = source


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
    :raises PhaseSweepError: The directory path is a link that does not lead to a directory.
    :return TrialDirectory: The directory, rebuilt under the current phase directory, with
        the trial number and attempt that opened it: ``trial`` and ``attempt_id`` themselves
        for a directory of its own.
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
    _refuse_unusable_link(trial, directory.path)
    return directory


def _refuse_unusable_link(trial: FrozenTrial, trial_dir: Path) -> None:
    """Refuse a rerun whose trial directory path is a link PhaseSweep cannot follow to a directory.

    A dangling link most plausibly names a checkpoint volume that is not mounted, so running
    the trial again from step 0 would orphan that checkpoint; the operator restores the target,
    or deletes the link to rerun from scratch. A missing path is left to the caller, which
    creates the directory.

    :param FrozenTrial trial: The trial about to rerun.
    :param Path trial_dir: The directory path it reruns in.
    :raises PhaseSweepError: The path is a broken link, or cannot be inspected.
    """
    try:
        is_link = S_ISLNK(trial_dir.lstat().st_mode)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PhaseSweepError(
            f"Trial {trial.number} reruns in {trial_dir}, which could not be inspected: {exc}.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    broken = (
        broken_symlink_detail(trial_dir.parent, trial_dir.name, require_directory=True)
        if is_link
        else None
    )
    if broken is not None:
        raise PhaseSweepError(
            f"Trial {trial.number} reruns in its trial directory, but {broken}.",
            action=OperatorAction.RESTORE_TREE,
            remediation=(
                f"Restore the link's target, or delete the link {trial_dir} to rerun this trial "
                "from step 0."
            ),
        )
