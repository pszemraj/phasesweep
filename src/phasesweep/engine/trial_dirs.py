"""Choose an attempt's training directory and track which attempts trained there."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from stat import S_ISLNK
from typing import NamedTuple

from optuna import Study
from optuna.trial import FrozenTrial, TrialState

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _attempt_process_resolution
from phasesweep.engine.errors import StudySchemaMismatchError, WinnerIntegrityError
from phasesweep.engine.fingerprints import _recorded_inheritance, phase_study_name
from phasesweep.engine.optuna import _draw_positions
from phasesweep.engine.paths import (
    _attempt_dir,
    _phase_dir,
    _trial_dir_for,
    _trial_dir_name,
    _trial_dir_opener,
    _valid_trial_dir_name,
)
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    REREAD_FILL_ATTR,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    TRIAL_DIR_ATTR,
    _recorded_identity,
    _recorded_trial_dir,
    _required_trial_dir,
    _trial_lifecycle,
    _unreplaced_attempts,
)
from phasesweep.errors import OperatorAction, PhaseSweepError, ProcessCleanupUncertainError
from phasesweep.evidence.baseline import read_evidence_baseline
from phasesweep.runtime.files import broken_symlink_detail
from phasesweep.runtime.json import json_object
from phasesweep.runtime.process import read_attempt_lifecycle
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE

REREAD_FILL_SCHEMA_VERSION = 3
_FILL_FIELDS = frozenset({"schema_version", "study_name"})


def _fill_source_name(target: Study | None) -> str | None:
    """Read the source study a filled study explicitly names.

    :param Study | None target: Study whose copy provenance to read.
    :raises StudySchemaMismatchError: The saved source record is malformed.
    :return str | None: Source study name, or None when no copy started.
    """
    if target is None or (raw := target.user_attrs.get(REREAD_FILL_ATTR)) is None:
        return None
    try:
        record = json_object(raw)
        name = record["study_name"]
        if (
            record.keys() != _FILL_FIELDS
            or type(record.get("schema_version")) is not int
            or record["schema_version"] != REREAD_FILL_SCHEMA_VERSION
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError("invalid source name or schema version")
    except (KeyError, ValueError) as exc:
        raise StudySchemaMismatchError(
            f"Study {target.study_name!r} has a malformed {REREAD_FILL_ATTR!r} record: {exc}."
        ) from exc
    return name


class ProducingAttempt(NamedTuple):
    """The study and row that allocated an attempt, rather than a copy of that row."""

    study: Study
    trial: FrozenTrial


def producing_attempts(studies: Sequence[Study]) -> dict[str, ProducingAttempt]:
    """Resolve physical attempts by identity across all studies of a phase.

    Each fill pins its source, and never changes the attempt IDs it copies. A row whose
    direct source contains its ID is a copy, even if its source was itself filled.
    Later trials can have the same number in siblings, so numbers cannot identify copies.
    If the source was lost, the generation that launched the trainer still proves its
    address from its frozen config and the inherited values kept by both studies.

    :param Sequence[Study] studies: Studies of the phase from one ledger or live run.
    :raises StudySchemaMismatchError: A study's fill provenance is malformed.
    :return dict[str, ProducingAttempt]: Unambiguous producing rows by attempt ID.
    """
    rows = {
        study.study_name: {
            attempt_id: trial
            for trial in study.get_trials(deepcopy=False)
            if isinstance(attempt_id := trial.user_attrs.get(ATTEMPT_ID_ATTR), str)
        }
        for study in studies
    }
    owners: dict[str, list[ProducingAttempt]] = defaultdict(list)
    snapshots: dict[tuple[Path, str], Experiment | None] = {}
    for study in studies:
        source_name = _fill_source_name(study)
        source = None if source_name is None else rows.get(source_name)
        for attempt_id, trial in rows[study.study_name].items():
            if (
                source_name is None
                or (source is not None and attempt_id not in source)
                or source is None
                and _generation_addresses(study, trial, snapshots)
            ):
                owners[attempt_id].append(ProducingAttempt(study, trial))
    return {attempt_id: matches[0] for attempt_id, matches in owners.items() if len(matches) == 1}


def _generation_addresses(
    study: Study, trial: FrozenTrial, snapshots: dict[tuple[Path, str], Experiment | None]
) -> bool:
    """Check whether retained launch settings identify a row whose fill source is gone.

    :param Study study: Possible launching study.
    :param FrozenTrial trial: Row with a physical attempt identity.
    :param dict[tuple[Path, str], Experiment | None] snapshots: Validated generation configs
        by artifact root and generation, reused across their trials.
    :return bool: Whether the generation's configuration names this study.
    """
    generation = trial.user_attrs.get(GENERATION_ID_ATTR)
    identity = _recorded_identity(study)
    if not isinstance(generation, str) or identity is None:
        return False
    try:
        directory = _recorded_trial_dir(trial)
    except ValueError:
        return False
    if directory is None:
        return False
    path = directory.parent.parent
    key = path, generation
    if key not in snapshots:
        try:
            snapshots[key] = _load_generation_experiment(path, generation)
        except WinnerIntegrityError:
            snapshots[key] = None
    producing = snapshots[key]
    if producing is None:
        return False
    return any(
        phase_study_name(producing, phase, _recorded_inheritance(identity)) == study.study_name
        for phase in producing.phases
        if phase.name == directory.parent.name
    )


class DirectoryHistory(NamedTuple):
    """Identity of earlier launches and the cutoff for evidence shared with later ones."""

    earlier: tuple[tuple[str, str], ...] = ()
    evidence_before: float | str | None = None


class DirectoryLaunch(NamedTuple):
    """A physical attempt and the evidence of when it started."""

    attempt_id: str
    generation: str
    epoch: float | None
    order: float


def directory_launches(
    attempts: Mapping[str, ProducingAttempt], *, directory: Path | None = None
) -> dict[str, list[DirectoryLaunch]]:
    """Index the physical launches sharing each training directory.

    The opener always sorts first. The launch baseline provides the epoch of each reuse.
    An attempt that reached RUNNING before starting has no shared evidence if its lifecycle
    records that it never launched.

    :param Mapping[str, ProducingAttempt] attempts: Producing rows across the phase's studies.
    :param Path | None directory: When supplied, inspect only this directory's launches.
    :return dict[str, list[DirectoryLaunch]]: Launch records indexed by training directory name.
    """
    launches: dict[str, list[DirectoryLaunch]] = defaultdict(list)
    for identity, producing in attempts.items():
        row = producing.trial
        gen = row.user_attrs.get(GENERATION_ID_ATTR)
        try:
            path = _recorded_trial_dir(row)
            opener = None if path is None else _trial_dir_opener(path)
        except ValueError:
            continue
        if path is None or not isinstance(gen, str) or (directory and path.name != directory.name):
            continue
        if type(row.user_attrs.get(RETURN_CODE_ATTR)) is not int:
            try:
                lifecycle = _trial_lifecycle(row)
            except ValueError:
                lifecycle = None
            if lifecycle is not None and (
                lifecycle.state == "allocated" or lifecycle.ended_by == "not_started"
            ):
                continue
        if identity == opener:
            epoch = float("-inf")
            launches[path.name].append(DirectoryLaunch(identity, gen, epoch, epoch))
            continue
        try:
            epoch = read_evidence_baseline(_attempt_dir(path.parent, identity)).launched_at
        except ValueError:
            epoch = None
        start = row.datetime_start
        fallback = 0.0 if start is None else start.timestamp()
        launches[path.name].append(
            DirectoryLaunch(identity, gen, epoch, fallback if epoch is None else epoch)
        )
    return launches


def directory_history(
    launches: Sequence[DirectoryLaunch],
    *,
    attempt_id: str | None = None,
    launched_at: float | None = None,
    ancestors: Sequence[tuple[str, str]] = (),
) -> DirectoryHistory:
    """Resolve earlier attempts and the next reuse of a shared trainer directory.

    A known ancestor without an epoch is still earlier. An undated non-ancestor can
    prevent attribution of a previously finished attempt's shared evidence.

    :param Sequence[DirectoryLaunch] launches: One directory's physical launch records.
    :param str | None attempt_id: Existing attempt to attribute, or None before a new launch.
    :param float | None launched_at: Existing reused attempt's launch epoch.
    :param Sequence[tuple[str, str]] ancestors: Proven earlier generations and attempt IDs,
        used when an ancestor's launch epoch is unavailable.
    :return DirectoryHistory: Earlier launches in order and cutoff for other shared evidence.
    """
    known_earlier = {identity for _, identity in ancestors}
    earlier: list[DirectoryLaunch] = []
    later: list[float] = []
    undated: list[str] = []
    for launch in launches:
        identity, _, epoch, _ = launch
        if identity == attempt_id:
            continue
        if epoch is None:
            if attempt_id is None or identity in known_earlier:
                earlier.append(launch)
            else:
                undated.append(identity)
        elif (
            attempt_id is None
            or epoch == float("-inf")
            or (launched_at is not None and epoch < launched_at)
        ):
            earlier.append(launch)
        else:
            later.append(epoch)
    before: float | str | None = min(later, default=None)
    if undated:
        before = (
            f"the launch time of attempt {min(undated)!r} in its shared directory is unavailable"
        )
    return DirectoryHistory(
        tuple(
            (launch.generation, launch.attempt_id)
            for launch in sorted(earlier, key=lambda launch: launch.order)
        ),
        before,
    )


def earlier_directory_launches(
    studies: Sequence[Study], directory: Path
) -> tuple[tuple[str, str], ...]:
    """Return the launch order before a new attempt starts in this directory.

    :param Sequence[Study] studies: All studies that can share the directory.
    :param Path directory: Trainer directory the new attempt will continue.
    :return tuple[tuple[str, str], ...]: Earlier generation and attempt IDs in launch order.
    """
    launches = directory_launches(producing_attempts(studies), directory=directory)
    return directory_history(launches.get(directory.name, ())).earlier


def standing_repeat_attempts(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    """Choose one reading's representative per draw for repeat ranking and reporting.

    A measurement recovered by a re-read still stands when a later retry did not measure.
    When several attempts measured a draw, the last measurement stands.

    :param Sequence[FrozenTrial] trials: One study's trials in trial-number order.
    :return list[FrozenTrial]: One representative per draw with a non-interrupted attempt.
    """
    positions = _draw_positions(trials)
    standing = {
        positions[trial.number]: trial for trial in _unreplaced_attempts(trials, interrupted=False)
    }
    for trial in trials:
        if trial.state == TrialState.COMPLETE:
            standing[positions[trial.number]] = trial
    return list(standing.values())


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
