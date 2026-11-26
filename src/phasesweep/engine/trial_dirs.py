"""Choose an attempt's training directory and track which attempts trained there."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Container, Iterator, Mapping, Sequence
from pathlib import Path
from stat import S_ISLNK
from typing import NamedTuple

from optuna import Study
from optuna.trial import FrozenTrial, TrialState

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import _attempt_process_resolution
from phasesweep.engine.errors import WinnerIntegrityError
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
from phasesweep.runtime.process import read_attempt_lifecycle
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE


class ProducingAttempt(NamedTuple):
    """The study and row that allocated an attempt, rather than a copy of that row."""

    study: Study
    trial: FrozenTrial
    #: Numbered rows of the producing study, shared by all its attempts.
    rows: Mapping[int, FrozenTrial]


def producing_attempts(studies: Sequence[Study]) -> dict[str, ProducingAttempt]:
    """Resolve physical attempts by identity across all studies of a phase.

    An attempt belongs to the study that launched it. A study that adopts another's attempt
    keeps its attempt ID but never copies its allocation context, so a row whose generation is
    not among its own study's allocation contexts is a copy. Where those contexts cannot tell,
    the frozen generation's study address does. Trial numbers never identify copies, since
    studies of a phase number their rows independently.

    :param Sequence[Study] studies: Studies of the phase from one ledger or live run.
    :return dict[str, ProducingAttempt]: Unambiguous producing rows by attempt ID.
    """
    from phasesweep.engine.study_policy import _allocation_contexts

    numbered = {
        study.study_name: {trial.number: trial for trial in study.get_trials(deepcopy=False)}
        for study in studies
    }
    rows = {
        study.study_name: {
            attempt_id: trial
            for trial in numbered[study.study_name].values()
            if isinstance(attempt_id := trial.user_attrs.get(ATTEMPT_ID_ATTR), str)
        }
        for study in studies
    }
    owners: dict[str, list[ProducingAttempt]] = defaultdict(list)
    snapshots: dict[tuple[Path, str], Experiment | None] = {}
    generations = {
        study.study_name: {record["generation_id"] for record in _allocation_contexts(study)}
        for study in studies
    }
    have_allocations = any(generations.values())
    for study in studies:
        for attempt_id, trial in rows[study.study_name].items():
            generation = trial.user_attrs.get(GENERATION_ID_ATTR)
            if (
                have_allocations
                and isinstance(generation, str)
                and generation not in generations[study.study_name]
            ):
                continue
            owners[attempt_id].append(ProducingAttempt(study, trial, numbered[study.study_name]))
    for attempt_id, matches in owners.items():
        if len(matches) > 1:
            launched = [
                item for item in matches if _generation_addresses(item.study, item.trial, snapshots)
            ]
            if launched:
                owners[attempt_id] = launched
    return {attempt_id: matches[0] for attempt_id, matches in owners.items() if len(matches) == 1}


def _generation_addresses(
    study: Study, trial: FrozenTrial, snapshots: dict[tuple[Path, str], Experiment | None]
) -> bool:
    """Check whether retained launch settings identify a producing row.

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
    """A directory's recorded attempt and what is known about its start."""

    attempt_id: str
    generation: str
    epoch: float | None
    order: float
    may_have_started: bool


def directory_launches(
    attempts: Mapping[str, ProducingAttempt], *, directory: Path | None = None
) -> dict[str, list[DirectoryLaunch]]:
    """Index the physical launches sharing each training directory.

    The opener always sorts first. The launch baseline provides the epoch of each reuse.
    An attempt that never started can belong to the rerun's ancestry without writing evidence.

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
        may_have_started = True
        if type(row.user_attrs.get(RETURN_CODE_ATTR)) is not int:
            try:
                lifecycle = _trial_lifecycle(row)
            except ValueError:
                lifecycle = None
            if lifecycle is not None and (
                lifecycle.state == "allocated" or lifecycle.ended_by == "not_started"
            ):
                may_have_started = False
        if identity == opener:
            opener_epoch = float("-inf")
            launches[path.name].append(
                DirectoryLaunch(identity, gen, opener_epoch, opener_epoch, may_have_started)
            )
            continue
        try:
            epoch = read_evidence_baseline(_attempt_dir(path.parent, identity)).launched_at
        except ValueError:
            epoch = None
        start = row.datetime_start
        fallback = 0.0 if start is None else start.timestamp()
        launches[path.name].append(
            DirectoryLaunch(
                identity, gen, epoch, fallback if epoch is None else epoch, may_have_started
            )
        )
    return launches


def overwritable_results(
    launches: Sequence[DirectoryLaunch], standing: Container[str], *, launching: bool
) -> list[DirectoryLaunch]:
    """Pick the standing results in a directory that a later launch can overwrite.

    A trainer owns its checkpoints, so only a finished attempt whose result a study can still
    publish is disclosed; an interrupted or failed attempt holds nothing to lose.

    :param Sequence[DirectoryLaunch] launches: One directory's physical launch records.
    :param Container[str] standing: IDs of attempts whose result a study can still publish.
    :param bool launching: Whether a launch about to start follows every recorded one; when
        False the ledger is read back and the last recorded launch overwrites nothing.
    :return list[DirectoryLaunch]: Standing launches a later launch followed, in launch order.
    """
    started = sorted((item for item in launches if item.may_have_started), key=lambda i: i.order)
    return [
        item for item in (started if launching else started[:-1]) if item.attempt_id in standing
    ]


def reused_checkpoint_warnings(
    studies: Sequence[Study], winning_trials: Mapping[int, str | None]
) -> tuple[str, ...]:
    """Name winning attempts whose trainer directory a later physical launch could have changed.

    :param Sequence[Study] studies: Studies of the phase from the same ledger capture.
    :param Mapping[int, str | None] winning_trials: Selected trial numbers and physical attempt IDs, including repeat members.
    :return tuple[str, ...]: One checkpoint warning per affected member, in trial order.
    """
    numbers = {attempt: number for number, attempt in winning_trials.items() if attempt is not None}
    if not numbers:
        return ()
    warnings: dict[int, str] = {}
    for directory, launches in directory_launches(producing_attempts(studies)).items():
        for item in overwritable_results(launches, numbers, launching=False):
            number = numbers[item.attempt_id]
            warnings[number] = (
                f"Trial {number}'s checkpoint directory {directory} was reused by a later attempt. "
                f"Use a checkpoint saved for winning attempt {item.attempt_id}. A shared filename may contain a later model."
            )
    return tuple(warnings[number] for number in sorted(warnings))


def reruns_over_finished_results(
    studies: Sequence[Study], trials: Mapping[int, FrozenTrial], reruns: Sequence[int]
) -> list[tuple[int, str]]:
    """Find the reruns whose directory holds a finished attempt's model that they can overwrite.

    :param Sequence[Study] studies: Every study of the phase that can share a directory.
    :param Mapping[int, FrozenTrial] trials: The rerunning study's trials by number.
    :param Sequence[int] reruns: Trial numbers of reruns about to launch.
    :return list[tuple[int, str]]: ``(rerun, directory name)`` for each affected rerun. A
        rerun whose directory cannot be resolved is left to its launch, which refuses it.
    """
    attempts = producing_attempts(studies)
    finished = {
        attempt_id
        for attempt_id, owner in attempts.items()
        if owner.trial.state == TrialState.COMPLETE
    }
    affected: list[tuple[int, str]] = []
    for number in reruns:
        try:
            name = _expected_trial_dir_name(trials, trials[number], attempts=attempts)
        except PhaseSweepError:
            continue
        if name is None:
            continue
        launches = directory_launches(attempts, directory=Path(name)).get(name, ())
        if overwritable_results(launches, finished, launching=True):
            affected.append((number, name))
    return affected


def directory_history(
    launches: Sequence[DirectoryLaunch],
    *,
    attempt_id: str | None = None,
    launched_at: float | None = None,
    ancestors: Sequence[tuple[str, str]] = (),
) -> DirectoryHistory:
    """Resolve earlier attempts and the next reuse of a shared trainer directory.

    An explicit ancestor is known to be earlier, even if it stopped before launch. An
    unrelated attempt can affect shared evidence only if it may have started.

    :param Sequence[DirectoryLaunch] launches: One directory's physical launch records.
    :param str | None attempt_id: Existing attempt to attribute, or None before a new launch.
    :param float | None launched_at: Existing reused attempt's launch epoch.
    :param Sequence[tuple[str, str]] ancestors: Earlier generations and attempt IDs from
        the recorded rerun chain.
    :return DirectoryHistory: Earlier launches in order and cutoff for other shared evidence.
    """
    known_earlier = {identity for _, identity in ancestors}
    earlier: list[DirectoryLaunch] = []
    later: list[float] = []
    undated: list[str] = []
    for launch in launches:
        identity, epoch = launch.attempt_id, launch.epoch
        if identity == attempt_id:
            continue
        if identity in known_earlier:
            earlier.append(launch)
        elif not launch.may_have_started:
            continue
        elif epoch is None:
            if attempt_id is None:
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
    studies: Sequence[Study],
    directory: Path,
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
) -> tuple[tuple[str, str], ...]:
    """Return the launch order before a new attempt starts in this directory.

    :param Sequence[Study] studies: All studies that can share the directory.
    :param Path directory: Trainer directory the new attempt will continue.
    :param Mapping[int, FrozenTrial] trials: The current study's trials by number.
    :param FrozenTrial trial: Trial about to launch, with its recorded rerun source.
    :return tuple[tuple[str, str], ...]: Earlier generation and attempt IDs in launch order.
    """
    from phasesweep.engine.evidence import _earlier_attempts

    attempts = producing_attempts(studies)
    launches = directory_launches(attempts, directory=directory)
    return directory_history(
        launches.get(directory.name, ()),
        ancestors=_earlier_attempts(trials, trial, attempts=attempts),
    ).earlier


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


def _rerun_chain(
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
    *,
    attempts: Mapping[str, ProducingAttempt] | None = None,
) -> Iterator[FrozenTrial]:
    """Walk a draw from its latest trial to its first, including directory restarts.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: The first trial to yield.
    :param Mapping[str, ProducingAttempt] | None attempts: Producers to follow for aliases.
    :raises ProcessCleanupUncertainError: A rerun link does not name an earlier draw ancestor.
    :return Iterator[FrozenTrial]: Trials in newest-to-oldest order.
    """
    current = trial
    seen: set[str] = set()
    while True:
        identity = current.user_attrs.get(ATTEMPT_ID_ATTR)
        if attempts is not None and isinstance(identity, str):
            if identity in seen:
                raise ProcessCleanupUncertainError(
                    f"Trial {current.number} repeats attempt {identity!r} in its rerun lineage.",
                    action=OperatorAction.RESTORE_LEDGER,
                )
            seen.add(identity)
            owner = attempts.get(identity)
            if owner is not None:
                trials, current = owner.rows, owner.trial
        source_number = current.user_attrs.get(RERUN_OF_ATTR)
        source = trials.get(source_number) if type(source_number) is int else None
        if source_number is not None and (source is None or source.number >= current.number):
            raise ProcessCleanupUncertainError(
                f"Trial {current.number} names {source_number!r} under {RERUN_OF_ATTR!r}, "
                "which is not an earlier trial of its study.",
                action=OperatorAction.RESTORE_LEDGER,
            )
        yield current
        if source is None:
            return
        current = source


def _rerun_directory_sources(
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
    *,
    attempts: Mapping[str, ProducingAttempt] | None = None,
) -> Iterator[FrozenTrial]:
    """Yield earlier attempts in the same directory, through its physical opener.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: A trial about to launch or one that recorded its directory.
    :param Mapping[str, ProducingAttempt] | None attempts: Producers to follow for aliases.
    :raises ProcessCleanupUncertainError: A source is invalid or its recorded directory is malformed.
    :return Iterator[FrozenTrial]: Directory sources, from the most recent to the opener.
    """
    for index, current in enumerate(_rerun_chain(trials, trial, attempts=attempts)):
        try:
            recorded = _recorded_trial_dir(current)
        except ValueError as exc:
            raise _unusable_directory_error(current, exc) from exc
        if recorded is not None:
            if index:
                yield current
            if recorded.name == _own_directory_name(current):
                return


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


def _expected_trial_dir_name(
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
    *,
    attempts: Mapping[str, ProducingAttempt] | None = None,
) -> str | None:
    """Return the directory name a launched trial must have recorded.

    :param Mapping[int, FrozenTrial] trials: The study's trials by number.
    :param FrozenTrial trial: A launched trial.
    :param Mapping[str, ProducingAttempt] | None attempts: Producers for copied rows.
    :return str | None: The name :func:`_trial_dir_for` builds for the trial that opened its
        directory, or ``None`` when that trial records no attempt.
    """
    sources = list(_rerun_directory_sources(trials, trial, attempts=attempts))
    return (
        _own_directory_name(sources[-1])
        if sources
        else _own_directory_name(next(_rerun_chain(trials, trial, attempts=attempts)))
    )


def _trial_dir_for_attempt(
    experiment: Experiment,
    phase_name: str,
    trials: Mapping[int, FrozenTrial],
    trial: FrozenTrial,
    *,
    attempt_id: str,
    recovered: frozenset[int] = frozenset(),
    studies: Sequence[Study] = (),
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
    :param Sequence[Study] studies: Studies that may supply a continuation's physical opener.
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
    opener = sources[-1]
    if studies and _required_trial_dir(source).name != _own_directory_name(opener):
        opener = list(
            _rerun_directory_sources(trials, trial, attempts=producing_attempts(studies))
        )[-1]
    directory = _owner_directory(phase_dir, opener)
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
