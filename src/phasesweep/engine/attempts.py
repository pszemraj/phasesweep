"""Active-attempt registry and stale-process resolution."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    OperatorAction,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.ledger import open_registry_study
from phasesweep.engine.paths import _attempts_dir, _trial_dir_for
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.errors import RERUN_CLEANUP_RECOVERY
from phasesweep.runtime.files import (
    PlatformCapabilityError,
    UnsafePrivatePathError,
    canonical_storage_identity,
    open_directory_fd,
    private_atomic_write_text,
    read_private_text_at,
    storage_recovery_locator,
)
from phasesweep.runtime.json import strict_json_loads
from phasesweep.runtime.process import AttemptLifecycle, read_attempt_lifecycle
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    StaleProcessIdentity,
    cleanup_stale_trial_process,
    read_stale_process_identity,
)

_TRIAL_OUTCOMES = frozenset({"success", "failure", "pruned", "cancelled", "fatal"})
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ParsedTrialOutcome:
    """Validated durable terminal-outcome fields for one trial."""

    sequence: int
    outcome: str
    cause: str | None
    policy: str | None
    abort: dict[str, Any] | None
    """The phase abort this outcome tripped, recorded with the outcome itself."""


@dataclass
class _PreflightCleanupReport:
    """Cleanup evidence accumulated while inspecting all existing phase studies."""

    cleanup_confirmed: bool = True
    recovered_attempt_ids: set[str] = field(default_factory=set)
    recovered_attempt_generations: dict[str, str] = field(default_factory=dict)
    uncertain_attempt_ids: set[str] = field(default_factory=set)
    error: BaseException | None = None

    def mark_uncertain(self, error: BaseException) -> None:
        """Record the first cleanup uncertainty and fail the aggregate closed."""
        self.cleanup_confirmed = False
        if self.error is None:
            self.error = error

    def record_generations(self, attempts: Mapping[str, _AttemptRecord]) -> None:
        """Record the producing generation of each attempt that has one.

        :param Mapping[str, _AttemptRecord] attempts: Recovered attempts keyed by id.
        """
        self.recovered_attempt_generations.update(
            {
                attempt_id: record.generation_id
                for attempt_id, record in attempts.items()
                if record.generation_id is not None
            }
        )


def _trial_dir_for_reaping(
    trial: optuna.trial.FrozenTrial,
    experiment: Experiment,
    phase_name: str,
    study_name: str,
) -> Path:
    """Return the trial directory to inspect during stale-trial reaping.

    Prefer the persisted ``phasesweep_trial_dir`` attr because it preserves the
    original workdir even if the operator resumes from a different cwd or edits
    ``experiment.workdir``. If the attr is absent, the trial died before the
    current launch path could persist the directory and before any subprocess
    could be started, so the canonical directory is safe to use for recovery.

    :param optuna.trial.FrozenTrial trial: Running trial being reaped.
    :param Experiment experiment: Parsed experiment.
    :param str phase_name: Phase containing the trial.
    :param str study_name: Study name used in diagnostics.
    :raises ProcessCleanupUncertainError: A persisted trial-directory value is
        invalid, so identity files cannot be located safely.
    :return Path: Persisted trial directory, or the canonical pre-launch directory.
    """
    if TRIAL_DIR_ATTR not in trial.user_attrs:
        trial_dir = _trial_dir_for(experiment, phase_name, trial.number)
        log.warning(
            "RUNNING trial %d in study %s is missing %r; falling back to "
            "canonical trial_dir=%s. No subprocess can be launched by the "
            "current orchestrator before this attr is normally persisted.",
            trial.number,
            study_name,
            TRIAL_DIR_ATTR,
            trial_dir,
        )
        return trial_dir

    stored = trial.user_attrs[TRIAL_DIR_ATTR]
    # recover-run reaps through here too, so it cannot repair the attribute;
    # only the ledger that recorded it can.
    if not isinstance(stored, str) or not stored:
        raise ProcessCleanupUncertainError(
            f"Refusing to reap RUNNING trial {trial.number}: invalid persisted "
            f"{TRIAL_DIR_ATTR!r} user attribute {stored!r}. The trial cannot be "
            "tied to its identity files safely. Restore the original storage ledger "
            "before retrying recovery.",
            action=OperatorAction.RESTORE_LEDGER,
        )
    return Path(stored)


def _trial_attempt_record(
    trial: optuna.trial.FrozenTrial,
    phase_name: str,
    trial_dir: Path,
) -> _AttemptRecord | None:
    """Build the attempt record a trial's durable attributes name.

    :param optuna.trial.FrozenTrial trial: Trial whose attempt is recorded.
    :param str phase_name: Phase that owns ``trial``.
    :param Path trial_dir: The trial's persisted (or canonical pre-launch) directory.
    :return _AttemptRecord | None: The record, or ``None`` when the trial has
        no valid persisted attempt id.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    return _AttemptRecord(
        attempt_id=attempt_id,
        trial_dir=trial_dir,
        generation_id=generation_id if isinstance(generation_id, str) and generation_id else None,
        phase_name=phase_name,
        trial_number=trial.number,
    )


def _require_trial_attempt(
    trial: optuna.trial.FrozenTrial,
    phase_name: str,
    trial_dir: Path,
    study_name: str,
) -> _AttemptRecord:
    """Normalize a trial whose process must be resolved into its attempt record.

    :param optuna.trial.FrozenTrial trial: RUNNING or terminal trial being recovered.
    :param str phase_name: Phase that owns ``trial``.
    :param Path trial_dir: The trial's persisted directory.
    :param str study_name: Study name, used only for diagnostics.
    :return _AttemptRecord: The trial's attempt record.
    :raises ProcessCleanupUncertainError: The trial has no valid persisted
        attempt id, so its process identity is unknown.
    """
    record = _trial_attempt_record(trial, phase_name, trial_dir)
    # recover-run resolves trials through here too, so this refusal must not
    # send the operator back to recover-run; neither may the trial refusals in
    # _TrialDiscovery.
    if record is None:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover trial {trial.number} in study {study_name}: missing or "
            f"invalid {ATTEMPT_ID_ATTR!r} user attribute. Process identity is unknown. "
            "Restore the original storage ledger with its durable attempt and generation "
            "identities before retrying recovery; do not infer replacement identities.",
            action=OperatorAction.RESTORE_LEDGER,
        )
    return record


ATTEMPT_REGISTRY_SCHEMA_VERSION = 3
_ATTEMPT_ENTRY_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_id",
        "experiment",
        "phase",
        "study_name",
        "storage_identity",
        "storage_locator",
        "trial_number",
        "trial_dir",
        "generation_id",
    }
)


def _register_active_attempt(
    experiment: Experiment,
    *,
    attempt_id: str,
    phase_name: str,
    study_name: str,
    trial_number: int,
    trial_dir: Path,
    generation_id: str,
) -> None:
    """Durably register a newly allocated attempt before anything is launched.

    The entry binds the attempt to the *producing* phase name, study name,
    and a frozen storage locator, so preflight can find and resolve it even after the
    phase was renamed or removed, or the storage URL changed (review
    v0.5.17 / blocker 3).

    Creation is fail-closed. The write used to be swallowed on the theory
    that "the per-phase reaper still recovers it under the same config", but
    that is exactly wrong for the case this registry exists to cover: the
    reaper walks only the phases the *current* config declares, so a rename,
    a removal, or a changed storage URL is precisely when the lost entry was
    the sole record of a trainer that may still be alive. Refusing to launch
    an attempt PhaseSweep could not record keeps that unreachable state from
    ever existing, and costs nothing to undo because this runs before the GPU
    lease and the trainer (PR #5 review / reviewer 2 pass 2, blocker 5).

    :param Experiment experiment: Experiment supplying the registry root.
    :param str attempt_id: Immutable attempt identity and entry filename.
    :param str phase_name: Producing phase name.
    :param str study_name: Fully qualified Optuna study name.
    :param int trial_number: Optuna trial number bound to the attempt.
    :param Path trial_dir: Trial directory holding lifecycle and process identity.
    :param str generation_id: Engine invocation identity.
    :raises ActiveAttemptPersistenceError: The entry could not be written, so
        no trainer was started or GPU lease consumed.
    """
    entry_path = _attempts_dir(experiment) / f"{attempt_id}.json"
    try:
        entry = {
            "schema_version": ATTEMPT_REGISTRY_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "experiment": experiment.experiment,
            "phase": phase_name,
            "study_name": study_name,
            "storage_identity": canonical_storage_identity(experiment.resolved_storage),
            "storage_locator": storage_recovery_locator(experiment.resolved_storage),
            "trial_number": trial_number,
            "trial_dir": str(trial_dir),
            "generation_id": generation_id,
        }
        private_atomic_write_text(entry_path, json.dumps(entry, sort_keys=True) + "\n")
    except (OSError, PlatformCapabilityError, UnsafePrivatePathError) as exc:
        raise ActiveAttemptPersistenceError(
            f"Could not register active attempt {attempt_id} for phase {phase_name!r} "
            f"at {entry_path}: {exc}. No trainer was started and no GPU lease was "
            "consumed. This entry is the only durable record that would let a later "
            "run discover this attempt after a phase rename/removal or a storage-URL "
            "change, so PhaseSweep refuses to launch a trainer it could not account "
            "for. Restore write access to the experiment workdir, then run again. With "
            "persistent storage this refusal durably aborts the phase, so that run also "
            "needs n_trials raised above its accepted target, or a new experiment name."
        ) from exc


def _retire_active_attempt(experiment: Experiment, attempt_id: str) -> None:
    """Best-effort removal of a registry entry whose trial is durably terminal.

    Deletion stays best-effort even though creation is fail-closed: a retained
    entry is safe by construction. It names an already-terminal trial, so
    preflight resolves it against a durable ``exited`` lifecycle, changes
    nothing in the study, and garbage-collects the file. Only a *missing*
    entry loses information (PR #5 review / reviewer 2 pass 2, blocker 5).

    :param Experiment experiment: Experiment supplying the registry root.
    :param str attempt_id: Attempt whose trial reached a terminal state.
    """
    with (
        contextlib.suppress(OSError, PlatformCapabilityError, ProcessCleanupUncertainError),
        _open_attempt_registry(experiment) as (directory_fd, _entry_paths),
    ):
        if directory_fd is not None:
            os.unlink(f"{attempt_id}.json", dir_fd=directory_fd)


@contextlib.contextmanager
def _open_attempt_registry(
    experiment: Experiment,
) -> Iterator[tuple[int | None, list[Path]]]:
    """Open and enumerate the private attempt registry without following links.

    :param Experiment experiment: Experiment whose registry is inspected.
    :return Iterator[tuple[int | None, list[Path]]]: Stable directory descriptor
        and sorted JSON display paths; ``(None, [])`` when no registry exists.
    :raises ProcessCleanupUncertainError: The registry path or permissions are unsafe.
    """
    attempts_dir = _attempts_dir(experiment)
    try:
        directory_fd = open_directory_fd(
            attempts_dir,
            create=False,
            private_final=True,
        )
    except FileNotFoundError:
        yield None, []
        return
    # A missing platform capability is not a damaged registry, so it
    # propagates with its own routing instead of this repair.
    except (OSError, UnsafePrivatePathError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry {attempts_dir} is not a real owner-only directory. "
            "Recovery cannot trust process authority reached through a symlink or "
            "shared path. Restore the original registry with mode 0700 before retrying.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    try:
        try:
            names = sorted(name for name in os.listdir(directory_fd) if name.endswith(".json"))
        except OSError as exc:
            # recover-run enumerates the same registry, so it cannot clear this.
            raise ProcessCleanupUncertainError(
                f"Attempt registry {attempts_dir} cannot be enumerated safely. "
                "Restore the original registry and access to it before retrying.",
                action=OperatorAction.RESTORE_TREE,
            ) from exc
        yield directory_fd, [attempts_dir / name for name in names]
    finally:
        os.close(directory_fd)


def _load_attempt_entry(entry_path: Path, *, directory_fd: int) -> dict[str, Any]:
    """Load and validate one attempt registry entry.

    :param Path entry_path: Registry entry file to parse.
    :param int directory_fd: Open descriptor for the entry's validated parent directory.
    :return dict[str, Any]: The validated entry payload.
    :raises ProcessCleanupUncertainError: The entry is unreadable or malformed
        — recovery cannot know whether a process from it is still alive.
    """
    # recover-run loads every entry through here too, so it cannot clear a bad
    # one: the entry file is what the operator repairs or removes.
    try:
        payload = strict_json_loads(read_private_text_at(directory_fd, entry_path.name, entry_path))
    except (OSError, UnsafePrivatePathError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} is unreadable or malformed. "
            "Recovery cannot prove whether a process from this attempt is still "
            "alive. Investigate the attempt's trial directory, then delete the "
            "entry file if you are certain nothing is running.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    locator_identity: str | None = None
    if isinstance(payload, dict) and isinstance(payload.get("storage_locator"), str):
        try:
            locator_identity = canonical_storage_identity(payload["storage_locator"])
        except ValueError as exc:
            raise ProcessCleanupUncertainError(
                f"Attempt registry entry {entry_path} has an unsupported or partial "
                "schema. Delete the entry file only if you are certain no process "
                "from this attempt is running.",
                action=OperatorAction.RESTORE_TREE,
            ) from exc
    if (
        not isinstance(payload, dict)
        or not _ATTEMPT_ENTRY_REQUIRED_FIELDS.issubset(payload)
        or payload.get("schema_version") != ATTEMPT_REGISTRY_SCHEMA_VERSION
        or not isinstance(payload.get("attempt_id"), str)
        or not isinstance(payload.get("trial_dir"), str)
        or not isinstance(payload.get("study_name"), str)
        or (
            payload.get("storage_identity") is not None
            and not isinstance(payload.get("storage_identity"), str)
        )
        or (
            payload.get("storage_locator") is not None
            and not isinstance(payload.get("storage_locator"), str)
        )
        or (
            payload.get("storage_locator") is not None
            and locator_identity != payload.get("storage_identity")
        )
        or not isinstance(payload.get("generation_id"), str)
        or not payload.get("generation_id")
        or type(payload.get("trial_number")) is not int
    ):
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} has an unsupported or partial "
            "schema. Delete the entry file only if you are certain no process "
            "from this attempt is running.",
            action=OperatorAction.RESTORE_TREE,
        )
    return payload


def _attempt_process_resolution(
    lifecycle: AttemptLifecycle | None,
    *,
    identity_exists: bool,
) -> str:
    """Classify the fail-closed recovery authority for one interrupted attempt.

    :param AttemptLifecycle | None lifecycle: Validated durable lifecycle, if present.
    :param bool identity_exists: Whether a retained process-identity file exists.
    :return str: ``"exited"`` for confirmed cleanup, ``"allocated"`` when launch
        provably never began, or ``"identity"`` when the process identity must
        be validated (and, outside inspection mode, cleaned).
    """
    if lifecycle is not None and lifecycle.state == "exited" and lifecycle.cleanup_confirmed:
        return "exited"
    if not identity_exists and lifecycle is not None and lifecycle.state == "allocated":
        return "allocated"
    return "identity"


def _registry_attempt_record(entry: Mapping[str, Any], entry_path: Path) -> _AttemptRecord:
    """Normalize one validated registry entry into the attempt it registered.

    The entry alone identifies the attempt, so it resolves without the
    producing phase still existing in the config or the producing storage
    being reachable.

    :param Mapping[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :return _AttemptRecord: The registered attempt.
    :raises ProcessCleanupUncertainError: The entry's trial directory is
        missing, so its process state cannot be verified.
    """
    trial_dir = Path(entry["trial_dir"])
    # Inspection and confirmed recovery both stop at each evidence check, so
    # recover-run cannot clear this entry; the operator repairs or removes it.
    if not trial_dir.is_dir():
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} points at a missing trial "
            f"directory {trial_dir}; its process state cannot be verified. "
            "Delete the entry file only if you are certain nothing is running.",
            action=OperatorAction.RESTORE_TREE,
        )
    return _AttemptRecord(
        attempt_id=entry["attempt_id"],
        trial_dir=trial_dir,
        generation_id=entry["generation_id"],
        phase_name=entry["phase"],
        trial_number=entry["trial_number"],
    )


def _valid_abort_record(value: object, *, sequence: int) -> bool:
    """Return whether the abort record stored with an outcome is well formed.

    :param object value: Raw abort record carried by a terminal outcome.
    :param int sequence: Completion sequence of the outcome that carries it.
    :return bool: Whether the record names that sequence, a positive trial
        target, and a non-empty policy and cause.
    """
    if not isinstance(value, dict):
        return False
    completion_sequence = value.get("completion_sequence")
    trial_target = value.get("trial_target")
    policy = value.get("policy")
    cause = value.get("cause")
    return (
        value.get("schema_version") == 1
        and type(completion_sequence) is int
        and completion_sequence == sequence
        and type(trial_target) is int
        and trial_target >= 1
        and isinstance(policy, str)
        and bool(policy)
        and isinstance(cause, str)
        and bool(cause)
    )


def _parsed_trial_outcome(value: Any) -> _ParsedTrialOutcome | None:
    """Return the ordered outcome fields when a trial attr is well formed.

    :param Any value: Raw ``TRIAL_OUTCOME_ATTR`` user-attr value to validate.
    :return _ParsedTrialOutcome | None: Validated outcome, including the fatal
        policy needed to distinguish cleanup uncertainty from ordinary errors
        and the abort record a failure or fatal outcome tripped; ``None`` when
        the payload is malformed.
    """
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schema_version")
    sequence = value.get("sequence")
    outcome = value.get("outcome")
    cause = value.get("cause")
    policy = value.get("policy")
    abort = value.get(TRIAL_OUTCOME_ABORT_KEY)
    if (
        schema_version != TRIAL_OUTCOME_SCHEMA_VERSION
        or type(sequence) is not int
        or sequence < 1
        or outcome not in _TRIAL_OUTCOMES
        or (cause is not None and not isinstance(cause, str))
        or (
            policy is not None and (outcome != "fatal" or not isinstance(policy, str) or not policy)
        )
        or (
            abort is not None
            and (
                outcome not in {"failure", "fatal"}
                or not _valid_abort_record(abort, sequence=sequence)
            )
        )
    ):
        return None
    return _ParsedTrialOutcome(
        sequence=sequence,
        outcome=outcome,
        cause=cause,
        policy=policy,
        abort=abort,
    )


def _trial_requires_cleanup_recovery(trial: optuna.trial.FrozenTrial) -> bool:
    """Return whether a terminal trial still lacks positive cleanup evidence.

    The explicit cleanup attribute is forensic redundancy, not the sole
    authority: its best-effort write can fail after the fatal outcome ledger
    durably records ``unsafe_process_cleanup``. The study-level recovery ledger
    is the only transition that consumes either form of uncertainty.

    :param optuna.trial.FrozenTrial trial: Trial whose durable cleanup state is inspected.
    :return bool: Whether ordinary preflight must confirm cleanup before new work.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    unsafe_outcome = (
        parsed is not None
        and parsed.outcome == "fatal"
        and parsed.policy == "unsafe_process_cleanup"
    )
    return trial.user_attrs.get(CLEANUP_CONFIRMED_ATTR) is False or unsafe_outcome


def _record_stale_trial_failure(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Persist a failure-policy outcome before marking a stale trial ``FAIL``.

    An objective may have written its outcome immediately before the
    orchestrator died. Preserve a recorded failure, fatal exception, or explicit
    host cancellation, with any abort it tripped; convert a not-yet-committed
    success/prune to failure because recovery is about to commit the trial as
    ``FAIL``. Recovery itself never records an abort decision. Malformed or duplicate
    records are replaced with a fresh sequence so current-schema validation
    can still diagnose any other corrupt row without stranding this stale
    process.

    :param optuna.Study study: Study containing ``trial``, used to read every
        trial's recorded outcome and assign a fresh sequence if needed.
    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial about to be
        marked ``FAIL``.
    :raises StudyStorageUnavailableError: The outcome could not be written to trial user
        attrs; the trial is left ``RUNNING`` rather than risk silently
        dropping the failure from ``max_consecutive_failures``.
    """
    trials = study.get_trials(deepcopy=False)
    parsed_by_trial = {
        candidate.number: parsed
        for candidate in trials
        if (parsed := _parsed_trial_outcome(candidate.user_attrs.get(TRIAL_OUTCOME_ATTR)))
        is not None
    }
    used_sequences = [parsed.sequence for parsed in parsed_by_trial.values()]
    existing = parsed_by_trial.get(trial.number)
    policy: str | None = None
    abort: dict[str, Any] | None = None
    if existing is not None and used_sequences.count(existing.sequence) == 1:
        sequence = existing.sequence
        outcome = existing.outcome
        cause = existing.cause
        policy = existing.policy
        abort = existing.abort
        if outcome not in {"failure", "fatal", "cancelled"}:
            outcome = "failure"
            cause = "orchestrator stopped before Optuna committed the terminal trial state"
            policy = None
    else:
        sequence = max(used_sequences, default=0) + 1
        outcome = "failure"
        cause = "stale RUNNING trial recovered after its orchestrator stopped"

    payload: dict[str, Any] = {
        "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
        "sequence": sequence,
        "outcome": outcome,
    }
    if cause is not None:
        payload["cause"] = cause
    if policy is not None:
        payload["policy"] = policy
    if abort is not None:
        payload[TRIAL_OUTCOME_ABORT_KEY] = abort
    try:
        active_trial = optuna.Trial(study, trial._trial_id)
        active_trial.set_user_attr(TRIAL_OUTCOME_ATTR, payload)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Process cleanup completed for stale RUNNING trial {trial.number} in study "
            f"{study.study_name!r}, but its durable failure outcome could not be recorded. "
            "The trial remains RUNNING so a later retry cannot silently omit this failure "
            "from max_consecutive_failures."
        ) from exc


def _registry_attempt_fail_stale_trial(
    entry: dict[str, Any],
    entry_path: Path,
    *,
    current_storage: str | None,
) -> str:
    """Mark the entry's Optuna trial FAIL through a matching storage locator.

    The allocation-time locator remains the authority when the current config
    names another target. When both credential-free identities match, the
    current locator is preferred so credential rotation cannot strand a stale
    trial behind an obsolete password or token. Phase rename/removal does not
    affect this comparison because registry recovery is phase-graph-independent.

    :param dict[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :param str | None current_storage: Current config's operational storage URL.
    :return str: ``"reaped"`` when the stale RUNNING trial was marked FAIL,
        ``"recovered"`` when an interrupted pass already recorded that
        transition in the study ledger, ``"terminal"`` when nothing needed
        to change (trial already terminal, study gone, or in-memory storage),
        or ``"unreachable"`` when the recorded storage could not be read
        and the entry must be retained for a later retry.
    :raises StudySchemaMismatchError: The registry and trial record
        conflicting generation identities, or the ledger the entry names
        holds unsupported PhaseSweep study state.
    :raises ProcessCleanupUncertainError: A terminal trial lacks the identity
        needed to attribute confirmed cleanup to the registry's attempt.
    :raises StudyStorageUnavailableError: The stale RUNNING trial could not be
        marked FAIL, or its durable recovery state could not be recorded; the
        study is left inconsistent rather than silently dropping the failure.
    """
    current_identity = canonical_storage_identity(current_storage)
    storage_url = (
        storage_recovery_locator(current_storage)
        if current_identity == entry["storage_identity"]
        else entry["storage_locator"]
    )
    if storage_url is None:
        # In-memory storage died with its orchestrator; nothing to update.
        return "terminal"
    try:
        study = open_registry_study(storage_url, entry["study_name"])
        if study is None:
            # The ledger confirmed the study is gone without being recreated
            # by the lookup, so nothing remains for this entry to repair.
            return "terminal"
        trials = study.get_trials(deepcopy=False)
    except StudySchemaMismatchError:
        # The entry was written by this release, so its ledger was current
        # when the attempt registered. Unsupported state there now is not
        # something to reap through; it stops the run with the entry intact.
        raise
    except Exception:  # noqa: BLE001 - unreachable storage keeps the entry for retry
        log.warning(
            "Attempt registry entry %s references storage that cannot be "
            "read right now; the stale trial will be retried on a later run.",
            entry_path,
        )
        return "unreachable"
    trial = next((t for t in trials if t.number == entry["trial_number"]), None)
    if trial is None:
        return "terminal"
    stored_attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if stored_attempt_id is not None and stored_attempt_id != entry["attempt_id"]:
        # A conflicting durable id proves the row belongs to another
        # attempt. A missing id is different: registration is written before
        # the first Optuna attr, so it is the expected crash/storage-failure
        # window. The registry entry plus the already-validated lifecycle or
        # process identity still binds this exact study and trial safely.
        return "terminal"
    stored_generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    if stored_generation_id is not None and stored_generation_id != entry["generation_id"]:
        raise StudySchemaMismatchError(
            f"Attempt registry entry {entry_path} identifies generation "
            f"{entry['generation_id']!r}, but its trial {trial.number} in study "
            f"{entry['study_name']!r} records {stored_generation_id!r}. Refusing to "
            "overwrite conflicting recovery identity."
        )
    if trial.state != optuna.trial.TrialState.RUNNING:
        recovered = trial.number in _cleanup_recovered_trial_numbers(study)
        if trial.state.is_finished() and (recovered or _trial_requires_cleanup_recovery(trial)):
            if stored_attempt_id is None or stored_generation_id is None:
                raise ProcessCleanupUncertainError(
                    f"Attempt registry entry {entry_path} cannot be matched to terminal "
                    f"trial {trial.number} in study {entry['study_name']!r}: its durable "
                    "attempt or generation identity is missing. The registry entry is "
                    "retained; cleanup recovery cannot be attributed to this trial. "
                    "Restore the original storage ledger with its durable attempt and "
                    "generation identities before retrying recovery; do not infer "
                    "replacement identities.",
                    action=OperatorAction.RESTORE_LEDGER,
                )
            if not recovered:
                _record_cleanup_recovery(study, trial)
            return "recovered"
        return "terminal"
    if stored_attempt_id is None or stored_generation_id is None:
        try:
            active_trial = optuna.Trial(study, trial._trial_id)
            if stored_attempt_id is None:
                active_trial.set_user_attr(ATTEMPT_ID_ATTR, entry["attempt_id"])
            if stored_generation_id is None:
                active_trial.set_user_attr(GENERATION_ID_ATTR, entry["generation_id"])
        except Exception as exc:
            raise StudyStorageUnavailableError(
                f"Process cleanup completed for registered attempt {entry['attempt_id']}, "
                f"but its durable trial identity could not be restored in study "
                f"{entry['study_name']!r}. The trial remains RUNNING and the registry "
                "entry is retained for retry."
            ) from exc
    _fail_stale_running_trial(
        study,
        trial,
        failure_message=(
            f"Process cleanup completed for registered attempt {entry['attempt_id']}, "
            f"but its stale RUNNING trial {trial.number} in study "
            f"{entry['study_name']!r} could not be marked FAIL. Refusing to "
            "continue with an inconsistent study."
        ),
    )
    log.warning(
        "Reaped stale RUNNING trial %d in study %s via the attempt registry",
        trial.number,
        entry["study_name"],
    )
    return "reaped"


def _preflight_active_attempts(
    experiment: Experiment,
    report: _PreflightCleanupReport,
    *,
    confirm: bool,
    retain_recovery_evidence: bool = False,
) -> dict[str, str]:
    """Resolve every registered nonterminal attempt before any launch.

    Runs before the per-phase study loop and is deliberately independent of
    the current phase graph: a stale attempt whose phase was renamed or
    removed, or whose storage URL changed, is still discovered, its process
    group verified/cleaned, and its recorded study repaired (review v0.5.17 /
    blocker 3). Inspection, the observational half of ``mcp recover-run``,
    scans the same registry, so renamed phases and unavailable current
    storage cannot hide a trainer from the dry run either.

    :param Experiment experiment: Parsed experiment whose registry is scanned.
    :param _PreflightCleanupReport report: Shared cleanup-evidence collector.
        Inspection records only the uncertainty it refuses on.
    :param bool confirm: Clean each attempt's process group, repair its
        recorded study, and retire its entry when true; otherwise validate
        each entry's recovery evidence without signalling or changing state.
    :param bool retain_recovery_evidence: Keep entries backed by uncommitted
        cleanup evidence so interrupted operator recovery can retry even when
        the attempt's recorded storage differs from the current config.
    :return dict[str, str]: Registered attempt ids mapped to their producing
        generation ids.
    :raises ProcessCleanupUncertainError: A registered attempt could not be
        proven safe.
    """
    inspected: dict[str, str] = {}
    with _open_attempt_registry(experiment) as (directory_fd, entry_paths):
        if directory_fd is None:
            return inspected
        for entry_path in entry_paths:
            entry = _load_attempt_entry(entry_path, directory_fd=directory_fd)
            attempt_id = entry["attempt_id"]
            inspected[attempt_id] = entry["generation_id"]
            try:
                _resolve_attempt(
                    _registry_attempt_record(entry, entry_path),
                    _RegistryEntryDiscovery(entry_path),
                    confirm=confirm,
                )
                if not confirm:
                    continue
                outcome = _registry_attempt_fail_stale_trial(
                    entry,
                    entry_path,
                    current_storage=experiment.resolved_storage,
                )
            except ProcessCleanupUncertainError as exc:
                report.uncertain_attempt_ids.add(attempt_id)
                report.mark_uncertain(exc)
                raise
            if outcome in {"reaped", "recovered"}:
                report.recovered_attempt_ids.add(attempt_id)
                report.recovered_attempt_generations[attempt_id] = entry["generation_id"]
            if outcome != "unreachable" and not (
                retain_recovery_evidence and outcome in {"reaped", "recovered"}
            ):
                with contextlib.suppress(OSError):
                    os.unlink(entry_path.name, dir_fd=directory_fd)
    return inspected


@dataclass(frozen=True)
class _AttemptRecord:
    """One discovered attempt: its identity, trial directory, and producing trial.

    Every source that discovers an attempt (a registry entry, a stale RUNNING
    trial, a terminal trial that recorded cleanup uncertainty) normalizes it
    into this record before :func:`_resolve_attempt` may inspect or signal
    anything. A status snapshot's trial reference never becomes one: without
    a durable attempt id and trial directory it is not a recovery identity.

    ``generation_id`` is ``None`` when the trial records no valid generation.
    """

    attempt_id: str
    trial_dir: Path
    generation_id: str | None
    phase_name: str
    trial_number: int


class _Discovery(ABC):
    """How one discovery source words the resolution of its attempts.

    A source chooses only the wording and operator action of each refusal,
    its log lines, and whether a durable lifecycle record may settle its
    attempts without a process identity; the rule itself is
    :func:`_resolve_attempt`.
    """

    lifecycle_settles: ClassVar[bool] = True

    @abstractmethod
    def lifecycle_malformed(self, record: _AttemptRecord) -> ProcessCleanupUncertainError:
        """Refuse an attempt whose lifecycle record is malformed or another attempt's.

        :param _AttemptRecord record: Attempt being resolved.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """

    @abstractmethod
    def identity_unreadable(self, record: _AttemptRecord) -> ProcessCleanupUncertainError:
        """Refuse an attempt whose durable process identity cannot be validated.

        :param _AttemptRecord record: Attempt being resolved.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """

    @abstractmethod
    def cleanup_unconfirmed(
        self, record: _AttemptRecord, identity: StaleProcessIdentity
    ) -> ProcessCleanupUncertainError:
        """Refuse an attempt whose process group could not be proven gone.

        :param _AttemptRecord record: Attempt being resolved.
        :param StaleProcessIdentity identity: The validated identity that was signalled.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """

    @abstractmethod
    def settled(self, record: _AttemptRecord, lifecycle: AttemptLifecycle) -> None:
        """Report an attempt its lifecycle settled without signalling anything.

        :param _AttemptRecord record: Attempt that was resolved.
        :param AttemptLifecycle lifecycle: The ``exited`` or ``allocated`` record.
        """

    @abstractmethod
    def cleaned(self, record: _AttemptRecord, identity: StaleProcessIdentity) -> None:
        """Report an attempt whose process group was confirmed gone.

        :param _AttemptRecord record: Attempt that was resolved.
        :param StaleProcessIdentity identity: The identity that was cleaned.
        """


@dataclass(frozen=True)
class _RegistryEntryDiscovery(_Discovery):
    """An attempt discovered through its entry in the attempt registry.

    Every refusal names the entry file, which is what the operator repairs
    or removes.
    """

    entry_path: Path

    def lifecycle_malformed(self, record: _AttemptRecord) -> ProcessCleanupUncertainError:
        """Refuse an entry whose lifecycle record is malformed.

        :param _AttemptRecord record: Attempt being resolved.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Attempt registry entry {self.entry_path} has a malformed lifecycle record in "
            f"{record.trial_dir}. Delete the entry file only if you are certain nothing is "
            "running.",
            action=OperatorAction.RESTORE_TREE,
        )

    def identity_unreadable(self, record: _AttemptRecord) -> ProcessCleanupUncertainError:
        """Refuse an entry whose process identity is missing or malformed.

        :param _AttemptRecord record: Attempt being resolved.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Attempt registry entry {self.entry_path} has a missing or malformed "
            f"process identity in {record.trial_dir}. "
            "Delete the entry file only if you are certain nothing is running.",
            action=OperatorAction.RESTORE_TREE,
        )

    def cleanup_unconfirmed(
        self, record: _AttemptRecord, identity: StaleProcessIdentity
    ) -> ProcessCleanupUncertainError:
        """Refuse a registered attempt that may still have a live process group.

        :param _AttemptRecord record: Attempt being resolved.
        :param StaleProcessIdentity identity: The validated identity that was signalled.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Registered attempt {record.attempt_id} (phase {record.phase_name!r}, from "
            f"{self.entry_path}) may still have a live process group. "
            f"trial_dir={record.trial_dir} pid={identity.pid} pgid={identity.pgid}. "
            f"Investigate (e.g. `ps -o pid,pgid,cmd -p {identity.pid}`), then "
            "re-run phasesweep."
        )

    def settled(self, record: _AttemptRecord, lifecycle: AttemptLifecycle) -> None:
        """Settle a registered attempt quietly; repairing its study reports the outcome.

        :param _AttemptRecord record: Attempt that was resolved.
        :param AttemptLifecycle lifecycle: The ``exited`` or ``allocated`` record.
        """

    def cleaned(self, record: _AttemptRecord, identity: StaleProcessIdentity) -> None:
        """Log the cleared process group of a registered attempt.

        :param _AttemptRecord record: Attempt that was resolved.
        :param StaleProcessIdentity identity: The identity that was cleaned.
        """
        log.warning(
            "Cleared orphaned group for registered attempt %s (pid=%s pgid=%s)",
            record.attempt_id,
            identity.pid,
            identity.pgid,
        )


@dataclass(frozen=True)
class _TrialDiscovery(_Discovery):
    """An attempt discovered through its stale RUNNING trial in a phase study.

    ``mcp recover-run`` resolves trials this way too, so no refusal here
    sends the operator back to recover-run.
    """

    study_name: str

    def lifecycle_malformed(self, record: _AttemptRecord) -> ProcessCleanupUncertainError:
        """Refuse a trial whose lifecycle record is malformed or another attempt's.

        :param _AttemptRecord record: Attempt being resolved.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Refusing to recover trial {record.trial_number} in study {self.study_name}: its "
            f"attempt lifecycle record is malformed or belongs to another attempt. "
            f"trial_dir={record.trial_dir}. Restore the original storage ledger and this "
            "attempt's lifecycle record before retrying recovery.",
            # Either side may be the one that changed, so no one repair is the remedy.
            action=OperatorAction.INSPECT_LOGS,
        )

    def identity_unreadable(self, record: _AttemptRecord) -> ProcessCleanupUncertainError:
        """Refuse a trial whose process identity is unusable or another attempt's.

        :param _AttemptRecord record: Attempt being resolved.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Refusing to recover trial {record.trial_number} in study {self.study_name}: its "
            f"durable process identity is missing, malformed, partial, or belongs to another "
            f"attempt. trial_dir={record.trial_dir}. Restore the original storage ledger and "
            "this attempt's process-identity files before retrying recovery.",
            # Either side may be the one that changed, so no one repair is the remedy.
            action=OperatorAction.INSPECT_LOGS,
        )

    def cleanup_unconfirmed(
        self, record: _AttemptRecord, identity: StaleProcessIdentity
    ) -> ProcessCleanupUncertainError:
        """Refuse to fail a RUNNING trial whose process group may still be alive.

        :param _AttemptRecord record: Attempt being resolved.
        :param StaleProcessIdentity identity: The validated identity that was signalled.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Refusing to mark RUNNING trial {record.trial_number}: stale process cleanup "
            f"could not prove the process group is gone. trial_dir={record.trial_dir} "
            f"pid={identity.pid} pgid={identity.pgid}. A leaked training "
            "process may still be holding GPU memory. Investigate "
            f"(e.g. `ps -o pid,pgid,cmd -p {identity.pid}` and "
            f"`kill -9 -- -{identity.pgid}` if appropriate), then re-run "
            "phasesweep."
        )

    def settled(self, record: _AttemptRecord, lifecycle: AttemptLifecycle) -> None:
        """Log a RUNNING trial that is failed without signalling.

        :param _AttemptRecord record: Attempt that was resolved.
        :param AttemptLifecycle lifecycle: The ``exited`` or ``allocated`` record.
        """
        if lifecycle.state == "exited":
            log.warning(
                "Trial %d in study %s exited (rc=%s) before its terminal state was "
                "committed; failing it without signalling.",
                record.trial_number,
                self.study_name,
                lifecycle.return_code,
            )
        else:
            log.warning(
                "Trial %d in study %s was allocated but no process was ever launched "
                "(orchestrator died while queued); failing it without signalling.",
                record.trial_number,
                self.study_name,
            )

    def cleaned(self, record: _AttemptRecord, identity: StaleProcessIdentity) -> None:
        """Log the cleared process group of a stale RUNNING trial.

        :param _AttemptRecord record: Attempt that was resolved.
        :param StaleProcessIdentity identity: The identity that was cleaned.
        """
        log.warning(
            "Cleared orphaned group for trial %d (pid=%s pgid=%s)",
            record.trial_number,
            identity.pid,
            identity.pgid,
        )


@dataclass(frozen=True)
class _CleanupUncertainTrialDiscovery(_TrialDiscovery):
    """An attempt discovered through a terminal trial that recorded cleanup uncertainty.

    The trial is already terminal, so no lifecycle record can settle it: only
    a validated identity whose group is confirmed gone clears the uncertainty.
    """

    lifecycle_settles: ClassVar[bool] = False
    experiment_name: str

    def cleanup_unconfirmed(
        self, record: _AttemptRecord, identity: StaleProcessIdentity
    ) -> ProcessCleanupUncertainError:
        """Refuse to clear the recorded uncertainty of a trial whose group may be alive.

        :param _AttemptRecord record: Attempt being resolved.
        :param StaleProcessIdentity identity: The validated identity that was signalled.
        :return ProcessCleanupUncertainError: The refusal to raise.
        """
        return ProcessCleanupUncertainError(
            f"Refusing to clear cleanup uncertainty for trial {record.trial_number} in "
            f"study {self.study_name}: process cleanup could not be confirmed. "
            f"experiment={self.experiment_name} phase={record.phase_name} "
            f"trial_dir={record.trial_dir} pid={identity.pid} pgid={identity.pgid}. "
            f"Investigate (e.g. `ps -o pid,pgid,cmd -p {identity.pid}`), then "
            f"{RERUN_CLEANUP_RECOVERY}."
        )

    def cleaned(self, record: _AttemptRecord, identity: StaleProcessIdentity) -> None:
        """Log the confirmed cleanup of a cleanup-uncertain terminal trial.

        :param _AttemptRecord record: Attempt that was resolved.
        :param StaleProcessIdentity identity: The identity that was cleaned.
        """
        log.warning(
            "Confirmed cleanup for terminal cleanup-uncertain trial %d in study %s "
            "(pid=%s pgid=%s)",
            record.trial_number,
            self.study_name,
            identity.pid,
            identity.pgid,
        )


def _resolve_attempt(record: _AttemptRecord, discovery: _Discovery, *, confirm: bool) -> None:
    """Prove that no live process can remain from one discovered attempt.

    Every discovered attempt resolves here, and this is the only place
    PhaseSweep signals a stale process group. Resolution order (review
    v0.5.17 / blocker 2):

    1. A durable ``exited`` lifecycle with confirmed cleanup means the
       supervised group was already proven gone by the launching orchestrator:
       the crash landed between process exit and the Optuna terminal commit.
       Nothing is signalled.
    2. A durable ``allocated`` lifecycle with no process identity means no
       launch was ever attempted (the worker died queued for a GPU). Nothing
       is signalled.
    3. Anything else means a process was or may have been launched. Its
       durable identity must validate, and a confirmed resolution cleans its
       group the fail-closed way.

    A source whose lifecycle cannot settle its attempts goes straight to step 3.

    :param _AttemptRecord record: Normalized attempt to resolve.
    :param _Discovery discovery: The discovering source's wording.
    :param bool confirm: Signal and clean a validated identity's group when
        true. When false, stop after validating it, as ``mcp recover-run``
        preflight does; a later confirmed call resolves the attempt again.
    :raises ProcessCleanupUncertainError: The lifecycle record or process
        identity is malformed or belongs to another attempt, or cleanup could
        not prove the process group gone.
    """
    if discovery.lifecycle_settles:
        try:
            lifecycle = read_attempt_lifecycle(
                record.trial_dir, expected_attempt_id=record.attempt_id
            )
        except ValueError as exc:
            raise discovery.lifecycle_malformed(record) from exc
        # A missing identity is exactly what 'allocated' predicts: the worker
        # died queued (e.g. waiting for a GPU) before launch began. The launcher
        # durably advances to 'launching' before Popen; that state and a
        # present-but-unreadable identity both fall through to the strict
        # identity read below and fail closed.
        resolution = _attempt_process_resolution(
            lifecycle,
            identity_exists=(record.trial_dir / PROCESS_IDENTITY_FILE).exists(),
        )
        if resolution != "identity":
            assert lifecycle is not None
            discovery.settled(record, lifecycle)
            return
    try:
        identity = read_stale_process_identity(
            record.trial_dir, expected_attempt_id=record.attempt_id
        )
    except (OSError, ValueError) as exc:
        raise discovery.identity_unreadable(record) from exc
    if not confirm:
        return
    if not cleanup_stale_trial_process(identity):
        raise discovery.cleanup_unconfirmed(record, identity)
    discovery.cleaned(record, identity)


def _fail_stale_running_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    failure_message: str,
) -> None:
    """Commit a stale RUNNING trial as ``FAIL`` once its attempt is resolved.

    The failure outcome and then the cleanup-recovery ledger are written
    before Optuna's terminal state retires the trial's RUNNING authority.

    :param optuna.Study study: Study containing ``trial``.
    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial whose attempt
        can no longer leak a process.
    :param str failure_message: The refusal to raise when Optuna cannot
        record ``FAIL``.
    :raises StudyStorageUnavailableError: The failure outcome, the recovery
        ledger, or the terminal state could not be recorded; the trial is left
        RUNNING rather than silently dropping the failure.
    """
    _record_stale_trial_failure(study, trial)
    _record_cleanup_recovery(study, trial)
    try:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    except Exception as exc:
        raise StudyStorageUnavailableError(failure_message) from exc


def _cleanup_recovered_trial_numbers(study: optuna.Study) -> set[int]:
    """Return trial numbers already consumed as cleanup recovery evidence.

    :param optuna.Study study: Study containing the cleanup recovery ledger.
    :return set[int]: Valid non-negative trial numbers recorded in the ledger.
    """
    raw = study.user_attrs.get(CLEANUP_RECOVERED_TRIALS_ATTR)
    if not isinstance(raw, list):
        return set()
    return {value for value in raw if type(value) is int and value >= 0}


def _record_cleanup_recovery(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Persist confirmed trial cleanup before retiring its other authorities.

    :param optuna.Study study: Study whose cleanup recovery ledger should be updated.
    :param optuna.trial.FrozenTrial trial: Trial whose cleanup evidence was consumed.
    :raises StudyStorageUnavailableError: The ledger could not be written; MCP cleanup
        uncertainty stays set rather than being cleared without consuming the
        trial evidence.
    """
    recovered = sorted(_cleanup_recovered_trial_numbers(study) | {trial.number})
    try:
        study.set_user_attr(CLEANUP_RECOVERED_TRIALS_ATTR, recovered)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Cleanup was confirmed for trial {trial.number} in study {study.study_name}, "
            "but the study-level cleanup recovery ledger could not be updated. "
            "Refusing to clear MCP cleanup uncertainty without consuming the trial evidence."
        ) from exc
