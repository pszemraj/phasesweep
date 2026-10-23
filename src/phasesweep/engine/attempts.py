"""Active-attempt registry and stale-process resolution."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.models import Experiment
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.ledger import open_registry_study
from phasesweep.engine.paths import _attempts_dir
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _outcome_sequence_counts,
    _parsed_trial_outcome,
    _stale_recovery_outcome,
    _trial_outcome_payload,
    _unduplicated_outcome,
)
from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError
from phasesweep.runtime.files import (
    UnsafePrivatePathError,
    canonical_storage_identity,
    open_directory_fd,
    private_atomic_write_text,
    read_private_text,
    read_private_text_at,
    storage_recovery_locator,
)
from phasesweep.runtime.json import JsonObject, strict_json_loads
from phasesweep.runtime.process import AttemptLifecycle, read_attempt_lifecycle
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    _process_group_alive,
    cleanup_stale_trial_process,
    is_same_live_process,
    read_boot_id,
    read_stale_process_identity,
    same_pid_namespace,
)

log = logging.getLogger(__name__)


@dataclass
class _PreflightCleanupReport:
    """Cleanup evidence accumulated while inspecting all existing phase studies."""

    cleanup_confirmed: bool = True
    recovered_attempt_ids: set[str] = field(default_factory=set)
    recovered_attempt_generations: dict[str, str] = field(default_factory=dict)
    reaped_running_trials: int = 0
    cleanup_recovered_trials: int = 0
    uncertain_attempt_ids: set[str] = field(default_factory=set)
    unreceipted_attempt_ids: set[str] = field(default_factory=set)
    """Recovered tracking operations whose cleanup receipt could not be written."""
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


def _trial_dir_from_attr(
    trial: optuna.trial.FrozenTrial,
    study_name: str,
    *,
    fallback: Callable[[], Path] | None = None,
) -> Path:
    """Read the persisted directory containing a trial's process identity.

    Persisted paths survive cwd and workdir changes. Only a RUNNING trial with
    no directory attribute may use the pre-launch fallback; invalid values and
    terminal recovery require the persisted path.

    :param optuna.trial.FrozenTrial trial: Trial whose process evidence is inspected.
    :param str study_name: Study name used in diagnostics.
    :param Callable[[], Path] | None fallback: Resolve the canonical pre-launch
        directory only when a RUNNING trial has no persisted directory attribute.
    :raises ProcessCleanupUncertainError: A persisted trial-directory value is
        invalid, so identity files cannot be located safely.
    :return Path: Persisted trial directory, or the canonical pre-launch directory.
    """
    if TRIAL_DIR_ATTR not in trial.user_attrs and fallback is not None:
        trial_dir = fallback()
        log.warning(
            "RUNNING trial %d in study %s is missing %r; falling back to "
            "canonical pre-launch trial_dir=%s.",
            trial.number,
            study_name,
            TRIAL_DIR_ATTR,
            trial_dir,
        )
        return trial_dir

    stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
    if not isinstance(stored, str) or not stored:
        if fallback is None:
            detail = (
                f"Refusing to recover cleanup-uncertain trial {trial.number} in study "
                f"{study_name}: missing or invalid {TRIAL_DIR_ATTR!r} user attribute "
                f"{stored!r}. The process group cannot be tied to its identity files."
            )
        else:
            detail = (
                f"Refusing to reap RUNNING trial {trial.number}: invalid persisted "
                f"{TRIAL_DIR_ATTR!r} user attribute {stored!r}. The trial cannot be "
                "tied to its identity files."
            )
        raise ProcessCleanupUncertainError(detail, action=OperatorAction.RESTORE_LEDGER)
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
    if record is None:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover trial {trial.number} in study {study_name}: missing or "
            f"invalid {ATTEMPT_ID_ATTR!r} user attribute. Process identity is unknown.",
            action=OperatorAction.RESTORE_LEDGER,
        )
    return record


ATTEMPT_REGISTRY_SCHEMA_VERSION = 4
_ATTEMPT_ENTRY_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
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
_TRACKING_RECEIPT_SCHEMA_VERSION = 1
_TRACKING_RECEIPT_FIELDS = frozenset(
    {"schema_version", "operation_id", "generation_id", "cleanup_confirmed"}
)


def _register_tracking_operation(
    experiment: Experiment, *, operation_id: str, generation_id: str, directory: Path
) -> None:
    """Register a supervised publisher with no study or Optuna trial.

    :param Experiment experiment: Local experiment namespace.
    :param str operation_id: Process identity.
    :param str generation_id: Producing generation.
    :param Path directory: Process lifecycle directory.
    """
    entry = {
        "schema_version": ATTEMPT_REGISTRY_SCHEMA_VERSION,
        "kind": "tracking",
        "attempt_id": operation_id,
        "experiment": experiment.experiment,
        "phase": "tracking",
        "trial_dir": str(directory),
        "generation_id": generation_id,
    }
    private_atomic_write_text(
        _attempts_dir(experiment) / f"{operation_id}.json", json.dumps(entry, sort_keys=True) + "\n"
    )


def _write_tracking_cleanup_receipt(
    experiment: Experiment, *, operation_id: str, generation_id: str
) -> None:
    """Durably record that a tracking operation's cleanup was confirmed.

    A tracking operation owns no Optuna trial, so its registry entry is the only
    record of the cleanup. The receipt outlives the entry and is filed under the
    originating generation, so ``recover-run`` can still attribute the cleanup to
    that generation after another pass retired the entry.

    :param Experiment experiment: Local experiment namespace.
    :param str operation_id: Tracking operation whose cleanup was confirmed.
    :param str generation_id: Generation that produced the operation.
    :raises OSError: The receipt could not be written.
    :raises UnsafePrivatePathError: The receipt destination is unsafe.
    """
    from phasesweep.tracking.session import tracking_root

    receipt = {
        "schema_version": _TRACKING_RECEIPT_SCHEMA_VERSION,
        "operation_id": operation_id,
        "generation_id": generation_id,
        "cleanup_confirmed": True,
    }
    private_atomic_write_text(
        tracking_root(experiment) / generation_id / "cleanup-receipts" / f"{operation_id}.json",
        json.dumps(receipt, sort_keys=True) + "\n",
        require_private_dir=False,
    )


def _tracking_cleanup_receipts(experiment: Experiment) -> dict[str, str]:
    """Read the tracking operations whose cleanup was confirmed, keyed to their generation.

    A receipt is proof only when it is an owner-only current-format record filed
    where it claims to belong. A receipt that is unreadable, malformed, or filed
    under another name is skipped with a warning, so damage can only withhold proof.

    :param Experiment experiment: Local experiment namespace.
    :return dict[str, str]: Operation ids mapped to their producing generation ids.
    """
    from phasesweep.tracking.session import tracking_root

    receipts: dict[str, str] = {}
    for path in sorted(tracking_root(experiment).glob("*/cleanup-receipts/*.json")):
        try:
            payload: object = strict_json_loads(read_private_text(path, require_private_dir=False))
        except (OSError, UnsafePrivatePathError, ValueError) as exc:
            log.warning("Ignoring unreadable tracking cleanup receipt %s: %s", path, exc)
            continue
        generation_id = path.parent.parent.name
        if not (
            isinstance(payload, dict)
            and set(payload) == _TRACKING_RECEIPT_FIELDS
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == _TRACKING_RECEIPT_SCHEMA_VERSION
            and payload["operation_id"] == path.stem
            and payload["generation_id"] == generation_id
            and payload["cleanup_confirmed"] is True
        ):
            log.warning("Ignoring malformed tracking cleanup receipt %s.", path)
            continue
        receipts[path.stem] = generation_id
    return receipts


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

    The producing phase, study and frozen storage locator support recovery
    after phase or storage configuration changes.

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
            "kind": "trial",
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
    except (OSError, UnsafePrivatePathError) as exc:
        raise ActiveAttemptPersistenceError(
            f"Could not register active attempt {attempt_id} for phase {phase_name!r} "
            f"at {entry_path}: {exc}. No trainer was started and no GPU lease was "
            "consumed. Persistent storage records a phase abort; retry requires "
            "n_trials above its accepted target in the config."
        ) from exc


def _retire_active_attempt(experiment: Experiment, attempt_id: str) -> None:
    """Best-effort removal of a registry entry whose trial is durably terminal.

    Deletion stays best-effort even though creation is fail-closed: a retained
    entry is safe by construction. It names an already-terminal trial, so
    preflight resolves it against a durable ``exited`` lifecycle, changes
    nothing in the study, and garbage-collects the file. Only a *missing*
    entry loses information.

    :param Experiment experiment: Experiment supplying the registry root.
    :param str attempt_id: Attempt whose trial reached a terminal state.
    """
    with (
        contextlib.suppress(OSError, ProcessCleanupUncertainError),
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
    except (OSError, UnsafePrivatePathError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry {attempts_dir} is not a real owner-only directory (mode 0700). "
            "Recovery cannot trust process authority reached through a symlink or "
            "shared path.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    try:
        try:
            names = sorted(name for name in os.listdir(directory_fd) if name.endswith(".json"))
        except OSError as exc:
            raise ProcessCleanupUncertainError(
                f"Attempt registry {attempts_dir} cannot be enumerated safely.",
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
    try:
        payload = strict_json_loads(read_private_text_at(directory_fd, entry_path.name, entry_path))
    except (OSError, UnsafePrivatePathError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} is unreadable or malformed. "
            "Recovery cannot prove whether a process from this attempt is still "
            "alive. Investigate the attempt's trial directory.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    if isinstance(payload, dict) and payload.get("kind") == "tracking":
        required = {
            "schema_version",
            "kind",
            "attempt_id",
            "experiment",
            "phase",
            "trial_dir",
            "generation_id",
        }
        if (
            payload.get("schema_version") == ATTEMPT_REGISTRY_SCHEMA_VERSION
            and set(payload) == required
            and all(
                isinstance(payload[k], str) and payload[k] for k in required - {"schema_version"}
            )
            # Both ids become path components of the operation's cleanup receipt.
            and SAFE_NAME_PATTERN.fullmatch(payload["attempt_id"])
            and SAFE_NAME_PATTERN.fullmatch(payload["generation_id"])
        ):
            return payload
        raise ProcessCleanupUncertainError(
            "Malformed tracking operation registry entry.", action=OperatorAction.RESTORE_TREE
        )
    locator_identity: str | None = None
    if isinstance(payload, dict) and isinstance(payload.get("storage_locator"), str):
        try:
            locator_identity = canonical_storage_identity(payload["storage_locator"])
        except ValueError as exc:
            raise ProcessCleanupUncertainError(
                f"Attempt registry entry {entry_path} has an unsupported or partial schema.",
                action=OperatorAction.RESTORE_TREE,
            ) from exc
    if (
        not isinstance(payload, dict)
        or not _ATTEMPT_ENTRY_REQUIRED_FIELDS.issubset(payload)
        or payload.get("kind") != "trial"
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
        or locator_identity != payload.get("storage_identity")
        or not isinstance(payload.get("generation_id"), str)
        or not payload.get("generation_id")
        or type(payload.get("trial_number")) is not int
    ):
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} has an unsupported or partial schema.",
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
    if not trial_dir.is_dir():
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} points at a missing trial "
            f"directory {trial_dir}; its process state cannot be verified.",
            action=OperatorAction.RESTORE_TREE,
        )
    return _AttemptRecord(
        attempt_id=entry["attempt_id"],
        trial_dir=trial_dir,
        generation_id=entry["generation_id"],
        phase_name=entry["phase"],
        trial_number=entry.get("trial_number"),
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


def _record_stale_trial_outcome(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Record a recovered outcome before marking a stale trial FAIL.

    Keep recorded failures, fatal errors, interruptions, and their aborts. An uncommitted
    success or missing outcome becomes interrupted and reruns. Recovery adds no abort.
    Malformed or duplicate outcomes receive a fresh sequence.

    :param optuna.Study study: Study providing outcome sequences.
    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial.
    :raises StudyStorageUnavailableError: Outcome persistence failed; leave the trial RUNNING.
    """
    trials = study.get_trials(deepcopy=False)
    sequence_counts = _outcome_sequence_counts(trials)
    current = next((candidate for candidate in trials if candidate.number == trial.number), trial)
    existing = _unduplicated_outcome(current, sequence_counts)
    outcome = _stale_recovery_outcome(existing)
    policy: str | None = None
    abort: JsonObject | None = None
    if existing is None:
        sequence = max(sequence_counts, default=0) + 1
        cause: str | None = "stale RUNNING trial recovered after its orchestrator stopped"
    elif existing.outcome == outcome:
        sequence = existing.sequence
        cause = existing.cause
        policy = existing.policy
        abort = existing.abort
    else:
        sequence = existing.sequence
        cause = "orchestrator stopped before Optuna committed the terminal trial state"

    payload = _trial_outcome_payload(sequence, outcome, cause=cause, policy=policy, abort=abort)
    try:
        active_trial = optuna.Trial(study, trial._trial_id)
        active_trial.set_user_attr(TRIAL_OUTCOME_ATTR, payload)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Process cleanup completed for stale RUNNING trial {trial.number} in study "
            f"{study.study_name!r}, but its durable {outcome} outcome could not be recorded. "
            "The trial remains RUNNING so a later retry records its outcome before anything "
            "counts it."
        ) from exc


def _registry_attempt_fail_stale_trial(
    entry: dict[str, Any],
    entry_path: Path,
    report: _PreflightCleanupReport,
) -> str:
    """Mark the entry's Optuna trial FAIL through its saved storage locator.

    Recovery remains independent of current storage and phase configuration.

    :param dict[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :param _PreflightCleanupReport report: Counts newly recorded trial cleanup actions.
    :return str: ``"reaped"`` when the stale RUNNING trial was marked FAIL,
        ``"recovered"`` when an interrupted pass already recorded that
        transition in the study ledger or the entry is a tracking operation
        (which owns no trial, so its resolved process is the recovery
        evidence), ``"terminal"`` when nothing needed to change (trial
        already terminal, study gone, or in-memory storage), or
        ``"unreachable"`` when the recorded storage could not be read and the
        entry must be retained for a later retry.
    :raises StudySchemaMismatchError: The registry and trial record
        conflicting generation identities, or the ledger the entry names
        holds unsupported PhaseSweep study state.
    :raises ProcessCleanupUncertainError: A terminal trial lacks the identity
        needed to attribute confirmed cleanup to the registry's attempt.
    :raises StudyStorageUnavailableError: The stale RUNNING trial could not be
        marked FAIL, or its durable recovery state could not be recorded; the
        study is left inconsistent rather than silently dropping the failure.
    """
    if entry["kind"] == "tracking":
        return "recovered"
    storage_url = entry["storage_locator"]
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
        raise
    except Exception:  # unreachable storage keeps the entry for retry
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
        # A missing id can follow registration before attrs; a conflicting id names another attempt.
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
                    "retained; cleanup recovery cannot be attributed to this trial.",
                    action=OperatorAction.RESTORE_LEDGER,
                )
            if not recovered:
                _record_cleanup_recovery(study, trial)
                report.cleanup_recovered_trials += 1
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
    report.reaped_running_trials += 1
    log.info(
        "Confirmed process cleanup for RUNNING trial %d in study %s via the attempt registry",
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
    tracking_only: bool = False,
) -> dict[str, str]:
    """Resolve registered attempts independently of the current phase graph and storage URL.

    :param Experiment experiment: Experiment whose registry is scanned.
    :param _PreflightCleanupReport report: Shared cleanup-evidence collector.
    :param bool confirm: Clean processes, repair studies, and retire resolved entries.
        False inspects evidence without signals or writes. Tracking entries require a
        durable cleanup receipt before retirement.
    :param bool retain_recovery_evidence: Keep uncommitted cleanup evidence for operator retries.
    :param bool tracking_only: Resolve only controller operations without Optuna trials.
    :return dict[str, str]: Attempt ids mapped to producing generation ids.
    :raises ProcessCleanupUncertainError: A registered attempt could not be proven safe.
    """
    inspected: dict[str, str] = {}
    with _open_attempt_registry(experiment) as (directory_fd, entry_paths):
        if directory_fd is None:
            return inspected
        for entry_path in entry_paths:
            entry = _load_attempt_entry(entry_path, directory_fd=directory_fd)
            if tracking_only and entry["kind"] != "tracking":
                continue
            attempt_id = entry["attempt_id"]
            inspected[attempt_id] = entry["generation_id"]
            try:
                _resolve_attempt(
                    _registry_attempt_record(entry, entry_path),
                    context=f"Attempt registry entry {entry_path}",
                    evidence_action=OperatorAction.RESTORE_TREE,
                    lifecycle_settles=True,
                    confirm=confirm,
                )
                if not confirm:
                    continue
                outcome = _registry_attempt_fail_stale_trial(entry, entry_path, report)
            except ProcessCleanupUncertainError as exc:
                report.uncertain_attempt_ids.add(attempt_id)
                report.mark_uncertain(exc)
                raise
            if outcome in {"reaped", "recovered"}:
                report.recovered_attempt_ids.add(attempt_id)
                report.recovered_attempt_generations[attempt_id] = entry["generation_id"]
            if entry["kind"] == "tracking" and outcome == "recovered":
                # The entry is the operation's only cleanup evidence, so no pass
                # may retire it before the originating generation holds a receipt.
                try:
                    _write_tracking_cleanup_receipt(
                        experiment, operation_id=attempt_id, generation_id=entry["generation_id"]
                    )
                except (OSError, UnsafePrivatePathError) as exc:
                    report.unreceipted_attempt_ids.add(attempt_id)
                    log.warning(
                        "Could not record the cleanup receipt for tracking operation %s: %s. "
                        "Its registry entry is retained for a later pass.",
                        attempt_id,
                        exc,
                    )
                else:
                    report.unreceipted_attempt_ids.discard(attempt_id)
            if (
                outcome != "unreachable"
                and attempt_id not in report.unreceipted_attempt_ids
                and not (retain_recovery_evidence and outcome in {"reaped", "recovered"})
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
    trial_number: int | None


def _resolve_attempt(
    record: _AttemptRecord,
    *,
    context: str,
    evidence_action: OperatorAction,
    lifecycle_settles: bool,
    confirm: bool,
) -> None:
    """Inspect or clean one attempt using its durable lifecycle and process identity.

    Confirmed exited attempts and allocated attempts without identities need no signal
    when ``lifecycle_settles`` is true. Terminal cleanup uncertainty requires a valid
    identity regardless of lifecycle. Confirmed resolution cleans the identified group.

    :param _AttemptRecord record: Attempt identity and evidence directory to resolve.
    :param str context: Discovery source, including its registry path or study and trial.
    :param OperatorAction evidence_action: Remedy for invalid lifecycle or identity evidence.
    :param bool lifecycle_settles: Whether lifecycle evidence can prove cleanup.
    :param bool confirm: Clean the process group; False validates evidence without signalling.
    :raises ProcessCleanupUncertainError: Invalid or mismatched lifecycle/identity,
        or unconfirmed process cleanup.
    """
    context = (
        f"{context}: attempt={record.attempt_id} phase={record.phase_name!r} "
        f"trial_dir={record.trial_dir}"
    )
    if lifecycle_settles:
        try:
            lifecycle = read_attempt_lifecycle(
                record.trial_dir, expected_attempt_id=record.attempt_id
            )
        except ValueError as exc:
            raise ProcessCleanupUncertainError(
                f"{context}: attempt lifecycle record is malformed or belongs to another attempt.",
                action=evidence_action,
            ) from exc
        # Only allocated with an absent identity settles without a strict identity read.
        resolution = _attempt_process_resolution(
            lifecycle,
            identity_exists=(record.trial_dir / PROCESS_IDENTITY_FILE).exists(),
        )
        if resolution != "identity":
            assert lifecycle is not None
            log.warning(
                "%s: lifecycle=%s rc=%s confirms cleanup without signalling.",
                context,
                lifecycle.state,
                lifecycle.return_code,
            )
            return
    try:
        identity = read_stale_process_identity(
            record.trial_dir, expected_attempt_id=record.attempt_id
        )
    except (OSError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"{context}: durable process identity is missing, malformed, partial, "
            "or belongs to another attempt.",
            action=evidence_action,
        ) from exc
    if not confirm:
        action = "--confirm must verify process-birth identity before signalling."
        if (
            identity.boot_id is not None
            and identity.boot_id == read_boot_id()
            and same_pid_namespace(identity.pid_namespace)
        ):
            state = "live" if _process_group_alive(identity.pgid) else "exited"
            if state == "live":
                if identity.proc_starttime is not None and is_same_live_process(
                    identity.pid, identity.proc_starttime
                ):
                    action = "--confirm will terminate it if still alive before reconciling this attempt."
                else:
                    state = "live, but its process-birth identity is unverified or mismatched"
        else:
            state = "not observable from this boot/PID namespace"
        log.warning(
            "%s: training process group is %s (pid=%s pgid=%s); %s",
            context,
            state,
            identity.pid,
            identity.pgid,
            action,
        )
        return
    if not cleanup_stale_trial_process(identity):
        raise ProcessCleanupUncertainError(
            f"{context}: process cleanup could not be confirmed. "
            f"pid={identity.pid} pgid={identity.pgid}. "
            "Inspect the recorded PID/PGID with ps and check GPU processes before retrying recovery."
        )
    log.warning(
        "%s: confirmed process cleanup (pid=%s pgid=%s).", context, identity.pid, identity.pgid
    )


def _fail_stale_running_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    failure_message: str,
) -> None:
    """Commit a stale RUNNING trial as ``FAIL`` once its attempt is resolved.

    The recovered outcome and then the cleanup-recovery ledger are written
    before Optuna's terminal state retires the trial's RUNNING authority.

    :param optuna.Study study: Study containing ``trial``.
    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial whose attempt
        can no longer leak a process.
    :param str failure_message: The refusal to raise when Optuna cannot
        record ``FAIL``.
    :raises StudyStorageUnavailableError: The outcome, the recovery ledger, or
        the terminal state could not be recorded; the trial is left RUNNING
        rather than committed without its outcome.
    """
    _record_stale_trial_outcome(study, trial)
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
