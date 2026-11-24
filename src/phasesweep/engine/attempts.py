"""Active-attempt registry and stale-process resolution."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrackingTargetUnavailableError,
    TrialEvidenceMissingError,
    tracker_unreadable,
)
from phasesweep.engine.ledger import open_registry_study
from phasesweep.engine.optuna import _finished_trial_count
from phasesweep.engine.paths import _attempt_dir, _attempts_dir, _experiment_dir, _trial_dir_opener
from phasesweep.engine.progress import _log_recovered_trial, _warn_deferred_extraction
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    GENERATION_ID_ATTR,
    OVERRIDES_ATTR,
    RETURN_CODE_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _completed_trainer_duration,
    _parsed_trial_outcome,
    _required_trial_dir,
    _trial_outcome_payload,
    _unique_trial_outcomes,
)
from phasesweep.errors import (
    FRESH_STUDY,
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    format_mismatch,
    operator_step,
)
from phasesweep.runtime.files import (
    UnsafePrivatePathError,
    canonical_storage_identity,
    open_directory_fd,
    private_atomic_write_text,
    read_private_text,
    read_private_text_at,
    storage_recovery_locator,
)
from phasesweep.runtime.json import finite_nonnegative, json_text, strict_json_loads
from phasesweep.runtime.process import (
    AttemptLifecycle,
    _record_attempt_exited,
    read_attempt_lifecycle,
)
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    cleanup_stale_trial_process,
    group_members_command,
    read_stale_process_identity,
    recorded_process_check,
    stale_trial_process_decision,
)
from phasesweep.runtime.shutdown import PhaseSweepShutdown, servicing_shutdown_signal

if TYPE_CHECKING:
    from phasesweep.engine.trial import TrialResult
    from phasesweep.evidence.baseline import EvidenceBaseline

log = logging.getLogger(__name__)


@dataclass
class _PreflightCleanupReport:
    """Cleanup evidence accumulated while inspecting all existing phase studies."""

    cleanup_confirmed: bool = True
    recovered_attempt_ids: set[str] = field(default_factory=set)
    recovered_attempt_generations: dict[str, str] = field(default_factory=dict)
    reaped_running_trials: int = 0
    cleanup_recovered_trials: int = 0
    retained_results: int = 0
    uncertain_attempt_ids: set[str] = field(default_factory=set)
    unreceipted_attempt_ids: set[str] = field(default_factory=set)
    """Recovered tracking operations whose cleanup receipt could not be written."""
    unreadable_evidence: dict[str, str] = field(default_factory=dict)
    """Completed attempts left RUNNING because their tracker gave no answer, each with its
    failure."""
    unreadable_tracker: str | None = None
    """First reason a tracker could not be read in this pass, from an extraction or the run
    that just stopped on it. Once set, later extractions in the pass skip their read, so an
    outage costs one read."""
    error: BaseException | None = None
    process_actions: dict[_AttemptRecord, bool] = field(default_factory=dict)
    """Inspected attempts mapped to whether their live group needs termination."""
    inspection_messages: dict[_AttemptRecord, str] | None = None
    """Deferred preview messages, replaced when a study requires stricter inspection."""

    def mark_uncertain(self, error: BaseException) -> None:
        """Record the first cleanup uncertainty and fail the aggregate closed."""
        self.cleanup_confirmed = False
        if self.error is None:
            self.error = error

    def mark_uncertain_if_unconfirmed(self, error: BaseException) -> bool:
        """Mark the report uncertain when ``error`` left process cleanup unconfirmed.

        That is a :class:`ProcessCleanupUncertainError`, or a :class:`PhaseSweepShutdown` whose
        handler could not confirm that the groups it stopped are gone.

        :param BaseException error: The error that ended or interrupted the work.
        :return bool: Whether ``error`` left cleanup unconfirmed, so the report was marked.
        """
        unconfirmed = isinstance(error, ProcessCleanupUncertainError) or (
            isinstance(error, PhaseSweepShutdown) and not error.report.cleanup_confirmed
        )
        if unconfirmed:
            self.mark_uncertain(error)
        return unconfirmed

    def record_generations(self, attempts: Mapping[str, _AttemptRecord]) -> None:
        """Record the producing generation of each attempt that has one.

        :param Mapping[str, _AttemptRecord] attempts: Attempts whose trials this pass made
            terminal or confirmed clean, keyed by id.
        """
        self.recovered_attempt_generations.update(
            {
                attempt_id: record.generation_id
                for attempt_id, record in attempts.items()
                if record.generation_id is not None
            }
        )


def _pending_evidence_refusal(pending: Mapping[str, str]) -> TrackingTargetUnavailableError:
    """Refuse new work while completed training waits for its tracker to answer.

    :param Mapping[str, str] pending: Pending attempts mapped to their tracker failure.
    :return TrackingTargetUnavailableError: Refusal naming the count and the first failure.
    """
    first = next(iter(pending.values())).rstrip(".")
    return tracker_unreadable(
        f"{len(pending)} completed trial(s) finished training, but the tracker could not be "
        f"read: {first}."
    )


def _trial_dir_from_attr(trial: optuna.trial.FrozenTrial, what: str) -> Path:
    """Read the persisted trial directory, beside which the attempt's records live.

    Persisted paths survive cwd and workdir changes.

    :param optuna.trial.FrozenTrial trial: Trial whose process evidence is inspected.
    :param str what: The recovery step and its trial, used in diagnostics.
    :raises ProcessCleanupUncertainError: The persisted trial-directory value is
        missing or invalid, so identity files cannot be located safely.
    :return Path: Persisted trial directory.
    """
    try:
        return _required_trial_dir(trial)
    except ValueError as exc:
        raise _unusable_attribute_error(
            what, TRIAL_DIR_ATTR, trial.user_attrs.get(TRIAL_DIR_ATTR)
        ) from exc


def _active_attempt_persistence_error(
    action: str, subject: str, path: Path, exc: Exception
) -> ActiveAttemptPersistenceError:
    """Refuse a launch whose attempt record could not be written.

    :param str action: What could not be done, such as ``register``.
    :param str subject: The record and its trial or phase.
    :param Path path: Where the write failed.
    :param Exception exc: The write failure, whose text names the OS error.
    :return ActiveAttemptPersistenceError: Error raised before any trainer starts.
    """
    return ActiveAttemptPersistenceError(
        f"Could not {action} {subject} at {path}: {exc}. No trainer was started."
    )


def _unusable_attribute_error(
    what: str, attribute: str, stored: object | None = None
) -> ProcessCleanupUncertainError:
    """Refuse recovery of a trial whose durable user attribute cannot locate its process.

    :param str what: The refused operation and its trial.
    :param str attribute: Name of the missing or invalid user attribute.
    :param object | None stored: The stored value, shown when there is one, defaults to None.
    :return ProcessCleanupUncertainError: Error routed to restoring the ledger.
    """
    shown = "" if stored is None else f" ({stored!r})"
    return ProcessCleanupUncertainError(
        f"Cannot {what}: its {attribute!r} user attribute is missing or invalid{shown}.",
        action=OperatorAction.RESTORE_LEDGER,
    )


def _trial_attempt_record(
    trial: optuna.trial.FrozenTrial,
    phase_name: str,
    trial_dir: Path,
) -> _AttemptRecord | None:
    """Build the attempt record a trial's durable attributes name.

    :param optuna.trial.FrozenTrial trial: Trial whose attempt is recorded.
    :param str phase_name: Phase that owns ``trial``.
    :param Path trial_dir: The trial's directory, as launch built or persisted it.
    :return _AttemptRecord | None: The record, or ``None`` when the trial has
        no valid persisted attempt id.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    return _AttemptRecord(
        attempt_id=attempt_id,
        attempt_dir=_attempt_dir(trial_dir.parent, attempt_id),
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
        raise _unusable_attribute_error(
            f"recover trial {trial.number} in study {study_name}", ATTEMPT_ID_ATTR
        )
    return record


ATTEMPT_REGISTRY_SCHEMA_VERSION = 5
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
        "attempt_dir",
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
        "attempt_dir": str(directory),
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
    attempt_dir: Path,
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
    :param Path trial_dir: The trainer's directory.
    :param Path attempt_dir: Attempt directory holding lifecycle and process identity.
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
            "attempt_dir": str(attempt_dir),
            "generation_id": generation_id,
        }
        private_atomic_write_text(entry_path, json.dumps(entry, sort_keys=True) + "\n")
    except (OSError, UnsafePrivatePathError) as exc:
        raise _active_attempt_persistence_error(
            "register", f"attempt {attempt_id} for phase {phase_name!r}", entry_path, exc
        ) from exc


def _retire_active_attempt(experiment: Experiment, attempt_id: str) -> None:
    """Best-effort removal of a registry entry whose trial is durably terminal.

    A failed deletion leaves the entry for the next preflight, which resolves
    its terminal trial's ``exited`` lifecycle without changing the study.

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
            f"Attempt registry {attempts_dir} is not a real owner-only directory (mode 0700): "
            f"{exc}",
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


_PARTIAL_SCHEMA = "has an unsupported or partial schema"


def _unusable_entry_error(
    entry_path: Path, problem: str, *, kind: str = "Attempt"
) -> ProcessCleanupUncertainError:
    """Refuse recovery over a registry entry it cannot use, with the check that clears it.

    :param Path entry_path: Registry entry that recovery cannot use.
    :param str problem: What is wrong with the entry.
    :param str kind: ``Attempt`` or ``Tracking operation``, defaults to ``Attempt``.
    :return ProcessCleanupUncertainError: Error routed to restoring the tree, whose fix line
        is the process check and the deletion.
    """
    return ProcessCleanupUncertainError(
        f"{kind} registry entry {entry_path} {problem}.",
        action=OperatorAction.RESTORE_TREE,
        remediation=_entry_removal_step(entry_path),
    )


def _entry_removal_step(entry_path: Path) -> str:
    """Name the check that makes deleting an attempt registry entry safe, and the deletion.

    :param Path entry_path: Registry entry that recovery cannot use.
    :return str: The process check for the attempt the entry names, then the deletion that
        lets the next run recover the attempt from its trial directory.
    """
    return recorded_process_check(
        f"the {PROCESS_IDENTITY_FILE} in the directory under {entry_path.parent.parent} "
        f"whose name ends in {entry_path.stem}"
    )


def _load_attempt_entry(entry_path: Path, *, directory_fd: int) -> dict[str, Any]:
    """Load and validate one attempt registry entry.

    :param Path entry_path: Registry entry file to parse.
    :param int directory_fd: Open descriptor for the entry's validated parent directory.
    :return dict[str, Any]: The validated entry payload.
    :raises ProcessCleanupUncertainError: The entry is unreadable, malformed or in another
        format — recovery cannot know whether a process from it is still alive.
    """
    try:
        payload = strict_json_loads(read_private_text_at(directory_fd, entry_path.name, entry_path))
    except (OSError, UnsafePrivatePathError, ValueError) as exc:
        raise _unusable_entry_error(entry_path, f"is unreadable or malformed: {exc}") from exc
    if (
        isinstance(payload, dict)
        and "schema_version" in payload
        and payload["schema_version"] != ATTEMPT_REGISTRY_SCHEMA_VERSION
    ):
        found = format_mismatch(payload["schema_version"], ATTEMPT_REGISTRY_SCHEMA_VERSION)
        raise _unusable_entry_error(entry_path, found)
    if isinstance(payload, dict) and payload.get("kind") == "tracking":
        required = {
            "schema_version",
            "kind",
            "attempt_id",
            "experiment",
            "phase",
            "attempt_dir",
            "generation_id",
        }
        if (
            set(payload) == required
            and all(
                isinstance(payload[k], str) and payload[k] for k in required - {"schema_version"}
            )
            # Both ids become path components of the operation's cleanup receipt.
            and SAFE_NAME_PATTERN.fullmatch(json_text(payload, "attempt_id"))
            and SAFE_NAME_PATTERN.fullmatch(json_text(payload, "generation_id"))
        ):
            return payload
        raise _unusable_entry_error(entry_path, "is malformed", kind="Tracking operation")
    locator_identity: str | None = None
    if isinstance(payload, dict) and isinstance(payload.get("storage_locator"), str):
        try:
            locator_identity = canonical_storage_identity(json_text(payload, "storage_locator"))
        except ValueError as exc:
            raise _unusable_entry_error(entry_path, _PARTIAL_SCHEMA) from exc
    if (
        not isinstance(payload, dict)
        or not _ATTEMPT_ENTRY_REQUIRED_FIELDS.issubset(payload)
        or payload.get("kind") != "trial"
        or not isinstance(payload.get("attempt_id"), str)
        or not isinstance(payload.get("trial_dir"), str)
        or not isinstance(payload.get("attempt_dir"), str)
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
        raise _unusable_entry_error(entry_path, _PARTIAL_SCHEMA)
    return payload


def _attempt_process_resolution(
    lifecycle: AttemptLifecycle | None,
    *,
    identity_exists: bool,
) -> str:
    """Classify the fail-closed recovery authority for one interrupted attempt.

    ``launching`` with no identity file settles without signalling. The launch writes
    ``launching`` before it spawns the supervisor and sends the payload only after the
    identity file is written, so a missing file means no payload crossed. Recovery acts on
    this only under the experiment lock, which proves the controller holding the payload
    pipe is dead, and a supervisor that reads end-of-file instead of a payload exits without
    running the trainer. An identity file that exists at all, even partial, keeps the strict
    read: its payload may have crossed.

    :param AttemptLifecycle | None lifecycle: Validated durable lifecycle, if present.
    :param bool identity_exists: Whether a retained process-identity file exists.
    :return str: ``"exited"`` for confirmed cleanup, ``"allocated"`` when launch
        provably never began, ``"payload_withheld"`` when a launch began but no worker
        received its payload, or ``"identity"`` when the process identity must be
        validated (and, outside inspection mode, cleaned).
    """
    if lifecycle is not None and lifecycle.state == "exited" and lifecycle.cleanup_confirmed:
        return "exited"
    if not identity_exists and lifecycle is not None and lifecycle.state == "allocated":
        return "allocated"
    if not identity_exists and lifecycle is not None and lifecycle.state == "launching":
        return "payload_withheld"
    return "identity"


def _registry_attempt_record(entry: Mapping[str, Any], entry_path: Path) -> _AttemptRecord:
    """Normalize one validated registry entry into the attempt it registered.

    The entry alone identifies the attempt, so it resolves without the
    producing phase still existing in the config or the producing storage
    being reachable.

    :param Mapping[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :return _AttemptRecord: The registered attempt.
    :raises ProcessCleanupUncertainError: The entry's attempt directory is
        missing, so its process state cannot be verified.
    """
    attempt_dir = Path(entry["attempt_dir"])
    if not attempt_dir.is_dir():
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} points at a missing attempt "
            f"directory {attempt_dir}; its process state cannot be verified.",
            action=OperatorAction.RESTORE_TREE,
        )
    trial_dir = entry.get("trial_dir")
    return _AttemptRecord(
        attempt_id=entry["attempt_id"],
        attempt_dir=attempt_dir,
        trial_dir=Path(trial_dir) if isinstance(trial_dir, str) else None,
        generation_id=entry["generation_id"],
        phase_name=entry["phase"],
        trial_number=entry.get("trial_number"),
    )


def _trial_requires_cleanup_recovery(trial: optuna.trial.FrozenTrial) -> bool:
    """Return whether a terminal trial still lacks positive cleanup evidence.

    A false cleanup attribute or a fatal ``unsafe_process_cleanup`` outcome
    requires recovery. The outcome covers a failed best-effort attribute write.
    Confirmed recovery is recorded in the study-level recovery ledger.

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


def _record_stale_trial_outcome(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    outcome: str = "interrupted",
    *,
    metric: float | None = None,
    cause: str | None = None,
    policy: tuple[Phase, Experiment] | None = None,
) -> None:
    """Commit a newly recovered verdict in completion order, preserving any existing decision.

    :param optuna.Study study: Owning study.
    :param optuna.trial.FrozenTrial trial: Trial being settled.
    :param str outcome: Success, failure, or an interruption when no measurement exists.
    :param float | None metric: Measured objective of a success, also outside the bounds.
    :param str | None cause: Reason for a failure or interruption.
    :param tuple[Phase, Experiment] | None policy: Producing phase, whose failure limit applies to
        a new verdict, and its configuration, whose bounds judge a success for the streak.
    :raises StudySchemaMismatchError: The durable completion order is invalid.
    :raises StudyStorageUnavailableError: Recovery could not read or commit the decision.
    """
    from phasesweep.engine.phase import _abort_decision
    from phasesweep.engine.study_policy import (
        _counted_outcome,
        _load_phase_policy_state,
        _recorded_outcome_sequence,
    )

    try:
        # Validating the completion order first refuses a duplicate sequence even when this
        # trial already holds its verdict.
        max_sequence = _recorded_outcome_sequence(study)
        # The stored trial holds the constraint values the caller recorded after ``trial``.
        stored = study._storage.get_trial(trial._trial_id)
        if _parsed_trial_outcome(stored.user_attrs.get(TRIAL_OUTCOME_ATTR)) is not None:
            return
        abort = None
        if policy is not None:
            policy_phase, producing = policy
            state = _load_phase_policy_state(study, producing)
            abort = _abort_decision(
                policy_phase,
                trial.number,
                _counted_outcome(outcome, metric, stored, producing),
                completion_sequence=max_sequence,
                consecutive_failures=state.consecutive_failures,
                abort_recorded=state.abort is not None,
            )
        if outcome == "interrupted" and cause is None:
            cause = "stale RUNNING trial recovered after its orchestrator stopped"
        payload = _trial_outcome_payload(
            max_sequence + 1, outcome, metric=metric, cause=cause, abort=abort
        )
        optuna.Trial(study, trial._trial_id).set_user_attr(TRIAL_OUTCOME_ATTR, payload)
    except StudySchemaMismatchError:
        raise
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Recovery of stale RUNNING trial {trial.number} in study {study.study_name!r} cannot advance "
            f"because its durable {outcome} outcome could not be recorded."
        ) from exc


def _record_recovery_interruption(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Preserve a trainer interruption before recovery can signal its process group.

    Trainer ledger attributes identify an already completed trainer whose process slot
    now belongs to an evidence worker. Terminating that worker keeps the trainer result.

    :param optuna.Study study: Owning study used to persist interruption intent.
    :param optuna.trial.FrozenTrial trial: Trial whose current process slot is being terminated.
    :raises StudyStorageUnavailableError: Interruption intent could not be made durable.
    """
    if (
        trial.state == optuna.trial.TrialState.RUNNING
        and _completed_trainer_duration(trial, None) is None
    ):
        _record_stale_trial_outcome(study, trial)


def _registry_attempt_fail_stale_trial(
    entry: dict[str, Any],
    entry_path: Path,
    report: _PreflightCleanupReport,
    experiment: Experiment,
) -> str:
    """Settle the entry's Optuna trial through its saved storage locator.

    Recovery remains independent of current storage and phase configuration.

    :param dict[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :param _PreflightCleanupReport report: Counts newly recorded trial cleanup actions.
    :param Experiment experiment: Current artifact namespace and polling settings.
    :return str: ``"reaped"`` when the stale RUNNING trial became terminal,
        ``"recovered"`` when an interrupted pass already recorded that
        transition in the study ledger or the entry is a tracking operation
        (which owns no trial, so its resolved process is the recovery
        evidence), ``"terminal"`` when nothing needed to change (trial
        already terminal, study gone, or in-memory storage), or
        ``"unreachable"`` when the recorded storage could not be read, or ``"pending"`` when shutdown or an unreadable tracker leaves completed training awaiting extraction; both retain the entry for retry.
    :raises StudySchemaMismatchError: The registry and trial record
        conflicting generation identities, or the ledger the entry names
        holds unsupported PhaseSweep study state.
    :raises ProcessCleanupUncertainError: A terminal trial lacks the identity
        needed to attribute confirmed cleanup to the registry's attempt.
    :raises StudyStorageUnavailableError: The trial's terminal result or durable recovery state could not be recorded; the attempt remains available for retry.
    """
    record = _registry_attempt_record(entry, entry_path)

    def resolve(
        study: optuna.Study | None = None,
        trial: optuna.trial.FrozenTrial | None = None,
        *,
        storage_unavailable: bool = False,
    ) -> None:
        """Clean the registered group after preserving its owning trainer outcome.

        :param optuna.Study | None study: Reachable owning study, when one exists.
        :param optuna.trial.FrozenTrial | None trial: Matching owning trial, when present.
        :param bool storage_unavailable: Whether durable interruption intent cannot be recorded.
        """

        def require_owning_ledger() -> None:
            """Keep a live attempt untouched until its owning ledger can be read."""
            if storage_unavailable:
                raise StudyStorageUnavailableError(
                    f"Cannot request cleanup for live attempt {record.attempt_id}: its study "
                    "cannot be read. Nothing was signalled; retry when its ledger is available."
                )

        def before_terminate() -> None:
            """Preserve interruption before directly signalling an unguarded trainer."""
            if study is not None and trial is not None:
                _record_recovery_interruption(study, trial)

        report.process_actions[record] = _resolve_attempt(
            record,
            context=f"Attempt registry entry {entry_path}",
            evidence_action=OperatorAction.RESTORE_TREE,
            lifecycle_settles=True,
            confirm=True,
            before_signal=require_owning_ledger,
            before_terminate=before_terminate,
        )

    if entry["kind"] == "tracking":
        resolve()
        return "recovered"
    storage_url = entry["storage_locator"]
    if storage_url is None:
        # In-memory storage died with its orchestrator, so no trial is left to update. The
        # attempt's cleanup is confirmed here, which is what lets its managed run be ended.
        resolve()
        return "terminal"
    try:
        study = open_registry_study(storage_url, entry["study_name"])
        trials = [] if study is None else study.get_trials(deepcopy=False)
    except StudySchemaMismatchError:
        raise
    except Exception:  # unreachable storage keeps the entry for retry
        resolve(storage_unavailable=True)
        log.warning(
            "Attempt registry entry %s references storage that cannot be "
            "read right now; the stale trial will be retried on a later run.",
            entry_path,
        )
        return "unreachable"
    if study is None:
        # The ledger confirmed the study is gone without being recreated.
        resolve()
        return "terminal"
    trial = next((t for t in trials if t.number == entry["trial_number"]), None)
    if trial is None:
        resolve()
        return "terminal"
    stored_attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if stored_attempt_id is not None and stored_attempt_id != entry["attempt_id"]:
        # A missing id can follow registration before attrs; a conflicting id names another attempt.
        resolve()
        return "terminal"
    stored_generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    if stored_generation_id is not None and stored_generation_id != entry["generation_id"]:
        raise StudySchemaMismatchError(
            f"Attempt registry entry {entry_path} identifies generation "
            f"{entry['generation_id']!r}, but its trial {trial.number} in study "
            f"{entry['study_name']!r} records {stored_generation_id!r}."
        )
    resolve(study, trial)
    if trial.state != optuna.trial.TrialState.RUNNING:
        recovered = trial.number in _cleanup_recovered_trial_numbers(study)
        if trial.state.is_finished() and (recovered or _trial_requires_cleanup_recovery(trial)):
            if stored_attempt_id is None or stored_generation_id is None:
                raise ProcessCleanupUncertainError(
                    f"Attempt registry entry {entry_path} cannot be matched to terminal "
                    f"trial {trial.number} in study {entry['study_name']!r}: its attempt or "
                    "generation identity is missing.",
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
                f"Process cleanup completed for attempt {entry['attempt_id']}, but its trial "
                f"identity could not be restored in study {entry['study_name']!r}, so the "
                f"trial stays RUNNING: {exc}"
            ) from exc
    settled = _fail_stale_running_trial(
        study,
        trial,
        record=_registry_attempt_record(entry, entry_path),
        experiment=experiment,
        report=report,
        where=f"attempt {entry['attempt_id']}",
    )
    if not settled:
        return "pending"
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
    generation_id: str | None = None,
) -> dict[str, str]:
    """Resolve registered attempts independently of the current phase graph and storage URL.

    :param Experiment experiment: Experiment whose registry is scanned.
    :param _PreflightCleanupReport report: Shared cleanup-evidence collector.
    :param bool confirm: Clean processes, repair studies, and retire resolved entries.
        False inspects evidence without signals or writes. Tracking entries require a
        durable cleanup receipt before retirement.
    :param bool retain_recovery_evidence: Keep uncommitted cleanup evidence for operator retries.
    :param bool tracking_only: Resolve only controller operations without Optuna trials.
    :param str | None generation_id: Limit recovery to entries registered under this generation, if given.
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
            if generation_id is not None and entry["generation_id"] != generation_id:
                continue
            attempt_id = entry["attempt_id"]
            inspected[attempt_id] = entry["generation_id"]
            try:
                if not confirm:
                    record = _registry_attempt_record(entry, entry_path)
                    report.process_actions[record] = _resolve_attempt(
                        record,
                        context=f"Attempt registry entry {entry_path}",
                        evidence_action=OperatorAction.RESTORE_TREE,
                        lifecycle_settles=True,
                        confirm=False,
                        inspection_messages=report.inspection_messages,
                    )
                    continue
                outcome = _registry_attempt_fail_stale_trial(entry, entry_path, report, experiment)
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
                outcome not in {"unreachable", "pending"}
                and attempt_id not in report.unreceipted_attempt_ids
                and not (retain_recovery_evidence and outcome in {"reaped", "recovered"})
            ):
                with contextlib.suppress(OSError):
                    os.unlink(entry_path.name, dir_fd=directory_fd)
    return inspected


@dataclass(frozen=True)
class _AttemptRecord:
    """One discovered attempt: its identity, its directories, and producing trial.

    Every source that discovers an attempt (a registry entry, a stale RUNNING
    trial, a terminal trial that recorded cleanup uncertainty) normalizes it
    into this record before :func:`_resolve_attempt` may inspect or signal
    anything. A status snapshot's trial reference never becomes one: without
    a durable attempt id and trial directory it is not a recovery identity.

    ``attempt_dir`` holds the lifecycle record and process identity. ``trial_dir`` is
    the trainer's directory, ``None`` for a tracking operation, which has none.
    ``generation_id`` is ``None`` when the trial records no valid generation.
    """

    attempt_id: str
    attempt_dir: Path
    trial_dir: Path | None
    generation_id: str | None
    phase_name: str
    trial_number: int | None


def _record_process_resolution(record: _AttemptRecord, lifecycle: AttemptLifecycle | None) -> str:
    """Classify one attempt's process slot from its lifecycle and its identity file.

    :param _AttemptRecord record: Attempt whose attempt directory holds the identity file.
    :param AttemptLifecycle | None lifecycle: Validated durable lifecycle, if present.
    :return str: The :func:`_attempt_process_resolution` verdict for the attempt.
    """
    return _attempt_process_resolution(
        lifecycle, identity_exists=(record.attempt_dir / PROCESS_IDENTITY_FILE).exists()
    )


def _resolve_attempt(
    record: _AttemptRecord,
    *,
    context: str,
    evidence_action: OperatorAction,
    lifecycle_settles: bool,
    confirm: bool,
    before_signal: Callable[[], None] | None = None,
    before_terminate: Callable[[], None] | None = None,
    inspection_messages: dict[_AttemptRecord, str] | None = None,
) -> bool:
    """Inspect or clean one attempt using its durable lifecycle and process identity.

    When ``lifecycle_settles`` is true, three slots need no signal: confirmed exited
    attempts, allocated attempts without an identity, and launching attempts without an
    identity, whose payload never crossed. Terminal cleanup uncertainty requires a valid
    identity regardless of lifecycle. Confirmed resolution cleans the identified group.

    :param _AttemptRecord record: Attempt identity and evidence directory to resolve.
    :param str context: Discovery source, including its registry path or study and trial.
    :param OperatorAction evidence_action: Remedy for invalid lifecycle or identity evidence.
    :param bool lifecycle_settles: Whether lifecycle evidence can prove cleanup.
    :param bool confirm: Clean the process group; False validates evidence without signalling.
    :param Callable[[], None] | None before_signal: Confirm authority before requesting any signal.
    :param Callable[[], None] | None before_terminate: Preserve the owning trainer outcome before directly terminating a live group.
    :param dict[_AttemptRecord, str] | None inspection_messages: Collect one preview message per attempt instead of logging immediately.
    :raises ProcessCleanupUncertainError: Invalid or mismatched lifecycle/identity,
        or unconfirmed process cleanup.
    :raises StudyStorageUnavailableError: Recovery intent could not be recorded durably before signalling.
    :return bool: Whether this pass would directly signal a group, or did so when confirmed.
        A request to its guardian is reported separately.
    """
    context = (
        f"{context}: attempt={record.attempt_id} phase={record.phase_name!r} "
        f"attempt_dir={record.attempt_dir}"
    )
    label = f"Attempt {record.attempt_id} phase={record.phase_name!r} trial={record.trial_number}"

    def report(message: str, *, level: int = logging.WARNING) -> None:
        """Defer a preview until all discovery sources have inspected the attempt.

        :param str message: Operator-facing process or lifecycle decision.
        :param int level: Level a real run logs it at; a settled attempt that needed no signal is
            routine, so only a decision about a live process group warns.
        """
        if inspection_messages is None:
            log.log(level, message)
        else:
            inspection_messages[record] = message

    def read_lifecycle() -> AttemptLifecycle | None:
        """Read this attempt's current lifecycle under the discovery source's error policy."""
        try:
            return read_attempt_lifecycle(record.attempt_dir, expected_attempt_id=record.attempt_id)
        except ValueError as exc:
            raise ProcessCleanupUncertainError(
                f"{context}: attempt lifecycle record is malformed or belongs to another attempt.",
                action=evidence_action,
            ) from exc

    if lifecycle_settles:
        lifecycle = read_lifecycle()
        resolution = _record_process_resolution(record, lifecycle)
        if resolution == "payload_withheld":
            report(
                f"{label}: lifecycle=launching with no process identity: no worker received "
                "its payload; settles without signalling.",
                level=logging.INFO,
            )
            return False
        if resolution != "identity":
            assert lifecycle is not None
            report(
                f"{label}: lifecycle={lifecycle.state} ended_by={lifecycle.ended_by} "
                f"rc={lifecycle.return_code} confirms cleanup without signalling.",
                level=logging.INFO,
            )
            return False
    try:
        identity = read_stale_process_identity(
            record.attempt_dir, expected_attempt_id=record.attempt_id
        )
    except (OSError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"{context}: durable process identity is missing, malformed, partial, "
            "or belongs to another attempt.",
            action=evidence_action,
        ) from exc
    if not confirm:
        decision = stale_trial_process_decision(identity)
        action = {
            "terminate": "--confirm will terminate it if still alive before reconciling this attempt",
            "guardian": "--confirm will ask the guardian to settle before reconciling this attempt",
            "gone": "--confirm will reconcile this attempt without signalling",
            "refuse": "--confirm will refuse cleanup and leave this attempt unresolved",
        }[decision.action]
        if (
            decision.action in {"terminate", "guardian"}
            and lifecycle_settles
            and record.trial_number is not None
        ):
            action += (
                ". With persistent storage, wait for the trainer to exit successfully "
                "to keep its result"
            )
        report(
            f"{label}: training process group: {decision.reason} "
            f"(pid={identity.pid} pgid={identity.pgid}); {action}."
        )
        return decision.action == "terminate"
    terminated = False
    guardian_settled = False

    def confirm_guardian_exit() -> bool:
        """Read the verified guardian's final verdict before claiming cleanup."""
        nonlocal guardian_settled
        lifecycle = read_lifecycle()
        guardian_settled = (
            lifecycle is not None
            and lifecycle.state == "exited"
            and lifecycle.cleanup_confirmed is True
            and lifecycle.duration_seconds is not None
        )
        return guardian_settled

    def record_termination() -> None:
        """Preserve interruption intent before marking the group for termination."""
        nonlocal terminated
        if before_terminate is not None:
            before_terminate()
        terminated = True

    if not cleanup_stale_trial_process(
        identity,
        before_signal=before_signal,
        before_terminate=record_termination,
        guardian_completed=confirm_guardian_exit,
    ):
        raise ProcessCleanupUncertainError(
            f"{context}: process cleanup could not be confirmed (pid={identity.pid} "
            f"pgid={identity.pgid} guardian={identity.guardian_pid}). List the group's members with "
            f"{group_members_command(str(identity.pgid))}, then retry recovery."
        )
    if guardian_settled:
        log.info("%s: lease guardian confirmed descendant cleanup.", label)
        return False
    # Recovery cleaned the group itself, with no guardian verdict to keep: it stopped the
    # attempt, or found it already gone without knowing how it ended. Either way it reruns.
    _record_attempt_exited(record.attempt_dir, attempt_id=record.attempt_id, ended_by="cancelled")
    # Signalling a live group is worth a warning; a group that was already gone is routine.
    log.log(
        logging.WARNING if terminated else logging.INFO,
        "%s: %s (pid=%s pgid=%s).",
        label,
        "terminated training process group; confirmed process cleanup"
        if terminated
        else "confirmed process cleanup without signalling",
        identity.pid,
        identity.pgid,
    )
    return terminated


def read_finished_attempt(
    trials: Mapping[int, optuna.trial.FrozenTrial],
    trial: optuna.trial.FrozenTrial,
    record: _AttemptRecord,
    *,
    producing: Experiment,
    reading: Experiment,
    current: Experiment,
    trainer_duration: float,
    baseline: EvidenceBaseline,
    evidence_before: float | str | None = None,
) -> tuple[TrialResult, str]:
    """Read a finished attempt's kept evidence after the fact, through the live extraction path.

    The one way PhaseSweep reads a trainer that finished while nothing was reading it: recovery
    of a finished orphan reads it under its producing configuration, and a study filled under
    changed reading settings (:mod:`phasesweep.engine.reread`) reads it under the current one.
    What the trainer ran and where it logged (run name, tracker target, environment) always
    come from ``producing``; the extractors, constraints and gates come from ``reading``; the
    attempt's own evidence baseline says which value files are its own.

    :param Mapping[int, optuna.trial.FrozenTrial] trials: The trial's study's trials by number,
        for the earlier attempts a rerun's directory holds.
    :param optuna.trial.FrozenTrial trial: The trial whose attempt finished.
    :param _AttemptRecord record: Its attempt, with a trial directory and generation.
    :param Experiment producing: The configuration that launched the attempt.
    :param Experiment reading: The configuration whose reading settings apply.
    :param Experiment current: The invocation's configuration, for its MLflow polling controls.
    :param float trainer_duration: The trainer's recorded wall-clock seconds.
    :param EvidenceBaseline baseline: The attempt's evidence baseline.
    :param float | str | None evidence_before: Unix time shared evidence must predate, or the
        reason it cannot be attributed to the attempt, defaults to None.
    :raises ProcessCleanupUncertainError: The attempt records an unusable trial directory, so
        the run its trial logs to cannot be named.
    :raises StudyStorageUnavailableError: The trial lacks its recorded trainer input or overrides.
    :return tuple[TrialResult, str]: The extraction result and the trial's recorded overrides.
    """
    from phasesweep.engine.evidence import _earlier_attempts
    from phasesweep.engine.trial import (
        ExecutedTrial,
        _environment_identity,
        extract_trial_result,
        trial_run_name,
    )
    from phasesweep.evidence.evaluation import TrialContext
    from phasesweep.evidence.mlflow import compile_query as compile_mlflow_query
    from phasesweep.runtime.process import ProcessResult

    if record.trial_dir is None or record.generation_id is None:
        raise StudyStorageUnavailableError(
            f"Attempt {record.attempt_id} records no trial directory or generation to read."
        )
    try:
        wandb_run_id = _trial_dir_opener(record.trial_dir)
    except ValueError as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt {record.attempt_id} records an unusable trial directory, so the run its "
            f"trial logs to cannot be named: {exc}.",
            action=OperatorAction.RESTORE_LEDGER,
        ) from exc
    environment = _environment_identity(
        producing, record.phase_name, require_wandb_online=False
    ).values
    trainer_input = trial.user_attrs.get(TRAINER_INPUT_ATTR)
    overrides_json = trial.user_attrs.get(OVERRIDES_ATTR)
    if (
        not isinstance(trainer_input, dict)
        or not isinstance(trainer_input.get("sha256"), str)
        or not isinstance(overrides_json, str)
    ):
        raise StudyStorageUnavailableError(
            f"Completed attempt {record.attempt_id} lacks its recorded trainer input or overrides."
        )
    gates = reading.phase(record.phase_name).gates
    ctx = TrialContext(
        experiment=producing.experiment,
        phase=record.phase_name,
        trial_id=trial.number,
        generation_id=record.generation_id,
        attempt_id=record.attempt_id,
        overrides_sha256=trainer_input["sha256"],
        trial_dir=record.trial_dir,
        attempt_dir=record.attempt_dir,
        run_name=trial_run_name(producing, record.phase_name, trial.number, record.attempt_id),
        return_code=0,
        duration_seconds=trainer_duration,
        wandb_run_id=wandb_run_id,
        mlflow_settings=(
            None if producing.mlflow is None else producing.mlflow.target(polling=current.mlflow)
        ),
        mlflow_query=compile_mlflow_query(reading, gates),
        wandb_environment=environment,
        mlflow_environment=environment,
        baseline=baseline,
        evidence_before=evidence_before,
        earlier_attempts=_earlier_attempts(trials, trial),
    )
    result = extract_trial_result(
        experiment=reading,
        executed=ExecutedTrial(ctx, ProcessResult(0, False, -1, trainer_duration)),
        gates=gates,
    )
    return result, overrides_json


def _recover_completed_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    record: _AttemptRecord,
    current: Experiment,
    report: _PreflightCleanupReport | None = None,
) -> bool | None:
    """Accept a finished orphan through the producing configuration's extraction path.

    An idle process slot and successful trainer exit evidence permit acceptance. The slot
    is idle once its group's cleanup is confirmed or its last launch never delivered a
    payload (see :func:`_attempt_process_resolution`). Trainer ledger attributes precede
    SDK workers and take priority over the reused process slot; the guardian covers a
    runner death before those attributes commit. A withheld payload carries no exit status,
    so it proves a finished trainer only through those attributes. Recorded process failures and interruptions remain authoritative; a measured
    bound violation may still commit as COMPLETE and infeasible. Evidence that answers
    with a failure commits FAIL with that failure as the trial's outcome, as the live path
    records it, so finished training is never trained again. Either verdict counts toward the
    producing phase's failure limit as a live one does. Only a ledger study reaches recovery,
    so training it defers stays retained for a later pass.

    :param optuna.Study study: Owning study whose trial is still RUNNING.
    :param optuna.trial.FrozenTrial trial: Stale trial with retained trainer evidence.
    :param _AttemptRecord record: Attempt identity and original trial directory.
    :param Experiment current: Recovery invocation supplying the artifact root and MLflow polling controls.
    :param _PreflightCleanupReport | None report: Collector for newly recovered completed
        results and for evidence the tracker gave no answer about.
    :return bool | None: True once the finished trainer's evidence settled the trial as
        COMPLETE or as a failure, False without proof the trainer finished, or None when
        shutdown or an unreadable tracker leaves extraction pending.
    :raises WinnerIntegrityError: The producing configuration snapshot is unavailable or invalid.
    :raises StudySchemaMismatchError: The study's recorded outcomes cannot be replayed to
        decide the verdict's abort; the trial remains RUNNING.
    :raises StudyStorageUnavailableError: The extracted evidence or its outcome could not be
        committed.
    :raises TrackingTargetUnavailableError: The tracker gave no answer and no report
        collects the pending attempt for a refusal after the pass.
    :raises ProcessCleanupUncertainError: The attempt or a source the rerun continued records
        a malformed directory, so the trial's W&B run or the attempts whose histories it
        continues cannot be named.
    """
    from phasesweep.engine.trial import record_trial_result
    from phasesweep.evidence.baseline import read_evidence_baseline

    lifecycle = read_attempt_lifecycle(record.attempt_dir, expected_attempt_id=record.attempt_id)
    trainer_duration = None if lifecycle is None else _completed_trainer_duration(trial, lifecycle)
    if (
        _record_process_resolution(record, lifecycle) not in {"exited", "payload_withheld"}
        or trainer_duration is None
        or record.generation_id is None
        or record.trial_dir is None
    ):
        return False
    if servicing_shutdown_signal() is not None:
        _warn_deferred_extraction(
            record.attempt_id, record.phase_name, trial.number, persistent=True
        )
        return None
    if report is not None and report.unreadable_tracker is not None:
        # This pass already found a tracker unreadable; keep the training without
        # spending another read's timeout on it.
        report.unreadable_evidence.setdefault(record.attempt_id, report.unreadable_tracker)
        _warn_deferred_extraction(
            record.attempt_id, record.phase_name, trial.number, persistent=True
        )
        return None
    try:
        baseline = read_evidence_baseline(record.attempt_dir)
    except ValueError as exc:
        # Launch writes it before the trainer starts; without it no value file can be shown
        # to be this attempt's own, so the training is not accepted and the trial reruns.
        log.warning("Attempt %s cannot be recovered as finished: %s", record.attempt_id, exc)
        return False
    experiment = _load_generation_experiment(_experiment_dir(current), record.generation_id)
    phase = experiment.phase(record.phase_name)
    active_trial = optuna.Trial(study, trial._trial_id)
    try:
        active_trial.set_user_attr(DURATION_ATTR, trainer_duration)
        active_trial.set_user_attr(RETURN_CODE_ATTR, 0)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Completed attempt {record.attempt_id} trainer exit evidence could not be recorded; its trial remains RUNNING."
        ) from exc
    try:
        result, overrides_json = read_finished_attempt(
            {other.number: other for other in study.get_trials(deepcopy=False)},
            trial,
            record,
            producing=experiment,
            reading=experiment,
            current=current,
            trainer_duration=trainer_duration,
            baseline=baseline,
        )
    except PhaseSweepShutdown:
        _warn_deferred_extraction(
            record.attempt_id, record.phase_name, trial.number, persistent=True
        )
        raise
    if servicing_shutdown_signal() is not None:
        _warn_deferred_extraction(
            record.attempt_id, record.phase_name, trial.number, persistent=True
        )
        return None
    if result.evidence_unreadable and result.failure_reason is not None:
        # No answer says nothing about the training. A pass with a report refuses new
        # work once it is done, since a raise here would strand the registry loop.
        if report is None:
            raise _pending_evidence_refusal({record.attempt_id: result.failure_reason})
        report.unreadable_evidence[record.attempt_id] = result.failure_reason
        if report.unreadable_tracker is None:
            report.unreadable_tracker = result.failure_reason
        log.warning(
            "Completed orphan attempt %s stays pending; its tracker could not be read: %s",
            record.attempt_id,
            result.failure_reason,
        )
        _warn_deferred_extraction(
            record.attempt_id, record.phase_name, trial.number, persistent=True
        )
        return None
    if result.failure_reason is None:
        assert result.metric is not None
    try:
        record_trial_result(active_trial, result, overrides_json=overrides_json)
        _record_stale_trial_outcome(
            study,
            trial,
            "failure" if result.failure_reason is not None else "success",
            # Only a success carries its measurement, as a live trial's outcome does.
            metric=result.metric if result.failure_reason is None else None,
            cause=result.failure_reason,
            policy=(phase, experiment),
        )
        if result.failure_reason is not None:
            active_trial.set_user_attr(FAILURE_REASON_ATTR, result.failure_reason)
    except StudySchemaMismatchError:
        raise
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Completed attempt {record.attempt_id} could not be committed; its retained evidence permits recovery on retry."
        ) from exc
    _tell_recovered_trial(study, trial, record, result.metric, report, producing=experiment)
    return True


def _tell_recovered_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    record: _AttemptRecord,
    metric: float | None,
    report: _PreflightCleanupReport | None,
    *,
    producing: Experiment,
) -> None:
    """Finish the Optuna transition for a trial whose outcome is already durable, and report it.

    :param optuna.Study study: Owning study.
    :param optuna.trial.FrozenTrial trial: Trial to settle.
    :param _AttemptRecord record: Attempt that produced the measurement.
    :param float | None metric: Accepted objective, or None for a failed extraction.
    :param _PreflightCleanupReport | None report: Collector for newly retained measurements.
    :param Experiment producing: The configuration that produced the attempt, for its line.
    :raises StudyStorageUnavailableError: The cleanup record or terminal transition failed.
    """
    _record_cleanup_recovery(study, trial)
    try:
        if metric is None:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial.number, metric)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Completed attempt {record.attempt_id} could not be committed; its retained evidence permits recovery on retry."
        ) from exc
    if metric is not None and report is not None:
        report.retained_results += 1
    _report_recovered_trial(study, trial, producing.phase(record.phase_name), producing)


def _report_recovered_trial(
    study: optuna.Study, trial: optuna.trial.FrozenTrial, phase: Phase, producing: Experiment
) -> None:
    """Print the trial line of a trial recovery just made terminal.

    A line that cannot be built never undoes the settlement already committed.

    :param optuna.Study study: Owning study.
    :param optuna.trial.FrozenTrial trial: The settled trial, as read before its transition.
    :param Phase phase: Its phase in the producing configuration.
    :param Experiment producing: The configuration that produced the attempt.
    """
    try:
        settled = study._storage.get_trial(trial._trial_id)
        finished = _finished_trial_count(study.get_trials(deepcopy=False))
    except Exception:
        log.debug("Recovered trial %d settled; its line could not be read.", trial.number)
        return
    _log_recovered_trial(producing, phase, settled, finished)


def _fail_stale_running_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    where: str,
    record: _AttemptRecord | None = None,
    experiment: Experiment | None = None,
    report: _PreflightCleanupReport | None = None,
) -> bool:
    """Settle a resolved RUNNING trial using completed evidence when available.

    Re-extract a confirmed successful trainer through its producing configuration.
    Otherwise preserve failures or record an interruption. The outcome and cleanup
    ledger precede Optuna's terminal transition. Recovery may read tracker evidence.
    The run delivers MLflow outcomes for settled generations from the ledger.

    :param optuna.Study study: Study containing ``trial``.
    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial whose attempt
        can no longer leak a process.
    :param _AttemptRecord | None record: Attempt record for recovering completed evidence.
    :param Experiment | None experiment: Recovery invocation supplying the artifact root and MLflow polling controls.
    :param _PreflightCleanupReport | None report: Collector for newly recovered completed
        results and for the generations whose trials were settled.
    :param str where: What the finished cleanup is named by, quoted in the refusal when Optuna
        cannot record ``FAIL``.
    :return bool: Whether the trial became terminal; False retains completed training during
        shutdown or while its tracker gives no answer.
    :raises WinnerIntegrityError: Completed trainer evidence requires an unavailable or invalid producing snapshot.
    :raises TrialEvidenceMissingError: Evidence for a recorded measurement is missing or has changed.
    :raises TrackingTargetUnavailableError: The tracker gave no answer and no report collects it.
    :raises StudyStorageUnavailableError: The outcome, the recovery ledger, or
        the terminal state could not be recorded; the trial is left RUNNING
        rather than committed without its outcome.
    """
    try:
        trial = study._storage.get_trial(trial._trial_id)
        outcome = _unique_trial_outcomes(study.study_name, study.get_trials(deepcopy=False)).get(
            trial.number
        )
    except StudySchemaMismatchError:
        raise
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Recovery of stale RUNNING trial {trial.number} in study {study.study_name!r} could not reload its durable outcome before settlement. The trial remains RUNNING for retry."
        ) from exc
    if outcome is not None and outcome.metric is not None:
        from phasesweep.engine.evidence import _EvidenceVerification

        if (
            record is None
            or record.generation_id is None
            or record.trial_dir is None
            or experiment is None
        ):
            raise TrialEvidenceMissingError(
                f"Trial {trial.number} in study {study.study_name!r} has a recorded measurement but no producing attempt to verify."
            )
        producing = _load_generation_experiment(_experiment_dir(experiment), record.generation_id)
        try:
            _EvidenceVerification(producing, producing.phase(record.phase_name)).verify_trial(
                trial,
                record.trial_dir,
                record.generation_id,
                record.attempt_id,
                outcome.metric,
                subject=f"Phase {record.phase_name!r} trial {trial.number}",
                verify_digest=True,
            )
        except TrialEvidenceMissingError as exc:
            if exc.action is not OperatorAction.RESTORE_TREE:
                raise
            raise TrialEvidenceMissingError.rewrap(
                exc, str(exc), remediation=f"{operator_step(exc).rstrip('.')}, or {FRESH_STUDY}."
            ) from exc
        _tell_recovered_trial(study, trial, record, outcome.metric, report, producing=producing)
        recovered = True
    elif outcome is None and record is not None and experiment is not None:
        extracted = _recover_completed_trial(
            study, trial, record=record, current=experiment, report=report
        )
        if extracted is None:
            return False
        recovered = extracted
    else:
        recovered = False
    if not recovered:
        crashed = None
        if outcome is None:
            crashed = _record_unfinished_trial_outcome(
                study, trial, record=record, experiment=experiment
            )
        elif outcome.outcome == "failure" and record is not None:
            # A run stopped right after recording a crash leaves its settlement here, which
            # times it and prints its line as for a crash recovery records itself. The line
            # never blocks the settlement.
            _save_failed_trainer_duration(study, trial, _attempt_lifecycle(record))
            with contextlib.suppress(PhaseSweepError, StopIteration):
                crashed = _producing_phase(record, experiment)
        _record_cleanup_recovery(study, trial)
        try:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        except Exception as exc:
            raise StudyStorageUnavailableError(
                f"Process cleanup completed ({where}), but stale RUNNING trial {trial.number} "
                f"in study {study.study_name!r} could not be marked FAIL: {exc}"
            ) from exc
        if crashed is not None:
            _report_recovered_trial(study, trial, *crashed)
    return True


def _record_unfinished_trial_outcome(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    *,
    record: _AttemptRecord | None,
    experiment: Experiment | None,
) -> tuple[Phase, Experiment] | None:
    """Record the verdict of a stale trial no outcome and no measurement settled.

    A trainer that crashed before its runner died keeps its failure, as on the live path,
    and counts toward the producing phase's ``max_consecutive_failures``: a deterministic
    crash must not rerun forever. Anything PhaseSweep stopped or never started, or whose
    ending is unknown, is interrupted and reruns.

    :param optuna.Study study: Owning study.
    :param optuna.trial.FrozenTrial trial: Stale trial being settled.
    :param _AttemptRecord | None record: Its attempt, whose directory holds the lifecycle record.
    :param Experiment | None experiment: Recovery invocation supplying the artifact root.
    :raises WinnerIntegrityError: A failure's producing configuration snapshot is unavailable.
    :raises StudySchemaMismatchError: The durable completion order is invalid.
    :raises StudyStorageUnavailableError: The verdict could not be recorded.
    :return tuple[Phase, Experiment] | None: For a recorded failure, the phase and configuration
        its trial line is printed under once the trial is terminal; ``None`` for an interruption.
    """
    from phasesweep.engine.trial import _trainer_failure_reason

    reason = None
    lifecycle = None
    if record is not None:
        lifecycle = _attempt_lifecycle(record)
        reason = _trainer_failure_reason(trial, lifecycle, record.attempt_dir)
    if reason is None or record is None:
        _record_stale_trial_outcome(study, trial)
        return None
    # The failure counts under its producing phase's policy, known only with its generation.
    known = record.generation_id is not None
    crashed = _producing_phase(record, experiment) if known else None
    _record_stale_trial_outcome(study, trial, "failure", cause=reason, policy=crashed)
    _save_failed_trainer_duration(study, trial, lifecycle)
    if FAILURE_REASON_ATTR not in trial.user_attrs:
        try:
            optuna.Trial(study, trial._trial_id).set_user_attr(FAILURE_REASON_ATTR, reason)
        except Exception as exc:
            raise StudyStorageUnavailableError(
                f"The failure of stale RUNNING trial {trial.number} in study "
                f"{study.study_name!r} was recorded, but its reason could not be saved."
            ) from exc
    return crashed if known else _producing_phase(record, experiment)


def _attempt_lifecycle(record: _AttemptRecord) -> AttemptLifecycle | None:
    """Read an attempt's lifecycle record, ``None`` when it is missing or unreadable.

    :param _AttemptRecord record: The attempt, whose directory holds the record.
    :return AttemptLifecycle | None: The validated record, if one could be read.
    """
    try:
        return read_attempt_lifecycle(record.attempt_dir, expected_attempt_id=record.attempt_id)
    except ValueError:
        return None


def _producing_phase(
    record: _AttemptRecord, experiment: Experiment | None
) -> tuple[Phase, Experiment] | None:
    """Return the configuration that produced an attempt, and its phase there.

    :param _AttemptRecord record: The attempt, naming its generation and phase.
    :param Experiment | None experiment: Recovery invocation supplying the artifact root.
    :raises WinnerIntegrityError: The generation's configuration snapshot is unavailable.
    :raises StopIteration: That configuration has no such phase.
    :return tuple[Phase, Experiment] | None: The phase and configuration, the invocation's own
        when the attempt names no generation, or ``None`` without an invocation.
    """
    if experiment is None:
        return None
    producing = experiment
    if record.generation_id is not None:
        producing = _load_generation_experiment(_experiment_dir(experiment), record.generation_id)
    return producing.phase(record.phase_name), producing


def _save_failed_trainer_duration(
    study: optuna.Study, trial: optuna.trial.FrozenTrial, lifecycle: AttemptLifecycle | None
) -> None:
    """Save a settled failure's trainer time from its lifecycle record, as a live failure saves it.

    A time the trial already recorded is kept.

    :param optuna.Study study: Owning study.
    :param optuna.trial.FrozenTrial trial: The stale trial settled as a failure.
    :param AttemptLifecycle | None lifecycle: Its attempt's lifecycle record, if one was read.
    :raises StudyStorageUnavailableError: The time could not be saved.
    """
    if DURATION_ATTR in trial.user_attrs or lifecycle is None:
        return
    seconds = finite_nonnegative(lifecycle.duration_seconds)
    if seconds is None:
        return
    try:
        optuna.Trial(study, trial._trial_id).set_user_attr(DURATION_ATTR, float(seconds))
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"The failure of stale RUNNING trial {trial.number} in study "
            f"{study.study_name!r} was recorded, but its trainer time could not be saved."
        ) from exc


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
    :raises StudyStorageUnavailableError: The ledger could not be read or written; MCP cleanup
        uncertainty stays set rather than being cleared without consuming the
        trial evidence.
    """
    try:
        recovered = sorted(_cleanup_recovered_trial_numbers(study) | {trial.number})
        study.set_user_attr(CLEANUP_RECOVERED_TRIALS_ATTR, recovered)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Cleanup was confirmed for trial {trial.number} in study {study.study_name}, "
            f"but the study's cleanup recovery ledger could not be updated: {exc}"
        ) from exc
