"""Engine locks, fingerprints, and stale-trial recovery guards."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config import Experiment, Phase, Suite
from phasesweep.engine.errors import (
    ExperimentLockBusyError,
    SamplerContinuationUnsupportedError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
)
from phasesweep.engine.optuna import _load_existing_phase_study
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    PHASE_FINGERPRINT_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    TRIAL_TARGET_ATTR,
    Winner,
    _attempts_dir,
    _experiment_dir,
    _suite_dir,
    _trial_dir_for,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import (
    atomic_write_text,
    canonical_storage_identity,
    exclusive_lock,
    try_lock_file,
    unlock_file,
)
from phasesweep.runtime.files import (
    lock_dir as _lock_dir,
)
from phasesweep.runtime.json import strict_json_loads
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    AttemptLifecycle,
    StaleProcessIdentity,
    cleanup_stale_trial_process,
    read_attempt_lifecycle,
    read_stale_process_identity,
)


@dataclass
class _PreflightCleanupReport:
    """Cleanup evidence accumulated while inspecting all existing phase studies."""

    cleanup_confirmed: bool = True
    recovered_attempt_ids: set[str] = field(default_factory=set)
    uncertain_attempt_ids: set[str] = field(default_factory=set)
    error: BaseException | None = None

    def mark_uncertain(self, error: BaseException) -> None:
        """Record the first cleanup uncertainty and fail the aggregate closed."""
        self.cleanup_confirmed = False
        if self.error is None:
            self.error = error


def _lock_digest(material: dict[str, Any]) -> str:
    """Hash a lock-material dict into a 24-char hex digest.

    Args:
        material: The output of :func:`_lock_material` (or any
            JSON-serialisable dict).

    Returns:
        First 24 hex characters of the SHA-256 of the canonicalised JSON. 24
        chars = 96 bits — well past collision risk for a same-host advisory
        lock filename.

    """
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _lock_path_from_material(experiment: Experiment, material: dict[str, str], label: str) -> Path:
    """Lock path under the configured host lock directory.

    Args:
        experiment: Parsed experiment config; the experiment name is part of
            the filename for human readability.
        material: Lock-material dict produced by :func:`_lock_material` or
            similar; hashed into the digest segment.
        label: A short human-readable label (``"output"``, ``"storage"``,
            phase name, ...).

    Returns:
        Resolved absolute path to the lock file (the file itself is not
        created here; ``open(...).flock()`` does that lazily).

    """
    return _lock_dir() / f"{experiment.experiment}__{label}__{_lock_digest(material)}.lock"


def _output_lock_material(experiment: Experiment) -> dict[str, str]:
    """Identity for the *output namespace* lock: which directory we write to.

    Catches the case where two configs share a workdir + experiment name but
    point at different storage backends — without an output lock those would
    silently overwrite each other's ``trial_*/``, ``winner.yaml``, and
    ``summary.yaml`` (review v0.5.6 / blocker 1). Always taken regardless of
    storage backend, including in-memory storage.

    Args:
        experiment: Parsed experiment config; supplies workdir + experiment name.

    Returns:
        Lock-material dict keyed on the resolved experiment directory.

    """
    return {"kind": "output", "experiment_dir": str(_experiment_dir(experiment))}


def _storage_run_lock_material(experiment: Experiment) -> dict[str, str] | None:
    """Identity for the *Optuna storage* lock: which study namespace we write to.

    Catches the case where two configs share storage + experiment name but
    point at different workdirs. Returns ``None`` for in-memory storage —
    there is no shared backend, so the output lock alone is sufficient.

    Args:
        experiment: Parsed experiment config; supplies storage + experiment name.

    Returns:
        Lock-material dict keyed on canonical storage identity, or ``None``
        when storage is in-memory.

    """
    storage_identity = canonical_storage_identity(experiment.storage)
    if storage_identity is None:
        return None
    return {
        "kind": "persistent_storage",
        "storage": storage_identity,
        "experiment": experiment.experiment,
    }


def _run_lock_paths(experiment: Experiment) -> list[Path]:
    """All same-host locks required for one full experiment run.

    For persistent storage we take *both* locks (output + storage); for
    in-memory storage we take only the output lock. The list is sorted by
    path so acquisition order is deterministic across processes — relevant
    for clear error messages, not for deadlock avoidance (we use
    ``LOCK_NB``).

    Args:
        experiment: Parsed experiment config.

    Returns:
        Lock-file paths sorted by string order. Length is 1 (output only) for
        in-memory storage, 2 (output + storage) otherwise.

    """
    materials: list[tuple[str, dict[str, str]]] = [
        ("output", _output_lock_material(experiment)),
    ]
    storage_material = _storage_run_lock_material(experiment)
    if storage_material is not None:
        materials.append(("storage", storage_material))
    return sorted(
        (_lock_path_from_material(experiment, m, label) for label, m in materials),
        key=str,
    )


@contextlib.contextmanager
def _experiment_lock(experiment: Experiment) -> Iterator[None]:
    """Take all same-host locks needed for one full experiment run.

    The phase-chained pipeline has cross-phase state — parent ``winner.yaml``,
    child fingerprints, ``summary.yaml``, ``--from-phase`` semantics — so the
    consistency domain is the entire run, not a single phase study. v0.5.7
    extends this further: the consistency domain spans both the *output
    namespace* (filesystem artifacts under ``<workdir>/<experiment>/``) and
    the *Optuna storage namespace* (review v0.5.6 / blocker 1).

    Two configs can disagree on storage but share output paths, or vice
    versa; either case can corrupt skipped-phase reuse and trial logs. We
    therefore take an output lock *always*, and a storage lock additionally
    whenever storage is persistent. In-memory storage has no shared backend,
    so the output lock alone suffices.

    Both locks are *same-host advisory only*; multi-host coordination would need durable per-trial leases and heartbeats rather than just host-local flock files.

    Args:
        experiment: Parsed experiment config.

    Yields:
        ``None``. Use as ``with _experiment_lock(exp): ...``.

    Raises:
        ExperimentLockBusyError: Another phasesweep process holds one of the required
            locks (output namespace or storage identity).

    """
    paths = _run_lock_paths(experiment)
    handles: list[Any] = []
    try:
        for path in paths:
            handle = try_lock_file(path)
            if handle is None:
                raise ExperimentLockBusyError(
                    f"Another phasesweep process appears to be using the same "
                    f"experiment backend or output namespace for "
                    f"{experiment.experiment!r} (lock file: {path}). phasesweep "
                    f"supports one active orchestrator per experiment output "
                    f"namespace and per persistent storage identity."
                )
            handles.append(handle)
        yield
    finally:
        # Reverse-order release isn't required by flock semantics, but it
        # keeps the "stack" mental model intact and parallels typical
        # acquire-A-then-B / release-B-then-A discipline.
        for handle in reversed(handles):
            unlock_file(handle)


@contextlib.contextmanager
def _suite_lock(suite: Suite) -> Iterator[None]:
    """Take a same-host lock for suite-level log and summary artifacts.

    :param Suite suite: Parsed suite config whose output directory names the lock.
    :return Iterator[None]: Context manager yielding ``None`` while the suite lock is held.
    """
    material = {"kind": "suite", "suite_dir": str(_suite_dir(suite))}
    path = _lock_dir() / f"{suite.suite}__suite__{_lock_digest(material)}.lock"
    with exclusive_lock(
        path,
        busy_message=(
            f"Another phasesweep suite process appears to be using {suite.suite!r} "
            f"(lock file: {path})."
        ),
    ):
        yield


_RUN_CONTROL_KEYS = frozenset(
    {
        # Fields excluded from the fingerprint because they don't change trial
        # meaning. Top-up workflow (re-run with a higher n_trials) must work;
        # throughput knobs (n_jobs / gpu_ids) and circuit breakers
        # (max_consecutive_failures) likewise must not invalidate a study.
        # `comment` is operator-facing documentation — editing it is never a
        # semantic change to the experiment.
        "n_trials",
        "n_jobs",
        "gpu_ids",
        "gpu_devices",
        "allow_no_gpu_isolation",
        "max_consecutive_failures",
        "comment",
        "allow_unbounded_trials",
        "timeout_seconds_per_phase",
        "allow_incomplete_on_timeout",
        "allow_partial_grid",
        "allow_seed_search",
    }
)
FINGERPRINT_SCHEMA_VERSION = 2
SUITE_FINGERPRINT_SCHEMA_VERSION = 1
EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 1


def _experiment_semantic_fingerprint(experiment: Experiment) -> str:
    """Hash the experiment semantics that give a published result its meaning.

    Stamped into each generation's summary manifest so reads can tell
    whether the config supplied *today* still matches the config that
    produced a published result (review v0.5.16 / blocker 4) — metric
    name/goal/extractor, constraints, trial command, env, provenance, and
    every phase's ordered semantic identity all contribute. Run-control
    fields (``n_trials`` top-ups, throughput knobs, comments) are excluded
    for the same reason :data:`_RUN_CONTROL_KEYS` excludes them from phase
    fingerprints: they never change what the published numbers mean, so they
    must not flag a published result as reinterpreted.

    :param Experiment experiment: Parsed experiment config to fingerprint.
    :return str: SHA-256 hex digest (64 characters) of the canonicalised
        semantic payload.
    """
    payload = {
        "fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "trial_command": experiment.trial_command,
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "provenance": dict(sorted(experiment.provenance.items())),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        "contracts": {
            name: contract.model_dump(mode="json")
            for name, contract in sorted(experiment.contracts.items())
        },
        "phases": [
            {
                "name": phase.name,
                **{
                    key: value
                    for key, value in phase.model_dump(mode="json").items()
                    if key not in _RUN_CONTROL_KEYS
                },
            }
            for phase in experiment.phases
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _suite_fingerprint(suite: Suite) -> str:
    """Hash the fully compiled suite plan, including historical annotations.

    Args:
        suite: Parsed suite config; each study's name, dependency edges,
            promotion rule, and fully resolved experiment contribute to the
            digest.

    Returns:
        SHA-256 hex digest (64 characters) of the canonicalised suite payload.
        Stamped onto suite-generation records to detect incompatible suite edits.

    """
    payload = {
        "fingerprint_schema_version": SUITE_FINGERPRINT_SCHEMA_VERSION,
        "suite": suite.suite,
        "studies": [
            {
                "name": study.name,
                "depends_on": study.depends_on,
                "promotion": (
                    None if study.promotion is None else study.promotion.model_dump(mode="json")
                ),
                "experiment": suite.experiment_for_study(study).model_dump(mode="json"),
            }
            for study in suite.studies
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _phase_semantic_payload(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> dict[str, Any]:
    """Return a dict capturing only fields that change *trial meaning*.

    Excludes run-control fields (review v0.5.2 / blocker 1) so that bumping
    ``n_trials`` to top up a study is a compatible operation. Includes
    ``experiment.env`` which v0.5.1 missed: env vars like ``CUBLAS_WORKSPACE_CONFIG``
    or ``MY_TRAINER_SEED`` change training behavior and must invalidate reuse.

    Args:
        experiment: The full experiment config; contributes ``trial_command``,
            ``override_format``, ``env``, metric, and constraints.
        phase: The phase being fingerprinted; ``_RUN_CONTROL_KEYS`` are stripped.
        inherited_winners: Winners loaded from parent phases; their
            ``effective_overrides`` are part of this phase's identity.

    Returns:
        A JSON-serialisable dict containing the configured trial semantics and
        operator-declared external provenance.

    """
    phase_dump = phase.model_dump(mode="json")
    semantic_phase = {k: v for k, v in phase_dump.items() if k not in _RUN_CONTROL_KEYS}
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trial_command": experiment.trial_command,
        "provenance": dict(sorted(experiment.provenance.items())),
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        "contracts": {
            name: experiment.contracts[name].model_dump(mode="json") for name in phase.contracts
        },
        "phase": semantic_phase,
        "inherited_effective_overrides": {
            parent: inherited_winners[parent].effective_overrides for parent in phase.inherits
        },
    }


def _phase_fingerprint(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Hash the semantic execution context for resume-compatibility checks.

    Uses the full SHA-256 hex digest. Earlier versions truncated to 16 hex
    chars (64 bits) — defensible against accidental collision but no reason
    to leave the door open in scientific-workflow metadata.

    Args:
        experiment: The experiment config (forwarded to
            :func:`_phase_semantic_payload`).
        phase: The phase being fingerprinted.
        inherited_winners: Parent-phase winners; their effective overrides
            contribute to identity.

    Returns:
        SHA-256 hex digest (64 characters) of the canonicalised semantic
        payload. Used to detect incompatible re-runs and stamped onto
        ``winner.yaml`` files for cross-version verification.

    """
    payload = _phase_semantic_payload(experiment, phase, inherited_winners)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_fingerprint(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Stamp a fresh study with its fingerprint or fail on mismatch.

    Args:
        study: The Optuna study being verified or stamped.
        experiment: The current experiment config.
        phase: The phase whose fingerprint should match the stored one.
        inherited_winners: Parent-phase winners contributing to identity.

    Raises:
        RuntimeError: The study already has a fingerprint and it does not
            match the current computed value (incompatible config edit).

    Returns:
        The verified fingerprint.

    """
    fp = _phase_fingerprint(experiment, phase, inherited_winners)
    existing = study.user_attrs.get(PHASE_FINGERPRINT_ATTR)
    if existing is None:
        study.set_user_attr(PHASE_FINGERPRINT_ATTR, fp)
    elif existing != fp:
        # A zero-trial study whose launch prerequisites failed (bad GPU
        # policy, broken command) must not permanently bind its semantic
        # identity: nothing was ever evaluated under the old fingerprint, so
        # rebinding cannot mix results, and refusing here would make the
        # corrected config a rejected regression (review v0.5.17 / finding A).
        # An accepted trial target is treated as identity too — it means an
        # earlier invocation got past every launch prerequisite.
        if not study.get_trials(deepcopy=False) and study.user_attrs.get(TRIAL_TARGET_ATTR) is None:
            log.warning(
                "Rebinding the fingerprint of empty study %s (%s -> %s): no trial "
                "ever ran under the previous config.",
                study.study_name,
                existing,
                fp,
            )
            study.set_user_attr(PHASE_FINGERPRINT_ATTR, fp)
            return fp
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} was created with a different phase config "
            f"(fingerprint {existing} != {fp}). Use a new experiment name, delete the "
            f"old study, or rename the phase."
        )
    return fp


log = logging.getLogger("phasesweep.engine.guards")


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

    Args:
        trial: RUNNING Optuna trial being reaped.
        experiment: Parsed experiment.
        phase_name: Phase containing the trial.
        study_name: Study name for operator-facing diagnostics.

    Returns:
        Persisted trial directory, or the canonical directory for a pre-launch
        RUNNING trial with no persisted directory attr.

    Raises:
        ProcessCleanupUncertainError: ``phasesweep_trial_dir`` exists but is not a non-empty string, so the reaper cannot safely locate the trial identity files.

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
    if not isinstance(stored, str) or not stored:
        raise ProcessCleanupUncertainError(
            f"Refusing to reap RUNNING trial {trial.number}: invalid persisted "
            f"{TRIAL_DIR_ATTR!r} user attribute {stored!r}. The trial cannot be "
            "tied to its identity files safely."
        )
    return Path(stored)


def _read_trial_process_identity(
    trial: optuna.trial.FrozenTrial,
    trial_dir: Path,
    study_name: str,
) -> StaleProcessIdentity:
    """Read one complete process identity bound to its persisted attempt.

    :param optuna.trial.FrozenTrial trial: RUNNING or terminal trial whose
        process identity is being read for stale-trial recovery.
    :param Path trial_dir: Persisted trial directory expected to contain the
        durable process identity files.
    :param str study_name: Study name, used only for diagnostics.
    :return StaleProcessIdentity: Process identity bound to the trial's
        persisted attempt id.
    :raises ProcessCleanupUncertainError: The trial has no valid persisted
        attempt id, or its durable process identity is missing, malformed,
        partial, or belongs to a different attempt.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover trial {trial.number} in study {study_name}: missing or "
            f"invalid {ATTEMPT_ID_ATTR!r} user attribute. Process identity is unknown."
        )
    try:
        return read_stale_process_identity(
            trial_dir,
            expected_attempt_id=attempt_id,
        )
    except (OSError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover trial {trial.number} in study {study_name}: its durable "
            f"process identity is missing, malformed, partial, or belongs to another attempt. "
            f"trial_dir={trial_dir}."
        ) from exc


ATTEMPT_REGISTRY_SCHEMA_VERSION = 1
_ATTEMPT_ENTRY_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_id",
        "experiment",
        "phase",
        "study_name",
        "storage",
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
    """Best-effort durable registration of a newly allocated attempt.

    The entry binds the attempt to the *producing* phase name, study name,
    and storage URL, so preflight can find and resolve it even after the
    phase was renamed or removed, or the storage URL changed (review
    v0.5.17 / blocker 3). A write failure only loses that cross-config
    coverage for this attempt — the per-phase reaper still recovers it under
    the same config — so it is logged, not raised.

    Args:
        experiment: Parsed experiment config (supplies the registry root).
        attempt_id: Immutable attempt identity; also the entry filename.
        phase_name: Phase the attempt belongs to, as configured *now*.
        study_name: Fully qualified Optuna study name.
        trial_number: Optuna trial number bound to this attempt.
        trial_dir: Resolved per-trial directory holding lifecycle/identity.
        generation_id: Engine invocation identity.

    """
    entry = {
        "schema_version": ATTEMPT_REGISTRY_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "experiment": experiment.experiment,
        "phase": phase_name,
        "study_name": study_name,
        "storage": experiment.storage,
        "trial_number": trial_number,
        "trial_dir": str(trial_dir),
        "generation_id": generation_id,
    }
    try:
        attempts_dir = _attempts_dir(experiment)
        attempts_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            attempts_dir / f"{attempt_id}.json",
            json.dumps(entry, sort_keys=True) + "\n",
        )
    except OSError:
        log.warning(
            "Could not register active attempt %s in the experiment attempt "
            "registry; recovery after a phase rename/removal will not see it.",
            attempt_id,
        )


def _retire_active_attempt(experiment: Experiment, attempt_id: str) -> None:
    """Best-effort removal of a registry entry whose trial is durably terminal.

    Args:
        experiment: Parsed experiment config (supplies the registry root).
        attempt_id: Attempt whose Optuna trial reached a terminal state.

    """
    with contextlib.suppress(OSError):
        (_attempts_dir(experiment) / f"{attempt_id}.json").unlink(missing_ok=True)


def _load_attempt_entry(entry_path: Path) -> dict[str, Any]:
    """Load and validate one attempt registry entry.

    :param Path entry_path: Registry entry file to parse.
    :return dict[str, Any]: The validated entry payload.
    :raises ProcessCleanupUncertainError: The entry is unreadable or malformed
        — recovery cannot know whether a process from it is still alive.
    """
    try:
        payload = strict_json_loads(entry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} is unreadable or malformed. "
            "Recovery cannot prove whether a process from this attempt is still "
            "alive. Investigate the attempt's trial directory, then delete the "
            "entry file if you are certain nothing is running."
        ) from exc
    if (
        not isinstance(payload, dict)
        or not _ATTEMPT_ENTRY_REQUIRED_FIELDS.issubset(payload)
        or payload.get("schema_version") != ATTEMPT_REGISTRY_SCHEMA_VERSION
        or not isinstance(payload.get("attempt_id"), str)
        or not isinstance(payload.get("trial_dir"), str)
        or not isinstance(payload.get("study_name"), str)
        or type(payload.get("trial_number")) is not int
    ):
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} has an unsupported or partial "
            "schema. Delete the entry file only if you are certain no process "
            "from this attempt is running."
        )
    return payload


def _registry_attempt_process_is_resolved(entry: dict[str, Any], entry_path: Path) -> None:
    """Prove no live process can remain from one registered attempt.

    Mirrors :func:`_resolve_attempt_for_reaping` but works from the registry
    entry instead of Optuna user attrs, so it needs neither the producing
    phase to still exist in the config nor the producing storage to be
    reachable.

    :param dict[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :raises ProcessCleanupUncertainError: The attempt cannot be proven safe.
    """
    attempt_id = entry["attempt_id"]
    trial_dir = Path(entry["trial_dir"])
    if not trial_dir.is_dir():
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} points at a missing trial "
            f"directory {trial_dir}; its process state cannot be verified. "
            "Delete the entry file only if you are certain nothing is running."
        )
    try:
        lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=attempt_id)
    except ValueError as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} has a malformed lifecycle record in {trial_dir}."
        ) from exc
    if lifecycle is not None and lifecycle.state == "exited" and lifecycle.cleanup_confirmed:
        return
    identity_missing = not (trial_dir / PROCESS_IDENTITY_FILE).exists()
    if identity_missing and lifecycle is not None and lifecycle.state == "allocated":
        return
    try:
        identity = read_stale_process_identity(trial_dir, expected_attempt_id=attempt_id)
    except (OSError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} has a missing or malformed "
            f"process identity in {trial_dir}."
        ) from exc
    if not cleanup_stale_trial_process(identity):
        raise ProcessCleanupUncertainError(
            f"Registered attempt {attempt_id} (phase {entry['phase']!r}, from "
            f"{entry_path}) may still have a live process group. "
            f"trial_dir={trial_dir} pid={identity.pid} pgid={identity.pgid}. "
            f"Investigate (e.g. `ps -o pid,pgid,cmd -p {identity.pid}`), then "
            "re-run phasesweep."
        )
    log.warning(
        "Cleared orphaned group for registered attempt %s (pid=%s pgid=%s)",
        attempt_id,
        identity.pid,
        identity.pgid,
    )


def _registry_attempt_fail_stale_trial(entry: dict[str, Any], entry_path: Path) -> str:
    """Mark the entry's Optuna trial FAIL through its *recorded* storage.

    Uses the study name and storage URL captured at allocation, not the
    current config, so a renamed phase or changed storage URL still reaches
    the right study.

    :param dict[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :return str: ``"reaped"`` when the stale RUNNING trial was marked FAIL,
        ``"terminal"`` when nothing needed to change (trial already terminal,
        study gone, or in-memory storage), or ``"unreachable"`` when the
        recorded storage could not be reached and the entry must be retained
        for a later retry.
    """
    storage_url = entry["storage"]
    if storage_url is None:
        # In-memory storage died with its orchestrator; nothing to update.
        return "terminal"
    from phasesweep.engine.optuna import _resolve_storage

    try:
        study = optuna.load_study(
            study_name=entry["study_name"],
            storage=_resolve_storage(storage_url),
        )
    except KeyError:
        # The study no longer exists; there is no RUNNING trial to fix.
        return "terminal"
    except Exception:  # noqa: BLE001 - unreachable storage keeps the entry for retry
        log.warning(
            "Attempt registry entry %s references storage that cannot be "
            "reached right now; the stale trial will be retried on a later run.",
            entry_path,
        )
        return "unreachable"
    trials = study.get_trials(deepcopy=False)
    trial = next((t for t in trials if t.number == entry["trial_number"]), None)
    if trial is None or trial.state != optuna.trial.TrialState.RUNNING:
        return "terminal"
    if trial.user_attrs.get(ATTEMPT_ID_ATTR) != entry["attempt_id"]:
        # The RUNNING trial belongs to a different attempt than this entry;
        # leave it for that attempt's own recovery evidence.
        return "terminal"
    try:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    except Exception as exc:
        raise RuntimeError(
            f"Process cleanup completed for registered attempt {entry['attempt_id']}, "
            f"but its stale RUNNING trial {trial.number} in study "
            f"{entry['study_name']!r} could not be marked FAIL. Refusing to "
            "continue with an inconsistent study."
        ) from exc
    log.warning(
        "Reaped stale RUNNING trial %d in study %s via the attempt registry",
        trial.number,
        entry["study_name"],
    )
    return "reaped"


def _preflight_active_attempts(
    experiment: Experiment,
    report: _PreflightCleanupReport,
) -> None:
    """Resolve every registered nonterminal attempt before any launch.

    Runs before the per-phase study loop and is deliberately independent of
    the current phase graph: a stale attempt whose phase was renamed or
    removed, or whose storage URL changed, is still discovered, its process
    group verified/cleaned, and its recorded study repaired (review v0.5.17 /
    blocker 3).

    :param Experiment experiment: Parsed experiment whose registry is scanned.
    :param _PreflightCleanupReport report: Shared cleanup-evidence collector.
    :raises ProcessCleanupUncertainError: A registered attempt could not be
        proven safe.
    """
    attempts_dir = _attempts_dir(experiment)
    if not attempts_dir.is_dir():
        return
    for entry_path in sorted(attempts_dir.glob("*.json")):
        entry = _load_attempt_entry(entry_path)
        attempt_id = entry["attempt_id"]
        try:
            _registry_attempt_process_is_resolved(entry, entry_path)
        except ProcessCleanupUncertainError as exc:
            report.uncertain_attempt_ids.add(attempt_id)
            report.mark_uncertain(exc)
            raise
        outcome = _registry_attempt_fail_stale_trial(entry, entry_path)
        if outcome == "reaped":
            report.recovered_attempt_ids.add(attempt_id)
        if outcome != "unreachable":
            with contextlib.suppress(OSError):
                entry_path.unlink(missing_ok=True)


def _attempt_lifecycle_for_reaping(
    trial: optuna.trial.FrozenTrial,
    trial_dir: Path,
    study_name: str,
) -> AttemptLifecycle | None:
    """Read the durable attempt lifecycle record bound to a stale trial.

    :param optuna.trial.FrozenTrial trial: Trial being inspected for recovery.
    :param Path trial_dir: Persisted trial directory.
    :param str study_name: Study name, used only for diagnostics.
    :return AttemptLifecycle | None: Validated record, or ``None`` for legacy
        attempts that never wrote one.
    :raises ProcessCleanupUncertainError: The trial has no valid persisted
        attempt id, or the record is malformed or belongs to another attempt.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover trial {trial.number} in study {study_name}: missing or "
            f"invalid {ATTEMPT_ID_ATTR!r} user attribute. Process identity is unknown."
        )
    try:
        return read_attempt_lifecycle(trial_dir, expected_attempt_id=attempt_id)
    except ValueError as exc:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover trial {trial.number} in study {study_name}: its attempt "
            f"lifecycle record is malformed or belongs to another attempt. "
            f"trial_dir={trial_dir}."
        ) from exc


def _resolve_attempt_for_reaping(
    trial: optuna.trial.FrozenTrial,
    trial_dir: Path,
    study_name: str,
    *,
    inspect_only: bool = False,
) -> None:
    """Prove that failing one stale RUNNING trial cannot leak a live process.

    Resolution order (review v0.5.17 / blocker 2):

    1. A durable ``exited`` lifecycle with confirmed cleanup means the
       supervised group was already proven gone by the launching orchestrator
       — the crash landed between process exit and the Optuna terminal
       commit. Safe to fail without signalling anything.
    2. A retained process identity means a process was launched — verify and
       clean it the fail-closed way.
    3. A durable ``allocated`` lifecycle with no identity means no process
       was ever created (the worker died queued for a GPU). Safe to fail.
    4. Anything else keeps today's fail-closed behavior.

    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial being resolved.
    :param Path trial_dir: Persisted trial directory.
    :param str study_name: Study name, used only for diagnostics.
    :param bool inspect_only: When ``True``, never signal a process — only
        prove the same resolution the confirming call would take (used by
        ``mcp recover-run`` preflight).
    :raises ProcessCleanupUncertainError: The attempt cannot be proven safe.
    """
    lifecycle = _attempt_lifecycle_for_reaping(trial, trial_dir, study_name)
    if lifecycle is not None and lifecycle.state == "exited" and lifecycle.cleanup_confirmed:
        log.warning(
            "Trial %d in study %s exited (rc=%s) before its terminal state was "
            "committed; failing it without signalling.",
            trial.number,
            study_name,
            lifecycle.return_code,
        )
        return
    identity_missing = not (trial_dir / PROCESS_IDENTITY_FILE).exists()
    if identity_missing and lifecycle is not None and lifecycle.state == "allocated":
        # A missing identity is exactly what 'allocated' predicts: the worker
        # died queued (e.g. waiting for a GPU) before any process existed. A
        # present-but-unreadable identity instead falls through to the strict
        # reader below and fails closed — a launch had begun.
        log.warning(
            "Trial %d in study %s was allocated but no process was ever launched "
            "(orchestrator died while queued); failing it without signalling.",
            trial.number,
            study_name,
        )
        return
    identity = _read_trial_process_identity(trial, trial_dir, study_name)
    if inspect_only:
        return
    safe_to_fail = cleanup_stale_trial_process(identity)
    if not safe_to_fail:
        raise ProcessCleanupUncertainError(
            f"Refusing to mark RUNNING trial {trial.number}: stale process cleanup "
            f"could not prove the process group is gone. trial_dir={trial_dir} "
            f"pid={identity.pid} pgid={identity.pgid}. A leaked training "
            "process may still be holding GPU memory. Investigate "
            f"(e.g. `ps -o pid,pgid,cmd -p {identity.pid}` and "
            f"`kill -9 -- -{identity.pgid}` if appropriate), then re-run "
            "phasesweep."
        )
    log.warning(
        "Cleared orphaned group for trial %d (pid=%s pgid=%s)",
        trial.number,
        identity.pid,
        identity.pgid,
    )


def _reap_stale_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    uncertain_attempt_ids: set[str] | None = None,
) -> int:
    """Mark RUNNING trials as FAIL after killing orphaned process groups.

    :param optuna.Study study: Study whose stale RUNNING trials should be reaped.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for exact
        attempt identities whose durable state was changed to FAIL.
    :param set[str] | None uncertain_attempt_ids: Optional collector for exact
        attempt identities whose cleanup could not be proven.
    :return int: Number of stale RUNNING trials marked as failed.
    """
    count = 0
    try:
        trials = study.get_trials(deepcopy=False)
    except Exception as exc:
        raise ProcessCleanupUncertainError(
            f"Could not inspect study {study.study_name!r} for stale RUNNING trials."
        ) from exc
    for trial in trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        try:
            trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)

            if TRIAL_DIR_ATTR in trial.user_attrs:
                _resolve_attempt_for_reaping(trial, trial_dir, study.study_name)
        except ProcessCleanupUncertainError:
            if uncertain_attempt_ids is not None and isinstance(attempt_id, str) and attempt_id:
                uncertain_attempt_ids.add(attempt_id)
            raise

        if trial.user_attrs.get(CLEANUP_CONFIRMED_ATTR) is False:
            _record_cleanup_recovery(study, trial)
        try:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        except Exception as exc:
            raise RuntimeError(
                f"Stale process cleanup completed for RUNNING trial {trial.number}, "
                f"but Optuna state could not be updated to FAIL. Refusing to continue "
                f"with an inconsistent study. trial_dir={trial_dir}"
            ) from exc

        if recovered_attempt_ids is not None and isinstance(attempt_id, str) and attempt_id:
            recovered_attempt_ids.add(attempt_id)

        log.warning("Reaped stale RUNNING trial %d in study %s", trial.number, study.study_name)
        count += 1
    return count


def _validate_study_schema(study: optuna.Study) -> None:
    """Initialize an empty study or reject populated incompatible storage."""
    trials = study.get_trials(deepcopy=False)
    version = study.user_attrs.get(STUDY_SCHEMA_ATTR)
    if not trials and version is None:
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
        return
    if version == STUDY_SCHEMA_VERSION:
        return

    trial_numbers = [trial.number for trial in trials]
    detail = "missing" if version is None else repr(version)
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} uses unsupported phasesweep storage schema {detail}; "
        f"current schema is {STUDY_SCHEMA_VERSION}. Affected trial numbers: {trial_numbers}. "
        "Use a new experiment name, or archive/delete the old study before running again."
    )


def _accepted_trial_target(study: optuna.Study) -> int:
    """Return the durable target, inferring old current-schema studies from history.

    :param optuna.Study study: Study whose accepted trial target is read.
    :return int: The stored ``phasesweep_trial_target`` user attr, or the
        number of finished trials when no target has been recorded yet.
    :raises StudySchemaMismatchError: The stored target is not a positive int,
        or is lower than the number of already-finished trials.
    """
    finished = sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())
    stored = study.user_attrs.get(TRIAL_TARGET_ATTR)
    if stored is None:
        return finished
    if type(stored) is not int or stored < 1 or finished > stored:
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has invalid {TRIAL_TARGET_ATTR!r}={stored!r} "
            f"for {finished} terminal trial(s). Use a new experiment name, or archive/delete "
            "the inconsistent study before running again."
        )
    return stored


def _validate_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Reject a target lower than the study's durable accepted target.

    :param optuna.Study study: Existing study whose accepted target is checked.
    :param Phase phase: Phase config supplying the requested ``n_trials`` target.
    :raises TrialTargetRegressionError: ``phase.n_trials`` is lower than the
        study's durable accepted target.
    """
    accepted_target = _accepted_trial_target(study)
    if phase.n_trials < accepted_target:
        raise TrialTargetRegressionError(
            f"Phase {phase.name!r} has already accepted a target of {accepted_target} terminal "
            f"trial(s), but the current config requests {phase.n_trials}. Use at least the prior "
            "target or a new experiment name."
        )


def _validate_sampler_continuation(study: optuna.Study, phase: Phase) -> None:
    """Reject any cross-process continuation of a partially complete stateful study.

    TPE and CMA-ES suggestions depend on process-local RNG/optimizer state that
    Optuna storage does not persist. Recreating a seeded sampler in a fresh
    process restarts that stream, so a mid-target resume can exactly repeat
    already-evaluated startup suggestions and spend the remaining budget on
    duplicates. Until PhaseSweep persists real sampler continuation state, a
    stateful phase is restartable only before its first terminal trial or after
    reaching its accepted target.

    :param optuna.Study study: Existing study whose finished-trial count is checked.
    :param Phase phase: Phase config supplying the sampler type and trial target.
    :raises SamplerContinuationUnsupportedError: The phase uses a stateful sampler
        (``tpe`` or ``cmaes``) and either raises its previously accepted trial
        target or was interrupted before reaching it.
    """
    finished = sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())
    if phase.sampler.type not in {"tpe", "cmaes"} or finished == 0:
        return

    accepted_target = _accepted_trial_target(study)
    if phase.n_trials > accepted_target:
        raise SamplerContinuationUnsupportedError(
            f"Phase {phase.name!r} uses {phase.sampler.type!r} and raises its accepted target "
            f"from {accepted_target} to {phase.n_trials} terminal trial(s). PhaseSweep cannot "
            "reproduce this sampler's process-local continuation state safely. Use a new "
            "experiment name, or run the full target in one invocation."
        )
    if finished < accepted_target:
        raise SamplerContinuationUnsupportedError(
            f"Phase {phase.name!r} uses {phase.sampler.type!r} and was interrupted at "
            f"{finished}/{accepted_target} terminal trial(s). PhaseSweep cannot reconstruct "
            "this sampler's exact process-local continuation state, so resuming could "
            "re-evaluate identical suggestions and waste the remaining budget. Use a new "
            "experiment name (optionally with a stateless random/grid sampler), or run the "
            "full target in one uninterrupted invocation."
        )


def _record_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Persist the highest accepted target before the phase launches work.

    :param optuna.Study study: Study whose accepted trial target is stored.
    :param Phase phase: Phase config supplying the new ``n_trials`` target.
    :raises TrialTargetRegressionError: ``phase.n_trials`` is lower than the
        study's already-accepted target.
    """
    accepted_target = _accepted_trial_target(study)
    if phase.n_trials < accepted_target:
        raise TrialTargetRegressionError(
            f"Phase {phase.name!r} cannot lower its accepted trial target from "
            f"{accepted_target} to {phase.n_trials}."
        )
    if phase.n_trials != study.user_attrs.get(TRIAL_TARGET_ATTR):
        study.set_user_attr(TRIAL_TARGET_ATTR, phase.n_trials)


def _preflight_existing_studies(
    experiment: Experiment,
    *,
    cleanup_report: _PreflightCleanupReport | None = None,
) -> dict[str, optuna.Study]:
    """Validate and reap every existing declared phase study before launch.

    :param Experiment experiment: Parsed experiment whose declared phases are inspected.
    :param _PreflightCleanupReport | None cleanup_report: Optional shared report to
        accumulate cleanup evidence into; a fresh one is created if omitted.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name (phases
        with no durable study yet are omitted).
    :raises StudyStorageUnavailableError: A phase's persistent storage could not
        be inspected.
    :raises StudySchemaMismatchError: A phase's study uses an incompatible
        storage schema.
    :raises TrialTargetRegressionError: A phase's study already accepted a
        higher trial target than the current config requests.
    :raises ProcessCleanupUncertainError: Stale-trial cleanup could not be
        confirmed safe for a phase's study.
    :raises RuntimeError: Multiple studies failed preflight for mixed reasons
        not covered by a single common exception type.
    """
    report = cleanup_report or _PreflightCleanupReport()
    studies: dict[str, optuna.Study] = {}
    errors: list[Exception] = []
    # The registry scan runs FIRST and is independent of the declared phase
    # list, so attempts from renamed/removed phases or changed storage URLs
    # are recovered before any current-config validation or launch (review
    # v0.5.17 / blocker 3).
    try:
        _preflight_active_attempts(experiment, report)
    except Exception as exc:
        errors.append(exc)
    for phase in experiment.phases:
        try:
            study = _load_existing_phase_study(experiment, phase)
        except Exception as exc:
            unavailable = StudyStorageUnavailableError(
                f"Could not inspect persistent study storage for phase {phase.name!r}."
            )
            unavailable.__cause__ = exc
            report.mark_uncertain(unavailable)
            errors.append(unavailable)
            continue
        if study is None:
            continue
        studies[phase.name] = study
        try:
            _reap_stale_trials(
                study,
                experiment,
                phase.name,
                recovered_attempt_ids=report.recovered_attempt_ids,
                uncertain_attempt_ids=report.uncertain_attempt_ids,
            )
        except Exception as exc:
            if isinstance(exc, ProcessCleanupUncertainError):
                report.mark_uncertain(exc)
            errors.append(exc)
            continue
        try:
            _validate_study_schema(study)
            _validate_trial_target(study, phase)
        except Exception as exc:
            errors.append(exc)
    if errors:
        first = errors[0]
        if len(errors) == 1:
            raise first
        message = "Experiment recovery preflight found multiple unsafe studies: " + "; ".join(
            str(error) for error in errors
        )
        if all(isinstance(error, StudySchemaMismatchError) for error in errors):
            raise StudySchemaMismatchError(message) from first
        if all(isinstance(error, StudyStorageUnavailableError) for error in errors):
            raise StudyStorageUnavailableError(message) from first
        if all(isinstance(error, TrialTargetRegressionError) for error in errors):
            raise TrialTargetRegressionError(message) from first
        cleanup_error = next(
            (error for error in errors if isinstance(error, ProcessCleanupUncertainError)),
            None,
        )
        if cleanup_error is not None:
            raise ProcessCleanupUncertainError(message) from cleanup_error
        raise RuntimeError(message) from first
    return studies


def _inspect_stale_running_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
) -> int:
    """Count stale RUNNING trials without signaling processes or writing state.

    Used by ``mcp recover-run`` preflight mode. The follow-up ``--confirm`` call
    must still find the same RUNNING trials so it can reap them and persist
    recovery evidence atomically with clearing MCP cleanup uncertainty.

    :param optuna.Study study: Study whose stale RUNNING trials should be inspected.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :return int: Number of stale RUNNING trials found.
    """
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)
        if TRIAL_DIR_ATTR in trial.user_attrs:
            _resolve_attempt_for_reaping(trial, trial_dir, study.study_name, inspect_only=True)
        count += 1
    return count


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
    """Persist that previously uncertain cleanup evidence has been consumed.

    :param optuna.Study study: Study whose cleanup recovery ledger should be updated.
    :param optuna.trial.FrozenTrial trial: Trial whose cleanup evidence was consumed.
    """
    recovered = sorted(_cleanup_recovered_trial_numbers(study) | {trial.number})
    try:
        study.set_user_attr(CLEANUP_RECOVERED_TRIALS_ATTR, recovered)
    except Exception as exc:
        raise RuntimeError(
            f"Cleanup was confirmed for trial {trial.number} in study {study.study_name}, "
            "but the study-level cleanup recovery ledger could not be updated. "
            "Refusing to clear MCP cleanup uncertainty without consuming the trial evidence."
        ) from exc


def _trial_dir_for_cleanup_recovery(
    trial: optuna.trial.FrozenTrial,
    study_name: str,
) -> Path:
    """Return the persisted trial directory for terminal cleanup recovery.

    :param optuna.trial.FrozenTrial trial: Terminal trial with uncertain cleanup.
    :param str study_name: Study name for diagnostics.
    :return Path: Persisted trial directory containing process identity files.
    :raises ProcessCleanupUncertainError: The trial has no safe persisted trial directory.
    """
    stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
    if not isinstance(stored, str) or not stored:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover cleanup-uncertain trial {trial.number} in study "
            f"{study_name}: missing or invalid {TRIAL_DIR_ATTR!r} user attribute "
            f"{stored!r}. The leaked process group cannot be tied to identity files safely."
        )
    return Path(stored)


def _recover_cleanup_uncertain_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
) -> int:
    """Confirm cleanup for terminal trials that explicitly recorded uncertainty.

    ``UnsafeProcessCleanupError`` can leave an Optuna trial in a terminal FAIL state with
    ``phasesweep_cleanup_confirmed=false``. The normal stale reaper intentionally visits
    only RUNNING trials, so operator recovery needs this separate fail-closed inspection
    before clearing MCP cleanup uncertainty.

    :param optuna.Study study: Existing Optuna study for the phase being recovered.
    :param Experiment experiment: Parsed experiment, used for diagnostics.
    :param str phase_name: Name of the phase being recovered.
    :return int: Number of cleanup-uncertain terminal trials confirmed clean.
    :raises ProcessCleanupUncertainError: A recorded trial cannot be inspected or cleaned.
    """
    recovered = 0
    recovered_trial_numbers = _cleanup_recovered_trial_numbers(study)
    for trial in study.get_trials(deepcopy=False):
        if not trial.state.is_finished():
            continue
        if trial.number in recovered_trial_numbers:
            continue
        if trial.user_attrs.get(CLEANUP_CONFIRMED_ATTR) is not False:
            continue

        trial_dir = _trial_dir_for_cleanup_recovery(trial, study.study_name)
        identity = _read_trial_process_identity(trial, trial_dir, study.study_name)
        safe_to_clear = cleanup_stale_trial_process(identity)
        if not safe_to_clear:
            raise ProcessCleanupUncertainError(
                f"Refusing to clear cleanup uncertainty for trial {trial.number} in "
                f"study {study.study_name}: process cleanup could not be confirmed. "
                f"experiment={experiment.experiment} phase={phase_name} "
                f"trial_dir={trial_dir} pid={identity.pid} pgid={identity.pgid}."
            )
        _record_cleanup_recovery(study, trial)
        recovered_trial_numbers.add(trial.number)
        recovered += 1
        log.warning(
            "Confirmed cleanup for terminal cleanup-uncertain trial %d in study %s "
            "(pid=%s pgid=%s)",
            trial.number,
            study.study_name,
            identity.pid,
            identity.pgid,
        )
    return recovered


def _inspect_cleanup_uncertain_trials(study: optuna.Study) -> int:
    """Count recoverable terminal cleanup evidence without signals or writes.

    :param optuna.Study study: Existing study inspected by recovery preflight.
    :return int: Number of unconsumed terminal trials that record cleanup uncertainty.
    :raises ProcessCleanupUncertainError: A trial lacks the persisted identity
        required for a safe confirmed recovery.
    """
    count = 0
    recovered_trial_numbers = _cleanup_recovered_trial_numbers(study)
    for trial in study.get_trials(deepcopy=False):
        if not trial.state.is_finished():
            continue
        if trial.number in recovered_trial_numbers:
            continue
        if trial.user_attrs.get(CLEANUP_CONFIRMED_ATTR) is not False:
            continue
        trial_dir = _trial_dir_for_cleanup_recovery(trial, study.study_name)
        _read_trial_process_identity(trial, trial_dir, study.study_name)
        count += 1
    return count
