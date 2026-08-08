"""Engine locks, fingerprints, and stale-trial recovery guards."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import optuna

from phasesweep.config import Experiment, Phase, Suite
from phasesweep.config.search import NON_RESUMABLE_SAMPLERS
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    ArtifactRootConflictError,
    ArtifactRootRebindError,
    ExperimentLockBusyError,
    LegacyArtifactRootMigrationRequiredError,
    SamplerContinuationUnsupportedError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialEvidenceMissingError,
    TrialTargetRegressionError,
)
from phasesweep.engine.optuna import _load_existing_phase_study
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    PHASE_FINGERPRINT_ATTR,
    PHASE_RECOVERY_ATTR,
    PHASE_RECOVERY_SCHEMA_VERSION,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    TRIAL_TARGET_ATTR,
    Winner,
    _attempts_dir,
    _experiment_dir,
    _last_successful_generation_id,
    _last_successful_suite_generation_path,
    _phase_dir,
    _suite_dir,
    _trial_dir_for,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import (
    atomic_write_text,
    canonical_storage_identity,
    exclusive_lock,
    file_sha256,
    storage_is_in_memory,
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

if TYPE_CHECKING:
    from phasesweep.engine.selection import SelectedTrial

_TRIAL_OUTCOMES = frozenset({"success", "failure", "pruned", "fatal"})


@dataclass(frozen=True)
class _PhasePolicyState:
    """Failure-policy state reconstructed from durable per-trial outcomes."""

    max_sequence: int
    consecutive_failures: int
    recovered_abort_sequence: int | None
    fatal_trial_number: int | None
    fatal_sequence: int | None
    fatal_cause: str | None


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


def _lock_path_from_material(name: str, material: dict[str, str], label: str) -> Path:
    """Lock path under the configured host lock directory.

    ``name`` is part of the lock *filename*, so two configs whose material
    hashes identically still miss each other unless their name components
    match too. Callers must therefore derive it from the same identity the
    material keys on — the resolved output namespace for the output lock —
    not from an arbitrary spelling of it (review v0.5.17 gap hunt).

    Args:
        name: Human-readable identity prefix, derived from the lock material's
            own identity.
        material: Lock-material dict produced by :func:`_lock_material` or
            similar; hashed into the digest segment.
        label: A short human-readable label (``"output"``, ``"storage"``,
            phase name, ...).

    Returns:
        Resolved absolute path to the lock file (the file itself is not
        created here; ``open(...).flock()`` does that lazily).

    """
    return _lock_dir() / f"{name}__{label}__{_lock_digest(material)}.lock"


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
    # Resolve the FULL directory, leaf included: _experiment_dir resolves only
    # the workdir prefix before appending the experiment name, so a symlinked
    # experiment leaf (runs/expA -> runs/expB) would otherwise mint a second
    # lock identity for one physical namespace — and the second orchestrator's
    # preflight would reap the first one's live trials (review v0.5.17 gap
    # hunt). A leaf that does not exist yet resolves to itself.
    return {"kind": "output", "experiment_dir": str(_experiment_dir(experiment).resolve())}


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
    if storage_is_in_memory(experiment.storage):
        # In-memory URLs (``sqlite:///:memory:`` and spellings thereof) have
        # no shared backend to guard; locking their canonical identity would
        # make two unrelated in-memory runs that share an experiment name
        # contend on a lock naming a backend that does not exist (review
        # v0.5.17 gap hunt).
        return None
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
    # The output lock's filename prefix comes from the RESOLVED namespace
    # leaf, not the configured experiment name: a symlinked experiment leaf
    # spells the same physical directory differently, and a prefix mismatch
    # alone would split the lock even with identical material (review
    # v0.5.17 gap hunt).
    paths = [
        _lock_path_from_material(
            _experiment_dir(experiment).resolve().name,
            _output_lock_material(experiment),
            "output",
        )
    ]
    storage_material = _storage_run_lock_material(experiment)
    if storage_material is not None:
        paths.append(_lock_path_from_material(experiment.experiment, storage_material, "storage"))
    return sorted(paths, key=str)


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
# v4 / v3 / v3: an omitted execution.cwd now contributes its effective
# invocation directory instead of an unbound null. The earlier execution
# contract work covered only configured cwd values, which still let identical
# persistent-study identities launch different relative commands from two
# invocation directories. whole_node phases additionally fingerprint their
# configured device-set size — the trainer's world size.
# Existing populated studies from earlier schemas fail the fingerprint check
# on resume; see docs/config.md's upgrade section.
FINGERPRINT_SCHEMA_VERSION = 4
SUITE_FINGERPRINT_SCHEMA_VERSION = 3
EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 3


def _execution_identity(experiment: Experiment) -> dict[str, Any]:
    """Return the execution contract's contribution to semantic fingerprints.

    The trainer's working directory and ambient-environment inheritance are
    semantic inputs: two invocations differing in either can evaluate
    different code or data under one study (review v0.5.17 / blocker 4). A
    configured cwd contributes its RESOLVED path — a relative cwd invoked
    from two directories is two different execution contexts and must not
    share a study. An unconfigured cwd contributes the resolved invocation
    directory because that is where the trainer actually runs. Ambient
    variable *values* are never hashed — they may hold secrets and are not
    declared semantic; put semantic values in ``env``, which is fingerprinted.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return dict[str, Any]: JSON-serialisable execution-identity payload.
    """
    contract = experiment.execution.inherit_env
    return {
        "cwd": str(
            Path(experiment.execution.cwd).expanduser().resolve()
            if experiment.execution.cwd is not None
            else Path.cwd().resolve()
        ),
        "inherit_env": sorted(contract) if isinstance(contract, list) else contract,
    }


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
        "execution": _execution_identity(experiment),
        "provenance": dict(sorted(experiment.provenance.items())),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        "contracts": {
            name: contract.model_dump(mode="json")
            for name, contract in sorted(experiment.contracts.items())
        },
        "phases": [
            {"name": phase.name, **_semantic_phase_dump(phase)} for phase in experiment.phases
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _semantic_phase_dump(phase: Phase) -> dict[str, Any]:
    """Return one phase's model dump reduced to its semantic fields.

    ``gpu_ids``/``gpu_devices`` are run-control (which card runs a trial does
    not change its meaning) — except under ``whole_node``, where the configured
    device-set SIZE is the trainer's world size and therefore semantic: a
    4-GPU DDP evaluation and a 1-GPU evaluation of the same phase must not
    share a study (review v0.5.17 gap hunt). Only the count joins the
    fingerprint, so respelling the same set (indices vs UUIDs) or moving
    hosts does not invalidate a study.

    :param Phase phase: Phase whose semantic payload is being built.
    :return dict[str, Any]: JSON-serializable semantic phase payload.
    """
    dump = {k: v for k, v in phase.model_dump(mode="json").items() if k not in _RUN_CONTROL_KEYS}
    # acknowledge_nonresumable is run-control, not semantics: it never changes
    # what a trial samples or means (on persistent storage its legal value is
    # fully determined by sampler.type), so it must not invalidate a study.
    dump["sampler"].pop("acknowledge_nonresumable", None)
    if phase.gpu_policy == "whole_node":
        tokens = phase.gpu_ids if phase.gpu_ids is not None else phase.gpu_devices
        dump["whole_node_device_count"] = len(tokens or [])
    return dump


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
                "execution_identity": _execution_identity(suite.experiment_for_study(study)),
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
    semantic_phase = _semantic_phase_dump(phase)
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trial_command": experiment.trial_command,
        "provenance": dict(sorted(experiment.provenance.items())),
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "execution": _execution_identity(experiment),
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
        # A zero-trial study must not permanently bind its semantic identity:
        # nothing was ever evaluated under the old fingerprint, so rebinding
        # cannot mix results. This includes a process that died after recording
        # its trial target but before Optuna created the first trial.
        if not study.get_trials(deepcopy=False):
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
    """Durably register a newly allocated attempt before anything is launched.

    The entry binds the attempt to the *producing* phase name, study name,
    and storage URL, so preflight can find and resolve it even after the
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

    Args:
        experiment: Parsed experiment config (supplies the registry root).
        attempt_id: Immutable attempt identity; also the entry filename.
        phase_name: Phase the attempt belongs to, as configured *now*.
        study_name: Fully qualified Optuna study name.
        trial_number: Optuna trial number bound to this attempt.
        trial_dir: Resolved per-trial directory holding lifecycle/identity.
        generation_id: Engine invocation identity.

    Raises:
        ActiveAttemptPersistenceError: The registry entry could not be
            written. No trainer was started and no GPU lease was consumed.

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
    entry_path = _attempts_dir(experiment) / f"{attempt_id}.json"
    try:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(entry_path, json.dumps(entry, sort_keys=True) + "\n")
    except OSError as exc:
        raise ActiveAttemptPersistenceError(
            f"Could not register active attempt {attempt_id} for phase {phase_name!r} "
            f"at {entry_path}: {exc}. No trainer was started and no GPU lease was "
            "consumed. This entry is the only durable record that would let a later "
            "run discover this attempt after a phase rename/removal or a storage-URL "
            "change, so PhaseSweep refuses to launch a trainer it could not account "
            "for. Restore write access to the experiment workdir, then run again."
        ) from exc


def _retire_active_attempt(experiment: Experiment, attempt_id: str) -> None:
    """Best-effort removal of a registry entry whose trial is durably terminal.

    Deletion stays best-effort even though creation is fail-closed: a retained
    entry is safe by construction. It names an already-terminal trial, so
    preflight resolves it against a durable ``exited`` lifecycle, changes
    nothing in the study, and garbage-collects the file. Only a *missing*
    entry loses information (PR #5 review / reviewer 2 pass 2, blocker 5).

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


def _registry_attempt_process_is_resolved(
    entry: dict[str, Any],
    entry_path: Path,
    *,
    inspect_only: bool = False,
) -> None:
    """Prove no live process can remain from one registered attempt.

    Mirrors :func:`_resolve_attempt_for_reaping` but works from the registry
    entry instead of Optuna user attrs, so it needs neither the producing
    phase to still exist in the config nor the producing storage to be
    reachable.

    :param dict[str, Any] entry: Validated registry entry payload.
    :param Path entry_path: Entry file, used only for diagnostics.
    :param bool inspect_only: Validate durable recovery evidence without
        signalling a process.
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
    if inspect_only:
        return
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


def _parsed_trial_outcome(value: Any) -> tuple[int, str, str | None] | None:
    """Return the ordered outcome fields when a trial attr is well formed.

    :param Any value: Raw ``TRIAL_OUTCOME_ATTR`` user-attr value to validate.
    :return tuple[int, str, str | None] | None: ``(sequence, outcome, cause)``
        when ``value`` is a dict with the current schema version, a positive
        int ``sequence``, an ``outcome`` in :data:`_TRIAL_OUTCOMES`, and a
        ``cause`` that is ``None`` or ``str``; ``None`` otherwise.
    """
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schema_version")
    sequence = value.get("sequence")
    outcome = value.get("outcome")
    cause = value.get("cause")
    if (
        schema_version != TRIAL_OUTCOME_SCHEMA_VERSION
        or type(sequence) is not int
        or sequence < 1
        or outcome not in _TRIAL_OUTCOMES
        or (cause is not None and not isinstance(cause, str))
    ):
        return None
    return sequence, outcome, cause


def _record_stale_trial_failure(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Persist a failure-policy outcome before marking a stale trial ``FAIL``.

    An objective may have written its outcome immediately before the
    orchestrator died. Preserve a recorded fatal exception; convert a
    not-yet-committed success/prune to failure because recovery is about to
    commit the trial as ``FAIL``. Malformed or duplicate records are replaced
    with a fresh sequence so current-schema validation can still diagnose any
    other corrupt row without stranding this stale process.

    :param optuna.Study study: Study containing ``trial``, used to read every
        trial's recorded outcome and assign a fresh sequence if needed.
    :param optuna.trial.FrozenTrial trial: Stale RUNNING trial about to be
        marked ``FAIL``.
    :raises RuntimeError: The outcome could not be written to trial user
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
    used_sequences = [parsed[0] for parsed in parsed_by_trial.values()]
    existing = parsed_by_trial.get(trial.number)
    if existing is not None and used_sequences.count(existing[0]) == 1:
        sequence, outcome, cause = existing
        if outcome not in {"failure", "fatal"}:
            outcome = "failure"
            cause = "orchestrator stopped before Optuna committed the terminal trial state"
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
    try:
        active_trial = optuna.Trial(study, trial._trial_id)
        active_trial.set_user_attr(TRIAL_OUTCOME_ATTR, payload)
    except Exception as exc:
        raise RuntimeError(
            f"Process cleanup completed for stale RUNNING trial {trial.number} in study "
            f"{study.study_name!r}, but its durable failure outcome could not be recorded. "
            "The trial remains RUNNING so a later retry cannot silently omit this failure "
            "from max_consecutive_failures."
        ) from exc


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
    :raises RuntimeError: The stale RUNNING trial could not be marked FAIL, or
        its durable failure outcome could not be recorded; the study is left
        inconsistent rather than silently dropping the failure.
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
    stored_attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if stored_attempt_id is not None and stored_attempt_id != entry["attempt_id"]:
        # A conflicting durable id proves the RUNNING row belongs to another
        # attempt. A missing id is different: registration is written before
        # the first Optuna attr, so it is the expected crash/storage-failure
        # window. The registry entry plus the already-validated lifecycle or
        # process identity still binds this exact study and trial safely.
        return "terminal"
    _record_stale_trial_failure(study, trial)
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
) -> set[str]:
    """Resolve every registered nonterminal attempt before any launch.

    Runs before the per-phase study loop and is deliberately independent of
    the current phase graph: a stale attempt whose phase was renamed or
    removed, or whose storage URL changed, is still discovered, its process
    group verified/cleaned, and its recorded study repaired (review v0.5.17 /
    blocker 3).

    :param Experiment experiment: Parsed experiment whose registry is scanned.
    :param _PreflightCleanupReport report: Shared cleanup-evidence collector.
    :return set[str]: Attempt ids inspected during this pass.
    :raises ProcessCleanupUncertainError: A registered attempt could not be
        proven safe.
    """
    attempts_dir = _attempts_dir(experiment)
    if not attempts_dir.is_dir():
        return set()
    inspected: set[str] = set()
    for entry_path in sorted(attempts_dir.glob("*.json")):
        entry = _load_attempt_entry(entry_path)
        attempt_id = entry["attempt_id"]
        inspected.add(attempt_id)
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
    return inspected


def _inspect_active_attempts(experiment: Experiment) -> set[str]:
    """Validate registered attempts without signalling or changing state.

    This is the observational half of ``mcp recover-run``. It scans the same
    phase-independent registry as normal preflight, so renamed phases and
    unavailable current storage cannot hide a trainer from the dry run.

    :param Experiment experiment: Parsed experiment whose registry is scanned.
    :return set[str]: Attempt ids a confirmed recovery would reconcile.
    :raises ProcessCleanupUncertainError: An entry lacks safe recovery evidence.
    """
    attempts_dir = _attempts_dir(experiment)
    if not attempts_dir.is_dir():
        return set()
    inspected: set[str] = set()
    for entry_path in sorted(attempts_dir.glob("*.json")):
        entry = _load_attempt_entry(entry_path)
        attempt_id = entry["attempt_id"]
        _registry_attempt_process_is_resolved(entry, entry_path, inspect_only=True)
        inspected.add(attempt_id)
    return inspected


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
    :raises ProcessCleanupUncertainError: The study's trials cannot be
        inspected, or a stale trial's directory, attempt identity, or process
        cleanup could not be proven safe.
    :raises RuntimeError: Cleanup succeeded but Optuna could not be updated to
        ``FAIL``; refusing to continue with an inconsistent study.
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
        _record_stale_trial_failure(study, trial)
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


def _phase_policy_schema_error(study: optuna.Study, detail: str) -> StudySchemaMismatchError:
    """Build the actionable error used for malformed durable policy state.

    :param optuna.Study study: Study whose durable state is inconsistent,
        used to name the study in the message.
    :param str detail: Specific description of the malformed state.
    :return StudySchemaMismatchError: Constructed error for the caller to
        raise, instructing them to use a new experiment name or
        archive/delete the inconsistent study.
    """
    return StudySchemaMismatchError(
        f"Study {study.study_name!r} has invalid durable failure-policy state: {detail}. "
        "Use a new experiment name, or archive/delete the inconsistent study before "
        "running again."
    )


def _load_phase_policy_state(study: optuna.Study) -> _PhasePolicyState:
    """Validate and reconstruct the durable consecutive-failure state.

    :param optuna.Study study: Study whose ``PHASE_RECOVERY_ATTR`` and
        per-trial outcome attrs are read and validated.
    :return _PhasePolicyState: Reconstructed state: the highest recorded
        outcome sequence, the consecutive-failure count since the last
        recovery boundary, the recovered abort sequence (if any), and the
        first fatal trial's number/sequence/cause (if any).
    :raises StudySchemaMismatchError: ``PHASE_RECOVERY_ATTR`` or a terminal
        trial's outcome attr is malformed, two trials share a completion
        sequence, or the recovery boundary exceeds the largest recorded
        sequence.
    """
    recovery = study.user_attrs.get(PHASE_RECOVERY_ATTR)
    recovery_boundary = 0
    recovered_abort_sequence: int | None = None
    if recovery is not None:
        if not isinstance(recovery, dict):
            raise _phase_policy_schema_error(
                study, f"{PHASE_RECOVERY_ATTR!r} must be an object, got {recovery!r}"
            )
        schema_version = recovery.get("schema_version")
        raw_recovery_boundary = recovery.get("start_after_sequence")
        raw_recovered_abort_sequence = recovery.get("recovered_abort_sequence")
        recovery_target = recovery.get("trial_target")
        if (
            schema_version != PHASE_RECOVERY_SCHEMA_VERSION
            or type(raw_recovery_boundary) is not int
            or raw_recovery_boundary < 1
            or type(raw_recovered_abort_sequence) is not int
            or raw_recovered_abort_sequence < 1
            or raw_recovered_abort_sequence > raw_recovery_boundary
            or type(recovery_target) is not int
            or recovery_target < 1
        ):
            raise _phase_policy_schema_error(
                study, f"{PHASE_RECOVERY_ATTR!r} has malformed fields: {recovery!r}"
            )
        recovery_boundary = raw_recovery_boundary
        recovered_abort_sequence = raw_recovered_abort_sequence

    events: list[tuple[int, int, str, str | None]] = []
    seen_sequences: dict[int, int] = {}
    for trial in study.get_trials(deepcopy=False):
        raw = trial.user_attrs.get(TRIAL_OUTCOME_ATTR)
        parsed = _parsed_trial_outcome(raw)
        if trial.state.is_finished() and parsed is None:
            raise _phase_policy_schema_error(
                study,
                f"terminal trial {trial.number} has missing or malformed "
                f"{TRIAL_OUTCOME_ATTR!r}: {raw!r}",
            )
        if parsed is None:
            continue
        sequence, outcome, cause = parsed
        other_trial = seen_sequences.get(sequence)
        if other_trial is not None:
            raise _phase_policy_schema_error(
                study,
                f"trials {other_trial} and {trial.number} both use completion sequence {sequence}",
            )
        seen_sequences[sequence] = trial.number
        events.append((sequence, trial.number, outcome, cause))

    events.sort()
    max_sequence = events[-1][0] if events else 0
    if recovery_boundary > max_sequence:
        raise _phase_policy_schema_error(
            study,
            f"{PHASE_RECOVERY_ATTR!r} starts after sequence {recovery_boundary}, "
            f"but the largest recorded sequence is {max_sequence}",
        )

    consecutive_failures = 0
    fatal_trial_number: int | None = None
    fatal_sequence: int | None = None
    fatal_cause: str | None = None
    for sequence, trial_number, outcome, cause in events:
        if sequence <= recovery_boundary:
            continue
        if outcome == "success":
            consecutive_failures = 0
        elif outcome in {"failure", "fatal"}:
            consecutive_failures += 1
        if outcome == "fatal" and fatal_trial_number is None:
            fatal_trial_number = trial_number
            fatal_sequence = sequence
            fatal_cause = cause

    return _PhasePolicyState(
        max_sequence=max_sequence,
        consecutive_failures=consecutive_failures,
        recovered_abort_sequence=recovered_abort_sequence,
        fatal_trial_number=fatal_trial_number,
        fatal_sequence=fatal_sequence,
        fatal_cause=fatal_cause,
    )


def _validate_study_schema(study: optuna.Study) -> None:
    """Initialize an empty study or reject populated incompatible storage.

    :param optuna.Study study: Study whose durable schema attr is stamped (when
        empty and unstamped) or validated against the current schema version.
    :raises StudySchemaMismatchError: The study already holds trials or a schema
        stamp from an unsupported version, or its durable failure-policy state
        is malformed.
    """
    trials = study.get_trials(deepcopy=False)
    version = study.user_attrs.get(STUDY_SCHEMA_ATTR)
    if not trials and version is None:
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
        return
    if version == STUDY_SCHEMA_VERSION:
        _load_phase_policy_state(study)
        return

    trial_numbers = [trial.number for trial in trials]
    detail = "missing" if version is None else repr(version)
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} uses unsupported phasesweep storage schema {detail}; "
        f"current schema is {STUDY_SCHEMA_VERSION}. Affected trial numbers: {trial_numbers}. "
        "Use a new experiment name, or archive/delete the old study before running again."
    )


def _validate_study_direction(
    study: optuna.Study,
    goal: Literal["minimize", "maximize"],
) -> None:
    """Reject a durable study whose objective direction differs from the config.

    Optuna's ``load_if_exists=True`` keeps the stored direction and silently
    ignores the direction supplied by a later caller. PhaseSweep therefore
    validates the durable value explicitly before any trial can be launched.

    :param optuna.Study study: Existing or newly created single-objective study.
    :param Literal goal: Direction required by the experiment metric.
    :raises StudySchemaMismatchError: The stored direction does not match ``goal``.
    """
    expected = (
        optuna.study.StudyDirection.MINIMIZE
        if goal == "minimize"
        else optuna.study.StudyDirection.MAXIMIZE
    )
    if study.directions == [expected]:
        return
    stored = ", ".join(direction.name.lower() for direction in study.directions)
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} optimizes {stored or 'no direction'}, but the current "
        f"config requires {goal}. Optuna does not change a persistent study's direction "
        "when load_if_exists=True. Use a new experiment/phase name or remove the "
        "incompatible study before running again."
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
    if phase.sampler.type not in NON_RESUMABLE_SAMPLERS or finished == 0:
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


# Warn-once keys for :func:`_warn_unbounded_environment_inheritance`; preflight
# runs again on the run's failure path, and a suite drives it once per study.
_FULL_ENV_INHERITANCE_WARNED: set[str] = set()


def _warn_unbounded_environment_inheritance(experiment: Experiment) -> None:
    """Warn once when a persisted study inherits the whole ambient environment.

    ``inherit_env: all`` makes every ambient variable an implicit input to
    trials that outlive this process. Their values are never fingerprinted (and
    must not be — they hold secrets), so a later top-up under a drifted shell
    writes into the same study under the same fingerprint, and only the
    recorded environment digest distinguishes them (review v0.5.18 / finding
    F3). In-memory studies keep nothing to drift against, so they stay quiet.

    :param Experiment experiment: Parsed experiment whose contract and storage
        are inspected.
    """
    if experiment.execution.inherit_env != "all":
        return
    if experiment.storage is None or storage_is_in_memory(experiment.storage):
        return
    if experiment.experiment in _FULL_ENV_INHERITANCE_WARNED:
        return
    _FULL_ENV_INHERITANCE_WARNED.add(experiment.experiment)
    log.warning(
        "[%s] execution.inherit_env='all' with persistent storage: trials persist "
        "beyond this process while inheriting every ambient variable, so a later "
        "resume or top-up under a drifted environment reuses the same study and "
        "fingerprint. Declare an explicit execution.inherit_env list (and put "
        "meaning-changing values in env, which is fingerprinted) to bound what "
        "trials can silently depend on.",
        experiment.experiment,
    )


def _artifact_root_identity(experiment: Experiment) -> str:
    """Return the single artifact root a persistent study is allowed to publish into.

    Derived through :func:`_experiment_dir` — the same helper every artifact
    path is built from — so the recorded binding and the namespace actually
    written can never drift apart.

    :param Experiment experiment: Parsed experiment supplying workdir and name.
    :return str: Resolved ``<workdir>/<experiment>`` namespace as a string.
    """
    return str(_experiment_dir(experiment))


def _artifact_root_binding_applies(experiment: Experiment) -> bool:
    """Return whether this experiment's storage can carry an artifact-root binding.

    :param Experiment experiment: Parsed experiment whose storage is inspected.
    :return bool: ``True`` only for persistent storage. In-memory studies do
        not outlive the process, so no later invocation can inherit them and
        no second publication root can conflict with the first.
    """
    return experiment.storage is not None and not storage_is_in_memory(experiment.storage)


def _claim_study_artifact_root(study: optuna.Study, offered: str) -> None:
    """Record the artifact root an already-checked study is allowed to claim.

    :param optuna.Study study: Phase study that :func:`_artifact_root_claim_needed`
        just reported as claimable.
    :param str offered: Resolved artifact root this config publishes into.
    """
    study.set_user_attr(ARTIFACT_ROOT_ATTR, offered)
    log.info("Bound study %s to artifact root %s", study.study_name, offered)


def _artifact_root_claim_needed(study: optuna.Study, experiment: Experiment) -> bool:
    """Decide, without writing anything, whether a study may claim the offered root.

    The read-only half of the binding, split out so a multi-phase run can
    check every declared phase before it claims any of them (re-review v0.5.19
    / blocker B3).

    :param optuna.Study study: Phase study whose recorded binding is inspected.
    :param Experiment experiment: Parsed experiment supplying the artifact root.
    :return bool: ``True`` when the study records no binding and holds no
        trials, so claiming the offered root is safe; ``False`` when it already
        records exactly that root.
    :raises LegacyArtifactRootMigrationRequiredError: The study holds trials but
        records no artifact root, so which workdir owns its evidence is unknown.
    :raises ArtifactRootConflictError: The study is already bound to a different
        artifact root, or carries a binding that is not a string.
    """
    offered = _artifact_root_identity(experiment)
    if ARTIFACT_ROOT_ATTR not in study.user_attrs:
        trial_count = len(study.get_trials(deepcopy=False))
        if trial_count:
            raise LegacyArtifactRootMigrationRequiredError(
                f"Study {study.study_name!r} holds {trial_count} trial(s) but records no "
                "artifact root: it predates artifact-root binding, and an ordinary run "
                "cannot infer which workdir owns its trial directories and publication. "
                f"Adopting the offered root {offered!r} here would let one study back two "
                "artifact trees. Run 'phasesweep rebind-workdir <config>' with a config "
                "whose workdir names the original, complete artifact tree; that validates "
                "the evidence is present there and records the binding. No trial ran and "
                "nothing was published."
            )
        return True
    bound = study.user_attrs[ARTIFACT_ROOT_ATTR]
    if bound == offered:
        return False
    raise ArtifactRootConflictError(
        f"Study {study.study_name!r} publishes into artifact root {bound!r}, but this "
        f"config offers {offered!r}. One persistent study backs exactly one publication "
        "root; running it against a second workdir would top up trials whose artifacts "
        "live under the bound root and publish a divergent result tree. Restore the "
        "original workdir, or - if you have already moved or copied the artifact tree "
        "to the new location - run 'phasesweep rebind-workdir <config>' to move the "
        "binding. No trial ran and nothing was published."
    )


def _bind_study_artifact_root(study: optuna.Study, experiment: Experiment) -> None:
    """Claim, or re-confirm, the one artifact root a single phase study publishes into.

    ``workdir`` is deliberately excluded from every semantic fingerprint so an
    artifact tree stays movable. Without a binding that mobility is unsound
    (review v0.5.19 / finding F5): re-running the same config with a second
    ``workdir`` matches the same fingerprints, tops up or re-selects against
    trial directories living under the *first* root, and publishes a second,
    divergent artifact tree — two roots each claiming to be the publication of
    one study, with ``trial_dir`` attrs pointing into the other.

    First contact claims the root only for a study that holds **no trials**,
    mirroring the zero-trial fingerprint rebind in :func:`_verify_fingerprint`:
    nothing was evaluated yet, so nothing can be stranded under another root.
    A *populated* study with no binding predates the attr, and an ordinary run
    cannot tell which workdir holds its trial directories; adopting whichever
    workdir happened to run first would leave two trees each reporting an
    intact publication of the same study (re-review v0.5.19 / blocker B1).
    Migrating one is the operator's explicit statement, made against the
    original tree with ``phasesweep rebind-workdir``.

    :param optuna.Study study: Phase study to bind.
    :param Experiment experiment: Parsed experiment supplying the artifact root.
    :raises LegacyArtifactRootMigrationRequiredError: The study holds trials but
        records no artifact root, so which workdir owns its evidence is unknown.
    :raises ArtifactRootConflictError: The study is already bound to a different
        artifact root, or carries a binding that is not a string.
    """
    if not _artifact_root_binding_applies(experiment):
        return
    if not _artifact_root_claim_needed(study, experiment):
        return
    _claim_study_artifact_root(study, _artifact_root_identity(experiment))


def _load_and_check_artifact_roots(experiment: Experiment) -> dict[str, optuna.Study]:
    """Load every existing phase study exactly once, then check and claim roots.

    Discovery on this mutating path is strict and tri-state (PR #5 review /
    reviewer 2, issue 1): a phase's study is *absent* (storage read fine, no
    such study), *present* (loaded and returned), or *unavailable* -- and
    unavailable raises before any claim, reaping, or registry recovery runs.
    The earlier shape swallowed a read failure here and let the main preflight
    loop re-read; a transient failure (a brief SQLite lock, an RDB reconnect)
    could then succeed on that second read, handing stale-trial reaping a
    study whose artifact-root binding was never checked -- mutation of a
    wrong-root ledger before the conflict was enforced. The returned mapping
    is therefore the *only* discovery pass: the study objects that passed the
    root check here are the exact objects preflight goes on to reap and
    validate.

    Every declared phase is checked before any claim is written (re-review
    v0.5.19 / blocker B3), so a refused multi-phase run leaves no study bound
    to the rejected root. Preflight holds the experiment lock across load,
    check, and claim, so nothing can bind in between. Phases whose study does
    not exist yet are bound by :func:`_bind_study_artifact_root` when the
    phase creates them.

    :param Experiment experiment: Parsed experiment whose declared phase
        studies are loaded and bound to its resolved artifact root.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name
        (phases with no durable study yet are omitted).
    :raises StudyStorageUnavailableError: A phase's persistent storage could
        not be inspected, so whether its study exists -- and what root it is
        bound to -- cannot be determined.
    :raises LegacyArtifactRootMigrationRequiredError: A populated phase study
        records no artifact root, so which workdir owns its evidence is unknown.
    :raises ArtifactRootConflictError: A phase study is already bound to a
        different artifact root, or carries a binding that is not a string.
    """
    loaded: dict[str, optuna.Study] = {}
    for phase in experiment.phases:
        try:
            study = _load_existing_phase_study(experiment, phase)
        except Exception as exc:
            unavailable = StudyStorageUnavailableError(
                f"Could not inspect persistent study storage for phase {phase.name!r}."
            )
            raise unavailable from exc
        if study is not None:
            loaded[phase.name] = study
    if not _artifact_root_binding_applies(experiment):
        return loaded
    claimable = [
        study for study in loaded.values() if _artifact_root_claim_needed(study, experiment)
    ]
    offered = _artifact_root_identity(experiment)
    for study in claimable:
        _claim_study_artifact_root(study, offered)
    return loaded


@dataclass(frozen=True)
class _ArtifactRootRebindEntry:
    """One existing phase study, its declared phase, and the root it records now."""

    phase_name: str
    study: optuna.Study
    previous: str | None

    @property
    def is_populated(self) -> bool:
        """Return whether this study holds any trial at all.

        :return bool: ``True`` when the study's ledger is non-empty.
        """
        return bool(self.study.get_trials(deepcopy=False))


@dataclass(frozen=True)
class _ArtifactRootRebindPlan:
    """One experiment's validated artifact-root rebind, before anything is written."""

    experiment: Experiment
    destination: str
    entries: tuple[_ArtifactRootRebindEntry, ...]

    @property
    def has_binding(self) -> bool:
        """Return whether any of this experiment's studies already records a binding.

        :return bool: ``True`` when at least one existing phase study is bound.
        """
        return any(entry.previous is not None for entry in self.entries)

    @property
    def has_unbound_populated_study(self) -> bool:
        """Return whether any existing study holds trials but records no binding.

        Such a study predates artifact-root binding, so no ordinary run will
        ever claim it (re-review v0.5.19 / blocker B1) and this command is its
        only migration path.

        :return bool: ``True`` when at least one existing phase study is
            populated and unbound.
        """
        return any(entry.previous is None and entry.is_populated for entry in self.entries)


def _artifact_root_rebind_entries(
    experiment: Experiment,
) -> tuple[_ArtifactRootRebindEntry, ...]:
    """Load every existing phase study with the artifact root it currently records.

    :param Experiment experiment: Parsed experiment whose phase studies are read.
    :return tuple[_ArtifactRootRebindEntry, ...]: Each existing study with its
        declared phase name and its recorded binding (``None`` when unbound).
    :raises ArtifactRootRebindError: A phase study exists but cannot be read, so
        what it is bound to is unknown and a rebind cannot be safe.
    """
    entries: list[_ArtifactRootRebindEntry] = []
    for phase in experiment.phases:
        try:
            study = _load_existing_phase_study(experiment, phase)
        except Exception as exc:
            raise ArtifactRootRebindError(
                f"Cannot inspect the persistent study for phase {phase.name!r} of experiment "
                f"{experiment.experiment!r}. Refusing to rebind while any study's current "
                "artifact root is unknown. Nothing was written."
            ) from exc
        if study is None:
            continue
        bound = study.user_attrs.get(ARTIFACT_ROOT_ATTR)
        entries.append(
            _ArtifactRootRebindEntry(
                phase_name=phase.name,
                study=study,
                previous=bound if isinstance(bound, str) else None,
            )
        )
    return tuple(entries)


def _studies_record_publication(entries: Sequence[_ArtifactRootRebindEntry]) -> bool:
    """Return whether storage records that this experiment produced a publishable result.

    A COMPLETE trial is the durable, root-independent evidence that the
    experiment reached the point of selecting a winner and publishing it. The
    source tree is gone by the time a rebind runs, so this is the only side of
    the move that can still be inspected.

    :param Sequence[_ArtifactRootRebindEntry] entries: Existing phase studies
        with their recorded bindings.
    :return bool: ``True`` when any phase study holds a COMPLETE trial.
    """
    return any(
        trial.state == optuna.trial.TrialState.COMPLETE
        for entry in entries
        for trial in entry.study.get_trials(deepcopy=False)
    )


def _running_trial_recoverable_in_place(
    entry: _ArtifactRootRebindEntry,
    trial: optuna.trial.FrozenTrial,
    phase_dir: Path,
    offered: str,
) -> bool:
    """Return whether a RUNNING trial's recovery can run at this destination as-is.

    True only for adopting an unbound study **at its original tree** (PR #5
    review / reviewer 2, issue 2): the study records no binding (or already
    records exactly this destination, so a partially applied earlier rebind
    still converges), and the trial's persisted directory resolves to exactly
    the structurally expected path under the offered destination. Exact
    equality is the proof that the destination *is* the original root rather
    than a relocation: a copied or moved tree holds the evidence, but the
    stored absolute path still names the source. For that proven case the
    binding can be written with the trial left RUNNING -- the next ordinary
    run recovers it through the standard stale-attempt protocol, which is the
    one implementation that writes the durable failure outcome, retires the
    registry entry, and keeps the failure-policy ledger valid. Telling the
    operator to ``tell(FAIL)`` the trial by hand instead produced a terminal
    trial with no ``phasesweep_trial_outcome`` and a permanently rejected
    study schema.

    :param _ArtifactRootRebindEntry entry: The study entry holding the trial.
    :param optuna.trial.FrozenTrial trial: The RUNNING trial being judged.
    :param Path phase_dir: Destination directory of the trial's declared phase.
    :param str offered: The destination artifact-root identity.
    :return bool: ``True`` when the binding may be written with this trial
        left RUNNING for the next ordinary run to recover.
    """
    if entry.previous is not None and entry.previous != offered:
        return False
    stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
    if not isinstance(stored, str) or not stored:
        return False
    stored_path = Path(stored)
    if not stored_path.is_absolute():
        return False
    expected = phase_dir / stored_path.name
    try:
        return stored_path.resolve(strict=True) == expected.resolve(strict=True)
    except OSError:
        return False


def _validate_relocated_trial_evidence(
    experiment: Experiment,
    entries: Sequence[_ArtifactRootRebindEntry],
) -> None:
    """Require the destination tree to hold the evidence every ledger trial names.

    The persisted ``phasesweep_trial_dir`` attr is an absolute path that
    stale-trial recovery, cleanup recovery, and every later diagnosis read back
    verbatim, so a rebind is only sound when the same evidence really is under
    the destination (re-review v0.5.19 / blocker B2). Each stored path is
    translated *structurally* - the destination's directory for the known phase
    plus the stored directory's own name - rather than by re-rooting it against
    a recorded previous root: that also covers a tree moved more than once, and
    the adoption of a study that never recorded a root at all.

    Two refusals fall out of the same walk. A ``RUNNING`` trial means an
    attempt was interrupted and never resolved; recovery follows the paths that
    attempt persisted, so a *relocation* must resolve it against the original
    root before the tree moves. The one allowed exception is adoption at the
    original root itself (:func:`_running_trial_recoverable_in_place`), where
    those persisted paths already point exactly where they should and the next
    ordinary run performs the recovery. A stale copy - taken before the ledger
    advanced - is rejected by the missing directory of the trial that ran
    after the copy, which is exactly the case that would otherwise publish a
    winner whose evidence exists only in the source tree. A terminal trial
    that never persisted the attr died before launch and has no evidence to
    move, so it is skipped.

    :param Experiment experiment: Parsed experiment naming the destination.
    :param Sequence[_ArtifactRootRebindEntry] entries: Existing phase studies
        with their recorded bindings.
    :raises ArtifactRootRebindError: A study holds a RUNNING trial that cannot
        be recovered at this destination, records a malformed trial directory,
        or names a trial directory that does not exist under the destination
        tree.
    """
    offered = _artifact_root_identity(experiment)
    for entry in entries:
        phase_dir = _phase_dir(experiment, entry.phase_name)
        for trial in entry.study.get_trials(deepcopy=False):
            if trial.state == optuna.trial.TrialState.RUNNING:
                if _running_trial_recoverable_in_place(entry, trial, phase_dir, offered):
                    continue
                raise ArtifactRootRebindError(
                    f"Study {entry.study.study_name!r} holds RUNNING trial {trial.number}: an "
                    "interrupted attempt must be resolved before its artifact tree moves, "
                    "because stale-attempt recovery follows the absolute trial paths that "
                    "attempt persisted. Run phasesweep against the original workdir to "
                    "recover it, then move the tree and rebind. A study that predates "
                    "artifact-root binding is instead adopted in place: run "
                    "'phasesweep rebind-workdir' with a config whose workdir IS the tree "
                    "this trial ran under (its persisted trial path must already lie "
                    "there), and the next ordinary run recovers the trial through the "
                    "standard stale-attempt protocol. Nothing was written."
                )
            if TRIAL_DIR_ATTR not in trial.user_attrs:
                continue
            stored = trial.user_attrs[TRIAL_DIR_ATTR]
            if not isinstance(stored, str) or not stored or not Path(stored).is_absolute():
                raise ArtifactRootRebindError(
                    f"Study {entry.study.study_name!r} trial {trial.number} records an invalid "
                    f"{TRIAL_DIR_ATTR!r} user attribute {stored!r}; a rebind cannot locate that "
                    "trial's evidence under the destination tree. Nothing was written."
                )
            translated = phase_dir / Path(stored).name
            if not translated.is_dir():
                raise ArtifactRootRebindError(
                    f"Destination artifact root {str(_experiment_dir(experiment))!r} is missing "
                    f"the evidence for trial {trial.number} of study "
                    f"{entry.study.study_name!r}: expected directory {str(translated)!r} "
                    f"(recorded as {stored!r}). The moved or copied tree is incomplete, or it "
                    "is a stale copy taken before that trial ran. Move the complete artifact "
                    "tree, then rebind. Nothing was written."
                )


def _attempt_entry_recoverable_in_place(entry_path: Path, destination: Path) -> bool:
    """Return whether a registry entry's recorded trial path lies inside this tree.

    An entry whose absolute trial directory resolves to an existing directory
    under the destination experiment tree was written *by* this tree: the
    attempt started under this exact root, so the next ordinary run here can
    follow the recorded path and resolve it (PR #5 review / reviewer 2,
    issue 2). An entry pointing anywhere else - or one that cannot be parsed -
    would be stranded by the rebind and refuses it.

    :param Path entry_path: Registry entry file to inspect.
    :param Path destination: Resolved destination experiment directory.
    :return bool: ``True`` when the entry's recorded trial directory exists
        under ``destination``.
    """
    try:
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    recorded = payload.get("trial_dir")
    if not isinstance(recorded, str) or not recorded or not Path(recorded).is_absolute():
        return False
    try:
        resolved = Path(recorded).resolve(strict=True)
    except OSError:
        return False
    return resolved.is_dir() and destination in resolved.parents


def _validate_no_live_attempts(experiment: Experiment) -> None:
    """Refuse a rebind that would strand an unresolved attempt's recorded paths.

    A registry entry is written when an attempt is allocated and unlinked once
    its trial is durably terminal, so a surviving entry is an attempt nobody
    resolved. Every entry stores the attempt's absolute trial directory and
    recovery fails closed when that path is missing, so relocating the tree
    under an unresolved attempt strands it permanently (re-review v0.5.19 /
    blocker B2). The registry lives inside the experiment tree, so this reads
    the copy that came along with the move.

    Entries whose recorded trial directory already exists *under this
    destination* are allowed through: those attempts started under this exact
    root (the adoption-in-place case), the recorded paths still resolve, and
    the next ordinary run recovers them through the standard protocol (PR #5
    review / reviewer 2, issue 2).

    :param Experiment experiment: Parsed experiment naming the destination.
    :raises ArtifactRootRebindError: The destination registry holds at least
        one unresolved attempt entry whose recorded paths do not resolve
        under this destination.
    """
    attempts_dir = _attempts_dir(experiment)
    if not attempts_dir.is_dir():
        return
    destination = _experiment_dir(experiment).resolve()
    stranded = [
        entry_path
        for entry_path in sorted(attempts_dir.glob("*.json"))
        if not _attempt_entry_recoverable_in_place(entry_path, destination)
    ]
    if not stranded:
        return
    raise ArtifactRootRebindError(
        f"Destination artifact root {str(_experiment_dir(experiment))!r} still holds "
        f"unresolved attempt registry entries whose recorded trial paths do not resolve "
        f"under it ({len(stranded)} total, first {str(stranded[0])!r}). Each entry "
        "references the absolute trial path its attempt was launched with, and recovery "
        "cannot follow those paths after a relocation. Run phasesweep against the "
        "original workdir until recovery clears these attempts, then move the tree and "
        "rebind. Nothing was written."
    )


def _validate_artifact_root_destination(
    experiment: Experiment,
    entries: Sequence[_ArtifactRootRebindEntry],
) -> None:
    """Confirm the config's workdir really holds this experiment's relocated tree.

    Four checks, in order: the destination experiment namespace must exist; the
    destination must hold the trial evidence every ledger trial names, with no
    RUNNING trial left to recover except one that is recoverable in place at
    its original root (:func:`_validate_relocated_trial_evidence`); the
    relocated registry must hold no unresolved attempt whose recorded paths
    do not resolve under this destination
    (:func:`_validate_no_live_attempts`); and - when storage records that a
    publishable result was produced - the destination's last-success pointer
    and its complete generation manifest must validate there through
    :func:`_last_successful_generation_id`, the same authoritative read every
    other publication consumer uses.

    Deliberately conservative in one case: an experiment whose trials completed
    but whose publication never succeeded is indistinguishable, from the
    storage side, from a destination that lost its publication during the move.
    That is refused. Nothing published means nothing to preserve, so the remedy
    is the original workdir or a new experiment name.

    :param Experiment experiment: Parsed experiment naming the destination.
    :param Sequence[_ArtifactRootRebindEntry] entries: Existing phase studies
        with their recorded bindings.
    :raises ArtifactRootRebindError: The destination namespace is missing, its
        per-trial evidence is incomplete, a trial is still RUNNING, an attempt
        is unresolved, or a recorded publication does not validate there.
    """
    destination = _experiment_dir(experiment)
    if not destination.is_dir():
        raise ArtifactRootRebindError(
            f"Destination artifact root {str(destination)!r} does not exist. Move or copy the "
            "experiment's artifact tree to the workdir this config declares before rebinding. "
            "Nothing was written."
        )
    _validate_relocated_trial_evidence(experiment, entries)
    _validate_no_live_attempts(experiment)
    if not _studies_record_publication(entries):
        return
    try:
        published = _last_successful_generation_id(experiment, raise_on_manifest_error=True)
    except RuntimeError as exc:
        raise ArtifactRootRebindError(
            f"Destination artifact root {str(destination)!r} records a publication that does "
            f"not validate ({exc}). Move the complete artifact tree, then rebind. "
            "Nothing was written."
        ) from exc
    if published is None:
        raise ArtifactRootRebindError(
            f"Destination artifact root {str(destination)!r} holds no valid published "
            "generation, but this storage's studies record completed trials. Refusing to "
            "rebind onto a tree that is missing the publication those trials produced. Move "
            "the complete artifact tree, then rebind. Nothing was written."
        )


def _plan_artifact_root_rebinds(
    experiments: Sequence[Experiment],
) -> list[_ArtifactRootRebindPlan]:
    """Validate every experiment's rebind destination without writing anything.

    Planning deliberately makes no claim about a *coherent* previous root: a
    crash between two per-study attr writes leaves a mixture of source-bound
    and destination-bound studies, and a second identical invocation has to
    converge instead of refusing (re-review v0.5.19 / blocker B2). Every study
    is validated against the destination and rewritten to it, so re-running the
    command is idempotent.

    :param Sequence[Experiment] experiments: Experiments the config compiles to;
        one for a single experiment, one per study for a suite.
    :return list[_ArtifactRootRebindPlan]: One validated plan per experiment,
        ready to apply.
    :raises ArtifactRootRebindError: Every experiment uses in-memory storage,
        every existing study is unbound *and* empty, a study cannot be read, or
        a destination fails validation.
    """
    persistent = [
        experiment for experiment in experiments if _artifact_root_binding_applies(experiment)
    ]
    if not persistent:
        raise ArtifactRootRebindError(
            "This config uses in-memory storage, so nothing is bound to an artifact root: "
            "in-memory studies do not outlive the process that created them and can never "
            "conflict with a second workdir. Nothing was written."
        )
    plans = [
        _ArtifactRootRebindPlan(
            experiment=experiment,
            destination=_artifact_root_identity(experiment),
            entries=_artifact_root_rebind_entries(experiment),
        )
        for experiment in persistent
    ]
    if not any(plan.has_binding or plan.has_unbound_populated_study for plan in plans):
        raise ArtifactRootRebindError(
            "No phase study in this storage is bound to an artifact root, and none holds a "
            "trial, so there is nothing to rebind or migrate; the next ordinary run binds "
            "these empty studies to the configured workdir. Nothing was written."
        )
    for plan in plans:
        if plan.entries:
            _validate_artifact_root_destination(plan.experiment, plan.entries)
    return plans


def _apply_artifact_root_rebind(plan: _ArtifactRootRebindPlan) -> list[tuple[str, str | None, str]]:
    """Write one validated plan's new artifact root onto every existing phase study.

    :param _ArtifactRootRebindPlan plan: Plan already validated by
        :func:`_plan_artifact_root_rebinds`.
    :return list[tuple[str, str | None, str]]: One ``(study name, previous
        root or None, new root)`` record per study written.
    """
    written: list[tuple[str, str | None, str]] = []
    for entry in plan.entries:
        entry.study.set_user_attr(ARTIFACT_ROOT_ATTR, plan.destination)
        written.append((entry.study.study_name, entry.previous, plan.destination))
    return written


def _validate_suite_artifact_root_rebind(
    suite: Suite,
    plans: Sequence[_ArtifactRootRebindPlan],
) -> None:
    """Refuse a suite rebind whose published summaries cannot survive relocation.

    A published suite summary anchors every study to the **absolute** path of
    the component generation summary it derives from, and validation re-reads
    that exact path, so a relocated suite publication is reported as corrupt by
    the read surfaces the moment it is rebound (re-review v0.5.19 / blocker
    B2). Component-level rebinds still proceed for a suite that never published
    one: those studies carry only per-experiment state, which this command does
    validate.

    Conservative in the same way :func:`_validate_artifact_root_destination`
    is: when every declared study records a completed trial, the suite may well
    have published, and a destination with no suite pointer is
    indistinguishable from one whose pointer was lost in the move.

    :param Suite suite: Suite config naming the destination suite namespace.
    :param Sequence[_ArtifactRootRebindPlan] plans: Validated per-study plans.
    :raises ArtifactRootRebindError: The destination holds a suite publication,
        or every declared study completed a trial while the destination holds
        no suite publication at all.
    """
    pointer = _last_successful_suite_generation_path(suite)
    if pointer.exists():
        raise ArtifactRootRebindError(
            f"Suite {suite.suite!r} records a published suite generation at {str(pointer)!r}. "
            "A published suite summary pins each study to the absolute path of the component "
            "summary it derives from, and those paths do not survive relocation: the rebound "
            "tree would report a corrupt suite publication instead of a result. Keep the suite "
            "at its original workdir, or use a new suite name for the relocated tree. Nothing "
            "was written."
        )
    if (
        plans
        and len(plans) == len(suite.studies)
        and all(plan.entries and _studies_record_publication(plan.entries) for plan in plans)
    ):
        raise ArtifactRootRebindError(
            f"Every study of suite {suite.suite!r} records a completed trial, but the "
            f"destination suite namespace {str(_suite_dir(suite))!r} holds no published suite "
            "generation. A fully completed suite may have published one, and a missing pointer "
            "is indistinguishable from a pointer lost in the move; a suite publication cannot "
            "be relocated because its summaries record absolute component paths. Keep the "
            "suite at its original workdir, or use a new suite name for the relocated tree. "
            "Nothing was written."
        )


_TRIAL_EVIDENCE_REMEDY = (
    "PhaseSweep will not select or republish a result whose evidence is gone: restore the "
    "artifact tree from a backup, or start a new experiment name so nothing ranks against "
    "trials that can no longer be inspected."
)

# Audit artifacts every launched attempt writes into its own trial directory
# before the trainer starts, so their absence is proof the directory is no
# longer the one that attempt produced (PR #5 review / reviewer 2, blocker 7).
_REQUIRED_TRIAL_EVIDENCE_FILES = ("overrides_resolved.json", "command.txt")


def _selection_candidate_identity(trial: optuna.trial.FrozenTrial) -> tuple[str, str] | None:
    """Return a trial's execution identity when it could win winner selection.

    Mirrors the eligibility filter in
    :func:`phasesweep.engine.selection.select_winner` exactly: COMPLETE state, a
    finite value, a truthy feasibility attr, and nonempty generation/attempt
    ids. A trial that fails any of these can never be selected, so declining to
    verify its evidence is not result-biasing - unlike skipping an eligible
    trial, which would change which trial wins.

    The constraint-bounds half of that filter is deliberately *not* mirrored:
    constraint bounds are config-mutable, so a trial outside today's bounds can
    re-enter the candidate set under a later config and its evidence must still
    be there when it does.

    :param optuna.trial.FrozenTrial trial: Persisted trial to classify.
    :return tuple[str, str] | None: ``(generation_id, attempt_id)`` for a
        selection-eligible trial, ``None`` for one that can never win.
    """
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return None
    if trial.value is None or not math.isfinite(trial.value):
        return None
    if not trial.user_attrs.get(FEASIBLE_ATTR, False):
        return None
    generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(generation_id, str) or not generation_id:
        return None
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    return generation_id, attempt_id


def _trial_objective_provenance(trial: optuna.trial.FrozenTrial) -> Mapping[str, Any] | None:
    """Decode a trial's frozen objective-evidence provenance record.

    :param optuna.trial.FrozenTrial trial: Trial whose provenance attr is read.
    :return Mapping[str, Any] | None: The parsed record, or ``None`` when the
        trial predates the record (review v0.5.17 / finding F) or stored
        something this build cannot parse as a mapping.
    """
    raw = trial.user_attrs.get(OBJECTIVE_PROVENANCE_ATTR)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _verify_objective_source_evidence(
    trial_dir: Path,
    provenance: Mapping[str, Any] | None,
    *,
    subject: str,
    verify_digest: bool,
) -> None:
    """Require a trial's frozen objective source to still be on disk as recorded.

    A missing or absent provenance record is tolerated: trials persisted before
    the record existed simply have no source to locate (see
    :class:`phasesweep.engine.selection.SelectedTrial`). A ``wandb`` source is
    tolerated too - it names a remote run, not a file in this tree.

    :param Path trial_dir: Structurally translated directory for the trial.
    :param Mapping[str, Any] | None provenance: Parsed objective provenance.
    :param str subject: Caller-built label naming the study, phase, and trial.
    :param bool verify_digest: Also re-hash the source and compare it against
        the recorded ``sha256``. Reserved for the winner (see
        :func:`_verify_winner_objective_evidence`).
    :raises TrialEvidenceMissingError: The recorded file source is missing,
        unreadable, or no longer the bytes the published scalar came from.
    """
    if provenance is None:
        return
    source = provenance.get("source")
    if not isinstance(source, Mapping) or source.get("kind") != "file":
        return
    raw_path = source.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        # A file source always records its path; a record that does not is not
        # a shape any PhaseSweep build writes, so there is nothing to locate.
        return
    candidate = Path(raw_path)
    source_path = candidate if candidate.is_absolute() else trial_dir / candidate
    try:
        stat_result = source_path.stat()
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r}, which is missing or "
            f"unreadable at {str(source_path)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        ) from exc
    recorded_size = source.get("size_bytes")
    size_recorded = isinstance(recorded_size, int) and not isinstance(recorded_size, bool)
    if size_recorded and stat_result.st_size != recorded_size:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r} as "
            f"{recorded_size} bytes, but that file is now {stat_result.st_size} bytes. "
            f"{_TRIAL_EVIDENCE_REMEDY}"
        )
    if not verify_digest:
        return
    recorded_digest = source.get("sha256")
    if not isinstance(recorded_digest, str) or not recorded_digest:
        return
    try:
        actual_digest = file_sha256(source_path)
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r}, which could not be "
            f"read back for verification at {str(source_path)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        ) from exc
    if actual_digest != recorded_digest:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r} with sha256 "
            f"{recorded_digest}, but that file now hashes to {actual_digest}: the bytes behind "
            f"the published metric have changed. {_TRIAL_EVIDENCE_REMEDY}"
        )


def _verify_trial_evidence_dir(
    trial_dir: Path,
    *,
    subject: str,
    attempt_id: str,
    provenance: Mapping[str, Any] | None,
    verify_objective_digest: bool,
) -> None:
    """Require one trial's evidence directory and audit artifacts to still exist.

    The single per-trial check shared by the launch preflight
    (:func:`_validate_selection_evidence`) and the selection-time winner check
    (:func:`_verify_winner_objective_evidence`), so the two can never drift
    apart on what "this trial's evidence is intact" means.

    ``trial_dir`` is always the *structurally translated* directory - the
    current config's phase directory plus the trial's own directory name -
    never the absolute path a trial persisted, which a relocated or rebound
    tree leaves pointing at the old root (same principle as
    :func:`_validate_relocated_trial_evidence`).

    A trial with no ``attempt_lifecycle.json`` is tolerated (legacy
    pre-lifecycle attempts), and a present record's ``state`` is deliberately
    not constrained: the transition to ``exited`` is documented best-effort, so
    an ``allocated`` record is an ordinary outcome for a trial that completed.
    Only a malformed or foreign record fails.

    :param Path trial_dir: Translated evidence directory for the trial.
    :param str subject: Caller-built label naming the study, phase, and trial.
    :param str attempt_id: Attempt identity the lifecycle record must belong to.
    :param Mapping[str, Any] | None provenance: Parsed objective provenance.
    :param bool verify_objective_digest: Re-hash the objective source as well.
    :raises TrialEvidenceMissingError: The directory, an audit artifact, or the
        recorded objective source is missing, foreign, or altered.
    """
    if not trial_dir.is_dir():
        raise TrialEvidenceMissingError(
            f"{subject} has no evidence directory at {str(trial_dir)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        )
    try:
        read_attempt_lifecycle(trial_dir, expected_attempt_id=attempt_id)
    except ValueError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} has an attempt lifecycle record that is malformed or belongs to "
            f"another attempt ({exc}). {_TRIAL_EVIDENCE_REMEDY}"
        ) from exc
    for filename in _REQUIRED_TRIAL_EVIDENCE_FILES:
        if not (trial_dir / filename).is_file():
            raise TrialEvidenceMissingError(
                f"{subject} is missing its {filename!r} audit artifact under "
                f"{str(trial_dir)!r}. {_TRIAL_EVIDENCE_REMEDY}"
            )
    _verify_objective_source_evidence(
        trial_dir,
        provenance,
        subject=subject,
        verify_digest=verify_objective_digest,
    )


def _validate_selection_evidence(
    experiment: Experiment,
    studies: Mapping[str, optuna.Study],
) -> None:
    """Require every trial that could win to still have its evidence on disk.

    Winner selection consults Optuna alone - state, value, feasibility,
    execution ids, constraint readings - and never touches the filesystem, so
    an experiment whose winning trial directory was deleted happily reselects
    that trial and republishes its number, metric, and provenance from a tree
    holding nothing behind them (PR #5 review / reviewer 2, blocker 7). This
    runs on the launch path before any generation claim or trial work, so a
    tree that cannot honestly be ranked is refused before it is added to.

    Only selection-eligible trials are candidates
    (:func:`_selection_candidate_identity`); the rest can never win, so passing
    over them cannot bias the result. Candidates are checked for existence and
    identity only - no objective digest - because the default objective source
    is the trainer's unbounded ``stdout.log``, and re-hashing every candidate's
    log on every top-up would cost the whole study's log volume per resume. The
    trial that actually becomes a winner is digest-verified at selection time
    instead (:func:`_verify_winner_objective_evidence`).

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param Mapping[str, optuna.Study] studies: Existing phase studies keyed by
        phase name, as returned by :func:`_preflight_existing_studies`.
    :raises TrialEvidenceMissingError: A selection-eligible trial records an
        unusable trial directory, or its evidence directory, audit artifacts,
        or recorded objective source are no longer in this tree.
    """
    for phase_name, study in studies.items():
        phase_dir = _phase_dir(experiment, phase_name)
        for trial in study.get_trials(deepcopy=False):
            identity = _selection_candidate_identity(trial)
            if identity is None:
                continue
            generation_id, attempt_id = identity
            subject = (
                f"Study {study.study_name!r} phase {phase_name!r} trial {trial.number} "
                "is eligible to win selection but"
            )
            stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
            if not isinstance(stored, str) or not stored or not Path(stored).is_absolute():
                raise TrialEvidenceMissingError(
                    f"{subject} records an invalid {TRIAL_DIR_ATTR!r} user attribute "
                    f"{stored!r}, so its evidence cannot be located. {_TRIAL_EVIDENCE_REMEDY}"
                )
            # Identity binding: the persisted directory name must be exactly the
            # one _trial_dir_for builds for this trial's number, generation, and
            # attempt. A name that disagrees means the study record and the
            # directory are not describing the same execution.
            expected_name = _trial_dir_for(
                experiment,
                phase_name,
                trial.number,
                generation_id=generation_id,
                attempt_id=attempt_id,
            ).name
            if Path(stored).name != expected_name:
                raise TrialEvidenceMissingError(
                    f"{subject} records evidence directory {Path(stored).name!r}, which does "
                    f"not name this trial's own generation/attempt identity (expected "
                    f"{expected_name!r}). {_TRIAL_EVIDENCE_REMEDY}"
                )
            _verify_trial_evidence_dir(
                phase_dir / expected_name,
                subject=subject,
                attempt_id=attempt_id,
                provenance=_trial_objective_provenance(trial),
                verify_objective_digest=False,
            )


def _verify_winner_objective_evidence(
    experiment: Experiment,
    phase_name: str,
    selected: SelectedTrial,
) -> None:
    """Digest-verify the evidence behind a trial that is about to be published.

    The launch preflight proves every candidate's evidence *exists*; this proves
    the one trial that actually won is still byte-for-byte the evidence its
    frozen provenance recorded. The split is deliberate: the default objective
    source is an uncapped trainer log, so hashing every candidate on every
    top-up is O(total trainer log bytes) per resume - potentially tens of
    gigabytes - while hashing only the published winner is bounded by one
    trial's log and still catches every deletion and every result-affecting
    edit, including a tamper that preserves byte length (PR #5 review /
    reviewer 2, blocker 7).

    It runs on every selection, so a deadline-truncated partial publication is
    covered on the same terms as a complete one.

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param str phase_name: Phase whose winner was just selected.
    :param SelectedTrial selected: The winning trial and its frozen provenance.
    :raises TrialEvidenceMissingError: The winner's evidence directory, audit
        artifacts, or objective source are missing, foreign, or altered.
    """
    trial_dir = _trial_dir_for(
        experiment,
        phase_name,
        selected.trial_number,
        generation_id=selected.generation_id,
        attempt_id=selected.attempt_id,
    )
    _verify_trial_evidence_dir(
        trial_dir,
        subject=(
            f"Phase {phase_name!r} winner trial {selected.trial_number} "
            f"(generation {selected.generation_id!r}, attempt {selected.attempt_id!r})"
        ),
        attempt_id=selected.attempt_id,
        provenance=selected.objective_provenance,
        verify_objective_digest=True,
    )


def _preflight_existing_studies(
    experiment: Experiment,
    *,
    cleanup_report: _PreflightCleanupReport | None = None,
    from_phase: str | None = None,
) -> dict[str, optuna.Study]:
    """Validate and reap every existing declared phase study before launch.

    :param Experiment experiment: Parsed experiment whose declared phases are inspected.
    :param _PreflightCleanupReport | None cleanup_report: Optional shared report to
        accumulate cleanup evidence into; a fresh one is created if omitted.
    :param str | None from_phase: Optional resume point. Recovery and schema checks
        still cover every phase; trial-target validation starts at this reached phase.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name (phases
        with no durable study yet are omitted).
    :raises ArtifactRootConflictError: A phase's persistent study is already
        bound to a different artifact root than this config's workdir offers;
        raised before any inspection, reaping, or trial work.
    :raises LegacyArtifactRootMigrationRequiredError: A populated phase study
        predates artifact-root binding, so which workdir owns its evidence
        cannot be inferred; raised on the same terms.
    :raises StudyStorageUnavailableError: A phase's persistent storage could not
        be inspected; raised before any claim, reaping, or registry recovery.
    :raises StudySchemaMismatchError: A phase's study uses an incompatible
        storage schema.
    :raises TrialTargetRegressionError: A phase's study already accepted a
        higher trial target than the current config requests.
    :raises ProcessCleanupUncertainError: Stale-trial cleanup could not be
        confirmed safe for a phase's study.
    :raises RuntimeError: Multiple studies failed preflight for mixed reasons
        not covered by a single common exception type.
    """
    _warn_unbounded_environment_inheritance(experiment)
    report = cleanup_report or _PreflightCleanupReport()
    # Discovery, root checks, and claims happen in ONE strict pass, and its
    # study objects are the ones every later step operates on: an invocation
    # offering a second publication root must not reap, inspect, or claim
    # anything in either tree (review v0.5.19 / finding F5), and a storage
    # read that fails must abort rather than let a second, luckier read hand
    # recovery a study whose root was never checked (PR #5 review /
    # reviewer 2, issue 1). The storage error still marks cleanup uncertain:
    # an unreadable ledger cannot prove its attempts are resolved.
    try:
        loaded = _load_and_check_artifact_roots(experiment)
    except StudyStorageUnavailableError as exc:
        report.mark_uncertain(exc)
        raise
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
    reached = from_phase is None
    for phase in experiment.phases:
        if phase.name == from_phase:
            reached = True
        study = loaded.get(phase.name)
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
            _validate_study_direction(study, experiment.metric.goal)
            _validate_study_schema(study)
            if reached:
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
    :raises RuntimeError: The ledger could not be written; MCP cleanup
        uncertainty stays set rather than being cleared without consuming the
        trial evidence.
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


def _previously_recovered_uncertain_trial_count(study: optuna.Study, generation_id: str) -> int:
    """Count this generation's cleanup-uncertain trials a prior recovery pass consumed.

    Recovery durably records each confirmed trial in the study-level ledger
    before the CLI can persist its run-level recovery record or clear the
    cleanup-uncertainty marker. A crash in that window must not erase the
    evidence: the retry skips these trials as already recovered, and without
    this count the trial-level-evidence guard would refuse to clear cleanup
    uncertainty forever (review v0.5.17 gap hunt).

    The count is scoped to ``generation_id`` — detached MCP runs use their run
    id as the generation id — so a *different* run's already-consumed evidence
    cannot clear this run's uncertainty; that cross-run refusal stays
    fail-closed.

    :param optuna.Study study: Study whose ledger and trials are inspected.
    :param str generation_id: Generation identity of the run being recovered.
    :return int: This generation's terminal trials that record cleanup
        uncertainty and appear in the durable recovery ledger.
    """
    recovered = _cleanup_recovered_trial_numbers(study)
    if not recovered:
        return 0
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if (
            trial.state.is_finished()
            and trial.number in recovered
            and trial.user_attrs.get(CLEANUP_CONFIRMED_ATTR) is False
            and trial.user_attrs.get(GENERATION_ID_ATTR) == generation_id
        ):
            count += 1
    return count


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


def _iter_cleanup_uncertain_trials(
    study: optuna.Study,
) -> Iterator[tuple[optuna.trial.FrozenTrial, Path, StaleProcessIdentity]]:
    """Yield unconsumed terminal trials with their validated process identity.

    :param optuna.Study study: Study whose cleanup evidence should be inspected.
    :return Iterator: Eligible trial, persisted trial directory, and process identity.
    """
    recovered_trial_numbers = _cleanup_recovered_trial_numbers(study)
    for trial in study.get_trials(deepcopy=False):
        if (
            trial.state.is_finished()
            and trial.number not in recovered_trial_numbers
            and trial.user_attrs.get(CLEANUP_CONFIRMED_ATTR) is False
        ):
            trial_dir = _trial_dir_for_cleanup_recovery(trial, study.study_name)
            identity = _read_trial_process_identity(trial, trial_dir, study.study_name)
            yield trial, trial_dir, identity


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
    for trial, trial_dir, identity in _iter_cleanup_uncertain_trials(study):
        safe_to_clear = cleanup_stale_trial_process(identity)
        if not safe_to_clear:
            raise ProcessCleanupUncertainError(
                f"Refusing to clear cleanup uncertainty for trial {trial.number} in "
                f"study {study.study_name}: process cleanup could not be confirmed. "
                f"experiment={experiment.experiment} phase={phase_name} "
                f"trial_dir={trial_dir} pid={identity.pid} pgid={identity.pgid}."
            )
        _record_cleanup_recovery(study, trial)
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
    return sum(1 for _ in _iter_cleanup_uncertain_trials(study))
