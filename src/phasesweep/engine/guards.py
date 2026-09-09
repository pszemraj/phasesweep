"""Engine locks, fingerprints, and stale-trial recovery guards."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
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
    PhaseSweepError,
    PublicationAccessError,
    PublicationIntegrityError,
    PublishedStudyMissingError,
    SamplerContinuationUnsupportedError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialEvidenceMissingError,
    TrialTargetRegressionError,
)
from phasesweep.engine.optuna import (
    _load_existing_phase_study,
    _load_journal_study_snapshot,
    _published_phase_trial_refs,
    _published_trial_matches,
)
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    PHASE_FINGERPRINT_ATTR,
    PHASE_RECOVERY_ATTR,
    PHASE_RECOVERY_SCHEMA_VERSION,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    TRAINER_INPUT_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    TRIAL_TARGET_ATTR,
    Winner,
    _artifact_root_binding_path,
    _attempts_dir,
    _experiment_dir,
    _last_successful_generation_id,
    _last_successful_suite_generation_path,
    _phase_dir,
    _resolve_publication_pointer,
    _suite_dir,
    _trial_dir_for,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError, _environment_identity
from phasesweep.runtime.files import (
    PlatformCapabilityError,
    UnsafePrivatePathError,
    atomic_write_text,
    canonical_storage_identity,
    exclusive_lock,
    file_sha256,
    open_directory_fd,
    private_atomic_write_text,
    read_private_text_at,
    storage_backend,
    storage_is_in_memory,
    storage_recovery_locator,
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

_TRIAL_OUTCOMES = frozenset({"success", "failure", "pruned", "cancelled", "fatal"})


@dataclass(frozen=True)
class _PhasePolicyState:
    """Failure-policy state reconstructed from durable per-trial outcomes."""

    max_sequence: int
    consecutive_failures: int
    recovered_abort_sequence: int | None
    fatal_trial_number: int | None
    fatal_sequence: int | None
    fatal_cause: str | None
    fatal_policy: str | None


@dataclass(frozen=True)
class _ParsedTrialOutcome:
    """Validated durable terminal-outcome fields for one trial."""

    sequence: int
    outcome: str
    cause: str | None
    policy: str | None


@dataclass(frozen=True)
class _AcceptedPartialDecision:
    """Durable terminal decision to select from an incomplete timed-out phase."""

    trial_target: int
    outcome_sequence: int
    finished_trials: int
    completed_trials: int
    timeout_scope: str
    recovered_abort_sequence: int | None


def _load_accepted_partial_decision(
    study: optuna.Study,
) -> _AcceptedPartialDecision | None:
    """Load and validate a persisted accepted-partial timeout decision.

    :param optuna.Study study: Study whose terminal phase decision is inspected.
    :raises StudySchemaMismatchError: The stored decision has an unsupported or
        internally inconsistent shape.
    :return _AcceptedPartialDecision | None: Validated decision, or ``None``
        when the study has no accepted-partial decision.
    """
    raw = study.user_attrs.get(PHASE_DECISION_ATTR)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has malformed {PHASE_DECISION_ATTR!r}={raw!r}. "
            "Use a new experiment name, or archive/delete the inconsistent study."
        )
    trial_target = raw.get("trial_target")
    outcome_sequence = raw.get("outcome_sequence")
    finished_trials = raw.get("finished_trials")
    completed_trials = raw.get("completed_trials")
    timeout_scope = raw.get("timeout_scope")
    recovered_abort_sequence = raw.get("recovered_abort_sequence")
    if (
        raw.get("schema_version") != PHASE_DECISION_SCHEMA_VERSION
        or raw.get("decision") != "accepted_partial_timeout"
        or type(trial_target) is not int
        or trial_target < 1
        or type(outcome_sequence) is not int
        or outcome_sequence < 0
        or type(finished_trials) is not int
        or finished_trials < 0
        or finished_trials >= trial_target
        or type(completed_trials) is not int
        or completed_trials < 0
        or completed_trials > finished_trials
        or timeout_scope not in {"phase", "run"}
        or (
            recovered_abort_sequence is not None
            and (
                type(recovered_abort_sequence) is not int
                or recovered_abort_sequence < 1
                or recovered_abort_sequence > outcome_sequence
            )
        )
    ):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has malformed {PHASE_DECISION_ATTR!r} fields: "
            f"{raw!r}. Use a new experiment name, or archive/delete the inconsistent study."
        )
    return _AcceptedPartialDecision(
        trial_target=trial_target,
        outcome_sequence=outcome_sequence,
        finished_trials=finished_trials,
        completed_trials=completed_trials,
        timeout_scope=timeout_scope,
        recovered_abort_sequence=recovered_abort_sequence,
    )


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


def _lock_digest(material: dict[str, Any]) -> str:
    """Hash a lock-material dict into a 24-char hex digest.

    :param dict[str, Any] material: Lock identity material, represented as a
        JSON-serializable mapping.
    :return str: First 24 hex characters of the canonical JSON's SHA-256.
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

    :param str name: Human-readable prefix derived from the lock identity.
    :param dict[str, str] material: Identity material hashed into the filename.
    :param str label: Short lock role such as ``"output"`` or ``"storage"``.
    :return Path: Absolute lock path; the lock file is created lazily.
    """
    return _lock_dir() / f"{name}__{label}__{_lock_digest(material)}.lock"


def _output_lock_material(experiment: Experiment) -> dict[str, str]:
    """Identity for the *output namespace* lock: which directory we write to.

    Catches the case where two configs share a workdir + experiment name but
    point at different storage backends — without an output lock those would
    silently overwrite each other's ``trial_*/``, ``winner.yaml``, and
    ``summary.yaml`` (review v0.5.6 / blocker 1). Always taken regardless of
    storage backend, including in-memory storage.

    :param Experiment experiment: Experiment supplying the output namespace.
    :return dict[str, str]: Lock material keyed on the resolved experiment directory.
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

    :param Experiment experiment: Experiment supplying storage and study namespace.
    :return dict[str, str] | None: Canonical persistent-storage lock material,
        or ``None`` for in-memory storage.
    """
    if storage_is_in_memory(experiment.resolved_storage):
        # In-memory URLs (``sqlite:///:memory:`` and spellings thereof) have
        # no shared backend to guard; locking their canonical identity would
        # make two unrelated in-memory runs that share an experiment name
        # contend on a lock naming a backend that does not exist (review
        # v0.5.17 gap hunt).
        return None
    storage_identity = canonical_storage_identity(experiment.resolved_storage)
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

    :param Experiment experiment: Experiment whose lock set is derived.
    :return list[Path]: Lock paths in deterministic string order: one output
        lock for in-memory storage, otherwise output and storage locks.
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

    :param Experiment experiment: Experiment whose complete lock set is acquired.
    :raises ExperimentLockBusyError: Another process holds the output or storage lock.
    :return Iterator[None]: Context manager iterator for the held lock set.
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
# v5 / v4 / v4: the embedded complete trainer_config is now part of the core
# experiment contract and contributes to experiment, suite, and phase identity.
# v4 / v3 / v3: an omitted execution.cwd contributes its effective
# invocation directory instead of an unbound null. The earlier execution
# contract work covered only configured cwd values, which still let identical
# persistent-study identities launch different relative commands from two
# invocation directories. whole_node phases additionally fingerprint their
# configured device-set size — the trainer's world size.
# Existing populated studies from earlier schemas fail the fingerprint check
# on resume; see docs/config.md's upgrade section.
FINGERPRINT_SCHEMA_VERSION = 5
SUITE_FINGERPRINT_SCHEMA_VERSION = 4
EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 4


def _execution_identity(experiment: Experiment) -> dict[str, Any]:
    """Return the execution contract's contribution to semantic fingerprints.

    The trainer's working directory and ambient-environment inheritance are
    semantic inputs: two invocations differing in either can evaluate
    different code or data under one study (review v0.5.17 / blocker 4). A
    configured cwd contributes its RESOLVED path — a relative cwd invoked
    from two directories is two different execution contexts and must not
    share a study. An unconfigured cwd contributes the resolved invocation
    directory because that is where the trainer actually runs. Ambient
    Ambient variable *values* are enforced through the trial environment
    cohort rather than embedded in this config digest. Pass-through names and
    their classification are config semantics, but their credential values
    may rotate. Put fixed semantic values in ``env``, which is fingerprinted.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return dict[str, Any]: JSON-serialisable execution-identity payload.
    """
    contract = experiment.execution.inherit_env
    identity = {
        "cwd": str(
            Path(experiment.execution.cwd).expanduser().resolve()
            if experiment.execution.cwd is not None
            else Path.cwd().resolve()
        ),
        "inherit_env": sorted(contract) if isinstance(contract, list) else contract,
    }
    if experiment.execution.passthrough_env:
        identity["passthrough_env"] = sorted(experiment.execution.passthrough_env)
    return identity


def _semantic_payload_digest(payload: object) -> str:
    """Return the canonical digest used for persisted experiment semantics.

    :param object payload: JSON-compatible semantic identity payload.
    :return str: SHA-256 of the canonical compact JSON representation.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        "trainer_config": experiment.trainer_config,
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
    return _semantic_payload_digest(payload)


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

    :param Suite suite: Suite whose study names, dependency edges, promotion
        rules, and resolved experiments contribute to the digest.
    :return str: SHA-256 of the canonical suite payload.
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

    :param Experiment experiment: Experiment supplying command, input format,
        environment, metric, constraints, and provenance.
    :param Phase phase: Phase whose run-control keys are excluded.
    :param dict[str, Winner] inherited_winners: Parent winners whose effective
        overrides contribute to identity.
    :return dict[str, Any]: JSON-serializable configured trial semantics.
    """
    semantic_phase = _semantic_phase_dump(phase)
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trial_command": experiment.trial_command,
        "trainer_config": experiment.trainer_config,
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

    :param Experiment experiment: Experiment forwarded to
        :func:`_phase_semantic_payload`.
    :param Phase phase: Phase being fingerprinted.
    :param dict[str, Winner] inherited_winners: Parent winners contributing
        effective overrides to identity.
    :return str: SHA-256 of the canonical semantic payload.
    """
    payload = _phase_semantic_payload(experiment, phase, inherited_winners)
    return _semantic_payload_digest(payload)


def _verify_fingerprint(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Stamp a fresh study with its fingerprint or fail on mismatch.

    :param optuna.Study study: Study being verified or stamped.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase whose fingerprint must match.
    :param dict[str, Winner] inherited_winners: Parent winners contributing to identity.
    :raises StudyFingerprintMismatchError: The populated study has an incompatible
        persisted fingerprint.
    :return str: Verified current fingerprint.
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
            "for. Restore write access to the experiment workdir, then run again."
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
    except (OSError, PlatformCapabilityError, UnsafePrivatePathError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry {attempts_dir} is not a real owner-only directory. "
            "Recovery cannot trust process authority reached through a symlink or "
            "shared path. Restore the original registry with mode 0700 before retrying."
        ) from exc
    try:
        try:
            names = sorted(name for name in os.listdir(directory_fd) if name.endswith(".json"))
        except OSError as exc:
            raise ProcessCleanupUncertainError(
                f"Attempt registry {attempts_dir} cannot be enumerated safely."
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
    except (OSError, PlatformCapabilityError, UnsafePrivatePathError, ValueError) as exc:
        raise ProcessCleanupUncertainError(
            f"Attempt registry entry {entry_path} is unreadable or malformed. "
            "Recovery cannot prove whether a process from this attempt is still "
            "alive. Investigate the attempt's trial directory, then delete the "
            "entry file if you are certain nothing is running."
        ) from exc
    locator_identity: str | None = None
    if isinstance(payload, dict) and isinstance(payload.get("storage_locator"), str):
        try:
            locator_identity = canonical_storage_identity(payload["storage_locator"])
        except ValueError as exc:
            raise ProcessCleanupUncertainError(
                f"Attempt registry entry {entry_path} has an unsupported or partial "
                "schema. Delete the entry file only if you are certain no process "
                "from this attempt is running."
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
            "from this attempt is running."
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
    resolution = _attempt_process_resolution(
        lifecycle,
        identity_exists=(trial_dir / PROCESS_IDENTITY_FILE).exists(),
    )
    if resolution == "exited":
        return
    if resolution == "allocated":
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


def _parsed_trial_outcome(value: Any) -> _ParsedTrialOutcome | None:
    """Return the ordered outcome fields when a trial attr is well formed.

    :param Any value: Raw ``TRIAL_OUTCOME_ATTR`` user-attr value to validate.
    :return _ParsedTrialOutcome | None: Validated outcome, including the fatal
        policy needed to distinguish cleanup uncertainty from ordinary errors;
        ``None`` when the payload is malformed.
    """
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schema_version")
    sequence = value.get("sequence")
    outcome = value.get("outcome")
    cause = value.get("cause")
    policy = value.get("policy")
    if (
        schema_version != TRIAL_OUTCOME_SCHEMA_VERSION
        or type(sequence) is not int
        or sequence < 1
        or outcome not in _TRIAL_OUTCOMES
        or (cause is not None and not isinstance(cause, str))
        or (
            policy is not None and (outcome != "fatal" or not isinstance(policy, str) or not policy)
        )
    ):
        return None
    return _ParsedTrialOutcome(
        sequence=sequence,
        outcome=outcome,
        cause=cause,
        policy=policy,
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
    orchestrator died. Preserve a recorded fatal exception or explicit host
    cancellation; convert a not-yet-committed success/prune to failure because
    recovery is about to commit the trial as ``FAIL``. Malformed or duplicate
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
    if existing is not None and used_sequences.count(existing.sequence) == 1:
        sequence = existing.sequence
        outcome = existing.outcome
        cause = existing.cause
        policy = existing.policy
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
    :raises StudySchemaMismatchError: The registry and RUNNING trial record
        conflicting generation identities.
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
    from phasesweep.engine.optuna import _resolve_storage

    captured_trial = None
    try:
        journal = storage_backend(storage_url) == "journal"
        if journal:
            snapshot = _load_journal_study_snapshot(storage_url, entry["study_name"])
            if snapshot is None:
                return "terminal"
            captured_trial = next(
                (
                    trial
                    for trial in snapshot.get_trials(deepcopy=False)
                    if trial.number == entry["trial_number"]
                ),
                None,
            )
            if captured_trial is None:
                return "terminal"
        try:
            study = optuna.load_study(
                study_name=entry["study_name"],
                storage=_resolve_storage(storage_url),
            )
        except KeyError:
            if journal:
                # The captured journal contained the study. A later replay or
                # lookup failure is uncertainty, not proof that recovery is done.
                raise
            return "terminal"
        trials = study.get_trials(deepcopy=False)
    except Exception:  # noqa: BLE001 - unreachable storage keeps the entry for retry
        log.warning(
            "Attempt registry entry %s references storage that cannot be "
            "read right now; the stale trial will be retried on a later run.",
            entry_path,
        )
        return "unreachable"
    trial = next((t for t in trials if t.number == entry["trial_number"]), None)
    if captured_trial is not None and (
        trial is None
        or trial._trial_id != captured_trial._trial_id
        or any(
            captured_trial.user_attrs.get(key) is not None
            and trial.user_attrs.get(key) != captured_trial.user_attrs[key]
            for key in (GENERATION_ID_ATTR, ATTEMPT_ID_ATTR)
        )
    ):
        log.warning(
            "Attempt registry entry %s changed in storage during recovery inspection; "
            "retaining the entry for a later retry.",
            entry_path,
        )
        return "unreachable"
    if trial is None:
        return "terminal"
    if trial.state != optuna.trial.TrialState.RUNNING:
        if (
            trial.state.is_finished()
            and trial.number in _cleanup_recovered_trial_numbers(study)
            and trial.user_attrs.get(ATTEMPT_ID_ATTR) == entry["attempt_id"]
            and trial.user_attrs.get(GENERATION_ID_ATTR) == entry["generation_id"]
        ):
            return "recovered"
        if trial.state.is_finished() and _trial_requires_cleanup_recovery(trial):
            _record_cleanup_recovery(study, trial)
            return "recovered"
        return "terminal"
    stored_attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if stored_attempt_id is not None and stored_attempt_id != entry["attempt_id"]:
        # A conflicting durable id proves the RUNNING row belongs to another
        # attempt. A missing id is different: registration is written before
        # the first Optuna attr, so it is the expected crash/storage-failure
        # window. The registry entry plus the already-validated lifecycle or
        # process identity still binds this exact study and trial safely.
        return "terminal"
    stored_generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    if stored_generation_id is not None and stored_generation_id != entry["generation_id"]:
        raise StudySchemaMismatchError(
            f"Attempt registry entry {entry_path} identifies generation "
            f"{entry['generation_id']!r}, but its RUNNING trial {trial.number} in study "
            f"{entry['study_name']!r} records {stored_generation_id!r}. Refusing to "
            "overwrite conflicting recovery identity."
        )
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
    _record_stale_trial_failure(study, trial)
    _record_cleanup_recovery(study, trial)
    try:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    except Exception as exc:
        raise StudyStorageUnavailableError(
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
    *,
    retain_recovery_evidence: bool = False,
) -> dict[str, str]:
    """Resolve every registered nonterminal attempt before any launch.

    Runs before the per-phase study loop and is deliberately independent of
    the current phase graph: a stale attempt whose phase was renamed or
    removed, or whose storage URL changed, is still discovered, its process
    group verified/cleaned, and its recorded study repaired (review v0.5.17 /
    blocker 3).

    :param Experiment experiment: Parsed experiment whose registry is scanned.
    :param _PreflightCleanupReport report: Shared cleanup-evidence collector.
    :param bool retain_recovery_evidence: Keep entries backed by uncommitted
        cleanup evidence so interrupted operator recovery can retry even when
        the attempt's recorded storage differs from the current config.
    :return dict[str, str]: Inspected attempt ids mapped to their producing
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
                _registry_attempt_process_is_resolved(entry, entry_path)
            except ProcessCleanupUncertainError as exc:
                report.uncertain_attempt_ids.add(attempt_id)
                report.mark_uncertain(exc)
                raise
            outcome = _registry_attempt_fail_stale_trial(
                entry,
                entry_path,
                current_storage=experiment.resolved_storage,
            )
            if outcome in {"reaped", "recovered"}:
                report.recovered_attempt_ids.add(attempt_id)
                report.recovered_attempt_generations[attempt_id] = entry["generation_id"]
            if outcome != "unreachable" and not (
                retain_recovery_evidence and outcome in {"reaped", "recovered"}
            ):
                with contextlib.suppress(OSError):
                    os.unlink(entry_path.name, dir_fd=directory_fd)
    return inspected


def _inspect_active_attempts(experiment: Experiment) -> dict[str, str]:
    """Validate registered attempts without signalling or changing state.

    This is the observational half of ``mcp recover-run``. It scans the same
    phase-independent registry as normal preflight, so renamed phases and
    unavailable current storage cannot hide a trainer from the dry run.

    :param Experiment experiment: Parsed experiment whose registry is scanned.
    :return dict[str, str]: Attempt ids a confirmed recovery would reconcile,
        mapped to their producing generation ids.
    :raises ProcessCleanupUncertainError: An entry lacks safe recovery evidence.
    """
    inspected: dict[str, str] = {}
    with _open_attempt_registry(experiment) as (directory_fd, entry_paths):
        if directory_fd is None:
            return inspected
        for entry_path in entry_paths:
            entry = _load_attempt_entry(entry_path, directory_fd=directory_fd)
            attempt_id = entry["attempt_id"]
            _registry_attempt_process_is_resolved(entry, entry_path, inspect_only=True)
            inspected[attempt_id] = entry["generation_id"]
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
       launch was ever attempted (the worker died queued for a GPU). The
       launcher advances to ``launching`` before Popen. Safe to fail.
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
    resolution = _attempt_process_resolution(
        lifecycle,
        identity_exists=(trial_dir / PROCESS_IDENTITY_FILE).exists(),
    )
    if resolution == "exited":
        assert lifecycle is not None
        log.warning(
            "Trial %d in study %s exited (rc=%s) before its terminal state was "
            "committed; failing it without signalling.",
            trial.number,
            study_name,
            lifecycle.return_code,
        )
        return
    if resolution == "allocated":
        # A missing identity is exactly what 'allocated' predicts: the worker
        # died queued (e.g. waiting for a GPU) before launch began. The
        # launcher durably advances to 'launching' before Popen; that state and
        # a present-but-unreadable identity both fall through to the strict
        # reader below and fail closed.
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


def _collect_attempt_generation(
    trial: optuna.trial.FrozenTrial,
    phase_name: str,
    attempt_ids: set[str] | None,
    attempt_generations: dict[str, str] | None,
    attempt_locations: dict[str, tuple[str, int, str]] | None,
) -> None:
    """Collect one trial's durable attempt, generation, and study-local locator.

    :param optuna.trial.FrozenTrial trial: Trial whose durable identity is collected.
    :param str phase_name: Phase that owns ``trial``.
    :param set[str] | None attempt_ids: Optional destination for known attempt IDs.
    :param dict[str, str] | None attempt_generations: Optional attempt-to-generation map.
    :param dict[str, tuple[str, int, str]] | None attempt_locations: Optional
        attempt-to-phase/trial/generation locator map.
    """
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(attempt_id, str) or not attempt_id:
        return
    if attempt_ids is not None:
        attempt_ids.add(attempt_id)
    generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    if not isinstance(generation_id, str) or not generation_id:
        return
    if attempt_generations is not None:
        attempt_generations[attempt_id] = generation_id
    if attempt_locations is not None:
        attempt_locations[attempt_id] = (phase_name, trial.number, generation_id)


def _reap_stale_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
    uncertain_attempt_ids: set[str] | None = None,
) -> int:
    """Mark RUNNING trials as FAIL after killing orphaned process groups.

    :param optuna.Study study: Study whose stale RUNNING trials should be reaped.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for exact
        attempt identities whose durable state was changed to FAIL.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each recovered attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :param set[str] | None uncertain_attempt_ids: Optional collector for exact
        attempt identities whose cleanup could not be proven.
    :return int: Number of stale RUNNING trials marked as failed.
    :raises ProcessCleanupUncertainError: The study's trials cannot be
        inspected, or a stale trial's directory, attempt identity, or process
        cleanup could not be proven safe.
    :raises StudyStorageUnavailableError: Cleanup succeeded but Optuna could
        not be updated to ``FAIL``; refusing to continue with an inconsistent
        study.
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

        _record_stale_trial_failure(study, trial)
        _record_cleanup_recovery(study, trial)
        try:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        except Exception as exc:
            raise StudyStorageUnavailableError(
                f"Stale process cleanup completed for RUNNING trial {trial.number}, "
                f"but Optuna state could not be updated to FAIL. Refusing to continue "
                f"with an inconsistent study. trial_dir={trial_dir}"
            ) from exc

        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )

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

    events: list[tuple[int, int, str, str | None, str | None]] = []
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
        sequence = parsed.sequence
        outcome = parsed.outcome
        cause = parsed.cause
        policy = parsed.policy
        other_trial = seen_sequences.get(sequence)
        if other_trial is not None:
            raise _phase_policy_schema_error(
                study,
                f"trials {other_trial} and {trial.number} both use completion sequence {sequence}",
            )
        seen_sequences[sequence] = trial.number
        events.append((sequence, trial.number, outcome, cause, policy))

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
    fatal_policy: str | None = None
    for sequence, trial_number, outcome, cause, policy in events:
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
            fatal_policy = policy

    return _PhasePolicyState(
        max_sequence=max_sequence,
        consecutive_failures=consecutive_failures,
        recovered_abort_sequence=recovered_abort_sequence,
        fatal_trial_number=fatal_trial_number,
        fatal_sequence=fatal_sequence,
        fatal_cause=fatal_cause,
        fatal_policy=fatal_policy,
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
    partial_decision = _load_accepted_partial_decision(study)
    if partial_decision is not None and phase.n_trials == partial_decision.trial_target:
        # An accepted partial timeout is terminal at its frozen target.
        # Identical replay launches no suggestions, so no process-local
        # sampler state needs to be reconstructed.
        return
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


def _validate_environment_cohort(study: optuna.Study, current_digest: str) -> None:
    """Refuse to allocate into a different or unknown semantic environment cohort.

    :param optuna.Study study: Existing study whose trials define the cohort.
    :param str current_digest: Semantic environment digest for this invocation.
    :raises StudySchemaMismatchError: A populated legacy study has a trial with
        no usable environment identity.
    :raises StudyFingerprintMismatchError: Recorded trials belong to another
        semantic environment cohort.
    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    missing = [
        trial.number
        for trial in trials
        if not isinstance(trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR), str)
        or not trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR)
    ]
    if missing:
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} contains populated legacy trial(s) without a "
            f"semantic trainer-environment identity: {missing}. PhaseSweep cannot guess "
            "which environment cohort owns those results. Use a new experiment name, or "
            "archive/delete the legacy study before running again."
        )
    recorded = {trial.user_attrs[TRAINER_ENV_DIGEST_ATTR] for trial in trials}
    if recorded != {current_digest}:
        rendered = ", ".join(sorted(digest[:12] for digest in recorded))
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} contains trial(s) from semantic trainer "
            f"environment cohort(s) [{rendered}], but this invocation composes "
            f"{current_digest[:12]}. No trial was allocated. Restore the original "
            "semantic environment, classify rotating credentials under "
            "execution.passthrough_env, or use a new experiment name."
        )


def _artifact_root_identity(experiment: Experiment) -> str:
    """Return the single artifact root a persistent study is allowed to publish into.

    Derived through :func:`_experiment_dir` — the same helper every artifact
    path is built from — so the recorded binding and the namespace actually
    written can never drift apart.

    :param Experiment experiment: Parsed experiment supplying workdir and name.
    :return str: Resolved ``<workdir>/<experiment>`` namespace as a string.
    """
    return str(_experiment_dir(experiment).resolve())


def _artifact_root_binding_applies(experiment: Experiment) -> bool:
    """Return whether this experiment's storage can carry an artifact-root binding.

    :param Experiment experiment: Parsed experiment whose storage is inspected.
    :return bool: ``True`` only for persistent storage. In-memory studies do
        not outlive the process, so no later invocation can inherit them and
        no second publication root can conflict with the first.
    """
    return experiment.resolved_storage is not None and not storage_is_in_memory(
        experiment.resolved_storage
    )


ARTIFACT_ROOT_BINDING_SCHEMA_VERSION = 2


def _artifact_root_storage_key(experiment: Experiment) -> str:
    """Return an opaque comparison key for one persistent storage ledger.

    The canonical identity is credential-free but may contain other target
    selectors from a query or nested connection string. The artifact tree is
    intentionally shareable, so it stores only this digest while private
    recovery state retains the operational URL.

    :param Experiment experiment: Experiment whose persistent ledger is identified.
    :return str: Full SHA-256 hex digest of the canonical storage identity.
    """
    storage_identity = canonical_storage_identity(experiment.resolved_storage)
    assert storage_identity is not None
    return hashlib.sha256(storage_identity.encode("utf-8")).hexdigest()


def _artifact_root_binding_payload(experiment: Experiment) -> dict[str, Any]:
    """Build the reverse ownership record for one persistent artifact root.

    :param Experiment experiment: Experiment whose artifact root is bound.
    :return dict[str, Any]: Versioned experiment, root, and storage identity record.
    """
    return {
        "schema_version": ARTIFACT_ROOT_BINDING_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "artifact_root": _artifact_root_identity(experiment),
        "storage_key": _artifact_root_storage_key(experiment),
    }


def _auto_storage_backend_conflict(experiment: Experiment, raw: Any) -> str | None:
    """Explain when parallelism selects the other auto-storage ledger.

    :param Experiment experiment: Config whose selected backend is being checked.
    :param Any raw: Recorded artifact-root binding.
    :return str | None: Actionable diagnostic for a backend change, otherwise ``None``.
    """
    if experiment.storage != "auto" or not isinstance(raw, dict):
        return None
    recorded_root = raw.get("artifact_root")
    if (
        raw.get("schema_version") != ARTIFACT_ROOT_BINDING_SCHEMA_VERSION
        or raw.get("experiment") != experiment.experiment
        or not isinstance(recorded_root, str)
        or not Path(recorded_root).is_absolute()
    ):
        return None
    parallel = any(phase.n_jobs > 1 for phase in experiment.phases)
    backend, previous = ("sqlite", "study.db") if parallel else ("journal", "study.journal")
    selected = "study.journal" if parallel else "study.db"
    # Reconstruct lexically: a moved tree's previous root may no longer exist.
    identity = f"{backend}:///{Path(recorded_root) / previous}"
    if raw.get("storage_key") != hashlib.sha256(identity.encode("utf-8")).hexdigest():
        return None
    return (
        f"Artifact root {_artifact_root_identity(experiment)!r} is bound to {previous}, "
        f"but storage: auto now selects {selected} because n_jobs changed between "
        "sequential and parallel execution. Restore the previous n_jobs setting to "
        "continue this tree, or use a new experiment name or workdir for the new backend. "
        "'phasesweep rebind-workdir' does not convert study.db and study.journal. "
        "No trial ran and nothing was published."
    )


def _root_durable_state_entry(experiment: Experiment) -> str | None:
    """Return the first known PhaseSweep state entry in an unbound root.

    Arbitrary operator files do not establish storage ownership. A note,
    ``.DS_Store``, or other unrelated file may already live in a chosen output
    directory before the first run, and treating it as a legacy PhaseSweep
    tree makes the experiment name unusable. Only paths PhaseSweep itself
    creates can require explicit adoption.

    :param Experiment experiment: Experiment whose artifact root is inspected.
    :raises ArtifactRootConflictError: The artifact root cannot be enumerated.
    :return str | None: Known durable entry name, or ``None`` for a fresh root.
    """
    root = _experiment_dir(experiment)
    if not root.is_dir():
        return None
    binding_name = _artifact_root_binding_path(experiment).name
    durable_names = {
        "attempts",
        "generation.yaml",
        "generations",
        "last_successful_generation.yaml",
        "summary.yaml",
        *(phase.name.casefold() for phase in experiment.phases),
    }
    staging_prefix = f".{binding_name}."
    try:
        for path in root.iterdir():
            name = path.name
            if name in {"run.log", binding_name}:
                continue
            if (
                name.startswith(staging_prefix)
                and name.endswith(".tmp")
                and path.is_file()
                and not path.is_symlink()
            ):
                token = name[len(staging_prefix) : -len(".tmp")]
                if len(token) == 16 and all(character in "0123456789abcdef" for character in token):
                    continue
            # These entries share a namespace on case-insensitive filesystems.
            if name.casefold() in durable_names:
                return name
        return None
    except OSError as exc:
        raise ArtifactRootConflictError(
            f"Cannot inspect artifact root {_artifact_root_identity(experiment)!r}: {exc}."
        ) from exc


def _validate_artifact_root_binding(
    experiment: Experiment,
    *,
    claim_fresh: bool,
    rebind: bool = False,
) -> None:
    """Validate or claim the storage ledger that owns an artifact tree.

    Study attributes bind ledger to tree. This reverse record binds tree to
    ledger, preventing a second database from combining its trial counts with
    another database's publication. A non-empty legacy tree is adopted only by
    the explicit ``rebind-workdir`` workflow.

    :param Experiment experiment: Config whose root and storage must agree.
    :param bool claim_fresh: Write the record when the root has no durable state.
    :param bool rebind: Explain empty-storage refusals for the rebind command.
    :raises ArtifactRootConflictError: The binding cannot be validated as the
        current user, or is malformed or names another owner.
    :raises LegacyArtifactRootMigrationRequiredError: A non-empty tree predates
        the reverse binding and requires explicit adoption.
    """
    path = _artifact_root_binding_path(experiment)
    if not _artifact_root_binding_applies(experiment):
        try:
            path.stat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ArtifactRootConflictError(
                f"Cannot inspect artifact-root binding {path}: {exc}."
            ) from exc
        raise ArtifactRootConflictError(
            f"Artifact root {_artifact_root_identity(experiment)!r} already records a "
            "persistent storage binding. An in-memory configuration cannot reuse this "
            "tree. Restore its persistent storage setting, or use a new experiment name "
            "or workdir for an in-memory run. Nothing was written."
        )
    expected = _artifact_root_binding_payload(experiment)
    try:
        raw = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        durable_entry = _root_durable_state_entry(experiment)
        if durable_entry is not None:
            if rebind:
                raise LegacyArtifactRootMigrationRequiredError(
                    f"Artifact root {expected['artifact_root']!r} contains legacy PhaseSweep "
                    "state, but the configured storage has no populated phase studies to "
                    "adopt. Restore the original storage setting and complete ledger for "
                    "this tree; keep the explicit storage setting when switching to "
                    "storage: auto would select an empty ledger. Rebinding does not move "
                    "or convert storage ledgers. Nothing was written."
                ) from None
            raise LegacyArtifactRootMigrationRequiredError(
                f"Artifact root {expected['artifact_root']!r} contains PhaseSweep state "
                f"entry {durable_entry!r} but records no storage-ledger binding. An "
                "ordinary run or read cannot infer "
                "which database owns this publication. Run 'phasesweep rebind-workdir "
                "<config>' with a config naming this complete tree to adopt it explicitly. "
                "No trial ran and nothing was published."
            ) from None
        if claim_fresh:
            try:
                atomic_write_text(path, json.dumps(expected, sort_keys=True) + "\n")
            except OSError as exc:
                raise ArtifactRootConflictError(
                    f"Could not bind fresh artifact root {expected['artifact_root']!r} to "
                    f"its storage ledger: {exc}. No trial ran and nothing was published."
                ) from exc
        return
    except PermissionError as exc:
        raise ArtifactRootConflictError(
            f"Artifact-root binding {path} cannot be validated as the current user "
            "(permission denied). Use the user that owns this artifact tree or restore "
            "read permission; do not run rebind-workdir to change an ownership record "
            "you could not inspect."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ArtifactRootConflictError(
            f"Artifact-root binding {path} is unreadable or malformed: {exc}. Refusing "
            "to combine this tree with an unverified storage ledger."
        ) from exc
    if raw != expected:
        backend_conflict = _auto_storage_backend_conflict(experiment, raw)
        if backend_conflict is not None:
            raise ArtifactRootConflictError(backend_conflict)
        if rebind:
            raise ArtifactRootConflictError(
                f"Artifact root {expected['artifact_root']!r} is bound to a different storage "
                "ledger or experiment. The configured storage has no populated phase "
                "studies to rebind. Restore the original experiment and storage setting "
                "and complete ledger for this tree; keep the explicit storage setting "
                "when switching to storage: auto would select an empty ledger. Rebinding "
                "does not move or convert storage ledgers. Nothing was written."
            )
        raise ArtifactRootConflictError(
            f"Artifact root {expected['artifact_root']!r} is bound to a different storage "
            f"ledger or experiment than {experiment.experiment!r}. "
            "Use the config that owns this tree, or move the complete tree and run "
            "'phasesweep rebind-workdir <config>'. No trial ran and nothing was published."
        )


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


def _load_and_check_artifact_roots(
    experiment: Experiment, *, from_phase: str | None = None
) -> dict[str, optuna.Study]:
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
    phase creates them. The exception is a phase with a winner in the current
    published generation: that publication proves the phase previously had
    trial rows with a specific generation and attempt identity. Missing or
    replaced published trials mean durable evidence was lost rather than that
    the phase is new. Skipped phases load their saved winners instead.

    :param Experiment experiment: Parsed experiment whose declared phase
        studies are loaded and bound to its resolved artifact root.
    :param str | None from_phase: Resume point; earlier phases will not execute.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name
        (phases with no durable study yet are omitted).
    :raises StudyStorageUnavailableError: A phase's persistent storage could
        not be inspected.
    :raises PublishedStudyMissingError: A phase to execute has a published
        result whose local trial identity is missing from durable storage.
    :raises LegacyArtifactRootMigrationRequiredError: A populated phase study
        records no artifact root, so which workdir owns its evidence is unknown.
    :raises ArtifactRootConflictError: A phase study is already bound to a
        different artifact root, or carries a binding that is not a string.
    """
    # Reject an existing foreign/legacy root before touching storage, including
    # an in-memory configuration offered a persistently bound root.
    _validate_artifact_root_binding(experiment, claim_fresh=False)
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
    _check_published_phase_studies(experiment, loaded, from_phase=from_phase)
    claimable = [
        study for study in loaded.values() if _artifact_root_claim_needed(study, experiment)
    ]
    # Both directions are now known-compatible. Claim the tree first, then
    # empty studies: a crash cannot leave a study pointing at a tree that does
    # not itself name the same ledger. Crucially, neither claim occurs when a
    # symlink-retargeted leaf exposed a study still bound to the old target.
    _validate_artifact_root_binding(experiment, claim_fresh=True)
    offered = _artifact_root_identity(experiment)
    for study in claimable:
        _claim_study_artifact_root(study, offered)
    return loaded


def _check_published_phase_studies(
    experiment: Experiment,
    loaded: Mapping[str, optuna.Study],
    *,
    from_phase: str | None = None,
) -> None:
    """Require the published local trial identity for phases that would execute.

    :param Experiment experiment: Experiment whose current publication is checked.
    :param Mapping[str, optuna.Study] loaded: Already-inspected persistent studies.
    :param str | None from_phase: Resume point; earlier phases only load winners.
    :raises PublishedStudyMissingError: A reached published trial is absent or replaced.
    :raises StudyStorageUnavailableError: A published study's trials cannot be read.
    """
    publication = _resolve_publication_pointer(experiment)
    published_trials = _published_phase_trial_refs(publication.summary)
    reached = from_phase is None
    for phase in experiment.phases:
        if phase.name == from_phase:
            reached = True
        if not reached or phase.name not in published_trials:
            continue
        study = loaded.get(phase.name)
        expected = published_trials[phase.name]
        missing = "is missing"
        if study is not None:
            try:
                trials = study.get_trials(deepcopy=False)
                if expected is not None and any(
                    _published_trial_matches(trial, expected) for trial in trials
                ):
                    continue
            except Exception as exc:
                raise StudyStorageUnavailableError(
                    "Could not inspect persistent study storage for published phase "
                    f"{phase.name!r}."
                ) from exc
            if not trials:
                missing = "contains no trials"
            else:
                missing = "does not contain the published trial identity"
                if expected is not None:
                    missing += (
                        f" (trial {expected.trial_number}, generation {expected.generation_id!r}, "
                        f"attempt {expected.attempt_id!r})"
                    )
                else:
                    missing += " because the publication records no complete local trial identity"
        raise PublishedStudyMissingError(
            f"Published generation {publication.generation_id!r} includes a winner for "
            f"phase {phase.name!r}, but its persistent study {missing}. That publication "
            "requires its original trial history; continuing could reuse incomplete or "
            "unrelated trials and replace the current publication. Restore the "
            "original complete storage ledger and study, or use a new experiment identity "
            "for a fresh run. Nothing was written."
        )


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
        names a trial directory that does not exist under the destination tree,
        or a selection candidate's recorded trainer input is missing or altered.
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
            if _selection_candidate_identity(trial) is not None:
                try:
                    _verify_trainer_input_evidence(
                        translated,
                        trial.user_attrs.get(TRAINER_INPUT_ATTR),
                        subject=(
                            f"Destination trial {trial.number} of study {entry.study.study_name!r}"
                        ),
                    )
                except TrialEvidenceMissingError as exc:
                    raise ArtifactRootRebindError(
                        f"Destination artifact root {str(_experiment_dir(experiment))!r} "
                        f"does not retain the recorded generated trainer input for trial "
                        f"{trial.number} of study {entry.study.study_name!r}: {exc} "
                        "Move the complete artifact tree, then rebind. Nothing was written."
                    ) from exc


def _attempt_entry_recoverable_in_place(
    entry_path: Path,
    destination: Path,
    *,
    directory_fd: int,
) -> bool:
    """Return whether a registry entry's recorded trial path lies inside this tree.

    An entry whose absolute trial directory resolves to an existing directory
    under the destination experiment tree was written *by* this tree: the
    attempt started under this exact root, so the next ordinary run here can
    follow the recorded path and resolve it (PR #5 review / reviewer 2,
    issue 2). An entry pointing anywhere else - or one that cannot be parsed -
    would be stranded by the rebind and refuses it.

    :param Path entry_path: Registry entry file to inspect.
    :param Path destination: Resolved destination experiment directory.
    :param int directory_fd: Open descriptor for the entry's validated parent directory.
    :return bool: ``True`` when the entry's recorded trial directory exists
        under ``destination``.
    """
    try:
        payload = json.loads(read_private_text_at(directory_fd, entry_path.name, entry_path))
    except (OSError, PlatformCapabilityError, UnsafePrivatePathError, ValueError):
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
    destination = _experiment_dir(experiment).resolve()
    try:
        with _open_attempt_registry(experiment) as (directory_fd, entry_paths):
            if directory_fd is None:
                return
            stranded = [
                entry_path
                for entry_path in entry_paths
                if not _attempt_entry_recoverable_in_place(
                    entry_path,
                    destination,
                    directory_fd=directory_fd,
                )
            ]
    except ProcessCleanupUncertainError as exc:
        raise ArtifactRootRebindError(
            f"Destination artifact root {str(_experiment_dir(experiment))!r} has an "
            "unsafe attempt registry. Restore its owner-only 0700 directory and 0600 "
            "entry files before rebinding."
        ) from exc
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
    except PublicationAccessError as exc:
        raise ArtifactRootRebindError(
            f"Destination artifact root {str(destination)!r} records a publication that "
            "cannot be validated as the current user (permission denied). Use the user "
            "that owns this artifact tree or restore read permission; refusing to rebind "
            "evidence that was not validated. Nothing was written."
        ) from exc
    except PublicationIntegrityError as exc:
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
            "This config uses in-memory storage and cannot rebind a persistent artifact "
            "tree. Restore the owning persistent storage setting, or use a new experiment "
            "name or workdir for an in-memory run. Nothing was written."
        )
    plans = [
        _ArtifactRootRebindPlan(
            experiment=experiment,
            destination=_artifact_root_identity(experiment),
            entries=_artifact_root_rebind_entries(experiment),
        )
        for experiment in persistent
    ]
    has_studies_to_rebind = any(
        plan.has_binding or plan.has_unbound_populated_study for plan in plans
    )
    for plan in plans:
        try:
            if has_studies_to_rebind:
                _validate_artifact_root_binding_for_rebind(plan)
            else:
                _validate_artifact_root_binding(plan.experiment, claim_fresh=False, rebind=True)
            _check_published_phase_studies(
                plan.experiment, {entry.phase_name: entry.study for entry in plan.entries}
            )
        except (
            ArtifactRootConflictError,
            PublishedStudyMissingError,
            StudyStorageUnavailableError,
        ) as exc:
            raise ArtifactRootRebindError(str(exc)) from exc
        if has_studies_to_rebind and plan.entries:
            _validate_artifact_root_destination(plan.experiment, plan.entries)
    if not has_studies_to_rebind:
        raise ArtifactRootRebindError(
            "No phase study in this storage is bound to an artifact root, and none holds a "
            "trial, so there is nothing to rebind or migrate; the next ordinary run binds "
            "these empty studies to the configured workdir. Nothing was written."
        )
    return plans


def _validate_artifact_root_binding_for_rebind(plan: _ArtifactRootRebindPlan) -> None:
    """Require an existing reverse binding to agree with the ledger being rebound.

    Auto storage may retain the identity of its database under the recorded
    previous root, provided the loaded studies agree with that source tree.

    :param _ArtifactRootRebindPlan plan: Planned binding mutation to validate.
    :raises ArtifactRootRebindError: The binding is unreadable, malformed, or
        cannot be validated as the current user.
    """
    path = _artifact_root_binding_path(plan.experiment)
    try:
        raw = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Explicit rebind is the migration path for a legacy tree.
        return
    except PermissionError as exc:
        raise ArtifactRootRebindError(
            f"Cannot validate artifact-root binding {path} as the current user "
            "(permission denied). Use the user that owns this artifact tree or restore "
            "read permission; refusing to rebind an ownership record that was not read. "
            "Nothing was written."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ArtifactRootRebindError(
            f"Cannot read artifact-root binding {path}: {exc}. Nothing was written."
        ) from exc
    expected = _artifact_root_binding_payload(plan.experiment)
    backend_conflict = _auto_storage_backend_conflict(plan.experiment, raw)
    if backend_conflict is not None:
        raise ArtifactRootRebindError(backend_conflict)
    recorded_root = raw.get("artifact_root") if isinstance(raw, dict) else None
    storage_matches = isinstance(raw, dict) and raw.get("storage_key") == expected["storage_key"]
    if (
        not storage_matches
        and plan.experiment.storage == "auto"
        and isinstance(recorded_root, str)
        and Path(recorded_root).is_absolute()
        and Path(recorded_root).name == plan.experiment.experiment
        and all(entry.previous in {recorded_root, plan.destination} for entry in plan.entries)
    ):
        # Reconstruct the old, already-resolved filename lexically. The old
        # tree may be gone or replaced by a symlink after the move.
        parallel = any(phase.n_jobs > 1 for phase in plan.experiment.phases)
        backend, filename = ("journal", "study.journal") if parallel else ("sqlite", "study.db")
        previous_identity = f"{backend}:///{Path(recorded_root) / filename}"
        storage_matches = (
            raw.get("storage_key") == hashlib.sha256(previous_identity.encode("utf-8")).hexdigest()
        )
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != ARTIFACT_ROOT_BINDING_SCHEMA_VERSION
        or raw.get("experiment") != expected["experiment"]
        or not storage_matches
        or not isinstance(recorded_root, str)
        or not Path(recorded_root).is_absolute()
    ):
        raise ArtifactRootRebindError(
            f"Artifact root {plan.destination!r} records ownership by another storage "
            "ledger, experiment, or source tree. Refusing to rebind the "
            "current studies onto it. Nothing was written."
        )


def _apply_artifact_root_rebind(plan: _ArtifactRootRebindPlan) -> list[tuple[str, str | None, str]]:
    """Write one validated plan's new artifact root onto every existing phase study.

    :param _ArtifactRootRebindPlan plan: Plan already validated by
        :func:`_plan_artifact_root_rebinds`.
    :raises ArtifactRootRebindError: The destination ownership record cannot be written.
    :return list[tuple[str, str | None, str]]: One ``(study name, previous
        root or None, new root)`` record per study written.
    """
    if not plan.entries:
        return []
    binding_path = _artifact_root_binding_path(plan.experiment)
    try:
        atomic_write_text(
            binding_path,
            json.dumps(_artifact_root_binding_payload(plan.experiment), sort_keys=True) + "\n",
        )
    except OSError as exc:
        raise ArtifactRootRebindError(
            f"Could not write artifact-root binding {binding_path}: {exc}. No study "
            "binding was changed."
        ) from exc
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
_TRAINER_INPUT_FILENAMES = {
    "yaml_file": "trainer_config.yaml",
    "json_file": "overrides.json",
    "argparse": "overrides_resolved.json",
    "hydra": "overrides_resolved.json",
}


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
        trial predates the record (review v0.5.17 / finding F).
    :raises TrialEvidenceMissingError: A present provenance record is corrupt
        or has an unsupported shape.
    """
    if OBJECTIVE_PROVENANCE_ATTR not in trial.user_attrs:
        return None
    raw = trial.user_attrs[OBJECTIVE_PROVENANCE_ATTR]
    if not isinstance(raw, str) or not raw:
        raise TrialEvidenceMissingError(
            f"Trial {trial.number} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence."
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrialEvidenceMissingError(
            f"Trial {trial.number} has corrupt {OBJECTIVE_PROVENANCE_ATTR!r} JSON evidence."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise TrialEvidenceMissingError(
            f"Trial {trial.number} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence."
        )
    return parsed


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


def _verify_trainer_input_evidence(
    trial_dir: Path,
    record: Any,
    *,
    subject: str,
) -> None:
    """Content-verify the historical generated input one trainer consumed.

    The filename comes from the trial's versioned record, not the current
    experiment config. Validating the format/filename pair keeps that
    historical path trial-relative and prevents a malformed ledger value from
    redirecting verification outside the evidence directory.

    :param Path trial_dir: Structurally translated trial evidence directory.
    :param Any record: Raw ``TRAINER_INPUT_ATTR`` Optuna user attribute.
    :param str subject: Caller-built trial label for diagnostics.
    :raises TrialEvidenceMissingError: The record is absent or malformed, or
        its exact file bytes are missing or changed.
    """
    if not isinstance(record, Mapping):
        raise TrialEvidenceMissingError(
            f"{subject} has no valid {TRAINER_INPUT_ATTR!r} record, so its historical "
            f"trainer input cannot be verified. {_TRIAL_EVIDENCE_REMEDY}"
        )
    input_format = record.get("format")
    filename = record.get("filename")
    expected_filename = (
        _TRAINER_INPUT_FILENAMES.get(input_format) if isinstance(input_format, str) else None
    )
    if (
        record.get("schema_version") != TRAINER_INPUT_SCHEMA_VERSION
        or expected_filename is None
        or filename != expected_filename
    ):
        raise TrialEvidenceMissingError(
            f"{subject} has a malformed or unsupported {TRAINER_INPUT_ATTR!r} record "
            f"{dict(record)!r}, so its historical trainer input cannot be located safely. "
            f"{_TRIAL_EVIDENCE_REMEDY}"
        )
    recorded_size = record.get("size_bytes")
    recorded_digest = record.get("sha256")
    if (
        not isinstance(recorded_size, int)
        or isinstance(recorded_size, bool)
        or recorded_size < 0
        or not isinstance(recorded_digest, str)
        or len(recorded_digest) != 64
        or any(character not in "0123456789abcdef" for character in recorded_digest)
    ):
        raise TrialEvidenceMissingError(
            f"{subject} has an invalid size or content identity in its "
            f"{TRAINER_INPUT_ATTR!r} record. {_TRIAL_EVIDENCE_REMEDY}"
        )

    input_path = trial_dir / filename
    if not input_path.is_file():
        raise TrialEvidenceMissingError(
            f"{subject} is missing its recorded trainer input {filename!r} under "
            f"{str(trial_dir)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        )
    try:
        actual_size = input_path.stat().st_size
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} cannot read its recorded trainer input {filename!r} at "
            f"{str(input_path)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        ) from exc
    if actual_size != recorded_size:
        raise TrialEvidenceMissingError(
            f"{subject} recorded trainer input {filename!r} as {recorded_size} bytes, but "
            f"that file is now {actual_size} bytes. {_TRIAL_EVIDENCE_REMEDY}"
        )
    try:
        actual_digest = file_sha256(input_path)
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} cannot content-verify its recorded trainer input {filename!r} at "
            f"{str(input_path)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        ) from exc
    if actual_digest != recorded_digest:
        raise TrialEvidenceMissingError(
            f"{subject} recorded trainer input {filename!r} with sha256 {recorded_digest}, "
            f"but that file now hashes to {actual_digest}: the exact trainer input bytes "
            f"have changed. {_TRIAL_EVIDENCE_REMEDY}"
        )


def _verify_trial_evidence_dir(
    trial_dir: Path,
    *,
    subject: str,
    attempt_id: str,
    provenance: Mapping[str, Any] | None,
    trainer_input: Any,
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
    :param Any trainer_input: Versioned historical generated-input record.
    :param bool verify_objective_digest: Re-hash the objective source as well.
    :raises TrialEvidenceMissingError: The directory, an audit artifact, the
        generated trainer input, or recorded objective source is missing,
        foreign, or altered.
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
    _verify_trainer_input_evidence(trial_dir, trainer_input, subject=subject)
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
    over them cannot bias the result. Every candidate's small generated trainer
    input is content-verified. The potentially unbounded objective source is
    checked for existence and recorded size only, because re-hashing every
    candidate's ``stdout.log`` on every top-up would cost the whole study's log
    volume per resume. The winning objective source is digest-verified at
    selection time instead (:func:`_verify_winner_objective_evidence`).

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param Mapping[str, optuna.Study] studies: Existing phase studies keyed by
        phase name, as returned by :func:`_preflight_existing_studies`.
    :raises TrialEvidenceMissingError: A selection-eligible trial records an
        unusable trial directory, or its evidence directory, audit artifacts,
        generated trainer input, or recorded objective source are no longer in
        this tree.
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
                trainer_input=trial.user_attrs.get(TRAINER_INPUT_ATTR),
                verify_objective_digest=False,
            )


def _verify_winner_objective_evidence(
    experiment: Experiment,
    phase_name: str,
    selected: SelectedTrial,
) -> None:
    """Digest-verify the evidence behind a trial that is about to be published.

    The launch preflight content-verifies every candidate's small generated
    trainer input and proves its objective source still exists; this additionally
    proves the winning objective source is byte-for-byte the evidence its frozen
    provenance recorded. The split is deliberate: the default objective source
    is an uncapped trainer log, so hashing every candidate on every top-up is
    O(total trainer log bytes) per resume - potentially tens of gigabytes -
    while hashing only the published winner is bounded by one trial's log and
    still catches every deletion and every result-affecting edit, including a
    tamper that preserves byte length (PR #5 review / reviewer 2, blocker 7).

    It runs on every selection, so a deadline-truncated partial publication is
    covered on the same terms as a complete one.

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param str phase_name: Phase whose winner was just selected.
    :param SelectedTrial selected: The winning trial and its frozen provenance.
    :raises TrialEvidenceMissingError: The winner's evidence directory, audit
        artifacts, generated trainer input, or objective source are missing,
        foreign, or altered.
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
        trainer_input=selected.trainer_input,
        verify_objective_digest=True,
    )


def _preflight_existing_studies(
    experiment: Experiment,
    *,
    cleanup_report: _PreflightCleanupReport | None = None,
    from_phase: str | None = None,
    preloaded_studies: Mapping[str, optuna.Study] | None = None,
) -> dict[str, optuna.Study]:
    """Validate and reap every existing declared phase study before launch.

    :param Experiment experiment: Parsed experiment whose declared phases are inspected.
    :param _PreflightCleanupReport | None cleanup_report: Optional shared report to
        accumulate cleanup evidence into; a fresh one is created if omitted.
    :param str | None from_phase: Optional resume point. Recovery and schema checks
        still cover every phase; trial-target validation starts at this reached phase.
    :param Mapping[str, optuna.Study] | None preloaded_studies: Studies already
        discovered and ownership-checked under the experiment lock before the
        generation claim. Direct callers omit this and perform the same strict
        discovery here.
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
    :raises PublishedStudyMissingError: A reached phase has a published winner
        but its persistent study is absent or empty.
    :raises StudySchemaMismatchError: A phase's study uses an incompatible
        storage schema.
    :raises TrialTargetRegressionError: A phase's study already accepted a
        higher trial target than the current config requests.
    :raises ProcessCleanupUncertainError: Stale-trial cleanup could not be
        confirmed safe for a phase's study.
    :raises PhaseSweepError: Multiple studies failed preflight for mixed
        expected reasons not covered by a more specific common exception type.
    :raises RuntimeError: Multiple studies failed and at least one error is an
        unexpected implementation failure that must retain traceback reporting.
    """
    current_environment_digest = _environment_identity(experiment).digest
    report = cleanup_report or _PreflightCleanupReport()
    # Discovery, root checks, and claims happen in ONE strict pass, and its
    # study objects are the ones every later step operates on: an invocation
    # offering a second publication root must not reap, inspect, or claim
    # anything in either tree (review v0.5.19 / finding F5), and a storage
    # read that fails must abort rather than let a second, luckier read hand
    # recovery a study whose root was never checked (PR #5 review /
    # reviewer 2, issue 1). The storage error still marks cleanup uncertain:
    # an unreadable ledger cannot prove its attempts are resolved.
    if preloaded_studies is None:
        try:
            loaded = _load_and_check_artifact_roots(experiment, from_phase=from_phase)
        except (StudyStorageUnavailableError, PublishedStudyMissingError) as exc:
            # This discovery also runs during post-execution reconciliation.
            # A lost ledger then aborts recovery before the attempt registry
            # can be inspected, so cleanup cannot be reported as confirmed.
            report.mark_uncertain(exc)
            raise
    else:
        loaded = dict(preloaded_studies)
    studies: dict[str, optuna.Study] = {}
    errors: list[Exception] = []
    # The registry scan runs FIRST and is independent of the declared phase
    # list, so attempts from renamed/removed phases or changed storage URLs
    # are recovered before any current-config validation or launch (review
    # v0.5.17 / blocker 3).
    try:
        _preflight_active_attempts(experiment, report)
    except Exception as exc:
        if isinstance(exc, (ProcessCleanupUncertainError, StudyStorageUnavailableError)):
            report.mark_uncertain(exc)
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
            recovered_terminal_attempts: set[str] = set()
            _recover_cleanup_uncertain_trials(
                study,
                experiment,
                phase.name,
                recovered_attempt_ids=recovered_terminal_attempts,
                recovered_attempt_generations=report.recovered_attempt_generations,
            )
            report.recovered_attempt_ids.update(recovered_terminal_attempts)
            for attempt_id in recovered_terminal_attempts:
                _retire_active_attempt(experiment, attempt_id)
            _reap_stale_trials(
                study,
                experiment,
                phase.name,
                recovered_attempt_ids=report.recovered_attempt_ids,
                recovered_attempt_generations=report.recovered_attempt_generations,
                uncertain_attempt_ids=report.uncertain_attempt_ids,
            )
        except Exception as exc:
            if isinstance(exc, (ProcessCleanupUncertainError, StudyStorageUnavailableError)):
                report.mark_uncertain(exc)
            errors.append(exc)
            continue
        try:
            _validate_study_direction(study, experiment.metric.goal)
            _validate_study_schema(study)
            if reached:
                _validate_environment_cohort(study, current_environment_digest)
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
        if all(isinstance(error, StudyFingerprintMismatchError) for error in errors):
            raise StudyFingerprintMismatchError(message) from first
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
        unexpected = next(
            (error for error in errors if not isinstance(error, PhaseSweepError)),
            None,
        )
        if unexpected is not None:
            raise RuntimeError(message) from unexpected
        raise PhaseSweepError(message) from first
    return studies


def _inspect_stale_running_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
) -> int:
    """Count stale RUNNING trials without signaling processes or writing state.

    Used by ``mcp recover-run`` preflight mode. The follow-up ``--confirm`` call
    must still find the same RUNNING trials so it can reap them and persist
    recovery evidence atomically with clearing MCP cleanup uncertainty.

    :param optuna.Study study: Study whose stale RUNNING trials should be inspected.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for
        attempts a confirmed pass would reap.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each collected attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :return int: Number of stale RUNNING trials found.
    """
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)
        if TRIAL_DIR_ATTR in trial.user_attrs:
            _resolve_attempt_for_reaping(trial, trial_dir, study.study_name, inspect_only=True)
        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )
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


def _previously_recovered_attempt_locations(
    study: optuna.Study,
    phase_name: str,
    generation_id: str,
    *,
    causal_attempt_ids: set[str] | None = None,
) -> dict[str, tuple[str, int, str]]:
    """Return this run's trial cleanup evidence from a prior interrupted pass.

    Recovery durably records each confirmed trial in the study-level ledger
    before the CLI can persist its run-level recovery record or clear the
    cleanup-uncertainty marker. A crash in that window must not erase the
    evidence: the retry skips these trials as already recovered, and without
    this mapping the trial-level-evidence guard would refuse to clear cleanup
    uncertainty forever (review v0.5.17 gap hunt).

    The mapping is scoped to ``generation_id`` — detached MCP runs use their run
    id as the generation id — so a *different* run's already-consumed evidence
    cannot clear this run's uncertainty; that cross-run refusal stays
    fail-closed.

    :param optuna.Study study: Study whose ledger and trials are inspected.
    :param str phase_name: Configured phase whose study is being inspected.
    :param str generation_id: Generation identity of the run being recovered.
    :param set[str] | None causal_attempt_ids: Older-generation attempts the
        run's own terminal report explicitly named as cleanup-uncertain.
    :return dict[str, tuple[str, int, str]]: Recovered attempt ids causally
        bound to this run, mapped to phase, trial number, and generation.
    """
    recovered = _cleanup_recovered_trial_numbers(study)
    if not recovered:
        return {}
    attempts: dict[str, tuple[str, int, str]] = {}
    for trial in study.get_trials(deepcopy=False):
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        attempt_generation = trial.user_attrs.get(GENERATION_ID_ATTR)
        if not (
            trial.state.is_finished()
            and trial.number in recovered
            and isinstance(attempt_id, str)
            and attempt_id
            and isinstance(attempt_generation, str)
            and attempt_generation
        ):
            continue
        if attempt_generation == generation_id or (
            causal_attempt_ids is not None and attempt_id in causal_attempt_ids
        ):
            attempts[attempt_id] = (phase_name, trial.number, attempt_generation)
    return attempts


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
            and _trial_requires_cleanup_recovery(trial)
        ):
            trial_dir = _trial_dir_for_cleanup_recovery(trial, study.study_name)
            identity = _read_trial_process_identity(trial, trial_dir, study.study_name)
            yield trial, trial_dir, identity


def _recover_cleanup_uncertain_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
) -> int:
    """Confirm cleanup for terminal trials that explicitly recorded uncertainty.

    ``UnsafeProcessCleanupError`` can leave an Optuna trial in a terminal FAIL state with
    ``phasesweep_cleanup_confirmed=false``. The normal stale reaper intentionally visits
    only RUNNING trials, so operator recovery needs this separate fail-closed inspection
    before clearing MCP cleanup uncertainty.

    :param optuna.Study study: Existing Optuna study for the phase being recovered.
    :param Experiment experiment: Parsed experiment, used for diagnostics.
    :param str phase_name: Name of the phase being recovered.
    :param set[str] | None recovered_attempt_ids: Optional collector for exact
        attempts whose cleanup was confirmed.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each collected attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
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
        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )
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


def _inspect_cleanup_uncertain_trials(
    study: optuna.Study,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
) -> int:
    """Count recoverable terminal cleanup evidence without signals or writes.

    :param optuna.Study study: Existing study inspected by recovery preflight.
    :param str phase_name: Name of the phase containing the recovered trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for
        attempts a confirmed pass would recover.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each collected attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :return int: Number of unconsumed terminal trials that record cleanup uncertainty.
    :raises ProcessCleanupUncertainError: A trial lacks the persisted identity
        required for a safe confirmed recovery.
    """
    count = 0
    for trial, _trial_dir, _identity in _iter_cleanup_uncertain_trials(study):
        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )
        count += 1
    return count
