"""Engine locks, fingerprints, and stale-trial recovery guards."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config import Experiment, Phase, Suite
from phasesweep.engine.optuna import _load_existing_phase_study
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    Winner,
    _experiment_dir,
    _suite_dir,
    _trial_dir_for,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import (
    canonical_storage_identity,
    exclusive_lock,
    try_lock_file,
    unlock_file,
)
from phasesweep.runtime.files import (
    lock_dir as _lock_dir,
)
from phasesweep.runtime.process import kill_stale_group, read_stale_process_identity


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
        RuntimeError: Another phasesweep process holds one of the required
            locks (output namespace or storage identity).

    """
    paths = _run_lock_paths(experiment)
    handles: list[Any] = []
    try:
        for path in paths:
            handle = try_lock_file(path)
            if handle is None:
                raise RuntimeError(
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
FINGERPRINT_SCHEMA_VERSION = 1


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
    existing = study.user_attrs.get("phasesweep_fingerprint")
    if existing is None:
        study.set_user_attr("phasesweep_fingerprint", fp)
    elif existing != fp:
        raise RuntimeError(
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


def _reap_stale_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
) -> int:
    """Mark RUNNING trials as FAIL after killing orphaned process groups.

    :param optuna.Study study: Study whose stale RUNNING trials should be reaped.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for exact
        attempt identities whose durable state was changed to FAIL.
    :return int: Number of stale RUNNING trials marked as failed.
    """
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue

        trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)

        identity = read_stale_process_identity(trial_dir)
        if identity.pid is not None or identity.pgid is not None:
            safe_to_fail = kill_stale_group(identity.pid, identity.starttime, pgid=identity.pgid)
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

        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
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
    raise RuntimeError(
        f"Study {study.study_name!r} uses unsupported phasesweep storage schema {detail}; "
        f"current schema is {STUDY_SCHEMA_VERSION}. Affected trial numbers: {trial_numbers}. "
        "Use a new experiment name, or archive/delete the old study before running again."
    )


def _preflight_existing_studies(experiment: Experiment) -> dict[str, optuna.Study]:
    """Validate and reap every existing declared phase study before launch."""
    studies: dict[str, optuna.Study] = {}
    errors: list[BaseException] = []
    for phase in experiment.phases:
        study = _load_existing_phase_study(experiment, phase)
        if study is None:
            continue
        studies[phase.name] = study
        try:
            _validate_study_schema(study)
            _reap_stale_trials(study, experiment, phase.name)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        first = errors[0]
        if len(errors) == 1:
            raise first
        raise RuntimeError(
            "Experiment recovery preflight found multiple unsafe studies: "
            + "; ".join(str(error) for error in errors)
        ) from first
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
        _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)
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
        identity = read_stale_process_identity(trial_dir)
        if identity.pid is None and identity.pgid is None:
            raise ProcessCleanupUncertainError(
                f"Refusing to clear cleanup uncertainty for trial {trial.number} in "
                f"study {study.study_name}: no persisted process identity was found "
                f"under trial_dir={trial_dir}. A leaked process group cannot be "
                "ruled out."
            )
        safe_to_clear = kill_stale_group(identity.pid, identity.starttime, pgid=identity.pgid)
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
        identity = read_stale_process_identity(trial_dir)
        if identity.pid is None and identity.pgid is None:
            raise ProcessCleanupUncertainError(
                f"Refusing to plan cleanup recovery for trial {trial.number} in "
                f"study {study.study_name}: no persisted process identity was found "
                f"under trial_dir={trial_dir}. A leaked process group cannot be ruled out."
            )
        count += 1
    return count


def _reap_skipped_phase(experiment: Experiment, phase: Phase) -> None:
    """Reap stale RUNNING trials for a phase skipped by ``--from-phase``.

    :param Experiment experiment: Parsed experiment config containing storage details.
    :param Phase phase: Phase being skipped and recovered before loading its winner.
    """
    if experiment.storage is None:
        return
    study = _load_existing_phase_study(experiment, phase)
    if study is None:
        return
    _validate_study_schema(study)
    _reap_stale_trials(study, experiment, phase.name)
