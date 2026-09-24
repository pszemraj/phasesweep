"""Artifact-root ownership and published-study binding."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    OperatorAction,
    PublicationAccessError,
    PublicationIntegrityError,
    PublishedStudyMissingError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.optuna import (
    _published_phase_trial_refs,
    _published_trial_matches,
)
from phasesweep.engine.paths import _artifact_root_binding_path, _experiment_dir
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR
from phasesweep.runtime.files import (
    atomic_write_text,
    canonical_storage_identity,
    storage_is_in_memory,
)
from phasesweep.runtime.json import strict_json_loads

log = logging.getLogger(__name__)

#: What an artifact tree's own ownership record says about it. ``"unbound"`` is
#: a fresh tree that has never recorded a ledger; every other shape either
#: matches exactly (``"bound"``) or is refused.
BindingState = Literal["bound", "unbound"]


def _artifact_root_identity(experiment: Experiment) -> str:
    """Return the single artifact root a persistent study is allowed to publish into.

    Derived through :func:`phasesweep.engine.paths._experiment_dir` — the same
    helper every artifact
    path is built from — so the recorded binding and the namespace actually
    written can never drift apart.

    :param Experiment experiment: Parsed experiment supplying workdir and name.
    :return str: Resolved ``<workdir>/<experiment>`` namespace as a string.
    """
    return str(_experiment_dir(experiment).resolve())


def _artifact_root_binding_applies(experiment: Experiment) -> bool:
    """Return whether the ledger can carry the reverse artifact-root binding.

    :param Experiment experiment: Parsed experiment whose storage is inspected.
    :return bool: ``True`` only for persistent storage. Every artifact root has
        its own format record; only a persistent ledger also needs to record
        the root that owns it across invocations.
    """
    return experiment.resolved_storage is not None and not storage_is_in_memory(
        experiment.resolved_storage
    )


ARTIFACT_ROOT_BINDING_SCHEMA_VERSION = 3


def _artifact_root_storage_key(experiment: Experiment) -> str | None:
    """Return an opaque comparison key, or an explicit in-memory identity.

    The canonical identity is credential-free but may contain other target
    selectors from a query or nested connection string. The artifact tree is
    intentionally shareable, so it stores only this digest while private
    recovery state retains the operational URL.

    :param Experiment experiment: Experiment whose persistent ledger is identified.
    :return str | None: Full SHA-256 hex digest of the canonical persistent
        storage identity, or ``None`` for deliberate no-ledger execution.
    """
    storage_identity = canonical_storage_identity(experiment.resolved_storage)
    if storage_identity is None or storage_is_in_memory(experiment.resolved_storage):
        return None
    return hashlib.sha256(storage_identity.encode("utf-8")).hexdigest()


def _artifact_root_binding_payload(experiment: Experiment) -> dict[str, Any]:
    """Build the format and ownership record for one artifact root.

    :param Experiment experiment: Experiment whose artifact root is bound.
    :return dict[str, Any]: Versioned experiment, root, and optional storage identity record.
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
        "No trial ran and nothing was published."
    )


def _root_durable_state_entry(experiment: Experiment) -> str | None:
    """Return the first known PhaseSweep state entry in an unbound root.

    Arbitrary operator files do not establish storage ownership. A note,
    ``.DS_Store``, or other unrelated file may already live in a chosen output
    directory before the first run, and treating it as PhaseSweep state makes
    the experiment name unusable. Only paths PhaseSweep itself creates require
    a format-boundary refusal.

    :param Experiment experiment: Experiment whose artifact root is inspected.
    :raises ArtifactRootConflictError: The artifact root cannot be enumerated.
    :return str | None: Known durable entry name, or ``None`` when no known
        PhaseSweep state entry is present.
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
        "study.db",
        "study.journal",
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
            f"Cannot inspect artifact root {_artifact_root_identity(experiment)!r}: {exc}.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc


def _check_artifact_root_binding(experiment: Experiment) -> BindingState:
    """Classify an artifact tree's recorded ownership, writing nothing.

    Study attributes bind ledger to tree. This reverse record binds tree to
    ledger, preventing a second database from combining its trial counts with
    another database's publication. Existing unmarked PhaseSweep state is
    pre-cutover state and is refused by this release.

    This is step one of the fixed ledger order, so it is strictly read-only: it
    neither writes the record (:func:`_write_artifact_root_binding`) nor opens
    or inspects the ledger itself
    (:func:`phasesweep.engine.ledger._scan_ledger_format`). Separating the two
    is what lets a pure read path classify a tree without touching a byte.

    :param Experiment experiment: Config whose root and storage must agree.
    :return BindingState: ``"bound"`` when the tree already records exactly
        this experiment and ledger; ``"unbound"`` when it records nothing yet.
    :raises ArtifactRootConflictError: The binding cannot be validated as the
        current user, is malformed or names another owner, or the root holds
        unmarked pre-cutover PhaseSweep state.
    """
    path = _artifact_root_binding_path(experiment)
    expected = _artifact_root_binding_payload(experiment)
    try:
        raw = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        durable_entry = _root_durable_state_entry(experiment)
        if durable_entry is not None:
            raise ArtifactRootConflictError(
                f"Artifact root {expected['artifact_root']!r} contains pre-cutover "
                f"PhaseSweep state entry {durable_entry!r} but no supported format marker. "
                "Use a fresh artifact root and fresh local storage with this PhaseSweep "
                "release, or use the preserved PhaseSweep 0.3.1 environment to operate "
                "the existing state. Nothing was written.",
                action=OperatorAction.USE_PRIOR_RELEASE,
            ) from None
        return "unbound"
    except PermissionError as exc:
        raise ArtifactRootConflictError(
            f"Artifact-root binding {path} cannot be validated as the current user "
            "(permission denied). Use the user that owns this artifact tree or restore "
            "read permission before using this artifact root.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    except (OSError, ValueError) as exc:
        raise ArtifactRootConflictError(
            f"Artifact-root binding {path} is unreadable or malformed: {exc}. Refusing "
            "to combine this tree with an unverified storage ledger.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    if isinstance(raw, dict) and raw.get("schema_version") != ARTIFACT_ROOT_BINDING_SCHEMA_VERSION:
        raise ArtifactRootConflictError(
            f"Artifact root {expected['artifact_root']!r} uses unsupported pre-cutover "
            f"PhaseSweep format {raw.get('schema_version')!r}. Use a fresh artifact root "
            "and fresh local storage with this PhaseSweep release, or use the preserved "
            "PhaseSweep 0.3.1 environment to operate the existing state. Nothing was written.",
            action=OperatorAction.USE_PRIOR_RELEASE,
        )
    if raw != expected:
        backend_conflict = _auto_storage_backend_conflict(experiment, raw)
        if backend_conflict is not None:
            raise ArtifactRootConflictError(backend_conflict, action=OperatorAction.FIX_CONFIG)
        raise ArtifactRootConflictError(
            f"Artifact root {expected['artifact_root']!r} is bound to a different storage "
            f"ledger or experiment than {experiment.experiment!r}. Use the config that owns "
            "this current-format tree, or use a fresh artifact root and local storage. "
            "Nothing was written.",
            action=OperatorAction.FIX_CONFIG,
        )
    return "bound"


def _write_artifact_root_binding(experiment: Experiment) -> None:
    """Record this experiment's ownership of a tree the caller already checked.

    Callers reach here only after :func:`_check_artifact_root_binding` returned
    ``"unbound"``, so this writes the first ownership record rather than
    replacing one.

    :param Experiment experiment: Experiment whose artifact root is claimed.
    :raises ArtifactRootConflictError: The record could not be written.
    """
    expected = _artifact_root_binding_payload(experiment)
    try:
        atomic_write_text(
            _artifact_root_binding_path(experiment),
            json.dumps(expected, sort_keys=True) + "\n",
        )
    except OSError as exc:
        raise ArtifactRootConflictError(
            f"Could not bind fresh artifact root {expected['artifact_root']!r} to "
            f"its storage ledger: {exc}. No trial ran and nothing was published.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc


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
    :raises ArtifactRootConflictError: The study is already bound to a different
        artifact root, carries a binding that is not a string, or holds trials
        without recording any root.
    """
    if ARTIFACT_ROOT_ATTR not in study.user_attrs:
        trial_count = len(study.get_trials(deepcopy=False))
        if trial_count:
            raise ArtifactRootConflictError(
                f"Study {study.study_name!r} holds {trial_count} trial(s) but records no "
                "artifact root and is pre-cutover state. Use a fresh artifact root and "
                "local storage with this release, or use the preserved PhaseSweep 0.3.1 "
                "environment to operate the existing state. Nothing was written.",
                action=OperatorAction.USE_PRIOR_RELEASE,
            )
        return True
    _check_study_artifact_root(study, experiment)
    return False


def _check_study_artifact_root(study: optuna.Study, experiment: Experiment) -> None:
    """Refuse a study that records an artifact root other than this config's.

    The ownership half of :func:`_artifact_root_claim_needed`, split out for
    callers that use existing studies without ever claiming one (MCP recovery
    reaps through them). A study that records no root passes, because it is
    not bound to anyone else; whether it may be claimed is the claim's
    question, not this one.

    :param optuna.Study study: Phase study whose recorded binding is inspected.
    :param Experiment experiment: Parsed experiment supplying the artifact root.
    :raises ArtifactRootConflictError: The study is bound to a different
        artifact root, or carries a binding that is not a string.
    """
    if ARTIFACT_ROOT_ATTR not in study.user_attrs:
        return
    offered = _artifact_root_identity(experiment)
    bound = study.user_attrs[ARTIFACT_ROOT_ATTR]
    if bound == offered:
        return
    raise ArtifactRootConflictError(
        f"Study {study.study_name!r} publishes into artifact root {bound!r}, but this "
        f"config offers {offered!r}. One persistent study backs exactly one publication "
        "root; running it against a second workdir would top up trials whose artifacts "
        "live under the bound root and publish a divergent result tree. Restore the "
        "original workdir, or use a fresh artifact root and local storage. No trial ran "
        "and nothing was published.",
        action=OperatorAction.FIX_CONFIG,
    )


def _bind_study_artifact_root(study: optuna.Study, experiment: Experiment) -> None:
    """Claim, or re-confirm, the one artifact root a single phase study publishes into.

    The shared :func:`_artifact_root_claim_needed` check permits first contact
    only for an empty study.

    :param optuna.Study study: Phase study to bind.
    :param Experiment experiment: Parsed experiment supplying the artifact root.
    :raises ArtifactRootConflictError: The study is already bound to a different
        artifact root, or carries a binding that is not a string.
    """
    if not _artifact_root_binding_applies(experiment):
        return
    if not _artifact_root_claim_needed(study, experiment):
        return
    _claim_study_artifact_root(study, _artifact_root_identity(experiment))


def _check_published_phase_studies(
    experiment: Experiment,
    loaded: Mapping[str, optuna.Study],
    *,
    from_phase: str | None = None,
) -> None:
    """Require published trial identity and history for phases that would execute.

    :param Experiment experiment: Experiment whose current publication is checked.
    :param Mapping[str, optuna.Study] loaded: Already-inspected persistent studies.
    :param str | None from_phase: Resume point; earlier phases only load winners.
    :raises PublishedStudyMissingError: A reached published trial is absent,
        replaced, or its recorded history boundary is missing.
    :raises PublicationAccessError: The last publication cannot be read.
    :raises PublicationIntegrityError: The last publication no longer validates.
    :raises StudyStorageUnavailableError: A published study's trials cannot be read.
    """
    publication = _resolve_publication_pointer(experiment)
    if publication.state == "permission_denied":
        raise PublicationAccessError(publication.error or "Published result is unreadable.")
    if publication.state == "failed":
        raise PublicationIntegrityError(publication.error or "Published result is invalid.")
    published_trials = _published_phase_trial_refs(publication.summary)
    reached = from_phase is None
    for phase_name in (phase.name for phase in experiment.phases):
        if phase_name == from_phase:
            reached = True
        if not reached or phase_name not in published_trials:
            continue
        study = loaded.get(phase_name)
        expected = published_trials[phase_name]
        missing = "is missing"
        if study is not None:
            try:
                trials = study.get_trials(deepcopy=False)
                matched = expected is not None and any(
                    _published_trial_matches(trial, expected) for trial in trials
                )
                if matched and expected is not None:
                    finished = sum(trial.state.is_finished() for trial in trials)
                    completed = sum(
                        trial.state == optuna.trial.TrialState.COMPLETE for trial in trials
                    )
                    if (
                        expected.finished_trials is None or finished >= expected.finished_trials
                    ) and (
                        expected.completed_trials is None or completed >= expected.completed_trials
                    ):
                        continue
                    missing = (
                        f"has only {finished} terminal and {completed} complete trials, below "
                        "the published completion boundary"
                    )
                    if expected.finished_trials is not None:
                        missing += f" ({expected.finished_trials} terminal"
                        if expected.completed_trials is not None:
                            missing += f", {expected.completed_trials} complete"
                        missing += ")"
            except Exception as exc:
                raise StudyStorageUnavailableError.rewrap(
                    exc,
                    "Could not inspect persistent study storage for published phase "
                    f"{phase_name!r}.",
                ) from exc
            if not matched:
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
                        missing += (
                            " because the publication records no complete local trial identity"
                        )
        raise PublishedStudyMissingError(
            f"Published generation {publication.generation_id!r} records a selected trial for "
            f"phase {phase_name!r}, but its persistent study {missing}. That publication "
            "requires its original trial history; continuing could reuse incomplete or "
            "unrelated trials and replace the current publication. Restore the "
            "original complete storage ledger and study, or use a new experiment identity "
            "for a fresh run. Nothing was written."
        )
