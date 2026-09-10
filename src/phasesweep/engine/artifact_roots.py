"""Artifact-root ownership and published-study binding."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    LegacyArtifactRootMigrationRequiredError,
    PublishedStudyMissingError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.optuna import (
    _load_existing_phase_study,
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

    ``workdir`` is excluded from semantic fingerprints so artifact trees can
    move. The shared :func:`_artifact_root_claim_needed` check permits first
    contact only for an empty study; moving a populated study's binding
    requires explicit ``phasesweep rebind-workdir``.

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
