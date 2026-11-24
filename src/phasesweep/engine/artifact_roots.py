"""Artifact-root ownership and published-study binding."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import optuna

from phasesweep.config.models import Experiment
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    ArtifactRootSchemaMismatchError,
    PublishedStudyMissingError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.optuna import (
    _finished_trial_count,
    _measured_trial_count,
    _meets_published_trial_history_boundary,
    _published_phase_trial_refs,
    _published_trial_matches,
)
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _artifact_root_identity,
    _experiment_dir,
)
from phasesweep.engine.publication import (
    _resolve_publication_pointer,
    raise_for_unusable_publication,
)
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR
from phasesweep.errors import (
    FRESH_STUDY,
    OTHER_FORMAT_STEP,
    OperatorAction,
    format_mismatch,
    restore_or_new_study,
)
from phasesweep.runtime.files import (
    atomic_write_text,
    canonical_storage_identity,
)
from phasesweep.runtime.json import strict_json_loads
from phasesweep.runtime.terminal import plural

log = logging.getLogger(__name__)

#: What an artifact tree's own ownership record says about it. ``"unbound"`` is
#: a fresh tree that has never recorded a ledger; every other shape either
#: matches exactly (``"bound"``) or is refused.
BindingState = Literal["bound", "unbound"]


ARTIFACT_ROOT_BINDING_SCHEMA_VERSION = 27


def _artifact_root_storage_key(experiment: Experiment) -> str | None:
    """Hash the canonical local ledger identity, or return None for in-memory storage.

    :param Experiment experiment: Experiment whose persistent ledger is identified.
    :return str | None: Full SHA-256 hex digest of the canonical persistent
        storage identity, or ``None`` for in-memory execution.
    """
    storage_identity = canonical_storage_identity(experiment.resolved_storage)
    if storage_identity is None:
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


def _root_durable_state_entry(experiment: Experiment) -> str | None:
    """Return the first known PhaseSweep state entry in an unbound root.

    Unrelated operator files do not establish ownership or require a format refusal.

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
        "study.journal",
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


_MOVED_TREE_STEP = f"Use the tree at the recorded path, or {FRESH_STUDY}."


def _binding_mismatch(
    path: Path, raw: object, expected: Mapping[str, object], experiment: Experiment
) -> ArtifactRootConflictError:
    """Name which recorded field of an artifact root's binding differs from this config's.

    A tree moved or copied to another path differs in its root, which also changes the
    ledger identity under ``storage: auto``, so the root is reported first and alone.

    :param Path path: Binding file that was read.
    :param object raw: The parsed binding.
    :param Mapping[str, object] expected: The binding this config would write.
    :param Experiment experiment: Config whose root, experiment and storage are checked.
    :return ArtifactRootConflictError: The refusal, with the step that fits its cause.
    """
    root = expected["artifact_root"]
    if not isinstance(raw, dict):
        return ArtifactRootConflictError(
            f"Artifact-root binding {path} is malformed: it is not a JSON object.",
            action=OperatorAction.RESTORE_TREE,
        )
    if raw.get("artifact_root") != root:
        return ArtifactRootConflictError(
            f"The artifact tree at {root!r} was created at {raw.get('artifact_root')!r}.",
            action=OperatorAction.FIX_CONFIG,
            remediation=_MOVED_TREE_STEP,
        )
    causes: list[str] = []
    names: list[str] = []
    if raw.get("experiment") != expected["experiment"]:
        causes.append(f"for experiment {raw.get('experiment')!r}, not {experiment.experiment!r}")
        names.append("`experiment`")
    if raw.get("storage_key") != expected["storage_key"]:
        causes.append("with a different storage ledger than this config names")
        names.append("`storage`")
    if not causes:
        return ArtifactRootConflictError(
            f"Artifact-root binding {path} is malformed: it records fields this PhaseSweep "
            "does not write.",
            action=OperatorAction.RESTORE_TREE,
        )
    return ArtifactRootConflictError(
        f"Artifact root {root!r} was created {', and '.join(causes)}.",
        action=OperatorAction.FIX_CONFIG,
        remediation=restore_or_new_study(f"the {' and '.join(names)} this tree was created with"),
    )


def _artifact_access_refusal(error: PermissionError, fallback: Path) -> ArtifactRootConflictError:
    """Explain how to restore access to a path the artifact tree needs.

    :param PermissionError error: OS refusal, which normally names the path.
    :param Path fallback: Path to report if the refusal names none.
    :return ArtifactRootConflictError: Refusal carrying the restore-permissions action.
    """
    return ArtifactRootConflictError(
        f"The current user cannot access {error.filename or fallback} (permission denied).",
        action=OperatorAction.RESTORE_TREE,
    )


def _artifact_path_mode(path: Path) -> int:
    """Read a path's mode without treating denied access as a missing entry.

    :param Path path: Path to inspect, following symlinks.
    :raises ArtifactRootConflictError: The current user cannot access the path.
    :return int: The mode from stat, or zero when no entry exists at the path.
    """
    try:
        return path.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return 0
    except PermissionError as exc:
        raise _artifact_access_refusal(exc, path) from exc


def _check_artifact_root_binding(experiment: Experiment) -> BindingState:
    """Classify a tree's experiment and ledger ownership without reading the ledger.

    The tree-to-ledger record complements each study's tree binding. Refuse foreign
    ownership and unmarked PhaseSweep state; do not write or scan storage here.

    :param Experiment experiment: Config whose root and storage must agree.
    :return BindingState: ``"bound"`` when the tree already records exactly
        this experiment and ledger; ``"unbound"`` when it records nothing yet.
    :raises ArtifactRootConflictError: The binding cannot be validated as the
        current user, is malformed or names another owner, or the root holds
        unmarked PhaseSweep state.
    """
    path = _artifact_root_binding_path(experiment)
    expected = _artifact_root_binding_payload(experiment)
    try:
        raw = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        durable_entry = _root_durable_state_entry(experiment)
        if durable_entry is not None:
            raise ArtifactRootConflictError(
                f"Artifact root {expected['artifact_root']!r} contains unmarked "
                f"PhaseSweep state entry {durable_entry!r} but no supported format marker.",
            ) from None
        return "unbound"
    except PermissionError as exc:
        raise _artifact_access_refusal(exc, path) from exc
    except (OSError, ValueError) as exc:
        raise ArtifactRootConflictError(
            f"Artifact-root binding {path} is unreadable or malformed: {exc}.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc
    if isinstance(raw, dict) and raw.get("schema_version") != ARTIFACT_ROOT_BINDING_SCHEMA_VERSION:
        raise ArtifactRootSchemaMismatchError(
            f"Artifact root {expected['artifact_root']!r} "
            f"{format_mismatch(raw.get('schema_version'), ARTIFACT_ROOT_BINDING_SCHEMA_VERSION)}.",
            remediation=OTHER_FORMAT_STEP,
        )
    if raw != expected:
        raise _binding_mismatch(path, raw, expected, experiment)
    return "bound"


def _write_artifact_root_binding(experiment: Experiment) -> None:
    """Write the first ownership record after an unbound tree passes inspection.

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
            f"its storage ledger: {exc}.",
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
    """Check whether an empty study may claim the offered root, without writing.

    Multi-phase callers inspect every declared study before claiming any.

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
                f"Study {study.study_name!r} holds {plural(trial_count, 'trial')} but records no "
                "artifact root, which is unsupported PhaseSweep state.",
            )
        return True
    _check_study_artifact_root(study, experiment)
    return False


def _check_study_artifact_root(study: optuna.Study, experiment: Experiment) -> None:
    """Refuse an existing study bound to another root, without claiming it.

    An unbound study passes; claimability is checked separately. Recovery callers
    apply this ownership check before using a live study.

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
        f"config offers {offered!r}.",
        action=OperatorAction.FIX_CONFIG,
        remediation=restore_or_new_study(
            "the `workdir` and `experiment` that give the recorded artifact root"
        ),
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
    # Only a persistent ledger records the artifact root that owns it across invocations.
    if not experiment.persistent:
        return
    if not _artifact_root_claim_needed(study, experiment):
        return
    _claim_study_artifact_root(study, _artifact_root_identity(experiment))


def _check_published_phase_studies(
    experiment: Experiment,
    loaded: Mapping[str, Sequence[optuna.Study]],
    *,
    from_phase: str | None = None,
) -> None:
    """Refuse an unusable publication, then require published history for phases that would execute.

    The refusal applies to every storage backend, so a launch never publishes past
    a publication that needs operator attention. Only persistent storage carries
    published trials to require; an in-memory ledger starts empty on every launch.

    A published winner came from whichever study of its phase its parents' published winners
    addressed, so every study of the phase is searched for it; attempt identities are unique
    across them.

    :param Experiment experiment: Experiment whose current publication is checked.
    :param Mapping[str, Sequence[optuna.Study]] loaded: Already-inspected persistent studies,
        every study of each phase.
    :param str | None from_phase: Resume point; earlier phases only load winners.
    :raises PublishedStudyMissingError: A reached published trial is absent,
        replaced, or its recorded history boundary is missing.
    :raises PublicationAccessError: The last publication cannot be read.
    :raises PublicationIntegrityError: The last publication no longer validates.
    :raises StudyStorageUnavailableError: A published study's trials cannot be read.
    """
    publication = _resolve_publication_pointer(experiment)
    raise_for_unusable_publication(
        experiment.experiment, publication.state, publication.error, _experiment_dir(experiment)
    )
    if not experiment.persistent:
        return
    published_trials = _published_phase_trial_refs(
        None if publication.validated is None else publication.validated.summary
    )
    for _index, phase in experiment.phases_from(from_phase):
        phase_name = phase.name
        if phase_name not in published_trials:
            continue
        studies = loaded.get(phase_name, ())
        expected = published_trials[phase_name]
        missing = "is missing"
        if studies:
            try:
                every = [study.get_trials(deepcopy=False) for study in studies]
                holding = [
                    trials
                    for trials in every
                    if expected is not None
                    and any(_published_trial_matches(trial, expected) for trial in trials)
                ]
                matched = bool(holding)
                trials = holding[0] if holding else [trial for t in every for trial in t]
                if matched and expected is not None:
                    finished = _finished_trial_count(trials)
                    measured = _measured_trial_count(trials)
                    if _meets_published_trial_history_boundary(finished, measured, expected):
                        continue
                    boundary = ", ".join(
                        f"{count} {label}"
                        for count, label in (
                            (expected.finished_trials, "finished"),
                            (expected.measured_trials, "measured"),
                        )
                        if count is not None
                    )
                    missing = (
                        f"has only {finished} finished and {measured} measured trials, "
                        f"below the published completion boundary ({boundary})"
                    )
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
            f"phase {phase_name!r}, but its persistent study {missing}."
        )
