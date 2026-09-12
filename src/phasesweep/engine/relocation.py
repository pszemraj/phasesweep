"""Artifact-root relocation planning and rebinding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna

from phasesweep.config import Experiment, Suite
from phasesweep.engine.artifact_roots import (
    ARTIFACT_ROOT_BINDING_SCHEMA_VERSION,
    _artifact_root_binding_applies,
    _artifact_root_binding_payload,
    _artifact_root_identity,
    _auto_storage_backend_conflict,
    _check_published_phase_studies,
    _validate_artifact_root_binding,
)
from phasesweep.engine.attempts import _open_attempt_registry
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    ArtifactRootRebindError,
    PublicationAccessError,
    PublicationIntegrityError,
    PublishedStudyMissingError,
    StudyStorageUnavailableError,
    TrialEvidenceMissingError,
)
from phasesweep.engine.evidence import (
    _selection_candidate_identity,
    _verify_trainer_input_evidence,
)
from phasesweep.engine.optuna import (
    _load_existing_phase_study,
)
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _experiment_dir,
    _last_successful_suite_generation_path,
    _phase_dir,
    _suite_dir,
)
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.files import (
    PlatformCapabilityError,
    UnsafePrivatePathError,
    atomic_write_text,
    read_private_text_at,
)
from phasesweep.runtime.json import strict_json_loads


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
    :func:`phasesweep.engine.publication._last_successful_generation_id`,
    the same authoritative read every
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
