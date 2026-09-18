"""Trial and winner evidence validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import optuna

from phasesweep.config import Experiment, Phase
from phasesweep.engine.errors import (
    TrialEvidenceMissingError,
)
from phasesweep.engine.paths import _phase_dir, _trial_dir_for
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_INPUT_ATTR,
    TRAINER_INPUT_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    Winner,
)
from phasesweep.evidence.evaluation import EVIDENCE_PROVENANCE_SCHEMA_VERSION
from phasesweep.evidence.models import _validate_trial_file_path
from phasesweep.runtime.files import (
    file_sha256,
)
from phasesweep.runtime.process import (
    read_attempt_lifecycle,
)

if TYPE_CHECKING:
    from phasesweep.engine.selection import SelectedTrial

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


def _trial_objective_provenance(trial: optuna.trial.FrozenTrial) -> dict[str, Any] | None:
    """Decode a trial's frozen objective-evidence provenance record.

    :param optuna.trial.FrozenTrial trial: Trial whose provenance attr is read.
    :return dict[str, Any] | None: The parsed record, or ``None`` when the
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
    if not isinstance(parsed, dict):
        raise TrialEvidenceMissingError(
            f"Trial {trial.number} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence."
        )
    _validate_objective_provenance(parsed, subject=f"Trial {trial.number}")
    return parsed


def _valid_sha256(value: object) -> bool:
    """Recognize a SHA-256 digest as written by PhaseSweep.

    :param object value: Recorded digest candidate.
    :return bool: Whether it is a lowercase 64-character hex digest.
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_objective_provenance(provenance: Mapping[str, Any], *, subject: str) -> None:
    """Require a present objective record to retain its source binding.

    :param Mapping[str, Any] provenance: Parsed trial or winner provenance.
    :param str subject: Trial or winner label for the diagnostic.
    :raises TrialEvidenceMissingError: The record is incomplete or malformed.
    """

    def fail(reason: str) -> NoReturn:
        """Raise a standardized malformed-provenance error."""
        raise TrialEvidenceMissingError(
            f"{subject} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence "
            f"({reason}). {_TRIAL_EVIDENCE_REMEDY}"
        )

    if provenance.get("schema_version") != EVIDENCE_PROVENANCE_SCHEMA_VERSION:
        fail("unsupported schema version")
    extractor = provenance.get("extractor")
    if not isinstance(extractor, Mapping):
        fail("missing extractor identity")
    kind = extractor.get("kind")
    if kind not in {"json", "json_envelope", "log_regex", "wandb"} or not _valid_sha256(
        extractor.get("config_sha256")
    ):
        fail("invalid extractor identity")
    if not isinstance(provenance.get("recorded_at"), str) or not provenance["recorded_at"]:
        fail("missing capture time")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        fail("missing source")
    if kind == "wandb":
        if source.get("kind") != "wandb" or any(
            not isinstance(source.get(field), str) or not source[field]
            for field in ("base_url", "entity", "project", "run_id", "retrieved_at")
        ):
            fail("missing W&B source address")
        if (
            source.get("run_state") != "finished"
            or not isinstance(source.get("summary"), Mapping)
            or not source["summary"]
        ):
            fail("missing W&B terminal summary")
    elif (
        source.get("kind") != "file"
        or not isinstance(source.get("path"), str)
        or not source["path"]
        or type(source.get("size_bytes")) is not int
        or source["size_bytes"] < 0
        or not _valid_sha256(source.get("sha256"))
    ):
        fail("missing file source path, size, or digest")
    else:
        try:
            _validate_trial_file_path(source["path"])
        except ValueError:
            fail("invalid file source path")


def _verify_objective_source_evidence(
    trial_dir: Path,
    provenance: Mapping[str, Any] | None,
    *,
    subject: str,
    verify_digest: bool,
) -> None:
    """Require a trial's frozen objective source to still be on disk as recorded.

    An absent provenance record is tolerated for trials persisted before the
    record existed (see :class:`phasesweep.engine.selection.SelectedTrial`). A
    present record must be complete. A valid ``wandb`` source names a remote
    run, not a file in this tree.

    :param Path trial_dir: Structurally translated directory for the trial.
    :param Mapping[str, Any] | None provenance: Parsed objective provenance.
    :param str subject: Caller-built label naming the study, phase, and trial.
    :param bool verify_digest: Also re-hash the source and compare it against
        the recorded ``sha256``. Reserved for the winner (see
        :func:`_verify_winner_objective_evidence`).
    :raises TrialEvidenceMissingError: A present record is malformed, or its
        file source is missing, unreadable, or no longer the recorded bytes.
    """
    if provenance is None:
        return
    _validate_objective_provenance(provenance, subject=subject)
    source = provenance["source"]
    assert isinstance(source, Mapping)  # required by _validate_objective_provenance
    if source.get("kind") == "wandb":
        return
    raw_path = str(source["path"])
    candidate = Path(raw_path)
    source_path = candidate if candidate.is_absolute() else trial_dir / candidate
    try:
        stat_result = source_path.stat()
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r}, which is missing or "
            f"unreadable at {str(source_path)!r}. {_TRIAL_EVIDENCE_REMEDY}"
        ) from exc
    recorded_size = int(source["size_bytes"])
    if stat_result.st_size != recorded_size:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r} as "
            f"{recorded_size} bytes, but that file is now {stat_result.st_size} bytes. "
            f"{_TRIAL_EVIDENCE_REMEDY}"
        )
    if not verify_digest:
        return
    recorded_digest = str(source["sha256"])
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
    verify_trainer_input: bool = True,
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
    :func:`phasesweep.engine.relocation._validate_relocated_trial_evidence`).

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
    :param bool verify_trainer_input: Content-verify the generated trainer input.
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
    if verify_trainer_input:
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
        phase name, as returned by :func:`phasesweep.engine.guards._preflight_existing_studies`.
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


def _verify_skipped_winner_evidence(
    experiment: Experiment,
    phase: Phase,
    winner: Winner,
) -> None:
    """Content-verify a skipped winner against its concrete source trial.

    ``--from-phase`` has no reason to require the prior phase's Optuna study:
    the authenticated winner artifact contains the source phase, trial, generation,
    attempt, objective provenance, and (for new publications) generated-input
    identity needed to inspect the source tree directly. This matters for a
    promotion fallback, whose exposed phase did not run the selected trial.

    Older winner artifacts predate serialized ``trainer_input``. Their explicit
    compatibility policy is to retain structural and objective-content verification
    while omitting only the generated-input hash check that the historical artifact
    cannot supply; no ledger lookup is used to fill that gap.

    :param Experiment experiment: Experiment owning the artifact tree.
    :param Phase phase: Exposed skipped phase whose winner is being carried.
    :param Winner winner: Authenticated winner loaded from the prior publication.
    :raises TrialEvidenceMissingError: The winner lacks a coherent concrete source,
        or that source's evidence has been removed or changed.
    """
    source = winner.source
    phase_names = {candidate.name for candidate in experiment.phases}
    if (
        source is None
        or source.phase not in phase_names
        or source.trial_number != winner.trial_number
        or source.generation_id != winner.generation_id
        or source.attempt_id != winner.attempt_id
        or source.generation_id is None
        or source.attempt_id is None
    ):
        raise TrialEvidenceMissingError(
            f"Skipped phase {phase.name!r} winner has no coherent concrete source trial. "
            f"{_TRIAL_EVIDENCE_REMEDY}"
        )
    trial_dir = _trial_dir_for(
        experiment,
        source.phase,
        source.trial_number,
        generation_id=source.generation_id,
        attempt_id=source.attempt_id,
    )
    _verify_trial_evidence_dir(
        trial_dir,
        subject=(
            f"Skipped phase {phase.name!r} winner source phase {source.phase!r} trial "
            f"{source.trial_number} (generation {source.generation_id!r}, attempt "
            f"{source.attempt_id!r})"
        ),
        attempt_id=source.attempt_id,
        provenance=winner.objective_provenance,
        trainer_input=winner.trainer_input,
        verify_objective_digest=True,
        verify_trainer_input=winner.trainer_input is not None,
    )
