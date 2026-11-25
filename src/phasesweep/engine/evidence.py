"""Trial and winner evidence validation."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

import optuna

from phasesweep.config.common import is_sha256_hex
from phasesweep.config.models import Experiment, Phase, _wandb_query
from phasesweep.engine.errors import (
    TrialEvidenceMissingError,
)
from phasesweep.engine.paths import _attempt_dir, _phase_dir, _trial_dir_opener
from phasesweep.engine.repeats import evidence_members, repeat_axis
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_INPUT_ATTR,
    TRAINER_INPUT_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    Winner,
    _required_trial_dir,
    constraint_attr,
)
from phasesweep.engine.trial_dirs import (
    _expected_trial_dir_name,
    _rerun_directory_sources,
    _unavailable_trial_directory,
)
from phasesweep.errors import (
    NEW_STUDY_STEP,
    OTHER_FORMAT_STEP,
    OperatorAction,
    format_mismatch,
    short_digest,
)
from phasesweep.evidence.evaluation import (
    EVIDENCE_PROVENANCE_SCHEMA_VERSION,
    TrialPathError,
    attempt_file,
    extractor_config_fingerprint,
    json_float,
    retained_evidence_dir,
    retained_evidence_file,
    trial_file,
)
from phasesweep.evidence.mlflow import compile_query as compile_mlflow_query
from phasesweep.evidence.models import (
    EXTRACTOR_KINDS,
    JsonExtractor,
    WandbExtractor,
    _validate_trial_file_path,
    _WandbSummarySource,
    is_captured_stream,
)
from phasesweep.runtime.commands import TRAINER_INPUT_FILENAMES
from phasesweep.runtime.files import (
    file_sha256,
    fsync_directory,
)
from phasesweep.runtime.json import finite_nonnegative
from phasesweep.runtime.process import (
    read_attempt_lifecycle,
)
from phasesweep.runtime.terminal import REPORT_LOGGER, plural

if TYPE_CHECKING:
    from phasesweep.engine.selection import SelectedTrial

report = logging.getLogger(REPORT_LOGGER)


# Audit artifacts every launched attempt writes into its own trial directory
# before the trainer starts, so their absence is proof the directory is no
# longer the one that attempt produced.
def _selection_candidate_identity(trial: optuna.trial.FrozenTrial) -> tuple[str, str] | None:
    """Return execution identity for COMPLETE, finite trials with nonempty ids.

    Selection additionally applies the current metric and constraint bounds. Evidence
    preflight excludes that mutable filter, preserving evidence for trials that may become
    eligible after a bounds edit. Ineligible trials cannot affect winner selection.

    :param optuna.trial.FrozenTrial trial: Persisted trial to classify.
    :return tuple[str, str] | None: ``(generation_id, attempt_id)`` for a
        selection-eligible trial, ``None`` for one that can never win.
    """
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return None
    if trial.value is None or not math.isfinite(trial.value):
        return None
    generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if not isinstance(generation_id, str) or not generation_id:
        return None
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    return generation_id, attempt_id


def _earlier_attempts(
    trials: Mapping[int, optuna.trial.FrozenTrial], trial: optuna.trial.FrozenTrial
) -> tuple[tuple[str, str], ...]:
    """Name the earlier attempts that trained in the directory a rerun continues.

    :param Mapping[int, optuna.trial.FrozenTrial] trials: The study's trials by number.
    :param optuna.trial.FrozenTrial trial: A trial continuing its source's directory, or
        one that recorded its directory.
    :raises ProcessCleanupUncertainError: A source is invalid or its recorded directory is malformed.
    :return tuple[tuple[str, str], ...]: ``(generation_id, attempt_id)`` per earlier attempt,
        oldest first; empty for a trial in a directory of its own.
    """
    attempts: list[tuple[str, str]] = []
    for source in _rerun_directory_sources(trials, trial):
        generation_id = source.user_attrs.get(GENERATION_ID_ATTR)
        attempt_id = source.user_attrs.get(ATTEMPT_ID_ATTR)
        if isinstance(generation_id, str) and isinstance(attempt_id, str):
            attempts.append((generation_id, attempt_id))
    return tuple(reversed(attempts))


def _trial_objective_provenance(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Decode a trial's frozen objective-evidence provenance record.

    :param optuna.trial.FrozenTrial trial: Trial whose provenance attr is read.
    :return dict[str, Any]: The parsed record.
    :raises TrialEvidenceMissingError: The record is absent, corrupt, or has an
        unsupported shape.
    """
    if OBJECTIVE_PROVENANCE_ATTR not in trial.user_attrs:
        raise TrialEvidenceMissingError(
            f"Trial {trial.number} has no {OBJECTIVE_PROVENANCE_ATTR!r} evidence."
        )
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


def _file_evidence_fields(source: Mapping[str, object]) -> tuple[str, str, int]:
    """Decode the provenance fields that identify exact scalar evidence bytes.

    :param Mapping[str, object] source: Recorded local file evidence.
    :raises ValueError: A path, digest or size is missing or invalid.
    :return tuple[str, str, int]: Configured path, recorded digest and number of bytes.
    """
    path, digest, size = source.get("path"), source.get("sha256"), source.get("size_bytes")
    if (
        not isinstance(path, str)
        or not path
        or not is_sha256_hex(digest)
        or type(size) is not int
        or size < 0
    ):
        raise ValueError("missing file source path, size, or digest")
    return _validate_trial_file_path(path), digest, size


@dataclass(frozen=True)
class _SharedMeasurement:
    """A measured scalar that still refers to a reused trainer directory."""

    attempt_id: str
    path: str
    digest: str
    size: int


def _shared_measurements(studies: Sequence[optuna.Study]) -> dict[str, set[_SharedMeasurement]]:
    """Index scores in sibling reading studies whose files a retried trainer may replace.

    :param Sequence[optuna.Study] studies: Other studies of the phase.
    :return dict[str, set[_SharedMeasurement]]: Referenced score files by trainer directory name.
    """
    by_directory: dict[str, set[_SharedMeasurement]] = {}
    for study in studies:
        for trial in study.get_trials(deepcopy=False):
            identity = _selection_candidate_identity(trial)
            if identity is None:
                continue
            try:
                directory = _required_trial_dir(trial).name
                provenance = _trial_objective_provenance(trial)
            except (TrialEvidenceMissingError, ValueError):
                continue
            sources = list(provenance.get("local_constraint_files", ()))
            if provenance["source"]["kind"] == "file":
                sources.append(provenance["source"])
            for source in sources:
                path, digest, size = _file_evidence_fields(source)
                if not is_captured_stream(path):
                    measurement = _SharedMeasurement(identity[1], path, digest, size)
                    by_directory.setdefault(directory, set()).add(measurement)
    return by_directory


def _preserve_shared_measurements(
    trial_dir: Path, measurements: Collection[_SharedMeasurement]
) -> None:
    """Retain exact measured files before a later trainer writes in their directory.

    A successful rename publishes only complete, digest-matching bytes. A file already changed
    in the trainer directory cannot be saved as this attempt's evidence.

    :param Path trial_dir: Directory the trainer will reuse.
    :param Collection[_SharedMeasurement] measurements: Measurements at risk from writes here.
    :raises OSError: A source could not be inspected, or matching bytes could not be retained.
    """
    for measurement in measurements:
        attempt_dir = _attempt_dir(trial_dir.parent, measurement.attempt_id)
        if (
            retained_evidence_file(attempt_dir, measurement.path, digest=measurement.digest)
            is not None
        ):
            continue
        try:
            original = trial_file(trial_dir, attempt_dir, measurement.path)
            if original.stat().st_size != measurement.size:
                continue
            source = original.open("rb")
        except (FileNotFoundError, TrialPathError):
            continue
        staged = None
        with source:
            directory = retained_evidence_dir(attempt_dir, measurement.path)
            destination = attempt_file(
                attempt_dir, (directory / measurement.digest).relative_to(attempt_dir)
            )
            directory.mkdir(parents=True, exist_ok=True)
            prefix = f".capture-{measurement.digest}-"
            # Reruns of this directory are serialized. A killed run can leave its stage behind.
            for abandoned in directory.glob(f"{prefix}*"):
                abandoned.unlink()
            try:
                with NamedTemporaryFile(dir=directory, prefix=prefix, delete=False) as handle:
                    staged = Path(handle.name)
                    shutil.copyfileobj(source, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                if file_sha256(staged) == measurement.digest:
                    staged.replace(destination)
                    fsync_directory(directory)
                    fsync_directory(directory.parent)
                    fsync_directory(attempt_dir)
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)


def _validate_objective_provenance(provenance: Mapping[str, Any], *, subject: str) -> None:
    """Require a present objective record to retain its source binding.

    :param Mapping[str, Any] provenance: Parsed trial or winner provenance.
    :param str subject: Trial or winner label for the diagnostic.
    :raises TrialEvidenceMissingError: The record is incomplete or malformed.
    """

    def fail(reason: str) -> NoReturn:
        """Raise a standardized malformed-provenance error.

        :raises TrialEvidenceMissingError: Always, naming ``subject`` and ``reason``.
        """
        raise TrialEvidenceMissingError(
            f"{subject} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence ({reason})."
        )

    if provenance.get("schema_version") != EVIDENCE_PROVENANCE_SCHEMA_VERSION:
        found = format_mismatch(
            provenance.get("schema_version"), EVIDENCE_PROVENANCE_SCHEMA_VERSION
        )
        raise TrialEvidenceMissingError(
            f"{subject} has {OBJECTIVE_PROVENANCE_ATTR!r} evidence that {found}.",
            action=OperatorAction.FRESH_NAMESPACE,
            # The evidence is a trial attribute, so it stays in an explicit ledger.
            remediation=OTHER_FORMAT_STEP,
        )
    extractor = provenance.get("extractor")
    if not isinstance(extractor, Mapping):
        fail("missing extractor identity")
    kind = extractor.get("kind")
    if kind not in EXTRACTOR_KINDS or not is_sha256_hex(extractor.get("config_sha256")):
        fail("invalid extractor identity")
    if not isinstance(provenance.get("recorded_at"), str) or not provenance["recorded_at"]:
        fail("missing capture time")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        fail("missing source")
    constraint_files = provenance.get("local_constraint_files", [])
    if not isinstance(constraint_files, list):
        fail("invalid local constraint files")
    for file in constraint_files:
        if not isinstance(file, Mapping):
            fail("invalid local constraint file")
        try:
            _file_evidence_fields(file)
        except ValueError as exc:
            fail(f"invalid local constraint file: {exc}")
    capture = provenance.get("remote_capture")
    if isinstance(capture, Mapping) and capture.get("kind") == "mlflow":
        from phasesweep.evidence.mlflow import validate_capture

        try:
            validate_capture(capture)
        except (ValueError, KeyError, TypeError) as exc:
            fail(f"invalid MLflow capture: {exc}")
    elif capture is not None:
        if not isinstance(capture, Mapping) or capture.get("kind") != "wandb":
            fail("invalid remote capture")
        for field in ("base_url", "entity", "project", "run_id", "retrieved_at"):
            if not isinstance(capture.get(field), str) or not capture[field]:
                fail(f"missing remote {field}")
        try:
            target = _WandbSummarySource(
                base_url=capture["base_url"], entity=capture["entity"], project=capture["project"]
            )
        except ValueError:
            fail("invalid remote target")
        if target.base_url != capture["base_url"] or capture.get("run_state") != "finished":
            fail("remote target is not normalized or run is not finished")
        values = capture.get("values")
        present = capture.get("present_keys")
        if (
            not isinstance(values, Mapping)
            or not isinstance(present, list)
            or any(not isinstance(key, str) or not key for key in present)
        ):
            fail("invalid remote summary subset")
        for key, value in values.items():
            try:
                valid = (
                    isinstance(key, str)
                    and bool(key)
                    and math.isfinite(json_float(value, label=key))
                )
            except ValueError:
                valid = False
            if not valid:
                fail("invalid remote numeric evidence")
    if kind == "mlflow":
        if (
            source.get("kind") != "mlflow"
            or not isinstance(capture, Mapping)
            or capture.get("kind") != "mlflow"
        ):
            fail("MLflow objective has no native capture")
        point = capture["observations"].get(source.get("selection"))
        evaluation = source.get("evaluation")
        if (
            not isinstance(point, Mapping)
            or not isinstance(evaluation, Mapping)
            or evaluation.get("progress") != {"axis": "step", "value": point["step"]}
            or evaluation.get("timestamp") != point["timestamp"]
        ):
            fail("MLflow objective has no matching native position")
        return
    if kind == "wandb":
        if source.get("kind") != "wandb" or not isinstance(capture, Mapping):
            fail("W&B objective has no remote capture")
        key = source.get("metric_key")
        if not isinstance(key, str) or not key or key not in capture["values"]:
            fail("W&B objective has no captured metric key")
        evaluation = source.get("evaluation")
        progress = evaluation.get("progress") if isinstance(evaluation, Mapping) else None
        if not _valid_progress_record(progress, nullable=True):
            fail("W&B objective has no evaluation-axis record")
        return
    if source.get("kind") != "file":
        fail("missing file source")
    try:
        _file_evidence_fields(source)
    except ValueError as exc:
        fail(str(exc))
    if kind == "json_envelope":
        evaluation = source.get("evaluation")
        if (
            not isinstance(evaluation, Mapping)
            or not isinstance(evaluation.get("checkpoint"), str)
            or not evaluation["checkpoint"]
            or type(evaluation.get("step")) is not int
            or evaluation["step"] < 0
        ):
            fail("envelope objective has no evaluation checkpoint or step")
        if not _valid_progress_record(evaluation.get("progress"), nullable=False):
            fail("envelope objective has no evaluation-axis position")


def _valid_progress_record(record: object, *, nullable: bool) -> TypeGuard[Mapping[str, Any]]:
    """Check a recorded evaluation-axis position: ``{axis, value}``.

    :param object record: Parsed ``source.evaluation.progress`` record.
    :param bool nullable: Whether ``value`` may be ``None``, meaning the
        source could not say where the objective was measured.
    :return TypeGuard[Mapping[str, Any]]: Whether the record names an axis
        and a valid position.
    """
    if not isinstance(record, Mapping) or set(record) != {"axis", "value"}:
        return False
    if not isinstance(record["axis"], str) or not record["axis"]:
        return False
    if record["value"] is None:
        return nullable
    return finite_nonnegative(record["value"]) is not None


def _verify_objective_source_evidence(
    trial_dir: Path,
    attempt_dir: Path,
    provenance: Mapping[str, Any] | None,
    *,
    subject: str,
    verify_digest: bool,
) -> None:
    """Verify the retained local source or the frozen remote capture.

    Local records name a file found where extraction read it, through
    :func:`phasesweep.evidence.evaluation.trial_file`. Remote records are validated
    without contacting the service again.

    :param Path trial_dir: Structurally translated directory for the trial.
    :param Path attempt_dir: The producing attempt's directory, holding its captured streams.
    :param Mapping[str, Any] | None provenance: Parsed objective provenance.
    :param str subject: Caller-built label naming the study, phase, and trial.
    :param bool verify_digest: Also re-hash the source and compare it against
        the recorded ``sha256``. Reserved for the winner (see
        :func:`_verify_winner_evidence`).
    :raises TrialEvidenceMissingError: The record is absent or malformed, or
        its file source is missing, unreadable, outside the attempt, or no longer the
        recorded bytes.
    """
    if provenance is None:
        raise TrialEvidenceMissingError(f"{subject} has no objective provenance record.")
    _validate_objective_provenance(provenance, subject=subject)
    source = provenance["source"]
    assert isinstance(source, Mapping)  # required by _validate_objective_provenance
    if source["kind"] in {"wandb", "mlflow"}:
        return
    # _validate_objective_provenance refused any path but a trial-relative one.
    raw_path = str(source["path"])
    try:
        retained = retained_evidence_file(attempt_dir, raw_path, digest=str(source["sha256"]))
        source_path = (
            retained if retained is not None else trial_file(trial_dir, attempt_dir, raw_path)
        )
    except TrialPathError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r}, but {exc}"
        ) from exc
    try:
        stat_result = source_path.stat()
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r}, which is missing or "
            f"unreadable at {str(source_path)!r}."
        ) from exc
    recorded_size = int(source["size_bytes"])
    if stat_result.st_size != recorded_size:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r} as "
            f"{recorded_size} bytes, but that file is now {stat_result.st_size} bytes at "
            f"{str(source_path)!r}."
        )
    if not verify_digest:
        return
    recorded_digest = str(source["sha256"])
    try:
        actual_digest = file_sha256(source_path)
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r}, which could not be "
            f"read back for verification at {str(source_path)!r}."
        ) from exc
    if actual_digest != recorded_digest:
        raise TrialEvidenceMissingError(
            f"{subject} recorded its objective evidence in {raw_path!r} with sha256 "
            f"{short_digest(recorded_digest)}, but that file now hashes to "
            f"{short_digest(actual_digest)} at {str(source_path)!r}: its bytes have changed."
        )


def _verify_trainer_input_evidence(
    trial_dir: Path,
    record: Any,
    *,
    subject: str,
) -> None:
    """Verify the generated input recorded by a trial, independently of current config.

    Require the versioned format/filename pair to identify a trial-relative file that
    resolves inside the attempt directory.

    :param Path trial_dir: Structurally translated trial evidence directory.
    :param Any record: Raw ``TRAINER_INPUT_ATTR`` Optuna user attribute.
    :param str subject: Caller-built trial label for diagnostics.
    :raises TrialEvidenceMissingError: The record is absent or malformed, or
        its exact file bytes are missing, outside the attempt, or changed.
    """
    if not isinstance(record, Mapping):
        raise TrialEvidenceMissingError(
            f"{subject} has no valid {TRAINER_INPUT_ATTR!r} record, so its historical "
            f"trainer input cannot be verified."
        )
    input_format = record.get("format")
    filename = record.get("filename")
    expected_filename = (
        TRAINER_INPUT_FILENAMES.get(input_format) if isinstance(input_format, str) else None
    )
    if (
        record.get("schema_version") != TRAINER_INPUT_SCHEMA_VERSION
        or expected_filename is None
        or filename != expected_filename
    ):
        raise TrialEvidenceMissingError(
            f"{subject} has a malformed or unsupported {TRAINER_INPUT_ATTR!r} record "
            f"{dict(record)!r}, so its historical trainer input cannot be located safely."
        )
    recorded_size = record.get("size_bytes")
    recorded_digest = record.get("sha256")
    if (
        not isinstance(recorded_size, int)
        or isinstance(recorded_size, bool)
        or recorded_size < 0
        or not is_sha256_hex(recorded_digest)
    ):
        raise TrialEvidenceMissingError(
            f"{subject} has an invalid size or content identity in its "
            f"{TRAINER_INPUT_ATTR!r} record."
        )

    try:
        input_path = attempt_file(trial_dir, filename)
    except TrialPathError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} recorded trainer input {filename!r}, but {exc}"
        ) from exc
    if not input_path.is_file():
        raise TrialEvidenceMissingError(
            f"{subject} is missing its recorded trainer input {filename!r} under "
            f"{str(trial_dir)!r}."
        )
    try:
        actual_size = input_path.stat().st_size
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} cannot read its recorded trainer input {filename!r} at {str(input_path)!r}."
        ) from exc
    if actual_size != recorded_size:
        raise TrialEvidenceMissingError(
            f"{subject} recorded trainer input {filename!r} as {recorded_size} bytes, but "
            f"that file is now {actual_size} bytes at {str(input_path)!r}."
        )
    try:
        actual_digest = file_sha256(input_path)
    except OSError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} cannot content-verify its recorded trainer input {filename!r} at "
            f"{str(input_path)!r}."
        ) from exc
    if actual_digest != recorded_digest:
        raise TrialEvidenceMissingError(
            f"{subject} recorded trainer input {filename!r} with sha256 "
            f"{short_digest(recorded_digest)}, but that file now hashes to "
            f"{short_digest(actual_digest)} at {str(input_path)!r}: its bytes have changed."
        )


class _EvidenceVerification:
    """Phase evidence bindings shared by all trial-directory checks."""

    def __init__(self, experiment: Experiment, phase: Phase) -> None:
        """Compile the exposed phase's evidence requirements.

        :param Experiment experiment: Configured primary extractor and remote target.
        :param Phase phase: Exposed phase supplying its constraint and gate requirements.
        """
        self.constraint_names = [constraint.name for constraint in experiment.constraints]
        self.extractor = experiment.metric.extractor
        self.wandb_query = _wandb_query(experiment, phase.gates)
        self.mlflow_query = compile_mlflow_query(experiment, phase.gates)
        self.mlflow_target = (
            None if experiment.mlflow is None else experiment.mlflow.evidence_identity()
        )

    def verify_trial(
        self,
        trial: optuna.trial.FrozenTrial,
        trial_dir: Path,
        generation_id: str,
        attempt_id: str,
        metric: float,
        *,
        subject: str,
        verify_digest: bool,
    ) -> None:
        """Verify a ledger measurement against the attempt's retained evidence.

        :param optuna.trial.FrozenTrial trial: Trial holding the input, provenance and constraints.
        :param Path trial_dir: Trial directory the producing attempt ran in.
        :param str generation_id: Generation that owns the attempt.
        :param str attempt_id: Attempt whose evidence is required.
        :param float metric: Recorded objective.
        :param str subject: Trial description for a refusal.
        :param bool verify_digest: Whether to hash the primary objective file.
        :raises TrialEvidenceMissingError: Evidence for the measurement is absent or changed.
        """
        _verify_trial_evidence_dir(
            trial_dir,
            subject=subject,
            generation_id=generation_id,
            attempt_id=attempt_id,
            provenance=_trial_objective_provenance(trial),
            trainer_input=trial.user_attrs.get(TRAINER_INPUT_ATTR),
            verify_objective_digest=verify_digest,
            verification=self,
            metric_value=metric,
            constraints={
                name: trial.user_attrs.get(constraint_attr(name)) for name in self.constraint_names
            },
        )


def _verify_trial_evidence_dir(
    trial_dir: Path,
    *,
    subject: str,
    generation_id: str,
    attempt_id: str,
    provenance: Mapping[str, Any] | None,
    trainer_input: Any,
    verify_objective_digest: bool,
    verification: _EvidenceVerification,
    metric_value: float,
    constraints: Mapping[str, Any],
) -> None:
    """Require a trial's translated directories and its recorded artifacts.

    The trainer input and objective source live in the trial directory; the lifecycle
    record and audit artifacts in the attempt's directory beside it. MLflow metadata
    stays in the producing generation's tracking directory. A lifecycle record must
    belong to the attempt. Its state is not constrained: writing
    ``exited`` is best-effort, so completed trials may retain ``launching``.

    :param Path trial_dir: Translated trial directory the attempt ran in.
    :param str subject: Caller-built label naming the study, phase, and trial.
    :param str generation_id: Producing generation that owns the attempt.
    :param str attempt_id: Attempt identity the lifecycle record must belong to.
    :param Mapping[str, Any] | None provenance: Parsed objective provenance.
    :param Any trainer_input: Versioned historical generated-input record.
    :param bool verify_objective_digest: Re-hash the objective source as well.
    :param _EvidenceVerification verification: Phase evidence bindings.
    :param float metric_value: Selected scalar recorded by the study or winner.
    :param Mapping[str, Any] constraints: Saved constraint measurements.
    :raises TrialEvidenceMissingError: The directory, an audit artifact, the
        generated trainer input, or recorded objective source is missing,
        foreign, or altered.
    """
    attempt_dir = _attempt_dir(trial_dir.parent, attempt_id)
    for directory in (trial_dir, attempt_dir):
        if not directory.is_dir():
            raise TrialEvidenceMissingError(
                f"{subject} has no evidence directory at {str(directory)!r}.",
                remediation=f"Restore {directory}.",
            )
    try:
        lifecycle = read_attempt_lifecycle(attempt_dir, expected_attempt_id=attempt_id)
    except ValueError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} has an attempt lifecycle record that is malformed or belongs to "
            f"another attempt ({exc})."
        ) from exc
    if lifecycle is None:
        raise TrialEvidenceMissingError(f"{subject} has no attempt lifecycle record.")
    # The resolved overrides are the trial's input; the command is the attempt's own record.
    for directory, filename in (
        (trial_dir, "overrides_resolved.json"),
        (attempt_dir, "command.txt"),
    ):
        if not (directory / filename).is_file():
            raise TrialEvidenceMissingError(
                f"{subject} is missing its {filename!r} audit artifact under {str(directory)!r}."
            )
    _verify_trainer_input_evidence(trial_dir, trainer_input, subject=subject)
    _verify_objective_source_evidence(
        trial_dir,
        attempt_dir,
        provenance,
        subject=subject,
        verify_digest=verify_objective_digest,
    )
    assert provenance is not None
    recorded_extractor = provenance["extractor"]
    if recorded_extractor["kind"] != verification.extractor.type or recorded_extractor[
        "config_sha256"
    ] != extractor_config_fingerprint(verification.extractor):
        raise TrialEvidenceMissingError(
            f"{subject} objective evidence no longer matches its configured extractor.",
            action=OperatorAction.FRESH_NAMESPACE,
            remediation=NEW_STUDY_STEP,
        )
    capture = provenance.get("remote_capture")
    if capture is not None and capture.get("kind") == "wandb":
        try:
            trial_run = _trial_dir_opener(trial_dir)
        except ValueError:
            trial_run = None
        if capture["run_id"] != trial_run:
            raise TrialEvidenceMissingError(
                f"{subject} W&B capture belongs to another trial's run."
            )
    if verification.wandb_query is not None:
        if (
            capture is None
            or any(
                capture[field] != getattr(verification.wandb_query.source, field)
                for field in ("base_url", "entity", "project")
            )
            or set(capture["values"]) != set(verification.wandb_query.numeric_keys)
            or capture.get("constraint_keys") != dict(verification.wandb_query.constraint_keys)
            or capture.get("gate_keys")
            != {identity: list(keys) for identity, keys in verification.wandb_query.gate_keys}
            or any(
                capture["values"].get(key) != constraints.get(name)
                for name, key in verification.wandb_query.constraint_keys
            )
            or not set(verification.wandb_query.presence_keys).issubset(capture["present_keys"])
        ):
            raise TrialEvidenceMissingError(
                f"{subject} W&B capture disagrees with its shared constraint/gate evidence."
            )
    elif verification.mlflow_query is not None:
        from phasesweep.evidence.mlflow import observation_id
        from phasesweep.evidence.models import MlflowExtractor
        from phasesweep.tracking.session import (
            EXPECTED_BINDING_FILE,
            attempt_tracking_dir,
            read_binding,
            read_record,
        )

        tracking_dir = attempt_tracking_dir(trial_dir.parent.parent, generation_id, attempt_id)
        try:
            binding = read_binding(tracking_dir)
            expected = read_record(tracking_dir / EXPECTED_BINDING_FILE)
            if (
                capture is None
                or capture.get("kind") != "mlflow"
                or capture["binding"] != binding
                or capture["query"] != verification.mlflow_query
                or binding["generation_id"] != generation_id
                or binding["attempt_id"] != attempt_id
                or binding["overrides_sha256"] != trainer_input["sha256"]
                or any(binding.get(k) != v for k, v in expected.items())
                or verification.mlflow_target is None
                or binding["tracking_uri"] != verification.mlflow_target["tracking_uri"]
                or binding["run_mode"] != verification.mlflow_target["run_mode"]
                or (
                    verification.mlflow_target["experiment_id"] is not None
                    and binding["experiment_id"] != verification.mlflow_target["experiment_id"]
                )
            ):
                raise ValueError("binding, target, or shared query differs")
            # The merged history's earlier runs must still be the ones their attempts bound.
            for prior in capture["earlier_bindings"]:
                prior_dir = attempt_tracking_dir(
                    trial_dir.parent.parent, prior["generation_id"], prior["attempt_id"]
                )
                if read_binding(prior_dir) != prior:
                    raise ValueError(f"earlier attempt {prior['attempt_id']}'s binding differs")
            for name, specification in verification.mlflow_query["constraints"].items():
                point = capture["observations"][
                    observation_id(MlflowExtractor.model_validate(specification))
                ]
                if point["value"] != constraints.get(name):
                    raise ValueError("constraint selection differs")
            spec = MlflowExtractor.model_validate(verification.mlflow_query["objective"])
            identity = observation_id(spec)
            if (
                capture["observations"][identity]["value"] != metric_value
                or provenance["source"].get("selection") != identity
                or provenance["source"].get("metric_key") != spec.metric_key
            ):
                raise ValueError("objective selection differs")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TrialEvidenceMissingError(
                f"{subject} MLflow evidence disagrees with its bound attempt: {exc}."
            ) from exc
    elif capture is not None:
        raise TrialEvidenceMissingError(f"{subject} has undeclared remote evidence.")
    if isinstance(verification.extractor, WandbExtractor):
        source = provenance["source"]
        if (
            capture is None
            or any(
                capture[field] != getattr(verification.extractor, field)
                for field in ("base_url", "entity", "project")
            )
            or source.get("metric_key") != verification.extractor.metric_key
            or source["evaluation"]["progress"]["axis"] != verification.extractor.evaluation_axis
            or capture["values"].get(verification.extractor.metric_key) != metric_value
        ):
            raise TrialEvidenceMissingError(
                f"{subject} W&B evidence disagrees with its configured source or selected scalar."
            )
    if isinstance(verification.extractor, JsonExtractor) and (
        provenance["source"].get("path") != verification.extractor.path
        or provenance["source"].get("key") != verification.extractor.key
    ):
        raise TrialEvidenceMissingError(
            f"{subject} JSON evidence disagrees with its configured path/key."
        )


def pruned_trials_note(numbers: Sequence[int]) -> str:
    """Say which finished trials have no directory and that they keep their ranking.

    :param Sequence[int] numbers: Trial numbers whose directory is gone, ascending; not empty.
    :return str: One line for the phase's note, such as ``1 finished trial has no directory
        (trial 0); it ranks by its recorded value``.
    """
    if len(numbers) == 1:
        return f"1 finished trial has no directory (trial {numbers[0]}); it ranks by its recorded value"
    listed = ", ".join(str(number) for number in numbers)
    return (
        f"{plural(len(numbers), 'finished trial')} have no directory (trials {listed}); "
        "they rank by their recorded values"
    )


def _selection_evidence(
    experiment: Experiment, phase: Phase, study: optuna.Study
) -> tuple[int, ...]:
    """Verify the evidence of every trial that could win a phase's selection.

    Check selection eligibility through :func:`_selection_candidate_identity`. A candidate
    whose whole trial directory is gone, by the rule a rerun uses to open a fresh one
    (:func:`phasesweep.engine.trial_dirs._unavailable_trial_directory`), keeps its recorded
    value and its place in the ranking: that value was verified when it was measured, and
    deleting a finished trial's directory is how an operator frees disk. Only the trial the
    ranking selects needs its directory, because a published winner names it and selection
    digest-verifies it (:func:`_verify_winner_evidence`). Every other candidate's generated
    input is content-verified and its objective source's existence and recorded size checked.

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param Phase phase: The phase the study belongs to.
    :param optuna.Study study: The phase study a run would select from.
    :raises TrialEvidenceMissingError: A selection-eligible trial records an unusable trial
        directory name; its directory exists but its attempt records, audit artifacts,
        generated trainer input, or recorded objective source are no longer in this tree; or
        the trial the ranking selects (any member of its group, with ``repeats``) has no
        directory.
    :raises PhaseSweepError: A trial directory's path could not be inspected.
    :return tuple[int, ...]: The candidates whose directory is gone, ascending.
    """
    phase_dir = _phase_dir(experiment, phase.name)
    verification = _EvidenceVerification(experiment, phase)
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    pruned: dict[int, Path] = {}
    for trial in trials.values():
        identity = _selection_candidate_identity(trial)
        if identity is None:
            continue
        generation_id, attempt_id = identity
        assert trial.value is not None
        subject = (
            f"Study {study.study_name!r} phase {phase.name!r} trial {trial.number} "
            "is eligible to win selection but"
        )
        try:
            recorded = _required_trial_dir(trial).name
        except ValueError:
            raise TrialEvidenceMissingError(
                f"{subject} records an invalid {TRIAL_DIR_ATTR!r} user attribute "
                f"{trial.user_attrs.get(TRIAL_DIR_ATTR)!r}, so its evidence cannot be "
                "located."
            ) from None
        # The directory must name its own opener, even after a fresh-directory rerun.
        expected_name = _expected_trial_dir_name(trials, trial)
        if expected_name is None or recorded != expected_name:
            raise TrialEvidenceMissingError(
                f"{subject} records evidence directory {recorded!r}, which does "
                f"not name the attempt that opened it (expected {expected_name!r})."
            )
        trial_dir = phase_dir / expected_name
        if _unavailable_trial_directory(trial_dir) is not None:
            pruned[trial.number] = trial_dir
            continue
        verification.verify_trial(
            trial,
            trial_dir,
            generation_id,
            attempt_id,
            float(trial.value),
            subject=subject,
            verify_digest=False,
        )
    if pruned:
        _refuse_a_pruned_winner(experiment, phase, study, pruned)
    return tuple(sorted(pruned))


def _refuse_a_pruned_winner(
    experiment: Experiment, phase: Phase, study: optuna.Study, pruned: Mapping[int, Path]
) -> None:
    """Refuse a phase whose ranking selects a trial whose directory is gone.

    No mechanism measures one finished trial again, so the remedy is the directory itself, or
    a renamed phase, whose new study trains every trial again while a child whose inherited
    values it selects again keeps its own study.

    :param Experiment experiment: Parsed experiment whose goal and bounds rank the trials.
    :param Phase phase: The phase the study belongs to.
    :param optuna.Study study: The phase study a run would select from.
    :param Mapping[int, Path] pruned: Each candidate whose directory is gone, by trial number.
    :raises TrialEvidenceMissingError: The selected trial, or a member of its group, is pruned.
    """
    from phasesweep.engine.selection import NoFeasibleTrialError, select_winner

    try:
        selected = select_winner(
            study,
            experiment,
            phase_name=phase.name,
            repeats=repeat_axis(phase),
            emit_advisories=False,
        )
    except NoFeasibleTrialError:
        return
    for member in evidence_members(selected):
        trial_dir = pruned.get(member.trial_number)
        if trial_dir is not None:
            raise TrialEvidenceMissingError(
                f"Study {study.study_name!r} phase {phase.name!r} would select trial "
                f"{member.trial_number}, whose directory {str(trial_dir)!r} is gone; only a "
                "winner needs its directory.",
                remediation=(
                    f"Restore {trial_dir}, or rename phase {phase.name!r} to retrain its trials "
                    "in a new study (later phases keep theirs if it selects the same values)."
                ),
            )


def _validate_selection_evidence(
    experiment: Experiment,
    studies: Mapping[str, optuna.Study],
) -> None:
    """Require on-disk evidence for the trials that could win, and note pruned ones.

    Each phase's candidates are judged by :func:`_selection_evidence`. A phase with
    candidates whose directory is gone logs one line naming them.

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param Mapping[str, optuna.Study] studies: Phase studies a run would select from, keyed by
        phase name.
    :raises TrialEvidenceMissingError: A candidate's evidence is present but no longer the
        recorded evidence, or the trial a phase would select has no directory.
    :raises PhaseSweepError: A trial directory's path could not be inspected.
    """
    for phase_name, study in studies.items():
        pruned = _selection_evidence(experiment, experiment.phase(phase_name), study)
        if pruned:
            report.info("%s  %s", phase_name, pruned_trials_note(pruned))


def _verify_winner_evidence(
    experiment: Experiment,
    phase: Phase,
    winner: SelectedTrial | Winner,
) -> None:
    """Digest-verify a selected or carried winner against each member's own evidence.

    Carried winners use their recorded source phase without loading its study.
    Scalar winners require only their source trial; repeat winners require every member.

    :param Experiment experiment: Parsed experiment naming the artifact tree.
    :param Phase phase: Phase whose winner is selected or carried into the publication.
    :param SelectedTrial | Winner winner: Selected trial or authenticated carried winner.
    :raises TrialEvidenceMissingError: The evidence directory, audit artifacts,
        generated trainer input, or objective source of the winner or of any
        member of its group are missing, foreign, or altered.
    """
    if isinstance(winner, Winner):
        source = winner.source
        assert source is not None
        phase_name = source.phase
        subject = f"Skipped phase {phase.name!r} winner source phase {phase_name!r} trial"
    else:
        phase_name = phase.name
        label = "winner trial" if winner.repeats is None else "winner group member trial"
        subject = f"Phase {phase_name!r} {label}"
    verification = _EvidenceVerification(experiment, phase)
    for member in evidence_members(winner):
        _verify_trial_evidence_dir(
            _phase_dir(experiment, phase_name) / member.trial_dir,
            subject=(
                f"{subject} {member.trial_number} "
                f"(generation {member.generation_id!r}, attempt {member.attempt_id!r})"
            ),
            generation_id=member.generation_id,
            attempt_id=member.attempt_id,
            provenance=member.objective_provenance,
            trainer_input=member.trainer_input,
            verify_objective_digest=True,
            verification=verification,
            metric_value=member.metric,
            constraints=member.constraints,
        )
