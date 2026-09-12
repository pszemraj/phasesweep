"""Artifact serialization, logging, and winner persistence."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import optuna
import yaml

import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.paths as path_ops
import phasesweep.engine.publication as publication_ops
from phasesweep.config import Experiment, Phase
from phasesweep.engine.errors import StudyFingerprintMismatchError, WinnerIntegrityError
from phasesweep.engine.state import (
    Winner,
    WinnerSourceKind,
    _parse_winner_source,
    _winner_source_or_default,
)
from phasesweep.runtime.files import atomic_text_writer, fsync_directory

log = logging.getLogger(__name__)


def _write_yaml_atomic(path: Path, payload: Any) -> None:
    """Atomically write a YAML document to ``path``.

    :param Path path: Destination YAML path to replace.
    :param Any payload: YAML-serializable value to write.
    """
    text = yaml.safe_dump(payload, sort_keys=False)
    with atomic_text_writer(path, newline="") as handle:
        handle.write(text)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write a JSON document to ``path`` with the artifact-tree mode.

    :param Path path: Destination JSON path to replace.
    :param Any payload: JSON-serializable value to write.
    :raises TypeError: ``payload`` is not JSON-serializable.
    :raises OSError: The document could not be staged, written, or renamed.
    """
    with atomic_text_writer(path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml_exclusive(path: Path, payload: Any) -> bool:
    """Create a YAML document at ``path`` exactly once, never overwriting it.

    Unlike :func:`_write_yaml_atomic` (always-overwrite, used for mutable
    pointers), this is the create-exclusive primitive backing truly immutable
    per-generation lifecycle records (review v0.5.15 / blocker 3): the file is
    opened with ``O_CREAT | O_EXCL`` so a second call for an already-written
    path can never clobber the first write, even a same-content rewrite. This
    is a plain create-once-and-fsync, not a full atomic-rename dance like
    :func:`atomic_text_writer` -- there is nothing to make atomic against a
    concurrent *reader* here, only against a second *writer*, and ``O_EXCL``
    already rules that out.

    :param Path path: Destination path to create; the parent directory is
        created if missing.
    :param Any payload: YAML-serializable value to write.
    :return bool: ``True`` when this call created and wrote the file;
        ``False`` when the destination already existed and nothing was
        written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)
    return True


@contextlib.contextmanager
def _file_log_handler(path: Path) -> Iterator[None]:
    """Attach a durable file handler for phasesweep logs.

    :param Path path: Log file path to append to.
    :return Iterator[None]: Context manager that removes the handler on exit.
    """
    logger = logging.getLogger("phasesweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname).1s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.DEBUG)
    old_level = logger.level
    if old_level in (logging.NOTSET, 0) or old_level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)


def _write_trials_csv(study: optuna.Study, path: Path) -> None:
    """Snapshot every trial in ``study`` to ``path`` as stdlib CSV.

    :param optuna.Study study: Study whose trials are serialized.
    :param Path path: Destination CSV path.
    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    param_names = sorted({n for t in trials for n in t.params})
    attr_names = sorted({n for t in trials for n in t.user_attrs})
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration",
        *[f"param:{n}" for n in param_names],
        *[f"user_attr:{n}" for n in attr_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            row: dict[str, Any] = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete,
                "duration": t.duration,
            }
            for n in param_names:
                row[f"param:{n}"] = t.params.get(n)
            for n in attr_names:
                row[f"user_attr:{n}"] = t.user_attrs.get(n)
            writer.writerow(row)


def _save_winner(
    experiment: Experiment,
    phase_name: str,
    winner: Winner,
    *,
    generation_id: str,
) -> None:
    """Persist a phase winner into its immutable generation namespace.

    The phase fingerprint is included so ``_load_winner`` can refuse stale
    winners on ``--from-phase`` resume (review v0.5.6 / blocker 3). Real
    winners always carry a fingerprint by construction in ``_run_phase``;
    placeholder winners (dry-run skip) are never saved.

    ``trainer_env_digest`` / ``trainer_inherit_env`` record which environment
    produced the winning trial (review v0.5.18 / finding F3). Both are
    ``None`` on winners selected from trials that predate the record; neither
    ever carries ambient variable values.

    Args:
        experiment: Parsed experiment config; supplies the metric name used
            in the persisted payload.
        phase_name: Name of the phase whose winner is being saved.
        winner: The winning trial.
        generation_id: Immutable generation namespace to write into. The
            legacy compatibility projection is produced separately by
            :func:`phasesweep.engine.generation._copy_yaml_projection` once a generation is published.

    """
    path = path_ops._generation_winner_path(experiment, generation_id, phase_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase_name,
        "metric": {experiment.metric.name: winner.metric, "goal": experiment.metric.goal},
        **_winner_common_payload(winner, phase_name),
        "phase_fingerprint": winner.phase_fingerprint,
        "objective_provenance": winner.objective_provenance,
        "trainer_env_digest": winner.trainer_env_digest,
        "trainer_inherit_env": winner.trainer_inherit_env,
    }
    _write_yaml_atomic(path, payload)


def _winner_common_payload(winner: Winner, phase_name: str) -> dict[str, Any]:
    """Serialize winner fields shared by persisted and summary representations.

    :param Winner winner: Winner whose common fields should be serialized.
    :param str phase_name: Phase exposed by this winner.
    :return dict[str, Any]: Shared trial, parameter, evidence, identity, and source fields.
    """
    payload = {
        "trial_number": winner.trial_number,
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
        "completion": winner.completion,
        "generation_id": winner.generation_id,
        "attempt_id": winner.attempt_id,
        "winner_source": _winner_source_payload(winner, phase_name),
    }
    if winner.promotion is not None:
        payload["promotion"] = winner.promotion
    return payload


def _winner_source_payload(winner: Winner, phase_name: str) -> dict[str, Any]:
    """Serialize the concrete source trial for an exposed winner.

    :param Winner winner: Winner whose recorded ``source`` is serialized; when
        unset, a ``phase_trial`` source is synthesized from the winner's own fields.
    :param str phase_name: Phase name used to synthesize a fallback source when
        ``winner.source`` is unset.
    :return dict[str, Any]: JSON-serializable winner-source payload with
        ``kind``, ``phase``, ``trial_number``, ``generation_id``, ``attempt_id``,
        and ``study`` keys.
    """
    source = _winner_source_or_default(winner, phase_name)
    return {
        "kind": source.kind,
        "phase": source.phase,
        "trial_number": source.trial_number,
        "generation_id": source.generation_id,
        "attempt_id": source.attempt_id,
        "study": source.study,
    }


def _save_promotion_decision(
    experiment: Experiment,
    phase_name: str,
    decision: dict[str, Any],
    *,
    generation_id: str,
) -> None:
    """Persist a phase promotion decision into its immutable generation namespace.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose promotion decision is being saved.
    :param dict[str, Any] decision: Promotion decision payload to persist.
    :param str generation_id: Immutable generation namespace to write into. The
        legacy compatibility projection is produced separately by
        :func:`phasesweep.engine.generation._copy_yaml_projection` once a generation is published.
    """
    path = path_ops._generation_promotion_decision_path(experiment, generation_id, phase_name)
    _write_yaml_atomic(path, decision)


# Warn-once keys for :func:`_warn_environment_drift`. A resume can load the
# same winner twice (preflight, then the run itself) and a suite can inherit it
# across studies; the operator needs the divergence once, not once per read.
_ENVIRONMENT_DRIFT_WARNED: set[tuple[str, str, str]] = set()


def _warn_environment_drift(
    experiment: Experiment,
    phase_name: str,
    stored_digest: str | None,
) -> None:
    """Warn once when an inherited winner was produced under another environment.

    Persistent-study preflight separately refuses a top-up across semantic
    environment cohorts. ``--from-phase`` deliberately skips this phase, so no
    trial is allocated into that study; this warning tells the operator that a
    later phase is building on a winner from another cohort. Winners without a
    recorded digest predate the record and are left alone.

    :param Experiment experiment: Parsed experiment supplying the current contract.
    :param str phase_name: Phase whose winner was loaded, used in the warn-once key.
    :param str | None stored_digest: Digest recorded on the loaded winner.
    """
    if stored_digest is None:
        return
    # Deferred: ``engine.trial`` pulls in the evidence/W&B stack, which the
    # read-only paths that import this module never need.
    from phasesweep.engine.trial import _environment_identity

    current_digest = _environment_identity(experiment).digest
    if stored_digest == current_digest:
        return
    key = (experiment.experiment, phase_name, stored_digest)
    if key in _ENVIRONMENT_DRIFT_WARNED:
        return
    _ENVIRONMENT_DRIFT_WARNED.add(key)
    log.warning(
        "[%s] inherited winner ran under trainer environment %s..., but this process "
        "composes %s... under execution.inherit_env=%r. The inherited result is being "
        "reused across an environment change; confirm the difference is irrelevant to "
        "the metric, or re-run the phase.",
        phase_name,
        stored_digest[:12],
        current_digest[:12],
        experiment.execution.inherit_env,
    )


def _load_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> Winner:
    """Load a phase winner from disk and verify it matches the *current* config.

    ``--from-phase`` skips earlier phases by reading their persisted winners.
    Without verification, editing a parent phase's YAML between runs leaves
    the child phase silently inheriting the *old* winner against the *new*
    parent config — a correctness bug, not just a performance one.

    We re-compute the fingerprint of the current parent ``phase`` against the
    currently-resolved ``inherited_winners`` and refuse the load if either
    (a) the stored winner has no fingerprint at all (legacy or hand-edited),
    or (b) the fingerprints disagree (review v0.5.6 / blocker 3).

    A recorded semantic environment digest that disagrees with this process is
    a warning on this explicit skipped-phase path; ordinary study top-ups are
    refused before allocation (see :func:`_warn_environment_drift`). Winners
    written before those fields existed load with it set to ``None``.

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose winner is being loaded.
        inherited_winners: Winners loaded for phases earlier in the chain;
            contribute to the recomputed fingerprint.

    Returns:
        The reconstructed :class:`Winner` for ``phase``.

    Raises:
        FileNotFoundError: ``winner.yaml`` does not exist for the phase.
        WinnerIntegrityError: The file is unreadable, incomplete, ambiguously
            scoped, or incompatible with the current partial-result policy.
        StudyFingerprintMismatchError: The stored fingerprint disagrees with
            the freshly computed one.

    """
    published_generation_id = publication_ops._last_successful_generation_id(
        experiment,
        raise_on_manifest_error=True,
    )
    path = publication_ops._published_winner_path_for(
        experiment, published_generation_id, phase.name
    )
    if path is None:
        raise FileNotFoundError(
            f"Winner file missing for phase {phase.name!r}: no generation has completed."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Winner file missing for phase {phase.name!r}: {path}")

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise WinnerIntegrityError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise WinnerIntegrityError(
            f"Winner file {path} is invalid or incomplete for skipped phase "
            f"{phase.name!r}: top level must be a mapping."
        )

    current_fp = fingerprint_ops._phase_fingerprint(experiment, phase, inherited_winners)
    stored_fp = data.get("phase_fingerprint")

    if stored_fp is None:
        raise WinnerIntegrityError(
            f"Winner file {path} has no phase_fingerprint. Refusing to use it "
            f"for --from-phase because phasesweep cannot prove it matches the "
            f"current config for skipped phase {phase.name!r}. Re-run the "
            f"phase, or — if you know the config is unchanged — delete the "
            f"file and re-run to regenerate it with a fingerprint."
        )

    if stored_fp != current_fp:
        raise StudyFingerprintMismatchError(
            f"Winner file {path} was produced by a different phase config "
            f"(stored fingerprint {stored_fp[:16]}... != current "
            f"{current_fp[:16]}...). Re-run phase {phase.name!r}, change the "
            f"experiment name, or restore the matching config before resuming."
        )

    completion = data.get("completion")
    if not isinstance(completion, dict):
        raise WinnerIntegrityError(
            f"Winner file {path} is invalid or incomplete for skipped phase "
            f"{phase.name!r}: missing mapping field 'completion'."
        )
    if completion.get("incomplete") is True and not phase.allow_incomplete_on_timeout:
        raise WinnerIntegrityError(
            f"Winner file {path} records an incomplete phase result. Refusing to "
            f"use it for skipped phase {phase.name!r} unless the current config "
            "sets allow_incomplete_on_timeout: true."
        )
    generation_id = data.get("generation_id")
    attempt_id = data.get("attempt_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise WinnerIntegrityError(
            f"Winner file {path} has no valid generation_id; refusing unscoped evidence."
        )
    if not isinstance(attempt_id, str) or not attempt_id:
        raise WinnerIntegrityError(
            f"Winner file {path} has no valid attempt_id; refusing unscoped evidence."
        )
    source_data = data.get("winner_source")
    if not isinstance(source_data, dict):
        raise WinnerIntegrityError(
            f"Winner file {path} has no valid winner_source; refusing ambiguous provenance."
        )
    source_kind = source_data.get("kind")
    if source_kind not in ("phase_trial", "promotion_baseline", "suite_baseline"):
        raise WinnerIntegrityError(f"Winner file {path} has an invalid winner_source kind.")

    stored_env_digest = data.get("trainer_env_digest")
    if not isinstance(stored_env_digest, str) or not stored_env_digest:
        stored_env_digest = None
    stored_inherit_env = data.get("trainer_inherit_env")
    if not isinstance(stored_inherit_env, str | list):
        stored_inherit_env = None
    _warn_environment_drift(experiment, phase.name, stored_env_digest)

    try:
        source = _parse_winner_source(source_data, cast(WinnerSourceKind, source_kind))
        return Winner(
            trial_number=int(data["trial_number"]),
            params=dict(data["params"]),
            effective_overrides=dict(data["effective_overrides"]),
            metric=float(data["metric"][experiment.metric.name]),
            constraints={k: float(v) for k, v in (data.get("constraints") or {}).items()},
            gates=[item for item in (data.get("gates") or []) if isinstance(item, dict)],
            completion=dict(completion),
            promotion=data.get("promotion") if isinstance(data.get("promotion"), dict) else None,
            phase_fingerprint=str(stored_fp),
            generation_id=generation_id,
            attempt_id=attempt_id,
            source=source,
            objective_provenance=(
                dict(data["objective_provenance"])
                if isinstance(data.get("objective_provenance"), dict)
                else None
            ),
            trainer_env_digest=stored_env_digest,
            trainer_inherit_env=(
                [str(name) for name in stored_inherit_env]
                if isinstance(stored_inherit_env, list)
                else stored_inherit_env
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WinnerIntegrityError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
