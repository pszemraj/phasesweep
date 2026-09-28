"""Artifact serialization, logging, and winner persistence."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import optuna
import yaml

import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.paths as path_ops
from phasesweep.config import Experiment, Phase
from phasesweep.engine.confound import _validate_confound_block
from phasesweep.engine.errors import (
    OperatorAction,
    StudyFingerprintMismatchError,
    WinnerIntegrityError,
)
from phasesweep.engine.publication_validation import ValidatedPublication
from phasesweep.engine.state import (
    Winner,
    _parse_winner_source,
)
from phasesweep.runtime.files import atomic_create_text, atomic_text_writer

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
    per-generation lifecycle records (review v0.5.15 / blocker 3): a second
    call for an already-written path can never clobber the first write, even
    a same-content rewrite. Delegates to :func:`atomic_create_text`, so
    ``path`` is staged and fsynced beside itself and only hard-linked into
    place once complete -- no reader can ever observe a partial write.

    :param Path path: Destination path to create; the parent directory is
        created if missing.
    :param Any payload: YAML-serializable value to write.
    :return bool: ``True`` when this call created and wrote the file;
        ``False`` when the destination already existed and nothing was
        written.
    """
    text = yaml.safe_dump(payload, sort_keys=False)
    return atomic_create_text(path, text)


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
    produced the winning trial (review v0.5.18 / finding F3). Neither ever
    carries ambient variable values. ``confound`` records the advisory
    verdicts on the population the source selection ranked; like the
    provenance fields it is winner-only and never enters ``summary.yaml``.

    Args:
        experiment: Parsed experiment config; supplies the metric name used
            in the persisted payload.
        phase_name: Name of the phase whose winner is being saved.
        winner: The winning trial.
        generation_id: Immutable generation namespace to write into.

    """
    path = path_ops._generation_winner_path(experiment, generation_id, phase_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase_name,
        "metric": {experiment.metric.name: winner.metric, "goal": experiment.metric.goal},
        **_winner_common_payload(winner),
        "phase_fingerprint": winner.phase_fingerprint,
        "objective_provenance": winner.objective_provenance,
        "trainer_env_digest": winner.trainer_env_digest,
        "trainer_inherit_env": winner.trainer_inherit_env,
        "confound": winner.confound,
    }
    _write_yaml_atomic(path, payload)


def _winner_common_payload(winner: Winner) -> dict[str, Any]:
    """Serialize winner fields shared by persisted and summary representations.

    :param Winner winner: Winner whose common fields should be serialized.
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
        "winner_source": _winner_source_payload(winner),
        "trainer_input": winner.trainer_input,
    }
    return payload


def _winner_source_payload(winner: Winner) -> dict[str, Any]:
    """Serialize the concrete source trial for an exposed winner.

    :param Winner winner: Winner whose recorded ``source`` is serialized.
    :return dict[str, Any]: JSON-serializable winner-source payload with
        ``kind``, ``phase``, ``trial_number``, ``generation_id``, ``attempt_id``,
        keys.
    """
    source = winner.source
    assert source is not None, "only the never-persisted dry-run placeholder has no source"
    return {
        "kind": source.kind,
        "phase": source.phase,
        "trial_number": source.trial_number,
        "generation_id": source.generation_id,
        "attempt_id": source.attempt_id,
    }


# Warn-once keys for :func:`_warn_environment_drift`. A resume can load the
# same winner twice (preflight, then the run itself); the operator needs the
# divergence once, not once per read.
_ENVIRONMENT_DRIFT_WARNED: set[tuple[str, str, str]] = set()


def _warn_environment_drift(
    experiment: Experiment,
    phase_name: str,
    stored_digest: str,
) -> None:
    """Warn once when a reused winner was produced under another environment.

    Persistent-study preflight refuses to allocate a trial across semantic
    environment cohorts, but two paths reuse a recorded result without
    allocating: ``--from-phase`` loads a skipped phase's published winner, and
    a replay re-selects from a study with no remaining trial slots. This
    warning tells the operator that the result being built on came from
    another cohort.

    :param Experiment experiment: Parsed experiment supplying the current contract.
    :param str phase_name: Phase whose winner was reused, used in the warn-once key.
    :param str stored_digest: Digest recorded on the reused winner or its trial.
    """
    # Deferred: ``engine.trial`` pulls in the evidence/W&B stack, which the
    # read-only paths that import this module never need.
    from phasesweep.engine.trial import _environment_identity

    current_digest = _environment_identity(
        experiment,
        phase_name,
        require_wandb_online=False,
    ).digest
    if stored_digest == current_digest:
        return
    key = (experiment.experiment, phase_name, stored_digest)
    if key in _ENVIRONMENT_DRIFT_WARNED:
        return
    _ENVIRONMENT_DRIFT_WARNED.add(key)
    log.warning(
        "[%s] reused winner ran under trainer environment %s..., but this process "
        "composes %s... under execution.inherit_env=%r. The recorded result is being "
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
    *,
    publication: ValidatedPublication | None,
) -> Winner:
    """Load a published phase winner and verify it matches the *current* config.

    ``--from-phase`` skips earlier phases by reading their persisted winners.
    Without verification, editing a parent phase's YAML between runs leaves
    the child phase silently inheriting the *old* winner against the *new*
    parent config — a correctness bug, not just a performance one.

    We re-compute the fingerprint of the current parent ``phase`` against the
    currently-resolved ``inherited_winners`` and refuse the load if the
    fingerprints disagree (review v0.5.6 / blocker 3).

    A recorded semantic environment digest that disagrees with this process is
    a warning on this explicit skipped-phase path; ordinary study top-ups are
    refused before allocation (see :func:`_warn_environment_drift`).

    The winner is decoded from the bytes publication validation read, never
    reread from disk, so what a resume builds on is exactly what was
    validated. That validation already proved the winner's phase, generation
    and attempt ids, fingerprint format, source, and completion shape; this
    function checks only what depends on the current config.

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose winner is being loaded.
        inherited_winners: Winners loaded for phases earlier in the chain;
            contribute to the recomputed fingerprint.
        publication: The last-success publication the caller resolved with
            ``raise_on_manifest_error=True``; ``None`` when nothing is
            published. A caller loading several phases resolves it once.

    Returns:
        The reconstructed :class:`Winner` for ``phase``.

    Raises:
        FileNotFoundError: The publication holds no winner for the phase.
        WinnerIntegrityError: The winner is incomplete or incompatible with
            the current partial-result policy.
        StudyFingerprintMismatchError: The stored fingerprint disagrees with
            the freshly computed one.

    """
    if publication is None:
        raise FileNotFoundError(
            f"Winner file missing for phase {phase.name!r}: no generation has completed."
        )
    path = path_ops._generation_winner_path(experiment, publication.generation_id, phase.name)
    validated = publication.winners.get(phase.name)
    if validated is None:
        raise FileNotFoundError(f"Winner file missing for phase {phase.name!r}: {path}")
    data = validated.payload

    current_fp = fingerprint_ops._phase_fingerprint(experiment, phase, inherited_winners)
    stored_fp = data["phase_fingerprint"]
    if stored_fp != current_fp:
        raise StudyFingerprintMismatchError(
            f"Winner file {path} was produced by a different phase config "
            f"(stored fingerprint {stored_fp[:16]}... != current "
            f"{current_fp[:16]}...). Re-run phase {phase.name!r}, change the "
            f"experiment name, or restore the matching config before resuming.",
            action=OperatorAction.FIX_CONFIG,
        )

    completion = data["completion"]
    if completion.get("incomplete") is True and not phase.allow_incomplete_on_timeout:
        raise WinnerIntegrityError(
            f"Winner file {path} records an incomplete phase result. Refusing to "
            f"use it for skipped phase {phase.name!r} unless the current config "
            "sets allow_incomplete_on_timeout: true.",
            action=OperatorAction.FIX_CONFIG,
        )

    try:
        source = _parse_winner_source(data["winner_source"], expected_phase=phase.name)
        stored_env_digest = data["trainer_env_digest"]
        stored_inherit_env = data["trainer_inherit_env"]
        winner = Winner(
            trial_number=int(data["trial_number"]),
            params=dict(data["params"]),
            effective_overrides=dict(data["effective_overrides"]),
            metric=float(data["metric"][experiment.metric.name]),
            constraints={k: float(v) for k, v in (data.get("constraints") or {}).items()},
            gates=[item for item in (data.get("gates") or []) if isinstance(item, dict)],
            completion=dict(completion),
            phase_fingerprint=stored_fp,
            generation_id=data["generation_id"],
            attempt_id=data["attempt_id"],
            source=source,
            objective_provenance=dict(data["objective_provenance"]),
            trainer_input=dict(data["trainer_input"]),
            trainer_env_digest=stored_env_digest,
            trainer_inherit_env=(
                [str(name) for name in stored_inherit_env]
                if isinstance(stored_inherit_env, list)
                else stored_inherit_env
            ),
            confound=_validate_confound_block(data["confound"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WinnerIntegrityError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
    _warn_environment_drift(experiment, phase.name, stored_env_digest)
    return winner
