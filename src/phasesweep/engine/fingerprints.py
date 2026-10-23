"""Semantic identities and study fingerprint validation."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.state import PHASE_FINGERPRINT_ATTR, PHASE_FINGERPRINT_PAYLOAD_ATTR, Winner
from phasesweep.errors import OperatorAction
from phasesweep.runtime.json import JsonObject, json_object

log = logging.getLogger(__name__)

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
    }
)
# Increment the schema versions when their fingerprint payloads change.
FINGERPRINT_SCHEMA_VERSION = 7
EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 6


def _execution_identity(experiment: Experiment) -> dict[str, Any]:
    """Return execution-context semantics for fingerprinting.

    Include the resolved trainer cwd, inheritance policy,
    and effective pass-through names. Ambient tracked values are checked by the trial
    cohort; pass-through values may change without altering semantics.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return dict[str, Any]: JSON-serialisable execution-identity payload.
    """
    return {
        "cwd": str(experiment.execution.resolved_cwd() or Path.cwd().resolve()),
        "inherit_env": experiment.execution.inherit_env_contract(),
        "passthrough_env": list(experiment.execution.effective_passthrough_env()),
    }


def _semantic_payload_digest(payload: object) -> str:
    """Return the canonical digest used for persisted experiment semantics.

    :param object payload: JSON-compatible semantic identity payload.
    :return str: SHA-256 of the canonical compact JSON representation.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _semantic_core(experiment: Experiment) -> dict[str, Any]:
    """Return the experiment-level fields both semantic fingerprints hash.

    :param Experiment experiment: Experiment whose command, trainer input,
        environment, execution, provenance, metric, and constraints are read.
    :return dict[str, Any]: JSON-serializable payload fragment.
    """
    return {
        "trial_command": experiment.trial_command,
        "trainer_config": experiment.trainer_config,
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "execution": _execution_identity(experiment),
        "provenance": dict(sorted(experiment.provenance.items())),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        **(
            {"mlflow": experiment.mlflow.evidence_identity()}
            if experiment.mlflow is not None
            else {}
        ),
    }


def _experiment_semantic_fingerprint(experiment: Experiment) -> str:
    """Hash the experiment semantics recorded with each published result.

    Include metric/extractor, constraints, command, environment, provenance, and
    ordered phase semantics. Exclude run-control targets, throughput, and comments.

    :param Experiment experiment: Parsed experiment config to fingerprint.
    :return str: SHA-256 hex digest (64 characters) of the canonicalised
        semantic payload.
    """
    payload = {
        "fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        **_semantic_core(experiment),
        "phases": [
            {"name": phase.name, **_semantic_phase_dump(phase)} for phase in experiment.phases
        ],
    }
    return _semantic_payload_digest(payload)


def _semantic_phase_dump(phase: Phase) -> dict[str, Any]:
    """Return phase execution and output semantics, excluding run control.

    GPU identities are excluded. Under ``whole_node``, their count defines trainer
    world size and is included; equivalent index and UUID spellings retain identity.

    :param Phase phase: Phase whose semantic payload is being built.
    :return dict[str, Any]: JSON-serializable semantic phase payload.
    """
    dump = {k: v for k, v in phase.model_dump(mode="json").items() if k not in _RUN_CONTROL_KEYS}
    if phase.gpu_policy == "whole_node":
        tokens = phase.gpu_ids if phase.gpu_ids is not None else phase.gpu_devices
        # Only macOS permits a discovered whole-node pool: one MPS device on
        # Apple silicon and no device on Intel. CUDA configurations declare the count.
        dump["whole_node_device_count"] = (
            len(tokens)
            if tokens is not None
            else int(platform.machine().lower() in {"arm64", "aarch64"})
        )
    return dump


def _phase_fingerprint_payload(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> JsonObject:
    """Return canonical JSON fields that change the meaning of a trial.

    Run-control fields such as ``n_trials`` are excluded, allowing study top-ups.

    :param Experiment experiment: Experiment supplying command, input format,
        environment, metric, constraints, and provenance.
    :param Phase phase: Phase whose run-control keys are excluded.
    :param dict[str, Winner] inherited_winners: Parent winners whose effective
        overrides contribute to identity.
    :return JsonObject: Configured trial semantics normalized to the JSON representation being
        hashed.
    """
    semantic_phase = _semantic_phase_dump(phase)
    payload = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        **_semantic_core(experiment),
        "phase": semantic_phase,
        "inherited_effective_overrides": {
            parent: inherited_winners[parent].effective_overrides for parent in phase.inherits
        },
    }
    return json_object(json.loads(json.dumps(payload, default=str)))


def _phase_fingerprint(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Hash the semantic execution context for resume-compatibility checks.

    :param Experiment experiment: Experiment forwarded to
        :func:`_phase_fingerprint_payload`.
    :param Phase phase: Phase being fingerprinted.
    :param dict[str, Winner] inherited_winners: Parent winners contributing
        effective overrides to identity.
    :return str: SHA-256 of the canonical semantic payload.
    """
    return _semantic_payload_digest(
        _phase_fingerprint_payload(experiment, phase, inherited_winners)
    )


_ABSENT = object()
_MAX_REPORTED_CHANGES = 10
_MAX_REPORTED_VALUE_CHARS = 80
_REPORTED_VALUE_LEAD_CHARS = 24
_CWD_FOLLOWS_INVOCATION = (
    "The trainer working directory follows the directory phasesweep runs from when "
    "execution.cwd is unset; run from the original directory or set execution.cwd in the "
    "config (relative paths resolve against the config file)."
)
# An explicit storage keeps the same study under a new workdir; only auto storage follows it.
_NEW_STUDY_STEP = (
    "To run this config, set a new `experiment` name in the config (with `storage: auto` a new "
    "`workdir` also works); the old study stays readable under the old name."
)


def _changed_payload_keys(
    old: object, new: object, path: str = ""
) -> list[tuple[str, object, object]]:
    """List the dotted key paths whose values differ between two canonical payloads.

    Mappings are compared key by key; any other value, a list included, is
    compared whole. A key present on one side only is reported against ``_ABSENT``.

    :param object old: Stored payload, or the value at ``path`` within it.
    :param object new: Current payload, or the value at ``path`` within it.
    :param str path: Dotted path of the values being compared; empty at the root.
    :return list[tuple[str, object, object]]: Each differing path with its old and new
        value, in key order.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[tuple[str, object, object]] = []
        for key in sorted(old.keys() | new.keys()):
            changes += _changed_payload_keys(
                old.get(key, _ABSENT), new.get(key, _ABSENT), f"{path}.{key}" if path else key
            )
        return changes
    return [] if old == new else [(path, old, new)]


def _render_changed_values(old: object, new: object) -> tuple[str, str]:
    """Render both sides of a changed key as compact JSON, cut to a readable length.

    A long pair is cut to a window that starts just before the first character
    where the two differ, so two paths that share a long prefix still show the
    part that tells them apart.

    :param object old: Stored payload value, or ``_ABSENT``.
    :param object new: Current payload value, or ``_ABSENT``.
    :return tuple[str, str]: The old and new text; ``<absent>`` stands for a missing key.
    """
    old_text, new_text = (
        "<absent>"
        if value is _ABSENT
        else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for value in (old, new)
    )
    if max(len(old_text), len(new_text)) <= _MAX_REPORTED_VALUE_CHARS:
        return old_text, new_text
    shared = next(
        (
            index
            for index, pair in enumerate(zip(old_text, new_text, strict=False))
            if pair[0] != pair[1]
        ),
        min(len(old_text), len(new_text)),
    )
    start = max(0, shared - _REPORTED_VALUE_LEAD_CHARS)

    def window(text: str) -> str:
        """Cut ``text`` to the window that starts at ``start``, marking each cut end.

        :param str text: Full rendered value.
        :return str: The window, with ``...`` where text was dropped.
        """
        stop = start + _MAX_REPORTED_VALUE_CHARS
        return f"{'...' if start else ''}{text[start:stop]}{'...' if stop < len(text) else ''}"

    return window(old_text), window(new_text)


def _describe_changed_keys(changes: list[tuple[str, object, object]]) -> str:
    """Name the keys that changed, with both values, for a fingerprint refusal.

    :param list[tuple[str, object, object]] changes: Output of :func:`_changed_payload_keys`.
    :return str: A sentence naming up to :data:`_MAX_REPORTED_CHANGES` keys and a count of
        the rest, followed by a space; empty when nothing changed.
    """
    if not changes:
        return ""
    shown = []
    for path, old, new in changes[:_MAX_REPORTED_CHANGES]:
        old_text, new_text = _render_changed_values(old, new)
        shown.append(f"{path}: {old_text} -> {new_text}")
    if len(changes) > _MAX_REPORTED_CHANGES:
        shown.append(f"and {len(changes) - _MAX_REPORTED_CHANGES} more")
    return f"Changed: {'; '.join(shown)}. "


def _stamp_fingerprint(study: optuna.Study, fingerprint: str, payload: JsonObject) -> None:
    """Record a study's fingerprint beside the payload it digests.

    The payload goes first: a process that dies between the two writes leaves
    a study with no fingerprint, which the next run stamps again, never one
    whose stored digest has no payload to explain it.

    :param optuna.Study study: Study being bound to the phase config.
    :param str fingerprint: Digest of ``payload``.
    :param JsonObject payload: Canonical payload the fingerprint covers.
    """
    study.set_user_attr(PHASE_FINGERPRINT_PAYLOAD_ATTR, payload)
    study.set_user_attr(PHASE_FINGERPRINT_ATTR, fingerprint)


def _verify_fingerprint(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Stamp a fresh study, or rebind an empty one, with its fingerprint; refuse a populated mismatch.

    :param optuna.Study study: Study being verified or stamped.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase whose fingerprint must match.
    :param dict[str, Winner] inherited_winners: Parent winners contributing to identity.
    :raises StudyFingerprintMismatchError: The populated study has no persisted
        fingerprint or an incompatible one; the message names the keys that changed.
    :return str: Verified current fingerprint.
    """
    payload = _phase_fingerprint_payload(experiment, phase, inherited_winners)
    fp = _semantic_payload_digest(payload)
    existing = study.user_attrs.get(PHASE_FINGERPRINT_ATTR)
    trials = study.get_trials(deepcopy=False)
    if existing is None:
        if trials:
            raise StudyFingerprintMismatchError(
                f"Phase {phase.name!r} study {study.study_name!r} has no semantic "
                "fingerprint. Its populated state is unsupported by this release.",
            )
        _stamp_fingerprint(study, fp, payload)
    elif existing != fp:
        # A zero-trial study has no result semantics to preserve. This includes
        # a process that died after recording its trial target but before Optuna
        # created the first trial.
        if not trials:
            log.warning(
                "Rebinding the fingerprint of empty study %s (%s -> %s): no trial "
                "ever ran under the previous config.",
                study.study_name,
                existing,
                fp,
            )
            _stamp_fingerprint(study, fp, payload)
            return fp
        stored = study.user_attrs.get(PHASE_FINGERPRINT_PAYLOAD_ATTR)
        changes = _changed_payload_keys(stored, payload) if isinstance(stored, dict) else []
        cwd_unset = experiment.execution.cwd is None and any(
            path == "execution.cwd" for path, _old, _new in changes
        )
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} was created with a different phase config "
            f"(fingerprint {existing} != {fp}). {_describe_changed_keys(changes)}"
            f"{_CWD_FOLLOWS_INVOCATION if cwd_unset else _NEW_STUDY_STEP}",
            # Running from the original directory keeps the study; a new name does not.
            action=OperatorAction.FIX_CONFIG if cwd_unset else None,
        )
    return fp
