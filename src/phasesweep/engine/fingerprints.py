"""Semantic identities and study fingerprint validation."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import optuna

from phasesweep.config.common import TRACKER_READ_CONTROLS
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.bounds import selection_bounds
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.growth import _GrowthVerdict, _search_space_growth, describe_gains
from phasesweep.engine.optuna import _draw_positions, _phase_namespace
from phasesweep.engine.state import (
    PHASE_IDENTITY_ATTR,
    PHASE_IDENTITY_SCHEMA_VERSION,
    PhaseIdentityRecord,
    SearchSpaceEntry,
    Winner,
    _recorded_identity,
)
from phasesweep.errors import (
    NEW_STUDY_STEP,
    OperatorAction,
    rename_step,
    restore_or_new_study,
    short_digest,
)
from phasesweep.runtime.json import JsonObject, json_object
from phasesweep.runtime.terminal import plural

log = logging.getLogger(__name__)

_RUN_CONTROL_KEYS = frozenset(
    {
        # Run-control settings bound or pace a run without changing what a finished trial
        # measured, so they stay outside fingerprints and changing one continues the study.
        "n_trials",
        "n_jobs",
        "gpu_ids",
        "gpu_devices",
        "allow_no_gpu_isolation",
        "max_consecutive_failures",
        "comment",
        "cost_key",
        "allow_unbounded_trials",
        "timeout_seconds_per_phase",
        "timeout_seconds_per_trial",
        "allow_incomplete_on_timeout",
        "allow_partial_grid",
    }
)
#: Where W&B read controls sit inside a dumped phase: a summary-presence gate carries them.
_PHASE_READ_CONTROLS = {"gates": {"__all__": set(TRACKER_READ_CONTROLS)}}
# Increment the schema versions when their fingerprint payloads change.
FINGERPRINT_SCHEMA_VERSION = 13
#: Where a phase identity payload keeps its reading settings (:func:`reading_settings`).
READING_KEY = "reading"
EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 10
#: The viability bounds of the metric and each constraint. They judge what a trial measured,
#: so a phase study's identity excludes them; a published result was selected under them, so
#: the experiment fingerprint it records includes them.
_BOUND_KEYS = frozenset({"min", "max"})


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


def _trainer_core(experiment: Experiment) -> JsonObject:
    """Return the experiment-level fields that say what a trainer ran and where it logged.

    :param Experiment experiment: Experiment whose command, trainer input, environment,
        execution, provenance and MLflow source are read.
    :return JsonObject: JSON-serializable payload fragment.
    """
    core = {
        "trial_command": experiment.trial_command,
        "trainer_config": experiment.trainer_config,
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "execution": _execution_identity(experiment),
        "provenance": dict(sorted(experiment.provenance.items())),
        **(
            {"mlflow": experiment.mlflow.evidence_identity()}
            if experiment.mlflow is not None
            else {}
        ),
    }
    return json_object(json.loads(json.dumps(core, default=str)))


def _experiment_reading(experiment: Experiment) -> JsonObject:
    """Return the experiment-level settings that say how a finished trainer's output is read.

    The metric's name only labels output, since every extractor reads by its own key, and a
    tracker's read controls only bound the read, so neither is part of any identity. The
    bounds (:data:`_BOUND_KEYS`) judge what was read (:mod:`phasesweep.engine.bounds`):
    :func:`_experiment_semantic_fingerprint` adds them back for a published result.

    :param Experiment experiment: Experiment whose metric and constraints are read.
    :return JsonObject: The metric's goal and extractor and each constraint's name and
        extractor.
    """
    extractor_exclude = {"extractor": set(TRACKER_READ_CONTROLS)}
    bound_exclude = dict.fromkeys(_BOUND_KEYS, True)
    reading = {
        "metric": experiment.metric.model_dump(
            mode="json", exclude={"name": True, **bound_exclude, **extractor_exclude}
        ),
        "constraints": [
            c.model_dump(mode="json", exclude={**bound_exclude, **extractor_exclude})
            for c in experiment.constraints
        ],
    }
    return json_object(json.loads(json.dumps(reading, default=str)))


def _semantic_core(experiment: Experiment) -> JsonObject:
    """Return the experiment-level fields a published result's fingerprint hashes.

    :param Experiment experiment: Experiment whose trainer and reading settings are read.
    :return JsonObject: :func:`_trainer_core` and :func:`_experiment_reading` together.
    """
    return {**_trainer_core(experiment), **_experiment_reading(experiment)}


def reading_settings(experiment: Experiment, phase: Phase) -> JsonObject:
    """Return every setting that says how a phase's finished trainer output becomes results.

    The metric's extractor and goal, the constraint list with each extractor, and the phase's
    gates: changing any of them changes what a finished trial's evidence reads as, never what
    the trainer ran, so they address the study (:func:`phase_study_name`) instead of being
    compared, and a study under other reading settings is re-read rather than retrained
    (:mod:`phasesweep.engine.reread`).

    :param Experiment experiment: Experiment supplying the metric and constraints.
    :param Phase phase: Phase supplying the gates.
    :return JsonObject: The settings, normalized to the JSON representation being hashed.
    """
    gates = phase.model_dump(mode="json", include={"gates"}, exclude=_PHASE_READ_CONTROLS)
    return {**_experiment_reading(experiment), **json_object(gates)}


def _experiment_semantic_fingerprint(
    experiment: Experiment, *, phase_count: int | None = None
) -> str:
    """Hash the experiment semantics recorded with each published result.

    Include metric/extractor, constraints and their bounds, command, environment, provenance,
    and ordered phase semantics. Exclude run-control targets, throughput, and comments. The
    bounds selected the published winner, so a result published under other bounds no longer
    matches the config.

    :param Experiment experiment: Parsed experiment config to fingerprint.
    :param int | None phase_count: Producing phase-prefix length, or None for the full plan.
    :return str: SHA-256 hex digest of the canonicalised semantic payload.
    """
    payload = {
        "fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        **_semantic_core(experiment),
        "bounds": selection_bounds(experiment.metric, experiment.constraints),
        "phases": _semantic_phase_plan(experiment.phases[:phase_count]),
    }
    return _semantic_payload_digest(payload)


def _recorded_config_matches(
    experiment: Experiment, stored_fingerprint: object, *, phase_count: int
) -> bool | None:
    """Compare recorded semantics with the current plan's producing phase prefix.

    :param Experiment experiment: Current configuration to compare.
    :param object stored_fingerprint: Represented generation's semantic fingerprint.
    :param int phase_count: Number of phases in the represented generation's plan.
    :return bool | None: Semantic agreement, or unknown without a recorded fingerprint.
    """
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        return None
    return stored_fingerprint == _experiment_semantic_fingerprint(
        experiment, phase_count=phase_count
    )


def _semantic_phase_plan(phases: Sequence[Phase]) -> list[JsonObject]:
    """Describe the configured phases and their inheritance graph.

    :param Sequence[Phase] phases: The whole plan or a prefix in execution order.
    :return list[JsonObject]: Fingerprint payloads in plan order, including inheritance.
    """
    return [
        {"name": phase.name, "inherits": list(phase.inherits), **_semantic_phase_dump(phase)}
        for phase in phases
    ]


def _semantic_phase_dump(phase: Phase) -> dict[str, Any]:
    """Return phase execution and output semantics, excluding run control and parent names.

    GPU identities are excluded. Under ``whole_node``, their count defines trainer
    world size and is included; equivalent index and UUID spellings retain identity.
    ``inherits`` is excluded: a study holds the values its parents supply
    (:func:`_inherited_override_payload`), not which parent supplied them, and the plan-level
    fingerprints that record the graph add it themselves.

    :param Phase phase: Phase whose semantic payload is being built.
    :return dict[str, Any]: JSON-serializable semantic phase payload.
    """
    dump = {
        k: v
        for k, v in phase.model_dump(mode="json", exclude=_PHASE_READ_CONTROLS).items()
        if k not in _RUN_CONTROL_KEYS and k != "inherits"
    }
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


def _identity_phase_dump(phase: Phase) -> JsonObject:
    """Return what a phase's study identity compares of the phase: its dump without its gates.

    The gates are reading settings (:func:`reading_settings`), which address the study.

    :param Phase phase: Phase whose payload is being built.
    :return JsonObject: :func:`_semantic_phase_dump` without ``gates``.
    """
    dump = {key: value for key, value in _semantic_phase_dump(phase).items() if key != "gates"}
    return json_object(json.loads(json.dumps(dump, default=str)))


class CarriesOverrides(Protocol):
    """A winner as a child sees it: the composed overrides it passes on."""

    @property
    def effective_overrides(self) -> Mapping[str, object]:
        """Return the winner's composed overrides.

        :return Mapping[str, object]: Override value by trainer key.
        """
        ...


def _inherited_override_suppliers(
    phase: Phase, inherited_winners: Mapping[str, CarriesOverrides]
) -> dict[str, str]:
    """Name the last parent supplying each key.

    :param Phase phase: The phase whose inheritance order determines precedence.
    :param Mapping[str, CarriesOverrides] inherited_winners: Parent winners supplying overrides.
    :return dict[str, str]: Parent by override key.
    """
    return {
        key: parent
        for parent in phase.inherits
        for key in inherited_winners[parent].effective_overrides
    }


def _inherited_override_payload(
    phase: Phase, inherited_winners: Mapping[str, CarriesOverrides]
) -> dict[str, object]:
    """Return the parent layer of a trial's composed overrides.

    :param Phase phase: The phase whose local keys and inheritance precedence apply.
    :param Mapping[str, CarriesOverrides] inherited_winners: Parent winners supplying overrides.
    :return dict[str, object]: Parent values, including keys the phase replaces.
    """
    return {
        key: inherited_winners[parent].effective_overrides[key]
        for key, parent in _inherited_override_suppliers(phase, inherited_winners).items()
    }


def _inherited_values(phase: Phase, inherited: Mapping[str, object]) -> JsonObject:
    """Keep the inherited values the trainer receives after the local overrides.

    :param Phase phase: The phase that can replace a parent's value.
    :param Mapping[str, object] inherited: The parent layer before local overrides.
    :return JsonObject: Effective inherited values by key, in JSON representation.
    """
    local_keys = _local_keys(phase)
    values = {key: value for key, value in inherited.items() if key not in local_keys}
    return json_object(json.loads(json.dumps(values, default=str)))


def _local_keys(phase: Phase) -> set[str]:
    """Return the trainer keys a phase sets itself, which no parent's value reaches.

    :param Phase phase: The phase.
    :return set[str]: Its fixed, searched and repeat keys.
    """
    local_keys = set(phase.fixed_overrides) | phase.search_space.keys()
    if phase.repeats is not None:
        local_keys.add(phase.repeats.key)
    return local_keys


#: Hex digits of the address digest in a study name; 64 bits keeps a collision between two
#: addresses of one phase out of reach.
_ADDRESS_DIGEST_CHARS = 16


def phase_study_name(experiment: Experiment, phase: Phase, inherited: Mapping[str, object]) -> str:
    """Return the study that holds a phase's trials as read under the current reading settings.

    The one rule for finding a phase's study: its namespace, ``::``, and a digest of the values
    its parents' winners pass it and its reading settings
    (:func:`reading_settings`). Returning to earlier inputs and settings addresses the study
    recorded under them again. Optuna keeps a finished trial's value fixed, so each reading
    change needs a study of its own (:mod:`phasesweep.engine.reread`).

    :param Experiment experiment: Experiment supplying the namespace and the reading settings.
    :param Phase phase: The phase.
    :param Mapping[str, object] inherited: The parent layer, from the parents' winners
        (:func:`_inherited_override_payload`) or a study's record (:func:`_recorded_inheritance`).
    :return str: The study name.
    """
    namespace = _phase_namespace(experiment.experiment, phase.name)
    digest = _semantic_payload_digest(
        {
            "inherited": _inherited_values(phase, inherited),
            "reading": reading_settings(experiment, phase),
        }
    )
    return f"{namespace}::{digest[:_ADDRESS_DIGEST_CHARS]}"


def _recorded_goal(study: optuna.Study) -> Literal["minimize", "maximize"] | None:
    """Read the metric goal a study's identity records among its reading settings.

    The goal addresses the study, so a study the current config does not address may record
    another one; its own direction is checked against that.

    :param optuna.Study study: A phase's study.
    :return Literal["minimize", "maximize"] | None: The recorded goal, or ``None`` when the
        study records no identity.
    """
    record = _recorded_identity(study)
    reading = None if record is None else record["payload"].get(READING_KEY)
    metric = reading.get("metric") if isinstance(reading, dict) else None
    goal = metric.get("goal") if isinstance(metric, dict) else None
    if goal == "minimize":
        return "minimize"
    if goal == "maximize":
        return "maximize"
    return None


def addressed_study_names(
    experiment: Experiment, winners: Mapping[str, CarriesOverrides]
) -> dict[str, str]:
    """Return the study each phase addresses under the given parent winners.

    :param Experiment experiment: The experiment whose phases are addressed.
    :param Mapping[str, CarriesOverrides] winners: Winners by phase name, from one capture.
    :return dict[str, str]: Study name by phase name, for each phase whose parents all have a
        winner; a phase missing from it has no study to address yet.
    """
    return {
        phase.name: phase_study_name(experiment, phase, _inherited_override_payload(phase, winners))
        for phase in experiment.phases
        if all(parent in winners for parent in phase.inherits)
    }


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
    return _identity_payload(
        experiment,
        phase,
        _inherited_override_payload(phase, inherited_winners),
    )


def _identity_payload(
    experiment: Experiment, phase: Phase, inherited: Mapping[str, object]
) -> JsonObject:
    """Return the fingerprint payload of a phase with the given inherited overrides.

    The inherited values and the reading settings address the study (:func:`phase_study_name`),
    so a study's record always agrees with the config on both; the record keeps them so the
    study a reading change re-reads from can say what changed.

    :param Experiment experiment: Experiment supplying the experiment-level semantics.
    :param Phase phase: Phase whose semantic dump the payload holds.
    :param Mapping[str, object] inherited: The parent layer of the phase's overrides.
    :return JsonObject: The payload, normalized to the JSON representation being hashed.
    """
    payload = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        **_trainer_core(experiment),
        "phase": _identity_phase_dump(phase),
        "inherited_effective_overrides": _inherited_values(phase, inherited),
        READING_KEY: reading_settings(experiment, phase),
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
#: The refusal shows each recorded value, so restoring it is the first way out.
_RECORDED_VALUES = "the recorded value(s)"
_CWD_FOLLOWS_INVOCATION = (
    "execution.cwd is unset, so the trainer runs where phasesweep is started: run from the "
    "original directory, or set execution.cwd to it."
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
    return [] if _same_value(old, new) else [(path, old, new)]


def _render_value(value: object) -> str:
    """Render one side of a changed key as compact JSON.

    :param object value: Payload value, or ``_ABSENT``.
    :return str: The JSON text; ``<absent>`` stands for a missing key.
    """
    if value is _ABSENT:
        return "<absent>"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _same_value(old: object, new: object) -> bool:
    """Say whether two payload values have the same JSON text, which is what the digest hashes.

    Python equality would equate ``8`` with ``8.0`` and ``1`` with ``True``, values whose
    fingerprints differ.

    :param object old: Stored payload value, or ``_ABSENT``.
    :param object new: Current payload value, or ``_ABSENT``.
    :return bool: Whether both render to the same text.
    """
    return _render_value(old) == _render_value(new)


def _render_changed_values(old: object, new: object) -> tuple[str, str]:
    """Render both sides of a changed key as compact JSON, cut to a readable length.

    A long pair is cut to a window that starts just before the first character
    where the two differ, so two paths that share a long prefix still show the
    part that tells them apart.

    :param object old: Stored payload value, or ``_ABSENT``.
    :param object new: Current payload value, or ``_ABSENT``.
    :return tuple[str, str]: The old and new text; ``<absent>`` stands for a missing key.
    """
    old_text, new_text = _render_value(old), _render_value(new)
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
        the rest; empty when nothing changed.
    """
    if not changes:
        return ""
    shown = []
    for path, old, new in changes[:_MAX_REPORTED_CHANGES]:
        old_text, new_text = _render_changed_values(old, new)
        shown.append(f"{path}: {old_text} -> {new_text}")
    if len(changes) > _MAX_REPORTED_CHANGES:
        shown.append(f"and {len(changes) - _MAX_REPORTED_CHANGES} more")
    return f"Changed: {'; '.join(shown)}."


def restore_or_rename_step(phase: Phase, *, recorded: str) -> str:
    """Offer the rename that reruns a phase whose own keys changed, beside restoring them.

    A phase's studies are named after the phase, so a new name starts a new study and leaves
    the other phases' studies valid. A phase inheriting it needs no new name: its study is
    found by the values the renamed phase's winner passes it. A persistent study's refusals
    share this step, whether the recorded identity is the study's or its winner's.

    :param Phase phase: Phase whose own configuration differs from its study's.
    :param str recorded: What to restore, as the refusal can name it.
    :return str: The operator's step.
    """
    return f"Restore {recorded}, or {rename_step(phase.name)}."


def new_study_note(
    phase: Phase,
    inherited: Mapping[str, object],
    others: Sequence[optuna.Study],
) -> str | None:
    """Say why a phase starts a new study beside the ones it already has, and that they stay.

    :param Phase phase: The phase whose addressed study holds no trials yet.
    :param Mapping[str, object] inherited: The parent layer it now inherits.
    :param Sequence[optuna.Study] others: The phase's other studies, oldest first.
    :raises StudySchemaMismatchError: The most recent populated study's identity is malformed.
    :return str | None: The sentence, or None when no other study holds a trial.
    """
    populated = [(study, study.get_trials(deepcopy=False)) for study in others]
    populated = [(study, trials) for study, trials in populated if trials]
    if not populated:
        return None
    latest, trials = populated[-1]
    record = _recorded_identity(latest)
    recorded = None if record is None else record["payload"].get("inherited_effective_overrides")
    current = _inherited_values(phase, inherited)
    changes = _changed_payload_keys(recorded if isinstance(recorded, dict) else {}, current)
    shown = "; ".join(
        f"{path}: {' -> '.join(_render_changed_values(old, new))}"
        for path, old, new in changes[:_MAX_REPORTED_CHANGES]
    )
    more = len(changes) - _MAX_REPORTED_CHANGES
    if more > 0:
        shown += f"; and {more} more"
    change = f"inherits {shown or 'different values'}"
    return f"{change}, so it starts a new study; the earlier study keeps its {plural(len(trials), 'trial')}."


def _identity_record(
    fingerprint: str,
    payload: JsonObject,
    search_spaces: list[SearchSpaceEntry],
) -> PhaseIdentityRecord:
    """Build the identity record a study is bound to.

    :param str fingerprint: Digest of ``payload``.
    :param JsonObject payload: Canonical payload the fingerprint covers.
    :param list[SearchSpaceEntry] search_spaces: The study's search-space history, ending with
        ``payload``'s own search space.
    :return PhaseIdentityRecord: The record.
    """
    return {
        "schema_version": PHASE_IDENTITY_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "payload": payload,
        "search_spaces": search_spaces,
    }


def _recorded_inheritance(record: PhaseIdentityRecord) -> JsonObject:
    """Read the parent values that reached the trainer.

    :param PhaseIdentityRecord record: The study's identity.
    :return JsonObject: The effective inherited override values by key.
    """
    return json_object(record["payload"]["inherited_effective_overrides"])


def _payload_search_space(payload: JsonObject) -> JsonObject:
    """Return the phase search space a fingerprint payload holds.

    :param JsonObject payload: A canonical phase fingerprint payload.
    :raises ValueError: The payload holds no phase search space object.
    :return JsonObject: Its ``phase.search_space``.
    """
    phase = payload.get("phase")
    space = phase.get("search_space") if isinstance(phase, dict) else None
    if not isinstance(space, dict):
        raise ValueError("The fingerprint payload holds no phase.search_space object.")
    return space


def _fresh_identity(
    experiment: Experiment, phase: Phase, inherited: Mapping[str, object]
) -> PhaseIdentityRecord:
    """Build the identity a study with no trials is bound to: one search space, from position 0.

    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase whose identity is built.
    :param Mapping[str, object] inherited: The parent layer.
    :return PhaseIdentityRecord: The record.
    """
    payload = _identity_payload(experiment, phase, inherited)
    return _identity_record(
        _semantic_payload_digest(payload),
        payload,
        [{"search_space": _payload_search_space(payload), "first_position": 0}],
    )


def _stamp_identity(study: optuna.Study, record: PhaseIdentityRecord) -> None:
    """Bind a study to an identity in one write.

    :param optuna.Study study: Study being bound to the phase config.
    :param PhaseIdentityRecord record: The identity to record.
    """
    study.set_user_attr(PHASE_IDENTITY_ATTR, record)


def _verify_fingerprint(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Bind the study to the phase's current identity, or refuse the change; see :func:`_judge_identity`.

    :param optuna.Study study: Study being verified or stamped.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase whose fingerprint must match.
    :param dict[str, Winner] inherited_winners: Parent winners contributing to identity.
    :raises StudyFingerprintMismatchError: The populated study has no persisted
        fingerprint or a change it cannot continue under; the message names the keys that
        changed.
    :return str: Verified current fingerprint.
    """
    inherited = _inherited_override_payload(phase, inherited_winners)
    verdict = _judge_identity(study, experiment, phase, inherited)
    if verdict.record is not None:
        if verdict.rebound_from is not None:
            log.warning(
                "Rebinding the fingerprint of empty study %s (%s -> %s): no trial "
                "ever ran under the previous config.",
                study.study_name,
                short_digest(verdict.rebound_from),
                short_digest(verdict.fingerprint),
            )
        if verdict.gained:
            log.info(
                "[%s] search_space gains %s: its finished trials stand and new configurations "
                "run from draw position %d.",
                phase.name,
                describe_gains(verdict.gained, sampler=phase.sampler),
                verdict.record["search_spaces"][-1]["first_position"],
            )
        _stamp_identity(study, verdict.record)
    return verdict.fingerprint


@dataclass(frozen=True)
class _IdentityVerdict:
    """What :func:`_judge_identity` found, and the record a run writes for it."""

    #: The current config's fingerprint.
    fingerprint: str
    #: The identity to record, or None when the recorded one stands unchanged.
    record: PhaseIdentityRecord | None
    #: The values each grown key gains over the space the study continues; empty otherwise.
    gained: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: The fingerprint an empty study was bound to before, when it is rebound.
    rebound_from: str | None = None


def _next_draw_position(trials: Sequence[optuna.trial.FrozenTrial], phase: Phase) -> int:
    """Return the draw position a search space taking effect now starts at.

    Under repeats it is rounded up to a whole configuration, so every member of a configuration
    is drawn under one space.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of the study.
    :param Phase phase: Phase whose repeat axis groups positions.
    :return int: The first position no trial holds, rounded up under repeats.
    """
    drawn = len(set(_draw_positions(trials).values()))
    count = 1 if phase.repeats is None else len(phase.repeats.values)
    return -(-drawn // count) * count


def _judge_identity(
    study: optuna.Study, experiment: Experiment, phase: Phase, inherited: Mapping[str, object]
) -> _IdentityVerdict:
    """Judge the current config against a study's recorded identity, writing nothing.

    A study with no trials takes the current identity outright. A populated one continues when
    only its ``search_space`` changed and every changed key only gains values
    (:func:`~phasesweep.engine.growth._search_space_growth`): the record gains an entry that
    takes effect at the next whole configuration, so every finished trial stands and only new
    configurations run. A newest entry no trial was drawn under is replaced, or dropped when the
    config is back to the space before it, so an unused growth never binds the study.

    :param optuna.Study study: The phase's study, live or a read-only snapshot.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase the study belongs to.
    :param Mapping[str, object] inherited: The parent layer.
    :raises StudyFingerprintMismatchError: The populated study has no recorded identity, or its
        config changed in a way it cannot continue under; the message names what changed.
    :raises StudySchemaMismatchError: The recorded identity is malformed.
    :return _IdentityVerdict: The current fingerprint and the record to write, if any.
    """
    return _judge_record(
        _recorded_identity(study),
        study.get_trials(deepcopy=False),
        study.study_name,
        experiment,
        phase,
        inherited,
    )


def _judge_record(
    record: PhaseIdentityRecord | None,
    trials: Sequence[optuna.trial.FrozenTrial],
    study_name: str,
    experiment: Experiment,
    phase: Phase,
    inherited: Mapping[str, object],
) -> _IdentityVerdict:
    """Judge the current config against an identity record and the trials it holds.

    :func:`_judge_identity` judges a study's own record; a study filled under changed reading
    settings is judged on its source's record and trials before it is written
    (:mod:`phasesweep.engine.reread`).

    :param PhaseIdentityRecord | None record: The recorded identity, or ``None``.
    :param Sequence[optuna.trial.FrozenTrial] trials: The trials the record covers.
    :param str study_name: The study the refusal names.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase the record belongs to.
    :param Mapping[str, object] inherited: The parent layer.
    :raises StudyFingerprintMismatchError: Trials without a record, or a config change the
        trials cannot continue under; the message names what changed.
    :return _IdentityVerdict: The current fingerprint and the record to write, if any.
    """
    fresh = _fresh_identity(experiment, phase, inherited)
    payload, fingerprint = fresh["payload"], fresh["fingerprint"]
    space = fresh["search_spaces"][0]["search_space"]
    if record is None:
        if trials:
            raise StudyFingerprintMismatchError(
                f"Phase {phase.name!r} study {study_name!r} has no semantic fingerprint.",
                remediation=NEW_STUDY_STEP,
            )
        return _IdentityVerdict(fingerprint, fresh)
    if record["fingerprint"] == fingerprint:
        return _IdentityVerdict(fingerprint, None)
    if not trials:
        # A zero-trial study has no result semantics to preserve. This includes a process that
        # died after recording its trial target but before Optuna created the first trial.
        return _IdentityVerdict(fingerprint, fresh, rebound_from=record["fingerprint"])
    entries = record["search_spaces"]
    first = _next_draw_position(trials, phase)
    base = entries[:-1] if len(entries) > 1 and first <= entries[-1]["first_position"] else entries
    base_payload = _with_search_space(record["payload"], base[-1]["search_space"])
    if base is not entries and _same_value(base_payload, payload):
        return _IdentityVerdict(
            fingerprint,
            _identity_record(fingerprint, payload, list(base)),
        )
    growth = _growth_verdicts(base_payload, payload, phase)
    if growth is not None and all(verdict.grows for verdict in growth.values()):
        entry: SearchSpaceEntry = {"search_space": space, "first_position": first}
        grown = [*base, entry]
        return _IdentityVerdict(
            fingerprint,
            _identity_record(fingerprint, payload, grown),
            gained={key: verdict.gained for key, verdict in growth.items() if verdict.gained},
        )
    raise _mismatch_refusal(
        study_name,
        experiment,
        phase,
        existing=record["fingerprint"],
        recorded=base_payload,
        payload=payload,
        growth=growth,
    )


def _with_search_space(payload: JsonObject, space: JsonObject) -> JsonObject:
    """Return a fingerprint payload with its phase search space replaced.

    :param JsonObject payload: A canonical phase fingerprint payload.
    :param JsonObject space: The search space to put in its place.
    :return JsonObject: The new payload; ``payload`` is unchanged.
    """
    phase = payload.get("phase")
    if not isinstance(phase, dict):
        return payload
    return {**payload, "phase": {**phase, "search_space": space}}


def _growth_verdicts(
    recorded_payload: JsonObject, payload: JsonObject, phase: Phase
) -> dict[str, _GrowthVerdict] | None:
    """Judge each changed search-space key, when ``search_space`` is all that changed.

    Keys are compared whole, not through :func:`_changed_payload_keys`'s dotted paths, since a
    search-space key such as ``optimizer.lr`` holds dots itself.

    :param JsonObject recorded_payload: The payload of the space the study continues.
    :param JsonObject payload: The current payload.
    :param Phase phase: Phase whose sampler judges the change.
    :return dict[str, _GrowthVerdict] | None: A verdict per changed key, or None when anything
        outside ``search_space`` changed, or a key was added or removed, which old trials never
        recorded or still hold.
    """
    recorded, current = _payload_search_space(recorded_payload), _payload_search_space(payload)
    if _changed_payload_keys(_with_search_space(recorded_payload, current), payload) or set(
        recorded
    ) != set(current):
        return None
    verdicts: dict[str, _GrowthVerdict] = {}
    for key in sorted(key for key in current if not _same_value(recorded[key], current[key])):
        old, new = recorded[key], current[key]
        if not isinstance(old, dict) or not isinstance(new, dict):
            return None
        verdicts[key] = _search_space_growth(old, new, sampler=phase.sampler)
    return verdicts or None


def _verify_recorded_identity(study: optuna.Study, experiment: Experiment, phase: Phase) -> None:
    """Refuse a populated study whose experiment or phase config changed, before any phase runs.

    A study is addressed by the values it inherits (:func:`phase_study_name`), so the values it
    recorded are the ones it would run under, and only the phase's own config can differ. A
    study without a recorded payload or trials is left to :func:`_verify_fingerprint` once the
    phase is reached.

    :param optuna.Study study: A reached phase's existing study, which this never writes.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase the study belongs to.
    :raises StudyFingerprintMismatchError: The study's own config differs from the one it
        recorded in a way it cannot continue under; the refusal :func:`_judge_identity` gives.
    """
    record = _recorded_identity(study)
    if (
        record is None
        or "inherited_effective_overrides" not in record["payload"]
        or not study.get_trials(deepcopy=False)
    ):
        return
    _judge_identity(study, experiment, phase, _recorded_inheritance(record))


def _growth_refusal(phase: Phase, growth: Mapping[str, _GrowthVerdict]) -> tuple[str, str] | None:
    """Say why a search space that changed alone cannot continue, and the step for it.

    :param Phase phase: Phase whose search space changed.
    :param Mapping[str, _GrowthVerdict] growth: A verdict per changed key.
    :return tuple[str, str] | None: The sentence and the step, or None when every key grows.
    """
    rename = rename_step(phase.name)
    dropped = {key: verdict.dropped for key, verdict in growth.items() if verdict.dropped}
    if dropped:
        named = "; ".join(f"{key} {', '.join(values)}" for key, values in dropped.items())
        sentences = " ".join(
            f"search_space {key!r} dropped {', '.join(values)};" for key, values in dropped.items()
        )
        return (
            f"{sentences} added values alone keep the study and run only the new ones.",
            f"Put back {named}, or {rename}.",
        )
    for key, verdict in growth.items():
        if verdict.refusal is not None:
            lead = (
                f"Restore {_RECORDED_VALUES}"
                if verdict.route is None
                else verdict.route.format(phase=phase.name)
            )
            return f"search_space {key!r} {verdict.refusal}.", f"{lead}, or {rename}."
    return None


def _mismatch_refusal(
    study_name: str,
    experiment: Experiment,
    phase: Phase,
    *,
    existing: str,
    recorded: JsonObject,
    payload: JsonObject,
    growth: Mapping[str, _GrowthVerdict] | None = None,
) -> StudyFingerprintMismatchError:
    """Build the refusal of a populated study whose recorded fingerprint differs from ``payload``.

    :param str study_name: The populated study.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase the study belongs to.
    :param str existing: The fingerprint the study recorded.
    :param JsonObject recorded: The payload of the config the study's trials ran under.
    :param JsonObject payload: The current payload, whose digest differs from ``existing``.
    :param Mapping[str, _GrowthVerdict] | None growth: A verdict per changed key when only the
        search space changed, so the refusal names what was dropped and the step for it.
    :return StudyFingerprintMismatchError: The refusal, naming the keys that changed.
    """
    changes = _changed_payload_keys(recorded, payload)
    cwd_unset = experiment.execution.cwd is None and any(
        path == "execution.cwd" for path, _old, _new in changes
    )
    refused = None if growth is None else _growth_refusal(phase, growth)
    # A recorded payload equal to the current one under a different digest was not written by
    # PhaseSweep, so the two digests are all the message can show.
    if not changes:
        way_forward = NEW_STUDY_STEP
        found = (
            f" (fingerprint {short_digest(existing)} != "
            f"{short_digest(_semantic_payload_digest(payload))})."
        )
    elif refused is not None:
        found = f". {_describe_changed_keys(changes)} {refused[0]}"
        way_forward = refused[1]
    else:
        found = f". {_describe_changed_keys(changes)}"
        if all(path.startswith("phase.") for path, _old, _new in changes):
            way_forward = restore_or_rename_step(phase, recorded=_RECORDED_VALUES)
        else:
            way_forward = restore_or_new_study(_RECORDED_VALUES)
    return StudyFingerprintMismatchError(
        f"Study {study_name!r} was created with a different config{found}",
        # Running from the original directory keeps the study; a new name does not.
        action=OperatorAction.FIX_CONFIG if cwd_unset else None,
        remediation=_CWD_FOLLOWS_INVOCATION if cwd_unset else way_forward,
    )
