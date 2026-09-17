"""Semantic identities and study fingerprint validation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import optuna

from phasesweep.config import Experiment, Phase, Suite
from phasesweep.engine.errors import (
    StudyFingerprintMismatchError,
)
from phasesweep.engine.state import (
    FAILURE_REASON_ATTR,
    PHASE_EVALUATION_SEMANTICS_ATTR,
    PHASE_FINGERPRINT_ATTR,
    Winner,
)
from phasesweep.evidence.evaluation import (
    DIRECTORY_SIZE_EVALUATION_REVISION,
    LOG_REGEX_EVALUATION_REVISION,
)
from phasesweep.evidence.models import ArtifactSizeGate, LogRegexExtractor

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
        "allow_seed_search",
    }
)
# v5 / v4 / v4: the embedded complete trainer_config is now part of the core
# experiment contract and contributes to experiment, suite, and phase identity.
# v4 / v3 / v3: an omitted execution.cwd contributes its effective
# invocation directory instead of an unbound null. The earlier execution
# contract work covered only configured cwd values, which still let identical
# persistent-study identities launch different relative commands from two
# invocation directories. whole_node phases additionally fingerprint their
# configured device-set size — the trainer's world size.
# Existing populated studies from earlier schemas fail the fingerprint check
# on resume; see docs/config.md's upgrade section.
FINGERPRINT_SCHEMA_VERSION = 5
SUITE_FINGERPRINT_SCHEMA_VERSION = 4
EXPERIMENT_FINGERPRINT_SCHEMA_VERSION = 4


def _evaluation_semantics(experiment: Experiment, phase: Phase | None = None) -> dict[str, int]:
    """Return revisions only for evaluators that can affect these results.

    :param Experiment experiment: Config supplying objectives, constraints, and contracts.
    :param Phase | None phase: One phase, or all phases for a publication identity.
    :return dict[str, int]: Active evaluator kinds and their semantic revisions.
    """
    revisions: dict[str, int] = {}
    if isinstance(experiment.metric.extractor, LogRegexExtractor) or any(
        isinstance(constraint.extractor, LogRegexExtractor) for constraint in experiment.constraints
    ):
        revisions["log_regex extractor"] = LOG_REGEX_EVALUATION_REVISION
    phases = experiment.phases if phase is None else [phase]
    for candidate in phases:
        gates = list(candidate.gates)
        for contract_name in candidate.contracts:
            gates.extend(experiment.contracts[contract_name].gates)
        if any(isinstance(gate, ArtifactSizeGate) and gate.source == "directory" for gate in gates):
            revisions["artifact_size directory gate"] = DIRECTORY_SIZE_EVALUATION_REVISION
    return revisions


def _execution_identity(experiment: Experiment) -> dict[str, Any]:
    """Return the execution contract's contribution to semantic fingerprints.

    The trainer's working directory and ambient-environment inheritance are
    semantic inputs: two invocations differing in either can evaluate
    different code or data under one study (review v0.5.17 / blocker 4). A
    configured cwd contributes its RESOLVED path — a relative cwd invoked
    from two directories is two different execution contexts and must not
    share a study. An unconfigured cwd contributes the resolved invocation
    directory because that is where the trainer actually runs. Ambient
    Ambient variable *values* are enforced through the trial environment
    cohort rather than embedded in this config digest. Pass-through names and
    their classification are config semantics, but their credential values
    may rotate. Put fixed semantic values in ``env``, which is fingerprinted.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return dict[str, Any]: JSON-serialisable execution-identity payload.
    """
    contract = experiment.execution.inherit_env
    identity = {
        "cwd": str(
            Path(experiment.execution.cwd).expanduser().resolve()
            if experiment.execution.cwd is not None
            else Path.cwd().resolve()
        ),
        "inherit_env": sorted(contract) if isinstance(contract, list) else contract,
    }
    if experiment.execution.passthrough_env:
        identity["passthrough_env"] = sorted(experiment.execution.passthrough_env)
    return identity


def _semantic_payload_digest(payload: object) -> str:
    """Return the canonical digest used for persisted experiment semantics.

    :param object payload: JSON-compatible semantic identity payload.
    :return str: SHA-256 of the canonical compact JSON representation.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _experiment_semantic_fingerprint(experiment: Experiment) -> str:
    """Hash the experiment semantics that give a published result its meaning.

    Stamped into each generation's summary manifest so reads can tell
    whether the config supplied *today* still matches the config that
    produced a published result (review v0.5.16 / blocker 4) — metric
    name/goal/extractor, constraints, trial command, env, provenance, and
    every phase's ordered semantic identity all contribute. Run-control
    fields (``n_trials`` top-ups, throughput knobs, comments) are excluded
    for the same reason :data:`_RUN_CONTROL_KEYS` excludes them from phase
    fingerprints: they never change what the published numbers mean, so they
    must not flag a published result as reinterpreted.

    :param Experiment experiment: Parsed experiment config to fingerprint.
    :return str: SHA-256 hex digest (64 characters) of the canonicalised
        semantic payload.
    """
    payload = {
        "fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "trial_command": experiment.trial_command,
        "trainer_config": experiment.trainer_config,
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "execution": _execution_identity(experiment),
        "provenance": dict(sorted(experiment.provenance.items())),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        "contracts": {
            name: contract.model_dump(mode="json")
            for name, contract in sorted(experiment.contracts.items())
        },
        "phases": [
            {"name": phase.name, **_semantic_phase_dump(phase)} for phase in experiment.phases
        ],
    }
    if revisions := _evaluation_semantics(experiment):
        payload["evaluation_semantics"] = revisions
    return _semantic_payload_digest(payload)


def _semantic_phase_dump(phase: Phase) -> dict[str, Any]:
    """Return one phase's model dump reduced to execution and output semantics.

    ``gpu_ids``/``gpu_devices`` are run-control (which card runs a trial does
    not change its meaning) — except under ``whole_node``, where the configured
    device-set SIZE is the trainer's world size and therefore semantic: a
    4-GPU DDP evaluation and a 1-GPU evaluation of the same phase must not
    share a study (review v0.5.17 gap hunt). Only the count joins the
    fingerprint, so respelling the same set (indices vs UUIDs) or moving
    hosts does not invalidate a study.

    :param Phase phase: Phase whose semantic payload is being built.
    :return dict[str, Any]: JSON-serializable semantic phase payload.
    """
    # Promotion runs after a phase's individual trials, but is still semantic:
    # it determines the phase output persisted for this candidate (or whether
    # it is replaced by a baseline or omitted) and therefore the overrides
    # downstream phases inherit. A populated study must not replay that
    # transition under edited promotion rules.
    dump = {k: v for k, v in phase.model_dump(mode="json").items() if k not in _RUN_CONTROL_KEYS}
    # acknowledge_nonresumable is run-control, not semantics: it never changes
    # what a trial samples or means (on persistent storage its legal value is
    # fully determined by sampler.type), so it must not invalidate a study.
    dump["sampler"].pop("acknowledge_nonresumable", None)
    if phase.gpu_policy == "whole_node":
        tokens = phase.gpu_ids if phase.gpu_ids is not None else phase.gpu_devices
        dump["whole_node_device_count"] = len(tokens or [])
    return dump


def _suite_fingerprint(suite: Suite, experiments: Mapping[str, Experiment] | None = None) -> str:
    """Hash the fully compiled suite plan, including historical annotations.

    :param Suite suite: Suite whose study names, dependency edges, promotion
        rules, and resolved experiments contribute to the digest.
    :param Mapping[str, Experiment] | None experiments: Already resolved studies,
        when called during suite execution.
    :return str: SHA-256 of the canonical suite payload.
    """
    studies = []
    for study in suite.studies:
        experiment = (
            suite.experiment_for_study(study) if experiments is None else experiments[study.name]
        )
        item = {
            "name": study.name,
            "depends_on": study.depends_on,
            "promotion": (
                None if study.promotion is None else study.promotion.model_dump(mode="json")
            ),
            "experiment": experiment.model_dump(mode="json"),
            "execution_identity": _execution_identity(experiment),
        }
        if revisions := _evaluation_semantics(experiment):
            item["evaluation_semantics"] = revisions
        studies.append(item)
    payload = {
        "fingerprint_schema_version": SUITE_FINGERPRINT_SCHEMA_VERSION,
        "suite": suite.suite,
        "studies": studies,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _phase_semantic_payload(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> dict[str, Any]:
    """Return a dict capturing only fields that change *trial meaning*.

    Excludes run-control fields (review v0.5.2 / blocker 1) so that bumping
    ``n_trials`` to top up a study is a compatible operation. Includes
    ``experiment.env`` which v0.5.1 missed: env vars like ``CUBLAS_WORKSPACE_CONFIG``
    or ``MY_TRAINER_SEED`` change training behavior and must invalidate reuse.

    :param Experiment experiment: Experiment supplying command, input format,
        environment, metric, constraints, and provenance.
    :param Phase phase: Phase whose run-control keys are excluded.
    :param dict[str, Winner] inherited_winners: Parent winners whose effective
        overrides contribute to identity.
    :return dict[str, Any]: JSON-serializable configured trial semantics.
    """
    semantic_phase = _semantic_phase_dump(phase)
    payload = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "trial_command": experiment.trial_command,
        "trainer_config": experiment.trainer_config,
        "provenance": dict(sorted(experiment.provenance.items())),
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "execution": _execution_identity(experiment),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        "contracts": {
            name: experiment.contracts[name].model_dump(mode="json") for name in phase.contracts
        },
        "phase": semantic_phase,
        "inherited_effective_overrides": {
            parent: inherited_winners[parent].effective_overrides for parent in phase.inherits
        },
    }
    if revisions := _evaluation_semantics(experiment, phase):
        payload["evaluation_semantics"] = revisions
    return payload


def _phase_fingerprint(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Hash the semantic execution context for resume-compatibility checks.

    Uses the full SHA-256 hex digest. Earlier versions truncated to 16 hex
    chars (64 bits) — defensible against accidental collision but no reason
    to leave the door open in scientific-workflow metadata.

    :param Experiment experiment: Experiment forwarded to
        :func:`_phase_semantic_payload`.
    :param Phase phase: Phase being fingerprinted.
    :param dict[str, Winner] inherited_winners: Parent winners contributing
        effective overrides to identity.
    :return str: SHA-256 of the canonical semantic payload.
    """
    payload = _phase_semantic_payload(experiment, phase, inherited_winners)
    return _semantic_payload_digest(payload)


def _only_failed_trials(
    study: optuna.Study, experiment: Experiment, revisions: Mapping[str, int]
) -> bool:
    """Return whether failed trials cannot contain affected evaluator readings.

    :param optuna.Study study: Existing phase study after stale-trial reaping.
    :param Experiment experiment: Config naming the affected metric and constraints.
    :param Mapping[str, int] revisions: Evaluators whose semantics changed.
    :return bool: ``True`` for an empty study or failed trials whose recorded
        causes are independent of the affected evaluators.
    """
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.FAIL:
            return False
        reason = trial.user_attrs.get(FAILURE_REASON_ATTR)
        if not isinstance(reason, str) or not reason.strip():
            return False
        if (
            (
                isinstance(experiment.metric.extractor, LogRegexExtractor)
                and reason.startswith("metric extractor")
            )
            or any(
                isinstance(constraint.extractor, LogRegexExtractor)
                and reason.startswith(f"constraint extractor {constraint.name!r}")
                for constraint in experiment.constraints
            )
            or (
                "artifact_size directory gate" in revisions
                and reason.startswith("evidence gates failed:")
            )
        ):
            return False
    return True


def _verify_evaluation_semantics(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    *,
    stamp_safe: bool,
) -> None:
    """Refuse an affected study whose historical evaluator revision is unknown.

    :param optuna.Study study: Existing or newly created phase study.
    :param Experiment experiment: Current evidence configuration.
    :param Phase phase: Phase whose direct and contract gates contribute.
    :param bool stamp_safe: Record the revision after a safe fingerprint check.
    :raises StudyFingerprintMismatchError: Reusable or running trials have no
        matching evaluator revision.
    """
    revisions = _evaluation_semantics(experiment, phase)
    if not revisions:
        return
    recorded = study.user_attrs.get(PHASE_EVALUATION_SEMANTICS_ATTR)
    if recorded == revisions:
        return
    if not _only_failed_trials(study, experiment, revisions):
        evaluators = ", ".join(revisions)
        raise StudyFingerprintMismatchError(
            f"Phase {phase.name!r} study {study.study_name!r} has incompatible or "
            f"unrecorded evaluation semantics for {evaluators} (stored {recorded!r}, "
            f"current {revisions!r}). Its populated history cannot establish which "
            "evidence interpretation produced those readings. Preserve the historical "
            "artifacts and use a fresh experiment identity and ledger."
        )
    if stamp_safe:
        study.set_user_attr(PHASE_EVALUATION_SEMANTICS_ATTR, revisions)


def _verify_fingerprint(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Stamp a fresh study with its fingerprint or fail on mismatch.

    :param optuna.Study study: Study being verified or stamped.
    :param Experiment experiment: Current experiment config.
    :param Phase phase: Phase whose fingerprint must match.
    :param dict[str, Winner] inherited_winners: Parent winners contributing to identity.
    :raises StudyFingerprintMismatchError: The populated study has an incompatible
        persisted fingerprint or evaluator revision.
    :return str: Verified current fingerprint.
    """
    _verify_evaluation_semantics(study, experiment, phase, stamp_safe=False)
    fp = _phase_fingerprint(experiment, phase, inherited_winners)
    existing = study.user_attrs.get(PHASE_FINGERPRINT_ATTR)
    if existing is None:
        if (revisions := _evaluation_semantics(experiment, phase)) and not _only_failed_trials(
            study, experiment, revisions
        ):
            evaluators = ", ".join(revisions)
            raise StudyFingerprintMismatchError(
                f"Phase {phase.name!r} study {study.study_name!r} has no semantic "
                f"fingerprint for {evaluators}. Its populated history cannot establish "
                "which evidence interpretation produced those readings. Preserve the "
                "historical artifacts and use a fresh experiment identity and ledger."
            )
        study.set_user_attr(PHASE_FINGERPRINT_ATTR, fp)
    elif existing != fp:
        # A zero-trial study must not permanently bind its semantic identity:
        # nothing was ever evaluated under the old fingerprint, so rebinding
        # cannot mix results. This includes a process that died after recording
        # its trial target but before Optuna created the first trial.
        if not study.get_trials(deepcopy=False):
            log.warning(
                "Rebinding the fingerprint of empty study %s (%s -> %s): no trial "
                "ever ran under the previous config.",
                study.study_name,
                existing,
                fp,
            )
            study.set_user_attr(PHASE_FINGERPRINT_ATTR, fp)
            _verify_evaluation_semantics(study, experiment, phase, stamp_safe=True)
            return fp
        if (revisions := _evaluation_semantics(experiment, phase)) and _only_failed_trials(
            study, experiment, revisions
        ):
            legacy_payload = _phase_semantic_payload(experiment, phase, inherited_winners)
            legacy_payload.pop("evaluation_semantics", None)
            if existing == _semantic_payload_digest(legacy_payload):
                log.warning(
                    "Rebinding legacy evaluator fingerprint of failed-only study %s: "
                    "no reusable trial readings exist.",
                    study.study_name,
                )
                study.set_user_attr(PHASE_FINGERPRINT_ATTR, fp)
                _verify_evaluation_semantics(study, experiment, phase, stamp_safe=True)
                return fp
        if revisions := _evaluation_semantics(experiment, phase):
            evaluators = ", ".join(revisions)
            raise StudyFingerprintMismatchError(
                f"Phase {phase.name!r} study {study.study_name!r} was created with a "
                f"different phase config or evaluator interpretation for {evaluators} "
                f"(semantic fingerprint {existing} != {fp}). "
                "Resuming could mix incompatible evidence interpretations. Preserve the "
                "historical artifacts and use a fresh experiment identity and ledger."
            )
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} was created with a different phase config "
            f"(fingerprint {existing} != {fp}). Use a new experiment name, delete the "
            f"old study, or rename the phase."
        )
    _verify_evaluation_semantics(study, experiment, phase, stamp_safe=True)
    return fp


log = logging.getLogger(__name__)
