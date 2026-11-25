"""Artifact serialization, logging, and winner persistence."""

from __future__ import annotations

import contextlib
import csv
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import optuna
import yaml

import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.paths as path_ops
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import (
    ResumeWinnerMissingError,
    StudyFingerprintMismatchError,
    WinnerIntegrityError,
)
from phasesweep.engine.optuna import _trial_count_state
from phasesweep.engine.paths import _valid_trial_dir_name
from phasesweep.engine.progress import _warn_once
from phasesweep.engine.publication_validation import ValidatedPublication
from phasesweep.engine.repeats import (
    RepeatAxis,
    _check_repeats_axis,
    group_labels_for,
    repeat_axis,
)
from phasesweep.engine.state import (
    TRAINER_ENV_ATTR,
    Winner,
    _recorded_identity,
    _trial_attempt_dir,
)
from phasesweep.engine.study_policy import _describe_environment_changes
from phasesweep.engine.trial import _environment_identity
from phasesweep.errors import (
    ACCEPT_INCOMPLETE_STEP,
    OperatorAction,
    PhaseSweepError,
    operator_step,
    short_digest,
)
from phasesweep.runtime.files import atomic_text_writer

log = logging.getLogger(__name__)

# One log line shape for every file destination. ``run.log`` and the MCP runner's captured log
# carry the date (``%Y-%m-%d %H:%M:%S``) because those files outlive the day; the CLI console
# prints its own clock-time layout (``phasesweep.runtime.console``).
LOG_FORMAT = "%(asctime)s %(levelname).1s %(name)s %(message)s"

# The only way past a skipped phase's published winner when no study outlives the run.
_WITHOUT_FROM_PHASE_IN_MEMORY = "without --from-phase to start every phase again"


def read_yaml_or_none(path: Path) -> object:
    """Read optional YAML without propagating read or decoding failures.

    :param Path path: YAML artifact path to read with the platform-default encoding.
    :return object: Decoded YAML value, or ``None`` when absent, unreadable, or malformed.
    """
    try:
        value: object = yaml.safe_load(path.read_text())
        return value
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _write_yaml_atomic(path: Path, payload: Any) -> None:
    """Atomically write a YAML document to ``path``.

    :param Path path: Destination YAML path to replace.
    :param Any payload: YAML-serializable value to write.
    """
    text = yaml.safe_dump(payload, sort_keys=False)
    with atomic_text_writer(path, newline="") as handle:
        handle.write(text)


@contextlib.contextmanager
def _file_log_handler(path: Path) -> Iterator[None]:
    """Capture phasesweep logs, and a routed failure with its advice, in the run log.

    The logger keeps a ``DEBUG`` level a caller set, so a verbose run records debug messages and
    the traceback of a routed failure; otherwise it records from ``INFO``.

    :param Path path: Log file path to append to.
    :raises PhaseSweepError: The operational failure raised inside the context.
    :return Iterator[None]: Context manager that removes the handler on exit.
    """
    logger = logging.getLogger("phasesweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt=LOG_FORMAT,
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
    except PhaseSweepError as exc:
        # The caller prints the error to its own stderr, so the log records it with its step,
        # and with its traceback when DEBUG is on, which the console never shows.
        error_info = (
            (type(exc), exc, exc.__traceback__) if logger.isEnabledFor(logging.DEBUG) else None
        )
        for line, info in ((str(exc), error_info), (f"fix: {operator_step(exc)}", None)):
            handler.handle(
                logger.makeRecord(logger.name, logging.ERROR, __file__, 0, "%s", (line,), info)
            )
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)


def _write_trials_csv(
    study: optuna.Study,
    path: Path,
    experiment: Experiment,
    *,
    repeats: RepeatAxis | None = None,
) -> None:
    """Snapshot every trial in ``study`` to ``path`` as stdlib CSV.

    The ``state`` column is the label run logs and status tables use, so an
    interrupted trial reads ``INTERRUPTED`` and one outside the current bounds ``INFEASIBLE``.
    A repeats phase gets a ``repeat_group`` column: the highest trial number
    in the trial's group so far, grouped by the same key selection uses. An
    incomplete group's label moves as more members are allocated.

    Every user attribute is a column except the tracked trainer environment, which is
    ledger provenance: its ``PATH`` and ``HOME`` values are noise in a results table.
    A rerun's source therefore shows as ``user_attr:phasesweep_rerun_of``. The
    ``attempt_dir`` column names the directory holding the attempt's own logs and records;
    ``user_attr:phasesweep_trial_dir`` is the trainer's directory.

    :param optuna.Study study: Study whose trials are serialized.
    :param Path path: Destination CSV path.
    :param Experiment experiment: Config whose bounds judge each measured trial.
    :param RepeatAxis | None repeats: The phase's repeat axis, or ``None`` without repeats.
    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    param_names = sorted({n for t in trials for n in t.params})
    attr_names = sorted({n for t in trials for n in t.user_attrs} - {TRAINER_ENV_ATTR})
    labels = group_labels_for(trials, repeats)
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration",
        "attempt_dir",
        *(["repeat_group"] if repeats is not None else []),
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
                "state": _trial_count_state(t, experiment),
                "value": t.value,
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete,
                "duration": t.duration,
                "attempt_dir": _trial_attempt_dir(t),
            }
            if repeats is not None:
                row["repeat_group"] = labels.get(t.number)
            for n in param_names:
                row[f"param:{n}"] = t.params.get(n)
            for n in attr_names:
                row[f"user_attr:{n}"] = t.user_attrs.get(n)
            writer.writerow(row)


def _refresh_recovered_trials_csv(
    studies: Mapping[str, Sequence[optuna.Study]], experiment: Experiment
) -> None:
    """Refresh the same addressed studies as status, from the studies recovery opened.

    :param Mapping[str, Sequence[optuna.Study]] studies: Every recovered study by phase.
    :param Experiment experiment: Config that locates the CSV and checks the study semantics.
    """
    from phasesweep.engine.read import _latest_study_names, _read_generation_selection

    if not studies:
        return
    try:
        current, _, publication = _read_generation_selection(experiment)
        addressed = _latest_study_names(experiment, publication, current)
    except Exception as exc:
        log.warning("Could not address recovered studies for trials.csv: %s", exc)
        return

    for phase in experiment.phases:
        name = addressed.get(phase.name)
        study = next(
            (item for item in studies.get(phase.name, ()) if item.study_name == name), None
        )
        if study is None:
            continue
        # Recovery can settle work even when a launch would refuse the supplied phase.
        # Keep the prior projection when that phase cannot supply the study's repeat grouping.
        try:
            recorded = _recorded_identity(study)
            if recorded is not None and recorded["payload"].get(
                "phase"
            ) == fingerprint_ops._identity_phase_dump(phase):
                _write_trials_csv(
                    study,
                    path_ops._trials_csv_path(experiment, phase.name),
                    experiment,
                    repeats=repeat_axis(phase),
                )
        except Exception as exc:
            log.warning("Could not refresh trials.csv for study %r: %s", study.study_name, exc)


def _save_winner(
    experiment: Experiment,
    phase_name: str,
    winner: Winner,
    *,
    generation_id: str,
) -> None:
    """Persist a phase winner into its immutable generation namespace.

    Real winners carry a phase fingerprint for resume validation; dry-run placeholders
    are never saved. Environment identity excludes ambient values. Evidence
    provenance remains a winner-only field.

    :param Experiment experiment: Parsed experiment config; supplies the metric name used in
        the persisted payload.
    :param str phase_name: Name of the phase whose winner is being saved.
    :param Winner winner: The winning trial.
    :param str generation_id: Immutable generation namespace to write into.
    """
    path = path_ops._generation_winner_path(experiment, generation_id, phase_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml_atomic(path, _winner_payload(experiment, phase_name, winner))


def _winner_payload(experiment: Experiment, phase_name: str, winner: Winner) -> dict[str, object]:
    """Build the mapping a phase's ``winner.yaml`` holds.

    :param Experiment experiment: Parsed experiment config; supplies the metric name and goal.
    :param str phase_name: Name of the phase the winner belongs to.
    :param Winner winner: The winning trial.
    :return dict[str, object]: The winner file's payload.
    """
    return {
        "phase": phase_name,
        "metric": {experiment.metric.name: winner.metric, "goal": experiment.metric.goal},
        **_winner_common_payload(winner),
        "phase_fingerprint": winner.phase_fingerprint,
        "study_name": winner.study_name,
        "objective_provenance": winner.objective_provenance,
        "trainer_env_digest": winner.trainer_env_digest,
    }


def _winner_common_payload(winner: Winner) -> dict[str, Any]:
    """Serialize winner fields shared by persisted and summary representations.

    :param Winner winner: Winner whose common fields should be serialized.
    :return dict[str, Any]: Shared trial, parameter, evidence, identity, and source fields.
    """
    source = winner.source
    assert source is not None, "only the never-persisted dry-run placeholder has no source"
    payload = {
        "trial_number": winner.trial_number,
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
        "completion": winner.completion,
        "generation_id": winner.generation_id,
        "attempt_id": winner.attempt_id,
        "trial_dir": winner.trial_dir,
        "winner_source": asdict(source),
        "trainer_input": winner.trainer_input,
        "confound": winner.confound,
        "observed": winner.observed,
        "repeats": winner.repeats,
    }
    return payload


# Resume preflight and execution can read the same winner; both use the same
# warning identity so the environment divergence is reported once.
def _warn_environment_drift(
    experiment: Experiment,
    phase_name: str,
    stored_digest: str,
    stored_environment: dict[str, str] | None = None,
) -> None:
    """Warn once when result reuse crosses semantic environment cohorts.

    Skipped-phase loading and no-work replay reuse results without allocating trials;
    ordinary top-ups refuse cohort drift before allocation. The warning names the
    variables that differ as the cohort refusal does
    (:func:`phasesweep.engine.study_policy._describe_environment_changes`) when the caller
    holds the winning trial's recorded mapping, and shows the two digests otherwise.

    :param Experiment experiment: Parsed experiment supplying the current contract.
    :param str phase_name: Phase whose winner was reused, used in the warn-once key.
    :param str stored_digest: Digest recorded on the reused winner or its trial.
    :param dict[str, str] | None stored_environment: Tracked mapping the winning trial
        recorded, or ``None`` when the caller holds only the digest.
    """
    current = _environment_identity(
        experiment,
        phase_name,
        require_wandb_online=False,
    )
    if stored_digest == current.digest:
        return
    differences = (
        _describe_environment_changes(stored_environment, current.tracked)
        if stored_environment is not None
        else ""
    )
    _warn_once(
        log,
        ("environment-drift", experiment.experiment, phase_name, stored_digest),
        "[%s] reused winner ran under a different trainer environment: %s "
        "(execution.inherit_env=%r).",
        phase_name,
        differences or f"{short_digest(stored_digest)} -> {short_digest(current.digest)}",
        experiment.execution.inherit_env,
    )


def _load_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    publication: ValidatedPublication | None,
) -> Winner:
    """Load validated winner bytes and check reuse against the current config.

    Recompute the phase fingerprint with the current inherited overrides. A skipped
    winner's recorded environment drift warns; ordinary top-ups refuse that drift.
    Publication validation already checked persisted identities and structure, so
    this function checks only current-config-dependent requirements.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase whose winner is being loaded.
    :param dict[str, Winner] inherited_winners: Winners loaded for phases earlier in the chain;
        contribute to the recomputed fingerprint.
    :param ValidatedPublication | None publication: The last-success publication the caller
        resolved with ``raise_on_manifest_error=True``; ``None`` when nothing is published. A
        caller loading several phases resolves it once.
    :raises ResumeWinnerMissingError: The publication holds no winner for the phase.
    :raises WinnerIntegrityError: The winner is incomplete, incompatible with the current
        partial-result policy, or carries a ``repeats`` block that is not the phase's
        configured repeat axis.
    :raises StudyFingerprintMismatchError: The stored fingerprint disagrees with the freshly
        computed one.
    :return Winner: The reconstructed winner for ``phase``.
    """
    if publication is None or phase.name not in publication.winners:
        absent = (
            "no generation has been published."
            if publication is None
            else "The last published generation holds no winner for it."
        )
        raise ResumeWinnerMissingError(
            f"Winner file missing for phase {phase.name!r}: {absent}",
            remediation=(
                "Run without --from-phase so the earlier phases run first."
                if experiment.persistent
                else f"Run {_WITHOUT_FROM_PHASE_IN_MEMORY}."
            ),
        )
    path = path_ops._generation_winner_path(experiment, publication.generation_id, phase.name)
    validated = publication.winners[phase.name]
    data = validated.payload

    current_fp = fingerprint_ops._phase_fingerprint(experiment, phase, inherited_winners)
    stored_fp = data["phase_fingerprint"]
    if stored_fp != current_fp:
        # A winner file records only the hash, so the refusal can name no changed key. MCP shows
        # the step without the message, so the step names the winner it means.
        produced = f"the config that produced the winner of `{phase.name}`"
        raise StudyFingerprintMismatchError(
            f"Winner file {path} was produced by a different config (stored fingerprint "
            f"{short_digest(stored_fp)} != current {short_digest(current_fp)}).",
            action=OperatorAction.FIX_CONFIG,
            remediation=(
                f"{fingerprint_ops.restore_or_rename_step(phase, recorded=produced)} "
                f"If its search_space only gained values, run from `{phase.name}` "
                f"(`--from-phase {phase.name}`)."
                if experiment.persistent
                else f"Restore {produced}, or run {_WITHOUT_FROM_PHASE_IN_MEMORY}."
            ),
        )

    completion = data["completion"]
    if completion.get("incomplete") is True and not phase.allow_incomplete_on_timeout:
        raise WinnerIntegrityError(
            f"Winner file {path} records an incomplete result for skipped phase {phase.name!r}.",
            action=OperatorAction.FIX_CONFIG,
            remediation=ACCEPT_INCOMPLETE_STEP,
        )

    try:
        stored_env_digest = data["trainer_env_digest"]
        _check_repeats_axis(validated.repeats, phase.repeats)
        winner = Winner(
            trial_number=int(data["trial_number"]),
            params=dict(data["params"]),
            effective_overrides=dict(data["effective_overrides"]),
            metric=validated.metric,
            constraints={k: float(v) for k, v in (data.get("constraints") or {}).items()},
            gates=[item for item in (data.get("gates") or []) if isinstance(item, dict)],
            completion=dict(completion),
            phase_fingerprint=stored_fp,
            study_name=data["study_name"],
            generation_id=data["generation_id"],
            attempt_id=data["attempt_id"],
            trial_dir=_valid_trial_dir_name(data["trial_dir"], label="winner trial_dir"),
            source=validated.source,
            objective_provenance=dict(data["objective_provenance"]),
            trainer_input=dict(data["trainer_input"]),
            trainer_env_digest=stored_env_digest,
            confound=validated.confound.copy(),
            observed=dict(validated.observed),
            repeats=validated.repeats,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise WinnerIntegrityError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
    _warn_environment_drift(experiment, phase.name, stored_env_digest)
    return winner
