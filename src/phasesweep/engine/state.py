"""Engine state types, paths, logs, and persisted artifacts."""

from __future__ import annotations

import contextlib
import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import yaml

from phasesweep.config import Experiment, Phase, Suite
from phasesweep.runtime.files import atomic_text_writer


@dataclass
class Winner:
    """Phase winner: sampled params, full effective overrides, and metric value.

    ``phase_fingerprint`` is the SHA-256 of the phase's semantic execution
    context at the time the winner was selected (review v0.5.6 / blocker 3).
    Persisted into ``winner.yaml`` and re-checked when ``--from-phase`` skips
    earlier phases — without that check, editing a parent phase's search
    space, fixed overrides, env, metric, or trial command and then resuming
    would silently inherit the *old* winner against the *new* parent config.

    ``None`` only on placeholder winners produced for the dry-run skip path,
    which never get persisted.
    """

    trial_number: int
    params: dict[str, Any]  # sampled params only
    effective_overrides: dict[str, Any]  # full composed overrides (fixed + inherited + sampled)
    metric: float
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    completion: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] | None = None
    phase_fingerprint: str | None = None
    generation_id: str | None = None
    attempt_id: str | None = None


TRIAL_DIR_ATTR = "phasesweep_trial_dir"
GENERATION_ID_ATTR = "phasesweep_generation_id"
ATTEMPT_ID_ATTR = "phasesweep_attempt_id"
FEASIBLE_ATTR = "phasesweep_feasible"
GATES_ATTR = "phasesweep_gates"
RETURN_CODE_ATTR = "phasesweep_return_code"
DURATION_ATTR = "phasesweep_duration_s"
OVERRIDES_ATTR = "phasesweep_overrides"
CLEANUP_CONFIRMED_ATTR = "phasesweep_cleanup_confirmed"
CLEANUP_RECOVERED_TRIALS_ATTR = "phasesweep_cleanup_recovered_trials"
FAILURE_REASON_ATTR = "phasesweep_failure_reason"
CONSTRAINT_PREFIX = "constraint:"


def constraint_attr(name: str) -> str:
    """Return the persisted user-attr key for a constraint value.

    :param str name: Constraint name from the experiment config.
    :return str: Optuna user-attr key used to store the constraint value.
    """
    return f"{CONSTRAINT_PREFIX}{name}"


def _experiment_dir(experiment: Experiment) -> Path:
    """Return the artifact namespace for one experiment.

    :param Experiment experiment: Experiment config with workdir and name.
    :return Path: Absolute directory for experiment artifacts.
    """
    return Path(experiment.workdir).expanduser().resolve() / experiment.experiment


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Return the artifact namespace for one phase.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name to append under the experiment directory.
    :return Path: Directory for phase artifacts.
    """
    return _experiment_dir(experiment) / phase_name


def _summary_path(experiment: Experiment) -> Path:
    """Return the experiment summary path.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the experiment summary YAML file.
    """
    return _experiment_dir(experiment) / "summary.yaml"


def _run_log_path(experiment: Experiment) -> Path:
    """Path to the durable run log for one experiment.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the experiment run log.
    """
    return _experiment_dir(experiment) / "run.log"


def _trial_dir_for(
    experiment: Experiment,
    phase_name: str,
    trial_number: int,
    *,
    generation_id: str | None = None,
    attempt_id: str | None = None,
) -> Path:
    """Return a trial directory, uniquely scoped when execution ids are supplied.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name containing the trial.
    :param int trial_number: Optuna trial number.
    :param str | None generation_id: Current engine invocation id.
    :param str | None attempt_id: Current subprocess attempt id.
    :return Path: Directory for the trial artifacts.
    """
    if generation_id is None and attempt_id is None:
        return _phase_dir(experiment, phase_name) / f"trial_{trial_number:05d}"
    if generation_id is None or attempt_id is None:
        raise ValueError("generation_id and attempt_id must be supplied together")
    return _phase_dir(experiment, phase_name) / (
        f"trial_{trial_number:05d}__generation_{generation_id}__attempt_{attempt_id}"
    )


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation metadata path.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the current generation YAML file.
    """
    return _experiment_dir(experiment) / "generation.yaml"


def _winner_path(experiment: Experiment, phase_name: str) -> Path:
    """Return the path to a phase's persisted winner.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose winner path is requested.
    :return Path: Path to the persisted winner YAML file.
    """
    return _phase_dir(experiment, phase_name) / "winner.yaml"


def _promotion_decision_path(experiment: Experiment, phase_name: str) -> Path:
    """Path to the persisted phase promotion decision.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose promotion decision path is
        requested.
    :return Path: Path to the persisted promotion decision YAML file.
    """
    return _phase_dir(experiment, phase_name) / "promotion.yaml"


def _suite_dir(suite: Suite) -> Path:
    """Filesystem namespace for suite-level summary/log artifacts.

    :param Suite suite: Suite config with default artifact settings.
    :return Path: Absolute directory for suite artifacts.
    """
    return Path(suite.defaults.workdir).expanduser().resolve() / suite.suite


def _suite_summary_path(suite: Suite) -> Path:
    """Path to a suite-level summary.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the suite summary YAML file.
    """
    return _suite_dir(suite) / "suite_summary.yaml"


def _suite_log_path(suite: Suite) -> Path:
    """Path to a suite-level run log.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the suite run log.
    """
    return _suite_dir(suite) / "run.log"


def _write_yaml_atomic(path: Path, payload: Any) -> None:
    """Atomically write a YAML document to ``path``.

    :param Path path: Destination YAML path to replace.
    :param Any payload: YAML-serializable value to write.
    """
    with atomic_text_writer(path) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


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


def _save_winner(experiment: Experiment, phase_name: str, winner: Winner) -> None:
    """Persist a phase winner.

    The phase fingerprint is included so ``_load_winner`` can refuse stale
    winners on ``--from-phase`` resume (review v0.5.6 / blocker 3). Real
    winners always carry a fingerprint by construction in ``_run_phase``;
    placeholder winners (dry-run skip) are never saved.

    Args:
        experiment: Parsed experiment config; supplies the metric name used
            in the persisted payload.
        phase_name: Name of the phase whose winner is being saved.
        winner: The winning trial.

    """
    path = _winner_path(experiment, phase_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase_name,
        "trial_number": winner.trial_number,
        "metric": {experiment.metric.name: winner.metric, "goal": experiment.metric.goal},
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
        "completion": winner.completion,
        "phase_fingerprint": winner.phase_fingerprint,
        "generation_id": winner.generation_id,
        "attempt_id": winner.attempt_id,
    }
    if winner.promotion is not None:
        payload["promotion"] = winner.promotion
    _write_yaml_atomic(path, payload)


def _save_promotion_decision(
    experiment: Experiment,
    phase_name: str,
    decision: dict[str, Any],
) -> None:
    """Persist a phase promotion decision independently of exposed winner state.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose promotion decision is being saved.
    :param dict[str, Any] decision: Promotion decision payload to persist.
    """
    path = _promotion_decision_path(experiment, phase_name)
    _write_yaml_atomic(path, decision)


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

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose winner is being loaded.
        inherited_winners: Winners loaded for phases earlier in the chain;
            contribute to the recomputed fingerprint.

    Returns:
        The reconstructed :class:`Winner` for ``phase``.

    Raises:
        FileNotFoundError: ``winner.yaml`` does not exist for the phase.
        RuntimeError: The file is unfingerprinted (legacy/hand-edited) or its
            fingerprint disagrees with the freshly computed one.

    """
    path = _winner_path(experiment, phase.name)
    if not path.is_file():
        raise FileNotFoundError(f"Winner file missing for phase {phase.name!r}: {path}")

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase "
            f"{phase.name!r}: top level must be a mapping."
        )

    from phasesweep.engine.guards import _phase_fingerprint

    current_fp = _phase_fingerprint(experiment, phase, inherited_winners)
    stored_fp = data.get("phase_fingerprint")

    if stored_fp is None:
        raise RuntimeError(
            f"Winner file {path} has no phase_fingerprint. Refusing to use it "
            f"for --from-phase because phasesweep cannot prove it matches the "
            f"current config for skipped phase {phase.name!r}. Re-run the "
            f"phase, or — if you know the config is unchanged — delete the "
            f"file and re-run to regenerate it with a fingerprint."
        )

    if stored_fp != current_fp:
        raise RuntimeError(
            f"Winner file {path} was produced by a different phase config "
            f"(stored fingerprint {stored_fp[:16]}... != current "
            f"{current_fp[:16]}...). Re-run phase {phase.name!r}, change the "
            f"experiment name, or restore the matching config before resuming."
        )

    completion = data.get("completion")
    if not isinstance(completion, dict):
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase "
            f"{phase.name!r}: missing mapping field 'completion'."
        )
    if completion.get("incomplete") is True and not phase.allow_incomplete_on_timeout:
        raise RuntimeError(
            f"Winner file {path} records an incomplete phase result. Refusing to "
            f"use it for skipped phase {phase.name!r} unless the current config "
            "sets allow_incomplete_on_timeout: true."
        )
    generation_id = data.get("generation_id")
    attempt_id = data.get("attempt_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise RuntimeError(
            f"Winner file {path} has no valid generation_id; refusing unscoped evidence."
        )
    if not isinstance(attempt_id, str) or not attempt_id:
        raise RuntimeError(
            f"Winner file {path} has no valid attempt_id; refusing unscoped evidence."
        )

    try:
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
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
