"""Pure filesystem paths for engine artifacts."""

from __future__ import annotations

from pathlib import Path

from phasesweep.config import Experiment, Suite
from phasesweep.engine.state import (
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_REPRODUCIBILITY_FILENAME,
)


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
    :raises ValueError: Exactly one of ``generation_id`` and ``attempt_id`` was
        supplied; a uniquely scoped directory needs both.
    """
    if generation_id is None and attempt_id is None:
        return _phase_dir(experiment, phase_name) / f"trial_{trial_number:05d}"
    if generation_id is None or attempt_id is None:
        raise ValueError("generation_id and attempt_id must be supplied together")
    return _phase_dir(experiment, phase_name) / (
        f"trial_{trial_number:05d}__generation_{generation_id}__attempt_{attempt_id}"
    )


def _attempts_dir(experiment: Experiment) -> Path:
    """Return the experiment-level active-attempt registry directory.

    One JSON entry per nonterminal attempt, written at allocation and
    retired once the attempt's Optuna trial is durably terminal. Recovery
    scans this registry *independently of the current phase graph*, so a
    renamed or removed phase cannot hide a stale attempt from preflight
    (review v0.5.17 / blocker 3).

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Directory containing active-attempt registry entries.
    """
    return _experiment_dir(experiment) / "attempts"


def _artifact_root_binding_path(experiment: Experiment) -> Path:
    """Return the reverse artifact-root-to-storage ownership record.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Root binding JSON path inside the experiment namespace.
    """
    return _experiment_dir(experiment) / "artifact_root_binding.json"


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation metadata path.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the current generation YAML file.
    """
    return _experiment_dir(experiment) / "generation.yaml"


def _generations_dir(experiment: Experiment) -> Path:
    """Return the immutable generation-record root for an experiment.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Directory containing all immutable per-generation namespaces.
    """
    return _experiment_dir(experiment) / "generations"


def _generation_dir(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's immutable artifact namespace.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Directory scoped to the given generation.
    """
    return _generations_dir(experiment) / generation_id


def _generation_record_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's lifecycle record path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's lifecycle record YAML file.
    """
    return _generation_dir(experiment, generation_id) / "generation.yaml"


# A generation's summary is written only as the last step of a successful run,
# so its presence is what separates a published generation from a claim-time
# namespace whose run never got that far. Named once because manifest
# validation reasons about that distinction for *another* generation's
# namespace, where the helper below cannot be used (PR #5 review / reviewer 2,
# blocker 7).
GENERATION_SUMMARY_FILENAME = "summary.yaml"


def _generation_summary_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's summary path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's summary YAML file.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_SUMMARY_FILENAME


def _generation_winner_path(experiment: Experiment, generation_id: str, phase_name: str) -> Path:
    """Return one generation's phase-winner path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :param str phase_name: Phase name whose generation-scoped winner path is requested.
    :return Path: Path to the phase's winner YAML file within the generation namespace.
    """
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / "winner.yaml"


def _generation_promotion_decision_path(
    experiment: Experiment, generation_id: str, phase_name: str
) -> Path:
    """Return one generation's phase-promotion path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :param str phase_name: Phase name whose generation-scoped promotion path is requested.
    :return Path: Path to the phase's promotion-decision YAML file within the
        generation namespace.
    """
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / "promotion.yaml"


def _last_successful_generation_path(experiment: Experiment) -> Path:
    """Return the pointer to the last fully published generation.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the YAML file recording the last-successful generation id.
    """
    return _experiment_dir(experiment) / "last_successful_generation.yaml"


def _generation_config_snapshot_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's canonical config snapshot path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's owner-only ``config.snapshot.yaml``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_CONFIG_SNAPSHOT_FILENAME


def _generation_reproducibility_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's shareable reproducibility-record path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's ``reproducibility.json``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_REPRODUCIBILITY_FILENAME


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


def _suite_generation_path(suite: Suite) -> Path:
    """Return the current suite-generation lifecycle path.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the current suite-generation lifecycle YAML file.
    """
    return _suite_dir(suite) / "suite_generation.yaml"


def _suite_generations_dir(suite: Suite) -> Path:
    """Return the immutable suite-generation root.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Directory containing all immutable per-suite-generation namespaces.
    """
    return _suite_dir(suite) / "suite_generations"


def _suite_generation_dir(suite: Suite, generation_id: str) -> Path:
    """Return one immutable suite-generation directory.

    :param Suite suite: Suite config with artifact root details.
    :param str generation_id: Immutable suite-generation namespace identifier.
    :return Path: Directory scoped to the given suite generation.
    """
    return _suite_generations_dir(suite) / generation_id


def _suite_generation_record_path(suite: Suite, generation_id: str) -> Path:
    """Return one suite generation's lifecycle record path.

    :param Suite suite: Suite config with artifact root details.
    :param str generation_id: Immutable suite-generation namespace identifier.
    :return Path: Path to the suite generation's lifecycle record YAML file.
    """
    return _suite_generation_dir(suite, generation_id) / "generation.yaml"


def _suite_generation_summary_path(suite: Suite, generation_id: str) -> Path:
    """Return one suite generation's immutable summary path.

    :param Suite suite: Suite config with artifact root details.
    :param str generation_id: Immutable suite-generation namespace identifier.
    :return Path: Path to the suite generation's summary YAML file.
    """
    return _suite_generation_dir(suite, generation_id) / "summary.yaml"


def _last_successful_suite_generation_path(suite: Suite) -> Path:
    """Return the pointer to the last fully published suite generation.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the YAML file recording the last-successful suite generation id.
    """
    return _suite_dir(suite) / "last_successful_suite_generation.yaml"


def _suite_log_path(suite: Suite) -> Path:
    """Path to a suite-level run log.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the suite run log.
    """
    return _suite_dir(suite) / "run.log"
