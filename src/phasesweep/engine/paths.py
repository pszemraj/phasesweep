"""Pure filesystem paths for engine artifacts."""

from __future__ import annotations

from pathlib import Path

from phasesweep.config.models import Experiment
from phasesweep.engine.state import (
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_REPRODUCIBILITY_FILENAME,
    WINNER_FILENAME,
)


def _experiment_dir(experiment: Experiment) -> Path:
    """Return the artifact namespace for one experiment.

    :return Path: Absolute directory for experiment artifacts.
    """
    return Path(experiment.workdir).expanduser().resolve() / experiment.experiment


def _artifact_root_identity(experiment: Experiment) -> str:
    """Return the artifact namespace resolved through any symlinked experiment leaf.

    :return str: Resolved ``<workdir>/<experiment>`` namespace as a string.
    """
    return str(_experiment_dir(experiment).resolve())


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Return the artifact namespace for one phase.

    :param Experiment experiment: Experiment config.
    :param str phase_name: Phase name.
    :return Path: Directory for phase artifacts.
    """
    return _experiment_dir(experiment) / phase_name


def _summary_path(experiment: Experiment) -> Path:
    """Return the experiment summary path.

    :return Path: Experiment ``summary.yaml``.
    """
    return _experiment_dir(experiment) / "summary.yaml"


def _run_log_path(experiment: Experiment) -> Path:
    """Path to the durable run log for one experiment.

    :return Path: Experiment ``run.log``.
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

    :param Experiment experiment: Experiment config.
    :param str phase_name: Owning phase.
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
    """Return the trial and tracking attempt registry directory.

    :return Path: Directory containing active-attempt registry entries.
    """
    return _experiment_dir(experiment) / "attempts"


def _artifact_root_binding_path(experiment: Experiment) -> Path:
    """Return the reverse artifact-root-to-storage ownership record.

    :return Path: Experiment ``artifact_root_binding.json``.
    """
    return _experiment_dir(experiment) / "artifact_root_binding.json"


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation metadata path.

    :return Path: Current ``generation.yaml`` pointer.
    """
    return _experiment_dir(experiment) / "generation.yaml"


def _generations_dir(experiment: Experiment) -> Path:
    """Return the immutable generation-record root for an experiment.

    :return Path: Directory containing all immutable per-generation namespaces.
    """
    return _experiment_dir(experiment) / "generations"


def _generation_dir(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's immutable artifact namespace.

    :param Experiment experiment: Experiment config.
    :param str generation_id: Generation id.
    :return Path: Directory scoped to the given generation.
    """
    return _generations_dir(experiment) / generation_id


def _generation_record_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's lifecycle record path.

    :param Experiment experiment: Experiment config.
    :param str generation_id: Generation id.
    :return Path: Generation lifecycle ``generation.yaml``.
    """
    return _generation_dir(experiment, generation_id) / "generation.yaml"


GENERATION_SUMMARY_FILENAME = "summary.yaml"


def _generation_summary_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's summary path.

    :param Experiment experiment: Experiment config.
    :param str generation_id: Generation id.
    :return Path: Generation ``summary.yaml``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_SUMMARY_FILENAME


def _generation_winner_path(experiment: Experiment, generation_id: str, phase_name: str) -> Path:
    """Return one generation's phase-winner path.

    :param Experiment experiment: Experiment config.
    :param str generation_id: Generation id.
    :param str phase_name: Owning phase.
    :return Path: Generation-scoped phase winner YAML file.
    """
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / WINNER_FILENAME


def _last_successful_generation_path(experiment: Experiment) -> Path:
    """Return the pointer to the last fully published generation.

    :return Path: ``last_successful_generation.yaml`` pointer.
    """
    return _experiment_dir(experiment) / "last_successful_generation.yaml"


def _generation_config_snapshot_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's canonical config snapshot path.

    :param Experiment experiment: Experiment config.
    :param str generation_id: Generation id.
    :return Path: Generation's owner-only ``config.snapshot.yaml``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_CONFIG_SNAPSHOT_FILENAME


def _generation_reproducibility_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's shareable reproducibility-record path.

    :param Experiment experiment: Experiment config.
    :param str generation_id: Generation id.
    :return Path: Generation ``reproducibility.json``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_REPRODUCIBILITY_FILENAME


def _winner_path(experiment: Experiment, phase_name: str) -> Path:
    """Return the path to a phase's persisted winner.

    :param Experiment experiment: Experiment config.
    :param str phase_name: Owning phase.
    :return Path: Persisted phase winner YAML file.
    """
    return _phase_dir(experiment, phase_name) / WINNER_FILENAME
