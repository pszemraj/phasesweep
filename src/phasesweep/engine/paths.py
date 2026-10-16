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
    """Resolve the workdir while leaving the experiment leaf unresolved.

    :return: Absolute ``<workdir>/<experiment>`` directory.
    """
    return Path(experiment.workdir).expanduser().resolve() / experiment.experiment


def _artifact_root_identity(experiment: Experiment) -> str:
    """Return the artifact namespace resolved through any symlinked experiment leaf.

    :return: Canonical ``<workdir>/<experiment>`` identity.
    """
    return str(_experiment_dir(experiment).resolve())


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Return the artifact namespace for one phase.

    :param experiment: Supplies the artifact root.
    :param phase_name: Phase directory name.
    :return: ``<experiment>/<phase_name>``.
    """
    return _experiment_dir(experiment) / phase_name


def _summary_path(experiment: Experiment) -> Path:
    """Return the experiment summary path.

    :return: Experiment-root ``summary.yaml`` convenience copy.
    """
    return _experiment_dir(experiment) / "summary.yaml"


def _run_log_path(experiment: Experiment) -> Path:
    """Path to the durable run log for one experiment.

    :return: Experiment-root ``run.log``.
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

    :param experiment: Supplies the artifact root.
    :param phase_name: Owning phase directory.
    :param trial_number: Optuna trial number, padded to five digits.
    :param generation_id: Engine invocation id; requires ``attempt_id``.
    :param attempt_id: Subprocess attempt id; requires ``generation_id``.
    :return: Phase child named for the trial and any supplied execution ids.
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

    :return: Experiment-root ``attempts`` directory.
    """
    return _experiment_dir(experiment) / "attempts"


def _artifact_root_binding_path(experiment: Experiment) -> Path:
    """Return the reverse artifact-root-to-storage ownership record.

    :return: Experiment-root ``artifact_root_binding.json``.
    """
    return _experiment_dir(experiment) / "artifact_root_binding.json"


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation metadata path.

    :return: Experiment-root ``generation.yaml`` pointer.
    """
    return _experiment_dir(experiment) / "generation.yaml"


def _generations_dir(experiment: Experiment) -> Path:
    """Return the root of the experiment's generation namespaces.

    :return: Experiment-root ``generations`` directory.
    """
    return _experiment_dir(experiment) / "generations"


def _generation_dir(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's artifact namespace.

    :param experiment: Supplies the artifact root.
    :param generation_id: Child of the ``generations`` directory.
    :return: ``<experiment>/generations/<generation_id>``.
    """
    return _generations_dir(experiment) / generation_id


def _generation_record_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's lifecycle record path.

    :param experiment: Supplies the artifact root.
    :param generation_id: Generation namespace to inspect.
    :return: Generation-scoped lifecycle ``generation.yaml``.
    """
    return _generation_dir(experiment, generation_id) / "generation.yaml"


GENERATION_SUMMARY_FILENAME = "summary.yaml"


def _generation_summary_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's summary path.

    :param experiment: Supplies the artifact root.
    :param generation_id: Generation namespace to inspect.
    :return: Generation-scoped ``summary.yaml``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_SUMMARY_FILENAME


def _generation_winner_path(experiment: Experiment, generation_id: str, phase_name: str) -> Path:
    """Return one generation's phase-winner path.

    :param experiment: Supplies the artifact root.
    :param generation_id: Generation namespace to inspect.
    :param phase_name: Child of the generation's ``phases`` directory.
    :return: Generation-scoped ``phases/<phase_name>/winner.yaml``.
    """
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / WINNER_FILENAME


def _last_successful_generation_path(experiment: Experiment) -> Path:
    """Return the pointer to the last fully published generation.

    :return: Experiment-root ``last_successful_generation.yaml`` pointer.
    """
    return _experiment_dir(experiment) / "last_successful_generation.yaml"


def _generation_config_snapshot_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's canonical config snapshot path.

    :param experiment: Supplies the artifact root.
    :param generation_id: Generation namespace to inspect.
    :return: Generation-scoped ``config.snapshot.yaml``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_CONFIG_SNAPSHOT_FILENAME


def _generation_reproducibility_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's shareable reproducibility-record path.

    :param experiment: Supplies the artifact root.
    :param generation_id: Generation namespace to inspect.
    :return: Generation-scoped ``reproducibility.json``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_REPRODUCIBILITY_FILENAME


def _winner_path(experiment: Experiment, phase_name: str) -> Path:
    """Return the path to a phase's persisted winner.

    :param experiment: Supplies the artifact root.
    :param phase_name: Phase directory name.
    :return: Phase-root ``winner.yaml`` convenience copy.
    """
    return _phase_dir(experiment, phase_name) / WINNER_FILENAME
