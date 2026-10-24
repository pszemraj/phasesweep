"""Pure filesystem paths for engine artifacts."""

from __future__ import annotations

from pathlib import Path

from phasesweep.config.models import Experiment
from phasesweep.engine.state import (
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_REPRODUCIBILITY_FILENAME,
    WINNER_FILENAME,
)
from phasesweep.runtime.files import artifact_root_path


def _experiment_dir(experiment: Experiment) -> Path:
    """Resolve the workdir, leaving the experiment leaf unresolved.

    :param Experiment experiment: Configuration naming the experiment directory.
    :return Path: Absolute experiment directory with any leaf symlink still unresolved.
    """
    return artifact_root_path(experiment.workdir, experiment.experiment)


def _artifact_root_identity(experiment: Experiment) -> str:
    """Resolve the artifact namespace through any symlinked experiment leaf.

    :param Experiment experiment: Configuration naming the artifact namespace.
    :return str: Fully resolved artifact-root path used for ownership comparisons.
    """
    return str(_experiment_dir(experiment).resolve())


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Return the phase directory under the experiment artifact root.

    :param Experiment experiment: Configuration naming the experiment directory.
    :param str phase_name: Phase whose live artifacts are addressed.
    :return Path: Phase directory under the experiment namespace.
    """
    return _experiment_dir(experiment) / phase_name


def _run_log_path(experiment: Experiment) -> Path:
    """Return the experiment-root run.log path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :return Path: Shared run log path for the experiment.
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
    """Return a trial directory, scoped by generation and attempt when both are supplied.

    :param Experiment experiment: Configuration naming the experiment directory.
    :param str phase_name: Phase containing the trial.
    :param int trial_number: Study-local trial number encoded in the directory name.
    :param str | None generation_id: Generation identity, supplied together with attempt_id.
    :param str | None attempt_id: Execution attempt identity, supplied together with generation_id.
    :raises ValueError: Exactly one execution identity is supplied.
    :return Path: Phase-local trial path, including both execution identities when supplied.
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

    :param Experiment experiment: Configuration naming the experiment directory.
    :return Path: Directory containing durable attempt registry records.
    """
    return _experiment_dir(experiment) / "attempts"


def _artifact_root_binding_path(experiment: Experiment) -> Path:
    """Return the artifact-root-to-storage ownership record path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :return Path: Artifact-root ownership record binding this tree to its storage.
    """
    return _experiment_dir(experiment) / "artifact_root_binding.json"


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation pointer path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :return Path: Experiment-root pointer to the current generation.
    """
    return _experiment_dir(experiment) / "generation.yaml"


def _generations_dir(artifact_root: Path) -> Path:
    """Return the root of the experiment's generation namespaces.

    :param Path artifact_root: Experiment artifact directory.
    :return Path: Parent directory of all generation artifact namespaces.
    """
    return artifact_root / "generations"


def _generation_dir(artifact_root: Path, generation_id: str) -> Path:
    """Return one generation's artifact namespace.

    :param Path artifact_root: Experiment artifact directory.
    :param str generation_id: Generation whose artifacts are addressed.
    :return Path: Artifact directory for that generation.
    """
    return _generations_dir(artifact_root) / generation_id


def _generation_record_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's lifecycle generation.yaml path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :param str generation_id: Generation whose lifecycle record is addressed.
    :return Path: Lifecycle record inside the generation namespace.
    """
    return _generation_dir(_experiment_dir(experiment), generation_id) / "generation.yaml"


GENERATION_SUMMARY_FILENAME = "summary.yaml"


def _generation_summary_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's summary.yaml path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :param str generation_id: Generation whose result summary is addressed.
    :return Path: Summary record inside the generation namespace.
    """
    return _generation_dir(_experiment_dir(experiment), generation_id) / GENERATION_SUMMARY_FILENAME


def _generation_winner_path(experiment: Experiment, generation_id: str, phase_name: str) -> Path:
    """Return the generation-scoped phases/<phase_name>/winner.yaml path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :param str generation_id: Generation containing the selected phase result.
    :param str phase_name: Phase whose winner record is addressed.
    :return Path: Phase winner record inside the generation namespace.
    """
    return (
        _generation_dir(_experiment_dir(experiment), generation_id)
        / "phases"
        / phase_name
        / WINNER_FILENAME
    )


def _last_successful_generation_path(experiment: Experiment) -> Path:
    """Return the last fully published generation pointer path.

    :param Experiment experiment: Configuration naming the experiment directory.
    :return Path: Experiment-root pointer to the last fully published generation.
    """
    return _experiment_dir(experiment) / "last_successful_generation.yaml"


def _generation_config_snapshot_path(artifact_root: Path, generation_id: str) -> Path:
    """Return one generation's saved configuration path.

    :param Path artifact_root: Experiment artifact directory.
    :param str generation_id: Generation whose launch configuration is addressed.
    :return Path: Configuration snapshot inside the generation namespace.
    """
    return _generation_dir(artifact_root, generation_id) / GENERATION_CONFIG_SNAPSHOT_FILENAME


def _generation_reproducibility_path(artifact_root: Path, generation_id: str) -> Path:
    """Return one generation's shareable reproducibility record path.

    :param Path artifact_root: Experiment artifact directory.
    :param str generation_id: Generation whose reproducibility record is addressed.
    :return Path: Reproducibility record inside the generation namespace.
    """
    return _generation_dir(artifact_root, generation_id) / GENERATION_REPRODUCIBILITY_FILENAME
