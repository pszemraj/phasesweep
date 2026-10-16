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
    """Resolve the workdir, leaving the experiment leaf unresolved."""
    return Path(experiment.workdir).expanduser().resolve() / experiment.experiment


def _artifact_root_identity(experiment: Experiment) -> str:
    """Resolve the artifact namespace through any symlinked experiment leaf."""
    return str(_experiment_dir(experiment).resolve())


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Return the phase directory under the experiment artifact root."""
    return _experiment_dir(experiment) / phase_name


def _summary_path(experiment: Experiment) -> Path:
    """Return the experiment-root summary.yaml convenience copy path."""
    return _experiment_dir(experiment) / "summary.yaml"


def _run_log_path(experiment: Experiment) -> Path:
    """Return the experiment-root run.log path."""
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

    Supplying only one execution ID raises ValueError.
    """
    if generation_id is None and attempt_id is None:
        return _phase_dir(experiment, phase_name) / f"trial_{trial_number:05d}"
    if generation_id is None or attempt_id is None:
        raise ValueError("generation_id and attempt_id must be supplied together")
    return _phase_dir(experiment, phase_name) / (
        f"trial_{trial_number:05d}__generation_{generation_id}__attempt_{attempt_id}"
    )


def _attempts_dir(experiment: Experiment) -> Path:
    """Return the trial and tracking attempt registry directory."""
    return _experiment_dir(experiment) / "attempts"


def _artifact_root_binding_path(experiment: Experiment) -> Path:
    """Return the artifact-root-to-storage ownership record path."""
    return _experiment_dir(experiment) / "artifact_root_binding.json"


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation pointer path."""
    return _experiment_dir(experiment) / "generation.yaml"


def _generations_dir(experiment: Experiment) -> Path:
    """Return the root of the experiment's generation namespaces."""
    return _experiment_dir(experiment) / "generations"


def _generation_dir(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's artifact namespace."""
    return _generations_dir(experiment) / generation_id


def _generation_record_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's lifecycle generation.yaml path."""
    return _generation_dir(experiment, generation_id) / "generation.yaml"


GENERATION_SUMMARY_FILENAME = "summary.yaml"


def _generation_summary_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's summary.yaml path."""
    return _generation_dir(experiment, generation_id) / GENERATION_SUMMARY_FILENAME


def _generation_winner_path(experiment: Experiment, generation_id: str, phase_name: str) -> Path:
    """Return the generation-scoped phases/<phase_name>/winner.yaml path."""
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / WINNER_FILENAME


def _last_successful_generation_path(experiment: Experiment) -> Path:
    """Return the last fully published generation pointer path."""
    return _experiment_dir(experiment) / "last_successful_generation.yaml"


def _generation_config_snapshot_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's canonical config snapshot path."""
    return _generation_dir(experiment, generation_id) / GENERATION_CONFIG_SNAPSHOT_FILENAME


def _generation_reproducibility_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's shareable reproducibility record path."""
    return _generation_dir(experiment, generation_id) / GENERATION_REPRODUCIBILITY_FILENAME


def _winner_path(experiment: Experiment, phase_name: str) -> Path:
    """Return the phase-root winner.yaml convenience copy path."""
    return _phase_dir(experiment, phase_name) / WINNER_FILENAME
