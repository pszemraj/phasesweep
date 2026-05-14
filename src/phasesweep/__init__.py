"""phasesweep: YAML-driven phase-chained hyperparameter sweeps."""

# Define ``__version__`` BEFORE importing submodules. Source of truth is the
# static ``version`` field in ``pyproject.toml``; ``importlib.metadata`` reads
# it back from installed package metadata.
from phasesweep._metadata import __version__

# Public API exports follow the ``__version__`` assignment by design. Keep the
# order stable so metadata is available before importing heavier subpackages.
from phasesweep.config import Config, Experiment, Suite, load_config, load_experiment  # noqa: E402
from phasesweep.engine import (  # noqa: E402
    NoFeasibleTrialError,
    UnsafeProcessCleanupError,
    Winner,
    config_status,
    run_config,
    run_experiment,
    run_suite,
)

__all__ = [
    "Experiment",
    "NoFeasibleTrialError",
    "Config",
    "Suite",
    "UnsafeProcessCleanupError",
    "Winner",
    "__version__",
    "config_status",
    "load_config",
    "load_experiment",
    "run_config",
    "run_experiment",
    "run_suite",
]
