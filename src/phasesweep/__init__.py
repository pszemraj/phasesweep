"""phasesweep: YAML-driven phase-chained hyperparameter sweeps."""

# Define ``__version__`` BEFORE importing submodules so they can do
# ``from phasesweep import __version__`` during their own import cycle without
# tripping the partial-module lookup pattern. Source of truth is the static
# ``version`` field in ``pyproject.toml``; ``importlib.metadata`` reads it
# back from the installed package metadata. The ``PackageNotFoundError``
# fallback covers in-source-tree use without ``pip install -e .``.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("phasesweep")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

# These imports MUST follow the ``__version__`` assignment above. Both
# orchestrator and runner reach back into this module for ``__version__`` at
# import time; moving them up would break that chain. ``noqa: E402`` /
# ``isort: split`` keep formatters from "fixing" the order.
from phasesweep.config import Experiment, load_experiment  # noqa: E402
from phasesweep.orchestrator import (  # noqa: E402
    NoFeasibleTrialError,
    Winner,
    run_experiment,
)
from phasesweep.runner import UnsafeProcessCleanupError  # noqa: E402

__all__ = [
    "Experiment",
    "NoFeasibleTrialError",
    "UnsafeProcessCleanupError",
    "Winner",
    "__version__",
    "load_experiment",
    "run_experiment",
]
