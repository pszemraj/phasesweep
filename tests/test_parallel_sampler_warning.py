"""Explicit reproducibility guidance for adaptive samplers with parallel workers."""

from __future__ import annotations

import pytest

from phasesweep.config.models import Phase
from phasesweep.config.search import FloatParam, Sampler, sampler_warnings


@pytest.mark.parametrize("sampler_type", ["tpe", "cmaes"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_adaptive_sampler_parallelism_warns_about_completion_order(
    sampler_type: str, n_jobs: int
) -> None:
    phase = Phase(
        name="adaptive",
        n_trials=20,
        n_jobs=n_jobs,
        sampler=Sampler(type=sampler_type, seed=7),
        search_space={"x": FloatParam(type="float", low=0.1, high=1.0)},
    )

    warnings = " ".join(sampler_warnings(phase)).lower()

    if n_jobs > 1:
        assert "completion order" in warnings
        assert "not reproducible" in warnings
        assert "n_jobs: 1" in warnings
    else:
        assert "completion order" not in warnings
