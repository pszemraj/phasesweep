"""Which metric-adjacent config fields enter the phase and experiment fingerprints."""

from __future__ import annotations

from phasesweep.config import LogRegexExtractor, Metric
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint, _phase_fingerprint
from tests.conftest import make_experiment


def _metric(**bounds: float) -> Metric:
    """Return a log_regex ``x=<value>`` metric with the given viability bounds."""
    return Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        **bounds,
    )


def test_metric_bounds_are_semantic() -> None:
    """Selection reads the bounds, so moving or adding one changes both fingerprints."""
    variants = {
        "unbounded": make_experiment(metric=_metric()),
        "max": make_experiment(metric=_metric(max=2.0)),
        "max_moved": make_experiment(metric=_metric(max=3.0)),
        "min": make_experiment(metric=_metric(min=0.5)),
        "both": make_experiment(metric=_metric(min=0.5, max=2.0)),
    }

    phase_fps = {n: _phase_fingerprint(e, e.phases[0], {}) for n, e in variants.items()}
    experiment_fps = {n: _experiment_semantic_fingerprint(e) for n, e in variants.items()}

    assert len(set(phase_fps.values())) == len(variants)
    assert len(set(experiment_fps.values())) == len(variants)
