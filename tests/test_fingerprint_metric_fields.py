"""Which selection-adjacent config fields enter the phase and experiment fingerprints."""

from __future__ import annotations

from phasesweep.config.models import Metric, Observation, Phase
from phasesweep.config.search import Repeats, Sampler
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint, _phase_fingerprint
from phasesweep.evidence.models import LogRegexExtractor
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


def test_observe_is_not_semantic() -> None:
    """Observations never affect selection or an evidence binding, so they never move identity."""

    def observation(name: str, pattern: str) -> Observation:
        return Observation(
            name=name, extractor=LogRegexExtractor(type="log_regex", pattern=pattern)
        )

    variants = [
        make_experiment(),
        make_experiment(observe=[observation("peak", r"p=(?P<value>[0-9.]+)")]),
        make_experiment(observe=[observation("peak", r"q=(?P<value>[0-9.]+)")]),
        make_experiment(
            observe=[
                observation("peak", r"p=(?P<value>[0-9.]+)"),
                observation("temp", r"t=(?P<value>[0-9.]+)"),
            ]
        ),
    ]

    assert len({_phase_fingerprint(e, e.phases[0], {}) for e in variants}) == 1
    assert len({_experiment_semantic_fingerprint(e) for e in variants}) == 1


def test_repeats_block_is_semantic() -> None:
    """Selection groups and reduces by the repeats block, so every field of it moves identity."""

    def phase(repeats: Repeats | None) -> Phase:
        return Phase(
            name="p",
            n_trials=6,
            sampler=Sampler(type="random", seed=0),
            search_space={"x": {"type": "int", "low": 0, "high": 9}},
            repeats=repeats,
        )

    variants = {
        "none": phase(None),
        "seeds": phase(Repeats(key="seed", values=[0, 1, 2])),
        "values": phase(Repeats(key="seed", values=[3, 4, 5])),
        "key": phase(Repeats(key="data.seed", values=[0, 1, 2])),
        "median": phase(Repeats(key="seed", values=[0, 1, 2], reduce="median")),
    }
    experiments = {name: make_experiment(phases=[p]) for name, p in variants.items()}

    phase_fps = {_phase_fingerprint(e, e.phases[0], {}) for e in experiments.values()}
    experiment_fps = {_experiment_semantic_fingerprint(e) for e in experiments.values()}

    assert len(phase_fps) == len(variants)
    assert len(experiment_fps) == len(variants)
