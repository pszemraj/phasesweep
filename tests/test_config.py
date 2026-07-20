from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import JsonExtractor, Metric, Phase
from tests.conftest import write_yaml


def test_inherit_must_be_prior(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {trial_dir}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - name: a
    inherits: [b]
    n_trials: 1
    search_space: { x: { type: float, low: 0.0, high: 1.0 } }
  - name: b
    n_trials: 1
    search_space: { y: { type: float, low: 0.0, high: 1.0 } }
"""
    )
    with pytest.raises(ValueError, match="inherits from 'b'"):
        load_experiment(cfg)


def test_constraint_requires_bound(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
constraints:
  - name: bytes
    extractor: { type: json, path: r.json, key: bytes }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
"""
    )
    with pytest.raises(ValueError, match="must define at least one"):
        load_experiment(cfg)


def test_metric_constraint_name_collision(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
constraints:
  - name: loss
    max: 1
    extractor: { type: json, path: r.json, key: bytes }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
"""
    )
    with pytest.raises(ValueError, match="distinct"):
        load_experiment(cfg)


@pytest.mark.parametrize("name", ["bad name with spaces", "bad\n"])
def test_phase_name_validation(name: str) -> None:
    with pytest.raises(ValidationError):
        Phase(name=name, n_trials=1, search_space={})


# ---- migrated from version-named files ----


@pytest.mark.parametrize(
    "body",
    [
        """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "first {overrides}"
trial_command: "second {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
""",
        """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - name: a
    n_trials: 1
    search_space:
      lr: {type: float, low: 1e-5, high: 1e-3, log: true}
      lr: {type: float, low: 1e-4, high: 1e-2, log: true}
""",
        """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - name: a
    n_trials: 1
    n_trials: 5
    search_space: { x: { type: float, low: 0, high: 1 } }
""",
    ],
    ids=["top_level", "search_space", "phase_mapping"],
)
def test_duplicate_yaml_keys_rejected(tmp_path: Path, body: str) -> None:
    """Duplicate keys fail loudly anywhere in the config tree."""
    with pytest.raises(ValueError, match="duplicate key"):
        load_experiment(write_yaml(tmp_path, body))


def test_yaml_merge_keys_allow_explicit_overrides(tmp_path: Path) -> None:
    """Explicit keys may override values inherited through a YAML merge key."""
    body = """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - &phase_defaults
    name: baseline
    n_trials: 1
  - <<: *phase_defaults
    name: tuned
    n_trials: 2
"""
    exp = load_experiment(write_yaml(tmp_path, body))

    assert [(phase.name, phase.n_trials) for phase in exp.phases] == [
        ("baseline", 1),
        ("tuned", 2),
    ]


def test_n_jobs_default_is_one(tmp_path):
    body = """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
"""
    exp = load_experiment(write_yaml(tmp_path, body))
    assert exp.override_format == "argparse"
    assert exp.phases[0].n_jobs == 1
    assert exp.phases[0].max_consecutive_failures == 5


def test_plain_json_extractor_is_not_a_primary_objective() -> None:
    with pytest.raises(ValidationError, match="json_envelope"):
        Metric(extractor=JsonExtractor(type="json", path="result.json", key="loss"))
