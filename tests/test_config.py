from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import (
    Phase,
)
from tests.conftest import write_yaml


def test_load_example(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
experiment: t
storage: sqlite:///./db.sqlite
workdir: ./runs
trial_command: "echo {trial_dir} {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
phases:
  - name: a
    n_trials: 2
    search_space: { x: { type: float, low: 0.0, high: 1.0 } }
"""
    )
    exp = load_experiment(cfg)
    assert exp.phases[0].name == "a"


def test_inherit_must_be_prior(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        """
experiment: t
storage: ":memory:"
trial_command: "echo {trial_dir}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
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
trial_command: "echo"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
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
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
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


def test_phase_name_validation():
    with pytest.raises(ValidationError):
        Phase(name="bad name with spaces", n_trials=1, search_space={})


# ---- migrated from version-named files ----


def test_duplicate_top_level_key_rejected(tmp_path: Path) -> None:
    """Two ``trial_command`` entries → load fails. Pre-v0.5.7 this silently
    kept the second value with no warning.
    """
    body = """
experiment: t
storage: ":memory:"
trial_command: "first {overrides}"
trial_command: "second {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
"""
    with pytest.raises(ValueError, match="duplicate key"):
        load_experiment(write_yaml(tmp_path, body))


def test_duplicate_search_space_key_rejected(tmp_path: Path) -> None:
    """Two ``lr`` entries in a search space → load fails. This is the
    reviewer's example: silent overwrite of a sweep range.
    """
    body = """
experiment: t
storage: ":memory:"
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
phases:
  - name: a
    n_trials: 1
    search_space:
      lr: {type: float, low: 1e-5, high: 1e-3, log: true}
      lr: {type: float, low: 1e-4, high: 1e-2, log: true}
"""
    with pytest.raises(ValueError, match="duplicate key"):
        load_experiment(write_yaml(tmp_path, body))


def test_duplicate_phase_keys_within_one_phase_rejected(tmp_path: Path) -> None:
    body = """
experiment: t
storage: ":memory:"
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
phases:
  - name: a
    n_trials: 1
    n_trials: 5
    search_space: { x: { type: float, low: 0, high: 1 } }
"""
    with pytest.raises(ValueError, match="duplicate key"):
        load_experiment(write_yaml(tmp_path, body))


def test_n_jobs_default_is_one(tmp_path):
    body = """
experiment: t
storage: ":memory:"
trial_command: "echo {overrides}"
metric:
  name: loss
  goal: minimize
  extractor: { type: json, path: r.json, key: loss }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
"""
    exp = load_experiment(write_yaml(tmp_path, body))
    assert exp.phases[0].n_jobs == 1
    assert exp.phases[0].max_consecutive_failures == 5
