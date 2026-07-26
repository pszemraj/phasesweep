from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasesweep import load_config, load_experiment
from phasesweep.config import (
    ExecutionContext,
    Experiment,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Suite,
)
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


@pytest.mark.parametrize(
    "provenance",
    [
        {"": "trainer-v1"},
        {" ": "trainer-v1"},
        {"revision": ""},
        {"revision": " "},
    ],
)
def test_provenance_requires_nonempty_keys_and_values(provenance: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="provenance keys and values must be nonempty"):
        Experiment(
            experiment="invalid_provenance",
            trial_command="echo",
            provenance=provenance,
            metric=Metric(
                extractor=LogRegexExtractor(
                    type="log_regex",
                    pattern=r"x=(?P<value>[0-9.]+)",
                )
            ),
            phases=[Phase(name="p", n_trials=1)],
        )


@pytest.mark.parametrize(
    ("inherit_env", "message"),
    [
        ([""], "nonempty and unpadded"),
        ([" PATH"], "nonempty and unpadded"),
        (["PATH "], "nonempty and unpadded"),
        (["WANDB_API_KEY", "WANDB_API_KEY"], "must be unique"),
    ],
    ids=["empty", "leading_space", "trailing_space", "duplicate"],
)
def test_execution_inherit_env_names_validated(inherit_env: list[str], message: str) -> None:
    """A named-inheritance list must be a set of exact variable names.

    Padded or empty names would silently inherit nothing (``os.environ`` has
    no ``' PATH'``), and duplicates hide a typo in a second entry.
    """
    with pytest.raises(ValidationError, match=message):
        ExecutionContext(inherit_env=inherit_env)


def test_execution_accepts_named_inherit_list() -> None:
    execution = ExecutionContext(inherit_env=["WANDB_API_KEY", "HF_TOKEN"])

    assert execution.inherit_env == ["WANDB_API_KEY", "HF_TOKEN"]
    assert execution.cwd is None
    assert ExecutionContext().inherit_env == "all"


def test_suite_execution_is_inherited_or_replaced_wholesale(tmp_path: Path) -> None:
    """``defaults.execution`` reaches studies that omit it; a study that
    declares its own block replaces the default *wholesale* rather than
    merging per key, and an explicit null resets to the built-in default
    (see ``Suite.experiment_for_study``).
    """
    config = load_config(
        write_yaml(
            tmp_path,
            """
            suite: execution_suite
            defaults:
              trial_command: "echo"
              metric:
                name: x
                goal: minimize
                extractor: {type: log_regex, pattern: 'x=(?P<value>[0-9.]+)'}
              execution:
                inherit_env: none
            studies:
              - name: inherited
                phases: [{name: p, n_trials: 1}]
              - name: replaced_cwd
                execution: {cwd: /srv/trainer}
                phases: [{name: p, n_trials: 1}]
              - name: replaced_names
                execution: {inherit_env: [WANDB_API_KEY, HF_TOKEN]}
                phases: [{name: p, n_trials: 1}]
              - name: reset
                execution: null
                phases: [{name: p, n_trials: 1}]
            """,
        )
    )

    assert isinstance(config, Suite)
    inherited, replaced_cwd, replaced_names, reset = (
        config.experiment_for_study(study) for study in config.studies
    )

    assert inherited.execution == ExecutionContext(inherit_env="none")

    # Wholesale replacement: the study's block does not keep the suite's
    # inherit_env: none — the unset field falls back to the field default.
    assert replaced_cwd.execution == ExecutionContext(cwd="/srv/trainer")
    assert replaced_cwd.execution.inherit_env == "all"
    assert replaced_names.execution == ExecutionContext(inherit_env=["WANDB_API_KEY", "HF_TOKEN"])
    assert replaced_names.execution.cwd is None

    # Explicit null is not "inherit the default block" — it is a reset.
    assert reset.execution == ExecutionContext()
    assert reset.execution.inherit_env == "all"


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


def test_duplicate_yaml_merge_keys_rejected(tmp_path: Path) -> None:
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
  - &first
    name: first
    n_trials: 1
  - &second
    name: second
    n_trials: 2
  - <<: *first
    <<: *second
    name: merged
"""

    with pytest.raises(ValueError, match=r"duplicate key '<<'"):
        load_experiment(write_yaml(tmp_path, body))


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
