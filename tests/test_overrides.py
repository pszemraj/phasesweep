from __future__ import annotations

import shlex

import pytest
from pydantic import ValidationError

from phasesweep import load_experiment, run_experiment
from phasesweep.runtime.commands import format_argparse, format_hydra, render_command
from tests.conftest import copy_fake_train, write_yaml


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"n_layers": 8, "lr": 3e-4}, "n_layers=8 lr=0.0003", id="basic"),
        pytest.param({"model.n_layers": 12}, "model.n_layers=12", id="dotted-key"),
        pytest.param({"flag": True, "off": False}, "flag=true off=false", id="booleans"),
    ],
)
def test_hydra_scalar_rendering(overrides: dict[str, object], expected: str) -> None:
    assert format_hydra(overrides) == expected


def test_hydra_quotes_string_values_for_hydra_grammar():
    s = format_hydra({"optimizer": "adam,w", "tags": ["a,b", "c[d]"], "mode": "true"})

    assert shlex.split(s) == [
        'optimizer="adam,w"',
        'tags=["a,b","c[d]"]',
        'mode="true"',
    ]


def test_hydra_rejects_structured_values():
    with pytest.raises(TypeError, match="json_file"):
        format_hydra({"model": {"depth": 2}})


def test_argparse():
    s = format_argparse({"lr": 3e-4, "weight_decay": 0.05})
    assert s == "--lr 0.0003 --weight_decay 0.05"


def test_render_command_hydra(tmp_path):
    cmd = render_command(
        "python train.py {overrides} --out {trial_dir}/r.json",
        {"n_layers": 8},
        "hydra",
        trial_dir=tmp_path,
        trial_id=3,
        phase="depth",
        run_name="x-depth-3",
    )
    assert "n_layers=8" in cmd
    assert str(tmp_path) in cmd


def test_render_command_json_file(tmp_path):
    cmd = render_command(
        "python train.py --overrides-path {overrides_path}",
        {"a.b": 1, "a.c": 2, "d": "x"},
        "json_file",
        trial_dir=tmp_path,
        trial_id=0,
        phase="p",
        run_name="r",
    )
    assert "overrides.json" in cmd
    import json

    data = json.loads((tmp_path / "overrides.json").read_text())
    assert data == {"a": {"b": 1, "c": 2}, "d": "x"}


def test_validate_rejects_structured_hydra_fixed_override(tmp_path):
    p = write_yaml(
        tmp_path,
        """
        experiment: t
        trial_command: "echo {overrides}"
        override_format: hydra
        metric:
          name: x
          goal: minimize
          extractor: { type: json_envelope, objective_name: x, split: test, policy: test }
        phases:
          - name: p
            n_trials: 1
            fixed_overrides:
              model: { depth: 2 }
        """,
    )

    with pytest.raises(ValidationError, match="override_format='hydra'.*json_file"):
        load_experiment(p)


# ---- migrated from version-named files ----


def test_effective_overrides_include_fixed(tmp_path):
    """Winner's effective_overrides must include parent's fixed_overrides, not just sampled params."""
    trainer = copy_fake_train(tmp_path)

    db_path = tmp_path / "phases.db"
    yaml_text = f"""
experiment: eff_override_test
storage: sqlite:///{db_path}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} --out {{trial_dir}}/result.json {{overrides}}"
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: arch
    fixed_overrides:
      model_family: llama
    n_trials: 2
    sampler: {{ type: grid }}
    search_space:
      n_layers: {{ type: categorical, choices: [4, 8] }}
  - name: opt
    inherits: [arch]
    n_trials: 4
    sampler: {{ type: tpe, seed: 0 }}
    search_space:
      lr: {{ type: float, low: 1e-5, high: 1e-2, log: true }}
"""
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(yaml_text)
    exp = load_experiment(yaml_path)
    winners = run_experiment(exp)

    # The opt phase winner should have model_family in effective_overrides
    opt_winner = winners["opt"]
    assert "model_family" in opt_winner.effective_overrides
    assert opt_winner.effective_overrides["model_family"] == "llama"
    # And also the inherited n_layers
    assert "n_layers" in opt_winner.effective_overrides


def test_transitive_inherited_search_key_cannot_be_resampled(tmp_path):
    """A grandchild may not re-sample a key locked two levels up."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: arch
            n_trials: 1
            search_space:
              n_layers: {{ type: categorical, choices: [4, 8] }}
          - name: lr
            inherits: [arch]
            n_trials: 1
            search_space:
              lr: {{ type: float, low: 1e-5, high: 1e-3, log: true }}
          - name: reg
            inherits: [lr]
            n_trials: 1
            search_space:
              n_layers: {{ type: categorical, choices: [12, 16] }}
        """,
    )
    with pytest.raises(ValidationError, match="re-samples key"):
        load_experiment(p)


def test_multi_parent_collision_unresolved_errors(tmp_path):
    """Two independent parents both lock 'lr'; child must resolve via fixed_overrides."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: a
            n_trials: 1
            search_space:
              lr: {{ type: float, low: 1e-5, high: 1e-3, log: true }}
          - name: b
            n_trials: 1
            search_space:
              lr: {{ type: float, low: 1e-5, high: 1e-3, log: true }}
          - name: c
            inherits: [a, b]
            n_trials: 1
            search_space:
              dropout: {{ type: float, low: 0, high: 0.5 }}
        """,
    )
    with pytest.raises(ValidationError, match="conflicting locked key"):
        load_experiment(p)


def test_multi_parent_collision_resolved_by_fixed_override(tmp_path):
    """Same conflict but child explicitly resolves with fixed_overrides — accepted."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: a
            n_trials: 1
            search_space:
              lr: {{ type: float, low: 1e-5, high: 1e-3, log: true }}
          - name: b
            n_trials: 1
            search_space:
              lr: {{ type: float, low: 1e-5, high: 1e-3, log: true }}
          - name: c
            inherits: [a, b]
            fixed_overrides:
              lr: 5.0e-4
            n_trials: 1
            search_space:
              dropout: {{ type: float, low: 0, high: 0.5 }}
        """,
    )
    exp = load_experiment(p)  # must not raise
    assert exp.phases[-1].fixed_overrides["lr"] == 5.0e-4
