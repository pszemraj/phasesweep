from __future__ import annotations

import datetime
import shlex

import pytest
from pydantic import ValidationError

from phasesweep import load_config, load_experiment, run_experiment
from phasesweep.runtime.commands import (
    dump_overrides_json,
    format_argparse,
    format_hydra,
    render_command,
    write_json_file,
)
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_hydra_rejects_non_finite_values(value: float) -> None:
    """The renderer must not admit values the semantic JSON dump collapses."""
    with pytest.raises(TypeError, match="finite floats"):
        format_hydra({"x": [value]})


def test_argparse():
    s = format_argparse({"lr": 3e-4, "weight_decay": 0.05})
    assert s == "--lr 0.0003 --weight_decay 0.05"


def test_argparse_renders_the_documented_wire_forms():
    """The accepted value set renders exactly what config_reference.yaml promises."""
    s = format_argparse(
        {
            "none": None,
            "flag": True,
            "n": 3,
            "ratio": 2.5,
            "tag": "s",
            "items": [1, "a", False],
        }
    )

    assert shlex.split(s) == [
        "--none",
        "None",
        "--flag",
        "true",
        "--n",
        "3",
        "--ratio",
        "2.5",
        "--tag",
        "s",
        "--items",
        "[1,a,false]",
    ]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"depth": 2}, id="mapping"),
        pytest.param(datetime.date(2024, 1, 1), id="date"),
        pytest.param([1, {"depth": 2}], id="nested-mapping"),
        pytest.param(float("nan"), id="non-finite-float"),
    ],
)
def test_argparse_rendering_rejects_values_outside_the_wire_contract(value):
    """Defense in depth behind the config validator: str()-ing a mapping or a
    date into a command line is how two different commands end up sharing one
    fingerprint (PR #5 review / reviewer 2, blocker 3)."""
    with pytest.raises(TypeError, match="json_file"):
        format_argparse({"model": value})


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


def _override_yaml(tmp_path, override_format: str, body: str):
    """Write a minimal override config with caller-supplied phase/contract body."""
    trial_command = {
        "json_file": "python train.py --overrides {overrides_path}",
        "argparse": "python train.py {overrides}",
        "hydra": "python train.py {overrides}",
    }[override_format]
    return write_yaml(
        tmp_path,
        f"""
        experiment: t
        trial_command: "{trial_command}"
        override_format: {override_format}
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
{body}
        """,
    )


def test_write_json_file_uses_the_canonical_strict_serializer(tmp_path):
    """The wire artifact and the load-time check must share one encoder."""
    path = write_json_file({"a.b": 1, "c": "x"}, tmp_path)

    assert path.read_text() == dump_overrides_json({"a": {"b": 1}, "c": "x"})
    with pytest.raises(TypeError):
        dump_overrides_json({"cutoff": datetime.date(2024, 1, 1)})
    with pytest.raises(TypeError):
        write_json_file({"cutoff": datetime.date(2024, 1, 1)}, tmp_path)


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), float("nan")],
    ids=["inf", "-inf", "nan"],
)
def test_dump_overrides_json_and_write_json_file_reject_non_finite_floats(tmp_path, value):
    """allow_nan=False must reject values ``json.dumps`` would otherwise render as the
    non-standard Infinity/-Infinity/NaN tokens ``strict_json_loads`` refuses to parse
    (review v0.5.17 / finding B)."""
    with pytest.raises(ValueError):
        dump_overrides_json({"x": value})
    with pytest.raises(ValueError):
        write_json_file({"x": value}, tmp_path)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        pytest.param("2024-01-01", r"'knob'.*cannot encode.*type date", id="yaml-date"),
        pytest.param(".inf", r"'knob'.*cannot encode.*non-finite", id="non-finite-float"),
    ],
)
def test_validate_rejects_invalid_json_file_fixed_override(
    tmp_path, literal: str, expected: str
) -> None:
    """The canonical strict JSON serializer rejects unsupported fixed values at load time."""
    p = _override_yaml(
        tmp_path,
        "json_file",
        "        phases:\n"
        "          - name: p\n"
        "            n_trials: 1\n"
        "            fixed_overrides:\n"
        f"              knob: {literal}\n",
    )

    with pytest.raises(ValidationError, match=expected):
        load_experiment(p)


def test_validate_rejects_unserializable_json_file_contract_override(tmp_path):
    """Contract-supplied values are composed into the same artifact and checked too."""
    p = _override_yaml(
        tmp_path,
        "json_file",
        """
        contracts:
          frozen:
            fixed_overrides:
              cutoff: 2024-01-01
        phases:
          - name: p
            n_trials: 1
            contracts: [frozen]
        """,
    )

    with pytest.raises(ValidationError, match="contract 'frozen' fixed_overrides"):
        load_experiment(p)


# ``knob`` holds a value with no faithful argparse wire form in every case:
# a mapping renders through str() while the fingerprint dumps JSON-normalized
# keys, and a YAML-native date/non-finite float has no canonical rendering at
# all (PR #5 review / reviewer 2, blocker 3).
_UNRENDERABLE_ARGPARSE_VALUES = [
    pytest.param("{1: x}", r"Phase 't'.*'knob'.*type dict", id="int-keyed-mapping"),
    pytest.param('{1: x, "1": y}', r"Phase 't'.*'knob'.*type dict", id="mixed-key-mapping"),
    pytest.param("2024-01-01", r"Phase 't'.*'knob'.*type date", id="yaml-date"),
    pytest.param(".nan", r"Phase 't'.*'knob'.*type float", id="non-finite-float"),
    pytest.param(
        "[2024-01-01]",
        r"Phase 't'.*'knob'.*at position \[0\].*type date",
        id="nested-date-in-list",
    ),
]


@pytest.mark.parametrize(("value", "expected"), _UNRENDERABLE_ARGPARSE_VALUES)
def test_validate_rejects_unrenderable_argparse_fixed_override(tmp_path, value, expected):
    p = _override_yaml(
        tmp_path,
        "argparse",
        "        phases:\n"
        "          - name: t\n"
        "            n_trials: 1\n"
        "            fixed_overrides:\n"
        f"              knob: {value}\n",
    )

    with pytest.raises(ValidationError, match=expected):
        load_experiment(p)


def test_validate_rejects_unrenderable_argparse_contract_override(tmp_path):
    """Contract-supplied values compose into the same command line and are checked too."""
    p = _override_yaml(
        tmp_path,
        "argparse",
        "        contracts:\n"
        "          frozen:\n"
        "            fixed_overrides:\n"
        "              knob: {1: x}\n"
        "        phases:\n"
        "          - name: t\n"
        "            n_trials: 1\n"
        "            contracts: [frozen]\n",
    )

    with pytest.raises(ValidationError, match="contract 'frozen' fixed_overrides"):
        load_experiment(p)


@pytest.mark.parametrize("literal", [".nan", ".inf", "-.inf"])
@pytest.mark.parametrize("origin", ["phase", "contract"])
@pytest.mark.parametrize("nested", [False, True])
def test_validate_rejects_non_finite_hydra_fixed_override(
    tmp_path, literal: str, origin: str, nested: bool
) -> None:
    """Hydra wire values must remain distinguishable in semantic fingerprints."""
    value = f"[{literal}]" if nested else literal
    if origin == "phase":
        body = (
            "        phases:\n"
            "          - name: t\n"
            "            n_trials: 1\n"
            "            fixed_overrides:\n"
            f"              knob: {value}\n"
        )
        expected = r"fixed_overrides.*'knob'"
    else:
        body = (
            "        contracts:\n"
            "          frozen:\n"
            "            fixed_overrides:\n"
            f"              knob: {value}\n"
            "        phases:\n"
            "          - name: t\n"
            "            n_trials: 1\n"
            "            contracts: [frozen]\n"
        )
        expected = r"contract 'frozen' fixed_overrides.*'knob'"

    with pytest.raises(ValidationError, match=expected):
        load_experiment(_override_yaml(tmp_path, "hydra", body))


def test_validate_accepts_finite_hydra_fixed_override(tmp_path) -> None:
    config = _override_yaml(
        tmp_path,
        "hydra",
        "        phases:\n"
        "          - name: t\n"
        "            n_trials: 1\n"
        "            fixed_overrides:\n"
        "              knob: [1.5, -2.0]\n",
    )

    experiment = load_experiment(config)

    assert experiment.phases[0].fixed_overrides["knob"] == [1.5, -2.0]


def test_argparse_fixed_override_values_keep_distinct_phase_fingerprints(tmp_path):
    """With the value contract enforced, distinct Python values always produce
    distinct JSON-mode dumps — so no two configs that render different commands
    can share a study identity."""
    # Import only: the fingerprint code itself is deliberately untouched.
    from phasesweep.engine.guards import _phase_fingerprint

    fingerprints: dict[str, str] = {}
    for label, literal in {"int": "1", "str": '"1"', "bool": "true", "float": "1.0"}.items():
        p = _override_yaml(
            tmp_path,
            "argparse",
            "        phases:\n"
            "          - name: t\n"
            "            n_trials: 1\n"
            "            fixed_overrides:\n"
            f"              knob: {literal}\n",
        )
        exp = load_experiment(p)
        fingerprints[label] = _phase_fingerprint(exp, exp.phases[0], {})

    assert len(set(fingerprints.values())) == len(fingerprints)


def test_suite_argparse_study_rejects_a_shared_structured_contract_value(tmp_path):
    """A contract shared between a json_file study and an argparse study is only
    legal for the json_file one; the argparse study fails when it is compiled."""
    config = load_config(
        write_yaml(
            tmp_path,
            """
            suite: mixed_formats
            defaults:
              trial_command: "echo {overrides}"
              metric:
                name: x
                goal: minimize
                extractor: {type: log_regex, pattern: 'x=(?P<value>[0-9.]+)'}
              contracts:
                frozen:
                  fixed_overrides:
                    model: {depth: 2}
            studies:
              - name: structured
                override_format: json_file
                trial_command: "echo {overrides_path}"
                phases: [{name: p, n_trials: 1, contracts: [frozen]}]
              - name: flat
                override_format: argparse
                phases: [{name: p, n_trials: 1, contracts: [frozen]}]
            """,
        )
    )

    structured, flat = config.studies
    config.experiment_for_study(structured)
    with pytest.raises(ValidationError, match="override_format='argparse'.*type dict"):
        config.experiment_for_study(flat)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        pytest.param('"2024-01-01"', "2024-01-01", id="quoted-date"),
        pytest.param("{depth: 2}", {"depth": 2}, id="mapping"),
    ],
)
def test_validate_accepts_json_file_fixed_override(
    tmp_path, literal: str, expected: object
) -> None:
    """Values supported by strict JSON remain available only on the json_file wire."""
    p = _override_yaml(
        tmp_path,
        "json_file",
        "        phases:\n"
        "          - name: p\n"
        "            n_trials: 1\n"
        "            fixed_overrides:\n"
        f"              knob: {literal}\n",
    )

    exp = load_experiment(p)

    assert exp.phases[0].fixed_overrides["knob"] == expected


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
trial_command: "python {trainer} {{overrides}}"
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
    sampler: {{ type: tpe, seed: 0, acknowledge_nonresumable: true }}
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


@pytest.mark.parametrize(
    ("resolution", "valid"),
    [
        pytest.param("", False, id="unresolved"),
        pytest.param(
            "fixed_overrides:\n              lr: 5.0e-4\n            ", True, id="resolved"
        ),
    ],
)
def test_multi_parent_collision_requires_fixed_override(tmp_path, resolution: str, valid: bool):
    """A child must explicitly resolve a key locked by multiple parents."""
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
            {resolution}n_trials: 1
            search_space:
              dropout: {{ type: float, low: 0, high: 0.5 }}
        """,
    )
    if not valid:
        with pytest.raises(ValidationError, match="conflicting locked key"):
            load_experiment(p)
        return
    exp = load_experiment(p)
    child = exp.phases[-1]
    assert child.fixed_overrides["lr"] == 5.0e-4

    # Exercise the runtime merge too: inherited winners are the lowest layer,
    # the explicit fixed resolution replaces both, and sampled keys remain the
    # final layer.
    from phasesweep.engine.phase import _composed_overrides
    from phasesweep.engine.state import Winner

    inherited = {
        name: Winner(
            trial_number=index,
            params={"lr": value},
            effective_overrides={"lr": value},
            metric=value,
        )
        for index, (name, value) in enumerate((("a", 1.0e-4), ("b", 2.0e-4)))
    }
    assert _composed_overrides(exp, child, {"dropout": 0.25}, inherited) == {
        "lr": 5.0e-4,
        "dropout": 0.25,
    }
