from __future__ import annotations

import argparse
import datetime
import json
import shlex
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phasesweep import load_experiment, run_experiment
from phasesweep.config import JsonEnvelopeExtractor, Metric, Phase
from phasesweep.runtime.commands import (
    TRAINER_INPUT_FILENAMES,
    compose_trainer_config,
    dump_json_file_overrides,
    dump_trainer_config_yaml,
    dump_trial_trainer_config_yaml,
    format_argparse,
    format_hydra,
    render_command,
)
from tests.conftest import (
    assert_invalid_experiment_yaml,
    copy_fake_train,
    make_experiment,
    write_yaml,
)


def test_argparse():
    s = format_argparse({"lr": 3e-4, "weight_decay": 0.05})
    assert s == "--lr=0.0003 --weight_decay=0.05"


@pytest.mark.parametrize("tag", ["--other-option", "-x", "two words", "a=b", "", "$(echo hi)"])
def test_argparse_round_trips_option_like_values(tag):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--offset", type=float)
    parser.add_argument("--tag")

    parsed = parser.parse_args(shlex.split(format_argparse({"offset": -1e-5, "tag": tag})))

    assert parsed.offset == -1e-5
    assert parsed.tag == tag


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
        "--none=None",
        "--flag=true",
        "--n=3",
        "--ratio=2.5",
        "--tag=s",
        "--items=[1,a,false]",
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
    with pytest.raises(TypeError, match="yaml_file"):
        format_argparse({"model": value})


def test_yaml_file_materializes_one_complete_trainer_config(tmp_path: Path) -> None:
    """The launch path writes one complete trainer YAML and the command names it.

    ``render_command`` never writes the file itself: the trial writer does, and
    passes its exact path back as ``materialized_input_path``, as mirrored here.
    """
    trial_dir = tmp_path / "work-{phase}" / "trial-{trial_id}"
    trial_dir.mkdir(parents=True)
    base = {
        "model": {"name": "tiny", "depth": 4, "dropout": 0.1},
        "optimizer": {"name": "adamw", "lr": 3e-4},
        "data": {"path": "data/train.jsonl"},
        "output_dir": "{trial_dir}/trainer/{phase}-{trial_id}",
    }
    overrides = {
        "model.depth": 8,
        "optimizer.lr": 1e-4,
        "trainer.seed": 17,
        "trainer.label": "{trial_dir}",
    }

    config_path = trial_dir / TRAINER_INPUT_FILENAMES["yaml_file"]
    config_path.write_text(
        dump_trial_trainer_config_yaml(
            base,
            overrides,
            substitutions={
                "{trial_dir}": str(trial_dir),
                "{trial_id}": "3",
                "{phase}": "depth",
                "{run_name}": "x-depth-3",
            },
        ),
        encoding="utf-8",
    )

    command = render_command(
        "python train.py {config_path}",
        overrides,
        "yaml_file",
        trial_dir=trial_dir,
        trial_id=3,
        phase="depth",
        run_name="x-depth-3",
        trainer_config=base,
        materialized_input_path=config_path,
    )

    assert shlex.split(command) == ["python", "train.py", str(config_path)]
    assert yaml.safe_load(config_path.read_text()) == {
        "data": {"path": "data/train.jsonl"},
        "model": {"name": "tiny", "depth": 8, "dropout": 0.1},
        "optimizer": {"name": "adamw", "lr": 1e-4},
        "output_dir": f"{trial_dir}/trainer/depth-3",
        "trainer": {"label": "{trial_dir}", "seed": 17},
    }
    assert base["model"]["depth"] == 4


def test_yaml_file_dump_is_deterministic_across_mapping_order() -> None:
    left = {"z": 1, "nested": {"b": 2, "a": [True, None]}, "a": "first"}
    right = {"a": "first", "nested": {"a": [True, None], "b": 2}, "z": 1}

    assert dump_trainer_config_yaml(left) == dump_trainer_config_yaml(right)


def test_compose_trainer_config_replaces_exact_leaf_but_rejects_scalar_descent() -> None:
    assert compose_trainer_config(
        {"model": {"depth": 4, "legacy": True}},
        {"model": "pretrained/model"},
    ) == {"model": "pretrained/model"}

    with pytest.raises(ValueError, match=r"path 'model'.*not a mapping"):
        compose_trainer_config({"model": "pretrained/model"}, {"model.depth": 8})


def test_yaml_file_is_the_default_and_requires_config_path(tmp_path: Path) -> None:
    valid = write_yaml(
        tmp_path,
        """
        experiment: yaml_first
        trial_command: "python train.py {config_path}"
        trainer_config:
          model: {depth: 4}
          optimizer: {lr: 0.001}
        metric:
          name: loss
          goal: minimize
          extractor: {type: json_envelope, objective_name: loss, split: test, policy: test}
        phases:
          - name: depth
            n_trials: 1
            search_space:
              model.depth: {type: int, low: 4, high: 8}
        """,
    )

    experiment = load_experiment(valid)
    assert experiment.override_format == "yaml_file"
    assert experiment.trainer_config["model"]["depth"] == 4

    invalid = valid.with_name("missing-placeholder.yaml")
    invalid.write_text(valid.read_text().replace("{config_path}", "--fixed"))
    with pytest.raises(ValidationError, match=r"does not reference \{config_path\}"):
        load_experiment(invalid)


@pytest.mark.parametrize(
    ("body", "error_match"),
    [
        pytest.param(
            """
        experiment: bad_base
        trial_command: "python train.py {config_path}"
        trainer_config:
          model: pretrained/model
        metric:
          name: loss
          goal: minimize
          extractor: {type: json_envelope, objective_name: loss, split: test, policy: test}
        phases:
          - name: depth
            n_trials: 1
            search_space:
              model.depth: {type: int, low: 4, high: 8}
        """,
            r"trainer_config path 'model'.*not a mapping",
            id="sampled-path-crosses-scalar-base",
        ),
        pytest.param(
            """
        experiment: bad_value
        trial_command: "python train.py {config_path}"
        trainer_config:
          cutoff: 2024-01-01
        metric:
          name: loss
          goal: minimize
          extractor: {type: json_envelope, objective_name: loss, split: test, policy: test}
        phases: [{name: p, n_trials: 1}]
        """,
            r"trainer_config.cutoff.*type date",
            id="nonportable-yaml-date",
        ),
    ],
)
def test_yaml_file_rejects_invalid_trainer_config(
    tmp_path: Path,
    body: str,
    error_match: str,
) -> None:
    assert_invalid_experiment_yaml(tmp_path, body, error_match)


def test_compatibility_format_rejects_ignored_trainer_config() -> None:
    with pytest.raises(ValueError, match="would ignore it"):
        make_experiment(
            override_format="argparse",
            trial_command="echo {overrides}",
            trainer_config={"model": {"depth": 4}},
        )


@pytest.mark.parametrize("selector", ["hydra", "json_file"])
def test_restored_override_format_is_validated_without_artifacts(
    tmp_path: Path, selector: str
) -> None:
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path / "runs"}
        trial_command: "echo {{{"overrides_path" if selector == "json_file" else "overrides"}}}"
        override_format: {selector}
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases: [{{name: p, n_trials: 1}}]
        """,
    )

    assert load_experiment(p).override_format == selector
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        3,
        -4,
        2.5,
        -1e-5,
        "null",
        "true",
        "3",
        "2.5",
        "",
        "two words",
        "quote\"and'quote",
        "C:\\data\\file",
        "trailing\\",
        'backslash\\"quote',
        "two\\\\slashes",
        "a,b",
        "{a: b}",
        "$money",
        "unicode café",
        [None, "false", [True, "a,b", "C:\\data"]],
    ],
)
def test_hydra_values_round_trip_real_parser_and_omegaconf(value):
    pytest.importorskip("hydra")
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    argument = shlex.split(format_hydra({"value": value}))[0]
    parsed = OverridesParser.create().parse_override(argument)
    assert not parsed.is_sweep_override()
    actual = OmegaConf.to_container(OmegaConf.create({"value": parsed.value()}), resolve=True)[
        "value"
    ]
    assert actual == value
    assert type(actual) is type(value)
    assert json.dumps(actual) == json.dumps(value)


@pytest.mark.parametrize("value", ["${oc.env:HOME}", "literal ${other}", ["${x}"], "line\nbreak"])
def test_hydra_unsupported_literals_fail_before_artifacts(tmp_path, value):
    with pytest.raises(ValueError, match=r"Hydra cannot pass literal strings.*interpolation"):
        make_experiment(
            workdir=str(tmp_path / "runs"),
            override_format="hydra",
            trial_command="echo {overrides}",
            phases=[Phase(name="p", n_trials=1, fixed_overrides={"value": value})],
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("value", ["${oc.env:HOME}", "line\nbreak"])
def test_hydra_unsupported_categorical_literals_fail_before_artifacts(tmp_path, value):
    with pytest.raises(
        ValueError,
        match=r"categorical search_space key 'value'.*Hydra cannot pass literal strings.*interpolation",
    ):
        make_experiment(
            workdir=str(tmp_path / "runs"),
            override_format="hydra",
            trial_command="echo {overrides}",
            phases=[
                Phase(
                    name="p",
                    n_trials=1,
                    search_space={"value": {"type": "categorical", "choices": ["ok", value]}},
                )
            ],
        )
    assert not (tmp_path / "runs").exists()


def test_yaml_exponent_scalars_preserve_declared_types(tmp_path):
    """YAML's scalar typing remains visible to fixed and categorical overrides."""
    experiment = load_experiment(
        write_yaml(
            tmp_path,
            """
            experiment: exponent_scalars
            workdir: runs
            trial_command: "python train.py {overrides}"
            override_format: argparse
            metric:
              name: loss
              goal: minimize
              extractor: {type: json_envelope, objective_name: loss, split: validation, policy: final}
            phases:
              - name: p
                n_trials: 1
                fixed_overrides:
                  plain_exponent: 1e-3
                  decimal_exponent: 1.0e-3
                search_space:
                  learning_rate:
                    type: categorical
                    choices: [1e-3, 1.0e-3]
            """,
        )
    )

    phase = experiment.phases[0]
    assert phase.fixed_overrides["plain_exponent"] == "1e-3"
    assert type(phase.fixed_overrides["plain_exponent"]) is str
    assert phase.fixed_overrides["decimal_exponent"] == 0.001
    assert type(phase.fixed_overrides["decimal_exponent"]) is float
    choices = phase.search_space["learning_rate"].choices
    assert choices == ["1e-3", 0.001]
    assert type(choices[0]) is str
    assert type(choices[1]) is float


@pytest.mark.parametrize("overrides", [{"x": 1, "x.y": 2}, {"x.y": 2, "x": 1}, {"x": {}, "x.y": 2}])
def test_json_file_rejects_dotted_collisions(overrides):
    with pytest.raises(ValueError, match="parent key"):
        dump_json_file_overrides(overrides)


@pytest.mark.parametrize("value", [float("inf"), datetime.date(2024, 1, 1), {1: "value"}, (1, 2)])
def test_json_file_rejects_non_json_values(value):
    with pytest.raises((TypeError, ValueError)):
        dump_json_file_overrides({"value": value})


@pytest.mark.parametrize("selector", ["hydra", "json_file"])
@pytest.mark.integration
def test_restored_input_real_consumer_inherits_and_replays(tmp_path, selector):
    from phasesweep.config import (
        CategoricalParam,
        ExecutionContext,
        LogRegexExtractor,
        Metric,
        Sampler,
    )

    trainer = tmp_path / "trainer.py"
    if selector == "hydra":
        # The trainer below imports Hydra in a subprocess, where a missing
        # package would surface as a failed trial rather than a skip.
        pytest.importorskip("hydra")
        (tmp_path / "config.yaml").write_text(
            "model: {depth: 0}\nrate: 0.0\ntag: ''\n", encoding="utf-8"
        )
        trainer.write_text(
            "import hydra, json, os\nfrom pathlib import Path\n"
            "from omegaconf import OmegaConf\n"
            "@hydra.main(version_base=None, config_path='.', config_name='config')\n"
            "def main(cfg):\n"
            "    values = OmegaConf.to_container(cfg, resolve=True)\n"
            "    Path(os.environ['PHASESWEEP_TRIAL_DIR'], 'consumed.json').write_text(json.dumps(values))\n"
            "    print('loss=' + str(10 - values['model']['depth']))\n"
            "main()\n",
            encoding="utf-8",
        )
        placeholder = "{overrides} hydra.run.dir={trial_dir}/hydra hydra.output_subdir=null"
    else:
        trainer.write_text(
            "import json, sys, os\nfrom pathlib import Path\n"
            "values = json.loads(Path(sys.argv[1]).read_text())\n"
            "Path(os.environ['PHASESWEEP_TRIAL_DIR'], 'consumed.json').write_text(json.dumps(values))\n"
            "print('loss=' + str(10 - values['model']['depth']))\n",
            encoding="utf-8",
        )
        placeholder = "{overrides_path}"
    experiment = make_experiment(
        workdir=str(tmp_path / "runs"),
        storage="auto",
        provenance={"trainer": "fixture"},
        override_format=selector,
        execution=ExecutionContext(inherit_env="none"),
        trial_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(trainer))} {placeholder}",
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"loss=(?P<value>[0-9.]+)"),
        ),
        phases=[
            Phase(
                name="depth",
                n_trials=2,
                gpu_policy="none",
                sampler=Sampler(type="grid"),
                fixed_overrides={"tag": 'space,quote"backslash\\'},
                search_space={"model.depth": CategoricalParam(type="categorical", choices=[2, 4])},
            ),
            Phase(
                name="rate",
                inherits=["depth"],
                n_trials=2,
                gpu_policy="none",
                sampler=Sampler(type="grid"),
                search_space={"rate": CategoricalParam(type="categorical", choices=[0.1, 0.2])},
            ),
        ],
    )
    winners = run_experiment(experiment)
    root = Path(experiment.workdir) / experiment.experiment
    receipts = sorted((root / "rate").glob("trial_*/consumed.json"))
    assert len(receipts) == 2
    for receipt in receipts:
        consumed = json.loads(receipt.read_text())
        assert consumed["model"]["depth"] == 4
        assert type(consumed["model"]["depth"]) is int
        assert consumed["tag"] == 'space,quote"backslash\\'
    run_experiment(experiment, from_phase="rate")
    assert sorted((root / "rate").glob("trial_*/consumed.json")) == receipts
    assert winners["depth"].effective_overrides["model.depth"] == 4


@pytest.mark.parametrize("override_format", ["argparse", "hydra"])
def test_packaged_fake_trainer_consumes_dotted_cli_overrides(
    tmp_path: Path, override_format: str
) -> None:
    trainer = copy_fake_train(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        override_format=override_format,
        trial_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(trainer))} {{overrides}}",
        metric=Metric(
            name="eval_loss",
            goal="minimize",
            extractor=JsonEnvelopeExtractor(
                type="json_envelope",
                path="result.json",
                objective_name="eval_loss",
                split="validation",
                policy="synthetic",
            ),
        ),
        phases=[
            Phase(
                name="p",
                n_trials=1,
                fixed_overrides={
                    "model.n_layers": 6,
                    "optimizer.lr": 0.001,
                    "optimizer.weight_decay": 0.2,
                    "model.dropout": 0.25,
                },
            )
        ],
    )

    run_experiment(experiment)

    (result_path,) = (Path(experiment.workdir) / experiment.experiment / "p").glob(
        "trial_*/result.json"
    )
    assert json.loads(result_path.read_text())["config"] == {
        "n_layers": 6,
        "lr": 0.001,
        "weight_decay": 0.2,
        "dropout": 0.25,
    }


def _override_yaml(tmp_path, override_format: str, body: str):
    """Write a minimal override config with caller-supplied phase/contract body."""
    assert override_format == "argparse"
    trial_command = "python train.py {overrides}"
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


def test_validate_rejects_argparse_categorical_wire_collision(tmp_path: Path) -> None:
    """Distinct categorical points must not launch byte-identical argparse values."""
    config = _override_yaml(
        tmp_path,
        "argparse",
        "        phases:\n"
        "          - name: t\n"
        "            n_trials: 2\n"
        "            sampler: {type: grid}\n"
        "            search_space:\n"
        "              knob: {type: categorical, choices: [1, '1']}\n",
    )

    with pytest.raises(ValidationError, match="both render as '1'.*override_format='argparse'"):
        load_experiment(config)


def test_argparse_fixed_override_values_keep_distinct_phase_fingerprints():
    """With the value contract enforced, distinct Python values always produce
    distinct JSON-mode dumps — so no two configs that render different commands
    can share a study identity."""
    # Import only: the fingerprint code itself is deliberately untouched.
    from phasesweep.engine.fingerprints import _phase_fingerprint

    fingerprints: dict[str, str] = {}
    for label, value in {"int": 1, "str": "1", "bool": True, "float": 1.0}.items():
        exp = make_experiment(n_trials=1, search_space={}, fixed_overrides={"knob": value})
        fingerprints[label] = _phase_fingerprint(exp, exp.phases[0], {})

    assert len(set(fingerprints.values())) == len(fingerprints)


# ---- migrated from version-named files ----


@pytest.mark.parametrize("override_format", ["argparse", "hydra"])
@pytest.mark.integration
def test_effective_overrides_include_fixed(tmp_path, override_format):
    """The packaged trainer consumes dotted inherited CLI values in both supported forms."""
    trainer = copy_fake_train(tmp_path)

    db_path = tmp_path / "phases.journal"
    yaml_text = f"""
experiment: eff_override_test
storage: journal:///{db_path}
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path / "runs"}
trial_command: "python {trainer} {{overrides}}"
override_format: {override_format}
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: arch
    fixed_overrides:
      model.dropout: 0.2
    n_trials: 2
    sampler: {{ type: grid }}
    search_space:
      model.n_layers: {{ type: categorical, choices: [4, 8] }}
  - name: opt
    inherits: [arch]
    n_trials: 2
    sampler: {{ type: grid }}
    search_space:
      optimizer.lr: {{ type: categorical, choices: [1.0e-5, 0.0003] }}
"""
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(yaml_text)
    exp = load_experiment(yaml_path)
    winners = run_experiment(exp)

    # The opt phase winner should have the parent's fixed override and sampled
    # dotted key. The 1e-5 categorical value is rendered in exponent form,
    # which the fake trainer must parse on both compatibility wires.
    opt_winner = winners["opt"]
    assert opt_winner.effective_overrides["model.dropout"] == 0.2
    assert opt_winner.effective_overrides["model.n_layers"] == 8
    assert opt_winner.effective_overrides["optimizer.lr"] == 0.0003

    result_paths = sorted((Path(exp.workdir) / exp.experiment / "opt").glob("trial_*/result.json"))
    assert len(result_paths) == 2
    consumed = [json.loads(path.read_text())["config"] for path in result_paths]
    assert {values["lr"] for values in consumed} == {1e-5, 0.0003}
    assert {values["n_layers"] for values in consumed} == {8}


def test_transitive_inherited_search_key_cannot_be_resampled(tmp_path):
    """A grandchild may not re-sample a key locked two levels up."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        override_format: argparse
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


def test_runtime_refuses_sampling_an_inherited_winner_key() -> None:
    """The runtime keeps malformed or stale configs from replacing inherited winners."""
    from phasesweep.engine.phase import _composed_overrides
    from phasesweep.engine.state import Winner

    experiment = make_experiment(
        trial_command="echo {overrides}",
        phases=[
            Phase(name="base", n_trials=1),
            Phase(name="child", n_trials=1, inherits=["base"]),
        ],
    )
    stale_child = experiment.phases[1].model_copy(update={"search_space": {"depth": object()}})
    inherited = {
        "base": Winner(
            trial_number=0,
            params={"depth": 8},
            effective_overrides={"depth": 8},
            metric=1.0,
        )
    }

    with pytest.raises(ValueError, match="re-samples inherited winner key"):
        _composed_overrides(stale_child, {"depth": 16}, inherited)


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
    """A child must explicitly resolve a key from independent parent origins."""
    p = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        override_format: argparse
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
        with pytest.raises(ValidationError, match="conflicting key"):
            load_experiment(p)
        return
    exp = load_experiment(p)
    child = exp.phases[-1]
    assert child.fixed_overrides["lr"] == 5.0e-4
