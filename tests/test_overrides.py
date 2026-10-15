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
from phasesweep.config.models import Phase
from phasesweep.config.search import CategoricalParam, FloatParam, Sampler
from phasesweep.runtime.commands import (
    TRAINER_INPUT_FILENAMES,
    compose_trainer_config,
    dump_json_file_overrides,
    dump_trainer_config_yaml,
    dump_trial_trainer_config_yaml,
    flatten_trainer_config,
    format_argparse,
    format_hydra,
    merged_cli_overrides,
    render_command,
)
from tests.conftest import (
    FAKE_TRAIN_METRIC,
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
    fingerprint."""
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


_CLI_BASE = {
    "model": {"name": "tiny", "depth": 4},
    "optimizer": {"lr": 3e-4},
    "layers": [1, 2],
    "output_dir": "{trial_dir}/out/{phase}-{trial_id}",
}
_CLI_OVERRIDES = {"model.depth": 8, "trainer.seed": 17}


def test_trainer_config_is_the_base_for_every_format(tmp_path: Path) -> None:
    """Every format renders the base with the overrides applied and expands its placeholders."""
    trial_dir = tmp_path / "trial"
    render = {
        "trial_dir": trial_dir,
        "trial_id": 3,
        "phase": "depth",
        "run_name": "x-depth-3",
        "trainer_config": _CLI_BASE,
    }
    output_dir = f"{trial_dir}/out/depth-3"

    argparse_command = render_command("train {overrides}", _CLI_OVERRIDES, "argparse", **render)
    assert sorted(shlex.split(argparse_command)[1:]) == sorted(
        [
            "--model.name=tiny",
            "--model.depth=8",
            "--optimizer.lr=0.0003",
            "--layers=[1,2]",
            f"--output_dir={output_dir}",
            "--trainer.seed=17",
        ]
    )

    hydra_command = render_command("train {overrides}", _CLI_OVERRIDES, "hydra", **render)
    assert sorted(shlex.split(hydra_command)[1:]) == sorted(
        [
            'model.name="tiny"',
            "model.depth=8",
            "optimizer.lr=0.0003",
            "layers=[1,2]",
            f'output_dir="{output_dir}"',
            "trainer.seed=17",
        ]
    )

    assert json.loads(
        dump_json_file_overrides(
            _CLI_OVERRIDES,
            trainer_config=_CLI_BASE,
            substitutions={
                "{trial_dir}": str(trial_dir),
                "{trial_id}": "3",
                "{phase}": "depth",
                "{run_name}": "x-depth-3",
            },
        )
    ) == {
        "model": {"name": "tiny", "depth": 8},
        "optimizer": {"lr": 3e-4},
        "layers": [1, 2],
        "output_dir": output_dir,
        "trainer": {"seed": 17},
    }
    json_command = render_command("train {overrides_path}", _CLI_OVERRIDES, "json_file", **render)
    assert shlex.split(json_command) == ["train", str(trial_dir / "overrides.json")]


def test_empty_trainer_config_leaves_overrides_untouched() -> None:
    """Without a base the overrides pass through in their own order, as a json_file base-less dump."""
    overrides = {"a.x": 1, "b": 2, "a.y": 3}

    assert list(merged_cli_overrides({}, overrides)) == ["a.x", "b", "a.y"]
    assert json.loads(dump_json_file_overrides(overrides)) == {"a": {"x": 1, "y": 3}, "b": 2}


@pytest.mark.parametrize(
    ("trainer_config", "overrides"),
    [
        pytest.param(_CLI_BASE, _CLI_OVERRIDES, id="nested-base-with-new-and-replaced-leaves"),
        pytest.param(_CLI_BASE, {"model": "pretrained/tiny"}, id="override-replaces-subtree"),
        pytest.param({"a": {"b": {"c": 1}}}, {"a.b.d": 2, "a.e": [1, 2]}, id="deep-merge"),
    ],
)
def test_flattened_merge_agrees_with_the_nested_json_composition(
    trainer_config: dict, overrides: dict
) -> None:
    """argparse and hydra see the same values json_file nests, only dotted."""
    nested = json.loads(dump_json_file_overrides(overrides, trainer_config=trainer_config))

    assert merged_cli_overrides(trainer_config, overrides) == flatten_trainer_config(nested)


@pytest.mark.parametrize(
    "mapping",
    [
        pytest.param({"a.b": 1, "a": {"b": 2}}, id="literal-key-first"),
        pytest.param({"a": {"b": 2}, "a.b": 1}, id="nested-first"),
    ],
)
def test_flatten_trainer_config_rejects_duplicate_flat_keys(mapping: dict) -> None:
    with pytest.raises(ValueError, match=r"dotted key 'a\.b' twice"):
        flatten_trainer_config(mapping)


def test_flatten_trainer_config_keeps_lists_and_empty_mappings_as_leaves() -> None:
    assert flatten_trainer_config({"a": {"b": [{"c": 1}], "d": {}}, "e": None}) == {
        "a.b": [{"c": 1}],
        "a.d": {},
        "e": None,
    }
    with pytest.raises(TypeError, match="non-string mapping key 1"):
        flatten_trainer_config({"a": {1: "x"}})


@pytest.mark.parametrize("override_format", ["argparse", "hydra"])
@pytest.mark.parametrize(
    "leaf",
    [
        pytest.param([{"x": 1}], id="list-of-mappings"),
        pytest.param({}, id="empty-mapping"),
    ],
)
def test_cli_format_rejects_unrenderable_trainer_config_leaf(
    override_format: str, leaf: object
) -> None:
    with pytest.raises(
        ValidationError,
        match=rf"trainer_config key 'opt\.leaf': override_format='{override_format}' supports",
    ):
        make_experiment(
            override_format=override_format,
            trial_command="echo {overrides}",
            trainer_config={"opt": {"leaf": leaf}},
        )


@pytest.mark.parametrize(
    ("override_format", "placeholder"),
    [("argparse", "overrides"), ("hydra", "overrides"), ("json_file", "overrides_path")],
)
def test_cli_format_requires_overrides_placeholder_when_trainer_config_is_set(
    override_format: str, placeholder: str
) -> None:
    constant = {"override_format": override_format, "n_trials": 1, "search_space": {}}

    make_experiment(trial_command="echo constant", **constant)
    with pytest.raises(ValidationError, match=rf"does not reference \{{{placeholder}\}}"):
        make_experiment(
            trial_command="echo constant", trainer_config={"model": {"depth": 4}}, **constant
        )
    make_experiment(
        trial_command=f"echo {{{placeholder}}}", trainer_config={"model": {"depth": 4}}, **constant
    )


def test_cli_format_reports_a_base_the_overrides_cannot_descend_through() -> None:
    with pytest.raises(ValidationError, match=r"trainer_config and composed overrides are invalid"):
        make_experiment(
            override_format="argparse",
            trial_command="echo {overrides}",
            trainer_config={"model": "pretrained/tiny"},
            search_space={"model.depth": CategoricalParam(type="categorical", choices=[4, 8])},
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
    from phasesweep.config.models import ExecutionContext, Metric
    from phasesweep.config.search import CategoricalParam, Sampler
    from phasesweep.evidence.models import LogRegexExtractor

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
        metric=FAKE_TRAIN_METRIC,
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
# all.
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


def test_validate_rejects_argparse_categorical_wire_collision() -> None:
    """Distinct categorical points must not launch byte-identical argparse values."""
    with pytest.raises(ValidationError, match="both render as '1'.*override_format='argparse'"):
        make_experiment(
            n_trials=2,
            sampler=Sampler(type="grid"),
            search_space={"knob": CategoricalParam(type="categorical", choices=[1, "1"])},
        )


def test_argparse_fixed_override_values_keep_distinct_phase_fingerprints():
    """With the value contract enforced, distinct Python values always produce
    distinct JSON-mode dumps — so no two configs that render different commands
    can share a study identity."""
    from phasesweep.engine.fingerprints import _phase_fingerprint

    fingerprints: dict[str, str] = {}
    for label, value in {"int": 1, "str": "1", "bool": True, "float": 1.0}.items():
        exp = make_experiment(n_trials=1, search_space={}, fixed_overrides={"knob": value})
        fingerprints[label] = _phase_fingerprint(exp, exp.phases[0], {})

    assert len(set(fingerprints.values())) == len(fingerprints)


@pytest.mark.parametrize("override_format", ["argparse", "hydra"])
@pytest.mark.integration
def test_effective_overrides_include_fixed(tmp_path, override_format):
    """The packaged trainer consumes dotted inherited CLI values in both supported forms."""
    trainer = copy_fake_train(tmp_path)

    exp = make_experiment(
        experiment="eff_override_test",
        persistent=tmp_path,
        trial_command=f"python {trainer} {{overrides}}",
        override_format=override_format,
        metric=FAKE_TRAIN_METRIC,
        phases=[
            Phase(
                name="arch",
                n_trials=2,
                sampler=Sampler(type="grid"),
                fixed_overrides={"model.dropout": 0.2},
                search_space={
                    "model.n_layers": CategoricalParam(type="categorical", choices=[4, 8])
                },
            ),
            Phase(
                name="opt",
                inherits=["arch"],
                n_trials=2,
                sampler=Sampler(type="grid"),
                search_space={
                    "optimizer.lr": CategoricalParam(type="categorical", choices=[1.0e-5, 0.0003])
                },
            ),
        ],
    )
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


def test_transitive_inherited_search_key_cannot_be_resampled() -> None:
    """A grandchild may not re-sample a key locked two levels up."""
    with pytest.raises(
        ValidationError,
        match=(
            r"Phase 'reg' re-samples key\(s\) \['n_layers'\] that an inherited phase already "
            r"sampled \(n_layers from 'arch', possibly transitively\)\..*"
            r"remove the key from search_space or drop the inherit\..*"
            r"Keys a parent only fixed may be searched\."
        ),
    ):
        make_experiment(
            phases=[
                Phase(
                    name="arch",
                    n_trials=1,
                    search_space={"n_layers": CategoricalParam(type="categorical", choices=[4, 8])},
                ),
                Phase(
                    name="lr",
                    inherits=["arch"],
                    n_trials=1,
                    search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
                ),
                Phase(
                    name="reg",
                    inherits=["lr"],
                    n_trials=1,
                    search_space={
                        "n_layers": CategoricalParam(type="categorical", choices=[12, 16])
                    },
                ),
            ]
        )


def test_runtime_refuses_sampling_an_inherited_winner_key() -> None:
    """The runtime keeps malformed or stale configs from replacing inherited winners."""
    from phasesweep.engine.phase import _composed_overrides
    from phasesweep.engine.state import Winner

    experiment = make_experiment(
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="base",
                n_trials=1,
                search_space={"depth": CategoricalParam(type="categorical", choices=[8, 16])},
            ),
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

    with pytest.raises(
        ValueError, match="re-samples inherited winner key.*that an ancestor sampled"
    ):
        _composed_overrides(experiment, stale_child, {"depth": 16}, inherited)


def test_runtime_allows_resampling_a_key_the_parent_only_fixed() -> None:
    """A sampled value replaces an inherited value the parent only fixed."""
    from phasesweep.engine.phase import _composed_overrides
    from phasesweep.engine.state import Winner

    experiment = make_experiment(
        trial_command="echo {overrides}",
        phases=[
            Phase(name="base", n_trials=1, fixed_overrides={"epochs": 1}),
            Phase(
                name="child",
                n_trials=1,
                inherits=["base"],
                search_space={"epochs": CategoricalParam(type="categorical", choices=[2, 4])},
            ),
        ],
    )
    inherited = {
        "base": Winner(
            trial_number=0,
            params={},
            effective_overrides={"epochs": 1},
            metric=1.0,
        )
    }

    composed = _composed_overrides(experiment, experiment.phases[1], {"epochs": 4}, inherited)

    assert composed == {"epochs": 4}


def test_grandparent_sampled_then_parent_fixed_key_may_be_searched() -> None:
    """A parent that re-fixes a key takes over its origin, which unlocks the key."""
    experiment = make_experiment(
        phases=[
            Phase(
                name="arch",
                n_trials=1,
                search_space={"n_layers": CategoricalParam(type="categorical", choices=[4, 8])},
            ),
            Phase(name="pin", n_trials=1, inherits=["arch"], fixed_overrides={"n_layers": 6}),
            Phase(
                name="reg",
                n_trials=1,
                inherits=["pin"],
                search_space={"n_layers": CategoricalParam(type="categorical", choices=[12, 16])},
            ),
            Phase(name="final", n_trials=1, inherits=["reg"]),
        ]
    )

    assert experiment.sampled_locked_keys("pin") == {"n_layers"}
    assert experiment.sampled_locked_keys("reg") == frozenset()
    assert experiment.sampled_locked_keys("final") == {"n_layers"}


def test_diamond_after_resample_requires_explicit_fix() -> None:
    """Two parents with different origins for a key need a child fix, which stays unsearchable."""
    base = Phase(name="base", n_trials=1, fixed_overrides={"epochs": 1})
    resampled = Phase(
        name="resampled",
        n_trials=1,
        inherits=["base"],
        search_space={"epochs": CategoricalParam(type="categorical", choices=[2, 4])},
    )
    join = Phase(name="join", n_trials=1, inherits=["base", "resampled"])

    with pytest.raises(ValidationError, match=r"inherits conflicting key.*'epochs'"):
        make_experiment(phases=[base, resampled, join])

    fixed_join = join.model_copy(update={"fixed_overrides": {"epochs": 3}})
    experiment = make_experiment(phases=[base, resampled, fixed_join])

    assert experiment.sampled_locked_keys("join") == {"epochs"}
    assert experiment.sampled_locked_keys("resampled") == frozenset()


def test_sampled_locked_keys_rejects_an_unknown_phase() -> None:
    experiment = make_experiment()

    with pytest.raises(KeyError):
        experiment.sampled_locked_keys("missing")
