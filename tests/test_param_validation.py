"""Search-space and override validation: param-type bounds, categorical scalar-only, dotted-prefix collisions, override-key shell safety, sampler/search-space compatibility, and trial-command template placeholder enforcement."""

from __future__ import annotations

import textwrap
from pathlib import Path

import optuna
import pytest
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config import (
    CategoricalParam,
    Constraint,
    Experiment,
    FloatParam,
    IntParam,
    JsonExtractor,
    Metric,
    Phase,
    Sampler,
    check_bounds,
)
from phasesweep.config.common import _find_prefix_collisions
from phasesweep.engine.optuna import _build_sampler
from tests.conftest import make_experiment, write_yaml


def _grid_yaml(tmp_path: Path, search_space: str, *, n_trials: int = 1) -> Path:
    """Write a minimal grid-sampler config with caller-supplied search space."""
    return write_yaml(
        tmp_path,
        f"""
        experiment: t
        trial_command: "echo {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json, path: r.json, key: x }}
        phases:
          - name: p
            n_trials: {n_trials}
            sampler: {{ type: grid }}
            search_space:
{textwrap.indent(textwrap.dedent(search_space).strip(), "              ")}
        """,
    )


@pytest.mark.parametrize(("n_jobs", "constant_liar"), [(4, True), (1, False)])
def test_tpe_sampler_constant_liar_policy(n_jobs: int, constant_liar: bool) -> None:
    """TPE enables constant_liar only for parallel optimization."""
    cfg = Sampler(type="tpe", seed=0)
    space = {"x": CategoricalParam(type="categorical", choices=[1, 2, 3])}
    sampler = _build_sampler(cfg, space, n_jobs=n_jobs)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._constant_liar is constant_liar


def test_param_constructors_reject_invalid_scalar_settings() -> None:
    cases = [
        ("float_low_gt_high", FloatParam, {"type": "float", "low": 1.0, "high": 0.0}, "low.*high"),
        (
            "float_log_non_positive",
            FloatParam,
            {"type": "float", "low": 0.0, "high": 1.0, "log": True},
            "low > 0",
        ),
        (
            "int_step_zero",
            IntParam,
            {"type": "int", "low": 0, "high": 10, "step": 0},
            "step must be > 0",
        ),
        (
            "int_log_step_not_one",
            IntParam,
            {"type": "int", "low": 1, "high": 100, "log": True, "step": 2},
            "log=true with step != 1",
        ),
        (
            "categorical_non_scalar",
            CategoricalParam,
            {"type": "categorical", "choices": [[1, 2], [3, 4]]},
            "Optuna-compatible scalars",
        ),
        (
            "categorical_nan_float",
            CategoricalParam,
            {"type": "categorical", "choices": [1.0, float("nan")]},
            "must be finite",
        ),
    ]

    for _case, model, kwargs, match in cases:
        with pytest.raises(ValidationError, match=match):
            model(**kwargs)


def test_check_bounds_rejects_non_finite_values():
    """NaN/inf are always out of bounds."""
    assert check_bounds(50.0, min_value=0.0, max_value=100.0) is True
    assert check_bounds(float("nan"), min_value=0.0, max_value=100.0) is False
    assert check_bounds(float("inf"), min_value=0.0, max_value=100.0) is False
    assert check_bounds(float("-inf"), min_value=0.0, max_value=100.0) is False


def test_sampler_rejects_negative_n_startup_trials():
    with pytest.raises(ValidationError):
        Sampler(type="tpe", n_startup_trials=-1)


def test_validate_rejects_cmaes_with_categorical(tmp_path: Path) -> None:
    """`phasesweep validate` must catch CMA-ES + categorical, not first trial."""
    p = write_yaml(
        tmp_path,
        """
        experiment: t
        trial_command: "echo {overrides}"
        metric:
          name: x
          goal: minimize
          extractor: { type: json, path: r.json, key: x }
        phases:
          - name: p
            n_trials: 1
            sampler: { type: cmaes }
            search_space:
              model: { type: categorical, choices: ["a", "b"] }
        """,
    )
    with pytest.raises(ValidationError, match="cmaes.*does not support categorical"):
        load_experiment(p)


def test_validate_rejects_invalid_grid_configs(tmp_path: Path) -> None:
    """Grid sampler configs must be exactly enumerable and complete by default."""
    cases = [
        (
            "log_float",
            "lr: { type: float, low: 1e-5, high: 1e-2, log: true }",
            1,
            "grid.*log-scale float",
        ),
        ("log_int", "n: { type: int, low: 1, high: 1024, log: true }", 1, "grid.*log-scale int"),
        (
            "float_no_step",
            "x: { type: float, low: 0.0, high: 1.0 }",
            1,
            "grid sampler requires 'step'",
        ),
        (
            "non_divisible_float",
            "x: { type: float, low: 0.0, high: 1.0, step: 0.6 }",
            1,
            "must be an integer",
        ),
        (
            "partial_matrix",
            "x: { type: categorical, choices: [1, 2, 3] }",
            2,
            "grid sampler has 3 combinations",
        ),
    ]

    for _case, search_space, n_trials, match in cases:
        with pytest.raises(ValidationError, match=match):
            load_experiment(_grid_yaml(tmp_path, search_space, n_trials=n_trials))


def test_schema_rejects_non_finite_bounds() -> None:
    """NaN/inf bounds silently corrupt comparisons and must fail at config-load."""
    cases = [
        (
            "constraint_nan_max",
            lambda: Constraint(
                name="size",
                extractor=JsonExtractor(type="json", path="r.json", key="size"),
                max=float("nan"),
            ),
        ),
        (
            "constraint_inf_min",
            lambda: Constraint(
                name="size",
                extractor=JsonExtractor(type="json", path="r.json", key="size"),
                min=float("inf"),
            ),
        ),
        ("float_nan_low", lambda: FloatParam(type="float", low=float("nan"), high=1.0)),
        ("float_inf_high", lambda: FloatParam(type="float", low=0.0, high=float("inf"))),
        ("float_nan_step", lambda: FloatParam(type="float", low=0.0, high=1.0, step=float("nan"))),
    ]

    for _case, build in cases:
        with pytest.raises(ValidationError, match="must be finite"):
            build()


def test_validate_accepts_divisible_grid_float(tmp_path: Path) -> None:
    """low=0, high=1, step=0.25 -> exactly [0, 0.25, 0.5, 0.75, 1.0]."""
    load_experiment(
        _grid_yaml(tmp_path, "x: { type: float, low: 0.0, high: 1.0, step: 0.25 }", n_trials=5)
    )


def test_validate_accepts_explicit_partial_grid(tmp_path: Path) -> None:
    """Partial grid phases are allowed only when explicitly requested."""
    p = write_yaml(
        tmp_path,
        """
        experiment: t
        trial_command: "echo {overrides}"
        metric:
          name: x
          goal: minimize
          extractor: { type: json, path: r.json, key: x }
        phases:
          - name: p
            n_trials: 2
            sampler: { type: grid }
            allow_partial_grid: true
            search_space:
              x: { type: categorical, choices: [1, 2, 3] }
        """,
    )
    load_experiment(p)


def test_validate_rejects_local_fixed_and_sampled_collision(tmp_path: Path) -> None:
    """A key cannot be both fixed_overrides and search_space in the same phase."""
    p = write_yaml(
        tmp_path,
        """
        experiment: t
        trial_command: "echo {overrides}"
        metric:
          name: x
          goal: minimize
          extractor: { type: json, path: r.json, key: x }
        phases:
          - name: p
            n_trials: 1
            fixed_overrides: { lr: 0.001 }
            search_space:
              lr: { type: float, low: 1e-5, high: 1e-2, log: true }
        """,
    )
    with pytest.raises(ValidationError, match="both fixed_overrides and search_space"):
        load_experiment(p)


def test_find_prefix_collisions() -> None:
    """The detector must flag a parent/child pair on the *same* dotted path
    (e.g. ``model`` set as scalar AND ``model.depth`` set as nested), and must
    not flag siblings sharing a parent or fully unrelated keys."""
    cases = [
        ("direct_prefix", {"model", "model.depth"}, [("model", "model.depth")]),
        ("deep_prefix", {"a.b.c", "a.b.c.d"}, [("a.b.c", "a.b.c.d")]),
        ("similar_segment_names", {"model.depth", "model.depths"}, []),
        ("siblings", {"a.x", "a.y"}, []),
        ("disjoint", {"foo", "bar", "baz"}, []),
    ]

    for case, keys, expected in cases:
        assert _find_prefix_collisions(keys) == expected, case


def test_rejects_dotted_prefix_collisions() -> None:
    """Dotted key collisions are rejected locally and through inheritance."""
    cases = [
        (
            "local",
            [
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=1,
                    fixed_overrides={"model": "llama"},
                    search_space={"model.depth": IntParam(type="int", low=8, high=32)},
                )
            ],
        ),
        (
            "inherited_parent_scalar_child_subkey",
            [
                Phase(  # type: ignore[arg-type]
                    name="parent",
                    n_trials=1,
                    fixed_overrides={"optimizer": "adamw"},
                    search_space={"x": IntParam(type="int", low=0, high=1)},
                ),
                Phase(  # type: ignore[arg-type]
                    name="child",
                    n_trials=1,
                    inherits=["parent"],
                    search_space={"optimizer.lr": FloatParam(type="float", low=1e-4, high=1e-2)},
                ),
            ],
        ),
        (
            "inherited_parent_subkey_child_scalar",
            [
                Phase(  # type: ignore[arg-type]
                    name="parent",
                    n_trials=1,
                    fixed_overrides={"optimizer.lr": 0.001},
                    search_space={"x": IntParam(type="int", low=0, high=1)},
                ),
                Phase(  # type: ignore[arg-type]
                    name="child",
                    n_trials=1,
                    inherits=["parent"],
                    fixed_overrides={"optimizer": "adamw"},
                    search_space={"y": IntParam(type="int", low=0, high=1)},
                ),
            ],
        ),
    ]

    for case, phases in cases:
        with pytest.raises(ValueError, match="namespace collision"):
            Experiment(
                experiment=f"t_{case}",
                trial_command="echo {overrides}",
                metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
                phases=phases,
            )


def test_yaml_load_rejects_prefix_collision(tmp_path: Path) -> None:
    """End-to-end: load_experiment surfaces the prefix-collision error."""
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent("""
        experiment: t
        trial_command: "echo {overrides}"
        metric:
          extractor:
            type: json
            path: r.json
            key: x
        phases:
          - name: p
            n_trials: 1
            fixed_overrides:
              model: llama
            search_space:
              model.depth:
                type: int
                low: 8
                high: 32
        """)
    )
    with pytest.raises(ValueError, match="namespace collision"):
        load_experiment(p)


@pytest.mark.parametrize(
    ("template", "override_format", "search_space", "fixed_overrides", "match"),
    [
        (
            "python train.py --out {trial_dir}/result.json",
            "hydra",
            {"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
            None,
            r"does not reference \{overrides\}",
        ),
        (
            "python train.py --out {trial_dir}/result.json",
            "argparse",
            {"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
            None,
            r"does not reference \{overrides\}",
        ),
        (
            "python train.py --out {trial_dir}/result.json",
            "argparse",
            None,
            {"lr": 1e-3},
            r"does not reference \{overrides\}",
        ),
        (
            "python train.py --out {trial_dir} '{{overrides}}'",
            "hydra",
            {"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
            None,
            r"does not reference \{overrides\}",
        ),
        (
            "python train.py '{{overrides_path}}'",
            "json_file",
            {"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
            None,
            r"does not reference \{overrides_path\}",
        ),
        (
            "python train.py --out {trial_dir}/result.json {overrides}",
            "json_file",
            {"x": IntParam(type="int", low=0, high=10)},
            None,
            r"\{overrides_path\}",
        ),
    ],
)
def test_trial_command_requires_live_override_placeholder(
    template: str,
    override_format: str,
    search_space: dict | None,
    fixed_overrides: dict | None,
    match: str,
) -> None:
    """Override-bearing phases must expose the active placeholder to the trainer."""
    with pytest.raises(ValueError, match=match):
        make_experiment(
            trial_command=template,
            override_format=override_format,
            n_trials=1,
            search_space=search_space or {},
            fixed_overrides=fixed_overrides or {},
        )


def test_trial_command_unknown_placeholder_rejected() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        make_experiment(trial_command="echo {trail_dir} {overrides}", n_trials=1)


def test_trial_command_unbalanced_brace_rejected() -> None:
    with pytest.raises(ValueError, match="trial_command failed to render"):
        make_experiment(trial_command="echo {trial_dir", n_trials=1)


def test_trial_command_accepts_supported_templates() -> None:
    """Accepted templates include normal overrides, format specs, and no-override phases."""
    cases = [
        (
            "field_conversion",
            lambda: make_experiment(
                trial_command="python train.py {overrides!s}",
                override_format="hydra",
                n_trials=1,
                search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
            ),
        ),
        (
            "field_format_spec",
            lambda: make_experiment(
                trial_command="python train.py '{overrides:>1}'",
                override_format="argparse",
                n_trials=1,
                search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
            ),
        ),
        (
            "constant_no_overrides",
            lambda: make_experiment(
                trial_command="python train.py --out {trial_dir}/result.json",
                override_format="hydra",
                n_trials=1,
                search_space={},
                fixed_overrides={},
            ),
        ),
        (
            "normal_template",
            lambda: make_experiment(trial_command="echo {trial_dir} {overrides}", n_trials=1),
        ),
        (
            "normal_template_with_output",
            lambda: make_experiment(
                trial_command="python train.py --out {trial_dir}/result.json {overrides}",
                n_trials=1,
            ),
        ),
        (
            "json_file_overrides_path",
            lambda: make_experiment(
                override_format="json_file",
                trial_command="python train.py --out {trial_dir}/result.json --cfg {overrides_path}",
                n_trials=1,
            ),
        ),
    ]

    for _case, build in cases:
        build()


def test_yaml_load_surfaces_template_error(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent("""
        experiment: t
        trial_command: "echo {trail_dir} {overrides}"
        metric:
          extractor:
            type: json
            path: r.json
            key: x
        phases:
          - name: p
            n_trials: 1
            search_space:
              x:
                type: int
                low: 0
                high: 1
        """)
    )
    with pytest.raises(ValueError, match="unknown placeholder"):
        load_experiment(p)


def test_distinct_phase_names_with_same_field_keys_accepted(tmp_path: Path) -> None:
    """Sanity: the strict loader rejects duplicates *within* a mapping, not
    across siblings. Two phases each with their own ``n_trials`` is fine.
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
    search_space: { x: { type: float, low: 0, high: 1 } }
  - name: b
    inherits: [a]
    n_trials: 1
    search_space: { y: { type: float, low: 0, high: 1 } }
"""
    exp = load_experiment(write_yaml(tmp_path, body))
    assert [p.name for p in exp.phases] == ["a", "b"]


def test_override_keys_reject_malformed_and_shell_unsafe_values() -> None:
    """argparse/Hydra can't render an override of an empty or
    whitespace-bearing key correctly. We reject these at config-load.
    """
    malformed_keys = [
        "",  # empty string
        ".",  # single dot
        "..",  # only dots
        ".lr",  # leading dot
        "lr.",  # trailing dot
        "model..depth",  # consecutive dots
        " lr",  # leading whitespace
        "lr ",  # trailing whitespace
        "model depth",  # internal whitespace
        "lr\t",  # internal tab
        "model.\u200bdepth",  # zero-width space
    ]
    shell_unsafe_keys = [
        "lr$",  # shell-special
        "lr*",
        "lr?",
        "lr|x",
        "lr;x",
        "lr/x",  # path separator could escape namespacing
        "lr\\x",
    ]
    cases = [(key, r"override key|empty|whitespace|invalid") for key in malformed_keys]
    cases.extend((key, r"invalid characters") for key in shell_unsafe_keys)

    for target in ("search_space", "fixed_overrides"):
        for bad_key, match in cases:
            kwargs = {
                "name": "p",
                "n_trials": 1,
                "search_space": {"x": FloatParam(type="float", low=0.0, high=1.0)},
            }
            if target == "search_space":
                kwargs["search_space"] = {bad_key: FloatParam(type="float", low=0.0, high=1.0)}
            else:
                kwargs["fixed_overrides"] = {bad_key: 1}
            with pytest.raises(ValidationError, match=match):
                Phase(**kwargs)  # type: ignore[arg-type]


def test_search_space_accepts_well_formed_keys() -> None:
    """Sanity: legitimate hydra-style keys must not trigger the validator.
    Without these passing, the validator would be unusably strict.
    """
    good_keys = [
        "lr",
        "model.depth",
        "hydra.run.dir",
        "data.train_path",
        "optim.weight_decay",
        "x_y_z",
        "kebab-case",
        "model.weight-decay",  # mixed dash and dot
        "_underscored",
        "trailing_underscore_",
        "X1",
        "a.b.c.d",  # deeply nested
    ]

    for good_key in good_keys:
        phase = Phase(
            name="p",
            n_trials=1,
            search_space={good_key: FloatParam(type="float", low=0.0, high=1.0)},
        )
        assert good_key in phase.search_space


def test_cmaes_phase_rejected_at_config_load_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``phasesweep validate`` must catch a cmaes-without-cmaes-installed
    config; pre-v0.5.7 this fired only at first trial launch.

    We simulate the missing package by intercepting the import. The
    validator then surfaces a ``ValueError`` (raised through pydantic
    ``ValidationError``) at ``Experiment(...)`` construction time.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "cmaes":
            raise ImportError("simulated missing cmaes package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValidationError, match=r"cmaes.*not installed"):
        make_experiment(
            sampler=Sampler(type="cmaes", seed=0),
            search_space={"x": IntParam(type="int", low=0, high=10)},
        )


def test_cmaes_phase_loads_when_package_present() -> None:
    """Sanity: a cmaes phase loads cleanly with the packaged dependency."""

    import cmaes  # noqa: F401

    exp = make_experiment(
        sampler=Sampler(type="cmaes", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    assert exp.phases[0].sampler.type == "cmaes"


def test_inherit_search_space_collision_errors(tmp_path):
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
      lr: { type: float, low: 1e-5, high: 1e-2, log: true }
  - name: b
    inherits: [a]
    n_trials: 1
    search_space:
      lr: { type: float, low: 1e-5, high: 1e-2, log: true }
"""
    cfg = write_yaml(tmp_path, body)
    with pytest.raises(ValueError, match="re-samples key"):
        load_experiment(cfg)
