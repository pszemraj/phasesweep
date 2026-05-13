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
    _find_prefix_collisions,
    _key_parts,
    check_bounds,
)
from phasesweep.orchestrator import (
    _build_sampler,
)
from phasesweep.selector import select_winner
from tests.conftest import make_experiment, write_yaml


def _exp_with_template(
    trial_command: str,
    *,
    override_format: str = "hydra",
    search_space: dict | None = None,
    fixed_overrides: dict | None = None,
) -> Experiment:
    """Build an Experiment with one phase that has the given overrides shape.

    Used by the trial-command template validation tests below — they need
    fine-grained control over ``override_format``, search-space dict, and
    fixed-overrides dict, which the conftest ``make_experiment`` factory
    doesn't expose. Tests that don't need that control should use
    ``make_experiment`` from ``conftest`` instead.
    """
    return Experiment(
        experiment="t",
        trial_command=trial_command,
        override_format=override_format,
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        phases=[
            Phase(
                name="p",
                n_trials=1,
                search_space=search_space or {},
                fixed_overrides=fixed_overrides or {},
            )
        ],
    )


def test_tpe_sampler_constant_liar_with_parallel():
    """TPE with n_jobs > 1 should enable constant_liar."""
    cfg = Sampler(type="tpe", seed=0)
    space = {"x": CategoricalParam(type="categorical", choices=[1, 2, 3])}
    sampler = _build_sampler(cfg, space, n_jobs=4)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    # constant_liar is stored as _constant_liar in TPESampler internals
    assert sampler._constant_liar is True


def test_tpe_sampler_no_constant_liar_single_job():
    """TPE with n_jobs=1 should NOT enable constant_liar."""
    cfg = Sampler(type="tpe", seed=0)
    space = {"x": CategoricalParam(type="categorical", choices=[1, 2, 3])}
    sampler = _build_sampler(cfg, space, n_jobs=1)
    assert sampler._constant_liar is False


def test_float_param_low_gt_high():
    from pydantic import ValidationError

    from phasesweep.config import FloatParam

    with pytest.raises(ValidationError, match="low.*high"):
        FloatParam(type="float", low=1.0, high=0.0)


def test_float_param_log_requires_positive():
    from pydantic import ValidationError

    from phasesweep.config import FloatParam

    with pytest.raises(ValidationError, match="low > 0"):
        FloatParam(type="float", low=0.0, high=1.0, log=True)


def test_int_param_step_must_be_positive():
    from pydantic import ValidationError

    from phasesweep.config import IntParam

    with pytest.raises(ValidationError, match="step must be > 0"):
        IntParam(type="int", low=0, high=10, step=0)


def test_check_bounds_rejects_non_finite_values():
    """NaN/inf are always out of bounds."""
    assert check_bounds(50.0, min_value=0.0, max_value=100.0) is True
    assert check_bounds(float("nan"), min_value=0.0, max_value=100.0) is False
    assert check_bounds(float("inf"), min_value=0.0, max_value=100.0) is False
    assert check_bounds(float("-inf"), min_value=0.0, max_value=100.0) is False


def test_selector_rejects_nan_constraint_values_defensively(tmp_path):
    """If a NaN somehow made it into user_attrs (legacy study), selector must reject."""
    db = tmp_path / "s.db"
    storage = f"sqlite:///{db}"
    study = optuna.create_study(study_name="t", storage=storage, direction="minimize")

    # Trial 0: clean, feasible.
    t0 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t0.set_user_attr("phasesweep_feasible", True)
    t0.set_user_attr("constraint:size", 100.0)
    study.tell(t0, 0.5)

    # Trial 1: legacy NaN constraint value but mistakenly marked feasible.
    t1 = study.ask({"x": optuna.distributions.FloatDistribution(0, 1)})
    t1.set_user_attr("phasesweep_feasible", True)
    t1.set_user_attr("constraint:size", float("nan"))
    study.tell(t1, 0.1)  # Better metric — would beat trial 0 if not rejected.

    exp = Experiment(
        experiment="t",
        trial_command="echo {overrides}",
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        constraints=[
            Constraint(
                name="size",
                extractor=JsonExtractor(type="json", path="r.json", key="s"),
                max=1000.0,
            )
        ],
        phases=[
            Phase(name="a", n_trials=1, search_space={"x": IntParam(type="int", low=0, high=10)}),
        ],
    )
    sel = select_winner(study, exp)
    assert sel.trial_number == 0, (
        "NaN-constraint trial 1 must be rejected even though metric was lower"
    )


def test_int_param_rejects_log_with_step_neq_1():
    """Optuna's IntDistribution rejects this at construction; we catch at config-load."""
    with pytest.raises(ValidationError, match="log=true with step != 1"):
        IntParam(type="int", low=1, high=100, log=True, step=2)


def test_sampler_rejects_negative_n_startup_trials():
    with pytest.raises(ValidationError):
        Sampler(type="tpe", n_startup_trials=-1)


def test_categorical_rejects_non_scalar_choice():
    with pytest.raises(ValidationError, match="Optuna-compatible scalars"):
        CategoricalParam(type="categorical", choices=[[1, 2], [3, 4]])


def test_categorical_rejects_nan_float_choice():
    with pytest.raises(ValidationError, match="must be finite"):
        CategoricalParam(type="categorical", choices=[1.0, float("nan")])


def test_cmaes_rejects_categorical_search_space():
    """CMA-ES is float-only in Optuna; categorical params silently fail every trial."""
    with pytest.raises(ValueError, match="cmaes.*does not support categorical"):
        _build_sampler(
            Sampler(type="cmaes", seed=0),
            {"x": CategoricalParam(type="categorical", choices=["a", "b"])},
            n_jobs=1,
        )


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


def test_validate_rejects_grid_with_log_float(tmp_path: Path) -> None:
    """Grid sampler cannot enumerate log-spaced floats; reject at config-load."""
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
            sampler: { type: grid }
            search_space:
              lr: { type: float, low: 1e-5, high: 1e-2, log: true }
        """,
    )
    with pytest.raises(ValidationError, match="grid.*log-scale float"):
        load_experiment(p)


def test_validate_rejects_grid_with_log_int(tmp_path: Path) -> None:
    """Grid sampler cannot enumerate log-spaced ints; reject at config-load."""
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
            sampler: { type: grid }
            search_space:
              n: { type: int, low: 1, high: 1024, log: true }
        """,
    )
    with pytest.raises(ValidationError, match="grid.*log-scale int"):
        load_experiment(p)


def test_validate_rejects_grid_float_without_step(tmp_path: Path) -> None:
    """Grid float without explicit step is ambiguous; reject at config-load."""
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
            sampler: { type: grid }
            search_space:
              x: { type: float, low: 0.0, high: 1.0 }
        """,
    )
    with pytest.raises(ValidationError, match="grid sampler requires 'step'"):
        load_experiment(p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max": float("nan")},  # x > NaN is always False, vacuous
        {"min": float("inf")},  # x < +inf is almost always True, vacuous
    ],
    ids=["nan_max", "inf_min"],
)
def test_constraint_rejects_non_finite_bounds(kwargs: dict[str, float]) -> None:
    """A constraint with NaN/inf bounds is vacuous and would silently corrupt
    feasibility checks — must be rejected at config-load."""
    with pytest.raises(ValidationError, match="must be finite"):
        Constraint(
            name="size",
            extractor=JsonExtractor(type="json", path="r.json", key="size"),
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low": float("nan"), "high": 1.0},
        {"low": 0.0, "high": float("inf")},
        {"low": 0.0, "high": 1.0, "step": float("nan")},
    ],
    ids=["nan_low", "inf_high", "nan_step"],
)
def test_float_param_rejects_non_finite_bounds(kwargs: dict[str, float]) -> None:
    """Pydantic accepts NaN floats by default — we must reject explicitly,
    otherwise NaN bounds silently pass every comparison and corrupt the
    sampler. Covers low / high / step in one parametrize block."""
    with pytest.raises(ValidationError, match="must be finite"):
        FloatParam(type="float", **kwargs)


def test_validate_rejects_non_divisible_grid_float(tmp_path: Path) -> None:
    """low=0, high=1, step=0.6 would generate [0, 0.6, 1.2] -- 1.2 > high."""
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
            sampler: { type: grid }
            search_space:
              x: { type: float, low: 0.0, high: 1.0, step: 0.6 }
        """,
    )
    with pytest.raises(ValidationError, match="must be an integer"):
        load_experiment(p)


def test_validate_accepts_divisible_grid_float(tmp_path: Path) -> None:
    """low=0, high=1, step=0.25 -> exactly [0, 0.25, 0.5, 0.75, 1.0]."""
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
            n_trials: 5
            sampler: { type: grid }
            search_space:
              x: { type: float, low: 0.0, high: 1.0, step: 0.25 }
        """,
    )
    load_experiment(p)  # must not raise


def test_grid_sampler_requires_full_matrix_by_default(tmp_path: Path) -> None:
    """Grid phases must run every combination unless explicitly allowed partial."""
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
            search_space:
              x: { type: categorical, choices: [1, 2, 3] }
        """,
    )
    with pytest.raises(ValidationError, match="grid sampler has 3 combinations"):
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


def test_key_parts_handles_dots_and_empties() -> None:
    """Splitter normalizes leading/trailing/double dots so collision detection
    is order-insensitive."""
    assert _key_parts("a") == ("a",)
    assert _key_parts("a.b.c") == ("a", "b", "c")
    assert _key_parts(".a") == ("a",)
    assert _key_parts("a..b") == ("a", "b")


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        # B5.a — direct prefix pair
        ({"model", "model.depth"}, [("model", "model.depth")]),
        # B5.b — deeper prefix chain
        ({"a.b.c", "a.b.c.d"}, [("a.b.c", "a.b.c.d")]),
        # B5.c — siblings under a shared prefix are NOT collisions
        ({"model.depth", "model.depths"}, []),
        ({"a.x", "a.y"}, []),
        # B5.d — fully disjoint keys
        ({"foo", "bar", "baz"}, []),
    ],
)
def test_find_prefix_collisions(keys: set[str], expected: list[tuple[str, str]]) -> None:
    """The detector must flag a parent/child pair on the *same* dotted path
    (e.g. ``model`` set as scalar AND ``model.depth`` set as nested), and must
    not flag siblings sharing a parent or fully unrelated keys."""
    assert _find_prefix_collisions(keys) == expected


def test_rejects_local_dotted_prefix_collision() -> None:
    """Same phase: `model` fixed and `model.depth` sampled => render-time corruption."""
    with pytest.raises(ValueError, match="namespace collision"):
        Experiment(
            experiment="t",
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            phases=[
                Phase(  # type: ignore[arg-type]
                    name="p",
                    n_trials=1,
                    fixed_overrides={"model": "llama"},
                    search_space={
                        "model.depth": IntParam(type="int", low=8, high=32),
                    },
                )
            ],
        )


def test_rejects_inherited_dotted_prefix_collision() -> None:
    """Parent locks `optimizer`, child samples `optimizer.lr` => collision."""
    with pytest.raises(ValueError, match="namespace collision"):
        Experiment(
            experiment="t",
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            phases=[
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
                    search_space={
                        "optimizer.lr": FloatParam(type="float", low=1e-4, high=1e-2),
                    },
                ),
            ],
        )


def test_rejects_inherited_dotted_prefix_collision_other_direction() -> None:
    """Parent locks `optimizer.lr`, child fixes `optimizer` (parent's key is now a sub-key)."""
    with pytest.raises(ValueError, match="namespace collision"):
        Experiment(
            experiment="t",
            trial_command="echo {overrides}",
            metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
            phases=[
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


def test_hydra_phase_with_search_space_requires_overrides_placeholder() -> None:
    """A hydra trial_command without ``{overrides}`` would silently no-op
    every sweep — Optuna samples 20 different LRs, the trainer sees zero.
    """
    with pytest.raises(ValueError, match=r"does not reference \{overrides\}"):
        _exp_with_template(
            "python train.py --out {trial_dir}/result.json",
            override_format="hydra",
            search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
        )


def test_argparse_phase_with_search_space_requires_overrides_placeholder() -> None:
    with pytest.raises(ValueError, match=r"does not reference \{overrides\}"):
        _exp_with_template(
            "python train.py --out {trial_dir}/result.json",
            override_format="argparse",
            search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
        )


def test_argparse_phase_with_fixed_overrides_requires_overrides_placeholder() -> None:
    """Fixed overrides count as overrides — same silent-no-op risk as
    sampled params if the trial_command never references them.
    """
    with pytest.raises(ValueError, match=r"does not reference \{overrides\}"):
        _exp_with_template(
            "python train.py --out {trial_dir}/result.json",
            override_format="argparse",
            fixed_overrides={"lr": 1e-3},
        )


def test_escaped_overrides_does_not_count_as_real_placeholder() -> None:
    """``{{overrides}}`` renders as the *literal* string ``{overrides}``;
    the real ``str.format`` field set is empty. Pre-v0.5.7 the substring
    check was fooled and accepted this as referencing the placeholder.
    """
    with pytest.raises(ValueError, match=r"does not reference \{overrides\}"):
        _exp_with_template(
            "python train.py --out {trial_dir} '{{overrides}}'",
            override_format="hydra",
            search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
        )


def test_escaped_overrides_path_does_not_count_for_json_file() -> None:
    """Same trick, json_file branch: ``{{overrides_path}}`` is a literal,
    not a real placeholder.
    """
    with pytest.raises(ValueError, match=r"does not reference \{overrides_path\}"):
        _exp_with_template(
            "python train.py '{{overrides_path}}'",
            override_format="json_file",
            search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
        )


def test_field_format_specs_still_count_as_referencing_overrides() -> None:
    """Real format specs like ``{overrides!s}`` and ``{overrides:>10}`` are
    legitimate references — the validator must extract the *root* field
    name regardless of conversion or format spec.
    """
    # No exception expected.
    _exp_with_template(
        "python train.py {overrides!s}",
        override_format="hydra",
        search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
    )
    _exp_with_template(
        "python train.py '{overrides:>1}'",
        override_format="argparse",
        search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
    )


def test_constant_trial_command_accepted_when_phase_has_no_overrides() -> None:
    """A phase with empty search_space, no fixed overrides, no inherited
    keys is legitimate (variance/seed sweeps run the same configuration
    repeatedly). Pre-v0.5.7 the validator was silent here too; v0.5.7 must
    not over-tighten.
    """
    # No exception expected.
    _exp_with_template(
        "python train.py --out {trial_dir}/result.json",
        override_format="hydra",
        search_space={},
        fixed_overrides={},
    )


def test_trial_command_unknown_placeholder_rejected() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        make_experiment(trial_command="echo {trail_dir} {overrides}", n_trials=1)


def test_trial_command_valid_template_accepted() -> None:
    """Sanity: a normal template passes."""
    make_experiment(trial_command="echo {trial_dir} {overrides}", n_trials=1)
    make_experiment(
        trial_command="python train.py --out {trial_dir}/result.json {overrides}", n_trials=1
    )


def test_json_file_without_overrides_path_rejected() -> None:
    """`json_file` mode silently no-ops the override JSON if {overrides_path} is missing."""
    with pytest.raises(ValueError, match=r"\{overrides_path\}"):
        make_experiment(
            override_format="json_file",
            trial_command="python train.py --out {trial_dir}/result.json {overrides}",
            n_trials=1,
        )


def test_json_file_with_overrides_path_accepted() -> None:
    make_experiment(
        override_format="json_file",
        trial_command="python train.py --out {trial_dir}/result.json --cfg {overrides_path}",
        n_trials=1,
    )


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


@pytest.mark.parametrize(
    "bad_key",
    [
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
    ],
)
def test_search_space_rejects_malformed_keys(bad_key: str) -> None:
    """Hydra/argparse can't render an override of an empty or
    whitespace-bearing key correctly. We reject these at config-load.
    """
    with pytest.raises(ValidationError, match=r"override key|empty|whitespace|invalid"):
        Phase(
            name="p",
            n_trials=1,
            search_space={bad_key: FloatParam(type="float", low=0.0, high=1.0)},
        )


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        ".lr",
        "lr.",
        "model..depth",
        " lr",
        "lr ",
    ],
)
def test_fixed_overrides_rejects_malformed_keys(bad_key: str) -> None:
    with pytest.raises(ValidationError, match=r"override key|empty|whitespace|invalid"):
        Phase(
            name="p",
            n_trials=1,
            search_space={"x": FloatParam(type="float", low=0.0, high=1.0)},
            fixed_overrides={bad_key: 1},
        )


@pytest.mark.parametrize(
    "good_key",
    [
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
    ],
)
def test_search_space_accepts_well_formed_keys(good_key: str) -> None:
    """Sanity: legitimate hydra-style keys must not trigger the validator.
    Without these passing, the validator would be unusably strict.
    """
    phase = Phase(
        name="p",
        n_trials=1,
        search_space={good_key: FloatParam(type="float", low=0.0, high=1.0)},
    )
    assert good_key in phase.search_space


@pytest.mark.parametrize(
    "weird_segment_char",
    [
        "lr$",  # shell-special
        "lr*",
        "lr?",
        "lr|x",
        "lr;x",
        "lr/x",  # path separator — could escape namespacing
        "lr\\x",
    ],
)
def test_search_space_rejects_shell_unsafe_segment_chars(weird_segment_char: str) -> None:
    """Shell-special characters in override keys would render to argv
    fragments that the shell could interpret. Reject at config-load.
    """
    with pytest.raises(ValidationError, match=r"invalid characters"):
        Phase(
            name="p",
            n_trials=1,
            search_space={weird_segment_char: FloatParam(type="float", low=0.0, high=1.0)},
        )


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
