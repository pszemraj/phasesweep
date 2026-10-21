"""Search-space and override validation: param-type bounds, categorical scalar-only, dotted-prefix collisions, override-key shell safety, sampler/search-space compatibility, and trial-command template placeholder enforcement."""

from __future__ import annotations

import sys
import warnings

import optuna
import pytest
from optuna.exceptions import ExperimentalWarning
from pydantic import ValidationError

from phasesweep.config.common import _find_prefix_collisions, check_bounds, is_sha256_hex
from phasesweep.config.models import Constraint, Phase
from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Repeats,
    Sampler,
    _placeholder_values_for,
    grid_search_space,
    sampler_search_space,
)
from phasesweep.engine.optuna import _build_sampler, _suggest
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import make_experiment


@pytest.mark.parametrize(
    "key", ["data_seed", "random_state", "trainer.data_seed", "model.random_state"]
)
def test_seed_aliases_cannot_be_optimized_as_configuration(key: str) -> None:
    """Common trainer seed names belong in fixed inputs or the repeat axis."""
    with pytest.raises(ValidationError, match="seed keys cannot be searched"):
        Phase(
            name="p",
            n_trials=2,
            search_space={key: IntParam(type="int", low=0, high=1)},
        )


@pytest.mark.parametrize(("n_jobs", "constant_liar"), [(4, True), (1, False)])
def test_tpe_sampler_constant_liar_policy(n_jobs: int, constant_liar: bool) -> None:
    """TPE enables constant_liar only for parallel optimization."""
    cfg = Sampler(type="tpe", seed=0)
    space = {"x": CategoricalParam(type="categorical", choices=[1, 2, 3])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = _build_sampler(cfg, space, n_jobs=n_jobs)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._constant_liar is constant_liar
    assert not [warning for warning in caught if issubclass(warning.category, ExperimentalWarning)]


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


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Phase, {"name": "p", "n_trials": True}),
        (FloatParam, {"type": "float", "low": False, "high": 1.0}),
        (IntParam, {"type": "int", "low": 0, "high": True}),
    ],
)
def test_numeric_config_fields_reject_yaml_booleans(
    model: type, payload: dict[str, object]
) -> None:
    """Boolean YAML scalars must not become numeric sweep controls."""
    with pytest.raises(ValidationError, match="numbers, not booleans"):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("low", "high"),
    [(-(2**53) - 1, 0), (0, 2**53 + 1)],
)
def test_int_param_rejects_bounds_optuna_cannot_round_trip(low: int, high: int) -> None:
    """Numeric integer domains must survive Optuna's float representation."""
    with pytest.raises(ValidationError, match="round-trip exactly through Optuna"):
        IntParam(type="int", low=low, high=high)


def test_int_param_accepts_exact_boundaries_and_large_categorical_choices() -> None:
    """The exact numeric boundary and categorical 64-bit values remain available."""
    assert IntParam(type="int", low=-(2**53), high=2**53).model_dump() == {
        "type": "int",
        "low": -(2**53),
        "high": 2**53,
        "log": False,
        "step": 1,
    }
    choices = [2**63 - 2, 2**63 - 1]
    assert CategoricalParam(type="categorical", choices=choices).choices == choices


def test_check_bounds_rejects_non_finite_values():
    """NaN/inf are always out of bounds."""
    assert check_bounds(50.0, min_value=0.0, max_value=100.0) is True
    assert check_bounds(float("nan"), min_value=0.0, max_value=100.0) is False
    assert check_bounds(float("inf"), min_value=0.0, max_value=100.0) is False
    assert check_bounds(float("-inf"), min_value=0.0, max_value=100.0) is False


def test_sampler_rejects_negative_n_startup_trials():
    with pytest.raises(ValidationError):
        Sampler(type="tpe", n_startup_trials=-1)


@pytest.mark.parametrize("sampler_type", ["random", "grid", "cmaes"])
def test_sampler_rejects_non_default_n_startup_trials_off_tpe(sampler_type: str) -> None:
    """A non-default n_startup_trials on a sampler that ignores it is refused.

    Only TPE reads n_startup_trials, yet it is part of the semantic phase
    fingerprint, so a value that changes nothing would still split studies.
    """
    with pytest.raises(ValidationError, match="n_startup_trials"):
        Sampler(type=sampler_type, n_startup_trials=999)


@pytest.mark.parametrize(
    ("sampler_type", "n_startup_trials"),
    [
        pytest.param("tpe", 999, id="tpe_non_default"),
        pytest.param("random", 10, id="random_default"),
        pytest.param("grid", 10, id="grid_default"),
        pytest.param("cmaes", 10, id="cmaes_default"),
    ],
)
def test_sampler_accepts_n_startup_trials_it_can_honor(
    sampler_type: str, n_startup_trials: int
) -> None:
    """TPE accepts any n_startup_trials; every other sampler accepts only the default.

    TPE is the one sampler type that actually reads n_startup_trials. Explicitly
    spelling out the default on a non-tpe sampler is fine too: this is checked by
    value, not by whether the field was explicitly set, because config snapshot
    fixtures serialize every field, including n_startup_trials: 10 on
    sampler.type: random, and must keep loading.
    """
    sampler = Sampler(type=sampler_type, n_startup_trials=n_startup_trials)
    assert sampler.n_startup_trials == n_startup_trials


def test_validate_rejects_incompatible_phase_settings() -> None:
    """Validation rejects incompatible sampler and parameter declarations."""
    with pytest.raises(ValidationError, match="cmaes.*does not support categorical"):
        make_experiment(
            n_trials=1,
            sampler=Sampler(type="cmaes"),
            search_space={"model": CategoricalParam(type="categorical", choices=["a", "b"])},
        )


def test_grid_config_validation_does_not_enumerate_parameter_lists(monkeypatch) -> None:
    """Validation counts the full matrix without calling the value-list builder."""

    def refuse_enumeration(*args, **kwargs):
        pytest.fail("config validation materialized the grid")

    monkeypatch.setattr("phasesweep.config.search.grid_search_space", refuse_enumeration)
    experiment = make_experiment(
        n_trials=12,
        sampler=Sampler(type="grid"),
        search_space={
            "width": IntParam(type="int", low=1, high=3),
            "scale": FloatParam(type="float", low=0.0, high=0.5, step=0.5),
            "model": CategoricalParam(type="categorical", choices=["a", "b"]),
        },
    )
    assert experiment.phases[0].n_trials == 12


def test_validate_rejects_invalid_grid_configs() -> None:
    """Grid sampler configs must be exactly enumerable and complete by default."""
    cases = [
        (
            "log_float",
            {"lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
            1,
            "grid.*log-scale float",
        ),
        (
            "log_int",
            {"n": {"type": "int", "low": 1, "high": 1024, "log": True}},
            1,
            "grid.*log-scale int",
        ),
        (
            "float_no_step",
            {"x": {"type": "float", "low": 0.0, "high": 1.0}},
            1,
            "grid sampler requires 'step'",
        ),
        (
            "rounded_below_low",
            {"eps": {"type": "float", "low": 1.0e-13, "high": 1.1e-12, "step": 1.0e-12}},
            2,
            "cannot preserve its configured bounds and step",
        ),
        (
            "rounded_above_high",
            {"eps": {"type": "float", "low": 6.0e-13, "high": 1.6e-12, "step": 1.0e-12}},
            2,
            "cannot preserve its configured bounds and step",
        ),
        (
            "partial_matrix",
            {"x": {"type": "categorical", "choices": [1, 2, 3]}},
            2,
            "grid sampler has 3 combinations",
        ),
        (
            "over_budget",
            {"x": {"type": "categorical", "choices": [1, 2]}},
            3,
            "n_trials=3 exceeds the grid cardinality 2",
        ),
    ]

    for _case, search_space, n_trials, match in cases:
        with pytest.raises(ValidationError, match=match):
            make_experiment(
                n_trials=n_trials, sampler=Sampler(type="grid"), search_space=search_space
            )


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


def test_validate_accepts_divisible_grid_float() -> None:
    """low=0, high=1, step=0.25 -> exactly [0, 0.25, 0.5, 0.75, 1.0]."""
    experiment = make_experiment(
        n_trials=5,
        sampler=Sampler(type="grid"),
        search_space={"x": FloatParam(type="float", low=0.0, high=1.0, step=0.25)},
    )
    assert grid_search_space(experiment.phases[0].search_space, phase_name="p") == {
        "x": [0.0, 0.25, 0.5, 0.75, 1.0]
    }


@pytest.mark.parametrize(
    "search_space",
    [
        pytest.param(
            {"x": IntParam(type="int", low=0, high=4_096)},
            id="int-range",
        ),
        pytest.param(
            {"x": FloatParam(type="float", low=0.0, high=4_096.0, step=1.0)},
            id="float-range",
        ),
        pytest.param(
            {
                "left": CategoricalParam(type="categorical", choices=list(range(65))),
                "right": CategoricalParam(type="categorical", choices=list(range(64))),
            },
            id="product-of-small-dimensions",
        ),
        pytest.param(
            {
                "wide": IntParam(type="int", low=0, high=4_096),
                "tiny": FloatParam(type="float", low=0.0, high=1.0e-12, step=1.0e-13),
            },
            id="cardinality-cap-precedes-float-collapse",
        ),
    ],
)
def test_grid_rejects_oversized_cardinality_before_materializing_values(
    search_space: dict[str, IntParam | FloatParam | CategoricalParam],
) -> None:
    """Grid validation bounds ranges and products before expanding values."""
    with pytest.raises(ValidationError, match="combinations.*maximum of 4,096"):
        make_experiment(
            phases=[
                Phase(
                    name="p",
                    n_trials=1,
                    sampler=Sampler(type="grid"),
                    search_space=search_space,
                )
            ]
        )


def test_grid_accepts_maximum_cardinality() -> None:
    """The configured grid ceiling itself remains a valid full matrix."""
    experiment = make_experiment(
        phases=[
            Phase(
                name="p",
                n_trials=4_096,
                sampler=Sampler(type="grid"),
                search_space={
                    "left": CategoricalParam(type="categorical", choices=list(range(64))),
                    "right": CategoricalParam(type="categorical", choices=list(range(64))),
                },
            )
        ]
    )
    assert grid_search_space(experiment.phases[0].search_space, phase_name="p") == {
        "left": list(range(64)),
        "right": list(range(64)),
    }


@pytest.mark.parametrize("sampler", ["random", "tpe", "cmaes", "grid"])
@pytest.mark.parametrize(
    "param",
    [
        {"type": "int", "low": 0, "high": 5, "step": 2},
        {"type": "float", "low": 0, "high": 1, "step": 0.3},
        {"type": "float", "low": -0.4, "high": 0.5, "step": 0.2},
        {"type": "float", "low": 0, "high": 100.00000001, "step": 1.0},
    ],
)
def test_off_lattice_endpoints_are_rejected_for_every_sampler(sampler, param) -> None:
    """An upper bound off the step lattice is refused at construction, whatever the sampler.

    The IntParam/FloatParam validators reject it before any sampler or grid code runs,
    so a close float ratio can never silently omit the configured upper bound.
    """
    with pytest.raises(ValidationError, match="evenly divid"):
        make_experiment(
            phases=[
                {
                    "name": "p",
                    "n_trials": 1,
                    "sampler": {"type": sampler, "seed": 0},
                    "search_space": {"x": param},
                }
            ]
        )


@pytest.mark.parametrize(
    "param",
    [
        IntParam(type="int", low=0, high=6, step=2),
        IntParam(type="int", low=-5, high=1, step=2),
        IntParam(type="int", low=3, high=3, step=7),
        FloatParam(type="float", low=0, high=1, step=0.1),
        FloatParam(type="float", low=-0.3, high=0.3, step=0.2),
        FloatParam(type="float", low=0.1, high=0.7, step=0.3),
        FloatParam(type="float", low=-0.2, high=-0.2, step=0.3),
        FloatParam(type="float", low=0, high=1e-12, step=1e-13),
        FloatParam(type="float", low=-0.7, high=1.13),
    ],
)
def test_non_grid_distributions_retain_configured_endpoints(param) -> None:
    experiment = make_experiment(
        phases=[
            Phase(
                name="p",
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": param},
            )
        ]
    )
    retained = experiment.phases[0].search_space["x"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distribution = (
            optuna.distributions.IntDistribution(retained.low, retained.high, step=retained.step)
            if isinstance(retained, IntParam)
            else optuna.distributions.FloatDistribution(
                retained.low, retained.high, step=retained.step
            )
        )
    assert distribution.high == param.high
    assert retained.model_dump() == param.model_dump()
    assert not caught


def test_validate_accepts_explicit_partial_grid() -> None:
    """Partial grid phases are allowed only when explicitly requested."""
    make_experiment(
        n_trials=2,
        sampler=Sampler(type="grid"),
        allow_partial_grid=True,
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 3])},
    )


@pytest.mark.parametrize("sampler_type", ["tpe", "random", "cmaes"])
def test_validate_rejects_partial_grid_switch_on_a_non_grid_sampler(sampler_type: str) -> None:
    """``allow_partial_grid`` is read only for grid, so another sampler would silently ignore it."""
    with pytest.raises(
        ValidationError,
        match=rf"allow_partial_grid: true applies only to sampler\.type='grid'.*{sampler_type!r}",
    ):
        make_experiment(
            n_trials=2,
            sampler=Sampler(type=sampler_type),
            allow_partial_grid=True,
        )


def test_validate_rejects_partial_grid_above_cardinality() -> None:
    """Partial-grid mode permits a subset, not an impossible oversized budget."""
    with pytest.raises(ValidationError, match="n_trials=4 exceeds the grid cardinality 3"):
        make_experiment(
            n_trials=4,
            sampler=Sampler(type="grid"),
            allow_partial_grid=True,
            search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 3])},
        )


@pytest.mark.parametrize(
    ("choices", "message"),
    [
        pytest.param([1, 1, 2], "must remain distinguishable", id="duplicate_ints"),
        pytest.param(
            ["a", "b", "a"],
            "at index 2 compares equal to 'a' at index 0",
            id="duplicate_strings",
        ),
        pytest.param([1, 1.0], "1.0 at index 1 compares equal to 1 at index 0", id="int_float"),
        pytest.param([1, True], "True at index 1 compares equal to 1 at index 0", id="int_bool"),
        pytest.param([0, False], "False at index 1 compares equal to 0 at index 0", id="zero_bool"),
        pytest.param(
            [0.0, -0.0], r"-0\.0 at index 1 compares equal to 0\.0 at index 0", id="signed_zero"
        ),
    ],
)
def test_categorical_rejects_equal_but_distinct_choices(
    choices: list[object], message: str
) -> None:
    """Optuna's ``==`` index lookup cannot tell equal choices apart once persisted.

    ``CategoricalDistribution`` records a sampled value as ``choices.index(value)``,
    so ``1``, ``1.0``, and ``True`` all persist as the first equal choice. The
    trainer would run one value while the published winner and every inherited
    override claimed another. A plain repeated choice is the same hazard and
    also inflates cardinality and skews sampling weight.
    """
    with pytest.raises(ValidationError, match=message):
        CategoricalParam(type="categorical", choices=choices)


def test_categorical_config_error_uses_only_the_selected_union_branch() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_experiment(
            n_trials=2,
            sampler=Sampler(type="grid"),
            search_space={"x": {"type": "categorical", "choices": [1, 1.0]}},
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"][-2:] == ("categorical", "choices")
    assert "must remain distinguishable" in errors[0]["msg"]


def test_grid_float_rejects_post_canonicalization_collapse() -> None:
    """Sub-1e-12 steps collapse onto the same rounded value, so cardinality would lie."""
    with pytest.raises(ValidationError, match="collapses to 2 unique value"):
        make_experiment(
            n_trials=11,
            sampler=Sampler(type="grid"),
            search_space={"x": FloatParam(type="float", low=0.0, high=1.0e-12, step=1.0e-13)},
        )


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


def test_is_sha256_hex() -> None:
    """Only a ``str`` of exactly 64 lowercase hex characters is accepted."""
    valid = "a" * 64
    cases = [
        ("valid_lowercase_hex", valid, True),
        ("valid_digits_only", "0" * 64, True),
        ("uppercase_rejected", "A" * 64, False),
        ("too_short", valid[:-1], False),
        ("too_long", valid + "a", False),
        ("non_hex_character", "g" * 64, False),
        ("empty_string", "", False),
        ("not_a_string_int", 0, False),
        ("not_a_string_none", None, False),
        ("not_a_string_bytes", b"a" * 64, False),
    ]

    for case, value, expected in cases:
        assert is_sha256_hex(value) is expected, case


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
            make_experiment(experiment=f"t_{case}", phases=phases)


@pytest.mark.parametrize(
    ("template", "override_format", "search_space", "fixed_overrides", "match"),
    [
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


@pytest.mark.parametrize(
    ("template", "match"),
    [
        pytest.param(
            "echo {trail_dir} {overrides}", "unknown placeholder", id="unknown_placeholder"
        ),
        pytest.param("echo {trial_dir", "trial_command failed to render", id="unbalanced_brace"),
        pytest.param(
            "echo {trial_dir.foo} {overrides}",
            r"attribute or index access; found \{trial_dir\.foo\}",
            id="attribute_access",
        ),
        pytest.param(
            "echo {trial_dir[0]} {overrides}",
            r"attribute or index access; found \{trial_dir\[0\]\}",
            id="index_access",
        ),
        pytest.param(
            "echo {trial_dir:{phase.foo}} {overrides}",
            r"attribute or index access; found \{phase\.foo\}",
            id="access_inside_nested_format_spec",
        ),
    ],
)
def test_trial_command_malformed_template_rejected(template: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_experiment(trial_command=template, n_trials=1)


def test_invalid_trainer_config_is_not_misreported_as_a_command_template_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        make_experiment(
            trial_command="python train.py --config {config_path}",
            override_format="yaml_file",
            trainer_config={"optimizer": {"learning_rate": float("nan")}},
            n_trials=1,
        )

    message = str(exc_info.value)
    assert "trainer_config and composed overrides are invalid" in message
    assert "unbalanced braces" not in message


def test_trial_command_accepts_supported_templates() -> None:
    """Accepted templates include a format spec on {overrides} and plain ones naming {trial_dir} and {overrides}."""
    cases = [
        (
            "field_format_spec",
            lambda: make_experiment(
                trial_command="python train.py '{overrides:>1}'",
                n_trials=1,
                search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True)},
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
    ]

    for _case, build in cases:
        build()


def test_override_keys_reject_malformed_and_shell_unsafe_values() -> None:
    """Override keys with empty or dot-edge segments, whitespace, shell-special characters, or path separators are rejected at config load."""
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
    """Sanity: legitimate dotted keys must not trigger the validator.
    Without these passing, the validator would be unusably strict.
    """
    good_keys = [
        "lr",
        "model.depth",
        "trainer.run.dir",
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
    """CMA-ES availability is part of config validation, not mid-run sampler build.

    ``cmaes`` is a declared hard dependency, but a declared dependency is not
    an enforced one. If the preflight is dropped, a broken environment passes
    ``phasesweep validate`` and fails inside ``_build_sampler`` instead —
    after the generation is claimed and earlier phases have burned GPU time.
    """
    # A None entry makes `import cmaes` raise ImportError without touching
    # any other import, and monkeypatch restores the real module after.
    monkeypatch.setitem(sys.modules, "cmaes", None)

    with pytest.raises(ValidationError, match=r"cmaes.*not installed") as excinfo:
        make_experiment(
            sampler=Sampler(type="cmaes", seed=0),
            search_space={"x": IntParam(type="int", low=0, high=10)},
        )

    assert "pip install cmaes" in str(excinfo.value)


def test_cmaes_phase_loads_when_package_present() -> None:
    """A CMA-ES phase loads when the base dependency is importable."""
    import cmaes  # noqa: F401

    exp = make_experiment(
        sampler=Sampler(type="cmaes", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
    )
    assert exp.phases[0].sampler.type == "cmaes"


@pytest.mark.parametrize(
    ("override_format", "template"),
    [
        ("yaml_file", "python train.py --config {config_path}"),
        ("argparse", "python train.py {overrides}"),
        ("hydra", "python train.py {overrides}"),
        ("json_file", "python train.py {overrides_path}"),
    ],
)
def test_extreme_float_bounds_load_under_every_override_format(
    override_format: str, template: str
) -> None:
    """A legal near-float64-max FloatParam loads under every override format.

    Config load renders the trial command with a placeholder value for each
    parameter, and every format refuses a non-finite value, so the
    placeholder for ``low=1e308, high=1.1e308`` must stay finite.
    """
    exp = make_experiment(
        trial_command=template,
        override_format=override_format,
        search_space={"lr": FloatParam(type="float", low=1e308, high=1.1e308)},
        n_trials=1,
    )
    assert exp.override_format == override_format


def test_placeholder_values_for_stepped_int_is_on_the_lattice() -> None:
    """The placeholder for a stepped IntParam is a point on its step lattice."""
    param = IntParam(type="int", low=0, high=6, step=2)
    value = _placeholder_values_for({"x": param})["x"]
    assert value == param.low
    assert (value - param.low) % param.step == 0


def test_placeholder_values_for_categorical_is_first_choice() -> None:
    """The placeholder for a categorical param is its first listed choice."""
    param = CategoricalParam(type="categorical", choices=["b", "a", "c"])
    assert _placeholder_values_for({"x": param})["x"] == "b"


def test_grid_repeat_axis_visits_each_configuration_once_per_value() -> None:
    """The grid treats the repeat key as one more axis, so every cell runs exactly once."""
    phase = Phase(
        name="p",
        n_trials=6,
        sampler=Sampler(type="grid", seed=3),
        search_space={"x": CategoricalParam(type="categorical", choices=["a", "b"])},
        repeats=Repeats(key="seed", values=[0, 1, 2]),
    )
    space = sampler_search_space(phase)
    assert list(space) == ["x", "seed"]
    study = optuna.create_study(sampler=_build_sampler(phase.sampler, space, repeats=phase.repeats))
    study.optimize(
        lambda trial: float(sum(hash(_suggest(trial, k, p)) % 7 for k, p in space.items())),
        n_trials=phase.n_trials,
    )
    cells = [(trial.params["x"], trial.params["seed"]) for trial in study.trials]
    assert sorted(cells) == [(x, seed) for x in ("a", "b") for seed in (0, 1, 2)]


def test_repeat_values_need_distinct_cli_wire_forms() -> None:
    phase = Phase(
        name="p",
        n_trials=2,
        sampler=Sampler(type="grid"),
        repeats=Repeats(key="seed", values=[1, "1"]),
    )
    with pytest.raises(
        ValidationError,
        match=r"repeats.key 'seed' has choices 1 at index 0 and '1' at index 1, which both "
        r"render as '1' under override_format='argparse'",
    ):
        make_experiment(phases=[phase])
    assert (
        make_experiment(
            phases=[phase],
            override_format="yaml_file",
            trial_command="train --config {config_path}",
        )
        .phases[0]
        .repeats
    )


def test_repeat_axis_counts_toward_grid_cardinality_limit() -> None:
    """A grid that fits the ceiling alone is refused once the repeat axis multiplies it."""
    search_space = {"left": CategoricalParam(type="categorical", choices=list(range(64)))}
    search_space["right"] = CategoricalParam(type="categorical", choices=list(range(64)))
    with pytest.raises(ValidationError, match="combinations.*maximum of 4,096"):
        make_experiment(
            phases=[
                Phase(
                    name="p",
                    n_trials=8_192,
                    sampler=Sampler(type="grid"),
                    search_space=search_space,
                    repeats=Repeats(key="seed", values=[0, 1]),
                )
            ]
        )
