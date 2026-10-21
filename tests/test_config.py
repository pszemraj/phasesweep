from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phasesweep import load_experiment
from phasesweep.config.io import ConfigError
from phasesweep.config.models import (
    Constraint,
    ExecutionContext,
    Experiment,
    Metric,
    Observation,
    Phase,
    experiment_warnings,
)
from phasesweep.config.search import Sampler, sampler_capability_line
from phasesweep.evidence.models import (
    JsonEnvelopeExtractor,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from phasesweep.runtime.json import JsonObject, JsonValue
from phasesweep.tracking.models import MlflowSettings
from tests.conftest import make_experiment, write_yaml


@pytest.mark.parametrize(
    ("field", "model", "extra", "message"),
    [
        (
            "base_url",
            WandbExtractor,
            {"type": "wandb", "entity": "e", "project": "p", "metric_key": "loss"},
            "W&B base_url requires an HTTP(S) endpoint without credentials, query, or fragment.",
        ),
        (
            "tracking_uri",
            MlflowSettings,
            {"experiment_name": "existing"},
            "MLflow tracking_uri requires HTTP(S), databricks, or databricks://profile (no spaces in the profile name) without embedded credentials, query, or fragment.",
        ),
    ],
    ids=["wandb", "mlflow"],
)
@pytest.mark.parametrize(
    ("endpoint", "wandb_expected", "mlflow_expected"),
    [
        ("HTTPS://EXAMPLE.TEST/Path///", "https://example.test/Path", "https://example.test/Path"),
        ("http://@EXAMPLE.TEST/path/", None, "http://@example.test/path"),
        ("http://:@EXAMPLE.TEST/path/", None, "http://:@example.test/path"),
        ("http://user:password@example.test", None, None),
        ("ftp://example.test", None, None),
        ("http://", None, None),
        ("http://example.test?key=value", None, None),
        ("http://example.test#part", None, None),
        (
            "https://example.test:service/",
            "https://example.test:service",
            "https://example.test:service",
        ),
    ],
    ids=[
        "normalization",
        "empty-user",
        "empty-user-password",
        "credentials",
        "scheme",
        "missing-host",
        "query",
        "fragment",
        "unparsed-port",
    ],
)
def test_remote_http_endpoint_contracts(
    field, model, extra, message, endpoint, wandb_expected, mlflow_expected
):
    expected = wandb_expected if field == "base_url" else mlflow_expected
    arguments = {field: endpoint, **extra}
    if expected is None:
        with pytest.raises(ValidationError) as caught:
            model(**arguments)
        assert str(caught.value.errors()[0]["ctx"]["error"]) == message
    else:
        assert getattr(model(**arguments), field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", "https://another.test"),
        ("entity", "another"),
        ("project", "another"),
        ("poll_seconds", 3),
        ("timeout_seconds", 5),
    ],
)
def test_wandb_consumers_must_agree_per_phase(field, value):
    objective = WandbExtractor(type="wandb", entity="e", project="p", metric_key="eval/loss")
    conflicting = objective.model_copy(update={field: value, "metric_key": "memory"})
    with pytest.raises(ValueError, match=rf"constraints\[0\].extractor.{field}.*metric.extractor"):
        make_experiment(
            metric=Metric(extractor=objective),
            constraints=[Constraint(name="memory", extractor=conflicting, max=5)],
        )


def test_evaluation_axis_defaults_and_misplacements_are_rejected():
    envelope = JsonEnvelopeExtractor(
        type="json_envelope", objective_name="loss", split="validation", policy="final"
    )
    wandb = WandbExtractor(type="wandb", entity="e", project="p", metric_key="eval/loss.min")
    assert (envelope.evaluation_axis, wandb.evaluation_axis) == ("step", "_step")
    for axis in ("eval/loss.min", "eval/loss"):
        with pytest.raises(ValueError, match="names the metric's own series"):
            wandb.model_validate({**wandb.model_dump(), "evaluation_axis": axis})
    for extractor in (
        envelope.model_copy(update={"evaluation_axis": "tokens"}),
        wandb.model_copy(update={"evaluation_axis": "eval/step"}),
    ):
        with pytest.raises(ValueError, match="only the metric extractor's evaluation_point"):
            Constraint.model_validate(
                {"name": "memory", "extractor": extractor.model_dump(), "max": 5}
            )


def test_evaluation_axis_is_part_of_study_identity():
    from phasesweep.engine.fingerprints import _phase_fingerprint

    def fingerprint(axis):
        extractor = WandbExtractor(
            type="wandb", entity="e", project="p", metric_key="eval/loss", evaluation_axis=axis
        )
        experiment = make_experiment(metric=Metric(extractor=extractor))
        return _phase_fingerprint(experiment, experiment.phases[0], {})

    assert fingerprint("_step") != fingerprint("eval/step")


@pytest.mark.parametrize("backend", ["wandb", "mlflow"])
@pytest.mark.parametrize("source", ["trainer_config", "fixed_overrides"])
@pytest.mark.parametrize("key", ["load_best_model_at_end", "training.load_best_model_at_end"])
def test_best_model_reload_warns_about_remote_evaluation_position(backend, source, key):
    settings = {key: True}
    experiment = make_experiment(
        override_format="hydra",
        trainer_config=settings if source == "trainer_config" else {},
        phases=[
            Phase(
                name="p",
                n_trials=1,
                fixed_overrides=settings if source == "fixed_overrides" else {},
            )
        ],
    ).model_dump(mode="json")
    experiment["metric"]["extractor"] = (
        {
            "type": "wandb",
            "entity": "e",
            "project": "p",
            "metric_key": "eval/loss",
            "evaluation_axis": "train/global_step",
        }
        if backend == "wandb"
        else {"type": "mlflow", "metric_key": "eval/loss", "select": "last"}
    )
    if backend == "mlflow":
        experiment["mlflow"] = {"tracking_uri": "databricks", "experiment_id": "1"}

    warnings = experiment_warnings(Experiment.model_validate(experiment))

    assert len(warnings) == 1
    warning = warnings[0]
    assert "load_best_model_at_end=True" in warning
    assert "scored best-checkpoint weights may differ from the terminal training step" in warning
    assert "does not verify checkpoint identity" in warning
    assert all(
        field in warning
        for field in (
            "json_envelope",
            "policy",
            "checkpoint",
            "step",
            "evaluation.progress",
            "evaluation_axis",
        )
    )


@pytest.mark.parametrize("value", [None, False, "true", 1])
def test_best_model_warning_requires_explicit_true(value):
    experiment = make_experiment(
        override_format="hydra",
        trainer_config={} if value is None else {"load_best_model_at_end": value},
        metric=Metric(
            extractor=WandbExtractor(type="wandb", entity="e", project="p", metric_key="eval/loss")
        ),
    )

    assert experiment_warnings(experiment) == []


def test_best_model_envelope_does_not_warn_about_remote_history():
    experiment = make_experiment(
        override_format="hydra",
        trainer_config={"training": {"load_best_model_at_end": True}},
        metric=Metric(
            extractor=JsonEnvelopeExtractor(
                type="json_envelope", objective_name="loss", split="validation", policy="best"
            )
        ),
    )

    assert experiment_warnings(experiment) == []


def test_wandb_query_normalizes_targets_and_collects_exact_keys():
    from phasesweep.config.models import _wandb_query

    objective = WandbExtractor(
        type="wandb",
        base_url="https://API.WANDB.AI/",
        entity="e",
        project="p",
        metric_key="eval/loss",
    )
    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="e", project="p", keys=["complete"]
    )
    experiment = make_experiment(metric=Metric(extractor=objective), gates=[gate])
    query = _wandb_query(experiment, experiment.phases[0].gates)
    assert query.source.base_url == "https://api.wandb.ai"
    assert query.numeric_keys == ("eval/loss",)
    assert query.presence_keys == ("complete",)


def test_wandb_gate_query_identity_ignores_gate_list_order():
    from phasesweep.config.models import _wandb_query

    first = WandbSummaryRequiredGate(
        type="wandb_summary_required", entity="e", project="p", keys=["complete"]
    )
    second = first.model_copy(update={"keys": ["artifact_ready"]})
    experiment = make_experiment(gates=[first, second])

    ordered = _wandb_query(experiment, [first, second])
    reordered = _wandb_query(experiment, [second, first])

    assert ordered is not None
    assert reordered is not None
    assert dict(ordered.gate_keys) == dict(reordered.gate_keys)


def test_wandb_gate_query_identity_uses_only_target_and_required_keys():
    from phasesweep.config.models import _wandb_query

    gate = WandbSummaryRequiredGate(
        type="wandb_summary_required",
        base_url="https://API.WANDB.AI/",
        entity="e",
        project="p",
        keys=["complete", "artifact_ready"],
        poll_seconds=2,
        timeout_seconds=120,
    )
    operational_change = gate.model_copy(
        update={
            "keys": ["artifact_ready", "complete"],
            "poll_seconds": 5,
            "timeout_seconds": 300,
        }
    )
    target_change = gate.model_copy(update={"project": "another"})
    keys_change = gate.model_copy(update={"keys": ["complete"]})

    def gate_keys(candidate: WandbSummaryRequiredGate) -> tuple[tuple[str, tuple[str, ...]], ...]:
        experiment = make_experiment(gates=[candidate])
        query = _wandb_query(experiment, experiment.phases[0].gates)
        assert query is not None
        return query.gate_keys

    assert gate_keys(operational_change) == gate_keys(gate)
    assert gate_keys(target_change) != gate_keys(gate)
    assert gate_keys(keys_change) != gate_keys(gate)


def test_wandb_environment_identity_can_bind_offline_mode_without_launch():
    from phasesweep.config.models import _wandb_query
    from phasesweep.evidence.models import compose_wandb_environment

    experiment = make_experiment(
        metric=Metric(
            extractor=WandbExtractor(type="wandb", entity="e", project="p", metric_key="loss")
        )
    )
    query = _wandb_query(experiment, experiment.phases[0].gates)
    assert query is not None

    with pytest.raises(ValueError, match="requires online"):
        compose_wandb_environment(query, {}, {"WANDB_MODE": "offline"})

    environment = compose_wandb_environment(
        query, {}, {"WANDB_MODE": "offline"}, require_online=False
    )
    assert environment["WANDB_MODE"] == "offline"
    assert environment["WANDB_PROJECT"] == "p"


@pytest.mark.parametrize(
    "env",
    [
        {"WANDB_RUN_ID": "foreign"},
        {"WANDB_RESUME": "allow"},
        {"WANDB_PROJECT": "foreign"},
        {"WANDB_ENTITY": "foreign"},
        {"WANDB_BASE_URL": "https://foreign.test"},
        {"WANDB_MODE": "offline"},
        {"WANDB_MODE": "disabled"},
        {"WANDB_DISABLED": "true"},
    ],
)
def test_wandb_managed_conflicts_fail_during_config_validation(env):
    with pytest.raises(ValueError, match="W&B|WANDB"):
        make_experiment(
            env=env,
            metric=Metric(
                extractor=WandbExtractor(
                    type="wandb", entity="e", project="p", metric_key="eval/loss"
                )
            ),
        )


def test_phases_from_selects_reached_phases_by_name() -> None:
    experiment = make_experiment(
        phases=[
            Phase(name="base", n_trials=1, fixed_overrides={"model.depth": 8}),
            Phase(
                name="tune",
                n_trials=1,
                inherits=["base"],
                search_space={"learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3}},
            ),
            Phase(name="final", n_trials=1, inherits=["tune"], fixed_overrides={"seed": 0}),
        ]
    )

    # None reaches every phase from the start.
    assert [(index, phase.name) for index, phase in experiment.phases_from(None)] == [
        (0, "base"),
        (1, "tune"),
        (2, "final"),
    ]
    # The first phase's name reaches the same full sequence as None.
    assert [(index, phase.name) for index, phase in experiment.phases_from("base")] == [
        (0, "base"),
        (1, "tune"),
        (2, "final"),
    ]
    # A middle phase's name skips everything declared before it.
    assert [(index, phase.name) for index, phase in experiment.phases_from("tune")] == [
        (1, "tune"),
        (2, "final"),
    ]
    # An unknown name never becomes "reached", so nothing is yielded.
    assert list(experiment.phases_from("unknown")) == []


def test_phase_composition_accepts_a_same_origin_diamond() -> None:
    experiment = make_experiment(
        phases=[
            Phase(name="base", n_trials=1, fixed_overrides={"depth": 8}),
            Phase(name="left", n_trials=1, inherits=["base"], fixed_overrides={"left": 1}),
            Phase(name="right", n_trials=1, inherits=["base"], fixed_overrides={"right": 1}),
            Phase(name="join", n_trials=1, inherits=["left", "right"]),
        ]
    )

    assert experiment.phases[-1].inherits == ["left", "right"]


def test_phase_composition_accepts_child_fixed_resolution_as_new_origin() -> None:
    experiment = make_experiment(
        phases=[
            Phase(name="left", n_trials=1, fixed_overrides={"learning_rate": 1e-4}),
            Phase(name="right", n_trials=1, fixed_overrides={"learning_rate": 2e-4}),
            Phase(
                name="resolved",
                n_trials=1,
                inherits=["left", "right"],
                fixed_overrides={"learning_rate": 5e-4},
            ),
            Phase(name="branch_a", n_trials=1, inherits=["resolved"]),
            Phase(name="branch_b", n_trials=1, inherits=["resolved"]),
            Phase(name="join", n_trials=1, inherits=["branch_a", "branch_b"]),
        ]
    )

    assert experiment.phases[2].fixed_overrides == {"learning_rate": 5e-4}


def test_resampling_a_parent_fixed_key_is_valid() -> None:
    """A key every inherited origin only fixed may be searched by the child."""
    experiment = make_experiment(
        phases=[
            Phase(name="base", n_trials=1, fixed_overrides={"lr": 1e-4}),
            Phase(
                name="child",
                n_trials=1,
                inherits=["base"],
                search_space={"lr": {"type": "float", "low": 1e-5, "high": 1e-3}},
            ),
        ]
    )

    assert experiment.sampled_locked_keys("child") == frozenset()


@pytest.mark.parametrize(
    ("phases", "match"),
    [
        pytest.param(
            [
                Phase(name="left", n_trials=1, fixed_overrides={"lr": 1e-4}),
                Phase(name="right", n_trials=1, fixed_overrides={"lr": 2e-4}),
                Phase(name="join", n_trials=1, inherits=["left", "right"]),
            ],
            "inherits conflicting key",
            id="unresolved-origins",
        ),
        pytest.param(
            [
                Phase(
                    name="base",
                    n_trials=1,
                    search_space={"lr": {"type": "float", "low": 1e-5, "high": 1e-3}},
                ),
                Phase(
                    name="child",
                    n_trials=1,
                    inherits=["base"],
                    search_space={"lr": {"type": "float", "low": 1e-5, "high": 1e-3}},
                ),
            ],
            "re-samples key",
            id="inherited-resampling",
        ),
        pytest.param(
            [
                Phase(
                    name="invalid",
                    n_trials=1,
                    fixed_overrides={"lr": 1e-4},
                    search_space={"lr": {"type": "float", "low": 1e-5, "high": 1e-3}},
                )
            ],
            "both fixed_overrides and search_space",
            id="local-fixed-sampled",
        ),
    ],
)
def test_phase_composition_rejects_invalid_origins(phases: list[Phase], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        make_experiment(phases=phases)


_LOSS_METRIC = {
    "name": "loss",
    "goal": "minimize",
    "extractor": {
        "type": "json_envelope",
        "objective_name": "loss",
        "split": "test",
        "policy": "test",
    },
}
_BYTES_EXTRACTOR = {"type": "json", "path": "r.json", "key": "bytes"}


@pytest.mark.parametrize(
    ("options", "match"),
    [
        pytest.param(
            {
                "phases": [
                    Phase(name="a", n_trials=1, inherits=["b"]),
                    Phase(name="b", n_trials=1),
                ]
            },
            "inherits from 'b'",
            id="inherit-must-reference-prior-phase",
        ),
        pytest.param(
            {"constraints": [{"name": "bytes", "extractor": _BYTES_EXTRACTOR}]},
            "must define at least one",
            id="constraint-requires-bound",
        ),
        pytest.param(
            {
                "metric": _LOSS_METRIC,
                "constraints": [{"name": "loss", "max": 1, "extractor": _BYTES_EXTRACTOR}],
            },
            "distinct",
            id="metric-constraint-name-collision",
        ),
        pytest.param(
            {"metric": {**_LOSS_METRIC, "name": "goal"}},
            "reserved for winner direction metadata",
            id="metric-name-goal-is-reserved",
        ),
    ],
)
def test_invalid_experiment_relationships(options: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        make_experiment(**options)


@pytest.mark.parametrize("name", ["bad name with spaces", "bad\n"])
def test_phase_name_validation(name: str) -> None:
    with pytest.raises(ValidationError):
        Phase(name=name, n_trials=1, search_space={})


def test_phase_names_reject_casefold_equivalent_spellings() -> None:
    phases = [
        Phase(name="Foo", n_trials=1),
        Phase(name="foo", n_trials=1),
    ]

    with pytest.raises(
        ValidationError,
        match=r"Phase names 'Foo' and 'foo' must be unique case-insensitively",
    ):
        make_experiment(trial_command="echo", phases=phases)


def test_phase_names_preserve_authored_case_for_inherit_selectors() -> None:
    phases = [
        Phase(name="Foo", n_trials=1),
        Phase(name="next", n_trials=1, inherits=["Foo"]),
    ]

    experiment = make_experiment(trial_command="echo", phases=phases)

    assert [phase.name for phase in experiment.phases] == ["Foo", "next"]
    assert experiment.phases[1].inherits == ["Foo"]

    phases[1] = Phase(name="next", n_trials=1, inherits=["foo"])
    with pytest.raises(ValidationError, match="inherits from 'foo', which is not a prior phase"):
        make_experiment(trial_command="echo", phases=phases)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        pytest.param(
            ("metric", "extractor"),
            {"type": "log_regex", "pattern": "("},
            "Invalid metric regex",
            id="metric_regex_does_not_compile",
        ),
        pytest.param(
            ("metric", "extractor"),
            {"type": "log_regex", "pattern": r"loss=(?P<loss>\S+)"},
            "requires a named",
            id="metric_regex_lacks_the_value_group",
        ),
        pytest.param(
            ("phases", 0, "name"),
            "attempts",
            "reserved for the runtime recovery registry",
            id="reserved_attempts",
        ),
        pytest.param(
            ("phases", 0, "name"),
            "Attempts",
            "reserved for the runtime recovery registry",
            id="reserved_attempts_cased",
        ),
        pytest.param(
            ("phases", 0, "name"),
            "generations",
            "reserved for immutable generation records",
            id="reserved_generations",
        ),
        pytest.param(
            ("phases", 0, "name"),
            "Generations",
            "reserved for immutable generation records",
            id="reserved_generations_cased",
        ),
        pytest.param(
            ("phases", 0, "name"),
            "tracking",
            "reserved for MLflow tracking records",
            id="reserved_tracking",
        ),
    ],
)
def test_config_rejects_invalid_content_before_creating_workdir(
    tmp_path, path: tuple[str | int, ...], value: JsonValue, match: str
):
    """An invalid metric regex or reserved phase name fails the load before any workdir exists.

    ``path`` addresses the key of the dumped experiment payload to overwrite with ``value``.
    """
    payload = make_experiment(workdir=tmp_path / "work").model_dump(mode="json")
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path = write_yaml(tmp_path, yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=match):
        load_experiment(config_path)

    assert not (tmp_path / "work").exists()


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
        make_experiment(experiment="invalid_provenance", provenance=provenance)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: make_experiment(workdir=""), "workdir must be a nonempty path", id="workdir"
        ),
        pytest.param(
            lambda: ExecutionContext(cwd=""),
            "execution.cwd must be a nonempty path",
            id="execution_cwd",
        ),
    ],
)
def test_empty_path_strings_are_rejected(build: Callable[[], object], message: str) -> None:
    """An empty path would silently mean the invocation directory; omit or name it instead."""
    with pytest.raises(ValidationError, match=message):
        build()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        pytest.param({"A=B": "x"}, "env names must be nonempty", id="name_with_equals"),
        pytest.param({"": "x"}, "env names must be nonempty", id="empty_name"),
        pytest.param({"A\x00B": "x"}, "env names must be nonempty", id="nul_in_name"),
        pytest.param({"A": "x\x00y"}, r"env values cannot contain NUL.*\['A'\]", id="nul_in_value"),
    ],
)
def test_env_rejects_names_and_values_the_launch_cannot_pass(
    env: dict[str, str], message: str
) -> None:
    """``os.execve`` raises ValueError for these, which the supervisor does not catch.

    The launch would crash after the generation was claimed, so the load refuses them.
    """
    with pytest.raises(ValidationError, match=message):
        make_experiment(env=env)


@pytest.mark.parametrize(
    ("field", "names", "message"),
    [
        pytest.param("inherit_env", [""], "nonempty and unpadded", id="inherit_empty"),
        pytest.param("inherit_env", [" PATH"], "nonempty and unpadded", id="inherit_leading_space"),
        pytest.param(
            "inherit_env", ["PATH "], "nonempty and unpadded", id="inherit_trailing_space"
        ),
        pytest.param(
            "inherit_env",
            ["WANDB_API_KEY", "WANDB_API_KEY"],
            "must be unique",
            id="inherit_duplicate",
        ),
        pytest.param("inherit_env", ["A=B"], "no '=' or NUL", id="inherit_equals"),
        pytest.param("inherit_env", ["A\x00B"], "no '=' or NUL", id="inherit_nul"),
        pytest.param("passthrough_env", [""], "nonempty and unpadded", id="passthrough_empty"),
        pytest.param(
            "passthrough_env",
            [" HF_TOKEN"],
            "nonempty and unpadded",
            id="passthrough_leading_space",
        ),
        pytest.param("passthrough_env", ["A=B"], "no '=' or NUL", id="passthrough_equals"),
        pytest.param("passthrough_env", ["A\x00B"], "no '=' or NUL", id="passthrough_nul"),
        pytest.param(
            "passthrough_env",
            ["HF_TOKEN", "HF_TOKEN"],
            "must be unique",
            id="passthrough_duplicate",
        ),
    ],
)
def test_execution_env_names_validated(field: str, names: list[str], message: str) -> None:
    """A named-inheritance or passthrough list must be a set of exact variable names.

    Padded, empty, ``=``-bearing, or NUL-bearing names would silently inherit
    nothing (``os.environ`` has no ``' PATH'`` and cannot hold ``'A=B'``), and
    duplicates hide a typo in a second entry.
    """
    with pytest.raises(ValidationError, match=message):
        ExecutionContext(**{field: names})


def test_execution_accepts_named_inherit_list() -> None:
    execution = ExecutionContext(inherit_env=["WANDB_API_KEY", "HF_TOKEN"])

    assert execution.inherit_env == ["WANDB_API_KEY", "HF_TOKEN"]
    assert execution.cwd is None
    assert ExecutionContext().inherit_env == "none"


def test_execution_rejects_ambiguous_environment_classification() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ExecutionContext(inherit_env=["HF_TOKEN"], passthrough_env=["HF_TOKEN"])


def test_tmpdir_is_builtin_passthrough_and_a_written_list_adds_to_it() -> None:
    """``passthrough_env`` defaults to ``[]``; the effective set always holds ``TMPDIR``."""
    assert ExecutionContext().passthrough_env == []
    assert ExecutionContext().effective_passthrough_env() == ("TMPDIR",)
    assert ExecutionContext(passthrough_env=["TMPDIR"]).effective_passthrough_env() == ("TMPDIR",)
    written = ExecutionContext(passthrough_env=["WANDB_API_KEY", "HF_TOKEN"])
    assert written.effective_passthrough_env() == ("HF_TOKEN", "TMPDIR", "WANDB_API_KEY")


@pytest.mark.parametrize("passthrough", [[], ["TMPDIR"], ["HF_TOKEN"]])
def test_inherit_env_naming_tmpdir_is_refused(passthrough: list[str]) -> None:
    """``TMPDIR`` is always inherited and never digested, so an ``inherit_env`` list may not name it."""
    with pytest.raises(ValidationError, match=r"inherit_env must not name \['TMPDIR'\].*remove it"):
        ExecutionContext(inherit_env=["TMPDIR", "HF_HOME"], passthrough_env=passthrough)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            """
experiment: t
storage: null
provenance: {revision: test-fixture-v1}
trial_command: "first {overrides}"
trial_command: "second {overrides}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
""",
            "duplicate key",
        ),
        (
            """
experiment: t
storage: null
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
override_format: argparse
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
            "duplicate key",
        ),
        (
            """
experiment: t
storage: null
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
override_format: argparse
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
            "duplicate key",
        ),
        (
            """
experiment: t
storage: null
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
""",
            r"duplicate key '<<'",
        ),
    ],
    ids=["top_level", "search_space", "phase_mapping", "merge_key"],
)
def test_duplicate_yaml_keys_rejected(tmp_path: Path, body: str, match: str) -> None:
    """Duplicate keys fail loudly anywhere in the config tree, merge keys included."""
    with pytest.raises(ValueError, match=match):
        load_experiment(write_yaml(tmp_path, body))


def test_yaml_merge_keys_allow_explicit_overrides(tmp_path: Path) -> None:
    """Explicit keys may override values inherited through a YAML merge key."""
    body = """
experiment: t
storage: null
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: { type: json_envelope, objective_name: loss, split: test, policy: test }
phases:
  - &phase_defaults
    name: baseline
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space: { x: { type: int, low: 0, high: 9 } }
  - <<: *phase_defaults
    name: tuned
    n_trials: 2
"""
    exp = load_experiment(write_yaml(tmp_path, body))

    assert [(phase.name, phase.n_trials) for phase in exp.phases] == [
        ("baseline", 1),
        ("tuned", 2),
    ]


def test_override_format_n_jobs_and_failure_limit_defaults(tmp_path):
    body = """
experiment: t
storage: null
provenance: {revision: test-fixture-v1}
trial_command: "echo {config_path}"
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
    assert exp.override_format == "yaml_file"
    assert exp.phases[0].n_jobs == 1
    assert exp.phases[0].max_consecutive_failures == 5


def test_yaml_syntax_error_names_the_config_file(tmp_path: Path) -> None:
    """A parser/scanner error must carry the file path, not PyYAML's ``<unicode string>`` label."""
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("experiment: t\nphases:\n  - name: a\n   n_trials: 1\n")

    with pytest.raises(ConfigError) as excinfo:
        load_experiment(config_path)

    assert str(config_path) in str(excinfo.value)


@pytest.mark.parametrize(
    "env_line",
    ['env: { A: "x\\ud800" }\n', 'env: { "\\ud800": x }\n'],
    ids=["value", "key"],
)
def test_lone_surrogate_escape_is_rejected_at_load(tmp_path: Path, env_line: str) -> None:
    """A ``\\ud800`` escape loads as text that cannot reach a trainer's environment."""
    body = (
        "experiment: t\n"
        'trial_command: "echo {overrides}"\n'
        "override_format: argparse\n"
        f"{env_line}"
        "metric:\n"
        "  extractor: { type: json_envelope, objective_name: x, split: test, policy: test }\n"
        "phases:\n"
        "  - name: p\n"
        "    n_trials: 1\n"
    )
    config_path = write_yaml(tmp_path, body)

    with pytest.raises(ConfigError, match="lone surrogate") as excinfo:
        load_experiment(config_path)

    assert str(config_path) in str(excinfo.value)


def _sampler_policy_experiment(
    *,
    storage: str | None,
    sampler: JsonObject | None = None,
    n_trials: int = 2,
    search_space: JsonObject | None = None,
) -> Experiment:
    """Build a one-phase experiment for the persistent-storage sampler policy.

    The phase is explicit because ``make_experiment`` seeds a default phase's
    sampler on persistent storage, which would hide the policy under test.

    :param str | None storage: Storage URL to declare, or ``None`` for in-memory.
    :param JsonObject | None sampler: Sampler mapping, or ``None`` to leave the default.
    :param int n_trials: Trial budget for the single phase.
    :param JsonObject | None search_space: Search space; ``None`` means one float ``lr``,
        and an empty mapping means a fixed configuration.
    :return Experiment: Validated experiment.
    """
    phase: JsonObject = {
        "name": "lr",
        "n_trials": n_trials,
        "search_space": (
            {"lr": {"type": "float", "low": 0.1, "high": 1.0}}
            if search_space is None
            else search_space
        ),
    }
    if sampler is not None:
        phase["sampler"] = sampler
    return make_experiment(storage=storage, phases=[phase])


@pytest.mark.parametrize(
    ("sampler", "message"),
    [
        pytest.param(None, "requires an explicit sampler.seed", id="default_tpe_unseeded"),
        pytest.param({"type": "tpe"}, "requires an explicit sampler.seed", id="tpe_unseeded"),
        pytest.param({"type": "cmaes"}, "requires an explicit sampler.seed", id="cmaes_unseeded"),
        pytest.param({"type": "random"}, "requires an explicit sampler.seed", id="random_unseeded"),
    ],
)
def test_persistent_storage_rejects_unseeded_samplers(
    tmp_path: Path, sampler: JsonObject | None, message: str
) -> None:
    """Persistent storage refuses an unseeded stochastic sampler at config load.

    The seed keys every draw by its position in the study, which is what lets
    a durable study resume and grow without drawing a configuration twice.
    """
    storage = f"journal:///{tmp_path}/phases.journal"
    with pytest.raises(ValueError, match=message) as excinfo:
        _sampler_policy_experiment(storage=storage, sampler=sampler)
    assert "'lr'" in str(excinfo.value)


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param({"type": "tpe", "seed": 0}, id="tpe"),
        pytest.param({"type": "cmaes", "seed": 1}, id="cmaes"),
        pytest.param({"type": "random", "seed": 3}, id="random"),
    ],
)
def test_persistent_storage_accepts_seeded_samplers(tmp_path: Path, sampler: JsonObject) -> None:
    """A seeded sampler loads unchanged on persistent storage."""
    experiment = _sampler_policy_experiment(
        storage=f"journal:///{tmp_path}/phases.journal", sampler=sampler
    )
    assert experiment.phases[0].sampler.seed is not None


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_sampler_seed_rejects_values_outside_optuna_domain(seed: int) -> None:
    """Every accepted seed must be constructible by the supported Optuna samplers."""
    with pytest.raises(ValueError, match="seed"):
        Sampler(type="random", seed=seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_sampler_seed_accepts_optuna_domain_boundaries(seed: int) -> None:
    """The validation boundary matches NumPy/Optuna's unsigned 32-bit seed domain."""
    assert Sampler(type="random", seed=seed).seed == seed


def test_persistent_storage_accepts_grid_without_seed(tmp_path: Path) -> None:
    """Grid enumerates a fixed matrix, so it needs no seed."""
    experiment = _sampler_policy_experiment(
        storage=f"journal:///{tmp_path}/phases.journal",
        sampler={"type": "grid"},
        n_trials=3,
        search_space={"lr": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.5}},
    )
    assert experiment.phases[0].sampler.seed is None


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(None, id="default_tpe"),
        pytest.param({"type": "tpe"}, id="tpe"),
        pytest.param({"type": "cmaes"}, id="cmaes"),
        pytest.param({"type": "random"}, id="random"),
    ],
)
def test_empty_search_space_skips_the_seed_on_persistent_storage(
    tmp_path: Path, sampler: JsonObject | None
) -> None:
    """A fixed configuration draws nothing, so no seed is owed."""
    experiment = _sampler_policy_experiment(
        storage=f"journal:///{tmp_path}/phases.journal",
        sampler=sampler,
        n_trials=1,
        search_space={},
    )
    assert experiment.phases[0].sampler.seed is None


@pytest.mark.parametrize(
    "sampler", [{"type": "tpe"}, {"type": "random", "seed": 0}, {"type": "grid"}]
)
def test_in_memory_fixed_configuration_refuses_identical_trials(
    tmp_path: Path, sampler: JsonObject
) -> None:
    """With ``storage: null`` nothing is ever retried, so load refuses what every launch would.

    Several runs of one fixed configuration are repeats, so only repeats may ask for them. A
    persistent study may hold failed attempts for the extra trials to replace, which only a
    launch can see, so the same phase loads there.
    """
    phase = Phase.model_validate({"name": "x", "n_trials": 3, "sampler": sampler})
    with pytest.raises(ValidationError) as excinfo:
        make_experiment(phases=[phase])
    assert (
        "Phase 'x': n_trials=3 with an empty search_space would run 3 identical trials. Use "
        "n_trials: 1 for a fixed configuration, or repeats to measure seed noise."
    ) in str(excinfo.value)

    persistent = make_experiment(storage=f"journal:///{tmp_path}/phases.journal", phases=[phase])
    assert persistent.phases[0].n_trials == 3

    fixed = make_experiment(phases=[Phase.model_validate({"name": "x", "n_trials": 1})])
    assert fixed.phases[0].n_trials == 1
    audit = Phase.model_validate(
        {"name": "x", "n_trials": 3, "sampler": {"type": "grid"}, "repeats": _SEEDS}
    )
    assert make_experiment(phases=[audit]).phases[0].repeats is not None


@pytest.mark.parametrize("storage", ["", "storage: null\n"], ids=["omitted", "explicit_null"])
def test_in_memory_storage_keeps_the_sampler_block_optional(tmp_path: Path, storage: str) -> None:
    """Without a durable study there is nothing to resume, so the tpe default stands.

    Covers both YAML spellings that leave ``storage`` in-memory: the key
    omitted entirely, and an explicit ``storage: null``.
    """
    body = (
        f"experiment: t\n{storage}"
        'trial_command: "echo {overrides}"\n'
        "override_format: argparse\n"
        "metric:\n"
        "  extractor: { type: json_envelope, objective_name: x, split: test, policy: test }\n"
        "phases:\n"
        "  - name: lr\n"
        "    n_trials: 2\n"
        "    search_space: { lr: { type: float, low: 0.1, high: 1.0 } }\n"
    )
    experiment = load_experiment(write_yaml(tmp_path, body))
    assert experiment.phases[0].sampler.type == "tpe"
    assert experiment.phases[0].sampler.seed is None


def test_metric_bounds_are_optional_finite_and_ordered() -> None:
    """Each bound is optional alone, finite, never boolean, and ordered when both are set."""
    from phasesweep.config.models import experiment_contract_lines

    extractor: JsonObject = {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.]+)"}

    def metric(**bounds: JsonValue) -> Metric:
        return Metric.model_validate({"extractor": extractor, **bounds})

    unbounded = metric()
    assert (unbounded.min, unbounded.max) == (None, None)
    assert (metric(min=0.5).min, metric(min=0.5).max) == (0.5, None)
    assert (metric(max=2).min, metric(max=2).max) == (None, 2.0)
    assert (metric(min=0.5, max=0.5).min, metric(min=0.5, max=0.5).max) == (0.5, 0.5)

    with pytest.raises(ValidationError, match=r"min \(3.0\) > max \(2.0\)"):
        metric(min=3, max=2)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError, match="must be finite"):
            metric(min=bad)
        with pytest.raises(ValidationError, match="must be finite"):
            metric(max=bad)
    with pytest.raises(ValidationError, match="must be numbers, not booleans"):
        metric(max=True)

    plain = experiment_contract_lines(make_experiment(metric=unbounded))[0]
    bounded = experiment_contract_lines(make_experiment(metric=metric(min=0.5, max=2)))[0]
    assert "bounds" not in plain
    assert "bounds={'max': 2.0, 'min': 0.5}" in bounded
    only_max = experiment_contract_lines(make_experiment(metric=metric(max=2)))[0]
    assert "bounds={'max': 2.0}" in only_max


_LOG_EXTRACTOR: JsonObject = {"type": "log_regex", "pattern": r"m=(?P<value>[0-9.]+)"}


def _observation(name: str) -> Observation:
    """Return a local ``log_regex`` observation called ``name``."""
    return Observation.model_validate({"name": name, "extractor": _LOG_EXTRACTOR})


def test_observe_names_are_distinct_from_metric_and_constraints() -> None:
    """One namespace covers the metric, constraints and observations."""
    metric = Metric.model_validate({"name": "loss", "extractor": _LOG_EXTRACTOR})
    memory = Constraint.model_validate({"name": "memory", "extractor": _LOG_EXTRACTOR, "max": 5})

    def build(*names: str) -> Experiment:
        return make_experiment(
            metric=metric,
            constraints=[memory],
            observe=[_observation(name) for name in names],
        )

    for clash in (("loss",), ("memory",), ("peak", "peak")):
        with pytest.raises(ValidationError, match="Metric, constraint and observe names"):
            build(*clash)
    assert [o.name for o in build("peak", "temperature").observe] == ["peak", "temperature"]
    assert make_experiment().observe == []


@pytest.mark.parametrize(
    "extractor",
    [
        {"type": "wandb", "entity": "e", "project": "p", "metric_key": "peak"},
        {"type": "mlflow", "metric_key": "peak", "select": "last"},
    ],
    ids=["wandb", "mlflow"],
)
def test_observe_refuses_remote_extractors(extractor: JsonObject, tmp_path: Path) -> None:
    """Remote readers wait for every key, so an observation may only read local files."""
    reason = (
        r"Observation 'peak' cannot use a \w+ extractor: remote extractors wait for every "
        r"requested key.*report_objective\(\.\.\., extra=\.\.\.\).*local json extractor"
    )
    with pytest.raises(ValidationError, match=reason):
        Observation.model_validate({"name": "peak", "extractor": extractor})
    with pytest.raises(ValidationError, match=reason):
        Observation(
            name="peak",
            extractor=WandbExtractor(type="wandb", entity="e", project="p", metric_key="k"),
        )

    body = (
        "experiment: t\n"
        'trial_command: "echo {overrides}"\n'
        "override_format: argparse\n"
        "metric:\n"
        "  extractor: { type: json_envelope, objective_name: x, split: test, policy: test }\n"
        f"observe:\n  - {{ name: peak, extractor: {json.dumps(extractor)} }}\n"
        "phases:\n"
        "  - name: lr\n"
        "    n_trials: 2\n"
        "    search_space: { lr: { type: float, low: 0.1, high: 1.0 } }\n"
    )
    with pytest.raises(ValidationError, match="local json extractor"):
        load_experiment(write_yaml(tmp_path, body))
    local = body.replace(
        json.dumps(extractor), "{ type: json, path: result.json, key: peak_memory_gb }"
    )
    loaded = load_experiment(write_yaml(tmp_path, local))
    assert [(o.name, o.extractor.type) for o in loaded.observe] == [("peak", "json")]


def test_observe_rejects_evaluation_axis() -> None:
    """Only the metric extractor's axis is read, so an observation may not set one."""
    envelope: JsonObject = {
        "type": "json_envelope",
        "objective_name": "peak",
        "split": "test",
        "policy": "final",
    }
    assert Observation.model_validate({"name": "peak", "extractor": envelope}).name == "peak"
    with pytest.raises(
        ValidationError,
        match="Observation 'peak' sets evaluation_axis, which only the metric extractor's",
    ):
        Observation.model_validate(
            {"name": "peak", "extractor": {**envelope, "evaluation_axis": "tokens"}}
        )


_SEEDS: JsonObject = {"key": "seed", "values": [0, 1, 2]}


def _repeats_phase(name: str = "p", **fields: JsonValue) -> Phase:
    """Return a seeded random phase over two configurations x three seeds, with ``fields`` applied."""
    spec: JsonObject = {
        "name": name,
        "n_trials": 6,
        "sampler": {"type": "random", "seed": 0},
        "search_space": {"lr": {"type": "categorical", "choices": [0.1, 0.2]}},
        "repeats": _SEEDS,
    }
    spec.update(fields)
    return Phase.model_validate(spec)


@pytest.mark.parametrize(
    ("sampler", "spelled"),
    [
        ({"type": "tpe", "seed": 0}, "sampler.type='tpe'"),
        ({"type": "cmaes", "seed": 0}, "sampler.type='cmaes'"),
        ({"type": "random"}, "sampler.type='random' without a seed"),
    ],
)
def test_repeats_requires_grid_or_seeded_random(sampler: JsonObject, spelled: str) -> None:
    numeric: JsonObject = {"lr": {"type": "float", "low": 0.1, "high": 0.2}}
    with pytest.raises(
        ValidationError,
        match=f"Phase 'p': repeats needs sampler.type grid, or sampler.type random with a "
        f"seed, and this phase uses {spelled}. Only those two keep",
    ):
        make_experiment(phases=[_repeats_phase(sampler=sampler, search_space=numeric)])

    grid = make_experiment(phases=[_repeats_phase(sampler={"type": "grid"})])
    assert grid.phases[0].repeats is not None
    seeded = make_experiment(phases=[_repeats_phase()]).phases[0]
    assert seeded.sampler.seed == 0
    assert sampler_capability_line(seeded, "journal:///tmp/x.journal") == (
        "phase 'p': sampler=random seed=0 (resumable, reproducible), repeats=seed×3 (mean)"
    )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        pytest.param(
            lambda: [
                _repeats_phase(
                    repeats={"key": "data.shard", "values": [0, 1, 2]},
                    search_space={"data.shard": {"type": "int", "low": 0, "high": 1}},
                )
            ],
            "repeats.key 'data.shard' is also a search_space key",
            id="searched",
        ),
        pytest.param(
            lambda: [
                _repeats_phase(
                    repeats={"key": "data.shard", "values": [0, 1, 2]},
                    fixed_overrides={"data.shard": 4},
                )
            ],
            "repeats.key 'data.shard' is also a fixed_overrides key",
            id="fixed",
        ),
        pytest.param(
            lambda: [
                _repeats_phase(
                    repeats={"key": "data", "values": [0, 1, 2]},
                    fixed_overrides={"data.shard": 4},
                )
            ],
            "dotted-key namespace collision",
            id="local-prefix",
        ),
        pytest.param(
            lambda: [
                Phase(name="base", n_trials=1, fixed_overrides={"optim.lr": 0.1}),
                _repeats_phase(
                    "child", inherits=["base"], repeats={"key": "optim", "values": [0, 1, 2]}
                ),
            ],
            "dotted-key namespace collision",
            id="inherited-prefix",
        ),
        pytest.param(
            lambda: [_repeats_phase(repeats={"key": "data..shard", "values": [0, 1]})],
            "phase 'p' repeats.key: override key 'data..shard' has an empty dotted segment",
            id="syntax",
        ),
    ],
)
def test_repeats_key_is_free_of_search_space_fixed_overrides_and_prefixes(
    build: Callable[[], list[Phase]], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        make_experiment(phases=build())


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([0], "at least 2 items"),
        ([1, 1.0], "repeats.values must remain distinguishable"),
        ([0, False], "repeats.values must remain distinguishable"),
        ([0.0, float("nan")], "repeats.values must be finite floats"),
        ([[0], [1]], "repeats.values"),
        ([{"a": 1}, {"a": 2}], "repeats.values"),
    ],
)
def test_repeats_values_follow_categorical_choice_rules(values: JsonValue, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Phase.model_validate(
            {"name": "p", "n_trials": 2, "repeats": {"key": "seed", "values": values}}
        )
    accepted = Phase.model_validate(
        {"name": "p", "n_trials": 4, "repeats": {"key": "seed", "values": [None, True, 2, "s"]}}
    )
    assert accepted.repeats is not None
    assert accepted.repeats.values == [None, True, 2, "s"]
    assert accepted.repeats.reduce == "mean"


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        pytest.param(
            {"sampler": {"type": "grid"}, "n_trials": 3},
            r"grid sampler has 6 combinations \(2 configurations x 3 repeats\) but n_trials=3\. "
            r"Grid phases run the full matrix by default; a grid with repeats always runs the "
            r"full matrix, so set n_trials: 6\.",
            id="grid-short",
        ),
        pytest.param(
            {"sampler": {"type": "grid"}, "n_trials": 9},
            r"n_trials=9 exceeds the grid cardinality 6 \(2 configurations x 3 repeats\)",
            id="grid-over",
        ),
        pytest.param(
            {"n_trials": 7},
            "seeded random with repeats runs each configuration 3 times, once per "
            "repeats.values entry, so n_trials must be a multiple of 3; got 7.",
            id="random-multiple",
        ),
        pytest.param(
            {"sampler": {"type": "grid"}, "search_space": {}, "n_trials": 6},
            r"repeats over an empty search_space runs one configuration once per repeat value, "
            r"so n_trials must be 3 \(the number of repeats.values\), not 6\.",
            id="empty-space",
        ),
        pytest.param(
            {"sampler": {"type": "grid"}, "n_trials": 3, "allow_partial_grid": True},
            "allow_partial_grid cannot be combined with repeats",
            id="partial-grid",
        ),
    ],
)
def test_repeats_trial_count_covers_every_repeat(fields: JsonObject, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        make_experiment(phases=[_repeats_phase(**fields)])
    assert make_experiment(phases=[_repeats_phase(n_trials=12)]).phases[0].n_trials == 12


def test_repeats_key_overrides_inherited_value_and_is_never_exported() -> None:
    """The repeat key settles conflicting parents, and descendants never inherit it."""
    parents = [
        Phase(name="a", n_trials=1, fixed_overrides={"data.shard": 1}),
        Phase(name="b", n_trials=1, fixed_overrides={"data.shard": 2}),
    ]
    with pytest.raises(ValidationError, match="inherits conflicting key"):
        make_experiment(phases=[*parents, Phase(name="c", n_trials=1, inherits=["a", "b"])])
    repeats: JsonObject = {"key": "data.shard", "values": [0, 1, 2]}
    child = _repeats_phase("c", inherits=["a", "b"], repeats=repeats)
    # data.shard is not exported, so its prefix is free below the repeats phase.
    grandchild = Phase(name="d", n_trials=1, inherits=["c"], fixed_overrides={"data": "full"})
    experiment = make_experiment(phases=[*parents, child, grandchild])
    assert [phase.name for phase in experiment.phases] == ["a", "b", "c", "d"]
    with pytest.raises(ValidationError, match="dotted-key namespace collision"):
        make_experiment(
            phases=[*parents, grandchild.model_copy(update={"inherits": ["a"]})],
        )


@pytest.mark.parametrize("key", ["seed", "trainer.seed"])
def test_seed_search_keys_say_how_to_fix_the_seed(key: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Phase(name="p", n_trials=2, search_space={key: {"type": "int", "low": 0, "high": 9}})
    assert (
        f"Phase 'p': seed keys cannot be searched: ['{key}']. A searched seed lets seed noise "
        "win as if it were a configuration. Fix the seed in fixed_overrides or trainer_config. "
        f"To compare finalists across seeds, use repeats: {{key: {key}, values: [0, 1, 2]}}."
    ) in str(excinfo.value)
    assert Phase(
        name="p", n_trials=2, search_space={"seedling": {"type": "int", "low": 0, "high": 9}}
    )
    with pytest.raises(ValidationError, match="allow_seed_search"):
        Phase.model_validate({"name": "p", "n_trials": 1, "allow_seed_search": True})
