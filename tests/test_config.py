from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phasesweep import load_config, load_experiment
from phasesweep.config import (
    ConfigError,
    Constraint,
    ExecutionContext,
    Experiment,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from tests.conftest import assert_invalid_experiment_yaml, make_experiment, write_yaml


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


def test_phase_composition_accepts_a_linear_chain() -> None:
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

    assert [phase.name for phase in experiment.phases] == ["base", "tune", "final"]


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
                Phase(name="base", n_trials=1, fixed_overrides={"lr": 1e-4}),
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


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
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
""",
            "inherits from 'b'",
            id="inherit-must-reference-prior-phase",
        ),
        pytest.param(
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
""",
            "must define at least one",
            id="constraint-requires-bound",
        ),
        pytest.param(
            """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
override_format: argparse
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
""",
            "distinct",
            id="metric-constraint-name-collision",
        ),
        pytest.param(
            """
experiment: t
storage: ":memory:"
provenance: {revision: test-fixture-v1}
trial_command: "echo {overrides}"
override_format: argparse
metric:
  name: goal
  goal: minimize
  extractor: { type: json_envelope, objective_name: goal, split: test, policy: test }
phases:
  - name: a
    n_trials: 1
    search_space: { x: { type: float, low: 0, high: 1 } }
""",
            "reserved for winner direction metadata",
            id="metric-name-goal-is-reserved",
        ),
    ],
)
def test_invalid_experiment_relationships(
    tmp_path: Path,
    body: str,
    match: str,
) -> None:
    assert_invalid_experiment_yaml(tmp_path, body, match)


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
    ("pattern", "match"),
    [("(", "Invalid metric regex"), (r"loss=(?P<loss>\S+)", "requires a named")],
)
def test_config_rejects_invalid_metric_regex_before_creating_workdir(tmp_path, pattern, match):
    payload = make_experiment(workdir=tmp_path / "work").model_dump(mode="json")
    payload["metric"]["extractor"] = {"type": "log_regex", "pattern": pattern}
    path = write_yaml(tmp_path, yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=match):
        load_experiment(path)

    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("attempts", "reserved for the runtime recovery registry"),
        ("Attempts", "reserved for the runtime recovery registry"),
        ("generations", "reserved for immutable generation records"),
        ("Generations", "reserved for immutable generation records"),
    ],
)
def test_config_rejects_reserved_phase_name_before_creating_workdir(
    tmp_path, name: str, match: str
):
    payload = make_experiment(workdir=tmp_path / "work").model_dump(mode="json")
    payload["phases"][0]["name"] = name
    path = write_yaml(tmp_path, yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=match):
        load_experiment(path)

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


@pytest.mark.parametrize(
    ("passthrough_env", "message"),
    [
        ([""], "nonempty and unpadded"),
        ([" HF_TOKEN"], "nonempty and unpadded"),
        (["HF_TOKEN", "HF_TOKEN"], "must be unique"),
    ],
)
def test_execution_passthrough_env_names_validated(
    passthrough_env: list[str], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ExecutionContext(passthrough_env=passthrough_env)


def test_execution_rejects_ambiguous_environment_classification() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ExecutionContext(inherit_env=["HF_TOKEN"], passthrough_env=["HF_TOKEN"])


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
        """
experiment: t
storage: ":memory:"
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
        """
experiment: t
storage: ":memory:"
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
override_format: argparse
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


def test_plain_json_extractor_is_a_primary_objective() -> None:
    metric = Metric(extractor=JsonExtractor(type="json", path="result.json", key="loss"))
    assert metric.extractor.type == "json"


def test_suite_config_is_rejected_before_artifacts(tmp_path: Path) -> None:
    """The removed top-level selector must not be ignored as an experiment field."""
    path = write_yaml(
        tmp_path,
        """
        suite: retired
        defaults: {}
        studies: []
        """,
    )

    with pytest.raises(ConfigError, match="suite configs are no longer supported"):
        load_config(path)


def test_yaml_syntax_error_names_the_config_file(tmp_path: Path) -> None:
    """A parser/scanner error must carry the file path, not PyYAML's stream label.

    Bad indentation is one of the two most common YAML mistakes. PyYAML marks it
    ``in "<unicode string>"``, so an unwrapped error leaves an operator with a
    line number and no file - useless for a suite that loads several configs.
    """
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("experiment: t\nphases:\n  - name: a\n   n_trials: 1\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    assert str(config_path) in str(excinfo.value)


def test_config_error_stays_a_value_error() -> None:
    """Existing ``except ValueError`` callers must keep catching config failures."""
    assert issubclass(ConfigError, ValueError)


def _sampler_policy_yaml(
    *,
    storage: str | None,
    sampler: str | None = None,
    n_trials: int = 2,
    search_space: str = "{ lr: { type: float, low: 0.1, high: 1.0 } }",
) -> str:
    """Build a one-phase experiment YAML for the persistent-storage sampler policy.

    :param str | None storage: Storage URL to declare, or ``None`` to omit the key.
    :param str | None sampler: Inline sampler mapping, or ``None`` to omit the block.
    :param int n_trials: Trial budget for the single phase.
    :param str search_space: Inline search-space mapping for the single phase.
    :return str: Complete experiment YAML text.
    """
    header = (
        f"storage: {storage}\nprovenance: {{revision: test-fixture-v1}}\n"
        if storage is not None
        else ""
    )
    sampler_line = f"    sampler: {sampler}\n" if sampler is not None else ""
    return (
        "experiment: t\n"
        f"{header}"
        'trial_command: "echo {overrides}"\n'
        "override_format: argparse\n"
        "metric:\n"
        "  extractor: { type: json_envelope, objective_name: x, split: test, policy: test }\n"
        "phases:\n"
        "  - name: lr\n"
        f"    n_trials: {n_trials}\n"
        f"{sampler_line}"
        f"    search_space: {search_space}\n"
    )


@pytest.mark.parametrize(
    ("sampler", "message"),
    [
        pytest.param(None, "requires an explicit sampler.seed", id="default_tpe_unseeded"),
        pytest.param("{ type: tpe }", "requires an explicit sampler.seed", id="tpe_unseeded"),
        pytest.param("{ type: cmaes }", "requires an explicit sampler.seed", id="cmaes_unseeded"),
        pytest.param("{ type: random }", "requires an explicit sampler.seed", id="random_unseeded"),
        pytest.param(
            "{ type: tpe, seed: 0 }",
            "requires sampler.acknowledge_nonresumable: true",
            id="tpe_unacknowledged",
        ),
        pytest.param(
            "{ type: cmaes, seed: 1 }",
            "requires sampler.acknowledge_nonresumable: true",
            id="cmaes_unacknowledged",
        ),
    ],
)
def test_persistent_storage_rejects_unseeded_and_unacknowledged_samplers(
    tmp_path: Path, sampler: str | None, message: str
) -> None:
    """Persistent storage must state the sampler contract at config load, not mid-run.

    An unseeded stochastic sampler makes a durable study irreproducible, and
    TPE/CMA-ES hold process-local state Optuna storage does not persist, so
    ``_validate_sampler_continuation`` hard-rejects a mid-target resume — but
    only after the operator has already been interrupted.
    """
    body = _sampler_policy_yaml(storage=f"sqlite:///{tmp_path}/phases.db", sampler=sampler)
    with pytest.raises(ValueError, match=message) as excinfo:
        load_experiment(write_yaml(tmp_path, body))
    assert "'lr'" in str(excinfo.value)


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param("{ type: tpe, seed: 0, acknowledge_nonresumable: true }", id="tpe"),
        pytest.param("{ type: cmaes, seed: 1, acknowledge_nonresumable: true }", id="cmaes"),
        pytest.param("{ type: random, seed: 3 }", id="random_needs_no_acknowledgement"),
    ],
)
def test_persistent_storage_accepts_seeded_and_acknowledged_samplers(
    tmp_path: Path, sampler: str
) -> None:
    """A seeded sampler with the required acknowledgement loads unchanged."""
    body = _sampler_policy_yaml(storage=f"sqlite:///{tmp_path}/phases.db", sampler=sampler)
    experiment = load_experiment(write_yaml(tmp_path, body))
    assert experiment.phases[0].sampler.seed is not None


@pytest.mark.parametrize("seed", [-1, 2**32])
@pytest.mark.parametrize("sampler_type", ["grid", "random", "tpe", "cmaes"])
def test_sampler_seed_rejects_values_outside_optuna_domain(sampler_type: str, seed: int) -> None:
    """Every accepted seed must be constructible by the supported Optuna samplers."""
    with pytest.raises(ValueError, match="seed"):
        Sampler(type=sampler_type, seed=seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
@pytest.mark.parametrize("sampler_type", ["grid", "random", "tpe", "cmaes"])
def test_sampler_seed_accepts_optuna_domain_boundaries(sampler_type: str, seed: int) -> None:
    """The validation boundary matches NumPy/Optuna's unsigned 32-bit seed domain."""
    assert Sampler(type=sampler_type, seed=seed).seed == seed


def test_persistent_storage_accepts_grid_without_seed_or_acknowledgement(tmp_path: Path) -> None:
    """Grid enumerates a fixed matrix and resumes from stored assignments, so it is exempt."""
    body = _sampler_policy_yaml(
        storage=f"sqlite:///{tmp_path}/phases.db",
        sampler="{ type: grid }",
        n_trials=3,
        search_space="{ lr: { type: float, low: 0.0, high: 1.0, step: 0.5 } }",
    )
    experiment = load_experiment(write_yaml(tmp_path, body))
    assert experiment.phases[0].sampler.seed is None
    assert experiment.phases[0].sampler.acknowledge_nonresumable is False


@pytest.mark.parametrize("storage", [None, '":memory:"'])
def test_in_memory_storage_keeps_the_sampler_block_optional(
    tmp_path: Path, storage: str | None
) -> None:
    """Without a durable study there is nothing to resume, so the tpe default stands."""
    experiment = load_experiment(write_yaml(tmp_path, _sampler_policy_yaml(storage=storage)))
    assert experiment.phases[0].sampler.type == "tpe"
    assert experiment.phases[0].sampler.seed is None


@pytest.mark.parametrize("sampler_type", ["grid", "random"])
def test_acknowledge_nonresumable_rejected_on_resumable_samplers(sampler_type: str) -> None:
    """The acknowledgement states a contract these samplers do not impose."""
    with pytest.raises(ValidationError, match="acknowledge_nonresumable"):
        Sampler(type=sampler_type, seed=0, acknowledge_nonresumable=True)  # type: ignore[arg-type]
