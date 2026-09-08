from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phasesweep import load_config, load_experiment
from phasesweep.config import (
    ConfigError,
    ExecutionContext,
    Experiment,
    JsonExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
    Suite,
)
from tests.conftest import assert_invalid_experiment_yaml, make_experiment, write_yaml


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


def test_config_rejects_reserved_phase_name_before_creating_workdir(tmp_path):
    payload = make_experiment(workdir=tmp_path / "work").model_dump(mode="json")
    payload["phases"][0]["name"] = "attempts"
    path = write_yaml(tmp_path, yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="reserved for the runtime recovery registry"):
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
              override_format: argparse
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


def test_suite_trainer_config_is_inherited_replaced_or_cleared(tmp_path: Path) -> None:
    """Suites preserve the one-YAML trainer boundary without implicit deep merges."""
    config = load_config(
        write_yaml(
            tmp_path,
            """
            suite: trainer_config_suite
            defaults:
              trial_command: "python train.py {config_path}"
              trainer_config:
                model: {depth: 4, width: 128}
                optimizer: {lr: 0.001}
              metric:
                name: loss
                goal: minimize
                extractor: {type: log_regex, pattern: 'loss=(?P<value>[0-9.]+)'}
            studies:
              - name: inherited
                phases: [{name: p, n_trials: 1}]
              - name: replaced
                trainer_config:
                  model: {depth: 8}
                phases: [{name: p, n_trials: 1}]
              - name: cleared
                trainer_config: null
                phases: [{name: p, n_trials: 1}]
            """,
        )
    )

    assert isinstance(config, Suite)
    inherited, replaced, cleared = (config.experiment_for_study(study) for study in config.studies)
    assert inherited.override_format == "yaml_file"
    assert inherited.trainer_config == {
        "model": {"depth": 4, "width": 128},
        "optimizer": {"lr": 0.001},
    }
    assert replaced.trainer_config == {"model": {"depth": 8}}
    assert cleared.trainer_config == {}


@pytest.mark.parametrize(
    "candidate_metric",
    [
        "{name: accuracy, goal: minimize, "
        "extractor: {type: log_regex, pattern: 'accuracy=(?P<value>[0-9.]+)'}}",
        "{name: loss, goal: maximize, "
        "extractor: {type: log_regex, pattern: 'loss=(?P<value>[0-9.]+)'}}",
        "{name: loss, goal: minimize, "
        "extractor: {type: log_regex, pattern: 'eval_loss=(?P<value>[0-9.]+)'}}",
    ],
    ids=["name", "goal", "extractor"],
)
def test_suite_promotion_requires_identical_metric_contracts(
    tmp_path: Path, candidate_metric: str
) -> None:
    """Reject cross-study promotion when the compared scalars mean different things."""
    config = f"""
    suite: incompatible_metrics
    defaults:
      trial_command: "echo"
      metric:
        name: loss
        goal: minimize
        extractor: {{type: log_regex, pattern: 'loss=(?P<value>[0-9.]+)'}}
    studies:
      - name: baseline
        phases: [{{name: baseline_eval, n_trials: 1}}]
      - name: candidate
        metric: {candidate_metric}
        promotion:
          min_delta_vs: baseline
        phases: [{{name: candidate_eval, n_trials: 1}}]
    """

    with pytest.raises(
        ValidationError,
        match=r"promotion against 'baseline' requires the same resolved metric contract.*"
        r"Put the shared metric in suite.defaults",
    ):
        load_config(write_yaml(tmp_path, config))


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


def test_plain_json_extractor_is_not_a_primary_objective() -> None:
    with pytest.raises(ValidationError, match="json_envelope"):
        Metric(extractor=JsonExtractor(type="json", path="result.json", key="loss"))


def test_suite_and_study_names_reject_double_underscore() -> None:
    """'<suite>__<study>' compilation is injective only when neither part can
    contain the separator: suite 'sweep' / study 'bert__lr' and suite
    'sweep__bert' / study 'lr' would otherwise share one artifact namespace,
    study identity, and fingerprint (review v0.5.17 gap hunt)."""
    from phasesweep.config import IntParam, StudySpec

    phase = Phase(
        name="p",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    with pytest.raises(ValidationError, match="must not contain '__'"):
        StudySpec(name="bert__lr", phases=[phase])

    with pytest.raises(ValidationError, match="must not contain '__'"):
        Suite(suite="sweep__bert", studies=[StudySpec(name="lr", phases=[phase])])


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
