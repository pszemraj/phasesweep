"""CLI commands: validate, show-winners, --dry-run."""

from __future__ import annotations

import logging
import re
import shlex
import signal
import stat
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import cli as cli_main
from phasesweep.config import (
    CategoricalParam,
    ExecutionContext,
    Experiment,
    FloatParam,
    JsonEnvelopeExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Repeats,
    Sampler,
)
from phasesweep.config.models import _environment_contract_line
from phasesweep.engine import (
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    NoFeasibleTrialError,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRefusedError,
    UnsafeProcessCleanupError,
)
from phasesweep.engine.paths import (
    _generation_dir,
    _generation_path,
    _last_successful_generation_path,
    _winner_path,
)
from phasesweep.engine.state import Winner
from phasesweep.errors import GpuConfigurationError
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.runs import RunStore
from phasesweep.runtime.files import UnsafeLockPathError, lock_dir
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    requires_nonroot,
    write_constant_trainer,
    write_param_echo_trainer,
)
from tests.ledger_fixtures import materialize, published_generation_id
from tests.recovery_helpers import recover_run_cli


@pytest.mark.parametrize(
    ("extractor", "details"),
    [
        (
            {"type": "json", "path": "r.json", "key": "eval.loss"},
            [
                "r.json",
                "eval.loss",
                "evaluation_point check: n/a (json objectives record no evaluation position; "
                "json_envelope via report_objective, wandb with evaluation_axis, or mlflow would)",
            ],
        ),
        (
            {"type": "wandb", "entity": "e", "project": "p", "metric_key": "eval/loss"},
            [
                "eval/loss",
                "api.wandb.ai",
                "evaluation_point check: supported (axis '_step' located in W&B run history; "
                "advisory, n/a when no position is found)",
            ],
        ),
        (
            {"type": "log_regex", "pattern": "loss=(?P<value>[0-9.]+)"},
            [
                "select",
                "last",
                "stdout.log",
                "evaluation_point check: n/a (log_regex objectives record no evaluation "
                "position; json_envelope via report_objective, wandb with evaluation_axis, "
                "or mlflow would)",
            ],
        ),
        (
            {
                "type": "json_envelope",
                "objective_name": "loss",
                "split": "validation",
                "policy": "final",
                "expected_step": 20,
            },
            [
                "policy",
                "final",
                "expected_step",
                "20",
                "evaluation_point check: supported (axis 'step' from the envelope; any other "
                "axis from evaluation.progress)",
            ],
        ),
        (
            {
                "type": "json_envelope",
                "objective_name": "loss",
                "split": "validation",
                "policy": "final",
                "evaluation_axis": "tokens",
            },
            [
                "evaluation_point check: supported (axis 'tokens' from the envelope's "
                "evaluation.progress)",
            ],
        ),
        (
            {"type": "mlflow", "metric_key": "eval/loss", "select": "last"},
            [
                "eval/loss",
                "evaluation_point check: supported (position is the step of the selected "
                "observation)",
            ],
        ),
    ],
)
def test_validate_and_dry_run_explain_scoring_without_sdk(
    tmp_path, monkeypatch, caplog, extractor, details
):
    from phasesweep.config import Metric
    from phasesweep.tracking.models import MlflowSettings

    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    metric = Metric(name="loss", goal="minimize", extractor=extractor)
    if extractor["type"] == "mlflow":
        # The mlflow reader needs the top-level target block, which make_experiment cannot set.
        experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1).model_copy(
            update={
                "metric": metric,
                "mlflow": MlflowSettings(
                    tracking_uri="http://127.0.0.1:5000", experiment_name="phasesweep"
                ),
            }
        )
    else:
        experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1, metric=metric)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
    result = CliRunner().invoke(cli_main, ["validate", str(config)])
    assert result.exit_code == 0, result.output
    assert "goal=minimize" in result.output
    assert all(detail in result.output for detail in details)
    with caplog.at_level(logging.INFO):
        run_experiment(experiment, dry_run=True)
    assert "goal=minimize" in caplog.text
    assert all(detail in caplog.text for detail in details)
    assert not (tmp_path / "runs").exists()


def test_validate_does_not_preflight_ambient_wandb_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation remains useful from a shell reserved for offline W&B work."""
    from phasesweep.config import Metric, WandbExtractor

    experiment = make_experiment(
        workdir=tmp_path / "runs",
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=WandbExtractor(
                type="wandb", entity="entity", project="project", metric_key="eval/loss"
            ),
        ),
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
    monkeypatch.setenv("WANDB_MODE", "offline")

    result = CliRunner().invoke(cli_main, ["validate", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "goal=minimize" in result.output


_BASE_ENV_NAMES_WITHOUT_TMPDIR = "PATH, HOME, LANG, LC_ALL, USER, LOGNAME, TZ"
_REFUSES_TO_RESUME = "a persistent study refuses to resume when any digested value changes"
_ENVIRONMENT_CONTRACT_CASES = [
    pytest.param(
        {},
        "environment: inherit_env=none (inherited when set, and digested: "
        f"{_BASE_ENV_NAMES_WITHOUT_TMPDIR}); env keys: none; "
        "passthrough_env: TMPDIR (built in; inherited when set, excluded from the digest)",
        id="default-in-memory",
    ),
    pytest.param(
        {
            "execution": ExecutionContext(inherit_env=["HF_HOME"], passthrough_env=["HF_TOKEN"]),
            "env": {"B_FLAG": "1", "A_FLAG": "2"},
            "persistent": True,
        },
        "environment: inherit_env=[HF_HOME] (inherited when set, and digested: "
        f"{_BASE_ENV_NAMES_WITHOUT_TMPDIR}, HF_HOME); env keys: A_FLAG, B_FLAG; "
        "passthrough_env: HF_TOKEN and built-in TMPDIR "
        "(inherited when set, excluded from the digest); " + _REFUSES_TO_RESUME,
        id="list-and-passthrough-persistent",
    ),
    pytest.param(
        {"execution": ExecutionContext(inherit_env="all")},
        "environment: inherit_env=all (inherited whole, and digested: "
        f"{_BASE_ENV_NAMES_WITHOUT_TMPDIR}; other ambient variables are not compared); "
        "env keys: none; "
        "passthrough_env: TMPDIR (built in; inherited when set, excluded from the digest)",
        id="all",
    ),
    pytest.param(
        {"execution": ExecutionContext(passthrough_env=["TMPDIR"])},
        "environment: inherit_env=none (inherited when set, and digested: "
        f"{_BASE_ENV_NAMES_WITHOUT_TMPDIR}); env keys: none; "
        "passthrough_env: TMPDIR (built in; inherited when set, excluded from the digest)",
        id="listed-tmpdir-is-redundant",
    ),
    pytest.param(
        {"env": {"TMPDIR": "/scratch"}},
        "environment: inherit_env=none (inherited when set, and digested: "
        f"{_BASE_ENV_NAMES_WITHOUT_TMPDIR}); env keys: TMPDIR; "
        "passthrough_env: none (TMPDIR set under env, so the configured value is digested)",
        id="configured-tmpdir-stays-digested",
    ),
    pytest.param(
        {"execution": ExecutionContext(passthrough_env=["HF_TOKEN"]), "env": {"TMPDIR": "/s"}},
        "environment: inherit_env=none (inherited when set, and digested: "
        f"{_BASE_ENV_NAMES_WITHOUT_TMPDIR}); env keys: TMPDIR; "
        "passthrough_env: HF_TOKEN (inherited when set, excluded from the digest) "
        "(TMPDIR set under env, so the configured value is digested)",
        id="configured-tmpdir-beside-a-listed-token",
    ),
]


def _environment_contract_experiment(tmp_path: Path, case: dict) -> Experiment:
    """Build the experiment behind one environment-contract case.

    :param Path tmp_path: Scratch directory for the workdir and, when persistent, the journal.
    :param dict case: Keyword arguments for :func:`make_experiment`, where ``persistent``
        is a flag that selects journal storage under ``tmp_path``.
    :return Experiment: The experiment for the case.
    """
    kwargs = dict(case)
    if kwargs.pop("persistent", False):
        kwargs["persistent"] = tmp_path
    return make_experiment(workdir=tmp_path / "runs", n_trials=1, **kwargs)


@pytest.mark.parametrize(("case", "expected"), _ENVIRONMENT_CONTRACT_CASES)
def test_validate_discloses_environment_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict, expected: str
) -> None:
    """``validate`` prints the static environment contract and never reads the ambient one."""
    monkeypatch.setenv("PHASESWEEP_TEST_AMBIENT_SECRET", "leak")
    experiment = _environment_contract_experiment(tmp_path, case)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines() if "environment:" in line]
    assert lines == [expected]
    assert "leak" not in result.output
    assert "PHASESWEEP_TEST_AMBIENT_SECRET" not in result.output


@pytest.mark.parametrize(("case", "expected"), _ENVIRONMENT_CONTRACT_CASES)
def test_dry_run_logs_environment_contract(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, case: dict, expected: str
) -> None:
    """``run --dry-run`` logs the same environment line ``validate`` prints."""
    experiment = _environment_contract_experiment(tmp_path, case)

    with caplog.at_level(logging.INFO):
        run_experiment(experiment, dry_run=True)

    lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("DRY RUN environment:")
    ]
    assert lines == [f"DRY RUN {expected}"]


@pytest.mark.parametrize(("case", "expected"), _ENVIRONMENT_CONTRACT_CASES)
def test_environment_contract_line_lists_each_variable_once(
    tmp_path: Path, case: dict, expected: str
) -> None:
    """A variable is digested, set under ``env``, or passed through, never two of them."""
    line = _environment_contract_line(_environment_contract_experiment(tmp_path, case))
    assert line == expected
    digested = re.search(r"digested: ([^)]*)\)", line)
    env_keys = re.search(r"env keys: ([^;]*)", line)
    passed = re.search(r"passthrough_env: ([^(;]*?)(?: \(|;|$)", line)
    names = [
        name
        for match in (digested, env_keys, passed)
        if match is not None
        for name in re.split(r", | and built-in ", match.group(1))
        if name != "none"
    ]
    assert len(names) == len(set(names)), line


@pytest.mark.parametrize("configured", [False, True])
def test_validate_and_dry_run_disclose_the_trainer_cwd(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, configured: bool
) -> None:
    """The trainer cwd line says what pins the directory, since it is part of the study identity."""
    trainer_dir = tmp_path / "trainer"
    trainer_dir.mkdir()
    execution = ExecutionContext(cwd=str(trainer_dir)) if configured else ExecutionContext()
    expected = (
        f"trainer cwd: {trainer_dir.resolve()} (execution.cwd; part of the study identity)"
        if configured
        else "trainer cwd: follows the directory phasesweep runs from (the catalog cwd under "
        "MCP) and is part of the study identity; set execution.cwd to pin it"
    )
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1, execution=execution)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))

    result = CliRunner().invoke(cli_main, ["validate", str(config)])
    with caplog.at_level(logging.INFO):
        run_experiment(experiment, dry_run=True)

    assert result.exit_code == 0, result.output
    assert [line.strip() for line in result.output.splitlines() if "trainer cwd:" in line] == [
        expected
    ]
    assert [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("DRY RUN trainer cwd:")
    ] == [f"DRY RUN {expected}"]


def test_help_registers_commands_and_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"], terminal_width=120)

    assert result.exit_code == 0
    assert "-h, --help" in result.output
    assert "recover-run" not in result.output
    for command in ("explain", "init", "mcp", "run", "show-winners", "status", "validate"):
        assert command in result.output

    for command in ("run", "validate", "show-winners", "status"):
        result = runner.invoke(cli_main, [command, "--help"], terminal_width=120)
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "CONFIG" in result.output
        assert "-h, --help" in result.output

    init_help = runner.invoke(cli_main, ["init", "--help"], terminal_width=120)
    assert init_help.exit_code == 0
    assert "-o, --output PATH" in init_help.output
    assert "--template [starter|architectures|argparse|chained|noisy]" in init_help.output
    assert "--list" in init_help.output

    recovery_help = runner.invoke(cli_main, ["mcp", "recover-run", "--help"], terminal_width=120)
    assert recovery_help.exit_code == 0
    for flag in ("--state-dir", "--run-id", "--confirm", "-h, --help"):
        assert flag in recovery_help.output
    assert (
        "restore the original complete storage ledger and access to it before recovery"
        in " ".join(recovery_help.output.split()).lower()
    )

    run_help = runner.invoke(cli_main, ["run", "--help"], terminal_width=120).output
    assert "--from-phase PHASE" in run_help
    assert "[default: (first phase)]" in run_help
    assert "--dry-run" in run_help
    assert "-v, --verbose" in run_help

    winners_help = runner.invoke(cli_main, ["show-winners", "--help"], terminal_width=120).output
    assert "show-winners [OPTIONS] CONFIG_YAML" in winners_help
    assert "Print published experiment winners." in winners_help

    mcp_help = runner.invoke(cli_main, ["mcp", "--help"], terminal_width=120)
    assert mcp_help.exit_code == 0
    assert "Manage the optional MCP server, catalog, and operator recovery." in mcp_help.output
    assert "integrations" not in mcp_help.output
    for command in ("call", "check", "init-catalog", "recover-run", "serve"):
        assert command in mcp_help.output
    for command in ("check-install", "install", "uninstall"):
        assert command not in mcp_help.output
    assert "--catalog PATH" not in mcp_help.output

    serve_help = runner.invoke(cli_main, ["mcp", "serve", "--help"], terminal_width=120)
    assert serve_help.exit_code == 0
    assert "--catalog PATH" in serve_help.output

    call_help = runner.invoke(cli_main, ["mcp", "call", "--help"], terminal_width=120)
    assert call_help.exit_code == 0
    assert "call [OPTIONS] TOOL [JSON_ARGS]" in call_help.output
    assert "--catalog PATH" in call_help.output

    check_help = runner.invoke(cli_main, ["mcp", "check", "--help"], terminal_width=120)
    assert check_help.exit_code == 0
    assert "--catalog PATH" in check_help.output

    init_help = runner.invoke(cli_main, ["mcp", "init-catalog", "--help"], terminal_width=120)
    assert init_help.exit_code == 0
    assert "--from PATH" in init_help.output
    compact_init_help = "".join(init_help.output.split())
    assert "PHASESWEEP_HOME/mcp/<catalog-digest>" in compact_init_help
    assert "${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>" in compact_init_help
    assert "state_dir next to the catalog" not in init_help.output

    for command in ("install", "uninstall", "check-install"):
        result = runner.invoke(cli_main, ["mcp", command])
        assert result.exit_code == 2
        compact_output = " ".join(result.output.split())
        assert "No such command" in compact_output
        assert command in compact_output


def test_recover_run_expands_user_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    RunStore(tmp_path / "state")

    result = recover_run_cli("~/state", "missing")

    assert result.exit_code != 0
    assert "unknown run id: missing" in result.output
    assert "not an existing MCP state directory" not in result.output


def test_recover_run_surfaces_host_error_suggestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable host must report the remediation, not only the diagnosis.

    ``require_linux_mcp_host`` carries its fix in ``CatalogError.suggestion``;
    rendering only the message would leave the operator with a run stuck in
    ``recovery_required`` and nothing to act on.
    """
    RunStore(tmp_path / "state")

    def refuse_host() -> None:
        raise CatalogError(
            "cannot read this process's Linux /proc start time",
            suggestion="mount /proc with process stat access",
        )

    monkeypatch.setattr("phasesweep.cli.require_linux_mcp_host", refuse_host)

    result = recover_run_cli(tmp_path / "state", "missing")

    assert result.exit_code != 0
    assert "cannot read this process's Linux /proc start time" in result.output
    assert "fix: mount /proc with process stat access" in result.output


@pytest.mark.parametrize(
    ("expected_dry_run", "expected_events"),
    [
        (False, ["signals", "load", "run"]),
        (True, ["signals", "load", "run"]),
    ],
)
def test_run_installs_signal_handlers_before_config_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_dry_run: bool,
    expected_events: list[str],
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    events: list[str] = []
    config = object()

    monkeypatch.setattr("phasesweep.cli.install_signal_handlers", lambda: events.append("signals"))

    def fake_load_experiment(_path: Path) -> object:
        events.append("load")
        return config

    def fake_run_experiment(loaded: object, *, from_phase: str | None, dry_run: bool) -> None:
        assert loaded is config
        assert from_phase is None
        assert dry_run is expected_dry_run
        events.append("run")

    monkeypatch.setattr("phasesweep.cli.load_experiment", fake_load_experiment)
    monkeypatch.setattr("phasesweep.cli.run_experiment", fake_run_experiment)
    args = ["run", str(config_path)]
    if expected_dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(cli_main, args)

    assert result.exit_code == 0, result.output
    assert events == expected_events


def test_validate_cli_renders_comment(tmp_path: Path) -> None:
    """``phasesweep validate`` surfaces phase comments so the operator sees
    design intent next to the spec, with a ``#`` prefix to read as documentation."""
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent("""
        experiment: t
        trial_command: "echo {overrides}"
        override_format: argparse
        metric:
          extractor: { type: json_envelope, objective_name: x, split: test, policy: test }
        phases:
          - name: depth
            comment: |
              first phase: figure out the depth.
              grid because we want every choice to actually run.
            n_trials: 1
            search_space: { x: { type: int, low: 0, high: 1 } }
        """)
    )
    runner = CliRunner()
    result = runner.invoke(cli_main, ["validate", str(p)])
    assert result.exit_code == 0
    assert "first phase: figure out the depth." in result.output
    assert "grid because we want every choice" in result.output
    for line in result.output.splitlines():
        if "first phase" in line or "grid because" in line:
            assert line.lstrip().startswith("#"), f"comment line not prefixed: {line!r}"


def test_validate_cli_discloses_sampler_capability(tmp_path: Path) -> None:
    """``phasesweep validate`` states each phase's resume/reproduce contract up front.

    The runtime guard rejects a mid-target TPE/CMA-ES resume, but only after the
    operator has already been interrupted; the capability line puts the same
    contract in front of them before any trial runs.
    """
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent(f"""
        experiment: t
        storage: journal:///{tmp_path}/phases.journal
        provenance: {{revision: test-fixture-v1}}
        trial_command: "echo {{overrides}}"
        override_format: argparse
        metric:
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: depth
            n_trials: 2
            sampler: {{ type: grid }}
            search_space: {{ d: {{ type: categorical, choices: [4, 8] }} }}
          - name: lr
            n_trials: 2
            sampler: {{ type: tpe, seed: 0 }}
            search_space: {{ lr: {{ type: float, low: 0.1, high: 1.0 }} }}
          - name: wd
            n_trials: 2
            sampler: {{ type: random, seed: 3 }}
            search_space: {{ wd: {{ type: float, low: 0.0, high: 0.3 }} }}
          - name: reg
            n_trials: 2
            n_jobs: 2
            gpu_policy: none
            allow_no_gpu_isolation: true
            sampler: {{ type: cmaes, seed: 1 }}
            search_space: {{ dropout: {{ type: float, low: 0.0, high: 0.3 }} }}
        """)
    )

    result = CliRunner().invoke(cli_main, ["validate", str(p)])

    assert result.exit_code == 0, result.output
    # Grid shuffles with a fixed seed when none is set, so it reproduces unseeded.
    assert "phase 'depth': sampler=grid (resumable, reproducible)" in result.output
    assert "phase 'lr': sampler=tpe seed=0 (resumable, reproducible)" in result.output
    assert "phase 'wd': sampler=random seed=3 (resumable, reproducible)" in result.output
    # Under several workers a model sampler's draws depend on which trials have finished.
    assert "phase 'reg': sampler=cmaes seed=1 (resumable)" in result.output


def test_validate_calls_empty_search_space_a_fixed_configuration(tmp_path: Path) -> None:
    """A phase with no search_space needs no seed and is reported as fixed."""
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent(f"""
        experiment: t
        storage: journal:///{tmp_path}/phases.journal
        provenance: {{revision: test-fixture-v1}}
        trial_command: "echo {{overrides}}"
        override_format: argparse
        metric:
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: baseline
            n_trials: 1
            fixed_overrides: {{ lr: 0.1 }}
        """)
    )

    result = CliRunner().invoke(cli_main, ["validate", str(p)])

    assert result.exit_code == 0, result.output
    assert "- baseline: n_trials=1 (fixed configuration)" in result.output
    assert "phase 'baseline': fixed configuration, no sampler draws (resumable)" in result.output
    assert "sampler=" not in result.output


def test_validate_says_an_in_memory_phase_cannot_resume(tmp_path: Path) -> None:
    """Without persistent storage the study ends with the process, so no phase line claims a resume."""
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent("""
        experiment: t
        trial_command: "echo {overrides}"
        override_format: argparse
        metric:
          extractor: { type: json_envelope, objective_name: x, split: test, policy: test }
        phases:
          - name: lr
            n_trials: 11
            search_space: { lr: { type: float, low: 0.1, high: 1.0 } }
          - name: wd
            n_trials: 2
            sampler: { type: random, seed: 3 }
            search_space: { wd: { type: float, low: 0.0, high: 0.3 } }
          - name: baseline
            n_trials: 1
            fixed_overrides: { lr: 0.1 }
        """)
    )

    result = CliRunner().invoke(cli_main, ["validate", str(p)])

    assert result.exit_code == 0, result.output
    assert "phase 'lr': sampler=tpe (not resumable: in-memory storage)" in result.output
    assert (
        "phase 'wd': sampler=random seed=3 (reproducible, not resumable: in-memory storage)"
        in result.output
    )
    assert (
        "phase 'baseline': fixed configuration, no sampler draws "
        "(not resumable: in-memory storage)" in result.output
    )
    assert "(resumable" not in result.output


_TPE_STARTUP_WARNING = (
    "warning: Phase 'lr': tpe with n_trials=5 <= n_startup_trials=10, so every trial is "
    "sampled at random and TPE never fits its model. Use sampler.type: random with a seed "
    "(same coverage), or set sampler.n_startup_trials below n_trials."
)
_LR_SPACE = "{ lr: { type: float, low: 0.1, high: 1.0 } }"

# Phase bodies, after the name, for which no startup warning applies.
_QUIET_TPE_BUDGETS = [
    pytest.param(f"n_trials: 11\nsearch_space: {_LR_SPACE}", id="budget-exceeds-startup"),
    pytest.param(
        f"n_trials: 5\nsampler: {{ type: tpe, n_startup_trials: 4 }}\nsearch_space: {_LR_SPACE}",
        id="startup-below-budget",
    ),
    pytest.param(
        f"n_trials: 5\nsampler: {{ type: random }}\nsearch_space: {_LR_SPACE}",
        id="random-sampler",
    ),
    pytest.param("n_trials: 1\nfixed_overrides: { lr: 0.1 }", id="empty-search-space"),
]


def _write_tpe_budget_config(tmp_path: Path, phase_body: str) -> Path:
    """Write an in-memory experiment whose single phase ``lr`` has ``phase_body``."""
    path = tmp_path / "exp.yaml"
    body = textwrap.indent(phase_body, "    ")
    path.write_text(
        textwrap.dedent("""
        experiment: t
        trial_command: "echo {overrides}"
        override_format: argparse
        metric:
          extractor: { type: json_envelope, objective_name: x, split: test, policy: test }
        phases:
          - name: lr
        """)
        + body
        + "\n"
    )
    return path


def test_validate_warns_when_tpe_never_leaves_startup(tmp_path: Path) -> None:
    """A tpe budget at or below its startup trials is random search; validate says so."""
    path = _write_tpe_budget_config(tmp_path, f"n_trials: 5\nsearch_space: {_LR_SPACE}")

    result = CliRunner().invoke(cli_main, ["validate", str(path)])

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.splitlines()]
    capability = lines.index("phase 'lr': sampler=tpe (not resumable: in-memory storage)")
    assert lines[capability + 1] == _TPE_STARTUP_WARNING


def test_validate_warns_at_the_startup_boundary(tmp_path: Path) -> None:
    path = _write_tpe_budget_config(tmp_path, f"n_trials: 10\nsearch_space: {_LR_SPACE}")

    result = CliRunner().invoke(cli_main, ["validate", str(path)])

    assert "tpe with n_trials=10 <= n_startup_trials=10" in result.output


@pytest.mark.parametrize("phase_body", _QUIET_TPE_BUDGETS)
def test_validate_does_not_warn_when_tpe_leaves_startup(tmp_path: Path, phase_body: str) -> None:
    result = CliRunner().invoke(
        cli_main, ["validate", str(_write_tpe_budget_config(tmp_path, phase_body))]
    )

    assert result.exit_code == 0, result.output
    assert "warning:" not in result.output


def test_dry_run_warns_when_tpe_never_leaves_startup(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="lr",
                n_trials=5,
                search_space={"lr": FloatParam(type="float", low=0.1, high=1.0)},
            )
        ],
    )

    run_experiment(exp, dry_run=True)

    warnings = [r for r in caplog.records if r.getMessage().startswith("DRY RUN warning:")]
    assert [(r.levelno, r.getMessage()) for r in warnings] == [
        (logging.WARNING, f"DRY RUN {_TPE_STARTUP_WARNING}")
    ]


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param(
            Phase(
                name="lr",
                n_trials=11,
                search_space={"lr": FloatParam(type="float", low=0.1, high=1.0)},
            ),
            id="budget-exceeds-startup",
        ),
        pytest.param(
            Phase(
                name="lr",
                n_trials=5,
                sampler=Sampler(type="random", seed=0),
                search_space={"lr": FloatParam(type="float", low=0.1, high=1.0)},
            ),
            id="random-sampler",
        ),
        pytest.param(
            Phase(name="lr", n_trials=1, fixed_overrides={"lr": 0.1}), id="empty-search-space"
        ),
    ],
)
def test_dry_run_does_not_warn_when_tpe_leaves_startup(
    tmp_path: Path, caplog, phase: Phase
) -> None:
    caplog.set_level(logging.INFO)
    exp = make_experiment(trial_command="echo {overrides}", phases=[phase])

    run_experiment(exp, dry_run=True)

    assert not [r for r in caplog.records if r.getMessage().startswith("DRY RUN warning:")]


_ARGPARSE_BOOLEAN_WARNING = (
    "warning: argparse renders booleans as --key=true/--key=false; a store_true flag rejects a "
    "value, so set such flags in trial_command or have the trainer accept a value. "
    "Rendered that way: --load_best_model_at_end (trainer_config), "
    "--fp16 (phase 'p' fixed_overrides), --shuffle (phase 'p' search_space)."
)


def _boolean_experiment(tmp_path: Path, **overrides: object) -> Experiment:
    """Build an experiment with a boolean in its trainer config, a fixed override and a choice."""
    return make_experiment(
        workdir=tmp_path / "runs",
        trainer_config={"load_best_model_at_end": True, "epochs": 3},
        phases=[
            Phase(
                name="p",
                n_trials=2,
                sampler=Sampler(type="grid"),
                fixed_overrides={"fp16": False},
                search_space={
                    "shuffle": CategoricalParam(type="categorical", choices=[True, False])
                },
            )
        ],
        **overrides,
    )


def test_validate_and_dry_run_warn_when_argparse_renders_a_boolean(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One warning lists every flag argparse would hand the trainer as ``--key=true``."""
    experiment = _boolean_experiment(tmp_path)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    assert [line.strip() for line in result.output.splitlines() if "warning:" in line] == [
        _ARGPARSE_BOOLEAN_WARNING
    ]
    caplog.set_level(logging.INFO)
    run_experiment(experiment, dry_run=True)
    warnings = [r for r in caplog.records if r.getMessage().startswith("DRY RUN warning:")]
    assert [(r.levelno, r.getMessage()) for r in warnings] == [
        (logging.WARNING, f"DRY RUN {_ARGPARSE_BOOLEAN_WARNING}")
    ]


@pytest.mark.parametrize(
    ("override_format", "trial_command"),
    [
        pytest.param("hydra", "echo {overrides}", id="hydra"),
        pytest.param("yaml_file", "echo {config_path}", id="yaml_file"),
        pytest.param("json_file", "echo {overrides_path}", id="json_file"),
    ],
)
def test_boolean_warning_is_only_for_argparse(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, override_format: str, trial_command: str
) -> None:
    """Hydra and the file formats carry a boolean as a value, so they stay quiet."""
    experiment = _boolean_experiment(
        tmp_path, override_format=override_format, trial_command=trial_command
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    assert "warning:" not in result.output
    caplog.set_level(logging.INFO)
    run_experiment(experiment, dry_run=True)
    assert not [r for r in caplog.records if r.getMessage().startswith("DRY RUN warning:")]


_CREDENTIAL_WARNING = (
    "warning: execution.inherit_env lists names that look like credentials: {names}. "
    "Each trial records the value of every listed name in plain text in the study ledger, "
    "and rotating one refuses a resume. Credentials belong in execution.passthrough_env, "
    "which is inherited without being recorded or compared."
)


@pytest.mark.parametrize(
    ("inherit_env", "named"),
    [
        pytest.param(
            ["HF_HOME", "HF_TOKEN", "api_key", "DB_Password", "AWS_SECRET_ACCESS_KEY"],
            "AWS_SECRET_ACCESS_KEY, DB_Password, HF_TOKEN, api_key",
            id="each-marker-in-any-case",
        ),
        pytest.param(["MY_CREDENTIALS"], "MY_CREDENTIALS", id="credential"),
        pytest.param(["HF_HOME", "PYTHONPATH"], None, id="plain-names"),
        pytest.param("none", None, id="none"),
        pytest.param("all", None, id="all"),
    ],
)
def test_validate_and_dry_run_warn_about_a_credential_in_inherit_env(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    inherit_env: str | list[str],
    named: str | None,
) -> None:
    """A listed name that looks like a credential is warned about, never refused."""
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        execution=ExecutionContext(inherit_env=inherit_env, passthrough_env=["HF_PASSTHROUGH_KEY"]),
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
    expected = [] if named is None else [_CREDENTIAL_WARNING.format(names=named)]

    result = CliRunner().invoke(cli_main, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    printed = [line.strip() for line in result.output.splitlines() if "warning:" in line]
    assert [line for line in printed if "inherit_env" in line] == expected
    caplog.set_level(logging.INFO)
    run_experiment(experiment, dry_run=True)
    logged = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("DRY RUN warning:") and "inherit_env" in r.getMessage()
    ]
    assert logged == [f"DRY RUN {line}" for line in expected]


_REPEATS_BUDGET_WARNING = (
    "warning: Phase 'lr': max_consecutive_failures=3 <= 3 repeats.values, so one configuration "
    "that fails on every repeat value can abort the phase by itself; under seeded random its "
    "repeats run back to back. Set max_consecutive_failures above 3."
)
_REPEATS_BODY = (
    "n_trials: 6\nsampler: { type: random, seed: 0 }\nsearch_space: "
    f"{_LR_SPACE}\nrepeats: {{ key: seed, values: [0, 1, 2] }}\n"
)


@pytest.mark.parametrize(("limit", "warns"), [(3, True), (2, True), (4, False)])
def test_validate_warns_when_one_configuration_can_exhaust_the_failure_budget(
    tmp_path: Path, limit: int, warns: bool
) -> None:
    """A warning, never a refusal: the run is still the operator's call."""
    body = _REPEATS_BODY + f"max_consecutive_failures: {limit}"
    result = CliRunner().invoke(
        cli_main, ["validate", str(_write_tpe_budget_config(tmp_path, body))]
    )

    assert result.exit_code == 0, result.output
    warnings = [line.strip() for line in result.output.splitlines() if "warning:" in line]
    expected = _REPEATS_BUDGET_WARNING.replace("=3 <=", f"={limit} <=")
    assert warnings == ([expected] if warns else [])


def test_dry_run_warns_when_one_configuration_can_exhaust_the_failure_budget(
    tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="lr",
                n_trials=6,
                max_consecutive_failures=3,
                sampler=Sampler(type="random", seed=0),
                search_space={"lr": FloatParam(type="float", low=0.1, high=1.0)},
                repeats=Repeats(key="seed", values=[0, 1, 2]),
            )
        ],
    )

    run_experiment(exp, dry_run=True)

    warnings = [r for r in caplog.records if r.getMessage().startswith("DRY RUN warning:")]
    assert [(r.levelno, r.getMessage()) for r in warnings] == [
        (logging.WARNING, f"DRY RUN {_REPEATS_BUDGET_WARNING}")
    ]


def test_show_winners_renders_comment_before_winner(tmp_path: Path) -> None:
    """``show-winners`` prints comment before the winner block so the reader
    frames numerical results against intent. Also covers the no-winner-yet
    branch — the comment is still surfaced even before a phase has run."""

    def make_cfg(workdir: Path) -> Path:
        cfg = tmp_path / "exp.yaml"
        experiment = make_experiment(
            workdir=workdir,
            trial_command="echo x=0.5 {overrides}",
            name="depth",
            comment="settle the depth before anything else.",
            n_trials=1,
            sampler=Sampler(type="random", seed=0),
        )
        cfg.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
        return cfg

    runner = CliRunner()

    # With a winner: comment must come BEFORE the winner block.
    config_with = make_cfg(tmp_path / "wd")
    run_experiment(load_experiment(config_with))
    result_with = runner.invoke(cli_main, ["show-winners", str(config_with)])
    assert result_with.exit_code == 0
    comment_line = "# settle the depth before anything else."
    metric_line = "trial_number: 0"
    assert comment_line in result_with.output
    assert metric_line in result_with.output
    assert result_with.output.index(comment_line) < result_with.output.index(metric_line)

    # Without a winner (different workdir → no winner.yaml): comment is still surfaced.
    result_without = runner.invoke(cli_main, ["show-winners", str(make_cfg(tmp_path / "empty_wd"))])
    assert result_without.exit_code == 0
    assert "(no winner yet)" in result_without.output
    assert comment_line in result_without.output


def test_show_winners_uses_only_the_last_successful_generation(tmp_path: Path) -> None:
    """Mutable compatibility files must not outrank immutable generation results."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment, config_path = materialized.experiment, materialized.config_path
    _winner_path(experiment, "p").write_text("trial_number: 99\n")
    _generation_path(experiment).write_text("generation_id: interrupted\n")

    published = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])

    assert published.exit_code == 0
    assert "trial_number: 0" in published.output
    assert "trial_number: 99" not in published.output

    _last_successful_generation_path(experiment).unlink()
    unpublished = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])

    assert unpublished.exit_code == 0
    assert "(no winner yet)" in unpublished.output
    assert "trial_number: 99" not in unpublished.output


def test_show_winners_rejects_a_foreign_storage_ledger(tmp_path: Path) -> None:
    """Winner-only CLI reads enforce the artifact tree's reverse ownership."""
    owner_config = materialize("current-journal", tmp_path, mode="tree").config_path
    foreign_config = tmp_path / "foreign.yaml"
    foreign_config.write_text(owner_config.read_text().replace("study.journal", "foreign.journal"))
    result = CliRunner().invoke(cli_main, ["show-winners", str(foreign_config)])

    assert result.exit_code == 1
    assert isinstance(result.exception, ArtifactRootConflictError)
    assert "different storage ledger" in str(result.exception)
    assert "trial_number" not in result.output


def test_dry_run_does_not_launch(tmp_path, caplog, monkeypatch):
    """Dry-run should preview one coherent chain without launching anything."""

    caplog.set_level(logging.INFO)
    exp = make_experiment(
        experiment="dry",
        persistent=tmp_path,
        trial_command="false {overrides}",
        metric=Metric(
            name="loss",
            extractor=JsonEnvelopeExtractor(
                type="json_envelope", objective_name="loss", split="test", policy="test"
            ),
        ),
        phases=[
            Phase(
                name="a",
                n_trials=5,
                sampler=Sampler(type="random", seed=0),
                search_space={"lr": FloatParam(type="float", low=1e-5, high=1e-2, log=True)},
            ),
            Phase(
                name="b",
                inherits=["a"],
                n_trials=5,
                sampler=Sampler(type="random", seed=0),
                search_space={"wd": FloatParam(type="float", low=0, high=0.3)},
            ),
        ],
    )
    monkeypatch.setattr(
        "phasesweep.engine.preview._suggest",
        lambda _trial, _name, param: param.low,
    )
    winners = run_experiment(exp, dry_run=True)
    assert set(winners) == {"a", "b"}
    # No filesystem artifacts written.
    assert not (Path(tmp_path / "runs") / "dry").exists()
    assert not (Path(tmp_path / "runs") / "summary.yaml").exists()
    # An example command was logged
    assert any("DRY RUN example command" in r.message for r in caplog.records)
    # The sampler capability of every phase is disclosed before any trial would run.
    logged = [r.getMessage() for r in caplog.records]
    for phase_name in ("a", "b"):
        assert any(
            f"phase '{phase_name}': sampler=random seed=0 (resumable, reproducible)" in message
            for message in logged
        ), logged
    assert winners["a"].params["lr"] == exp.phases[0].search_space["lr"].low
    assert winners["b"].effective_overrides["lr"] == winners["a"].params["lr"]


@pytest.mark.parametrize(
    "search_space",
    [{"lr": CategoricalParam(type="categorical", choices=[0.1])}, {}],
    ids=["one-categorical-choice", "empty-search-space"],
)
def test_dry_run_previews_a_one_point_grid(tmp_path, caplog, search_space):
    """A one-point grid, or a phase with no search space, previews its one sample."""
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        persistent=tmp_path,
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="p",
                n_trials=1,
                sampler=Sampler(type="grid"),
                search_space=search_space,
                fixed_overrides={"x": 1},
            )
        ],
    )
    winners = run_experiment(exp, dry_run=True)
    assert set(winners) == {"p"}
    assert any("DRY RUN example command" in r.message for r in caplog.records)


def test_dry_run_marks_inherited_values_as_placeholders(tmp_path, caplog):
    """A preview command built from a previewed parent says its inherited values are placeholders."""
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        persistent=tmp_path,
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="a",
                n_trials=1,
                sampler=Sampler(type="grid"),
                search_space={"depth": CategoricalParam(type="categorical", choices=[6])},
            ),
            Phase(
                name="b",
                inherits=["a"],
                n_trials=1,
                sampler=Sampler(type="grid"),
                search_space={"lr": CategoricalParam(type="categorical", choices=[0.1])},
            ),
        ],
    )

    winners = run_experiment(exp, dry_run=True)

    commands = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("DRY RUN example")
    ]
    assert len(commands) == 2
    assert commands[0].startswith("DRY RUN example command:\n")
    assert commands[1].startswith(
        "DRY RUN example command (placeholders until 'a' publishes a winner: depth):\n"
    )
    assert "--depth=6" in commands[1]
    assert winners["a"].placeholder_keys == frozenset({"depth"})
    assert winners["b"].placeholder_keys == frozenset({"depth", "lr"})
    published = Winner(trial_number=3, params={}, effective_overrides={}, metric=1.0)
    assert published.placeholder_keys == frozenset()


def test_dry_run_names_the_placeholder_keys_by_parent(tmp_path, caplog):
    """The note names the inherited keys that are stand-ins, grouped by the parent that must publish.

    ``c`` inherits a key from each of two placeholder parents. ``d`` inherits from ``c`` alone,
    so it carries the key ``a`` and ``b`` gave ``c``, which stays a stand-in two phases down,
    and it fixes ``depth`` itself, which is then a real value and goes unnamed. ``b``'s fixed
    ``opt`` is real from the start.
    """
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        persistent=tmp_path,
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="a",
                n_trials=1,
                sampler=Sampler(type="grid"),
                search_space={"depth": CategoricalParam(type="categorical", choices=[6])},
            ),
            Phase(
                name="b",
                n_trials=1,
                sampler=Sampler(type="grid"),
                fixed_overrides={"opt": "adam"},
                search_space={"width": CategoricalParam(type="categorical", choices=[2])},
            ),
            Phase(
                name="c",
                inherits=["a", "b"],
                n_trials=1,
                sampler=Sampler(type="grid"),
                search_space={"lr": CategoricalParam(type="categorical", choices=[0.1])},
            ),
            Phase(
                name="d",
                inherits=["c"],
                n_trials=1,
                sampler=Sampler(type="grid"),
                fixed_overrides={"depth": 12},
                search_space={"wd": CategoricalParam(type="categorical", choices=[0.01])},
            ),
        ],
    )

    run_experiment(exp, dry_run=True)

    commands = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("DRY RUN example")
    ]
    assert [command.split(":\n")[0] for command in commands] == [
        "DRY RUN example command",
        "DRY RUN example command",
        "DRY RUN example command (placeholders until 'a' publishes a winner: depth; "
        "until 'b' publishes a winner: width)",
        "DRY RUN example command (placeholders until 'c' publishes a winner: lr, width)",
    ]
    assert "--opt=adam" in commands[2]
    assert "--depth=12" in commands[3]


def test_dry_run_leaves_a_parent_that_only_fixes_values_unnamed(tmp_path, caplog):
    """A parent that publishes nothing new supplies no stand-ins, so its child's command carries no note."""
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        persistent=tmp_path,
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="a", n_trials=1, sampler=Sampler(type="grid"), fixed_overrides={"opt": "adam"}
            ),
            Phase(
                name="b",
                inherits=["a"],
                n_trials=1,
                sampler=Sampler(type="grid"),
                search_space={"lr": CategoricalParam(type="categorical", choices=[0.1])},
            ),
        ],
    )

    run_experiment(exp, dry_run=True)

    commands = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("DRY RUN example")
    ]
    assert len(commands) == 2
    assert "placeholders" not in commands[1]
    assert "--opt=adam" in commands[1]


def test_dry_run_discloses_each_phases_sampler_capability(tmp_path, caplog):
    """``run --dry-run`` states every phase's resume/reproduce contract, as ``validate`` does.

    The empty-search-space phase is the fixed configuration: it draws nothing, so
    it owes no seed or acknowledgement on persistent storage and says so.
    """
    caplog.set_level(logging.INFO)
    exp = make_experiment(
        persistent=tmp_path,
        trial_command="echo {overrides}",
        phases=[
            Phase(
                name="depth",
                n_trials=2,
                sampler=Sampler(type="grid"),
                search_space={"d": CategoricalParam(type="categorical", choices=[4, 8])},
            ),
            Phase(
                name="lr",
                n_trials=2,
                sampler=Sampler(type="tpe", seed=0),
                search_space={"lr": FloatParam(type="float", low=0.1, high=1.0)},
            ),
            Phase(
                name="wd",
                n_trials=2,
                sampler=Sampler(type="random", seed=3),
                search_space={"wd": FloatParam(type="float", low=0.0, high=0.3)},
            ),
            Phase(name="fixed", n_trials=1, fixed_overrides={"x": 1}),
        ],
    )

    run_experiment(exp, dry_run=True)

    assert [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("DRY RUN phase '")
    ] == [
        "DRY RUN phase 'depth': sampler=grid (resumable, reproducible)",
        "DRY RUN phase 'lr': sampler=tpe seed=0 (resumable, reproducible)",
        "DRY RUN phase 'wd': sampler=random seed=3 (resumable, reproducible)",
        "DRY RUN phase 'fixed': fixed configuration, no sampler draws (resumable)",
    ]


@pytest.mark.integration
def test_status_cli_reports_phase_counts(tmp_path: Path) -> None:
    """``phasesweep status`` is read-only and reports study trial state counts."""
    experiment = make_experiment(
        experiment="status_test",
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path, value=1.0),
        n_trials=1,
    )
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
    run_experiment(load_experiment(p))

    result = CliRunner().invoke(cli_main, ["status", str(p)])
    assert result.exit_code == 0
    assert "status_test" in result.output
    assert "COMPLETE: 1" in result.output

    # The published generation now backs the phase's winner; current and
    # published identity must both be shown explicitly, never one unlabeled id.
    status_obj = yaml.safe_load(result.output)
    assert status_obj["current_generation_id"] is not None
    assert status_obj["published_generation_id"] == status_obj["current_generation_id"]
    # The ledger file Optuna's own CLI and dashboard are pointed at.
    assert status_obj["ledger_path"] == str((tmp_path / "studies.journal").resolve())


@pytest.mark.integration
def test_status_refuses_an_unreadable_journal_storage_at_format_boundary(tmp_path: Path) -> None:
    """An unreadable journal cannot be classified as current-format state.

    Only a bad line that another line follows is unreadable: a bad last line
    alone is skipped on reads, as Optuna's own reader skips it.
    """
    ledger = tmp_path / "studies.journal"
    ledger.write_text("not a journal record\nanother bad record\n")
    experiment = make_experiment(
        storage=f"journal:///{ledger}",
        workdir=tmp_path / "runs",
        n_trials=1,
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))

    result = subprocess.run(
        [sys.executable, "-m", "phasesweep", "status", str(config_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1
    assert str(ledger) in result.stderr
    assert "format boundary" in result.stderr


def test_show_winners_renders_historical_annotations_on_config_drift(tmp_path: Path) -> None:
    """A published result keeps its own comments and is labeled historical on drift.

    The experiment CLI must not decorate a historical winner with the
    *current* phase comments; it uses the saved annotations and prints an
    explicit marker when the current config no longer matches the published
    one.
    """
    trainer = write_param_echo_trainer(tmp_path)
    config_path = tmp_path / "exp.yaml"

    def write_config(comment: str, metric_name: str) -> None:
        pattern = f"{metric_name}=(?P<value>[0-9.eE+-]+)"
        experiment = make_experiment(
            experiment="drift_cli",
            workdir=tmp_path / "runs",
            trainer=trainer,
            metric=Metric(
                name=metric_name, extractor=LogRegexExtractor(type="log_regex", pattern=pattern)
            ),
            comment=comment,
            n_trials=1,
            sampler=Sampler(type="random", seed=0),
        )
        config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))

    write_config("original hypothesis", "x")
    run_experiment(load_experiment(config_path))

    # Semantic drift (metric rename) plus a new comment on the same phase.
    write_config("new hypothesis", "y")

    result = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])
    assert result.exit_code == 0
    assert "Historical experiment result" in result.output
    assert "# original hypothesis" in result.output
    assert "new hypothesis" not in result.output


@pytest.mark.parametrize(
    ("artifact", "appended"),
    [
        pytest.param("phases/p/winner.yaml", b"\n# edited after publication\n", id="winner"),
        pytest.param("reproducibility.json", b"\n", id="reproducibility_record"),
    ],
)
def test_status_and_show_winners_report_a_corrupt_publication_and_exit_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    artifact: str,
    appended: bytes,
) -> None:
    """A corrupt publication is not a fresh tree.

    ``status`` must not print a payload byte-identical to a never-run workdir
    and exit 0, or the operator's natural next move -- re-run -- would advance
    the pointer and erase the only evidence of corruption. ``show-winners``
    must likewise not answer "no winner yet" over it. Both commands are read-only,
    so they share one corrupted tree. Editing a published winner artifact or
    the claim-time reproducibility record (which feeds the same reporting as
    any winner) stops the generation manifest from validating.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    config_path = materialized.config_path
    generation_id = published_generation_id(materialized.experiment)
    assert generation_id is not None
    tampered = _generation_dir(materialized.experiment, generation_id) / artifact
    tampered.write_bytes(tampered.read_bytes() + appended)

    exit_code = invoke_cli_boundary(["status", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    # The payload still prints in full, on stdout, before the boundary reports.
    payload = yaml.safe_load(captured.out)
    assert payload["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in payload["publication_error"]
    assert payload["published_generation_id"] is None
    # The diagnostic names the failure and forbids the wrong move explicitly.
    assert generation_id in captured.err
    assert "does not match its recorded hash" in captured.err
    assert "Do not run anything over this tree" in captured.err

    exit_code = invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert "no winner yet" not in captured.out
    assert generation_id in captured.err
    assert "does not match its recorded hash" in captured.err
    assert "Do not run anything over this tree" in captured.err


@requires_nonroot
def test_status_reports_an_unreadable_snapshot_as_permission_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A snapshot this user may not read reports permission, not corruption.

    ``config.snapshot.yaml`` is owner-only, so a second operator inspecting a
    healthy tree must not be told the publication no longer validates and to
    restore the generation namespace. The read still fails closed -- nothing
    unvalidatable may read as published -- but the reason names the permission
    denial and the user who can validate it.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    config_path, experiment = materialized.config_path, materialized.experiment
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    snapshot = _generation_dir(experiment, generation_id) / "config.snapshot.yaml"
    original_mode = stat.S_IMODE(snapshot.stat().st_mode)
    snapshot.chmod(0o000)
    try:
        exit_code = invoke_cli_boundary(["status", str(config_path)], monkeypatch)
        captured = capsys.readouterr()
    finally:
        snapshot.chmod(original_mode)

    assert exit_code == 1
    assert "Traceback" not in captured.err
    payload = yaml.safe_load(captured.out)
    assert payload["publication_integrity"] == "permission_denied"
    assert "permission denied" in payload["publication_error"]
    assert "missing or unreadable" not in payload["publication_error"]
    assert "permission denied" in captured.err
    assert "not evidence of corruption" in captured.err
    assert "only the publishing user" in captured.err


def test_status_and_show_winners_stay_successful_without_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A healthy publication and a never-published tree both stay exit 0."""
    fresh_config = tmp_path / "fresh.yaml"
    experiment = make_experiment(workdir=tmp_path / "runs")
    fresh_config.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))

    assert invoke_cli_boundary(["status", str(fresh_config)], monkeypatch) == 0
    fresh = capsys.readouterr()
    assert yaml.safe_load(fresh.out)["publication_integrity"] == "absent"
    assert invoke_cli_boundary(["show-winners", str(fresh_config)], monkeypatch) == 0
    capsys.readouterr()

    config_path = materialize("current-journal", tmp_path, mode="tree").config_path

    assert invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 0
    published = capsys.readouterr()
    assert yaml.safe_load(published.out)["publication_integrity"] == "ok"
    assert invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch) == 0
    assert "trial_number" in capsys.readouterr().out


def test_status_trials_renders_rows_per_phase_and_keeps_publication_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``status --trials`` prints a table per phase, leaves plain status as YAML, and exits alike."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    config_path = materialized.config_path
    generation_id = published_generation_id(materialized.experiment)
    assert generation_id is not None
    heading = "=== p === (2 trials)"

    assert invoke_cli_boundary(["status", str(config_path), "--trials"], monkeypatch) == 0
    lines = capsys.readouterr().out.splitlines()
    assert generation_id in lines[0] and "winner (*)" in lines[0]
    assert lines[lines.index(heading) + 1].split() == [
        "trial",
        "state",
        "value",
        "params",
        "duration_s",
        "winner",
    ]
    rows = lines[lines.index(heading) + 2 :]
    assert [row.split()[:2] for row in rows] == [["0", "COMPLETE"], ["1", "COMPLETE"]]
    # The fixture's two trials tie on the metric, and the tie goes to the lower draw position.
    assert rows[0].endswith("*") and not rows[1].endswith("*")

    assert invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 0
    assert yaml.safe_load(capsys.readouterr().out)["kind"] == "experiment"

    tampered = _generation_dir(materialized.experiment, generation_id) / "phases/p/winner.yaml"
    tampered.write_bytes(tampered.read_bytes() + b"\n# edited after publication\n")

    exit_code = invoke_cli_boundary(["status", str(config_path), "--trials"], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    # The tables still print, with no winner starred, before the boundary reports.
    assert "no generation is published, so no winner is marked" in captured.out
    assert heading in captured.out
    assert not any(line.endswith("*") for line in captured.out.splitlines())
    assert generation_id in captured.err
    assert "does not match its recorded hash" in captured.err
    assert "Do not run anything over this tree" in captured.err


def _stub_run_command(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Make ``phasesweep run`` reach the engine and fail with ``error``.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to replace CLI collaborators.
    :param BaseException error: Exception ``run_experiment`` raises once the CLI calls it.
    """
    monkeypatch.setattr("phasesweep.cli.install_signal_handlers", lambda: None)
    monkeypatch.setattr("phasesweep.cli.load_experiment", lambda _path: object())

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("phasesweep.cli.run_experiment", fail)


@pytest.mark.parametrize(
    ("published_result_committed", "expects_notice"),
    [
        pytest.param(False, False, id="ordinary-pre-publication-shutdown"),
        pytest.param(True, True, id="post-publication-shutdown"),
    ],
)
def test_cli_only_explains_marked_post_publication_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    published_result_committed: bool,
    expects_notice: bool,
) -> None:
    """Only the engine's post-publication shutdown marker permits the CLI notice."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    shutdown = PhaseSweepShutdown(
        signal.SIGTERM,
        ShutdownCleanupReport(
            signum=signal.SIGTERM,
            cleanup_confirmed=True,
            child_pgids=(),
        ),
    )
    assert shutdown.published_result_committed is False
    if published_result_committed:
        shutdown.published_result_committed = True
    _stub_run_command(monkeypatch, shutdown)

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 128 + signal.SIGTERM
    notice = "shutdown was honored after the published result was committed"
    assert (notice in captured.err) is expects_notice


def test_cli_trial_cleanup_refusal_names_the_cli_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A trial cleanup refusal routes to recovery, which for a CLI run is running it again.

    The next ``phasesweep run`` preflight retries the recorded cleanup before it
    launches anything, so that is the step the CLI operator is told.
    """
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    refusal = UnsafeProcessCleanupError("Trial 0 cleanup could not be confirmed.")
    _stub_run_command(monkeypatch, refusal)

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.strip() == (
        "phasesweep: Trial 0 cleanup could not be confirmed. To retry its cleanup, "
        "run `phasesweep run` again with the same config."
    )


@pytest.mark.parametrize(
    "config_text",
    [
        pytest.param("experiment: t\nphases:\n  - name: a\n   n_trials: 1\n", id="syntax_error"),
        pytest.param("experiment: t\nphases: []\n", id="schema_validation_error"),
    ],
)
def test_cli_boundary_reports_config_error_without_traceback(
    config_text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bad config is bad input: exit 2, the file named, no traceback.

    Bad indentation is one of the two most common YAML mistakes, and PyYAML
    labels it ``in "<unicode string>"``; without the config loader's own source
    label the operator cannot tell which file failed. A schema validation error
    must name the file the same way.
    """
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(config_text)

    exit_code = invoke_cli_boundary(["validate", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert str(config_path) in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def _example_config_text(change: Callable[[dict], None]) -> str:
    """Return the minimal starting example as YAML after a change to its mapping.

    :param Callable[[dict], None] change: Edit applied to the loaded example mapping.
    :return str: YAML text of the changed mapping.
    """
    example = Path(__file__).resolve().parent.parent / "examples" / "example.yaml"
    data = yaml.safe_load(example.read_text())
    change(data)
    return yaml.safe_dump(data)


def test_config_error_appends_explain_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A validation error names the reference key it points into, as a runnable command."""

    def bad_sampler(data: dict) -> None:
        data["phases"][0]["sampler"] = {"type": "bogus"}

    config_path = tmp_path / "bad_sampler.yaml"
    config_path.write_text(_example_config_text(bad_sampler))

    exit_code = invoke_cli_boundary(["validate", str(config_path)], monkeypatch)

    err = capsys.readouterr().err
    assert exit_code == 2
    assert str(config_path) in err
    assert "Traceback" not in err
    hint = "see: phasesweep explain 'phases[].sampler.type'"
    assert err.rstrip().splitlines()[-1] == hint
    # The hint is a command the operator can paste.
    explained = CliRunner().invoke(cli_main, shlex.split(hint.removeprefix("see: phasesweep ")))
    assert explained.exit_code == 0
    assert explained.output.startswith("phases[].sampler.type: ")


def test_config_error_hints_stay_few_and_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Five errors under five keys give the first three hints; an error with no location gives none."""

    def many_errors(data: dict) -> None:
        data["trial_command"] = 3
        data["metric"]["extractor"]["path"] = 3
        data["phases"][0]["n_trials"] = "many"
        data["phases"][0]["gpu_policy"] = "bogus"
        data["env"] = {"A": 1}

    config_path = tmp_path / "many.yaml"
    config_path.write_text(_example_config_text(many_errors))
    assert invoke_cli_boundary(["validate", str(config_path)], monkeypatch) == 2
    hints = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("see: phasesweep")
    ]
    assert hints == [
        "see: phasesweep explain trial_command",
        "see: phasesweep explain metric.extractor.path",
        "see: phasesweep explain 'phases[].n_trials'",
    ]

    def model_level_only(data: dict) -> None:
        data["phases"][0]["inherits"] = ["baseline"]

    config_path.write_text(_example_config_text(model_level_only))
    assert invoke_cli_boundary(["validate", str(config_path)], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "is not a prior phase" in err
    assert "see:" not in err


@pytest.mark.parametrize(
    ("verbose", "expects_traceback"),
    [
        pytest.param(False, False, id="concise"),
        pytest.param(True, True, id="verbose"),
    ],
)
def test_cli_boundary_reports_expected_run_failure(
    verbose: bool,
    expects_traceback: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected failures gain a traceback only when verbose logging is active."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, NoFeasibleTrialError("no feasible trial in phase 'depth'"))

    argv = ["run", str(config_path)]
    if verbose:
        argv.append("-v")
    exit_code = invoke_cli_boundary(argv, monkeypatch, debug=verbose)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no feasible trial in phase 'depth'" in captured.err
    assert ("Traceback" in captured.err) is expects_traceback
    if not verbose:
        assert "Traceback" not in captured.out


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [("WANDB_MODE", "offline"), ("WANDB_DISABLED", "true")],
)
def test_cli_boundary_rejects_ambient_offline_wandb_before_generation(
    env_name: str,
    env_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from phasesweep.config import Metric, WandbExtractor

    experiment = make_experiment(
        workdir=tmp_path / "runs",
        execution=ExecutionContext(inherit_env=[env_name]),
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=WandbExtractor(
                type="wandb", entity="entity", project="project", metric_key="eval/loss"
            ),
        ),
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment.model_dump(mode="json")))
    monkeypatch.setenv(env_name, env_value)

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "W&B evidence requires online logging" in captured.err
    assert "internal error" not in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "runs").exists()


def test_cli_boundary_classifies_relative_lock_directory_as_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real lock-directory validator reaches exit 1 without a traceback."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", "relative-locks")
    monkeypatch.setattr("phasesweep.cli.install_signal_handlers", lambda: None)
    monkeypatch.setattr("phasesweep.cli.load_experiment", lambda _path: object())
    monkeypatch.setattr(
        "phasesweep.cli.run_experiment",
        lambda *_args, **_kwargs: lock_dir(),
    )

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PHASESWEEP_LOCK_DIR must be an absolute path" in captured.err
    assert "Traceback" not in captured.err
    assert "internal error" not in captured.err


def _unexpected_validation_error() -> ValidationError:
    """Build a real Pydantic error raised outside the config-loading layer.

    :return ValidationError: The error from validating an empty experiment payload.
    """
    with pytest.raises(ValidationError) as exc_info:
        Experiment.model_validate({})
    return exc_info.value


@pytest.mark.parametrize(
    ("error", "exit_code", "present", "absent"),
    [
        pytest.param(
            RuntimeError("injected-internal"),
            70,
            ["internal error", "Traceback", "injected-internal"],
            [],
            id="unexpected_failure",
        ),
        pytest.param(
            _unexpected_validation_error(),
            70,
            ["internal error", "Traceback"],
            [],
            id="unexpected_validation_error",
        ),
        pytest.param(
            PermissionError("workdir is not writable"),
            1,
            ["workdir is not writable"],
            ["Traceback", "internal error"],
            id="environmental_io_failure",
        ),
    ],
)
def test_cli_boundary_separates_internal_errors_from_operational_failures(
    error: BaseException,
    exit_code: int,
    present: list[str],
    absent: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An exception that is not an expected outcome is a bug: exit 70 with a traceback.

    Only the config-loading layer may classify Pydantic errors as bad input, so
    a validation error raised during a run is still an internal error. An
    environmental I/O failure is operational: exit 1, no traceback.
    """
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, error)

    actual_exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert actual_exit_code == exit_code
    for text in present:
        assert text in captured.err
    for text in absent:
        assert text not in captured.err


def test_cli_boundary_leaves_help_exit_status_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--help`` still succeeds through the boundary rather than being trapped."""
    exit_code = invoke_cli_boundary(["--help"], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 0
    # The program name is whatever Click detects for the running process, so
    # pin the usage line and body rather than the console-script name.
    assert "[OPTIONS] COMMAND [ARGS]..." in captured.out
    assert "Phase-chained hyperparameter sweeps" in captured.out


def test_expected_operational_failures_share_one_base() -> None:
    """Every operator-facing failure the boundary reports without a traceback.

    ``PhaseSweepError`` must stay a ``RuntimeError`` so existing
    ``except RuntimeError`` handlers keep catching these.
    """
    assert issubclass(PhaseSweepError, RuntimeError)
    for error_type in (
        NoFeasibleTrialError,
        ProcessCleanupUncertainError,
        UnsafeProcessCleanupError,
        ExperimentLockBusyError,
        GpuConfigurationError,
        StudyContextConflictError,
        StudyFingerprintMismatchError,
        StudySchemaMismatchError,
        StudyStorageUnavailableError,
        TrialTargetRefusedError,
        UnsafeLockPathError,
    ):
        assert issubclass(error_type, PhaseSweepError), error_type.__name__
