"""CLI commands: validate, show-winners, --dry-run."""

from __future__ import annotations

import logging
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import cli as cli_main
from phasesweep.config import (
    Experiment,
    FloatParam,
    JsonEnvelopeExtractor,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.engine import (
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    NoFeasibleTrialError,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    SamplerContinuationUnsupportedError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
    UnsafeProcessCleanupError,
)
from phasesweep.engine.paths import (
    _generation_dir,
    _generation_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _winner_path,
)
from phasesweep.engine.publication import _last_successful_generation_id
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
from tests.ledger_fixtures import materialize
from tests.recovery_helpers import recover_run_cli


@pytest.mark.parametrize(
    ("extractor", "details"),
    [
        ({"type": "json", "path": "r.json", "key": "eval.loss"}, ["r.json", "eval.loss"]),
        (
            {"type": "wandb", "entity": "e", "project": "p", "metric_key": "eval/loss"},
            ["eval/loss", "api.wandb.ai"],
        ),
        (
            {"type": "log_regex", "pattern": "loss=(?P<value>[0-9.]+)"},
            ["select", "last", "stdout.log"],
        ),
        (
            {
                "type": "json_envelope",
                "objective_name": "loss",
                "split": "validation",
                "policy": "final",
                "expected_step": 20,
            },
            ["policy", "final", "expected_step", "20"],
        ),
    ],
)
def test_validate_and_dry_run_explain_scoring_without_sdk(
    tmp_path, monkeypatch, caplog, extractor, details
):
    from phasesweep.config import Metric

    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        n_trials=1,
        metric=Metric(name="loss", goal="minimize", extractor=extractor),
    )
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


def test_help_registers_commands_and_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"], terminal_width=120)

    assert result.exit_code == 0
    assert "-h, --help" in result.output
    assert "recover-run" not in result.output
    for command in ("init", "mcp", "run", "show-winners", "status", "validate"):
        assert command in result.output

    for command in ("run", "validate", "show-winners", "status"):
        result = runner.invoke(cli_main, [command, "--help"], terminal_width=120)
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "CONFIG" in result.output
        assert "-h, --help" in result.output

    init_help = runner.invoke(cli_main, ["init", "--help"], terminal_width=120)
    assert init_help.exit_code == 0
    assert "-o, --output FILE" in init_help.output

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
    for command in ("check", "init-catalog", "recover-run", "serve"):
        assert command in mcp_help.output
    for command in ("check-install", "install", "uninstall"):
        assert command not in mcp_help.output
    assert "--catalog PATH" not in mcp_help.output

    serve_help = runner.invoke(cli_main, ["mcp", "serve", "--help"], terminal_width=120)
    assert serve_help.exit_code == 0
    assert "--catalog PATH" in serve_help.output

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
            sampler: {{ type: tpe, seed: 0, acknowledge_nonresumable: true }}
            search_space: {{ lr: {{ type: float, low: 0.1, high: 1.0 }} }}
          - name: wd
            n_trials: 2
            sampler: {{ type: random, seed: 3 }}
            search_space: {{ wd: {{ type: float, low: 0.0, high: 0.3 }} }}
        """)
    )

    result = CliRunner().invoke(cli_main, ["validate", str(p)])

    assert result.exit_code == 0, result.output
    assert "phase 'depth': sampler=grid (resumable)" in result.output
    assert (
        "phase 'lr': sampler=tpe seed=0 (non-resumable: run each target in one invocation)"
        in result.output
    )
    assert "phase 'wd': sampler=random seed=3 (resumable, reproducible)" in result.output


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
        "phasesweep.engine.phase._suggest",
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
    assert status_obj["ledger_path"] == str(tmp_path / "studies.journal")


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

    Review v0.5.16 / blocker 4: the experiment CLI used to decorate a
    historical winner with the *current* phase comments; it now uses saved
    annotations and an explicit marker when the current config no longer
    matches the published one.
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


def _corrupt_the_publication(experiment: Experiment) -> str:
    """Edit a published winner artifact so its generation manifest stops validating.

    :param Experiment experiment: Experiment whose published generation should be corrupted.
    :return str: The corrupted generation id.
    """
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")
    return generation_id


def test_status_and_show_winners_report_a_corrupt_publication_and_exit_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review v0.5.18 / finding F4: a corrupt publication is not a fresh tree.

    ``status`` used to print a payload byte-identical to a never-run workdir
    and exit 0, so the operator's natural next move -- re-run -- advanced the
    pointer and erased the only evidence of corruption. ``show-winners`` must
    likewise not answer "no winner yet" over it. Both commands are read-only,
    so they share one corrupted tree.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    config_path = materialized.config_path
    generation_id = _corrupt_the_publication(materialized.experiment)

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


def test_tampered_reproducibility_record_fails_both_reporting_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The claim-time provenance files feed the same reporting as any winner."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    config_path, experiment = materialized.config_path, materialized.experiment
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    record = _generation_dir(experiment, generation_id) / "reproducibility.json"
    record.write_bytes(record.read_bytes() + b"\n")

    assert invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 1
    status_captured = capsys.readouterr()
    assert yaml.safe_load(status_captured.out)["publication_integrity"] == "failed"
    assert "Do not run anything over this tree" in status_captured.err

    assert invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch) == 1
    winners_captured = capsys.readouterr()
    assert "Do not run anything over this tree" in winners_captured.err
    assert "Traceback" not in winners_captured.err


@requires_nonroot
def test_status_reports_an_unreadable_snapshot_as_permission_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A snapshot this user may not read reports permission, not corruption.

    Re-review v0.5.19 / observation N1: ``config.snapshot.yaml`` is owner-only,
    so a second operator inspecting a healthy tree was told the publication no
    longer validates and to restore the generation namespace. The read still
    fails closed -- nothing unvalidatable may read as published -- but the
    reason names the permission denial and the user who can validate it.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    config_path, experiment = materialized.config_path, materialized.experiment
    generation_id = _last_successful_generation_id(experiment)
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


def test_cli_boundary_reports_config_syntax_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A YAML syntax error is bad input: exit 2, the file named, no traceback.

    Bad indentation is one of the two most common YAML mistakes, and PyYAML
    labels it ``in "<unicode string>"``; without the config loader's own source
    label the operator cannot tell which file failed.
    """
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("experiment: t\nphases:\n  - name: a\n   n_trials: 1\n")

    exit_code = invoke_cli_boundary(["validate", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert str(config_path) in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_cli_boundary_names_config_for_schema_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "invalid-schema.yaml"
    config_path.write_text("experiment: t\nphases: []\n")

    exit_code = invoke_cli_boundary(["validate", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert str(config_path) in captured.err
    assert "Traceback" not in captured.err


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


def test_cli_boundary_reports_runtime_operational_failures_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lock contention is a failure, not a bug.

    Runtime lock configuration goes through the real validator in the next test.
    """
    error = ExperimentLockBusyError("another experiment process holds the lock")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, error)

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(error) in captured.err
    assert "Traceback" not in captured.err
    assert "internal error" not in captured.err


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


def test_cli_boundary_reports_unexpected_failure_as_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An exception that is not an expected outcome is a bug: exit 70 with a traceback."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, RuntimeError("injected-internal"))

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 70
    assert "internal error" in captured.err
    assert "Traceback" in captured.err
    assert "injected-internal" in captured.err


def test_cli_boundary_does_not_misclassify_unexpected_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only the config-loading layer may classify Pydantic errors as bad input."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    with pytest.raises(ValidationError) as exc_info:
        Experiment.model_validate({})
    _stub_run_command(monkeypatch, exc_info.value)

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 70
    assert "internal error" in captured.err
    assert "Traceback" in captured.err


def test_cli_boundary_reports_environmental_io_failure_as_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, PermissionError("workdir is not writable"))

    exit_code = invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "workdir is not writable" in captured.err
    assert "Traceback" not in captured.err
    assert "internal error" not in captured.err


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
        SamplerContinuationUnsupportedError,
        StudyContextConflictError,
        StudyFingerprintMismatchError,
        StudySchemaMismatchError,
        StudyStorageUnavailableError,
        TrialTargetRegressionError,
        UnsafeLockPathError,
    ):
        assert issubclass(error_type, PhaseSweepError), error_type.__name__
