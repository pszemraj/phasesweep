"""CLI commands: validate, show-winners, --dry-run."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import stat
import sys
import textwrap
from pathlib import Path

import optuna
import pytest
import yaml
from click.testing import CliRunner

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import cli as cli_main
from phasesweep.cli import main as cli_boundary
from phasesweep.config import Suite, load_config
from phasesweep.engine import (
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
from phasesweep.engine.guards import _register_active_attempt
from phasesweep.engine.run import run_suite
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    _experiment_dir,
    _generation_dir,
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _last_successful_generation_path,
    _last_successful_suite_generation_id,
    _suite_generation_summary_path,
    _winner_path,
)
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.runs import RunStore
from tests.conftest import (
    assert_published_winner_evidence_local,
    write_trainer,
    write_yaml,
)


def test_help_registers_commands_and_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"], terminal_width=120)

    assert result.exit_code == 0
    assert "-h, --help" in result.output
    assert "recover-run" not in result.output
    for command in ("init", "mcp", "rebind-workdir", "run", "show-winners", "status", "validate"):
        assert command in result.output

    for command in ("rebind-workdir", "run", "validate", "show-winners", "status"):
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

    run_help = runner.invoke(cli_main, ["run", "--help"], terminal_width=120).output
    assert "--from-phase PHASE" in run_help
    assert "[default: (first phase)]" in run_help
    assert "--dry-run" in run_help
    assert "-v, --verbose" in run_help

    winners_help = runner.invoke(cli_main, ["show-winners", "--help"], terminal_width=120).output
    assert "show-winners [OPTIONS] CONFIG_YAML" in winners_help
    assert "Pass the same experiment or suite" in winners_help
    assert "config YAML used for the run" in winners_help

    mcp_help = runner.invoke(cli_main, ["mcp", "--help"], terminal_width=120)
    assert mcp_help.exit_code == 0
    for command in (
        "check",
        "check-install",
        "init-catalog",
        "install",
        "recover-run",
        "serve",
        "uninstall",
    ):
        assert command in mcp_help.output
    assert "--catalog PATH" not in mcp_help.output

    serve_help = runner.invoke(cli_main, ["mcp", "serve", "--help"], terminal_width=120)
    assert serve_help.exit_code == 0
    assert "--catalog PATH" in serve_help.output

    check_help = runner.invoke(cli_main, ["mcp", "check", "--help"], terminal_width=120)
    assert check_help.exit_code == 0
    assert "--catalog PATH" in check_help.output

    check_install_help = runner.invoke(
        cli_main,
        ["mcp", "check-install", "--help"],
        terminal_width=120,
    )
    assert check_install_help.exit_code == 0
    assert "--agent" in check_install_help.output

    init_help = runner.invoke(cli_main, ["mcp", "init-catalog", "--help"], terminal_width=120)
    assert init_help.exit_code == 0
    assert "--from PATH" in init_help.output


def test_recover_run_expands_user_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    RunStore(tmp_path / "state")

    result = CliRunner().invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", "~/state", "--run-id", "missing"],
    )

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

    result = CliRunner().invoke(
        cli_main,
        ["mcp", "recover-run", "--state-dir", str(tmp_path / "state"), "--run-id", "missing"],
    )

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

    def fake_load_config(_path: Path) -> object:
        events.append("load")
        return config

    def fake_run_config(loaded: object, *, from_phase: str | None, dry_run: bool) -> None:
        assert loaded is config
        assert from_phase is None
        assert dry_run is expected_dry_run
        events.append("run")

    monkeypatch.setattr("phasesweep.cli.load_config", fake_load_config)
    monkeypatch.setattr("phasesweep.cli.run_config", fake_run_config)
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
        storage: sqlite:///{tmp_path}/phases.db
        provenance: {{revision: test-fixture-v1}}
        trial_command: "echo {{overrides}}"
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
    workdir = tmp_path / "wd"
    workdir.mkdir()
    phase_dir = workdir / "t" / "depth"
    phase_dir.mkdir(parents=True)
    (phase_dir / "winner.yaml").write_text(
        textwrap.dedent("""
        phase: depth
        trial_number: 2
        metric:
          x: 0.5
          goal: minimize
        params:
          x: 5
        effective_overrides:
          x: 5
        constraints: {}
        """).lstrip()
    )

    def make_cfg(workdir_str: str) -> Path:
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            textwrap.dedent(f"""
            experiment: t
            workdir: {workdir_str}
            trial_command: "echo {{overrides}}"
            metric:
              extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
            phases:
              - name: depth
                comment: settle the depth before anything else.
                n_trials: 1
                search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
            """)
        )
        return cfg

    runner = CliRunner()

    # With a winner: comment must come BEFORE the winner block.
    result_with = runner.invoke(cli_main, ["show-winners", str(make_cfg(str(workdir)))])
    assert result_with.exit_code == 0
    comment_line = "# settle the depth before anything else."
    metric_line = "trial_number: 2"
    assert comment_line in result_with.output
    assert metric_line in result_with.output
    assert result_with.output.index(comment_line) < result_with.output.index(metric_line)

    # Without a winner (different workdir → no winner.yaml): comment is still surfaced.
    result_without = runner.invoke(
        cli_main, ["show-winners", str(make_cfg(str(tmp_path / "empty_wd")))]
    )
    assert result_without.exit_code == 0
    assert "(no winner yet)" in result_without.output
    assert comment_line in result_without.output


def test_show_winners_uses_only_the_last_successful_generation(tmp_path: Path) -> None:
    """Mutable compatibility files must not outrank immutable generation results."""
    config_path = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: p
            n_trials: 1
            search_space: {{}}
        """,
    )
    experiment = load_experiment(config_path)
    immutable = _generation_winner_path(experiment, "successful", "p")
    immutable.parent.mkdir(parents=True)
    immutable.write_text("trial_number: 1\n")
    compatibility = _winner_path(experiment, "p")
    compatibility.parent.mkdir(parents=True, exist_ok=True)
    compatibility.write_text("trial_number: 99\n")
    # Pointer validation reads back the generation's own immutable summary,
    # not the (post-commit, informational) lifecycle record (review v0.5.15 /
    # blocker 3), so a matching summary is required for the pointer to
    # resolve as published.
    summary = _generation_summary_path(experiment, "successful")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("experiment: t\ngeneration_id: successful\n")
    _last_successful_generation_path(experiment).write_text(
        "experiment: t\ngeneration_id: successful\n"
    )
    _generation_path(experiment).write_text("generation_id: interrupted\n")

    published = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])

    assert published.exit_code == 0
    assert "trial_number: 1" in published.output
    assert "trial_number: 99" not in published.output

    _last_successful_generation_path(experiment).unlink()
    unpublished = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])

    assert unpublished.exit_code == 0
    assert "(no winner yet)" in unpublished.output
    assert "trial_number: 99" not in unpublished.output


def test_dry_run_does_not_launch(tmp_path, caplog, monkeypatch):
    """Dry-run should preview one coherent chain without launching anything."""

    caplog.set_level(logging.INFO)
    body = f"""
experiment: dry
storage: sqlite:///{tmp_path}/dry.db
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path}/runs
trial_command: "false {{overrides}}"
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json_envelope, objective_name: loss, split: test, policy: test }}
phases:
  - name: a
    n_trials: 5
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ lr: {{ type: float, low: 1e-5, high: 1e-2, log: true }} }}
  - name: b
    inherits: [a]
    n_trials: 5
    sampler: {{ type: random, seed: 0 }}
    search_space: {{ wd: {{ type: float, low: 0, high: 0.3 }} }}
"""
    exp = load_experiment(write_yaml(tmp_path, body))
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


def test_status_cli_reports_phase_counts(tmp_path: Path) -> None:
    """``phasesweep status`` is read-only and reports study trial state counts."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True)
        args,_=ap.parse_known_args(); open(args.out, 'w').write(json.dumps({'x': 1.0}))
        print('x=1.0')
        """,
    )
    p = write_yaml(
        tmp_path,
        f"""
        experiment: status_test
        storage: sqlite:///{tmp_path}/status.db
        provenance: {{revision: test-fixture-v1}}
        workdir: {tmp_path}/runs
        trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
        phases:
          - name: p
            n_trials: 1
            sampler: {{ type: random, seed: 0 }}
            search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
        """,
    )
    exp = load_experiment(p)
    run_experiment(exp)

    result = CliRunner().invoke(cli_main, ["status", str(p)])
    assert result.exit_code == 0
    assert "status_test" in result.output
    assert "COMPLETE: 1" in result.output

    # The published generation now backs the phase's winner; current and
    # published identity must both be shown explicitly, never one unlabeled id.
    status_obj = yaml.safe_load(result.output)
    assert status_obj["current_generation_id"] is not None
    assert status_obj["published_generation_id"] == status_obj["current_generation_id"]


def test_show_winners_renders_historical_annotations_on_config_drift(tmp_path: Path) -> None:
    """A published result keeps its own comments and is labeled historical on drift.

    Review v0.5.16 / blocker 4: the experiment CLI used to decorate a
    historical winner with the *current* phase comments; it now mirrors the
    suite CLI's historical rendering — saved annotations, plus an explicit
    marker when the current config no longer matches the published one.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--out")\n'
        'parser.add_argument("--x", type=int, default=0)\n'
        "args, _ = parser.parse_known_args()\n"
        'print(f"x={args.x}")\n',
    )

    def config_text(comment: str, metric_name: str) -> str:
        return f"""
        experiment: drift_cli
        workdir: {tmp_path}/runs
        trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
        metric:
          name: {metric_name}
          goal: minimize
          extractor: {{ type: log_regex, pattern: '{metric_name}=(?P<value>[0-9.eE+-]+)' }}
        phases:
          - name: p
            comment: {comment}
            n_trials: 1
            sampler: {{ type: random, seed: 0 }}
            search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
        """

    config_path = write_yaml(tmp_path, config_text("original hypothesis", "x"))
    run_experiment(load_experiment(config_path))

    # Semantic drift (metric rename) plus a new comment on the same phase.
    config_path.write_text(textwrap.dedent(config_text("new hypothesis", "y")).lstrip())

    result = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])
    assert result.exit_code == 0
    assert "Historical experiment result" in result.output
    assert "# original hypothesis" in result.output
    assert "new hypothesis" not in result.output


def _published_experiment_config(tmp_path: Path) -> Path:
    """Write and run a one-phase experiment so its workdir holds a real publication.

    :param Path tmp_path: Per-test temporary directory.
    :return Path: Config path whose experiment has published exactly one generation.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        'parser.add_argument("--out")\n'
        'parser.add_argument("--x", type=int, default=0)\n'
        "args, _ = parser.parse_known_args()\n"
        'print(f"x={args.x}")\n',
    )
    return write_yaml(
        tmp_path,
        f"""
        experiment: integrity_cli
        workdir: {tmp_path}/runs
        trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
        phases:
          - name: p
            n_trials: 1
            sampler: {{ type: random, seed: 0 }}
            search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
        """,
    )


def _corrupt_the_publication(config_path: Path) -> str:
    """Edit a published winner artifact so its generation manifest stops validating.

    :param Path config_path: Config whose published generation should be corrupted.
    :return str: The corrupted generation id.
    """
    experiment = load_experiment(config_path)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")
    return generation_id


def test_status_reports_a_corrupt_publication_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review v0.5.18 / finding F4: a corrupt publication is not a fresh tree.

    ``status`` used to print a payload byte-identical to a never-run workdir
    and exit 0, so the operator's natural next move -- re-run -- advanced the
    pointer and erased the only evidence of corruption.
    """
    config_path = _published_experiment_config(tmp_path)
    run_experiment(load_experiment(config_path))
    generation_id = _corrupt_the_publication(config_path)

    exit_code = _invoke_cli_boundary(["status", str(config_path)], monkeypatch)

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


def test_show_winners_reports_a_corrupt_publication_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``show-winners`` must not answer "no winner yet" over a corrupt publication."""
    config_path = _published_experiment_config(tmp_path)
    run_experiment(load_experiment(config_path))
    generation_id = _corrupt_the_publication(config_path)

    exit_code = _invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch)

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
    config_path = _published_experiment_config(tmp_path)
    run_experiment(load_experiment(config_path))
    experiment = load_experiment(config_path)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    record = _generation_dir(experiment, generation_id) / "reproducibility.json"
    record.write_bytes(record.read_bytes() + b"\n")

    assert _invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 1
    status_captured = capsys.readouterr()
    assert yaml.safe_load(status_captured.out)["publication_integrity"] == "failed"
    assert "Do not run anything over this tree" in status_captured.err

    assert _invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch) == 1
    winners_captured = capsys.readouterr()
    assert "Do not run anything over this tree" in winners_captured.err
    assert "Traceback" not in winners_captured.err


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root reads any mode, so no PermissionError can be provoked",
)
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
    config_path = _published_experiment_config(tmp_path)
    run_experiment(load_experiment(config_path))
    experiment = load_experiment(config_path)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot = _generation_dir(experiment, generation_id) / "config.snapshot.yaml"
    original_mode = stat.S_IMODE(snapshot.stat().st_mode)
    snapshot.chmod(0o000)
    try:
        exit_code = _invoke_cli_boundary(["status", str(config_path)], monkeypatch)
        captured = capsys.readouterr()
    finally:
        snapshot.chmod(original_mode)

    assert exit_code == 1
    assert "Traceback" not in captured.err
    payload = yaml.safe_load(captured.out)
    assert payload["publication_integrity"] == "failed"
    assert "permission denied" in payload["publication_error"]
    assert "missing or unreadable" not in payload["publication_error"]
    assert "permission denied" in captured.err
    assert "only the publishing user" in captured.err


def _suite_config(tmp_path: Path) -> Path:
    """Write a one-study suite config over persistent storage.

    :param Path tmp_path: Per-test temporary directory.
    :return Path: Config path for a suite that has not run yet.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        "import argparse\nparser = argparse.ArgumentParser()\n"
        'parser.add_argument("--out")\nparser.add_argument("--x", type=int, default=0)\n'
        'args, _ = parser.parse_known_args()\nprint(f"x={args.x}")\n',
    )
    return write_yaml(
        tmp_path,
        f"""
        suite: integrity_suite
        defaults:
          workdir: {tmp_path}/runs
          storage: sqlite:///{tmp_path}/suite.db
          provenance: {{revision: test-fixture-v1}}
          trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
        studies:
          - name: one
            phases:
              - name: p
                n_trials: 1
                sampler: {{ type: random, seed: 0 }}
                search_space: {{ x: {{ type: int, low: 0, high: 3 }} }}
        """,
    )


def test_suite_status_fails_on_a_corrupt_component_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A suite status embeds every study, so one corrupt component fails the read.

    A component-only run leaves the suite pointer ``absent`` -- a healthy state
    for a suite that never published -- so the escalation here can only come
    from the embedded study payload, which without this check would print the
    corruption and still exit 0.
    """
    config_path = _suite_config(tmp_path)
    suite = load_config(config_path)
    assert isinstance(suite, Suite)
    # config_status compiles each study independently, so publishing the one
    # component is enough to give the suite payload a publication to corrupt.
    component = suite.experiment_for_study(suite.studies[0])
    run_experiment(component)
    generation_id = _last_successful_generation_id(component)
    assert generation_id is not None
    winner_path = _generation_winner_path(component, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    exit_code = _invoke_cli_boundary(["status", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    payload = yaml.safe_load(captured.out)
    assert payload["publication_integrity"] == "absent"
    assert payload["published_suite_generation_id"] is None
    assert payload["studies"][0]["status"]["publication_integrity"] == "failed"
    assert "Do not run anything over this tree" in captured.err


def test_suite_status_reports_the_suite_publication_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A published suite reports its own generation identity and integrity.

    Re-review v0.5.19 / observation N2: the suite envelope carried no
    publication fields at all, so ``status`` could not report the one verdict
    ``show-winners`` resolves for the same tree.
    """
    config_path = _suite_config(tmp_path)
    suite = load_config(config_path)
    assert isinstance(suite, Suite)

    assert _invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 0
    fresh = yaml.safe_load(capsys.readouterr().out)
    assert fresh["publication_integrity"] == "absent"
    assert fresh["published_suite_generation_id"] is None

    run_suite(suite)
    generation_id = _last_successful_suite_generation_id(suite)
    assert generation_id is not None

    assert _invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 0
    published = yaml.safe_load(capsys.readouterr().out)
    assert published["publication_integrity"] == "ok"
    assert published["published_suite_generation_id"] == generation_id
    assert "publication_error" not in published
    assert published["studies"][0]["status"]["publication_integrity"] == "ok"


def test_suite_status_escalates_a_corrupt_suite_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both reporting surfaces escalate a suite publication that stopped validating.

    Re-review v0.5.19 / observation N2: the component studies still validate
    here, so before the suite envelope carried its own verdict ``status``
    printed ``publication_integrity: ok`` for every component and exited 0
    while ``show-winners`` reported the same tree as corrupt.
    """
    config_path = _suite_config(tmp_path)
    suite = load_config(config_path)
    assert isinstance(suite, Suite)
    run_suite(suite)
    generation_id = _last_successful_suite_generation_id(suite)
    assert generation_id is not None

    # Spoof a published winner fact in the suite summary alone: the component
    # generation it names stays intact, so only the suite verdict changes.
    summary_path = _suite_generation_summary_path(suite, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    exposed = [item for item in summary["studies"][0]["phases"] if item.get("exposed")]
    assert exposed, "test setup: the suite must expose at least one winner"
    exposed[0]["metric"] = exposed[0]["metric"] + 1.0
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    exit_code = _invoke_cli_boundary(["status", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    payload = yaml.safe_load(captured.out)
    assert payload["publication_integrity"] == "failed"
    assert payload["published_suite_generation_id"] is None
    assert payload["publication_error"]
    # The component study is untouched, so the suite level is the only escalation.
    assert payload["studies"][0]["status"]["publication_integrity"] == "ok"
    assert "Suite 'integrity_suite'" in captured.err
    assert "Do not run anything over this tree" in captured.err

    assert _invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch) == 1
    winners_captured = capsys.readouterr()
    assert "Suite 'integrity_suite'" in winners_captured.err


def test_status_and_show_winners_stay_successful_without_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A healthy publication and a never-published tree both stay exit 0."""
    config_path = _published_experiment_config(tmp_path)

    assert _invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 0
    fresh = capsys.readouterr()
    assert yaml.safe_load(fresh.out)["publication_integrity"] == "absent"
    assert _invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch) == 0
    capsys.readouterr()

    run_experiment(load_experiment(config_path))

    assert _invoke_cli_boundary(["status", str(config_path)], monkeypatch) == 0
    published = capsys.readouterr()
    assert yaml.safe_load(published.out)["publication_integrity"] == "ok"
    assert _invoke_cli_boundary(["show-winners", str(config_path)], monkeypatch) == 0
    assert "trial_number" in capsys.readouterr().out


def _movable_experiment_configs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Write two configs that differ only in ``workdir`` and share one SQLite storage.

    :param Path tmp_path: Per-test temporary directory.
    :return tuple[Path, Path, Path, Path]: ``(config_a, config_b, workdir_a, workdir_b)``.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"x": 0.5}))
        print("x=0.5")
        """,
    )
    workdir_a = tmp_path / "runs_a"
    workdir_b = tmp_path / "runs_b"

    def config_text(workdir: Path) -> str:
        return textwrap.dedent(f"""
            experiment: t
            storage: sqlite:///{tmp_path}/studies.db
            provenance: {{revision: test-fixture-v1}}
            workdir: {workdir}
            trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
            metric:
              name: x
              goal: minimize
              extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            phases:
              - name: p
                n_trials: 1
                sampler: {{ type: random, seed: 0 }}
                search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
            """).lstrip()

    config_a = tmp_path / "exp_a.yaml"
    config_a.write_text(config_text(workdir_a))
    config_b = tmp_path / "exp_b.yaml"
    config_b.write_text(config_text(workdir_b))
    return config_a, config_b, workdir_a, workdir_b


def test_rebind_workdir_moves_the_binding_to_a_relocated_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit rebind is the only way one study changes publication roots."""
    config_a, config_b, workdir_a, workdir_b = _movable_experiment_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    shutil.copytree(workdir_a, workdir_b)
    experiment_b = load_experiment(config_b)

    result = CliRunner().invoke(cli_main, ["rebind-workdir", str(config_b)])

    assert result.exit_code == 0, result.output
    assert "t::p" in result.output
    assert str(_experiment_dir(experiment_a)) in result.output
    assert str(_experiment_dir(experiment_b)) in result.output
    study = optuna.load_study(study_name="t::p", storage=experiment_b.storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment_b))

    # The relocated root is now the only one this study will publish into.
    run_experiment(load_experiment(config_b))
    exit_code = _invoke_cli_boundary(["run", str(config_a)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "rebind-workdir" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("drop_pointer", "holds no valid published generation"),
        ("tamper_winner", "does not match its recorded hash"),
    ],
)
def test_rebind_workdir_refuses_a_destination_without_the_recorded_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damage: str,
    expected: str,
) -> None:
    """An incomplete move must leave the original binding in place.

    Both refusal paths matter: a destination with no last-success pointer at
    all, and one whose pointer resolves to a generation manifest that no longer
    validates.
    """
    config_a, config_b, workdir_a, workdir_b = _movable_experiment_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    published = _last_successful_generation_id(experiment_a)
    assert published is not None
    shutil.copytree(workdir_a, workdir_b)
    experiment_b = load_experiment(config_b)
    if damage == "drop_pointer":
        _last_successful_generation_path(experiment_b).unlink()
    else:
        winner = _generation_winner_path(experiment_b, published, "p")
        winner.write_text(winner.read_text() + "\n# edited after publication\n")

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_b)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert str(_experiment_dir(experiment_b)) in captured.err
    assert expected in captured.err
    study = optuna.load_study(study_name="t::p", storage=experiment_a.storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment_a))


def test_rebind_workdir_refuses_when_no_study_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rebinding is never a backdoor for claiming a root a plain run would claim."""
    config_a, _config_b, _workdir_a, _workdir_b = _movable_experiment_configs(tmp_path)
    experiment = load_experiment(config_a)
    optuna.create_study(study_name="t::p", storage=experiment.storage, direction="minimize")
    _experiment_dir(experiment).mkdir(parents=True)

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_a)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    assert ARTIFACT_ROOT_ATTR not in study.user_attrs


def test_rebind_workdir_rebinds_every_compiled_suite_study(tmp_path: Path) -> None:
    """A suite compiles to one experiment per study, each with its own artifact root."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        open(args.out, "w").write(json.dumps({"x": 1.0}))
        print("x=1.0")
        """,
    )

    def suite_text(workdir: Path) -> str:
        return textwrap.dedent(f"""
            suite: s
            defaults:
              workdir: {workdir}
              storage: sqlite:///{tmp_path}/suite.db
              provenance: {{revision: test-fixture-v1}}
              trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
              metric:
                name: x
                goal: minimize
                extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            studies:
              - name: ran
                phases:
                  - name: p
                    n_trials: 1
                    sampler: {{ type: random, seed: 0 }}
                    search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
              - name: untouched
                phases:
                  - name: p
                    n_trials: 1
                    sampler: {{ type: random, seed: 0 }}
                    search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
            """).lstrip()

    workdir_a = tmp_path / "runs_a"
    workdir_b = tmp_path / "runs_b"
    config_a = tmp_path / "suite_a.yaml"
    config_a.write_text(suite_text(workdir_a))
    config_b = tmp_path / "suite_b.yaml"
    config_b.write_text(suite_text(workdir_b))
    suite_a = load_config(config_a)
    assert isinstance(suite_a, Suite)
    # Only the first study has ever run, so the second contributes no study to
    # rebind while the first still supplies the binding that authorizes it.
    run_experiment(suite_a.experiment_for_study(suite_a.studies[0]))
    shutil.copytree(workdir_a, workdir_b)
    suite_b = load_config(config_b)
    assert isinstance(suite_b, Suite)

    result = CliRunner().invoke(cli_main, ["rebind-workdir", str(config_b)])

    assert result.exit_code == 0, result.output
    assert "s__ran::p" in result.output
    study = optuna.load_study(study_name="s__ran::p", storage=f"sqlite:///{tmp_path}/suite.db")
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(
        _experiment_dir(suite_b.experiment_for_study(suite_b.studies[0]))
    )


def _movable_two_phase_configs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build one two-phase experiment as two configs differing only in workdir.

    Two phases means two studies, which is what makes a partially applied
    rebind observable at all.

    :param Path tmp_path: Per-test temporary directory.
    :return tuple[Path, Path, Path, Path]: ``(config_a, config_b, workdir_a, workdir_b)``.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"x": 0.5}))
        print("x=0.5")
        """,
    )
    workdir_a = tmp_path / "runs_a"
    workdir_b = tmp_path / "runs_b"

    def config_text(workdir: Path) -> str:
        return textwrap.dedent(f"""
            experiment: t
            storage: sqlite:///{tmp_path}/studies.db
            provenance: {{revision: test-fixture-v1}}
            workdir: {workdir}
            trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
            metric:
              name: x
              goal: minimize
              extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            phases:
              - name: p
                n_trials: 1
                sampler: {{ type: random, seed: 0 }}
                search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
              - name: q
                n_trials: 1
                sampler: {{ type: random, seed: 1 }}
                search_space: {{ y: {{ type: int, low: 0, high: 10 }} }}
            """).lstrip()

    config_a = tmp_path / "exp_two_a.yaml"
    config_a.write_text(config_text(workdir_a))
    config_b = tmp_path / "exp_two_b.yaml"
    config_b.write_text(config_text(workdir_b))
    return config_a, config_b, workdir_a, workdir_b


def _drop_artifact_root_binding(storage: str, study_name: str) -> None:
    """Reconstruct a pre-binding study by deleting its artifact-root user attr.

    Optuna's API can set a study user attr but never delete one, so the state
    a database written before the binding existed is in has to be rebuilt by
    removing the row from the SQLite file directly.

    :param str storage: ``sqlite:///`` storage URL backing the study.
    :param str study_name: Fully qualified Optuna study name to unbind.
    """
    with sqlite3.connect(storage.removeprefix("sqlite:///")) as connection:
        connection.execute(
            "DELETE FROM study_user_attributes WHERE key = ? AND study_id = "
            "(SELECT study_id FROM studies WHERE study_name = ?)",
            (ARTIFACT_ROOT_ATTR, study_name),
        )


def test_rebind_workdir_refuses_a_stale_copy_missing_trial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A copy taken before the ledger advanced is not the same artifact tree.

    Its publication still validates, so only the per-trial evidence check sees
    that the next run would select a trial whose artifacts exist solely in the
    source tree (re-review v0.5.19 / blocker B2).
    """
    config_a, config_b, workdir_a, workdir_b = _movable_experiment_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    shutil.copytree(workdir_a, workdir_b)
    topped_up = experiment_a.model_copy(
        update={"phases": [experiment_a.phases[0].model_copy(update={"n_trials": 2})]}
    )
    run_experiment(topped_up)

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_b)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert "trial 1" in captured.err
    assert str(workdir_b) in captured.err
    assert "stale copy" in captured.err
    study = optuna.load_study(study_name="t::p", storage=experiment_a.storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment_a))


def test_rebind_workdir_refuses_while_a_trial_is_still_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An interrupted attempt has to be recovered before its tree moves.

    Recovery reads the absolute trial path the attempt persisted, so it can
    only run against the root that attempt started under.
    """
    config_a, config_b, workdir_a, workdir_b = _movable_experiment_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    assert experiment_a.storage is not None
    optuna.load_study(study_name="t::p", storage=experiment_a.storage).ask()
    shutil.copytree(workdir_a, workdir_b)

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_b)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert "RUNNING trial" in captured.err
    study = optuna.load_study(study_name="t::p", storage=experiment_a.storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment_a))


def test_rebind_workdir_refuses_an_unresolved_attempt_at_the_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A surviving registry entry is an attempt nobody resolved.

    Entries are unlinked when their attempt resolves, and each one references
    the absolute trial directory recovery would have to inspect, which no
    longer exists at that path once the tree moves.
    """
    config_a, config_b, workdir_a, workdir_b = _movable_experiment_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    shutil.copytree(workdir_a, workdir_b)
    experiment_b = load_experiment(config_b)
    _register_active_attempt(
        experiment_b,
        attempt_id="unresolved-attempt",
        phase_name="p",
        study_name="t::p",
        trial_number=0,
        trial_dir=_experiment_dir(experiment_a) / "p" / "trial_00000",
        generation_id="old-generation",
    )

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_b)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert "unresolved attempt registry entries" in captured.err
    study = optuna.load_study(study_name="t::p", storage=experiment_a.storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment_a))


def test_rebind_workdir_refuses_a_suite_that_published_a_suite_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A published suite summary pins absolute component paths it cannot re-derive.

    Rebinding it would produce a tree whose suite publication reports corrupt
    on the next read, which is worse than refusing the move.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        args, _ = ap.parse_known_args()
        open(args.out, "w").write(json.dumps({"x": 1.0}))
        print("x=1.0")
        """,
    )

    def suite_text(workdir: Path) -> str:
        return textwrap.dedent(f"""
            suite: s
            defaults:
              workdir: {workdir}
              storage: sqlite:///{tmp_path}/suite.db
              provenance: {{revision: test-fixture-v1}}
              trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
              metric:
                name: x
                goal: minimize
                extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            studies:
              - name: only
                phases:
                  - name: p
                    n_trials: 1
                    sampler: {{ type: random, seed: 0 }}
                    search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
            """).lstrip()

    workdir_a = tmp_path / "runs_a"
    workdir_b = tmp_path / "runs_b"
    config_a = tmp_path / "suite_pub_a.yaml"
    config_a.write_text(suite_text(workdir_a))
    config_b = tmp_path / "suite_pub_b.yaml"
    config_b.write_text(suite_text(workdir_b))
    suite_a = load_config(config_a)
    assert isinstance(suite_a, Suite)
    run_suite(suite_a)
    assert _last_successful_suite_generation_id(suite_a) is not None
    shutil.copytree(workdir_a, workdir_b)

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_b)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert "published suite generation" in captured.err
    study = optuna.load_study(study_name="s__only::p", storage=f"sqlite:///{tmp_path}/suite.db")
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(
        _experiment_dir(suite_a.experiment_for_study(suite_a.studies[0]))
    )


def test_rebind_workdir_adopts_a_populated_study_that_predates_the_binding(
    tmp_path: Path,
) -> None:
    """The command is the migration path an ordinary run deliberately refuses.

    The config names the study's original tree, so the destination checks are
    what prove this workdir really owns the evidence (re-review v0.5.19 /
    blocker B1).
    """
    config_a, _config_b, _workdir_a, _workdir_b = _movable_experiment_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    assert experiment_a.storage is not None
    _drop_artifact_root_binding(experiment_a.storage, "t::p")

    result = CliRunner().invoke(cli_main, ["rebind-workdir", str(config_a)])

    assert result.exit_code == 0, result.output
    assert "(unbound)" in result.output
    study = optuna.load_study(study_name="t::p", storage=experiment_a.storage)
    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == str(_experiment_dir(experiment_a))

    # The migrated study now runs and publishes like any other bound study.
    run_experiment(load_experiment(config_a))
    assert_published_winner_evidence_local(_experiment_dir(experiment_a))


def test_rebind_workdir_converges_after_a_partially_applied_rebind(tmp_path: Path) -> None:
    """A crash between two per-study writes must be fixable by re-running the command.

    Planning therefore validates every study against the destination instead
    of requiring one coherent previous root.
    """
    config_a, config_b, workdir_a, workdir_b = _movable_two_phase_configs(tmp_path)
    experiment_a = load_experiment(config_a)
    run_experiment(experiment_a)
    shutil.copytree(workdir_a, workdir_b)
    experiment_b = load_experiment(config_b)
    assert experiment_b.storage is not None
    destination = str(_experiment_dir(experiment_b))
    optuna.load_study(study_name="t::p", storage=experiment_b.storage).set_user_attr(
        ARTIFACT_ROOT_ATTR, destination
    )

    result = CliRunner().invoke(cli_main, ["rebind-workdir", str(config_b)])

    assert result.exit_code == 0, result.output
    for phase_name in ("p", "q"):
        study = optuna.load_study(study_name=f"t::{phase_name}", storage=experiment_b.storage)
        assert study.user_attrs[ARTIFACT_ROOT_ATTR] == destination


def test_rebind_workdir_reports_that_in_memory_storage_binds_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In-memory studies never persist, so there is no binding to move."""
    config_path = write_yaml(
        tmp_path,
        f"""
        experiment: t
        workdir: {tmp_path}/runs
        trial_command: "echo {{overrides}}"
        metric:
          extractor: {{ type: json_envelope, objective_name: x, split: test, policy: test }}
        phases:
          - name: p
            n_trials: 1
            search_space: {{}}
        """,
    )
    _experiment_dir(load_experiment(config_path)).mkdir(parents=True)

    exit_code = _invoke_cli_boundary(["rebind-workdir", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "in-memory" in captured.err
    assert "Traceback" not in captured.err


def _invoke_cli_boundary(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug: bool = False,
) -> int:
    """Run the console-script entry point exactly as the installed command does.

    ``CliRunner`` invokes the Click group directly and therefore bypasses the
    process-level error boundary; these tests must exercise the boundary, so
    they call it with a patched ``sys.argv`` instead.

    :param list[str] argv: Arguments following the program name.
    :param pytest.MonkeyPatch monkeypatch: Fixture used to set ``sys.argv`` and
        the root log level.
    :param bool debug: Root log level the boundary observes. ``True`` mirrors
        what ``-v`` produces in a real process; ``_configure_logging`` cannot be
        used here because ``logging.basicConfig`` is a no-op once pytest's own
        root handler is installed.
    :return int: Status the boundary passed to ``sys.exit``.
    """
    monkeypatch.setattr(sys, "argv", ["phasesweep", *argv])
    monkeypatch.setattr(logging.getLogger(), "level", logging.DEBUG if debug else logging.INFO)
    with pytest.raises(SystemExit) as excinfo:
        cli_boundary()
    code = excinfo.value.code
    return 0 if code is None else int(code)


def _stub_run_command(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make ``phasesweep run`` reach the engine and fail with ``error``.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to replace CLI collaborators.
    :param Exception error: Exception ``run_config`` raises once the CLI calls it.
    """
    monkeypatch.setattr("phasesweep.cli.install_signal_handlers", lambda: None)
    monkeypatch.setattr("phasesweep.cli.load_config", lambda _path: object())

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("phasesweep.cli.run_config", fail)


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

    exit_code = _invoke_cli_boundary(["validate", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert str(config_path) in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_cli_boundary_reports_expected_run_failure_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An expected operational failure exits 1 with its message and no traceback."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, NoFeasibleTrialError("no feasible trial in phase 'depth'"))

    exit_code = _invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no feasible trial in phase 'depth'" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_cli_boundary_adds_traceback_for_expected_failure_when_verbose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-v`` keeps the one-line diagnostic and adds the traceback behind it."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, NoFeasibleTrialError("no feasible trial in phase 'depth'"))

    exit_code = _invoke_cli_boundary(["run", str(config_path), "-v"], monkeypatch, debug=True)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no feasible trial in phase 'depth'" in captured.err
    assert "Traceback" in captured.err


def test_cli_boundary_reports_unexpected_failure_as_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An exception that is not an expected outcome is a bug: exit 70 with a traceback."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    _stub_run_command(monkeypatch, RuntimeError("injected-internal"))

    exit_code = _invoke_cli_boundary(["run", str(config_path)], monkeypatch)

    captured = capsys.readouterr()
    assert exit_code == 70
    assert "internal error" in captured.err
    assert "Traceback" in captured.err
    assert "injected-internal" in captured.err


def test_cli_boundary_leaves_help_exit_status_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--help`` still succeeds through the boundary rather than being trapped."""
    exit_code = _invoke_cli_boundary(["--help"], monkeypatch)

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
        SamplerContinuationUnsupportedError,
        StudyContextConflictError,
        StudyFingerprintMismatchError,
        StudySchemaMismatchError,
        StudyStorageUnavailableError,
        TrialTargetRegressionError,
    ):
        assert issubclass(error_type, PhaseSweepError), error_type.__name__
