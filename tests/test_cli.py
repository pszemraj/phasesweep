"""CLI commands: validate, show-winners, --dry-run."""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import cli as cli_main
from phasesweep.cli import main as cli_boundary
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
from phasesweep.engine.state import (
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _winner_path,
)
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.runs import RunStore
from tests.conftest import write_trainer, write_yaml


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
