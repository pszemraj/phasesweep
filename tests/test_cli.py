"""CLI commands: validate, show-winners, --dry-run."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from click.testing import CliRunner

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import main as cli_main
from tests.conftest import write_trainer, write_yaml


def test_help_output_is_operator_readable() -> None:
    """Help should describe CLI usage without leaking Python call signatures."""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"], terminal_width=120)

    assert result.exit_code == 0
    assert "Phase-chained hyperparameter sweeps driven by a YAML file." in result.output
    assert "-h, --help" in result.output
    assert "mcp" in result.output and "Serve the MCP broker." in result.output
    assert "mcp-recover-run" in result.output
    assert "Recover MCP cleanup uncertainty." in result.output
    assert "run" in result.output and "Run configured phases." in result.output
    assert "show-winners" in result.output and "Print saved phase winners." in result.output
    assert "status" in result.output and "Print read-only run status." in result.output
    assert "validate" in result.output and "Validate a config file." in result.output
    assert "Args:" not in result.output

    for command in ("run", "validate", "show-winners", "status"):
        result = runner.invoke(cli_main, [command, "--help"], terminal_width=120)
        assert result.exit_code == 0
        assert "Args:" not in result.output
        assert "config_path:" not in result.output
        assert "Usage:" in result.output
        assert "CONFIG" in result.output
        assert "-h, --help" in result.output

    recovery_help = runner.invoke(cli_main, ["mcp-recover-run", "--help"], terminal_width=120)
    assert recovery_help.exit_code == 0
    assert "Args:" not in recovery_help.output
    assert "--state-dir" in recovery_help.output
    assert "--run-id" in recovery_help.output
    assert "--confirm" in recovery_help.output
    assert "-h, --help" in recovery_help.output

    run_help = runner.invoke(cli_main, ["run", "--help"], terminal_width=120).output
    assert "--from-phase PHASE" in run_help
    assert "[default: (first phase)]" in run_help
    assert "--dry-run" in run_help
    assert "-v, --verbose" in run_help

    mcp_help = runner.invoke(cli_main, ["mcp", "--help"], terminal_width=120)
    assert mcp_help.exit_code == 0
    assert "Serve the optional MCP broker over stdio" in mcp_help.output
    assert "--catalog PATH" in mcp_help.output


def test_validate_cli_renders_comment(tmp_path: Path) -> None:
    """``phasesweep validate`` surfaces phase comments so the operator sees
    design intent next to the spec, with a ``#`` prefix to read as documentation."""
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent("""
        experiment: t
        trial_command: "echo {overrides}"
        metric:
          extractor: { type: json, path: r.json, key: x }
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
              extractor: {{ type: json, path: r.json, key: x }}
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


def test_dry_run_does_not_launch(tmp_path, caplog):
    """dry-run should log an example command but never call the trial_command."""

    caplog.set_level(logging.INFO)
    body = f"""
experiment: dry
storage: sqlite:///{tmp_path}/dry.db
workdir: {tmp_path}/runs
trial_command: "false {{overrides}}"
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
phases:
  - name: a
    n_trials: 5
    search_space: {{ lr: {{ type: float, low: 1e-5, high: 1e-2, log: true }} }}
  - name: b
    inherits: [a]
    n_trials: 5
    search_space: {{ wd: {{ type: float, low: 0, high: 0.3 }} }}
"""
    exp = load_experiment(write_yaml(tmp_path, body))
    winners = run_experiment(exp, dry_run=True)
    assert set(winners) == {"a", "b"}
    # No filesystem artifacts written.
    assert not (Path(tmp_path / "runs") / "dry").exists()
    assert not (Path(tmp_path / "runs") / "summary.yaml").exists()
    # An example command was logged
    assert any("DRY RUN example command" in r.message for r in caplog.records)


def test_status_cli_reports_phase_counts(tmp_path: Path) -> None:
    """``phasesweep status`` is read-only and reports study trial state counts."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True)
        args,_=ap.parse_known_args(); open(args.out, 'w').write(json.dumps({'x': 1.0}))
        """,
    )
    p = write_yaml(
        tmp_path,
        f"""
        experiment: status_test
        storage: sqlite:///{tmp_path}/status.db
        workdir: {tmp_path}/runs
        trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
        metric:
          name: x
          goal: minimize
          extractor: {{ type: json, path: r.json, key: x }}
        phases:
          - name: p
            n_trials: 1
            search_space: {{ x: {{ type: int, low: 0, high: 1 }} }}
        """,
    )
    exp = load_experiment(p)
    run_experiment(exp)

    result = CliRunner().invoke(cli_main, ["status", str(p)])
    assert result.exit_code == 0
    assert "status_test" in result.output
    assert "COMPLETE: 1" in result.output
