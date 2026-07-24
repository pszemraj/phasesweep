"""CLI commands: validate, show-winners, --dry-run."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import main as cli_main
from phasesweep.engine.state import (
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_path,
    _winner_path,
)
from tests.conftest import write_trainer, write_yaml


def test_help_registers_commands_and_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"], terminal_width=120)

    assert result.exit_code == 0
    assert "-h, --help" in result.output
    assert "recover-run" not in result.output
    for command in ("mcp", "run", "show-winners", "status", "validate"):
        assert command in result.output

    for command in ("run", "validate", "show-winners", "status"):
        result = runner.invoke(cli_main, [command, "--help"], terminal_width=120)
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "CONFIG" in result.output
        assert "-h, --help" in result.output

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
    for command in ("check", "init-catalog", "install", "recover-run", "serve", "uninstall"):
        assert command in mcp_help.output
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
    record = _generation_record_path(experiment, "successful")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("experiment: t\ngeneration_id: successful\nstate: published\n")
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
    search_space: {{ lr: {{ type: float, low: 1e-5, high: 1e-2, log: true }} }}
  - name: b
    inherits: [a]
    n_trials: 5
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
