"""A categorical choice that YAML 1.1 loads as a numeric-looking string is warned about before any trial runs."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import cli as cli_main

_STRING_CHOICES_WARNING = (
    "Phase 'wd_sweep': search_space 'lr' has categorical choices that load as strings, not "
    "numbers: '1e-4', '3e-4', '1e-3', '3e-3'. YAML 1.1 leaves an exponent without a decimal "
    "point and sign as text (so does a quoted number), and the trainer receives a string. "
    "Write 1e-4 as 1.0e-4 or 0.0001, and the rest likewise, or ignore this if the strings "
    "are intended."
)


def _write_choices_config(tmp_path: Path, choices: str) -> Path:
    """Write a grid config whose categorical choices are the raw YAML text ``choices``."""
    path = tmp_path / "exp.yaml"
    path.write_text(
        textwrap.dedent(
            f"""\
            experiment: demo
            override_format: argparse
            trial_command: "echo x=1 {{overrides}}"
            metric:
              extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            phases:
              - name: wd_sweep
                n_trials: 4
                sampler: {{ type: grid }}
                search_space:
                  lr: {{ type: categorical, choices: {choices} }}
            """
        ),
        encoding="utf-8",
    )
    return path


def test_validate_and_dry_run_warn_about_numeric_looking_string_choices(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """YAML 1.1 loads 1e-4 as a string, so both previews say so before every trial fails on it."""
    path = _write_choices_config(tmp_path, "[1e-4, 3e-4, 1e-3, 3e-3]")

    result = CliRunner().invoke(cli_main, ["validate", str(path)])

    assert result.exit_code == 0, result.output
    assert [line.strip() for line in result.output.splitlines() if "warning:" in line] == [
        f"warning: {_STRING_CHOICES_WARNING}"
    ]
    caplog.set_level(logging.INFO)
    run_experiment(load_experiment(path), dry_run=True)
    warnings = [r for r in caplog.records if r.getMessage().startswith("DRY RUN warning:")]
    assert [(r.levelno, r.getMessage()) for r in warnings] == [
        (logging.WARNING, f"DRY RUN warning: {_STRING_CHOICES_WARNING}")
    ]


@pytest.mark.parametrize("exponent", ["999999999", "999999999999999999999"])
def test_extreme_exponent_warning_keeps_its_compact_spelling(
    tmp_path: Path, exponent: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Validation and dry runs warn without expanding enormous decimal alternatives."""
    path = _write_choices_config(tmp_path, f"[1e-{exponent}, adam, sgd, rmsprop]")
    result = CliRunner().invoke(cli_main, ["validate", str(path)])
    assert result.exit_code == 0, result.output
    spelling = f"Write 1e-{exponent} as 1.0e-{exponent},"
    assert spelling in result.output
    run_experiment(load_experiment(path), dry_run=True)
    assert any(spelling in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param("[1.0e-4, 3.0e-4, 0.001, 3.0e+3]", id="floats"),
        pytest.param("[adam, sgd, nan, inf]", id="words-and-nonfinite-names"),
        pytest.param("[2, 3, 5, null]", id="ints-and-null"),
    ],
)
def test_validate_does_not_warn_about_choices_that_are_not_numeric_strings(
    tmp_path: Path, choices: str
) -> None:
    result = CliRunner().invoke(
        cli_main, ["validate", str(_write_choices_config(tmp_path, choices))]
    )

    assert result.exit_code == 0, result.output
    assert "warning:" not in result.output
