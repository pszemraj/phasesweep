"""phasesweep CLI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from phasesweep.config import Experiment, Suite, load_config
from phasesweep.engine import config_status, run_config
from phasesweep.runtime.process import install_signal_handlers


def _configure_logging(verbose: bool) -> None:
    """Initialize root logging and tune Optuna's verbosity.

    Args:
        verbose: If True, set root logger to ``DEBUG`` and Optuna to ``INFO``;
            otherwise root is ``INFO`` and Optuna is quieted to ``WARNING``.

    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Optuna's per-trial INFO output is essentially 1:1 with phasesweep's own
    # runner.info "[phase/trial_N] <cmd>" line and adds nothing. Quiet it down
    # by default; -v restores INFO (DEBUG would surface RDB internals which we
    # don't want even in verbose mode).
    import optuna  # local import: keeps `phasesweep --help` snappy

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="phasesweep")
def main() -> None:
    """Phase-chained hyperparameter sweeps driven by a YAML file."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--from-phase", default=None, help="Skip prior phases; load their winners from disk.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Render and log an example command per phase. Don't launch any trials.",
)
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def run(config_path: Path, from_phase: str | None, dry_run: bool, verbose: bool) -> None:
    """Run all phases defined in CONFIG_PATH.

    Args:
        config_path: Path to the experiment YAML file.
        from_phase: If given, skip every phase before this one and load their
            winners from disk; the named phase must exist in the config.
        dry_run: Render and log one example trial command per phase instead of
            launching any subprocesses.
        verbose: Enable ``DEBUG`` logging for phasesweep and ``INFO`` for Optuna.

    """
    _configure_logging(verbose)
    install_signal_handlers()
    config = load_config(config_path)
    if from_phase is not None:
        if isinstance(config, Suite):
            click.echo("--from-phase is only supported for single experiment configs.", err=True)
            sys.exit(2)
        valid = [p.name for p in config.phases]
        if from_phase not in valid:
            click.echo(f"--from-phase={from_phase!r} not in {valid}", err=True)
            sys.exit(2)
    run_config(config, from_phase=from_phase, dry_run=dry_run)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config_path: Path) -> None:
    """Validate CONFIG_PATH without running anything."""
    config = load_config(config_path)
    if isinstance(config, Experiment):
        click.echo(f"OK: {config.experiment} ({len(config.phases)} phases)")
        _render_experiment_phases(config)
        return

    click.echo(f"OK: suite {config.suite} ({len(config.studies)} studies)")
    for study in config.studies:
        deps = f" depends_on={study.depends_on}" if study.depends_on else ""
        click.echo(f"  study {study.name}{deps}")
        _render_experiment_phases(config.experiment_for_study(study), indent="    ")


def _render_experiment_phases(experiment: Experiment, *, indent: str = "  ") -> None:
    """Render phase summaries for ``validate``."""
    for p in experiment.phases:
        deps = f" inherits={p.inherits}" if p.inherits else ""
        contracts = f" contracts={p.contracts}" if p.contracts else ""
        click.echo(
            f"{indent}- {p.name}: n_trials={p.n_trials} sampler={p.sampler.type}{deps}{contracts}"
        )
        if p.comment:
            for line in p.comment.strip().splitlines():
                click.echo(f"{indent}    # {line}")


@main.command(name="show-winners")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show_winners(config_path: Path) -> None:
    """Print winner.yaml from each phase that has one."""
    config = load_config(config_path)
    if isinstance(config, Suite):
        for study in config.studies:
            click.echo(f"### study {study.name}")
            _show_experiment_winners(config.experiment_for_study(study))
        return
    _show_experiment_winners(config)


def _show_experiment_winners(experiment: Experiment) -> None:
    """Print winner files for one experiment."""
    primary_root = Path(experiment.workdir).expanduser().resolve() / experiment.experiment
    for p in experiment.phases:
        wpath = primary_root / p.name / "winner.yaml"
        if wpath.is_file():
            click.echo(f"=== {p.name} ===")
            if p.comment:
                # Show design-intent before numerical results so the reader
                # frames "winner trial 7 with metric=0.32" against the original
                # hypothesis instead of the other way around.
                for line in p.comment.strip().splitlines():
                    click.echo(f"# {line}")
            click.echo(wpath.read_text())
        else:
            click.echo(f"=== {p.name} === (no winner yet)")
            if p.comment:
                for line in p.comment.strip().splitlines():
                    click.echo(f"# {line}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def status(config_path: Path) -> None:
    """Print read-only run status for CONFIG_PATH."""
    config = load_config(config_path)
    click.echo(_format_status(config_status(config)))


def _format_status(status_obj: dict) -> str:
    """Render status data as stable YAML."""
    import yaml

    return yaml.safe_dump(status_obj, sort_keys=False).rstrip()


if __name__ == "__main__":
    main()
