"""phasesweep CLI."""

from __future__ import annotations

import logging
import os
import shlex
import sys
import traceback
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import click
import optuna
import yaml

from phasesweep.config.common import CONFIG_DIR_CONTEXT
from phasesweep.config.io import (
    ConfigError,
    _load_yaml_mapping_from_text,
    describe_validation_error,
    load_experiment,
)
from phasesweep.config.models import Experiment, experiment_disclosures
from phasesweep.config.reference import (
    ReferenceEntry,
    first_sentence,
    lookup_or_suggest,
    pin_documentation_links,
    plain_path,
    reference_notes,
    rules_for,
    sub_key_count,
    top_level_entries,
)
from phasesweep.config.run_overrides import CreatedKey, SetPathError, apply_set, created_key_notes
from phasesweep.config.search import phase_draws
from phasesweep.engine.errors import ArtifactRootSchemaMismatchError, StudySchemaMismatchError
from phasesweep.engine.fingerprints import _recorded_config_matches
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_dir,
    _generation_winner_path,
    _trial_dir_for,
)
from phasesweep.engine.preview import validate_grid_targets
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.publication import (
    raise_for_unusable_publication,
)
from phasesweep.engine.read import (
    TrialTables,
    _read_generation_selection,
    _represented_winners,
    experiment_status,
    read_trial_tables,
    require_study,
)
from phasesweep.engine.run import run_experiment
from phasesweep.errors import OperatorAction, PhaseSweepError, operator_step
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.registry import (
    CatalogCheckReport,
    Registry,
    check_catalog,
    require_supported_mcp_host,
)
from phasesweep.mcp.scaffold import scaffold_catalog_text
from phasesweep.reporting import _progress_values, report_objective
from phasesweep.runtime.files import atomic_create_text, private_atomic_write_text
from phasesweep.runtime.gpu import GpuPool
from phasesweep.runtime.json import (
    JsonObject,
    JsonValue,
    json_object,
    strict_json_loads,
)
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    install_signal_handlers,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
CONFIG_PATH = click.Path(exists=True, dir_okay=False, path_type=Path)
CONFIG_OVERRIDES = click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Override a config value with a YAML VALUE; repeat as needed. KEY is dot-separated "
        "names (env.NOTE). Select a list element by zero-based index, as phases.0 or "
        "phases[0], and a phase by name, as phases[NAME]. Write [KEY] to take a mapping key "
        "verbatim when it contains dots: phases[NAME].search_space[model.n_layers].choices. "
        "Paths resolve against the config file's directory, just as in the config."
    ),
)

# Exit taxonomy for the process-level boundary in main(). Commands that already
# choose an exit status keep it; these are the codes the boundary itself uses.
_USAGE_EXIT = 2  # bad input: config or arguments
_FAILURE_EXIT = 1  # the run itself failed
_INTERNAL_EXIT = 70  # sysexits EX_SOFTWARE: a phasesweep bug
_ABORT_EXIT = 130  # 128 + SIGINT

# Column limits for the `explain` listing.
_SIGNATURE_COLUMN_CAP = 40
_SUMMARY_WIDTH = 72


def _configure_logging(verbose: bool) -> None:
    """Initialize root logging and tune Optuna's verbosity.

    :param bool verbose: If True, set root logger to ``DEBUG`` and Optuna to ``INFO``;
        otherwise root is ``INFO`` and Optuna is quieted to ``ERROR``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.ERROR)


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Phase-chained hyperparameter sweeps driven by a YAML file.",
)
@click.version_option(package_name="phasesweep")
def cli() -> None:
    """Run the phasesweep command line interface."""


def _operational_error_text(exc: PhaseSweepError, *, read_only: bool = False) -> str:
    """Format an operational diagnosis with advice for the requested operation.

    :param PhaseSweepError exc: Operational failure to render.
    :param bool read_only: Whether the command only inspects existing state.
    :return str: Diagnostic followed by its operator step.
    """
    if read_only and isinstance(exc, (ArtifactRootSchemaMismatchError, StudySchemaMismatchError)):
        return (
            f"{exc}\nfix: inspect the raw artifacts, or use the exact checkout that produced "
            "them for inspection if available; this checkout reads only its current format."
        )
    step = operator_step(exc)
    return f"{exc}\nfix: {step}"


def main() -> None:
    """Run the CLI behind a process-level error boundary.

    Click exceptions reach this boundary; explicit ``SystemExit`` statuses pass through.

    :raises SystemExit: Always, carrying one of ``_USAGE_EXIT`` (bad input),
        ``_FAILURE_EXIT`` (the run failed), ``_INTERNAL_EXIT`` (a phasesweep
        bug), ``_ABORT_EXIT`` (interrupted), or the status the command or
        Click chose.
    """
    operation = {"read_only": False}
    try:
        status = cli.main(standalone_mode=False, obj=operation)
    except click.ClickException as exc:
        # Includes UsageError, which already carries exit code 2.
        if (
            isinstance(exc, click.NoSuchOption)
            and exc.option_name == "--workdir"
            and exc.ctx is not None
            and any(
                isinstance(param, click.Option) and "--set" in param.opts
                for param in exc.ctx.command.params
            )
        ):
            exc = click.UsageError(
                f"{exc.message} Use --set workdir=PATH; relative paths resolve against the config file's directory.",
                ctx=exc.ctx,
            )
        exc.show()
        sys.exit(exc.exit_code)
    except (click.Abort, KeyboardInterrupt):
        click.echo("Aborted.", err=True)
        sys.exit(_ABORT_EXIT)
    except ConfigError as exc:
        click.echo(f"phasesweep: {exc}", err=True)
        sys.exit(_USAGE_EXIT)
    except PhaseSweepError as exc:
        click.echo(
            f"phasesweep: {_operational_error_text(exc, read_only=operation['read_only'])}",
            err=True,
        )
        # -v (see _configure_logging) means the operator asked for internals.
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        sys.exit(_FAILURE_EXIT)
    except OSError as exc:
        click.echo(f"phasesweep: operational error: {exc}", err=True)
        sys.exit(_FAILURE_EXIT)
    except Exception:  # the boundary's purpose is to report bugs
        click.echo("phasesweep: internal error — please report this traceback.", err=True)
        traceback.print_exc()
        sys.exit(_INTERNAL_EXIT)
    sys.exit(status if isinstance(status, int) else 0)


def _load_cli_config(path: Path) -> Experiment:
    """Load and fully validate a CLI config with path-aware diagnostics.

    :param Path path: Config file supplied to a CLI command.
    :return Experiment: Validated experiment configuration.
    :raises ConfigError: The file cannot be parsed or fails model validation.
    """
    from pydantic import ValidationError

    try:
        return load_experiment(path)
    except ConfigError:
        raise
    except ValidationError as exc:
        raise ConfigError(f"{path}: {describe_validation_error(exc)}") from exc
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


_DEFAULT_TEMPLATE = "starter"
_TEMPLATE_PURPOSE_PREFIX = "# phasesweep template: "


def _template_names() -> list[str]:
    """List the packaged templates: the default first, then the rest by name.

    The catalog is the ``templates`` directory listing, so a template is its file name
    without ``.yaml``.

    :return list[str]: Template names.
    """
    entries = resources.files("phasesweep").joinpath("templates").iterdir()
    names = sorted(
        entry.name.removesuffix(".yaml") for entry in entries if entry.name.endswith(".yaml")
    )
    return sorted(names, key=lambda name: name != _DEFAULT_TEMPLATE)


def _template_text(name: str) -> str:
    """Read one packaged template with links pinned to the installed source.

    :param str name: Template name from ``_template_names``.
    :return str: The template's YAML text.
    """
    return pin_documentation_links(
        resources.files("phasesweep")
        .joinpath("templates", f"{name}.yaml")
        .read_text(encoding="utf-8")
    )


def _template_purpose(name: str) -> str:
    """Read a template's one-line purpose from its first line.

    :param str name: Template name from ``_template_names``.
    :raises ValueError: The template's first line is not the purpose header.
    :return str: The purpose text after the header prefix.
    """
    first_line = _template_text(name).splitlines()[0]
    if not first_line.startswith(_TEMPLATE_PURPOSE_PREFIX):
        raise ValueError(f"Template {name!r} must start with {_TEMPLATE_PURPOSE_PREFIX!r}.")
    return first_line.removeprefix(_TEMPLATE_PURPOSE_PREFIX).strip()


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help="Write a runnable template experiment without overwriting files.",
    short_help="Create an experiment from a template.",
)
@click.option(
    "--template",
    type=click.Choice(_template_names()),
    default=_DEFAULT_TEMPLATE,
    show_default=True,
    help="Template to write. Every one runs as written on the packaged toy trainer.",
)
@click.option("--list", "list_templates", is_flag=True, help="List the templates and exit.")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=".",
    show_default="the current directory",
    help="Destination file, or a directory that receives TEMPLATE.yaml. A directory is an "
    "existing one or a path ending in a separator, and is created if needed.",
)
def init(template: str, list_templates: bool, output: str) -> None:
    """Write an annotated template experiment and print the review commands.

    :param str template: Name of the packaged template to write.
    :param bool list_templates: Print each template's name and purpose instead of writing.
    :param str output: Destination file, or a directory that receives ``<template>.yaml``;
        existing paths are never replaced.
    :raises click.exceptions.Exit: With code 2 when the destination already exists as a
        file or symlink, including when it appears between the check and the
        atomic publication; with code 1 when staging or publishing fails with
        an ``OSError`` such as a permission, disk-space, or hard-link error.
    """
    if list_templates:
        names = _template_names()
        width = max(len(name) for name in names)
        for name in names:
            click.echo(f"{name:<{width}}  {_template_purpose(name)}")
        return
    expanded = Path(output).expanduser()
    if expanded.is_dir() or output.endswith(("/", os.sep)):
        expanded = expanded / f"{template}.yaml"
    target = expanded.absolute()
    if target.exists() or target.is_symlink():
        click.echo(f"phasesweep init: refusing to overwrite existing path {target}", err=True)
        raise click.exceptions.Exit(2)
    text = _template_text(template)
    try:
        if not atomic_create_text(target, text):
            click.echo(f"phasesweep init: refusing to overwrite existing path {target}", err=True)
            raise click.exceptions.Exit(2)
    except OSError as exc:
        # `phasesweep init` is the first command a new user runs; an unwritable
        # directory or a filesystem without hard links must report one line, not
        # a traceback.
        click.echo(f"phasesweep init: cannot write {target}: {exc}", err=True)
        raise click.exceptions.Exit(1) from None

    # Quote the expanded path, never the raw option value: shell quoting
    # suppresses "~" expansion, so a quoted raw "~/x.yaml" would name a
    # different (nonexistent) file than the one just written.
    config_arg = shlex.quote(str(expanded))
    click.echo(f"Wrote the {template} template to {target}")
    click.echo("\nNext:")
    click.echo(f"  phasesweep validate {config_arg}")
    click.echo(f"  phasesweep run {config_arg} --dry-run")
    click.echo(f"  phasesweep run {config_arg}")
    click.echo(f"  phasesweep status {config_arg} --trials")
    click.echo(f"  phasesweep mcp init-catalog --from {config_arg}")
    click.echo("\nphasesweep explain lists every config key; phasesweep explain KEY describes one.")
    click.echo("phasesweep init --list describes the other templates.")


@cli.command(
    "report-objective",
    context_settings=CONTEXT_SETTINGS,
    help="Publish a JSON-envelope objective from inside a PhaseSweep trial.",
    short_help="Publish this trial's objective.",
)
@click.argument("value", type=float)
@click.option("--name", required=True, help="Objective name declared by the extractor.")
@click.option("--split", required=True, help="Evaluated data split.")
@click.option(
    "--policy",
    required=True,
    help="Evaluation policy, the exact string the metric extractor's policy names, such as final or best.",
)
@click.option("--checkpoint", required=True, help="Checkpoint identity for this evaluation.")
@click.option("--step", required=True, type=click.IntRange(min=0), help="Evaluation step.")
@click.option(
    "--progress",
    "progress_items",
    multiple=True,
    metavar="NAME=VALUE",
    help="Position on another progress axis at this evaluation, such as tokens=1048576. Repeatable.",
)
def report_objective_cmd(
    value: float,
    name: str,
    split: str,
    policy: str,
    checkpoint: str,
    step: int,
    progress_items: tuple[str, ...],
) -> None:
    """Publish one objective through the trainer-side reporting API.

    :param float value: Finite objective value.
    :param str name: Configured objective name.
    :param str split: Evaluated data split.
    :param str policy: Evaluation policy.
    :param str checkpoint: Checkpoint identity.
    :param int step: Non-negative evaluation step.
    :param tuple[str, ...] progress_items: ``NAME=VALUE`` progress positions.
    :raises click.BadParameter: If the objective metadata is invalid.
    :raises click.ClickException: If the PhaseSweep trial environment is absent.
    """
    progress: dict[str, int | float] = {}
    for item in progress_items:
        axis, separator, raw = item.partition("=")
        if not separator or axis in progress:
            raise click.BadParameter(
                f"expected unique NAME=VALUE pairs, got {item!r}.", param_hint="'--progress'"
            )
        try:
            progress[axis] = int(raw)
        except ValueError:
            try:
                progress[axis] = float(raw)
            except ValueError:
                raise click.BadParameter(
                    f"{item!r} does not end in a number.", param_hint="'--progress'"
                ) from None
    try:
        _progress_values(progress)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--progress'") from exc
    try:
        destination = report_objective(
            value,
            name=name,
            split=split,
            policy=policy,
            checkpoint=checkpoint,
            step=step,
            progress=progress or None,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="objective") from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote objective to {destination}")


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Run configured phases. Use --from-phase to skip earlier phases; only the winners "
        "inherited by the remaining phases are required."
    ),
    short_help="Run configured phases.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@click.option(
    "--from-phase",
    metavar="PHASE",
    default=None,
    show_default="first phase",
    help="Skip earlier phases and load winners required by the remaining phases.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    show_default=True,
    help=(
        "Print what a run would launch and one example command per phase, and refuse most of "
        "what a run would refuse over the recorded trials, without launching trials."
    ),
)
@CONFIG_OVERRIDES
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    show_default=True,
    help="Show debug logs from phasesweep and INFO logs from Optuna.",
)
def run(
    config_path: Path,
    from_phase: str | None,
    dry_run: bool,
    settings: tuple[str, ...],
    verbose: bool,
) -> None:
    """Run all phases defined in ``config_path``.

    :param Path config_path: Path to the experiment YAML file.
    :param str | None from_phase: Optional phase name to start from after loading
        required earlier winners from disk.
    :param bool dry_run: Preview the trials each phase would launch, refuse most of what a
        run would refuse over the recorded trials, and render example commands, without
        launching subprocesses.
    :param tuple[str, ...] settings: ``KEY=VALUE`` assignments with YAML values.
    :param bool verbose: Enable debug logging for phasesweep and INFO logging for Optuna.
    :raises ConfigError: The config file cannot be parsed or fails model validation.
    :raises SystemExit: With code 2 when ``from_phase`` names no configured phase; a
        ``PhaseSweepShutdown`` from an interrupted run is re-raised unchanged.
    """
    click.get_current_context().ensure_object(dict)["read_only"] = dry_run
    _configure_logging(verbose)
    # Install before config parsing so CLI SIGINT/SIGTERM behavior and exit
    # codes remain structured throughout startup, including dry-run previews.
    # run_experiment() repeats this idempotently for direct library callers.
    install_signal_handlers()
    config = _command_config(config_path, settings=settings)
    if from_phase is not None:
        valid = [p.name for p in config.phases]
        if from_phase not in valid:
            click.echo(f"--from-phase={from_phase!r} not in {valid}", err=True)
            sys.exit(2)
    try:
        run_experiment(config, from_phase=from_phase, dry_run=dry_run)
    except PhaseSweepShutdown as exc:
        if exc.published_result_committed:
            click.echo(
                "phasesweep: shutdown was honored after the published result was committed; "
                "the result is available for inspection.",
                err=True,
            )
        raise


def _command_config(path: Path, *, settings: tuple[str, ...]) -> Experiment:
    """Apply invocation overrides and validate the derived configuration.

    :param Path path: Original config file, which remains unchanged.
    :param tuple[str, ...] settings: ``KEY=VALUE`` assignments with YAML values; the key
        syntax is :data:`phasesweep.config.run_overrides.SET_SYNTAX`.
    :raises click.BadParameter: An assignment cannot address a config value.
    :raises ConfigError: The derived config fails normal model validation.
    :return Experiment: Fully validated config with original path anchoring.
    """
    from pydantic import ValidationError

    if not settings:
        return _load_cli_config(path)
    try:
        payload = json_object(_load_yaml_mapping_from_text(path.read_text(encoding="utf-8"), path))
    except ConfigError:
        raise
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    created: list[CreatedKey] = []
    for setting in settings:
        try:
            new_key = apply_set(payload, setting)
        except SetPathError as exc:
            raise click.BadParameter(str(exc), param_hint="--set") from exc
        if new_key is not None:
            created.append(new_key)
    try:
        return Experiment.model_validate(
            payload, context={CONFIG_DIR_CONTEXT: path.resolve().parent}
        )
    except ValidationError as exc:
        detail = "\n".join([describe_validation_error(exc), *created_key_notes(exc, created)])
        raise ConfigError(f"{path} (overrides): {detail}") from exc


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help="Validate a phasesweep experiment config without launching any trials.",
    short_help="Validate a config file.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@CONFIG_OVERRIDES
def validate(config_path: Path, settings: tuple[str, ...]) -> None:
    """Validate config and read-only GPU inventory without launching work.

    :param Path config_path: Experiment YAML whose relative paths anchor to its directory.
    :param tuple[str, ...] settings: Config overrides applied before validation.
    :raises ConfigError: The derived configuration cannot be loaded or validated.
    :raises TrialTargetRefusedError: A grid target exceeds its recorded retry ceiling.
    :raises StudyStorageUnavailableError: The ledger cannot be read to check a raised grid target.
    """
    config = _command_config(config_path, settings=settings)
    validate_grid_targets(config)
    for phase in config.phases:
        GpuPool.for_phase(phase, cuda_visible_devices=config.env.get("CUDA_VISIBLE_DEVICES"))
    click.echo(f"OK: {config.experiment} ({len(config.phases)} phases)")
    _render_experiment_phases(config)


def _render_experiment_phases(experiment: Experiment, *, indent: str = "  ") -> None:
    """Render phase summaries and sampler capability lines for ``validate``.

    :param Experiment experiment: Experiment whose phases should be printed.
    :param str indent: Prefix to place before each rendered phase line.
    """
    for phase, lines in experiment_disclosures(experiment):
        prefix = indent
        if phase is not None:
            deps = f" inherits={phase.inherits}" if phase.inherits else ""
            sampling = (
                f"sampler={phase.sampler.type}" if phase_draws(phase) else "(fixed configuration)"
            )
            click.echo(f"{indent}- {phase.name}: n_trials={phase.n_trials} {sampling}{deps}")
            prefix += "    "
        for warning, line in lines:
            click.echo(f"{prefix}{'warning: ' if warning else ''}{line}")
        if phase is not None:
            _render_phase_comment(phase.comment, prefix=f"{indent}    # ")


def _render_phase_comment(comment: str | None, *, prefix: str) -> None:
    """Render a phase comment one line at a time with ``prefix``.

    :param str | None comment: Optional phase comment to render.
    :param str prefix: Prefix to place before each rendered comment line.
    """
    if comment:
        for line in comment.strip().splitlines():
            click.echo(f"{prefix}{line}")


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Explain config keys from the packaged reference. Without KEY, list the top-level "
        "keys; with a KEY such as 'phases[].sampler', print the section rules and shared-block "
        "descriptions that govern it, then that key and every key below it. "
        "The spellings phases[].sampler, phases.sampler and phases.0.sampler are equivalent. "
        "The broad phases key prints the rules over the phase list and its immediate "
        "children; narrow it for full details. With --notes, print the general notes on the "
        "config file, paths, YAML typing and the study fingerprint instead."
    ),
    short_help="Explain config keys.",
)
@click.argument("key", metavar="[KEY]", required=False)
@click.option(
    "--notes",
    is_flag=True,
    help="Print the general notes on the config file, paths, YAML typing and the study fingerprint.",
)
def explain(key: str | None, notes: bool) -> None:
    """Print the configuration reference for one key, list the top-level keys, or print its notes.

    A key prints the section rules and shared-block descriptions that govern its entries
    first, then the entries themselves; the broad ``phases`` key prints the rules of its own
    section and a compact index of its fields.

    :param str | None key: Config key in any spelling the reference accepts, or None to list
        the top-level keys.
    :param bool notes: Print the reference's general notes instead of keys.
    :raises click.UsageError: ``notes`` is combined with a ``key``.
    :raises click.exceptions.Exit: With code 2 when ``key`` matches no documented key.
    """
    if notes:
        if key is not None:
            raise click.UsageError("--notes takes no KEY.")
        for line in reference_notes():
            click.echo(line)
        return
    if key is None:
        _render_reference_index()
        return
    entries, near = lookup_or_suggest(key)
    if not entries:
        hint = (
            f"Closest: {', '.join(near)}."
            if near
            else "Run `phasesweep explain` to list the top-level keys."
        )
        click.echo(f"phasesweep explain: no config key matches {key!r}. {hint}", err=True)
        raise click.exceptions.Exit(_USAGE_EXIT)
    if plain_path(key) == "phases":
        # The compact index carries only the rules over the whole phase list.
        _render_rules(entries[:1])
        for entry in entries:
            if plain_path(entry.path).count(".") == 1:
                click.echo(f"{entry.path}: {entry.signature}")
                click.echo(f"  {first_sentence(entry, _SUMMARY_WIDTH)}")
        click.echo("Narrow with a key: phasesweep explain 'phases[].search_space'")
        return
    _render_rules(entries)
    for index, entry in enumerate(entries):
        if index:
            click.echo()
        click.echo(f"{entry.path}: {entry.signature}")
        for line in entry.comments:
            click.echo(f"  {line}")


def _render_rules(entries: Sequence[ReferenceEntry]) -> None:
    """Print the section rules and shared-block descriptions that govern some entries.

    :param Sequence[ReferenceEntry] entries: Entries about to be printed, in file order.
    """
    for rules in rules_for(entries):
        click.echo(f"{rules.title}:")
        for line in rules.lines:
            click.echo(f"  {line}")
        click.echo()


def _render_reference_index() -> None:
    """Print each section's top-level keys with signature, summary and sub-key count."""
    entries = top_level_entries()
    width = max(len(entry.path) for entry in entries)
    # A signature longer than the cap overflows its column instead of widening every row.
    sig_width = max(len(e.signature) for e in entries if len(e.signature) <= _SIGNATURE_COLUMN_CAP)
    section = None
    for entry in entries:
        if entry.section != section:
            if section is not None:
                click.echo()
            click.echo(entry.section)
            section = entry.section
        count = sub_key_count(entry)
        suffix = f" ({count} sub-key{'' if count == 1 else 's'})" if count else ""
        summary = first_sentence(entry, _SUMMARY_WIDTH)
        click.echo(f"  {entry.path:<{width}}  {entry.signature:<{sig_width}}  {summary}{suffix}")
    click.echo()
    click.echo("Narrow with a key, for example: phasesweep explain 'phases[].sampler'")
    click.echo(
        "General notes on the config file, paths, YAML typing and the study fingerprint: "
        "phasesweep explain --notes"
    )


@cli.command(
    name="show-winners",
    context_settings=CONTEXT_SETTINGS,
    help="Print concise phase winners, including unpublished results from the latest run.",
    short_help="Print saved phase winners.",
)
@click.argument("config_path", metavar="CONFIG_YAML", type=CONFIG_PATH)
@CONFIG_OVERRIDES
def show_winners(config_path: Path, settings: tuple[str, ...]) -> None:
    """Print concise winners from published or unfinished generation records.

    :param Path config_path: Config used to locate the publication and label drift as historical.
    :param tuple[str, ...] settings: The same config overrides used by the run.
    :raises PublicationAccessError: This user cannot validate the publication.
    :raises PublicationIntegrityError: The publication is corrupt; nothing is rendered.
    :raises WinnerIntegrityError: An unpublished winner's producing snapshot is unavailable or invalid.
    """
    click.get_current_context().ensure_object(dict)["read_only"] = True
    experiment = _command_config(config_path, settings=settings)
    require_study(experiment)
    validate_ledger(experiment)
    _, generation_id, publication = _read_generation_selection(experiment, latest=True)
    raise_for_unusable_publication(
        experiment.experiment, publication.state, publication.error, _experiment_dir(experiment)
    )
    is_published = publication.validated is not None and generation_id == publication.generation_id
    validated = publication.validated if is_published else None
    producing = (
        _load_generation_experiment(_experiment_dir(experiment), generation_id)
        if generation_id is not None and not is_published
        else experiment
    )
    phase_names = (
        [phase.name for phase in validated.phase_plan]
        if validated is not None
        else [phase.name for phase in producing.phases]
    )
    views = _represented_winners(
        experiment,
        publication,
        generation_id,
        phase_names,
        fallback_published=True,
    )
    winners = {view.phase: view for view in views}
    phase_plan: list[tuple[str, str | None]] = [(p.name, p.comment) for p in producing.phases]
    if generation_id is not None:
        label = "" if is_published else " (unpublished)"
        generation_path = _generation_dir(_experiment_dir(experiment), generation_id)
        click.echo(f"# generation: {generation_path}{label}")
    if not is_published and any(
        not view.is_published and view.config_matches_current is False for view in views
    ):
        click.echo(
            f"# Historical experiment result: unpublished generation {generation_id} "
            "does not match the current config."
        )
    if validated is not None:
        phase_plan = [(phase.name, phase.comment) for phase in validated.phase_plan]
        config_matches = _recorded_config_matches(
            experiment,
            validated.summary.get("config_fingerprint"),
            phase_count=len(validated.phase_plan),
        )
        if config_matches is True and len(experiment.phases) > len(validated.phase_plan):
            click.echo("# Published phase results; appended phases have not run yet.")
            phase_plan.extend(
                (phase.name, phase.comment)
                for phase in experiment.phases[len(validated.phase_plan) :]
            )
        elif config_matches is False:
            click.echo(
                "# Historical experiment result: published generation "
                f"{validated.generation_id} does not match the current config."
            )
            click.echo("# Rendering the saved phase plan and annotations.")
        revision = validated.code_revision
        if revision is None:
            click.echo("# code revision: unknown (not a git work tree or git unavailable)")
        else:
            dirty = " (dirty)" if revision["dirty"] else ""
            click.echo(f"# code revision: {revision['commit']}{dirty}")
    trainer_bases: dict[str | None, JsonObject] = {}
    for view in views:
        winner_publication = publication.validated if view.is_published else None
        trainer_base = (
            winner_publication.argparse_trainer_config
            if winner_publication is not None
            else producing.trainer_config
            if generation_id is not None and producing.override_format == "argparse"
            else None
        )
        if trainer_base is not None:
            trainer_bases[view.generation_id] = trainer_base
    multiple_sources = len({view.generation_id for view in views}) > 1
    for source_id, trainer_base in trainer_bases.items():
        source_label = f" for generation {source_id}" if multiple_sources else ""
        click.echo(f"# Producing argparse trainer base{source_label}, before phase overrides.")
        click.echo(yaml.safe_dump({"trainer_config": trainer_base}, sort_keys=False))

    for name, comment in phase_plan:
        winner = winners.get(name)
        if winner is not None:
            winner_publication = publication.validated if winner.is_published else None
            metric_name = producing.metric.name
            if winner_publication is not None:
                metric_name = str(winner_publication.metric["name"])
                comment = next(
                    phase.comment for phase in winner_publication.phase_plan if phase.name == name
                )
            click.echo(f"=== {name} ===")
            click.echo(
                f"# winner generation: {winner.generation_id} "
                f"({'published' if winner.is_published else 'unpublished'})"
            )
            if winner.is_published and not is_published and winner.config_matches_current is False:
                click.echo(
                    "# Historical published phase result: does not match the current config."
                )
            _render_phase_comment(comment, prefix="# ")
            if winner.repeats is not None:
                repeats = winner.repeats
                click.echo(f"# metric is the {repeats['reduce']} over {repeats['n']} repeats.")
                click.echo(f"# trial_number {winner.trial_number} names the group's last member.")
                click.echo(
                    "# Choose a checkpoint explicitly from repeats.members below; each member includes its trial_dir."
                )
            if winner.generation_id is not None:
                click.echo(
                    f"# full record: {_generation_winner_path(experiment, winner.generation_id, name)}"
                )
            payload: JsonObject = {
                "trial_number": winner.trial_number,
                "metric": {metric_name: winner.metric},
                "params": json_object(winner.params),
                "effective_overrides": json_object(winner.effective_overrides),
                "gates_passed": winner.gates_passed,
                "incomplete": winner.incomplete,
                "observed": dict(winner.observed),
                "confound": None if winner.confound is None else json_object(winner.confound),
                "repeats": None if winner.repeats is None else json_object(dict(winner.repeats)),
                "cost_advisory": winner.cost_advisory,
            }
            if winner.repeats_detail is not None:
                detail = winner.repeats_detail
                if isinstance(payload["repeats"], dict):
                    payload["repeats"]["key"] = detail["key"]
                    payload["repeats"]["members"] = [
                        {
                            "trial_number": member["trial_number"],
                            "repeat": member["repeat"],
                            "trial_dir": str(
                                _trial_dir_for(
                                    experiment,
                                    name,
                                    member["trial_number"],
                                    generation_id=member["generation_id"],
                                    attempt_id=member["attempt_id"],
                                )
                            ),
                        }
                        for member in detail["members"]
                    ]
            click.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())
        else:
            click.echo(f"=== {name} === (no winner yet)")
            _render_phase_comment(comment, prefix="# ")


def _param_text(value: JsonValue) -> str:
    """Format one sampled param for a trial table, floats to four significant digits.

    :param JsonValue value: The recorded param value.
    :return str: Terminal text for the value.
    """
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _trial_cell(column: str, value: JsonValue) -> str:
    """Format one trial-table cell for the terminal.

    Metric-like numbers print to six significant digits, params to four, and
    durations to a tenth of a second; a missing value prints as ``-``.

    :param str column: Column header the cell sits under.
    :param JsonValue value: The row's value for that column.
    :return str: Terminal text for the cell.
    """
    if value is None:
        return "-"
    if isinstance(value, dict):
        return ", ".join(f"{name}={_param_text(item)}" for name, item in value.items()) or "-"
    if isinstance(value, float):
        return f"{value:.1f}" if column == "duration_s" else f"{value:.6g}"
    return str(value)


#: Columns whose text is not a number, which a trial table left-aligns.
_TEXT_TRIAL_COLUMNS = frozenset({"state", "params", "winner", "reduce", "eligible"})


def _aligned_row(columns: Sequence[str], texts: Sequence[str], widths: Sequence[int]) -> str:
    """Pad one table row to its column widths, text columns left and numbers right.

    :param Sequence[str] columns: Column headers, which pick each column's alignment.
    :param Sequence[str] texts: The row's cell texts, one per column.
    :param Sequence[int] widths: Width of each column.
    :return str: The row joined by two spaces, without trailing spaces.
    """
    padded = (
        text.ljust(width) if column in _TEXT_TRIAL_COLUMNS else text.rjust(width)
        for column, text, width in zip(columns, texts, widths, strict=True)
    )
    return "  ".join(padded).rstrip()


def _table_lines(columns: Sequence[str], rows: Sequence[JsonObject]) -> list[str]:
    """Render trial or group rows as aligned plain text.

    :param Sequence[str] columns: Column names in display order.
    :param Sequence[JsonObject] rows: Rows to display, with every named column present.
    :return list[str]: The header line followed by one line per row, without trailing spaces.
    """
    cells = [[_trial_cell(column, row[column]) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in cells))
        for index, column in enumerate(columns)
    ]
    return [
        _aligned_row(columns, columns, widths),
        *(_aligned_row(columns, row, widths) for row in cells),
    ]


def _render_trial_tables(tables: TrialTables, *, persisted: bool) -> str:
    """Render every phase's trial table under a header that names the winner marks' source.

    :param TrialTables tables: The tables one publication resolution produced.
    :param bool persisted: Whether the experiment keeps its trials in a ledger.
    :return str: The header line, then each phase's heading and table.
    """
    represented = tables.represented_generation_id or tables.published_generation_id
    sources = set(tables.winner_generations.values())
    if len(sources) == 1:
        represented, published = next(iter(sources))
        marks = f"winner (*) marks {'published' if published else 'unpublished'} generation {represented}"
    elif len(sources) > 1:
        marks = "winner (*) marks the phase generations listed below"
    elif represented is None:
        marks = "no generation is published, so no winner is marked"
    elif represented != tables.published_generation_id:
        marks = f"winner (*) marks unpublished generation {represented}"
    else:
        marks = f"winner (*) marks published generation {represented}"
    lines = [f"experiment {tables.experiment}: {marks}"]
    for table in tables.phases:
        if not table.available:
            note = "trial data unavailable"
        elif not persisted:
            note = "no trials recorded: storage is null"
        elif not table.rows:
            note = "no trials yet"
        else:
            note = f"{len(table.rows)} trial{'' if len(table.rows) == 1 else 's'}"
        lines.extend(["", f"=== {table.phase} === ({note})"])
        source = tables.winner_generations.get(table.phase)
        if source is not None:
            source_id, published = source
            lines.append(
                f"winner generation: {source_id} ({'published' if published else 'unpublished'})"
            )
        if table.rows:
            lines.extend(_table_lines(table.columns, table.rows))
        if table.groups:
            groups: list[JsonObject] = [
                {
                    "group": group.group,
                    "winner": "*" if group.winner else "",
                    "params": group.params,
                    "n/expected": f"{group.n}/{group.expected}",
                    "reduce": group.reduce,
                    "aggregate": group.aggregate,
                    "mean": group.mean,
                    "sd": group.sd,
                    "min": group.min,
                    "max": group.max,
                    "eligible": "yes" if group.eligible else "no",
                }
                for group in table.groups
            ]
            observed_names = sorted({name for group in table.groups for name in group.observed})
            for group, row in zip(table.groups, groups, strict=True):
                for name in observed_names:
                    row[f"observed:{name}"] = group.observed.get(name)
                    row[f"observed_n:{name}"] = group.observed_n.get(name, 0)
            lines.extend(
                [
                    "",
                    "Repeat groups (statistics include infeasible measured members):",
                    "Group numbers track the highest member trial number so far "
                    "and can change as members arrive.",
                ]
            )
            lines.extend(_table_lines(tuple(groups[0]), groups))
    return "\n".join(lines)


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Print read-only trial counts and phase state for a phasesweep experiment, as YAML; "
        "--trials prints trial and repeat-group tables per phase instead."
    ),
    short_help="Print read-only run status.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@CONFIG_OVERRIDES
@click.option(
    "--trials",
    "show_trials",
    is_flag=True,
    help=(
        "Print one aligned table of trials per phase (state, value, params, duration, "
        "repeat group, observed values, failure reasons, winner) and repeat-group statistics "
        "instead of YAML."
    ),
)
def status(config_path: Path, show_trials: bool, settings: tuple[str, ...]) -> None:
    """Print read-only run status for ``config_path``.

    The payload is printed first and in full even when it reports a corrupt
    publication: the operator needs the trial counts and generation identity
    to decide what to inspect, and the boundary's one-line diagnostic follows
    on stderr. ``--trials`` prints the trial tables in its place, from one
    publication resolution, and exits the same way.

    :param Path config_path: Experiment YAML file to inspect.
    :param bool show_trials: Print per-phase trial tables instead of the YAML status.
    :param tuple[str, ...] settings: The same config overrides used by the run.
    :raises PublicationAccessError: The reported publication cannot be
        validated as the current user.
    :raises PublicationIntegrityError: The reported publication no longer validates.
    """
    click.get_current_context().ensure_object(dict)["read_only"] = True
    config = _command_config(config_path, settings=settings)
    require_study(config)
    if show_trials:
        tables = read_trial_tables(config, latest=True)
        click.echo(_render_trial_tables(tables, persisted=config.resolved_storage is not None))
        raise_for_unusable_publication(
            tables.experiment,
            tables.publication_integrity,
            tables.publication_error,
            _experiment_dir(config),
        )
        return
    payload = experiment_status(config)
    click.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())
    raise_for_unusable_publication(
        str(payload["experiment"]),
        str(payload["publication_integrity"]),
        None if payload.get("publication_error") is None else str(payload["publication_error"]),
        Path(str(payload["workdir"])),
    )


@cli.command(
    "sync-tracking",
    context_settings=CONTEXT_SETTINGS,
    help="Retry pending MLflow uploads without training or metric selection.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@CONFIG_OVERRIDES
@click.option(
    "--generation",
    "generation_id",
    default=None,
    help="Retry only this generation's uploads instead of every pending one.",
)
def sync_tracking(config_path: Path, generation_id: str | None, settings: tuple[str, ...]) -> None:
    """Retry retained native MLflow result projections.

    :param Path config_path: Matching experiment configuration.
    :param str | None generation_id: Generation to synchronize, or ``None`` for all.
    :param tuple[str, ...] settings: The same config overrides used by the run.
    :raises ClickException: The generation name or a delivery record is malformed.
    :raises PhaseSweepError: Nothing matching can be synchronized, or uploads remain pending.
    """
    from phasesweep.tracking.publication import sync_tracking as sync

    config = _command_config(config_path, settings=settings)
    require_study(config)
    try:
        counts = sync(config, generation_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(yaml.safe_dump(counts, sort_keys=False).rstrip())
    if counts["pending"]:
        raise PhaseSweepError("MLflow uploads remain pending.", action=OperatorAction.RETRY)


@cli.group(
    context_settings=CONTEXT_SETTINGS,
    help="Manage the optional MCP server, catalog, and operator recovery.",
    short_help="Manage the MCP server and agent integrations.",
)
def mcp() -> None:
    """Run MCP operator commands."""


def _catalog_error_text(exc: CatalogError) -> str:
    """Render a catalog error with its optional actionable fix; paths may be included.

    :param CatalogError exc: Raised catalog error to render.
    :return str: Message, plus a trailing ``fix:`` line when a suggestion exists.
    """
    if exc.suggestion:
        return f"{exc}\nfix: {exc.suggestion}"
    return str(exc)


@mcp.command(
    name="recover-run",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Operator-only recovery for MCP cleanup uncertainty, interrupted publication, or "
        "terminal-result finalization. If study storage is unavailable, restore the original "
        "complete storage ledger and access to it before recovery. "
        "--confirm performs the reported actions."
    ),
    short_help="Recover MCP cleanup or result finalization.",
)
@click.option(
    "--state-dir",
    required=True,
    # RunStore.open_existing validates the normalized path without creating it.
    # Normalization happens inside the command because Click's resolve_path
    # does not expand "~", unlike the catalog loader.
    type=click.Path(file_okay=False, path_type=Path),
    help="MCP state_dir containing runs/ and logs/.",
)
@click.option("--run-id", required=True, help="MCP run id to recover.")
@click.option(
    "--confirm",
    is_flag=True,
    help="Perform the reported cleanup and stored terminal-result actions.",
)
def mcp_recover_run(state_dir: Path, run_id: str, confirm: bool) -> None:
    """Recover cleanup, interrupted publication, or result finalization.

    :param Path state_dir: MCP state directory containing the run metadata.
    :param str run_id: Identifier of the run to recover.
    :param bool confirm: Whether to perform recovery instead of only reporting actions.
    :raises click.ClickException: If the host is not a supported MCP host, the state
        directory or run id is unknown, the launch outcome is still unresolved, no
        immutable terminal snapshot exists, an interrupted publication cannot
        be reconciled from its pointer, runner identity cannot rule out PID
        reuse, the runner or a trial lives in another PID namespace and this
        process cannot prove its processes gone, the runner still appears live, the run config snapshot is missing or
        does not match its recorded digest, no trial-level cleanup evidence can be
        confirmed, restored storage cannot account for published history, or study
        recovery fails with a ``RuntimeError``.
    """
    state_dir = state_dir.expanduser().resolve()
    try:
        require_supported_mcp_host()
    except CatalogError as exc:
        raise click.ClickException(_catalog_error_text(exc)) from None
    try:
        recover_run(state_dir, run_id, confirm=confirm, emit=click.echo)
    except RunRecoveryError as exc:
        raise click.ClickException(_operational_error_text(exc)) from None


@mcp.command(
    name="serve",
    context_settings=CONTEXT_SETTINGS,
    help="Serve the optional MCP server over stdio using an operator-authored catalog.",
    short_help="Serve the MCP server.",
)
@click.option(
    "--catalog",
    required=True,
    metavar="PATH",
    type=CONFIG_PATH,
    help="MCP catalog that maps agent-visible experiment ids to config files.",
)
@click.pass_context
def mcp_serve(ctx: click.Context, catalog: Path) -> None:
    """Serve the MCP server over stdio.

    :param click.Context ctx: Active Click context used to exit with the server return code.
    :param Path catalog: Operator-authored MCP catalog to load.
    """
    from phasesweep.mcp.server import serve

    ctx.exit(serve(catalog))


def _tool_arguments(ctx: click.Context, param: click.Parameter, value: str) -> JsonObject:
    """Parse the ``mcp call`` argument text as one strict JSON object.

    :param click.Context ctx: Active Click context, unused.
    :param click.Parameter param: Parameter being parsed, unused.
    :param str value: Argument text, ``{}`` when omitted.
    :raises click.BadParameter: If the text is not strict JSON or not an object.
    :return JsonObject: The decoded tool arguments.
    """
    del ctx, param
    try:
        return json_object(strict_json_loads(value))
    except ValueError as exc:
        raise click.BadParameter(f"expected a JSON object of tool arguments: {exc}") from exc


@mcp.command(
    name="call",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Run one MCP tool against a catalog in this process and print its structured result as "
        "JSON, through the same dispatch, validation, redaction and audit as `mcp serve`. "
        "JSON_ARGS is a JSON object of tool arguments and defaults to {}. Exit code 0 prints the "
        "result; 1 prints the tool's error on stderr; 2 means bad input or a catalog that does "
        "not load. The call is real: launch_run starts a detached run."
    ),
    short_help="Call one MCP tool from the shell.",
)
@click.argument("tool")
@click.argument("json_args", metavar="[JSON_ARGS]", default="{}", callback=_tool_arguments)
@click.option(
    "--catalog",
    required=True,
    metavar="PATH",
    type=CONFIG_PATH,
    help="MCP catalog that maps agent-visible experiment ids to config files.",
)
@click.pass_context
def mcp_call(ctx: click.Context, tool: str, json_args: JsonObject, catalog: Path) -> None:
    """Call one MCP tool without a client, for scripting or debugging.

    :param click.Context ctx: Active Click context used to exit with the call's return code.
    :param str tool: Tool name, such as ``list_experiments``.
    :param JsonObject json_args: Tool arguments, already parsed as a JSON object.
    :param Path catalog: Operator-authored MCP catalog to load.
    """
    from phasesweep.mcp.server import call

    ctx.exit(call(catalog, tool, json_args))


@mcp.command(
    name="check",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Validate an MCP catalog with the exact rules the server applies at startup and "
        "print a per-experiment ok/FAIL report. A successful check provisions and probes "
        "the configured state directory. Exit code 0 when every entry loads, 2 otherwise."
    ),
    short_help="Preflight an MCP catalog.",
)
@click.option(
    "--catalog",
    required=True,
    metavar="PATH",
    type=CONFIG_PATH,
    help="MCP catalog that maps agent-visible experiment ids to config files.",
)
@click.pass_context
def mcp_check(ctx: click.Context, catalog: Path) -> None:
    """Preflight an MCP catalog for the operator.

    Shares the per-entry validation code path with ``Registry.load`` but
    collects every entry's verdict instead of failing fast, so a broken
    catalog is diagnosed here rather than inside an MCP client restart. A
    successful check also provisions the state layout through the exact server
    startup path. Operator-facing: output may include paths.

    :param click.Context ctx: Active Click context used for the exit code.
    :param Path catalog: Operator-authored MCP catalog to validate.
    """
    try:
        report = check_catalog(catalog)
    except CatalogError as exc:
        click.echo(f"phasesweep mcp check: {_catalog_error_text(exc)}", err=True)
        ctx.exit(2)
    _echo_catalog_report(report)
    if not report.ok:
        ctx.exit(2)


def _echo_catalog_report(report: CatalogCheckReport) -> None:
    """Print a per-experiment ok/FAIL table for an MCP catalog check.

    Operator-facing: messages and suggestions may include paths.

    :param CatalogCheckReport report: Collected per-entry catalog verdicts.
    """
    width = max(len(entry.experiment_id) for entry in report.entries)
    for entry in report.entries:
        if entry.ok:
            actions = f"({', '.join(entry.actions)})" if entry.actions else "(read-only)"
            click.echo(f"{entry.experiment_id:<{width}}  ok    {actions}")
            continue
        message = (entry.error or "").removeprefix(f"{entry.experiment_id!r}: ")
        click.echo(f"{entry.experiment_id:<{width}}  FAIL  {message}")
        if entry.suggestion:
            click.echo(f"{'':<{width}}        fix: {entry.suggestion}")


@mcp.command(
    name="init-catalog",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Write an annotated MCP catalog for existing experiment configs: absolute "
        "state_dir under PHASESWEEP_HOME/mcp/<catalog-digest> or "
        "${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>, one read-only "
        "entry per --from config (visible_params: none, no allow block). The staged "
        "catalog is validated with the server startup path before it is published. "
        "Existing files are never overwritten; edit the result, then re-check it with "
        "`phasesweep mcp check`."
    ),
    short_help="Scaffold an MCP catalog.",
)
@click.option(
    "--from",
    "from_configs",
    multiple=True,
    required=True,
    metavar="PATH",
    type=CONFIG_PATH,
    help="Experiment config to catalog; repeat for more entries.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("catalog.yaml"),
    show_default=True,
    help="Catalog file to write; an existing file is never overwritten.",
)
@click.pass_context
def init_catalog(ctx: click.Context, from_configs: tuple[Path, ...], output: Path) -> None:
    """Scaffold an MCP catalog for the operator without overwriting a file.

    :param click.Context ctx: Active Click context used for the exit code.
    :param tuple[Path, ...] from_configs: Experiment configs to catalog.
    :param Path output: Catalog destination; must not already exist.
    """
    if not _write_catalog_scaffold(output, from_configs):
        ctx.exit(2)
    click.echo(
        "next: fill in each description, decide allow/visible_params, then run "
        f"`phasesweep mcp check --catalog {output.resolve()}`"
    )


def _write_catalog_scaffold(output: Path, from_configs: tuple[Path, ...]) -> bool:
    """Stage, validate, and exclusively publish a scaffolded catalog.

    :param Path output: Catalog destination.
    :param tuple[Path, ...] from_configs: Experiment configs to catalog.
    :return bool: True when the catalog was written; False after printing why not.
    """
    if output.is_symlink() or output.exists():
        click.echo(
            f"phasesweep mcp init-catalog: {output} already exists; refusing to overwrite. "
            "Pass -o to choose another name.",
            err=True,
        )
        return False

    try:
        text = scaffold_catalog_text(output, from_configs)

        def validate_scaffold(staged: Path) -> None:
            """Validate the staged catalog and identify its final origin."""
            registry = Registry.load(staged)
            private_atomic_write_text(registry.state_dir / "origin", str(output.resolve()) + "\n")

        if not atomic_create_text(output, text, validate=validate_scaffold):
            click.echo(
                f"phasesweep mcp init-catalog: {output} already exists; refusing to "
                "overwrite. Pass -o to choose another name.",
                err=True,
            )
            return False
    except CatalogError as exc:
        click.echo(f"phasesweep mcp init-catalog: {_catalog_error_text(exc)}", err=True)
        return False
    except OSError as exc:
        click.echo(f"phasesweep mcp init-catalog: cannot write {output}: {exc}", err=True)
        return False
    click.echo(f"wrote {output}")
    return True


if __name__ == "__main__":
    main()
