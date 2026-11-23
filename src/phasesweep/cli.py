"""phasesweep CLI."""

from __future__ import annotations

import contextlib
import errno
import io
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TextIO

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
    explain_command,
    first_sentence,
    lookup_or_suggest,
    pin_documentation_links,
    plain_path,
    reference_notes,
    rules_for,
    sub_key_count,
    top_level_entries,
)
from phasesweep.config.run_overrides import (
    SET_SYNTAX,
    CreatedKey,
    SetPathError,
    apply_set,
    created_key_notes,
)
from phasesweep.config.search import phase_draws
from phasesweep.engine.errors import WinnerIntegrityError
from phasesweep.engine.fingerprints import _recorded_config_matches
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.locking import experiment_is_busy
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_dir,
    _generation_summary_path,
    _generation_winner_path,
    _phase_dir,
    _run_log_path,
    _trials_csv_path,
)
from phasesweep.engine.preview import validate_trial_targets
from phasesweep.engine.provenance import (
    CodeRevision,
    _load_generation_code_revision,
    _load_generation_experiment,
)
from phasesweep.engine.publication import (
    raise_for_unusable_publication,
)
from phasesweep.engine.read import (
    _read_generation_selection,
    _represented_winners,
    experiment_status,
    phase_refusals,
    read_trial_tables,
    require_study,
    unpublished_winner_step,
)
from phasesweep.engine.run import TerminalReport, run_experiment
from phasesweep.engine.study_policy import _require_launch_target
from phasesweep.errors import (
    STORAGE_NULL_KEEPS_NOTHING,
    OperatorAction,
    PhaseSweepError,
    operator_step,
    phase_names,
)
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.registry import (
    CatalogCheckReport,
    Registry,
    check_catalog,
    require_supported_mcp_host,
)
from phasesweep.mcp.scaffold import scaffold_catalog_text
from phasesweep.mcp.tool_names import TOOL_NAMES
from phasesweep.reporting import _progress_values, report_objective
from phasesweep.run_report import (
    Banner,
    LogLine,
    NothingToRun,
    Summary,
    banner,
    nothing_to_run,
    summary,
)
from phasesweep.runtime.console import ConsoleHandler
from phasesweep.runtime.files import (
    PlatformCapabilityError,
    UnsafeLockPathError,
    atomic_create_text,
    private_atomic_write_text,
)
from phasesweep.runtime.gpu import GpuPool
from phasesweep.runtime.json import (
    JsonObject,
    json_object,
    strict_json_loads,
)
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    install_signal_handlers,
)
from phasesweep.runtime.terminal import (
    command_line,
    display_path,
    fit_lines,
    plural,
    terminal_width,
    use_color,
    wrap,
    wrap_labeled,
)
from phasesweep.trial_table import render_trial_tables, unsettled_running_comment

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
CONFIG_PATH = click.Path(exists=True, dir_okay=False, path_type=Path)
CONFIG_OVERRIDES = click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Override a config value with a YAML VALUE; repeatable. KEY is dot-separated "
        # The \b paragraph keeps the URL unwrapped, so a terminal never breaks it mid-word.
        f"(env.NOTE); select a phase as phases[NAME] or phases[0].\n\n\b\n{SET_SYNTAX}"
    ),
)
MCP_CATALOG = click.option(
    "--catalog",
    required=True,
    metavar="PATH",
    type=CONFIG_PATH,
    help="MCP catalog that maps agent-visible experiment ids to config files.",
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

#: The ``run.log`` of the run in progress, which :func:`main` names last on every exit.
_RUN_LOG: ContextVar[Path | None] = ContextVar("phasesweep_run_log", default=None)


@contextlib.contextmanager
def _console_logging(verbose: bool, *, dry_run: bool = False) -> Iterator[None]:
    """Show a run's report and warnings on stderr, and leave the rest to ``run.log``.

    The ``phasesweep`` logger takes ``INFO``, or ``DEBUG`` under ``verbose``, which is the level
    ``run.log`` records at; the console shows the same lines either way. A dry run also shows
    the package's INFO lines, which are part of the launch contract it prints. Optuna stays at
    ``ERROR`` in both modes. Everything is restored on exit, so a process that runs several
    commands, as the tests do, keeps no handler or level behind.

    :param bool verbose: Record ``DEBUG`` messages in ``run.log``.
    :param bool dry_run: Print the package's INFO lines on the console too.
    :return Iterator[None]: Context manager for the command's lifetime.
    """
    package = logging.getLogger("phasesweep")
    root = logging.getLogger()
    handler = ConsoleHandler(dry_run=dry_run)
    previous_level = package.level
    previous_verbosity = optuna.logging.get_verbosity()
    package.setLevel(logging.DEBUG if verbose else logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        optuna.logging.set_verbosity(previous_verbosity)
        package.setLevel(previous_level)


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Phase-chained hyperparameter sweeps driven by a YAML file.",
)
@click.version_option(package_name="phasesweep")
def cli() -> None:
    """Run the phasesweep command line interface."""


def _operational_error_text(exc: PhaseSweepError) -> str:
    """Format an operational diagnosis followed by its operator step.

    :param PhaseSweepError exc: Operational failure to render.
    :return str: Diagnostic followed by its operator step.
    """
    return f"{exc}\nfix: {operator_step(exc)}"


def _print_operational_error(exc: PhaseSweepError) -> None:
    """Write ``phasesweep: <diagnosis>`` and its ``fix:`` step to stderr.

    On a terminal each wraps at word boundaries to the width with its text hanging under the
    label. Redirected, each stays one logical line: scripts and agents match on these sentences,
    and a refusal may have no ``run.log`` copy to read them from. A line the step sets off after
    its sentence is a command to paste, so it prints whole and flush left either way.

    :param PhaseSweepError exc: Operational failure to report.
    """
    stream = sys.stderr
    if not stream.isatty():
        click.echo(f"phasesweep: {_operational_error_text(exc)}", err=True)
        return
    width = terminal_width(stream)
    step, *commands = operator_step(exc).split("\n")
    lines = [
        *wrap_labeled("phasesweep:", str(exc), width),
        *wrap_labeled("fix:", step, width),
        *commands,
    ]
    click.echo("\n".join(lines), err=True)


class _StandardStreamFile(io.FileIO):
    """A standard stream's descriptor that discards what can no longer be written.

    Text for a gone reader is dropped as if written, so the command carries on, no text stays
    buffered for interpreter shutdown to fail on (which would turn any exit status into 120),
    and Click's own handling of a broken pipe, which ends a command with status 1, is never
    reached. Any other write error, such as a full disk, is raised once, so the command ends
    with it as an operational error and status 1; the stream then drops what it is given, so
    the flush at interpreter exit neither reports the lost text again nor changes that status.
    """

    #: Write errors that mean nobody can read the stream any more: a pipe whose reader exited
    #: (``EPIPE``) and a terminal that hung up, as a dropped SSH session's does (``EIO``). A
    #: class attribute, because a stream is flushed as it is destroyed at interpreter exit,
    #: after module globals are cleared.
    _READER_GONE = frozenset({errno.EPIPE, errno.EIO})

    #: Set once a write error other than a gone reader has been raised.
    _failed: bool = False

    def write(self, data: ReadableBuffer, /) -> int:
        """Write ``data``, or drop it when the reader is gone or an earlier write failed.

        :param ReadableBuffer data: Encoded text from the stream's buffer.
        :raises OSError: The first write failure other than a gone reader, such as a full disk.
        :return int: The number of bytes the descriptor took, all of them when dropped.
        """
        if self._failed:
            return memoryview(data).nbytes
        try:
            return super().write(data)
        except OSError as exc:
            if exc.errno in self._READER_GONE:
                return memoryview(data).nbytes
            self._failed = True
            raise


def _discard_unreadable_output(stream: TextIO | None) -> TextIO:
    """Rebuild a standard stream without letting an absent reader interrupt the command.

    :param TextIO | None stream: ``sys.stdout`` or ``sys.stderr``.
    :return TextIO: A sink if Python started without the stream, the stream itself for captures
        without a descriptor, or a rebuilt stream that preserves its text settings.
    """
    if stream is None:
        return Path(os.devnull).open("w", encoding="utf-8", errors="backslashreplace")
    if not isinstance(stream, io.TextIOWrapper):
        return stream
    try:
        descriptor = stream.fileno()
    except OSError:
        return stream
    stream.flush()
    return io.TextIOWrapper(
        io.BufferedWriter(_StandardStreamFile(descriptor, "w", closefd=False)),
        encoding=stream.encoding,
        errors=stream.errors,
        line_buffering=stream.line_buffering,
        write_through=stream.write_through,
    )


def main() -> None:
    """Run the CLI behind a process-level error boundary, then name the run's log.

    Click exceptions reach this boundary; explicit ``SystemExit`` statuses pass through. A
    ``run`` that went past its config prints ``log: <path>`` last, after its error or resume
    lines, whenever ``run.log`` exists, so the file is named however the command ended. A path
    wider than the terminal breaks after slashes. Output to a closed stream, a pipe whose reader
    exited or a terminal that hung up is discarded, so the status stays the command's own.

    :raises SystemExit: Always, carrying one of ``_USAGE_EXIT`` (bad input),
        ``_FAILURE_EXIT`` (the run failed), ``_INTERNAL_EXIT`` (a phasesweep
        bug), ``_ABORT_EXIT`` (interrupted), or the status the command or
        Click chose.
    """
    _RUN_LOG.set(None)
    sys.stdout = _discard_unreadable_output(sys.stdout)
    sys.stderr = _discard_unreadable_output(sys.stderr)
    try:
        _error_boundary()
    finally:
        run_log = _RUN_LOG.get()
        if run_log is not None and _is_readable_file(run_log):
            _print_block(LogLine(str(run_log.resolve())))


def _is_readable_file(path: Path) -> bool:
    """Tell whether ``path`` is an existing file, treating an unreachable one as absent.

    The exit path runs after a failure that may be an inaccessible directory, so asking about
    the file must not raise a second error over the first.

    :param Path path: The file to look for.
    :return bool: ``True`` when it exists as a file and can be examined.
    """
    try:
        return path.is_file()
    except OSError:
        return False


def _error_boundary() -> None:
    """Run the CLI and turn each kind of failure into its message and exit status.

    :raises SystemExit: Always, with the status :func:`main` documents.
    """
    try:
        # Click would take the program name from sys.argv[0], which is not "phasesweep"
        # when the CLI is launched as a module or from a test runner.
        status = cli.main(prog_name="phasesweep", standalone_mode=False)
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
        _print_operational_error(exc)
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


def _refuse_init_overwrite(target: Path) -> NoReturn:
    """Print that ``init`` will not replace ``target`` and exit with the usage status.

    :param Path target: Destination that already exists, or appeared during publication.
    :raises click.exceptions.Exit: Always, with code 2.
    """
    click.echo(f"phasesweep init: refusing to overwrite existing path {target}", err=True)
    raise click.exceptions.Exit(2)


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
    help="Template to write. Local templates use the packaged toy trainer; mlflow needs "
    "a tracking server and an existing experiment.",
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
        _refuse_init_overwrite(target)
    text = _template_text(template)
    try:
        if not atomic_create_text(target, text):
            _refuse_init_overwrite(target)
    except OSError as exc:
        # `phasesweep init` is the first command a new user runs; an unwritable
        # directory or a filesystem without hard links must report one line, not
        # a traceback.
        click.echo(f"phasesweep init: cannot write {target}: {exc}", err=True)
        raise click.exceptions.Exit(1) from None

    # Name the expanded path, never the raw option value: shell quoting
    # suppresses "~" expansion, so a quoted raw "~/x.yaml" would name a
    # different (nonexistent) file than the one just written.
    config_arg = str(expanded)
    click.echo(f"Wrote the {template} template to {target}")
    click.echo("\nNext: read the template header for prerequisites and trainer adaptation, then:")
    click.echo(f"  {command_line('validate', config_arg)}")
    click.echo(f"  {command_line('run', config_arg, '--dry-run')}")
    click.echo(f"  {command_line('run', config_arg)}")
    click.echo(f"  {command_line('status', config_arg, '--trials')}")
    click.echo(f"  {command_line('mcp', 'init-catalog', '--from', config_arg)}")
    click.echo(
        "\nphasesweep explain lists the top-level config keys; phasesweep explain KEY describes one."
    )
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
        "Run configured phases. --from-phase skips earlier phases and needs only the "
        "winners the remaining phases inherit. The full log is run.log in the experiment's "
        "directory."
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
    help="Record DEBUG messages in run.log; the console shows the same lines either way.",
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
    :param bool verbose: Record ``DEBUG`` messages in ``run.log``; the console is unchanged.
    :raises ConfigError: The config file cannot be parsed or fails model validation.
    :raises click.BadParameter: ``from_phase`` names no configured phase; the usage error exits
        with status 2.
    :raises PhaseSweepShutdown: An interrupted run is re-raised unchanged.
    """
    with _console_logging(verbose, dry_run=dry_run):
        # Install before config parsing so CLI SIGINT/SIGTERM behavior and exit
        # codes remain structured throughout startup, including dry-run previews.
        # run_experiment() repeats this idempotently for direct library callers.
        install_signal_handlers()
        config = _command_config(config_path, settings=settings)
        if from_phase is not None:
            valid = [p.name for p in config.phases]
            if from_phase not in valid:
                raise click.BadParameter(
                    f"{from_phase!r} is not a phase of this config; its phases are "
                    f"{phase_names(valid)}.",
                    param_hint="--from-phase",
                )
        run_log = _run_log_path(config)
        if not dry_run:
            _RUN_LOG.set(run_log)
            _print_block(
                banner(
                    config,
                    config_path=display_path(config_path),
                    workdir=display_path(_experiment_dir(config)),
                    log_path=str(run_log.resolve()),
                    from_phase=from_phase,
                )
            )
        finished: list[TerminalReport] = []
        wall_start, started = time.time(), time.monotonic()
        try:
            run_experiment(
                config, from_phase=from_phase, dry_run=dry_run, terminal_callback=finished.append
            )
        except PhaseSweepShutdown as exc:
            if exc.published_result_committed:
                click.echo("phasesweep: stopped after the result was published.", err=True)
            elif not dry_run:
                click.echo(
                    _resume_hint(
                        config,
                        command=_rerun_command(config_path, settings, from_phase),
                        width=terminal_width(sys.stderr),
                    ),
                    err=True,
                )
            raise
        else:
            if not dry_run and finished:
                _print_block(
                    _closing_block(
                        config,
                        finished[-1],
                        elapsed=time.monotonic() - started,
                        wall_start=wall_start,
                        config_path=config_path,
                        settings=settings,
                        from_phase=from_phase,
                    )
                )
        finally:
            # A terminal report means the engine logged its closing lines, which count what
            # stays pending, however the run ended; a run refused before that, as by a live
            # run's lock, gets no step to send records it may not own.
            if not dry_run and finished:
                _echo_sync_step(config, config_path, settings)


def _print_block(block: Banner | Summary | NothingToRun | LogLine) -> None:
    """Write a block the run prints but does not log to stderr, fitted to its width and colour.

    :param Banner | Summary | NothingToRun | LogLine block: The block to draw.
    """
    stream = sys.stderr
    lines = block.render(terminal_width(stream), color=use_color(stream))
    click.echo("\n".join(lines), err=True)


def _closing_block(
    config: Experiment,
    outcome: TerminalReport,
    *,
    elapsed: float,
    wall_start: float,
    config_path: Path,
    settings: tuple[str, ...],
    from_phase: str | None,
) -> Summary | NothingToRun:
    """Build what a successful run closes with: its summary, or that it had nothing to run.

    :param Experiment config: The config that ran.
    :param TerminalReport outcome: The engine's report of the finished invocation.
    :param float elapsed: Seconds the engine took.
    :param float wall_start: ``time.time()`` when the engine started, to tell the files this
        run wrote from those it found.
    :param Path config_path: The config path as typed.
    :param tuple[str, ...] settings: The ``--set`` assignments as typed.
    :param str | None from_phase: The ``--from-phase`` value, if given.
    :return Summary | NothingToRun: The block to print.
    """
    next_command = command_line("status", *_config_arguments(config_path, settings), "--trials")
    if outcome.replayed_generation_id is not None:
        return nothing_to_run(
            elapsed=elapsed,
            counts=outcome.phase_counts,
            generation_id=outcome.replayed_generation_id,
            next_command=next_command,
        )
    root = _experiment_dir(config)
    candidates = [
        _generation_summary_path(config, outcome.generation_id),
        *(_trials_csv_path(config, phase.name) for phase in config.phases),
    ]
    written = [
        path.relative_to(root).as_posix()
        for path in candidates
        if path.is_relative_to(root) and path.is_file() and path.stat().st_mtime >= wall_start
    ]
    return summary(
        config,
        elapsed=elapsed,
        counts=outcome.phase_counts,
        winners=outcome.winners or {},
        from_phase=from_phase,
        written_root=display_path(root),
        written=written,
        next_command=next_command,
        show_winners_command=command_line(
            "show-winners", *_config_arguments(config_path, settings)
        ),
    )


def _echo_sync_step(config: Experiment, config_path: Path, settings: tuple[str, ...]) -> None:
    """Print the ``sync-tracking`` command as typed when a run leaves MLflow work pending.

    The engine's closing warning says in words what that command needs, as it also goes to
    ``run.log``; this line gives the command itself, whole so it pastes. Nothing is printed when
    nothing is pending or the count fails, which the engine's closing lines already report.

    :param Experiment config: The config the run was started with.
    :param Path config_path: The config path as typed.
    :param tuple[str, ...] settings: The ``--set`` assignments as typed.
    """
    if config.mlflow is None:
        return
    from phasesweep.tracking.publication import delivery_status

    try:
        pending = delivery_status(config)["pending"]
    except (OSError, ValueError):
        return
    if pending:
        command = command_line("sync-tracking", *_config_arguments(config_path, settings))
        click.echo(f"MLflow: to send what is pending without training, run:\n{command}", err=True)


def _config_arguments(config_path: Path, settings: tuple[str, ...]) -> list[str]:
    """List the arguments that name an experiment's config on a command line, as typed.

    :param Path config_path: The config path as typed.
    :param tuple[str, ...] settings: The ``--set`` assignments as typed.
    :return list[str]: The path, then each ``--set`` and its assignment, ready for :func:`command_line`.
    """
    return [str(config_path), *(part for item in settings for part in ("--set", item))]


def _rerun_command(config_path: Path, settings: tuple[str, ...], from_phase: str | None) -> str:
    """Build the shell-quoted ``phasesweep run`` command that repeats an invocation.

    Only what decides which trials run is repeated; ``--verbose`` changes the log alone.

    :param Path config_path: The config path as typed.
    :param tuple[str, ...] settings: The ``--set`` assignments as typed.
    :param str | None from_phase: The ``--from-phase`` value, if given.
    :return str: The command, quoted so it can be pasted into a shell.
    """
    return command_line(
        "run",
        *_config_arguments(config_path, settings),
        *(() if from_phase is None else ("--from-phase", from_phase)),
    )


def _resume_hint(config: Experiment, *, command: str, width: int) -> str:
    """Name the next step after a run was stopped by SIGINT, SIGHUP or SIGTERM.

    :param Experiment config: The config the stopped run was started with.
    :param str command: The shell-quoted command that repeats the run.
    :param int width: Terminal columns the sentence wraps to; the command stays on one line.
    :return str: The sentence, then the command on a line of its own, printed last so the
        cleanup log above it is not the final word.
    """
    if config.persistent:
        sentence = "phasesweep: stopped. Rerun the same command to resume."
    else:
        sentence = (
            "phasesweep: stopped. Rerun the same command to start over: "
            f"{STORAGE_NULL_KEEPS_NOTHING}."
        )
    return "\n".join([*wrap(sentence, width), command])


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
    :raises LaunchEnvironmentError: ``execution.cwd`` is not a directory on this host, or the
        trial command's program is written with a ``/`` and is not an executable file. A program
        found through ``PATH`` is left to a launch, which judges it on the PATH it starts with.
    :raises TrialTargetRefusedError: A fixed-configuration or grid target exceeds its
        recorded retry ceiling.
    :raises StudyStorageUnavailableError: The ledger cannot be read to check a raised target.
    """
    config = _command_config(config_path, settings=settings)
    _require_launch_target(config)
    validate_trial_targets(config)
    for phase in config.phases:
        GpuPool.for_phase(phase, cuda_visible_devices=config.env.get("CUDA_VISIBLE_DEVICES"))
    click.echo(f"OK: {config.experiment} ({plural(len(config.phases), 'phase')})")
    _render_experiment_phases(config)
    # Validation never compares the config with a study's recorded identity or the environment
    # of its recorded trials, and it leaves a trial program found through PATH to the launch's
    # PATH, so an edit that a recorded study refuses and a program that is not installed both
    # pass it. The ledger is read only for some targets, so this cannot tell whether a study is
    # recorded and says it every time.
    sentence = (
        "Checked the config alone. A dry run also compares it with the recorded study and "
        "its trials' environment, and checks the trial program on PATH:"
    )
    dry_run = command_line("run", *_config_arguments(config_path, settings), "--dry-run")
    click.echo("\n".join([*wrap(sentence, terminal_width(sys.stdout)), dry_run]))


def _render_experiment_phases(experiment: Experiment, *, indent: str = "  ") -> None:
    """Render phase summaries and sampler capability lines for ``validate``.

    Each contract line wraps to the terminal under its own prefix, later lines hanging two more
    columns in, and each warning wraps under its ``warning:`` label.

    :param Experiment experiment: Experiment whose phases should be printed.
    :param str indent: Prefix to place before each rendered phase line.
    """
    width = terminal_width(sys.stdout)
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
            if warning:
                shown = wrap_labeled("warning:", line, width, indent=prefix)
            else:
                shown = [
                    f"{prefix}{part}" for part in fit_lines(line, width - len(prefix), hanging="  ")
                ]
            click.echo("\n".join(shown))
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
        "keys; with a KEY such as 'phases[].sampler' (quote brackets), print it and every "
        "key below it with the rules that govern it. --notes prints the general notes on "
        "the config file, paths, YAML typing and the study fingerprint."
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
        miss = f"phasesweep explain: no config key matches {key!r}."
        if near:
            # Each suggestion is a whole command on a line of its own, so it can be pasted.
            click.echo(
                "\n".join([f"{miss} Closest:", *(explain_command(path) for path in near)]), err=True
            )
        else:
            click.echo(f"{miss} Run `phasesweep explain` to list the top-level keys.", err=True)
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
        suffix = f" ({plural(count, 'sub-key')})" if count else ""
        summary = first_sentence(entry, _SUMMARY_WIDTH)
        click.echo(f"  {entry.path:<{width}}  {entry.signature:<{sig_width}}  {summary}{suffix}")
    click.echo()
    click.echo("Narrow with a key, for example: phasesweep explain 'phases[].sampler'")
    click.echo(
        "General notes on the config file, paths, YAML typing and the study fingerprint: "
        "phasesweep explain --notes"
    )


def _code_revision_text(revision: CodeRevision | None) -> str:
    """Render a generation's recorded trainer revision for a ``show-winners`` comment line.

    :param CodeRevision | None revision: The recorded revision, ``None`` when none was readable.
    :return str: The commit, marked ``(dirty)`` when tracked files differed, or ``unknown``.
    """
    if revision is None:
        return "unknown (not a git work tree or git unavailable)"
    return f"{revision['commit']}{' (dirty)' if revision['dirty'] else ''}"


@cli.command(
    name="show-winners",
    context_settings=CONTEXT_SETTINGS,
    help="Print concise phase winners, including unpublished results from the latest run.",
    short_help="Print saved phase winners.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@CONFIG_OVERRIDES
def show_winners(config_path: Path, settings: tuple[str, ...]) -> None:
    """Print concise winners from published or unfinished generation records.

    :param Path config_path: Config used to locate the publication and label drift as historical.
    :param tuple[str, ...] settings: The same config overrides used by the run.
    :raises PublicationAccessError: This user cannot validate the publication.
    :raises PublicationIntegrityError: The publication is corrupt; nothing is rendered.
    :raises WinnerIntegrityError: An unpublished winner's producing snapshot is unavailable or invalid.
    """
    experiment = _command_config(config_path, settings=settings)
    require_study(experiment)
    ledger = validate_ledger(experiment)
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
        ledger=ledger,
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
        click.echo(f"# publication code revision: {_code_revision_text(validated.code_revision)}")
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
        click.echo(
            yaml.dump({"trainer_config": trainer_base}, Dumper=_WinnerDumper, sort_keys=False)
        )

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
            carried = ""
            if winner.source is not None and winner.source.generation_id != winner.generation_id:
                try:
                    source_text = _code_revision_text(
                        _load_generation_code_revision(
                            _experiment_dir(experiment), winner.source.generation_id
                        )
                    )
                except WinnerIntegrityError:
                    # Only this display line needs the source generation's record, so its
                    # loss must not cut the listing short.
                    source_text = "unreadable (its reproducibility.json cannot be read)"
                carried = (
                    f"; winner trial from generation {winner.source.generation_id}, "
                    f"code revision {source_text}"
                )
            click.echo(
                f"# winner generation: {winner.generation_id} "
                f"({'published' if winner.is_published else 'unpublished'}){carried}"
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
                "cost_advisory_status": winner.cost_advisory_status,
                "tie_groups": (
                    None
                    if winner.tie_groups is None
                    else [json_object(dict(group)) for group in winner.tie_groups]
                ),
                "edge_advisory": winner.edge_advisory,
                "edge_advisory_status": winner.edge_advisory_status,
            }
            if winner.repeats_detail is not None:
                detail = winner.repeats_detail
                if isinstance(payload["repeats"], dict):
                    payload["repeats"]["key"] = detail["key"]
                    payload["repeats"]["members"] = [
                        {
                            "trial_number": member["trial_number"],
                            "repeat": member["repeat"],
                            "trial_dir": str(_phase_dir(experiment, name) / member["trial_dir"]),
                        }
                        for member in detail["members"]
                    ]
            click.echo(yaml.dump(payload, Dumper=_WinnerDumper, sort_keys=False).rstrip())
        else:
            click.echo(f"=== {name} === (no winner yet)")
            _render_phase_comment(comment, prefix="# ")


def _display_float(value: float) -> str:
    """Format a displayed float to twelve significant digits without changing storage.

    :param float value: Recorded value to display.
    :return str: Compact decimal or scientific notation.
    """
    return f"{value:.12g}"


class _WinnerDumper(yaml.SafeDumper):
    """Render winner floats with the shared display precision."""

    def represent_float(self, value: float) -> yaml.ScalarNode:
        """Apply display precision while preserving YAML's float type.

        :param float value: Stored value, left unchanged outside the display.
        :return yaml.ScalarNode: Safe YAML float scalar.
        """
        return super().represent_float(float(_display_float(value)))


_WinnerDumper.add_representer(float, _WinnerDumper.represent_float)


def _status_can_suggest_resume(config: Experiment) -> bool:
    """Check whether status can confirm the experiment is idle before suggesting a resume.

    A lock probe that cannot run leaves the trial data and publication verdict readable.

    :param Experiment config: Experiment whose run locks are probed.
    :return bool: Whether the lock probe found no active run.
    """
    try:
        return not experiment_is_busy(config)
    except (OSError, UnsafeLockPathError, PlatformCapabilityError):
        return False


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Print read-only trial counts and phase state for a phasesweep experiment, as YAML; "
        "--trials prints a table of trials per phase for reading instead."
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
        "Print one table per phase instead of YAML: a column per swept parameter, rows sorted "
        "by value, the winner marked, repeat groups below. trials.csv in each phase's "
        "directory is the machine-readable form."
    ),
)
def status(config_path: Path, show_trials: bool, settings: tuple[str, ...]) -> None:
    """Print read-only run status for ``config_path``.

    The payload is printed first and in full even when it reports a corrupt
    publication: the operator needs the trial counts and generation identity
    to decide what to inspect, and the boundary's one-line diagnostic follows
    on stderr. A phase whose recorded abort or failure streak would refuse a run gets a
    comment with that refusal and its fix, active run or not. Otherwise an unpublished winner
    gets a comment explaining how to resume publication, and trials recorded RUNNING while no
    run holds the experiment's locks get a comment saying that a rerun settles them; both
    require a successful idle lock probe. ``--trials`` prints the trial tables in its place,
    from one publication resolution, with each refusal under its phase's heading, styled only
    on a terminal, and exits the same way.

    :param Path config_path: Experiment YAML file to inspect.
    :param bool show_trials: Print per-phase trial tables instead of the YAML status.
    :param tuple[str, ...] settings: The same config overrides used by the run.
    :raises PublicationAccessError: The reported publication cannot be
        validated as the current user.
    :raises PublicationIntegrityError: The reported publication no longer validates.
    """
    config = _command_config(config_path, settings=settings)
    require_study(config)
    if show_trials:
        tables = read_trial_tables(config, latest=True)
        width = terminal_width(sys.stdout)
        running = sum(row["state"] == "RUNNING" for table in tables.phases for row in table.rows)
        if running and _status_can_suggest_resume(config):
            click.echo(unsettled_running_comment(running, width))
        click.echo(
            render_trial_tables(
                tables,
                config,
                csv_root=None if config.persistent else _experiment_dir(config),
                width=width,
                color=use_color(sys.stdout),
            )
        )
        raise_for_unusable_publication(
            tables.experiment,
            tables.publication_integrity,
            tables.publication_error,
            _experiment_dir(config),
        )
        return
    ledger = validate_ledger(config)
    payload = experiment_status(config, ledger=ledger)
    phases = payload.get("phases")
    phase_payloads = (
        [phase for phase in phases if isinstance(phase, dict)] if isinstance(phases, list) else []
    )
    running = 0
    for phase in phase_payloads:
        counts = phase.get("trials")
        recorded = counts.get("RUNNING") if isinstance(counts, dict) else None
        if isinstance(recorded, int):
            running += recorded
    retained = any(
        phase.get("winner") is not None and phase.get("winner_is_published") is False
        for phase in phase_payloads
    )
    # One probe serves both comments: flock cannot test a lock without taking it.
    suggest_resume = _status_can_suggest_resume(config) if running or retained else False
    if running and suggest_resume:
        click.echo(unsettled_running_comment(running))
    # A recorded refusal needs no idle probe: it holds whether or not a run is active.
    refusals = phase_refusals(config, ledger)
    for refusal in refusals.values():
        for line in _operational_error_text(refusal).splitlines():
            click.echo(f"# {line}")
    if retained and suggest_resume and not refusals:
        step = unpublished_winner_step(config, ledger)
        if step is not None:
            click.echo(
                "# Winner retained but not published"
                f"{'.' if step.cause is None else f'; {step.cause}'}"
            )
            click.echo(f"# fix: {step.fix}")
    click.echo(yaml.dump(payload, Dumper=_WinnerDumper, sort_keys=False).rstrip())
    raise_for_unusable_publication(
        str(payload["experiment"]),
        str(payload["publication_integrity"]),
        None if payload.get("publication_error") is None else str(payload["publication_error"]),
        Path(str(payload["workdir"])),
    )


@cli.command(
    "sync-tracking",
    context_settings=CONTEXT_SETTINGS,
    help="Reconcile MLflow run creations and retry uploads without training or selecting winners.",
    short_help="Reconcile and retry MLflow uploads.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@CONFIG_OVERRIDES
@click.option(
    "--generation",
    "generation_id",
    default=None,
    help="Limit reconciliation and uploads to this generation.",
)
def sync_tracking(config_path: Path, generation_id: str | None, settings: tuple[str, ...]) -> None:
    """Reconcile pending native MLflow runs and retry result delivery.

    Warnings print on the console as a run's do. Every step that ends with something pending
    names this command as typed, on a line of its own; a shutdown signal prints how much is
    still pending before the command exits with the signal's status.

    :param Path config_path: Matching experiment configuration.
    :param str | None generation_id: Generation to synchronize, or ``None`` for all.
    :param tuple[str, ...] settings: The same config overrides used by the run.
    :raises ClickException: The generation name or a delivery record is malformed.
    :raises PhaseSweepError: Nothing matching can be synchronized, or records remain pending;
        its step is a retry when MLflow gave no answer or a creation's window is still open,
        and fixing the causes when it refused.
    """
    from phasesweep.tracking.publication import sync_tracking as sync
    from phasesweep.tracking.session import (
        DeliveryTally,
        count_text,
        creation_window_end,
        pending_creation_records,
        refused_text,
    )

    command = command_line(
        "sync-tracking",
        *_config_arguments(config_path, settings),
        *(() if generation_id is None else ("--generation", generation_id)),
    )
    tally = DeliveryTally()
    with _console_logging(verbose=False):
        config = _command_config(config_path, settings=settings)
        require_study(config)
        # Every creation search starts after this, so a window open now was open at its search.
        started = time.time()
        try:
            counts = sync(config, generation_id, tally=tally)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        except PhaseSweepShutdown:
            click.echo(_sync_stopped_text(config, generation_id, command), err=True)
            raise
    click.echo(yaml.safe_dump(counts, sort_keys=False).rstrip())
    if not counts["pending"]:
        return
    again = f"run this command again:\n{command}"
    if tally.unanswered is not None:
        raise PhaseSweepError(
            f"MLflow delivery remains pending: {tally.unanswered}",
            action=OperatorAction.RETRY,
            remediation=f"Once the server answers, {again}",
        )
    if tally.refused:
        raise PhaseSweepError(
            "MLflow delivery remains pending:" + refused_text(tally.refused),
            action=OperatorAction.FIX_CONFIG,
            remediation=(
                f"Fix {'each cause' if len(tally.refused) > 1 else 'the cause'} in MLflow, "
                f"then {again}"
            ),
        )
    creations = pending_creation_records(config, generation_id)
    open_windows = [
        end for end in (creation_window_end(config, path) for path in creations) if end > started
    ]
    if open_windows:
        assert config.mlflow is not None
        # MLflow answered that no run carries the identity yet, and the lost request may still
        # make one, so only a search after the window can settle the creation. The time is
        # rounded up, so a retry at the printed second is past the window.
        until = math.ceil(max(open_windows))
        left = math.ceil(until - time.time())
        raise PhaseSweepError(
            f"MLflow delivery remains pending: no run yet for {count_text(0, len(open_windows))}, "
            "which may still be created until timeout_seconds "
            f"({config.mlflow.timeout_seconds:g} s) after the request was sent.",
            action=OperatorAction.RETRY,
            remediation=(
                f"Wait until {time.strftime('%H:%M:%S', time.localtime(until))}, about {left} s "
                f"from now, then {again}"
                if left > 0
                else f"That window has closed since the search, so {again}"
            ),
        )
    # The counts printed above, less the creations the window check just listed, so the line
    # agrees with them without reading the records again.
    records = counts["pending"] - len(creations)
    raise PhaseSweepError(
        f"MLflow delivery remains pending: {count_text(records, len(creations))} could not be "
        "sent or settled; the warnings above give each cause.",
        action=OperatorAction.RETRY,
        remediation=f"Once those causes are resolved, {again}",
    )


def _sync_stopped_text(config: Experiment, generation_id: str | None, command: str) -> str:
    """Say what a shutdown left pending in sync-tracking's scope, and how to send it.

    :param Experiment config: The config sync-tracking ran with.
    :param str | None generation_id: Its ``--generation`` value, if given.
    :param str command: The shell-quoted command that repeats it.
    :return str: The sentence, wrapped to the terminal, then the command on a line of its own.
    """
    from phasesweep.tracking.publication import delivery_counts
    from phasesweep.tracking.session import count_text

    try:
        counts = delivery_counts(config, generation_id)
    except (OSError, ValueError) as exc:
        stopped = f"phasesweep: stopped; what stays pending could not be counted ({exc})."
    else:
        if not counts.pending:
            return "phasesweep: stopped with nothing pending."
        stopped = (
            f"phasesweep: stopped with {count_text(counts.records, counts.creations)} pending."
        )
    sentence = f"{stopped} Rerun the same command to send what is left:"
    return "\n".join([*wrap(sentence, terminal_width(sys.stderr)), command])


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
        "Recover an MCP run after uncertain cleanup, an interrupted publication or an "
        "unfinished result snapshot. Without --confirm it only reports the actions; "
        "--confirm performs them."
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

    A shutdown signal during a confirmed pass stops its workers and prints the command that
    finishes the recovery, then the command exits with the signal's status.

    :param Path state_dir: MCP state directory containing the run metadata.
    :param str run_id: Identifier of the run to recover.
    :param bool confirm: Whether to perform recovery instead of only reporting actions.
    :raises PhaseSweepShutdown: A shutdown signal stopped a confirmed pass.
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
    # The engine's warnings, such as a record MLflow refused, print on the console as
    # sync-tracking's do; recovery writes no run.log of its own.
    with _console_logging(verbose=False):
        try:
            recover_run(state_dir, run_id, confirm=confirm, emit=click.echo)
        except RunRecoveryError as exc:
            _print_operational_error(exc)
            raise click.exceptions.Exit(_FAILURE_EXIT) from None
        except PhaseSweepShutdown:
            # A stopped pass leaves the run recovery-required unless it had already cleared it,
            # and either way the same command finishes or confirms the recovery.
            command = command_line(
                "mcp", "recover-run", "--state-dir", str(state_dir), "--run-id", run_id, "--confirm"
            )
            sentence = "phasesweep: stopped. Rerun the same command to finish the recovery:"
            click.echo("\n".join([*wrap(sentence, terminal_width(sys.stderr)), command]), err=True)
            raise


@mcp.command(
    name="serve",
    context_settings=CONTEXT_SETTINGS,
    help="Serve the optional MCP server over stdio using an operator-authored catalog.",
    short_help="Serve the MCP server.",
)
@MCP_CATALOG
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
        "Call one MCP tool against a catalog in this process and print its result (JSON, or "
        "get_guide's text) through the path `mcp serve` uses. "
        f"TOOL is one of {', '.join(TOOL_NAMES)}. "
        "JSON_ARGS is a JSON object of tool arguments, default {}. Exit status: 0 result, "
        "1 tool error on stderr, 2 bad input or an unloadable catalog. "
        "The call is real: launch_run starts a detached run."
    ),
    short_help="Call one MCP tool from the shell.",
)
@click.argument("tool", type=click.Choice(TOOL_NAMES), metavar="TOOL")
@click.argument("json_args", metavar="[JSON_ARGS]", default="{}", callback=_tool_arguments)
@MCP_CATALOG
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
        "Check an MCP catalog: print an ok/FAIL line per experiment, then provision and probe "
        "the state directory and report run records that block launches. Exit status 0 when "
        "all pass, 2 otherwise."
    ),
    short_help="Preflight an MCP catalog.",
)
@MCP_CATALOG
@click.pass_context
def mcp_check(ctx: click.Context, catalog: Path) -> None:
    """Preflight an MCP catalog for the operator.

    Shares the per-entry validation code path with ``Registry.load`` but
    collects every entry's verdict instead of failing fast, so a broken
    catalog is diagnosed here rather than inside an MCP client restart. If every
    entry passes, provision the state layout as the server does and report run
    records that block launches. Operator-facing: output may include paths.

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

    Operator-facing: messages and suggestions may include paths. Each ok line ends with the
    directory that entry's trainer starts in: its ``execution.cwd``, or the entry's ``cwd``
    when that is unset.

    :param CatalogCheckReport report: Collected per-entry catalog verdicts.
    """
    width = max(len(entry.experiment_id) for entry in report.entries)
    for entry in report.entries:
        if entry.ok:
            actions = f"({', '.join(entry.actions)})" if entry.actions else "(read-only)"
            click.echo(
                f"{entry.experiment_id:<{width}}  ok    {actions}  trainer cwd {entry.trainer_cwd}"
            )
            continue
        message = (entry.error or "").removeprefix(f"{entry.experiment_id!r}: ")
        click.echo(f"{entry.experiment_id:<{width}}  FAIL  {message}")
        if entry.suggestion:
            click.echo(f"{'':<{width}}        fix: {entry.suggestion}")


@mcp.command(
    name="init-catalog",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Write an annotated MCP catalog with one read-only entry per --from config "
        "(visible_params: none, no allow block) and an absolute state_dir under "
        "PHASESWEEP_HOME/mcp/<catalog-digest> or "
        "${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>. Existing files "
        "are never overwritten. Edit it, then run `phasesweep mcp check`."
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
    # Show where each trainer will start before the operator grants anything.
    _echo_catalog_report(check_catalog(output))
    click.echo(
        "next: fill in each description, decide allow/visible_params, then run "
        f"`{command_line('mcp', 'check', '--catalog', str(output.resolve()))}`"
    )


def _refuse_catalog_overwrite(output: Path) -> bool:
    """Print that ``init-catalog`` will not replace ``output``.

    :param Path output: Catalog destination that exists, or appeared during publication.
    :return bool: Always False, the scaffold writer's "not written" result.
    """
    click.echo(
        f"phasesweep mcp init-catalog: {output} already exists; refusing to overwrite. "
        "Pass -o to choose another name.",
        err=True,
    )
    return False


def _write_catalog_scaffold(output: Path, from_configs: tuple[Path, ...]) -> bool:
    """Stage, validate, and exclusively publish a scaffolded catalog.

    :param Path output: Catalog destination.
    :param tuple[Path, ...] from_configs: Experiment configs to catalog.
    :return bool: True when the catalog was written; False after printing why not.
    """
    if output.is_symlink() or output.exists():
        return _refuse_catalog_overwrite(output)

    try:
        text = scaffold_catalog_text(output, from_configs)

        def validate_scaffold(staged: Path) -> None:
            """Validate the staged catalog and identify its final origin."""
            registry = Registry.load(staged)
            private_atomic_write_text(registry.state_dir / "origin", str(output.resolve()) + "\n")

        if not atomic_create_text(output, text, validate=validate_scaffold):
            return _refuse_catalog_overwrite(output)
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
