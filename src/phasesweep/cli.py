"""phasesweep CLI."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import click

from phasesweep.config import Experiment, Suite, load_config
from phasesweep.engine import config_status, run_config
from phasesweep.engine.guards import (
    _experiment_lock,
    _inspect_cleanup_uncertain_trials,
    _inspect_stale_running_trials,
    _reap_stale_trials,
    _recover_cleanup_uncertain_trials,
)
from phasesweep.engine.optuna import _load_existing_phase_study
from phasesweep.engine.state import _winner_path
from phasesweep.mcp.config_snapshot import load_experiment_snapshot
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.install import installer as mcp_installer
from phasesweep.mcp.install.targets import agent_ids
from phasesweep.mcp.registry import CatalogCheckReport, Registry, check_catalog
from phasesweep.mcp.runs import RunStore, write_status_file
from phasesweep.mcp.scaffold import scaffold_catalog_text
from phasesweep.mcp.snapshots import finalize_result_snapshot, parse_result_snapshot
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.files import fsync_directory, private_atomic_write_text
from phasesweep.runtime.process import (
    install_signal_handlers,
    is_same_live_process,
    kill_stale_group,
    read_proc_starttime,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
CONFIG_PATH = click.Path(exists=True, dir_okay=False, path_type=Path)


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


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Phase-chained hyperparameter sweeps driven by a YAML file.",
)
@click.version_option(package_name="phasesweep")
def main() -> None:
    """Run the phasesweep command line interface."""


@main.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Run every phase in a phasesweep experiment config. Use --from-phase to skip earlier "
        "phases after their winner.yaml files already exist."
    ),
    short_help="Run configured phases.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
@click.option(
    "--from-phase",
    metavar="PHASE",
    default=None,
    show_default="first phase",
    help="Skip earlier phases and load their winners from disk.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    show_default=True,
    help="Render one example command per phase without launching trials.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    show_default=True,
    help="Show debug logs from phasesweep and INFO logs from Optuna.",
)
def run(config_path: Path, from_phase: str | None, dry_run: bool, verbose: bool) -> None:
    """Run all phases defined in ``config_path``.

    :param Path config_path: Path to the experiment or suite YAML file.
    :param str | None from_phase: Optional phase name to start from after loading
        earlier winners from disk.
    :param bool dry_run: Render example commands without launching subprocesses.
    :param bool verbose: Enable debug logging for phasesweep and INFO logging for Optuna.
    """
    _configure_logging(verbose)
    # Install before config parsing so CLI SIGINT/SIGTERM behavior and exit
    # codes remain structured throughout startup, including dry-run previews.
    # run_experiment() repeats this idempotently for direct library callers.
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


@main.command(
    context_settings=CONTEXT_SETTINGS,
    help="Validate a phasesweep experiment or suite config without launching any trials.",
    short_help="Validate a config file.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
def validate(config_path: Path) -> None:
    """Validate ``config_path`` without running anything."""
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
    """Render phase summaries for ``validate``.

    :param Experiment experiment: Experiment whose phases should be printed.
    :param str indent: Prefix to place before each rendered phase line.
    """
    for p in experiment.phases:
        deps = f" inherits={p.inherits}" if p.inherits else ""
        contracts = f" contracts={p.contracts}" if p.contracts else ""
        click.echo(
            f"{indent}- {p.name}: n_trials={p.n_trials} sampler={p.sampler.type}{deps}{contracts}"
        )
        _render_phase_comment(p.comment, prefix=f"{indent}    # ")


def _render_phase_comment(comment: str | None, *, prefix: str) -> None:
    """Render a phase comment one line at a time with ``prefix``.

    :param str | None comment: Optional phase comment to render.
    :param str prefix: Prefix to place before each rendered comment line.
    """
    if comment:
        for line in comment.strip().splitlines():
            click.echo(f"{prefix}{line}")


@main.command(
    name="show-winners",
    context_settings=CONTEXT_SETTINGS,
    help="Print winner.yaml content for every phase in a phasesweep experiment or suite.",
    short_help="Print saved phase winners.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
def show_winners(config_path: Path) -> None:
    """Print winner files from ``config_path``."""
    config = load_config(config_path)
    if isinstance(config, Suite):
        for study in config.studies:
            click.echo(f"### study {study.name}")
            _show_experiment_winners(config.experiment_for_study(study))
        return
    _show_experiment_winners(config)


def _show_experiment_winners(experiment: Experiment) -> None:
    """Print winner files for one experiment."""
    for p in experiment.phases:
        wpath = _winner_path(experiment, p.name)
        if wpath.is_file():
            click.echo(f"=== {p.name} ===")
            # Show design-intent before numerical results so the reader frames
            # them against the original hypothesis instead of the other way around.
            _render_phase_comment(p.comment, prefix="# ")
            click.echo(wpath.read_text())
        else:
            click.echo(f"=== {p.name} === (no winner yet)")
            _render_phase_comment(p.comment, prefix="# ")


@main.command(
    context_settings=CONTEXT_SETTINGS,
    help="Print read-only trial counts and phase state for a phasesweep experiment or suite.",
    short_help="Print read-only run status.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
def status(config_path: Path) -> None:
    """Print read-only run status for ``config_path``."""
    config = load_config(config_path)
    click.echo(_format_status(config_status(config)))


@main.group(
    context_settings=CONTEXT_SETTINGS,
    help="Manage the optional MCP server and coding-agent integrations.",
    short_help="Manage the MCP server and agent integrations.",
)
def mcp() -> None:
    """Run MCP operator commands."""


@mcp.command(
    name="recover-run",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Operator-only recovery for MCP cleanup uncertainty or failed terminal-result "
        "finalization. --confirm performs the reported cleanup and stored-snapshot actions."
    ),
    short_help="Recover MCP cleanup or result finalization.",
)
@click.option(
    "--state-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="MCP state_dir containing runs/ and logs/.",
)
@click.option("--run-id", required=True, help="MCP run id to recover.")
@click.option(
    "--confirm",
    is_flag=True,
    help="Perform the reported cleanup and stored terminal-result actions.",
)
def mcp_recover_run(state_dir: Path, run_id: str, confirm: bool) -> None:
    """Recover cleanup uncertainty or finalize an already-captured result snapshot.

    :param Path state_dir: MCP state directory containing the run metadata.
    :param str run_id: Identifier of the run to recover.
    :param bool confirm: Whether to perform recovery instead of only reporting actions.
    """
    if not sys.platform.startswith("linux"):
        raise click.ClickException(
            "MCP recovery is supported only on Linux because safe process cleanup "
            "requires /proc process identities"
        )
    if read_proc_starttime(os.getpid()) is None:
        raise click.ClickException(
            "MCP recovery cannot read this process's Linux /proc start time; "
            "mount /proc with process stat access before retrying"
        )
    try:
        store = RunStore.open_existing(state_dir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    handle = store.get(run_id)
    if handle is None:
        raise click.ClickException(f"unknown run id: {run_id}")
    terminal_status = store.recorded_terminal_status(handle)
    cleanup_recovery_required = store.cleanup_recovery_required(handle)
    snapshot_recovery_required = store.snapshot_recovery_required(handle)
    terminal_cleanup_uncertain = (
        terminal_status is not None
        and terminal_status.get("cleanup_confirmed") is False
        and cleanup_recovery_required
    )
    cleanup_already_recovered = (
        terminal_status is not None
        and terminal_status.get("cleanup_confirmed") is False
        and not cleanup_recovery_required
    )
    runner_without_status = handle.launch_state == "spawned" and terminal_status is None
    launch_without_status = handle.launch_state == "launching" and terminal_status is None
    cleanup_recovery_needed = cleanup_recovery_required or runner_without_status
    stored_snapshot = (
        parse_result_snapshot(terminal_status) if terminal_status is not None else None
    )
    snapshot_unavailable = (
        (terminal_status is not None and stored_snapshot is None)
        or launch_without_status
        or runner_without_status
    )
    snapshot_finalize_needed = stored_snapshot is not None and (
        terminal_cleanup_uncertain or cleanup_already_recovered or snapshot_recovery_required
    )
    if (
        not cleanup_recovery_needed
        and snapshot_unavailable
        and not launch_without_status
        and not snapshot_recovery_required
    ):
        raise click.ClickException(
            "this run has no immutable terminal result snapshot. Historical results cannot "
            "be rebuilt from the current shared study because later runs may have changed it."
        )
    if not cleanup_recovery_needed and not snapshot_finalize_needed and not snapshot_unavailable:
        click.echo("No cleanup uncertainty or terminal result repair is needed for this run.")
        return

    identity = store.cleanup_identity(handle)
    if cleanup_recovery_needed and identity.pid is not None and identity.pid_starttime is None:
        raise click.ClickException(
            "runner process identity has no Linux /proc start time; refusing automated "
            "recovery because PID reuse cannot be ruled out"
        )
    if is_same_live_process(identity.pid, identity.pid_starttime):
        raise click.ClickException("runner still appears live; use phasesweep_cancel_sweep first")
    snapshot = store.config_snapshot_path(run_id)
    if not snapshot.is_file():
        raise click.ClickException(f"run config snapshot is missing: {snapshot}")
    try:
        config = load_experiment_snapshot(
            snapshot,
            handle.config_sha256,
            source=f"run snapshot {run_id}",
        )
    except OSError as exc:
        raise click.ClickException(f"cannot read run config snapshot: {snapshot}") from exc
    except ValueError as exc:
        raise click.ClickException(f"{exc}; refusing recovery") from None

    recovery_lock = _experiment_lock(config) if confirm else contextlib.nullcontext()
    try:
        with recovery_lock:
            # Lock contention proves another ordinary CLI or MCP orchestrator is
            # using this exact output/storage namespace. Re-check runner liveness
            # only after acquiring the lock, then keep it through every signal,
            # study mutation, and recovery-state write.
            if confirm and is_same_live_process(identity.pid, identity.pid_starttime):
                raise click.ClickException(
                    "runner still appears live; use phasesweep_cancel_sweep first"
                )
            if (
                cleanup_recovery_needed
                and confirm
                and not kill_stale_group(
                    identity.pid,
                    identity.pid_starttime,
                    pgid=identity.pgid,
                    grace_seconds=30.0,
                )
            ):
                raise click.ClickException("runner process-group cleanup is still uncertain")

            reaped = 0
            cleanup_recovered = 0
            inspected_studies = 0
            if cleanup_recovery_needed:
                for phase in config.phases:
                    study = _load_existing_phase_study(config, phase)
                    if study is None:
                        continue
                    inspected_studies += 1
                    if confirm:
                        cleanup_recovered += _recover_cleanup_uncertain_trials(
                            study,
                            config,
                            phase.name,
                        )
                        reaped += _reap_stale_trials(study, config, phase.name)
                    else:
                        cleanup_recovered += _inspect_cleanup_uncertain_trials(study)
                        reaped += _inspect_stale_running_trials(study, config, phase.name)
            cleanup_evidence_count = reaped + cleanup_recovered
            if terminal_cleanup_uncertain and cleanup_evidence_count == 0:
                if inspected_studies == 0:
                    detail = (
                        "no existing Optuna studies could be loaded from the run snapshot storage"
                    )
                else:
                    detail = (
                        "no RUNNING trials were reaped and no terminal trials recorded cleanup "
                        "uncertainty"
                    )
                raise click.ClickException(
                    "runner status recorded cleanup_confirmed=false, but recovery could not "
                    f"confirm any trial-level cleanup evidence ({detail}). Refusing to clear "
                    "cleanup uncertainty."
                )

            if not confirm:
                actions = []
                if cleanup_recovery_needed:
                    actions.append(
                        "attempt runner process-group cleanup, "
                        f"reap {reaped} stale trial(s), and recover {cleanup_recovered} "
                        "cleanup-uncertain terminal trial(s)"
                    )
                if snapshot_finalize_needed:
                    actions.append("finalize the stored terminal snapshot with cleanup evidence")
                elif snapshot_unavailable:
                    actions.append("record that the historical terminal snapshot is unavailable")
                click.echo(
                    f"Recovery preflight for {run_id}: would {' and '.join(actions)}. "
                    "Re-run with --confirm to perform those actions."
                )
                return

            if terminal_status is None:
                terminal_status = {
                    "run_id": run_id,
                    "returncode": 1,
                    "error_class": "RunnerExitedWithoutStatus",
                    "cleanup_confirmed": True,
                    "ended_at": utc_now_iso(),
                    "result_snapshot_state": "failed",
                    "result_snapshot_error": "HistoricalSnapshotUnavailable",
                }
                write_status_file(store.status_path(run_id), terminal_status)

            if cleanup_recovery_needed:
                payload = {
                    "run_id": run_id,
                    "config_sha256": handle.config_sha256,
                    "recovered_at": utc_now_iso(),
                    "cleanup_confirmed": True,
                    "reaped_running_trials": reaped,
                    "cleanup_uncertain_terminal_trials": cleanup_recovered,
                }
                private_atomic_write_text(
                    store.cleanup_recovery_path(run_id),
                    json.dumps(payload, indent=2) + "\n",
                )
                store.clear_cleanup_uncertain(handle)
                click.echo(
                    f"Cleared cleanup uncertainty for {run_id}; reaped {reaped} stale trial(s) "
                    f"and confirmed {cleanup_recovered} cleanup-uncertain trial(s)."
                )

            if snapshot_recovery_required and stored_snapshot is None:
                assert terminal_status is not None
                terminal_status["result_snapshot_state"] = "failed"
                terminal_status["result_snapshot_error"] = "InterruptedFinalization"
                write_status_file(store.status_path(run_id), terminal_status)

            if snapshot_finalize_needed:
                _finalize_stored_terminal_result_snapshot(
                    store,
                    run_id,
                    terminal_status,
                    cleanup_confirmed=(
                        cleanup_recovery_needed
                        or cleanup_already_recovered
                        or terminal_status.get("cleanup_confirmed") is True
                    ),
                )

                click.echo(f"Finalized stored terminal result snapshot for {run_id}.")
            elif snapshot_unavailable:
                click.echo(
                    f"Historical terminal result snapshot for {run_id} is unavailable and was "
                    "not rebuilt from mutable shared state."
                )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from None


def _finalize_stored_terminal_result_snapshot(
    store: RunStore,
    run_id: str,
    terminal_status: dict,
    *,
    cleanup_confirmed: bool,
) -> None:
    """Finalize and persist a snapshot captured before the experiment lock was released.

    :param RunStore store: Existing run store containing the terminal status.
    :param str run_id: Run whose stored terminal snapshot should be finalized.
    :param dict terminal_status: Validated terminal process status to enrich.
    :param bool cleanup_confirmed: Whether RUNNING trials are terminal in reality.
    :raises click.ClickException: If the stored snapshot is unavailable or persistence fails.
    """
    snapshot = parse_result_snapshot(terminal_status)
    if snapshot is None:
        raise click.ClickException(
            f"run {run_id} has no immutable snapshot to finalize; refusing to read current "
            "shared state as historical evidence"
        )
    raw_snapshot = snapshot.model_dump(mode="json")
    terminal_status.pop("result_snapshot_error", None)
    terminal_status["result_snapshot_state"] = "pending"
    try:
        write_status_file(store.status_path(run_id), terminal_status)
        terminal_status["result_snapshot"] = finalize_result_snapshot(
            raw_snapshot,
            cleanup_confirmed=cleanup_confirmed,
        )
        terminal_status["result_snapshot_state"] = "complete"
        write_status_file(store.status_path(run_id), terminal_status)
    except Exception as exc:  # noqa: BLE001 - convert operator repair failures to CLI errors
        terminal_status["result_snapshot"] = raw_snapshot
        terminal_status["result_snapshot_state"] = "pending"
        terminal_status["result_snapshot_error"] = type(exc).__name__
        with contextlib.suppress(Exception):
            write_status_file(store.status_path(run_id), terminal_status)
        raise click.ClickException(
            f"failed to finalize terminal result snapshot for {run_id}: {type(exc).__name__}"
        ) from None


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

    The MCP SDK import stays behind this command so the base CLI still works
    without installing the ``mcp`` optional dependency.

    :param click.Context ctx: Active Click context used to exit with the server return code.
    :param Path catalog: Operator-authored MCP catalog to load.
    """
    from phasesweep.mcp.server import serve

    ctx.exit(serve(catalog))


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
        click.echo(f"phasesweep mcp check: {exc}", err=True)
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
        "state_dir next to the catalog, one read-only entry per --from config "
        "(visible_params: none, no allow block). The staged catalog is validated with "
        "the server startup path before it is published. Existing files are never overwritten; "
        "edit the result, then re-check it with `phasesweep mcp check`."
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
    if os.path.lexists(output):
        click.echo(
            f"phasesweep mcp init-catalog: {output} already exists; refusing to overwrite. "
            "Pass -o to choose another name.",
            err=True,
        )
        return False

    staged: Path | None = None
    try:
        text = scaffold_catalog_text(output, from_configs)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            staged = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        Registry.load(staged)
        try:
            os.link(staged, output)
        except FileExistsError:
            click.echo(
                f"phasesweep mcp init-catalog: {output} already exists; refusing to overwrite. "
                "Pass -o to choose another name.",
                err=True,
            )
            return False
        fsync_directory(output.parent)
    except CatalogError as exc:
        click.echo(f"phasesweep mcp init-catalog: {exc}", err=True)
        if exc.suggestion:
            click.echo(f"fix: {exc.suggestion}", err=True)
        return False
    except OSError as exc:
        click.echo(f"phasesweep mcp init-catalog: cannot write {output}: {exc}", err=True)
        return False
    finally:
        if staged is not None:
            with contextlib.suppress(OSError):
                staged.unlink()
    click.echo(f"wrote {output}")
    return True


@mcp.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Wire the phasesweep MCP server into coding-agent configs: an MCP server entry plus a "
        "marker-fenced instructions block per agent, project-scoped wherever the client "
        "supports it. The catalog is validated with the exact server startup rules before any "
        "client config is touched. Strict JSON configs are re-serialized and may be reformatted. "
        "Generated entries bind to the Python environment running this command; rerun install "
        "after replacing that environment. Without --agent, choose from all supported agents in "
        "one menu, with detected clients preselected."
    ),
    short_help="Connect coding agents to the MCP server.",
)
@click.option(
    "--catalog",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    metavar="PATH",
    show_default="<project>/catalog.yaml",
    help="MCP catalog the installed server entry will serve.",
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    type=click.Choice(agent_ids()),
    help="Agent to configure explicitly; repeat for more.",
)
@click.option(
    "--type",
    "integration",
    type=click.Choice(["mcp", "instructions", "all"]),
    default="all",
    show_default=True,
    help="Integration to write: the MCP server entry, the instructions block, or both.",
)
@click.option(
    "--project",
    "project_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default="current directory",
    help="Project root for project-scoped client files.",
)
@click.option("--yes", is_flag=True, help="Apply without confirmation prompts.")
@click.option(
    "--allow-user-scope",
    is_flag=True,
    help="Acknowledge user-scoped MCP config writes when using --yes.",
)
@click.option(
    "--dry-run", is_flag=True, help="Preview planned client-file edits without applying them."
)
@click.pass_context
def install(
    ctx: click.Context,
    catalog: Path | None,
    agents: tuple[str, ...],
    integration: str,
    project_dir: Path,
    yes: bool,
    allow_user_scope: bool,
    dry_run: bool,
) -> None:
    """Install phasesweep MCP and instructions integrations for coding agents.

    Validates (or interactively scaffolds) the catalog first, then delegates
    the plan-then-apply flow to :mod:`phasesweep.mcp.install.installer`.
    Operator-facing: output may include paths.

    :param click.Context ctx: Active Click context used for the exit code.
    :param Path | None catalog: Catalog path; defaults to ``<project>/catalog.yaml``.
    :param tuple[str, ...] agents: Explicit agent ids for unattended runs.
    :param str integration: ``mcp``, ``instructions``, or ``all``.
    :param Path project_dir: Project root for project-scoped client files.
    :param bool yes: Skip every confirmation prompt.
    :param bool allow_user_scope: Acknowledge unattended user-scoped MCP config writes.
    :param bool dry_run: Preview installer verdicts without changing client files.
    """
    project = project_dir.resolve()
    catalog_path: Path | None = None
    if integration != "instructions":
        if importlib.util.find_spec("mcp") is None:
            click.echo(
                "phasesweep mcp install: MCP support is not installed; install with "
                "`pip install 'phasesweep[mcp]'`; no client config was touched.",
                err=True,
            )
            ctx.exit(2)
        catalog_path = (catalog if catalog is not None else project / "catalog.yaml").resolve()
        if not catalog_path.exists():
            if dry_run:
                click.echo(
                    f"phasesweep mcp install: no catalog at {catalog_path}. Scaffold one first:\n"
                    f"  phasesweep mcp init-catalog --from <experiment.yaml> -o {catalog_path}; "
                    "nothing was changed.",
                    err=True,
                )
                ctx.exit(2)
            if not _offer_catalog_scaffold(catalog_path, yes):
                ctx.exit(2)
        try:
            report = check_catalog(catalog_path)
        except CatalogError as exc:
            click.echo(f"phasesweep mcp install: {exc}", err=True)
            ctx.exit(2)
        if not report.ok:
            _echo_catalog_report(report)
            click.echo(
                "phasesweep mcp install: fix the catalog (see report above); "
                "no client config was touched.",
                err=True,
            )
            ctx.exit(2)
    ctx.exit(
        mcp_installer.run(
            "install",
            project,
            catalog_path,
            list(agents) or None,
            integration,  # type: ignore[arg-type]
            yes,
            dry_run,
            allow_user_scope,
        )
    )


def _offer_catalog_scaffold(catalog_path: Path, yes: bool) -> bool:
    """Offer to scaffold a missing catalog before installing.

    Unattended runs fail with the exact command to run instead; interactive
    runs prompt for one experiment config and scaffold from it.

    :param Path catalog_path: Missing catalog destination.
    :param bool yes: Whether the run is unattended.
    :return bool: True when a validated catalog now exists at ``catalog_path``.
    """
    suggestion = f"phasesweep mcp init-catalog --from <experiment.yaml> -o {catalog_path}"
    if yes:
        click.echo(
            f"phasesweep mcp install: no catalog at {catalog_path}. Scaffold one first:\n"
            f"  {suggestion}",
            err=True,
        )
        return False
    click.echo(f"no catalog at {catalog_path}.")
    raw = click.prompt(
        "experiment config to scaffold it from (blank to abort)",
        default="",
        show_default=False,
    ).strip()
    if not raw:
        click.echo(f"aborted; scaffold a catalog with: {suggestion}", err=True)
        return False
    config = Path(raw)
    if not config.is_file():
        click.echo(f"phasesweep mcp install: no such config file: {config}", err=True)
        return False
    return _write_catalog_scaffold(catalog_path, (config,))


@mcp.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Remove installer-owned phasesweep integration data: recognizable generated-shape JSON "
        "entries and marker-fenced TOML or instruction blocks, per selected agent. Unmanaged "
        "same-name entries stay untouched."
    ),
    short_help="Disconnect coding agents.",
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    type=click.Choice(agent_ids()),
    help="Agent to clean up explicitly; repeat for more.",
)
@click.option(
    "--type",
    "integration",
    type=click.Choice(["mcp", "instructions", "all"]),
    default="all",
    show_default=True,
    help="Integration to remove.",
)
@click.option(
    "--project",
    "project_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default="current directory",
    help="Project root for project-scoped client files.",
)
@click.option("--yes", is_flag=True, help="Apply without confirmation prompts.")
@click.option(
    "--dry-run", is_flag=True, help="Preview planned client-file removals without applying them."
)
@click.pass_context
def uninstall(
    ctx: click.Context,
    agents: tuple[str, ...],
    integration: str,
    project_dir: Path,
    yes: bool,
    dry_run: bool,
) -> None:
    """Remove installed phasesweep integrations from coding agents.

    :param click.Context ctx: Active Click context used for the exit code.
    :param tuple[str, ...] agents: Explicit agent ids for unattended runs.
    :param str integration: ``mcp``, ``instructions``, or ``all``.
    :param Path project_dir: Project root for project-scoped client files.
    :param bool yes: Skip every confirmation prompt.
    :param bool dry_run: Preview uninstaller verdicts without changing client files.
    """
    ctx.exit(
        mcp_installer.run(
            "uninstall",
            project_dir.resolve(),
            None,
            list(agents) or None,
            integration,  # type: ignore[arg-type]
            yes,
            dry_run,
        )
    )


def _format_status(status_obj: dict) -> str:
    """Render status data as stable YAML.

    :param dict status_obj: Status payload returned by :func:`config_status`.
    :return str: YAML-formatted status text without a trailing newline.
    """
    import yaml

    return yaml.safe_dump(status_obj, sort_keys=False).rstrip()


if __name__ == "__main__":
    main()
