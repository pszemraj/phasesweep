"""phasesweep CLI."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import secrets
import shlex
import sys
import traceback
from collections.abc import Callable, Iterator
from importlib import resources
from pathlib import Path
from typing import Any

import click
import yaml

from phasesweep.config import ConfigError, Experiment, Suite, load_config
from phasesweep.config.search import sampler_capability_line
from phasesweep.engine import (
    PhaseSweepError,
    PublicationAccessError,
    PublicationIntegrityError,
    config_status,
    run_config,
)
from phasesweep.engine.artifact_roots import _validate_artifact_root_binding
from phasesweep.engine.fingerprints import (
    _experiment_semantic_fingerprint,
    _suite_fingerprint,
)
from phasesweep.engine.locking import _experiment_lock, _suite_lock
from phasesweep.engine.paths import _experiment_dir, _suite_dir
from phasesweep.engine.publication import (
    _published_summary_path_for,
    _published_winner_path_for,
    _resolve_publication_pointer,
    _resolve_suite_publication_pointer,
)
from phasesweep.engine.relocation import (
    _apply_artifact_root_rebind,
    _plan_artifact_root_rebinds,
    _validate_suite_artifact_root_rebind,
)
from phasesweep.mcp import MCP_EXTRA_INSTALL_COMMAND
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.install import installer as mcp_installer
from phasesweep.mcp.install.targets import agent_ids
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.registry import (
    CatalogCheckReport,
    Registry,
    check_catalog,
    require_linux_mcp_host,
)
from phasesweep.mcp.scaffold import scaffold_catalog_text
from phasesweep.reporting import report_objective
from phasesweep.runtime.files import fsync_directory, private_atomic_write_text
from phasesweep.runtime.process import (
    install_signal_handlers,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}
CONFIG_PATH = click.Path(exists=True, dir_okay=False, path_type=Path)

# Exit taxonomy for the process-level boundary in main(). Commands that already
# choose an exit status keep it; these are the codes the boundary itself uses.
_USAGE_EXIT = 2  # bad input: config or arguments
_FAILURE_EXIT = 1  # the run itself failed
_INTERNAL_EXIT = 70  # sysexits EX_SOFTWARE: a phasesweep bug
_ABORT_EXIT = 130  # 128 + SIGINT


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
def cli() -> None:
    """Run the phasesweep command line interface."""


def main() -> None:
    """Run the CLI behind a process-level error boundary.

    The console entry point, so every failure reaches the operator as a
    diagnostic rather than a traceback. Click runs in ``standalone_mode=False``
    so its exceptions surface here instead of being converted inside Click;
    it still returns the exit status for ``ctx.exit()`` paths, and commands
    that call ``sys.exit`` themselves raise ``SystemExit``, which is a
    ``BaseException`` and passes through untouched.

    :raises SystemExit: Always, carrying one of ``_USAGE_EXIT`` (bad input),
        ``_FAILURE_EXIT`` (the run failed), ``_INTERNAL_EXIT`` (a phasesweep
        bug), ``_ABORT_EXIT`` (interrupted), or the status the command or
        Click chose.
    """
    try:
        status = cli.main(standalone_mode=False)
    except click.ClickException as exc:
        # Includes UsageError, which already carries exit code 2.
        exc.show()
        sys.exit(exc.exit_code)
    except (click.Abort, KeyboardInterrupt):
        click.echo("Aborted.", err=True)
        sys.exit(_ABORT_EXIT)
    except ConfigError as exc:
        click.echo(f"phasesweep: {exc}", err=True)
        sys.exit(_USAGE_EXIT)
    except PhaseSweepError as exc:
        click.echo(f"phasesweep: {exc}", err=True)
        # -v (see _configure_logging) means the operator asked for internals.
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        sys.exit(_FAILURE_EXIT)
    except OSError as exc:
        click.echo(f"phasesweep: operational error: {exc}", err=True)
        sys.exit(_FAILURE_EXIT)
    except Exception:  # noqa: BLE001 - the boundary's purpose is to report bugs
        click.echo("phasesweep: internal error — please report this traceback.", err=True)
        traceback.print_exc()
        sys.exit(_INTERNAL_EXIT)
    sys.exit(status if isinstance(status, int) else 0)


def _load_cli_config(path: Path) -> Experiment | Suite:
    """Load and fully validate a CLI config with path-aware diagnostics.

    :param Path path: Config file supplied to a CLI command.
    :return Experiment | Suite: Validated config. Every suite study has also
        been compiled into an experiment to validate its resolved defaults.
    :raises ConfigError: The file cannot be parsed, fails model validation, or
        contains a suite study that cannot compile into an experiment.
    """
    from pydantic import ValidationError

    try:
        config = load_config(path)
        if isinstance(config, Suite):
            for study in config.studies:
                config.experiment_for_study(study)
        return config
    except ConfigError:
        raise
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _starter_experiment_text(target: Path) -> str:
    """Render the packaged starter config with target-local absolute paths.

    :param Path target: Absolute destination for the starter YAML.
    :return str: Rendered annotated YAML text.
    """
    template = (
        resources.files("phasesweep")
        .joinpath("templates", "starter_experiment.yaml")
        .read_text(encoding="utf-8")
    )
    runs_dir = target.parent / "runs"
    # JSON and YAML double-quoted scalars share escaping for valid Unicode.
    # Keep non-ASCII workdir characters literal to avoid JSON's UTF-16 surrogate
    # pairs.
    return template.replace("__PHASESWEEP_WORKDIR__", json.dumps(str(runs_dir), ensure_ascii=False))


@contextlib.contextmanager
def _staged_text(destination: Path, text: str) -> Iterator[Path]:
    """Write and fsync text beside a destination without publishing it.

    :param Path destination: Eventual destination used to locate and name the staging file.
    :param str text: Complete UTF-8 text to stage.
    :raises FileExistsError: If ten randomized staging names collide.
    :return Iterator[Path]: Staging path, removed when the context exits.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        handle = None
        for _ in range(10):
            candidate = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
            try:
                handle = candidate.open("x", encoding="utf-8")
            except FileExistsError:
                continue
            staged = candidate
            break
        if handle is None or staged is None:
            raise FileExistsError(f"cannot create a staging file beside {destination}")
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        yield staged
    finally:
        if staged is not None:
            with contextlib.suppress(OSError):
                staged.unlink()


def _publish_staged_text(
    destination: Path,
    text: str,
    *,
    validate: Callable[[Path], object] | None = None,
) -> bool:
    """Stage, optionally validate, and exclusively publish a text file.

    :param Path destination: Destination that must not already exist.
    :param str text: Complete UTF-8 text to publish.
    :param Callable[[Path], object] | None validate: Optional staged-file validator.
    :return bool: True when published, or False if another writer won the destination race.
    """
    with _staged_text(destination, text) as staged:
        if validate is not None:
            validate(staged)
        try:
            os.link(staged, destination)
        except FileExistsError:
            return False
        fsync_directory(destination.parent)
    return True


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help="Write a runnable two-phase starter experiment without overwriting files.",
    short_help="Create a starter experiment.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("experiment.yaml"),
    show_default=True,
    help="Starter YAML destination.",
)
def init(output: Path) -> None:
    """Write an annotated starter experiment and print the review commands.

    :param Path output: Destination YAML path; existing paths are never replaced.
    :raises click.exceptions.Exit: With code 2 when ``output`` already exists as a
        file or symlink, including when it appears between the check and the
        atomic publication; with code 1 when staging or publishing fails with
        an ``OSError`` such as a permission, disk-space, or hard-link error.
    """
    expanded = output.expanduser()
    target = expanded.absolute()
    if target.exists() or target.is_symlink():
        click.echo(f"phasesweep init: refusing to overwrite existing path {target}", err=True)
        raise click.exceptions.Exit(2)
    text = _starter_experiment_text(target)
    try:
        if not _publish_staged_text(target, text):
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
    click.echo(f"Wrote starter experiment to {target}")
    click.echo("\nNext:")
    click.echo(f"  phasesweep validate {config_arg}")
    click.echo(f"  phasesweep run {config_arg} --dry-run")
    click.echo(f"  phasesweep mcp init-catalog --from {config_arg}")


@cli.command(
    "report-objective",
    context_settings=CONTEXT_SETTINGS,
    help="Publish a JSON-envelope objective from inside a PhaseSweep trial.",
    short_help="Publish this trial's objective.",
)
@click.argument("value", type=float)
@click.option("--name", required=True, help="Objective name declared by the extractor.")
@click.option("--split", required=True, help="Evaluated data split.")
@click.option("--policy", required=True, help="Evaluation policy, such as final_checkpoint.")
@click.option("--checkpoint", required=True, help="Checkpoint identity for this evaluation.")
@click.option("--step", required=True, type=click.IntRange(min=0), help="Evaluation step.")
def report_objective_cmd(
    value: float,
    name: str,
    split: str,
    policy: str,
    checkpoint: str,
    step: int,
) -> None:
    """Publish one objective through the trainer-side reporting API.

    :param float value: Finite objective value.
    :param str name: Configured objective name.
    :param str split: Evaluated data split.
    :param str policy: Evaluation policy.
    :param str checkpoint: Checkpoint identity.
    :param int step: Non-negative evaluation step.
    :raises click.BadParameter: If the objective metadata is invalid.
    :raises click.ClickException: If the PhaseSweep trial environment is absent.
    """
    try:
        destination = report_objective(
            value,
            name=name,
            split=split,
            policy=policy,
            checkpoint=checkpoint,
            step=step,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="objective") from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote objective to {destination}")


@cli.command(
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
    config = _load_cli_config(config_path)
    if from_phase is not None:
        if isinstance(config, Suite):
            click.echo("--from-phase is only supported for single experiment configs.", err=True)
            sys.exit(2)
        valid = [p.name for p in config.phases]
        if from_phase not in valid:
            click.echo(f"--from-phase={from_phase!r} not in {valid}", err=True)
            sys.exit(2)
    run_config(config, from_phase=from_phase, dry_run=dry_run)


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help="Validate a phasesweep experiment or suite config without launching any trials.",
    short_help="Validate a config file.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
def validate(config_path: Path) -> None:
    """Validate ``config_path`` without running anything."""
    config = _load_cli_config(config_path)
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
    """Render phase summaries and sampler capability lines for ``validate``.

    :param Experiment experiment: Experiment whose phases should be printed.
    :param str indent: Prefix to place before each rendered phase line.
    """
    for p in experiment.phases:
        deps = f" inherits={p.inherits}" if p.inherits else ""
        contracts = f" contracts={p.contracts}" if p.contracts else ""
        click.echo(
            f"{indent}- {p.name}: n_trials={p.n_trials} sampler={p.sampler.type}{deps}{contracts}"
        )
        # Capability disclosure (review v0.5.18 / finding F7): state the
        # resume/reproduce contract before a trial runs, not after an
        # interrupted operator hits the runtime continuation guard.
        click.echo(f"{indent}    {sampler_capability_line(p)}")
        _render_phase_comment(p.comment, prefix=f"{indent}    # ")


def _render_phase_comment(comment: str | None, *, prefix: str) -> None:
    """Render a phase comment one line at a time with ``prefix``.

    :param str | None comment: Optional phase comment to render.
    :param str prefix: Prefix to place before each rendered comment line.
    """
    if comment:
        for line in comment.strip().splitlines():
            click.echo(f"{prefix}{line}")


def _publication_integrity_error(
    subject: str,
    detail: str,
    namespace_root: Path,
) -> PublicationIntegrityError:
    """Build the one diagnostic every corrupt-publication read surface reports.

    Carries the validation failure *and* the reason not to re-run (review
    v0.5.18 / finding F4): a successful re-run publishes a new generation, the
    last-success pointer advances past the corrupt one, and nothing reports
    the corruption afterwards. The immutable namespace is still on disk until
    then, so inspecting or restoring it first is the whole remedy.

    :param str subject: Experiment or suite the corrupt publication belongs to.
    :param str detail: Validation error explaining why it no longer validates.
        Punctuated here rather than at the raising site, since validators
        return bare reason clauses.
    :param Path namespace_root: Artifact root holding the generation namespaces
        the operator must inspect.
    :return PublicationIntegrityError: Complete single-line operator diagnostic.
    """
    detail = detail.rstrip()
    if not detail.endswith((".", "!", "?")):
        detail = f"{detail}."
    return PublicationIntegrityError(
        f"{subject} records a publication that no longer validates: {detail} "
        f"Do not run anything over this tree until you have inspected or restored the "
        f"generation namespace under {namespace_root} -- a successful run would advance the "
        "last-success pointer past the corrupt generation and nothing would report it again."
    )


def _publication_access_error(subject: str, detail: str) -> PublicationAccessError:
    """Build the diagnostic for a publication this user cannot validate.

    :param str subject: Experiment or suite the publication belongs to.
    :param str detail: Path-free permission diagnostic from the validator.
    :return PublicationAccessError: Complete single-line operator diagnostic.
    """
    detail = detail.rstrip()
    if not detail.endswith((".", "!", "?")):
        detail = f"{detail}."
    return PublicationAccessError(
        f"{subject} records a publication that cannot be validated as the current user: "
        f"{detail} Results remain hidden, but this is not evidence of corruption. Re-read "
        "the publication as the publishing user or restore read permission before launching "
        "another run; rebind-workdir is not a permission-repair command."
    )


def _raise_on_failed_publication(payload: dict[str, Any]) -> None:
    """Escalate an unusable publication verdict to its typed operator error.

    Reads the status payload that was just rendered rather than re-resolving
    the pointer, so the exit status can never disagree with what the operator
    was shown. A suite payload is checked at both levels: its own suite
    last-success pointer first — mirroring :func:`_show_suite_winners`, so the
    two surfaces name the same subject for the same tree (re-review v0.5.19 /
    observation N2) — and then one embedded study at a time, since a corrupt
    component publication is a corrupt suite result too.

    :param dict[str, Any] payload: ``config_status`` payload already rendered.
    :raises PublicationAccessError: Publication validation was denied by permissions.
    :raises PublicationIntegrityError: A reported publication no longer validates.
    """
    if payload.get("kind") == "suite":
        if payload.get("publication_integrity") == "permission_denied":
            raise _publication_access_error(
                f"Suite {str(payload.get('suite'))!r}",
                str(payload.get("publication_error")),
            )
        if payload.get("publication_integrity") == "failed":
            raise _publication_integrity_error(
                f"Suite {str(payload.get('suite'))!r}",
                str(payload.get("publication_error")),
                Path(str(payload.get("workdir"))),
            )
        studies = payload.get("studies")
        for study in studies if isinstance(studies, list) else []:
            if isinstance(study, dict) and isinstance(study.get("status"), dict):
                _raise_on_failed_publication(study["status"])
        return
    if payload.get("publication_integrity") == "permission_denied":
        raise _publication_access_error(
            f"Experiment {str(payload.get('experiment'))!r}",
            str(payload.get("publication_error")),
        )
    if payload.get("publication_integrity") != "failed":
        return
    raise _publication_integrity_error(
        f"Experiment {str(payload.get('experiment'))!r}",
        str(payload.get("publication_error")),
        Path(str(payload.get("workdir"))),
    )


@cli.command(
    name="show-winners",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Print published experiment winners or the last successful exposed suite winners. "
        "Pass the same experiment or suite config YAML used for the run."
    ),
    short_help="Print saved phase winners.",
)
@click.argument("config_path", metavar="CONFIG_YAML", type=CONFIG_PATH)
def show_winners(config_path: Path) -> None:
    """Print winner files referenced by ``config_path``."""
    config = _load_cli_config(config_path)
    if isinstance(config, Suite):
        _show_suite_winners(config)
        return
    _show_experiment_winners(config)


def _show_suite_winners(suite: Suite) -> None:
    """Print the authoritative exposed winners from the last successful suite run.

    :param Suite suite: Compiled suite whose published summary is rendered.
    :raises PublicationAccessError: This user cannot validate the published suite.
    :raises PublicationIntegrityError: The suite last-success pointer names a
        suite generation that no longer validates. Reported as corruption
        rather than as "no successful suite result yet", which is what a suite
        that has genuinely never published reports (review v0.5.18 / finding F4).
    :raises click.ClickException: If the published summary cannot be read, or its
        study, phase, or annotation records are malformed; raw component-experiment
        winners are never substituted for it.
    """
    publication = _resolve_suite_publication_pointer(suite)
    if publication.state == "permission_denied":
        raise _publication_access_error(
            f"Suite {suite.suite!r}",
            str(publication.error),
        )
    if publication.state == "failed":
        raise _publication_integrity_error(
            f"Suite {suite.suite!r}",
            str(publication.error),
            _suite_dir(suite),
        )
    summary_path = _published_summary_path_for(suite, publication.generation_id)
    if publication.generation_id is not None:
        summary = publication.summary
    elif summary_path is not None and summary_path.is_file():
        try:
            summary = yaml.safe_load(summary_path.read_text())
        except (OSError, yaml.YAMLError):
            summary = None
    else:
        summary = None
    if summary_path is None or summary is None:
        click.echo("(no successful suite result yet)")
        return
    try:
        if not isinstance(summary, dict):
            raise TypeError("summary must be a mapping")
        studies = summary["studies"]
        if not isinstance(studies, list):
            raise TypeError("studies must be a list")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise click.ClickException(
            "the last successful suite summary is unreadable; refusing to substitute "
            "raw component-experiment winners"
        ) from exc

    stored_fingerprint = summary.get("suite_fingerprint")
    if isinstance(stored_fingerprint, str) and stored_fingerprint != _suite_fingerprint(suite):
        generation_id = summary.get("suite_generation_id", "unknown")
        click.echo(
            "# Historical suite result: saved generation "
            f"{generation_id} does not match the current compiled suite config."
        )
        click.echo("# Rendering the saved study graph and annotations.")

    for study in studies:
        if not isinstance(study, dict) or not isinstance(study.get("name"), str):
            raise click.ClickException(
                "the last successful suite summary has invalid study records"
            )
        click.echo(f"### study {study['name']}")
        promotion = study.get("promotion")
        if isinstance(promotion, dict):
            click.echo("--- suite promotion decision ---")
            click.echo(yaml.safe_dump(promotion, sort_keys=False).rstrip())
        click.echo("--- exposed winners ---")
        phases = study.get("phases")
        if not isinstance(phases, list):
            raise click.ClickException(
                "the last successful suite summary has invalid phase records"
            )
        for winner in phases:
            if not isinstance(winner, dict) or not isinstance(winner.get("name"), str):
                raise click.ClickException(
                    "the last successful suite summary has invalid phase records"
                )
            phase_name = winner["name"]
            comment = winner.get("comment")
            if comment is not None and not isinstance(comment, str):
                raise click.ClickException(
                    "the last successful suite summary has an invalid phase annotation"
                )
            if winner.get("exposed") is False:
                click.echo(f"=== {phase_name} === (no exposed winner)")
                _render_phase_comment(comment, prefix="# ")
                continue
            click.echo(f"=== {phase_name} ===")
            _render_phase_comment(comment, prefix="# ")
            click.echo(yaml.safe_dump(winner, sort_keys=False).rstrip())


def _show_experiment_winners(experiment: Experiment) -> None:
    """Print winner files for one experiment.

    A published result is rendered against the phase plan and comments its
    own generation summary recorded, mirroring the suite path (review
    v0.5.16 / blocker 4): old evidence must never be decorated with the
    current config's annotations, and a config that has drifted since
    publication is labeled historical instead of silently reinterpreted.

    :param Experiment experiment: Experiment whose published winners are rendered.
    :raises PublicationAccessError: This user cannot validate the publication.
    :raises PublicationIntegrityError: The last-success pointer names a
        generation that no longer validates. Nothing is rendered in that case:
        printing "(no winner yet)" beside a corrupt publication reads as a
        phase that simply has not run (review v0.5.18 / finding F4).
    """
    _validate_artifact_root_binding(experiment, claim_fresh=False)
    publication = _resolve_publication_pointer(experiment)
    if publication.state == "permission_denied":
        raise _publication_access_error(
            f"Experiment {experiment.experiment!r}",
            str(publication.error),
        )
    if publication.state == "failed":
        raise _publication_integrity_error(
            f"Experiment {experiment.experiment!r}",
            str(publication.error),
            _experiment_dir(experiment),
        )
    generation_id = publication.generation_id
    phase_plan: list[tuple[str, str | None]] = [(p.name, p.comment) for p in experiment.phases]
    if generation_id is not None:
        summary = publication.summary
        if isinstance(summary, dict):
            stored_plan = summary.get("phase_plan")
            if isinstance(stored_plan, list) and all(
                isinstance(item, dict) and isinstance(item.get("name"), str) for item in stored_plan
            ):
                phase_plan = [
                    (
                        str(item["name"]),
                        item["comment"] if isinstance(item.get("comment"), str) else None,
                    )
                    for item in stored_plan
                ]
            stored_fingerprint = summary.get("config_fingerprint")
            if isinstance(
                stored_fingerprint, str
            ) and stored_fingerprint != _experiment_semantic_fingerprint(experiment):
                click.echo(
                    "# Historical experiment result: published generation "
                    f"{generation_id} does not match the current config."
                )
                click.echo("# Rendering the saved phase plan and annotations.")

    for name, comment in phase_plan:
        wpath = _published_winner_path_for(experiment, generation_id, name)
        if wpath is not None and wpath.is_file():
            click.echo(f"=== {name} ===")
            # Show design-intent before numerical results so the reader frames
            # them against the original hypothesis instead of the other way around.
            _render_phase_comment(comment, prefix="# ")
            click.echo(wpath.read_text())
        else:
            click.echo(f"=== {name} === (no winner yet)")
            _render_phase_comment(comment, prefix="# ")


@cli.command(
    context_settings=CONTEXT_SETTINGS,
    help="Print read-only trial counts and phase state for a phasesweep experiment or suite.",
    short_help="Print read-only run status.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
def status(config_path: Path) -> None:
    """Print read-only run status for ``config_path``.

    The payload is printed first and in full even when it reports a corrupt
    publication: the operator needs the trial counts and generation identity
    to decide what to inspect, and the boundary's one-line diagnostic follows
    on stderr (review v0.5.18 / finding F4).

    :param Path config_path: Experiment or suite YAML file to inspect.
    :raises PublicationAccessError: The reported publication cannot be
        validated as the current user.
    :raises PublicationIntegrityError: The reported publication - or, for a
        suite, its own suite-level publication or any component study's - no
        longer validates.
    """
    config = _load_cli_config(config_path)
    payload = config_status(config)
    click.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())
    _raise_on_failed_publication(payload)


@cli.command(
    name="rebind-workdir",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Point this config's persistent phase studies at the workdir it now declares, after "
        "you have already moved the experiment's complete artifact tree there. Verifies at the "
        "destination that every trial in the study ledger still has its evidence directory, "
        "that no trial is RUNNING and no attempt is unresolved, and that any recorded "
        "publication validates; refuses relocating a published suite. Also the migration path "
        "for a study that predates artifact-root binding: point the config at that study's "
        "original tree - there, an interrupted RUNNING trial whose persisted paths already "
        "lie under that tree is allowed through, and the next ordinary run recovers it. "
        "Updates both the tree-to-storage record and study-side bindings; writes nothing "
        "unless every check passes."
    ),
    short_help="Rebind studies to a moved artifact tree.",
)
@click.argument("config_path", metavar="CONFIG", type=CONFIG_PATH)
def rebind_workdir(config_path: Path) -> None:
    """Move each phase study's artifact-root binding to the configured workdir.

    Each persistent phase study is bound to the one artifact root it publishes
    into, so an ordinary run against a different ``workdir`` is refused rather
    than allowed to produce a second, divergent publication tree. This command
    is the operator's explicit statement that the tree itself was relocated,
    and the only way a study that predates the binding is adopted at all. It is
    a rebind, never a move: PhaseSweep does not copy, delete, or verify the
    original tree.

    With auto storage, the database must have moved with the artifact tree.
    Its recorded previous filename is recognized without converting backends
    or moving an explicit external database into the namespace.

    What it verifies at the destination, per experiment: the namespace exists;
    every trial the study ledger holds still has its evidence directory there,
    which is what rejects a stale copy taken before the ledger advanced; no
    trial is ``RUNNING`` and no attempt registry entry is unresolved, because
    recovery follows the absolute paths those attempts persisted; and, when the
    studies record completed trials, the recorded publication validates. A
    suite that published a suite generation is refused outright - suite
    summaries record absolute component paths that do not survive relocation.

    The one ``RUNNING`` exception is adoption in place: when a pre-binding
    study's interrupted trial persisted paths that already resolve exactly
    under the offered workdir - proof the destination is the original root,
    not a copy - the binding is written with the trial (and its registry
    entry) left as-is, and the next ordinary run recovers it through the
    standard stale-attempt protocol. That protocol, not a manual Optuna
    ``tell(FAIL)``, is what records the durable failure outcome the study
    schema requires.

    The destination's reverse root-to-storage binding is updated before the
    study attrs so an interrupted rebind converges on retry. No trial,
    publication, or attempt metadata is rewritten, so the refusals remain
    broader than the cases PhaseSweep can repair (re-review v0.5.19 / blocker
    B2; see the tracked relocation TODO in ``docs/development.md``).

    :param Path config_path: Path to the experiment or suite YAML file whose
        ``workdir`` already names the artifact tree these studies own.
    :raises ArtifactRootRebindError: Storage is in-memory, every existing study
        is unbound and empty, a study cannot be read, or a destination fails
        any of the checks above; validation refusals write nothing. A later
        apply-time failure can follow an earlier suite component already being
        rebound, because the per-study storage updates are not one transaction.
    :raises ExperimentLockBusyError: Another orchestrator owns one of the
        experiment (or suite) consistency locks.
    """
    config = _load_cli_config(config_path)
    experiments = (
        [config.experiment_for_study(study) for study in config.studies]
        if isinstance(config, Suite)
        else [config]
    )
    with contextlib.ExitStack() as locks:
        # Every lock is held across validation AND application so no other
        # orchestrator can run, publish, or bind between the two halves. The
        # locks are non-blocking, so contention reports busy rather than
        # deadlocking on acquisition order.
        if isinstance(config, Suite):
            locks.enter_context(_suite_lock(config))
        for experiment in experiments:
            locks.enter_context(_experiment_lock(experiment))
        plans = _plan_artifact_root_rebinds(experiments)
        if isinstance(config, Suite):
            # Suite-level publication state is validated after the per-study
            # plans and before any of them is applied, so a suite refusal still
            # leaves every component binding exactly as it was.
            _validate_suite_artifact_root_rebind(config, plans)
        for plan in plans:
            for study_name, previous, destination in _apply_artifact_root_rebind(plan):
                origin = previous if previous is not None else "(unbound)"
                click.echo(f"{study_name}: {origin} -> {destination}")


@cli.group(
    context_settings=CONTEXT_SETTINGS,
    help="Manage the optional MCP server and coding-agent integrations.",
    short_help="Manage the MCP server and agent integrations.",
)
def mcp() -> None:
    """Run MCP operator commands."""


def _catalog_error_text(exc: CatalogError) -> str:
    """Render a catalog error together with its actionable fix.

    ``CatalogError`` carries the remediation in ``suggestion``; formatting the
    exception alone silently drops it and leaves the operator with a diagnosis
    but no instruction. Matches the ``fix:`` line ``mcp check`` prints per
    entry. Operator-facing: messages and suggestions may include paths.

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
    # does not expand "~", unlike the catalog loader (review v0.5.17 gap hunt).
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
        reuse, the runner still appears live, the run config snapshot is missing or
        does not match its recorded digest, no trial-level cleanup evidence can be
        confirmed, restored storage cannot account for published history, or study
        recovery fails with a ``RuntimeError``.
    """
    state_dir = state_dir.expanduser().resolve()
    try:
        require_linux_mcp_host()
    except CatalogError as exc:
        raise click.ClickException(_catalog_error_text(exc)) from None
    try:
        recover_run(state_dir, run_id, confirm=confirm, emit=click.echo)
    except RunRecoveryError as exc:
        raise click.ClickException(str(exc)) from None


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
    :raises FileExistsError: If ten randomized staging names all collide beside
        ``output``; handled by this function's own ``OSError`` branch, which reports
        the failure and returns ``False`` rather than propagating.
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

        if not _publish_staged_text(output, text, validate=validate_scaffold):
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

    Validates the operator-reviewed catalog first, then delegates the
    plan-then-apply flow to :mod:`phasesweep.mcp.install.installer`.
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
    report: CatalogCheckReport | None = None
    if integration != "instructions":
        if importlib.util.find_spec("mcp") is None:
            click.echo(
                "phasesweep mcp install: MCP support is not installed; install with "
                f"`{MCP_EXTRA_INSTALL_COMMAND}`; no client config was touched.",
                err=True,
            )
            ctx.exit(2)
        catalog_path = (catalog if catalog is not None else project / "catalog.yaml").resolve()
        if not catalog_path.exists():
            output_arg = shlex.quote(str(catalog_path))
            click.echo(
                f"phasesweep mcp install: no catalog at {catalog_path}. Create and review one first:\n"
                "  phasesweep mcp init-catalog --from <experiment.yaml> "
                f"-o {output_arg}\n"
                "Then edit its descriptions, visibility, and permissions, run "
                "`phasesweep mcp check`, and retry install; nothing was changed.",
                err=True,
            )
            ctx.exit(2)
        try:
            report = check_catalog(catalog_path)
        except CatalogError as exc:
            click.echo(f"phasesweep mcp install: {_catalog_error_text(exc)}", err=True)
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
            catalog_report=report,
        )
    )


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


@mcp.command(
    name="check-install",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Verify each coding agent's configured phasesweep MCP launcher still resolves: the "
        "absolute executable exists and is executable, and the configured catalog is readable. "
        "Read-only; prints repair guidance for anything broken."
    ),
    short_help="Verify configured MCP launchers.",
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    type=click.Choice(agent_ids()),
    help="Agent to check explicitly; repeat for more. Default: every supported agent.",
)
@click.option(
    "--project",
    "project_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default="current directory",
    help="Project root for project-scoped client files.",
)
@click.pass_context
def check_install_cmd(ctx: click.Context, agents: tuple[str, ...], project_dir: Path) -> None:
    """Verify configured MCP launchers resolve, for the operator.

    Read-only repair guidance for stale installs (review v0.5.15 / item G):
    reports each configured launcher's health without editing any client file.

    :param click.Context ctx: Active Click context used for the exit code.
    :param tuple[str, ...] agents: Explicit agent ids, or empty for all.
    :param Path project_dir: Project root for project-scoped client files.
    """
    ctx.exit(mcp_installer.check_install(project_dir.resolve(), list(agents) or None))


if __name__ == "__main__":
    main()
