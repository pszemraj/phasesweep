"""Plan-then-apply orchestration for ``phasesweep install`` / ``uninstall``.

Selects agent targets (unattended by id, or interactively among detected
clients), prints exactly what will be written where, and applies the MCP
server entry and the marker-fenced instructions block per target. Every step
reports one of the edit :data:`~phasesweep.mcp.install.edits.Action` verdicts;
``skipped``/``error`` steps print the snippet to merge manually and make the
command exit nonzero so scripts notice.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import click

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from phasesweep.mcp import agent_prompt_text
from phasesweep.mcp.install.edits import (
    Action,
    _marked_span,
    manual_json_snippet,
    merge_json_member,
    remove_json_member,
    remove_marked,
    replace_or_append_marked,
    updated_marked_text,
)
from phasesweep.mcp.install.targets import (
    MARKDOWN_END,
    MARKDOWN_START,
    SERVER_NAME,
    TOML_END,
    TOML_START,
    AgentTarget,
    agent_targets,
    codex_toml_content,
    is_managed_mcp_entry,
    mcp_entry,
)

Mode = Literal["install", "uninstall"]
Integration = Literal["mcp", "instructions"]

_OK_ACTIONS: frozenset[str] = frozenset({"created", "updated", "unchanged", "removed", "not-found"})
_CODEX_TABLE_HEADER = f"[mcp_servers.{SERVER_NAME}]"
_INSTRUCTION_OWNERS_PREFIX = "<!-- PHASESWEEP_OWNERS: "
_INSTRUCTION_OWNERS_SUFFIX = " -->"


@dataclass(frozen=True)
class StepResult:
    """Outcome of one integration edit for one agent target."""

    integration: Integration
    path: Path | None
    action: Action | None
    note: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this step needs no further operator action.

        :return bool: True for successful, idempotent, and unsupported steps.
        """
        return self.action is None or self.action in _OK_ACTIONS


def resolve_server_command() -> str:
    """Resolve the ``phasesweep-mcp`` executable clients should launch.

    Prefer the script beside the running interpreter so ``conda run`` and
    explicit environment executables cannot be redirected by an unrelated
    ``PATH`` entry.

    :return str: Absolute path to an executable ``phasesweep-mcp`` script.
    :raises FileNotFoundError: If neither the active environment nor ``PATH``
        contains a launchable script.
    """
    sibling = Path(sys.executable).parent / "phasesweep-mcp"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling.resolve())
    found = shutil.which("phasesweep-mcp")
    if found:
        return str(Path(found).resolve())
    raise FileNotFoundError(
        "cannot find an executable phasesweep-mcp in the active Python environment or PATH"
    )


def _project_path_is_contained(path: Path, project: Path) -> bool:
    """Return whether resolving ``path`` stays beneath the project root.

    :param Path path: Project-scoped target path.
    :param Path project: Project root that must contain the resolved target.
    :return bool: False when a symlinked target or parent escapes the project.
    """
    try:
        path.resolve(strict=False).relative_to(project.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _apply_mcp(
    target: AgentTarget,
    mode: Mode,
    command: str,
    catalog: Path | None,
    project: Path,
    dry_run: bool,
) -> StepResult:
    """Apply or remove the MCP server entry for one target.

    :param AgentTarget target: Client being configured.
    :param Mode mode: ``install`` or ``uninstall``.
    :param str command: Absolute ``phasesweep-mcp`` executable path.
    :param Path | None catalog: Absolute catalog path; required for install.
    :param Path project: Project root used to contain project-scoped writes.
    :param bool dry_run: Compute the edit verdict without changing client files.
    :return StepResult: Edit verdict with a manual snippet on skips.
    """
    spec = target.mcp
    if spec is None:
        return StepResult("mcp", None, None)
    if spec.scope == "project" and not _project_path_is_contained(spec.path, project):
        return StepResult(
            "mcp",
            spec.path,
            "error",
            note="refusing project config path that resolves outside the project",
        )
    if spec.format == "toml":
        if mode == "uninstall":
            return StepResult(
                "mcp",
                spec.path,
                remove_marked(
                    spec.path,
                    start=TOML_START,
                    end=TOML_END,
                    dry_run=dry_run,
                ),
            )
        if catalog is None:
            raise ValueError("installing an MCP entry requires a catalog path")
        if spec.path.is_symlink() or (spec.path.exists() and not spec.path.is_file()):
            return StepResult("mcp", spec.path, "error", note="config path is not a regular file")
        try:
            existing = spec.path.read_text(encoding="utf-8") if spec.path.exists() else ""
        except OSError:
            return StepResult("mcp", spec.path, "error", note="config file could not be read")
        content = codex_toml_content(command, catalog)
        if TOML_START not in existing:
            try:
                parsed = tomllib.loads(existing)
            except tomllib.TOMLDecodeError as exc:
                return StepResult(
                    "mcp",
                    spec.path,
                    "skipped",
                    note=f"config contains invalid TOML ({exc}); merge this manually:\n{content}",
                )
            servers = parsed.get("mcp_servers")
            if isinstance(servers, dict) and SERVER_NAME in servers:
                return StepResult(
                    "mcp",
                    spec.path,
                    "skipped",
                    note=(
                        f"an unmanaged {_CODEX_TABLE_HEADER} table already exists; "
                        f"update it manually to:\n{content}"
                    ),
                )
        try:
            candidate = updated_marked_text(existing, content, start=TOML_START, end=TOML_END)
            tomllib.loads(candidate)
        except ValueError as exc:
            return StepResult(
                "mcp",
                spec.path,
                "skipped",
                note=(
                    f"automatic merge would produce invalid TOML ({exc}); "
                    f"merge this manually:\n{content}"
                ),
            )
        action = replace_or_append_marked(
            spec.path,
            content,
            start=TOML_START,
            end=TOML_END,
            dry_run=dry_run,
        )
        return StepResult("mcp", spec.path, action)

    if mode == "uninstall":
        managed = partial(is_managed_mcp_entry, spec.style)
        action = remove_json_member(
            spec.path,
            spec.key,
            SERVER_NAME,
            managed=managed,
            dry_run=dry_run,
        )
        if action == "skipped":
            note = "config is not strict JSON; remove the entry manually"
        elif action == "conflict":
            note = "an unmanaged phasesweep entry exists; it was left untouched"
        else:
            note = None
        return StepResult("mcp", spec.path, action, note=note)
    if catalog is None:
        raise ValueError("installing an MCP entry requires a catalog path")
    entry = mcp_entry(spec.style, command, catalog)
    managed = partial(is_managed_mcp_entry, spec.style)
    action = merge_json_member(
        spec.path,
        spec.key,
        SERVER_NAME,
        entry,
        managed=managed,
        dry_run=dry_run,
    )
    note = None
    if action in ("skipped", "conflict", "error"):
        if action == "skipped":
            reason = "config is not strict JSON (comments?)"
        elif action == "conflict":
            reason = "an unmanaged phasesweep entry already exists"
        else:
            reason = "config path or shape was unexpected"
        note = (
            f"{reason}; merge this manually:\n{manual_json_snippet(spec.key, SERVER_NAME, entry)}"
        )
    return StepResult("mcp", spec.path, action, note=note)


def _read_instruction_block(path: Path, valid_owner_ids: set[str]) -> tuple[set[str], str] | None:
    """Read the owner set and prompt body from a managed instructions block.

    :param Path path: Instructions file that may contain the managed block.
    :param set[str] valid_owner_ids: Agent ids allowed to own this exact path.
    :return tuple[set[str], str] | None: Owners and body, an empty pair when no
        block exists, or ``None`` when the file or ownership metadata is unsafe
        to edit automatically.
    """
    if path.is_symlink() or (path.exists() and not path.is_file()):
        return None
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return None
    try:
        span = _marked_span(existing, start=MARKDOWN_START, end=MARKDOWN_END)
    except ValueError:
        return None
    if span is None:
        return set(), ""
    start_idx, end_idx = span

    body = existing[start_idx + len(MARKDOWN_START) : end_idx].removeprefix("\n")
    owner_line, separator, prompt = body.partition("\n")
    if (
        not separator
        or not owner_line.startswith(_INSTRUCTION_OWNERS_PREFIX)
        or not owner_line.endswith(_INSTRUCTION_OWNERS_SUFFIX)
    ):
        return None
    encoded = owner_line[len(_INSTRUCTION_OWNERS_PREFIX) : -len(_INSTRUCTION_OWNERS_SUFFIX)]
    owner_ids = encoded.split(",") if encoded else []
    owners = set(owner_ids)
    if not owners or len(owners) != len(owner_ids) or not owners <= valid_owner_ids:
        return None
    return owners, prompt


def _owned_instruction_content(owners: set[str], prompt: str) -> str:
    """Render ownership metadata followed by the shared agent prompt.

    :param set[str] owners: Agent ids that currently use this instructions block.
    :param str prompt: Managed phasesweep instructions body.
    :return str: Complete content to place between the marker fence.
    """
    owner_line = (
        f"{_INSTRUCTION_OWNERS_PREFIX}{','.join(sorted(owners))}{_INSTRUCTION_OWNERS_SUFFIX}"
    )
    return f"{owner_line}\n{prompt}"


def _apply_instructions(
    target: AgentTarget,
    mode: Mode,
    project: Path,
    dry_run: bool,
) -> StepResult:
    """Apply or remove the instructions marker block for one target.

    :param AgentTarget target: Client being configured.
    :param Mode mode: ``install`` or ``uninstall``.
    :param Path project: Project root used to contain project-scoped writes.
    :param bool dry_run: Compute the edit verdict without changing the instructions file.
    :return StepResult: Edit verdict for the instructions file.
    """
    path = target.instructions_path
    if path is None:
        return StepResult("instructions", None, None)
    if not _project_path_is_contained(path, project):
        return StepResult(
            "instructions",
            path,
            "error",
            note="refusing instructions path that resolves outside the project",
        )
    valid_owner_ids = {
        candidate.id for candidate in agent_targets(project) if candidate.instructions_path == path
    }
    block = _read_instruction_block(path, valid_owner_ids)
    if block is None:
        return StepResult(
            "instructions",
            path,
            "error",
            note="instructions block has missing or invalid ownership metadata",
        )
    owners, installed_prompt = block
    if mode == "uninstall":
        if target.id not in owners:
            return StepResult("instructions", path, "not-found")
        owners.remove(target.id)
        if owners:
            action = replace_or_append_marked(
                path,
                _owned_instruction_content(owners, installed_prompt),
                start=MARKDOWN_START,
                end=MARKDOWN_END,
                dry_run=dry_run,
            )
            return StepResult(
                "instructions",
                path,
                action,
                note=f"retained for: {', '.join(sorted(owners))}",
            )
        return StepResult(
            "instructions",
            path,
            remove_marked(
                path,
                start=MARKDOWN_START,
                end=MARKDOWN_END,
                dry_run=dry_run,
            ),
        )
    owners.add(target.id)
    action = replace_or_append_marked(
        path,
        _owned_instruction_content(owners, agent_prompt_text()),
        start=MARKDOWN_START,
        end=MARKDOWN_END,
        dry_run=dry_run,
    )
    return StepResult("instructions", path, action)


def _select_targets(
    project: Path, agent_ids: Sequence[str] | None, mode: Mode, yes: bool
) -> list[AgentTarget] | None:
    """Choose which agent targets to act on.

    :param Path project: Project root anchoring project-scoped paths.
    :param Sequence[str] | None agent_ids: Explicit target ids, or ``None``
        to select interactively among detected clients.
    :param Mode mode: ``install`` or ``uninstall`` (verb used in prompts).
    :param bool yes: Skip the per-client confirmation prompts.
    :return list[AgentTarget] | None: Selected targets, or ``None`` when
        nothing is selectable.
    """
    targets = agent_targets(project)
    if agent_ids is not None:
        by_id = {target.id: target for target in targets}
        return [by_id[agent_id] for agent_id in dict.fromkeys(agent_ids)]
    detected = [target for target in targets if target.is_detected()]
    if not detected:
        click.echo(
            "no coding agents detected; pass --agent explicitly "
            f"(choices: {', '.join(t.id for t in targets)})",
            err=True,
        )
        return None
    if yes:
        return detected
    verb = "Configure" if mode == "install" else "Remove phasesweep from"
    chosen = [
        target
        for target in detected
        if click.confirm(f"{verb} {target.display_name}?", default=True)
    ]
    if not chosen:
        click.echo("nothing selected.")
        return None
    return chosen


def _integrations(integration: Literal["mcp", "instructions", "all"]) -> tuple[Integration, ...]:
    """Expand the ``--type`` flag into concrete integrations.

    :param Literal integration: ``mcp``, ``instructions``, or ``all``.
    :return tuple[Integration, ...]: Integrations to apply, in apply order.
    """
    if integration == "all":
        return ("mcp", "instructions")
    return (integration,)


def _print_plan(
    targets: Sequence[AgentTarget],
    integrations: tuple[Integration, ...],
    mode: Mode,
    dry_run: bool,
) -> None:
    """Print what will be written or removed before touching anything.

    :param Sequence[AgentTarget] targets: Selected agent targets.
    :param tuple[Integration, ...] integrations: Integrations to apply.
    :param Mode mode: ``install`` or ``uninstall``.
    :param bool dry_run: Whether this plan will only compute edit verdicts.
    """
    prefix = "Dry-run " if dry_run else ""
    click.echo(f"\n{prefix}{'install' if mode == 'install' else 'uninstall'} plan:")
    for target in targets:
        click.echo(f"  {target.display_name}")
        for integration in integrations:
            if integration == "mcp":
                path = target.mcp.path if target.mcp else None
                notice = target.mcp.notice if target.mcp else None
            else:
                path, notice = target.instructions_path, None
            if path is None:
                click.echo(f"    {integration:<13} (not supported)")
                continue
            click.echo(f"    {integration:<13} {path}")
            if notice and mode == "install":
                click.echo(f"    {'':<13} note: {notice}")
    click.echo("")


def run(
    mode: Mode,
    project: Path,
    catalog: Path | None,
    agent_ids: Sequence[str] | None,
    integration: Literal["mcp", "instructions", "all"],
    yes: bool,
    dry_run: bool = False,
    allow_user_scope: bool = False,
    before_apply: Callable[[], bool] | None = None,
) -> int:
    """Run the installer or uninstaller end to end.

    :param Mode mode: ``install`` or ``uninstall``.
    :param Path project: Project root anchoring project-scoped paths.
    :param Path | None catalog: Validated absolute catalog path (install only).
    :param Sequence[str] | None agent_ids: Explicit target ids, or ``None``
        for interactive selection among detected clients.
    :param Literal integration: ``mcp``, ``instructions``, or ``all``.
    :param bool yes: Skip every confirmation prompt.
    :param bool dry_run: Report planned edit verdicts without changing client files.
    :param bool allow_user_scope: Explicitly authorize unattended user-scoped MCP writes.
    :param Callable[[], bool] | None before_apply: Optional post-confirmation preparation;
        client edits proceed only when it returns True.
    :return int: ``0`` when every step succeeded, ``1`` when any step needs
        manual attention, ``2`` when nothing was selected or confirmed.
    """
    if mode == "install" and integration != "instructions" and catalog is None:
        raise ValueError("installing MCP entries requires a validated catalog path")
    targets = _select_targets(project, agent_ids, mode, yes)
    if targets is None:
        return 2
    integrations = _integrations(integration)
    _print_plan(targets, integrations, mode, dry_run)
    user_scoped_targets = [
        target
        for target in targets
        if "mcp" in integrations and target.mcp is not None and target.mcp.scope == "user"
    ]
    if mode == "install" and yes and not dry_run and user_scoped_targets and not allow_user_scope:
        names = ", ".join(target.display_name for target in user_scoped_targets)
        click.echo(
            "phasesweep install: --yes cannot authorize user-scoped MCP config writes "
            f"for {names}. Review the plan, then re-run with --allow-user-scope; "
            "no client config was touched.",
            err=True,
        )
        return 2
    if not dry_run and not yes and not click.confirm("Proceed?", default=True):
        click.echo("cancelled; nothing was changed.")
        return 2

    command = ""
    if mode == "install" and "mcp" in integrations:
        try:
            command = resolve_server_command()
        except FileNotFoundError as exc:
            click.echo(f"phasesweep install: {exc}; no client config was touched.", err=True)
            return 1
    if not dry_run and before_apply is not None and not before_apply():
        return 2
    attention = 0
    for target in targets:
        click.echo(f"  {target.display_name}")
        for kind in integrations:
            if kind == "mcp":
                result = _apply_mcp(target, mode, command, catalog, project, dry_run)
            else:
                result = _apply_instructions(target, mode, project, dry_run)
            if result.action is None:
                click.echo(f"    {result.integration:<13} not supported")
                continue
            display_action: str = result.action
            if dry_run:
                display_action = {
                    "created": "would-create",
                    "updated": "would-update",
                    "removed": "would-remove",
                }.get(result.action, result.action)
            click.echo(f"    {result.integration:<13} {display_action:<13} {result.path}")
            if result.note:
                for line in result.note.splitlines():
                    click.echo(f"      {line}")
            if not result.ok:
                attention += 1
    click.echo("")
    if attention:
        click.echo(
            f"{attention} step(s) need manual attention (see above).",
            err=True,
        )
        return 1
    if dry_run:
        click.echo("dry run complete; nothing was changed.")
        return 0
    if mode == "install":
        click.echo(
            "done. Restart your MCP client, then ask your agent to list phasesweep experiments."
        )
    else:
        click.echo("done. Restart your MCP client to drop the phasesweep server.")
    return 0
