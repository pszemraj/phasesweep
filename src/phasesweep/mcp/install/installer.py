"""Plan-then-apply orchestration for ``phasesweep install`` / ``uninstall``.

Selects agent targets (unattended by id, or interactively among detected
clients), prints exactly what will be written where, and applies the MCP
server entry and the marker-fenced instructions block per target. Every step
reports one of the edit :data:`~phasesweep.mcp.install.edits.Action` verdicts;
``skipped``/``error`` steps print the snippet to merge manually and make the
command exit nonzero so scripts notice.
"""

from __future__ import annotations

import functools
import importlib.resources
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click

from phasesweep.mcp.install.edits import (
    Action,
    manual_json_snippet,
    merge_json_member,
    remove_json_member,
    remove_marked,
    replace_or_append_marked,
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
    mcp_entry,
)

Mode = Literal["install", "uninstall"]
Integration = Literal["mcp", "instructions"]

_OK_ACTIONS: frozenset[str] = frozenset({"created", "updated", "unchanged", "removed", "not-found"})
_CODEX_TABLE_HEADER = f"[mcp_servers.{SERVER_NAME}]"


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


@functools.cache
def instructions_text() -> str:
    """Return the packaged agent instructions installed between markers.

    :return str: Contents of the canonical ``agent_prompt.md`` package data.
    """
    return (
        importlib.resources.files("phasesweep.mcp")
        .joinpath("agent_prompt.md")
        .read_text(encoding="utf-8")
    )


def resolve_server_command() -> str:
    """Resolve the ``phasesweep-mcp`` executable clients should launch.

    :return str: Absolute path from ``PATH`` when available, otherwise the
        script directory of the running interpreter (venv/conda layout).
    """
    found = shutil.which("phasesweep-mcp")
    if found:
        return found
    return str(Path(sys.executable).parent / "phasesweep-mcp")


def _apply_mcp(target: AgentTarget, mode: Mode, command: str, catalog: Path | None) -> StepResult:
    """Apply or remove the MCP server entry for one target.

    :param AgentTarget target: Client being configured.
    :param Mode mode: ``install`` or ``uninstall``.
    :param str command: Absolute ``phasesweep-mcp`` executable path.
    :param Path | None catalog: Absolute catalog path; required for install.
    :return StepResult: Edit verdict with a manual snippet on skips.
    """
    spec = target.mcp
    if spec is None:
        return StepResult("mcp", None, None)
    if spec.format == "toml":
        if mode == "uninstall":
            return StepResult(
                "mcp", spec.path, remove_marked(spec.path, start=TOML_START, end=TOML_END)
            )
        if catalog is None:
            raise ValueError("installing an MCP entry requires a catalog path")
        existing = spec.path.read_text(encoding="utf-8") if spec.path.exists() else ""
        if _CODEX_TABLE_HEADER in existing and TOML_START not in existing:
            return StepResult(
                "mcp",
                spec.path,
                "skipped",
                note=(
                    f"an unmanaged {_CODEX_TABLE_HEADER} table already exists; "
                    "update it manually to:\n" + codex_toml_content(command, catalog)
                ),
            )
        action = replace_or_append_marked(
            spec.path, codex_toml_content(command, catalog), start=TOML_START, end=TOML_END
        )
        return StepResult("mcp", spec.path, action)

    if mode == "uninstall":
        action = remove_json_member(spec.path, spec.key, SERVER_NAME)
        note = (
            "config is not strict JSON; remove the entry manually" if action == "skipped" else None
        )
        return StepResult("mcp", spec.path, action, note=note)
    if catalog is None:
        raise ValueError("installing an MCP entry requires a catalog path")
    entry = mcp_entry(spec.style, command, catalog)
    action = merge_json_member(spec.path, spec.key, SERVER_NAME, entry)
    note = None
    if action in ("skipped", "error"):
        reason = (
            "config is not strict JSON (comments?)"
            if action == "skipped"
            else "config shape was unexpected"
        )
        note = (
            f"{reason}; merge this manually:\n{manual_json_snippet(spec.key, SERVER_NAME, entry)}"
        )
    return StepResult("mcp", spec.path, action, note=note)


def _apply_instructions(target: AgentTarget, mode: Mode) -> StepResult:
    """Apply or remove the instructions marker block for one target.

    :param AgentTarget target: Client being configured.
    :param Mode mode: ``install`` or ``uninstall``.
    :return StepResult: Edit verdict for the instructions file.
    """
    path = target.instructions_path
    if path is None:
        return StepResult("instructions", None, None)
    if mode == "uninstall":
        return StepResult(
            "instructions", path, remove_marked(path, start=MARKDOWN_START, end=MARKDOWN_END)
        )
    action = replace_or_append_marked(
        path, instructions_text(), start=MARKDOWN_START, end=MARKDOWN_END
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
    targets: Sequence[AgentTarget], integrations: tuple[Integration, ...], mode: Mode
) -> None:
    """Print what will be written or removed before touching anything.

    :param Sequence[AgentTarget] targets: Selected agent targets.
    :param tuple[Integration, ...] integrations: Integrations to apply.
    :param Mode mode: ``install`` or ``uninstall``.
    """
    click.echo(f"\n{'Install' if mode == 'install' else 'Uninstall'} plan:")
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
) -> int:
    """Run the installer or uninstaller end to end.

    :param Mode mode: ``install`` or ``uninstall``.
    :param Path project: Project root anchoring project-scoped paths.
    :param Path | None catalog: Validated absolute catalog path (install only).
    :param Sequence[str] | None agent_ids: Explicit target ids, or ``None``
        for interactive selection among detected clients.
    :param Literal integration: ``mcp``, ``instructions``, or ``all``.
    :param bool yes: Skip every confirmation prompt.
    :return int: ``0`` when every step succeeded, ``1`` when any step needs
        manual attention, ``2`` when nothing was selected or confirmed.
    """
    if mode == "install" and integration != "instructions" and catalog is None:
        raise ValueError("installing MCP entries requires a validated catalog path")
    targets = _select_targets(project, agent_ids, mode, yes)
    if targets is None:
        return 2
    integrations = _integrations(integration)
    _print_plan(targets, integrations, mode)
    if not yes and not click.confirm("Proceed?", default=True):
        click.echo("cancelled; nothing was changed.")
        return 2

    command = resolve_server_command()
    attention = 0
    for target in targets:
        click.echo(f"  {target.display_name}")
        for kind in integrations:
            if kind == "mcp":
                result = _apply_mcp(target, mode, command, catalog)
            else:
                result = _apply_instructions(target, mode)
            if result.action is None:
                click.echo(f"    {result.integration:<13} not supported")
                continue
            click.echo(f"    {result.integration:<13} {result.action:<10} {result.path}")
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
    if mode == "install":
        click.echo(
            "done. Restart your MCP client, then ask your agent to list phasesweep experiments."
        )
    else:
        click.echo("done. Restart your MCP client to drop the phasesweep server.")
    return 0
