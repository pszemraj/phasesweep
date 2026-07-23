"""Plan-then-apply orchestration for ``phasesweep mcp install`` / ``uninstall``.

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
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import click

from phasesweep.mcp import agent_prompt_text
from phasesweep.mcp.install.edits import (
    Action,
    _atomic_write_text,
    _EditLockUnavailable,
    _locked_editable_text,
    _marked_span,
    manual_json_snippet,
    merge_json_member,
    remove_json_member,
    removed_marked_text,
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
_LOCK_UNAVAILABLE_NOTE = (
    "installer lock unavailable; ensure the phasesweep lock directory is writable and "
    "PHASESWEEP_LOCK_DIR, if set, names an existing safe directory"
)


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
        return str(sibling.absolute())
    found = shutil.which("phasesweep-mcp")
    if found:
        return str(Path(found).absolute())
    raise FileNotFoundError(
        "cannot find an executable phasesweep-mcp in the active Python environment or PATH"
    )


def _project_path_is_contained(path: Path, project: Path) -> bool:
    """Return whether resolving ``path`` stays beneath the project root.

    :param Path path: Candidate path to resolve.
    :param Path project: Existing project root that must contain the candidate.
    :return bool: True when the resolved candidate is the project root or one of its descendants.
    """
    try:
        path.resolve(strict=False).relative_to(project.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _toml_mcp_entry(parsed: dict[str, object]) -> object | None:
    """Return the parsed Codex PhaseSweep table, when present.

    :param dict[str, object] parsed: Parsed complete Codex TOML document.
    :return object | None: PhaseSweep server table or ``None`` when absent.
    """
    servers = parsed.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    return servers.get(SERVER_NAME)


def _apply_toml_mcp(
    path: Path,
    mode: Mode,
    command: str,
    catalog: Path | None,
    dry_run: bool,
) -> StepResult:
    """Apply one Codex TOML edit as a locked, semantically checked transaction.

    :param Path path: Codex config path.
    :param Mode mode: ``install`` or ``uninstall``.
    :param str command: Absolute lexical MCP launcher path.
    :param Path | None catalog: Absolute catalog path for installs.
    :param bool dry_run: Compute the verdict without committing the candidate.
    :return StepResult: Safe edit or manual-attention verdict.
    """
    content = codex_toml_content(command, catalog) if catalog is not None else ""

    with _locked_editable_text(path) as loaded:
        if isinstance(loaded, _EditLockUnavailable):
            return StepResult(
                "mcp",
                path,
                "error",
                note=_LOCK_UNAVAILABLE_NOTE,
            )
        if loaded is None:
            guidance = (
                f"; merge this manually:\n{content}"
                if mode == "install"
                else "; remove the managed block manually"
            )
            return StepResult(
                "mcp",
                path,
                "error",
                note=f"config path is not a readable regular UTF-8 file{guidance}",
            )
        if mode == "uninstall" and not loaded.existed:
            return StepResult("mcp", path, "not-found")
        try:
            parsed = tomllib.loads(loaded.text)
        except tomllib.TOMLDecodeError as exc:
            suffix = f"; merge this manually:\n{content}" if content else ""
            return StepResult(
                "mcp",
                path,
                "skipped",
                note=f"config contains invalid TOML ({exc}){suffix}",
            )
        try:
            span = _marked_span(loaded.text, start=TOML_START, end=TOML_END)
        except ValueError as exc:
            suffix = f"; merge this manually:\n{content}" if content else ""
            return StepResult(
                "mcp",
                path,
                "skipped",
                note=f"config contains incomplete or repeated PhaseSweep markers ({exc}){suffix}",
            )

        current = _toml_mcp_entry(parsed)
        if mode == "uninstall":
            if span is None:
                return StepResult("mcp", path, "not-found")
            if not is_managed_mcp_entry("stdio", current):
                return StepResult(
                    "mcp",
                    path,
                    "skipped",
                    note=(
                        "PhaseSweep marker lines do not contain a recognizable managed "
                        f"{_CODEX_TABLE_HEADER} table; config was left untouched"
                    ),
                )
            candidate = removed_marked_text(loaded.text, start=TOML_START, end=TOML_END)
            assert candidate is not None
            try:
                parsed_candidate = tomllib.loads(candidate)
            except tomllib.TOMLDecodeError as exc:
                return StepResult(
                    "mcp",
                    path,
                    "skipped",
                    note=f"automatic removal would produce invalid TOML ({exc})",
                )
            if _toml_mcp_entry(parsed_candidate) is not None:
                return StepResult(
                    "mcp",
                    path,
                    "skipped",
                    note="automatic removal did not remove the managed table; config was untouched",
                )
            if dry_run:
                return StepResult("mcp", path, "removed")
            action: Action = (
                "removed" if _atomic_write_text(path, candidate, expected=loaded) else "error"
            )
            note = None if action == "removed" else "config changed before it could be replaced"
            return StepResult("mcp", path, action, note=note)

        assert catalog is not None  # narrowed by the install-only guard above
        expected = mcp_entry("stdio", command, catalog)
        if span is None and current is not None:
            return StepResult(
                "mcp",
                path,
                "skipped",
                note=(
                    f"an unmanaged {_CODEX_TABLE_HEADER} table already exists; "
                    f"update it manually to:\n{content}"
                ),
            )
        if span is not None and not is_managed_mcp_entry("stdio", current):
            return StepResult(
                "mcp",
                path,
                "skipped",
                note=(
                    "PhaseSweep marker lines do not contain a recognizable managed "
                    f"{_CODEX_TABLE_HEADER} table; merge this manually:\n{content}"
                ),
            )
        try:
            candidate = updated_marked_text(
                loaded.text,
                content,
                start=TOML_START,
                end=TOML_END,
            )
            parsed_candidate = tomllib.loads(candidate)
        except ValueError as exc:
            return StepResult(
                "mcp",
                path,
                "skipped",
                note=(
                    f"automatic merge would produce invalid TOML ({exc}); "
                    f"merge this manually:\n{content}"
                ),
            )
        if _toml_mcp_entry(parsed_candidate) != expected:
            return StepResult(
                "mcp",
                path,
                "skipped",
                note=(
                    "automatic merge did not produce the requested managed table; "
                    f"merge this manually:\n{content}"
                ),
            )
        if candidate == loaded.text:
            return StepResult("mcp", path, "unchanged")
        if dry_run:
            return StepResult("mcp", path, "updated" if loaded.existed else "created")
        action = (
            ("updated" if loaded.existed else "created")
            if _atomic_write_text(path, candidate, expected=loaded)
            else "error"
        )
        note = None if action != "error" else "config changed before it could be replaced"
        return StepResult("mcp", path, action, note=note)


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
    :param Path project: Project root that must contain project-scoped writes.
    :param bool dry_run: Compute the edit verdict without changing client files.
    :return StepResult: Edit verdict with a manual snippet on skips.
    """
    spec = target.mcp
    if spec.scope == "project" and not _project_path_is_contained(spec.path, project):
        return StepResult(
            "mcp",
            spec.path,
            "error",
            note="refusing project config path that resolves outside the project",
        )
    # Pin the physical target once. Project paths have already passed the
    # containment check; user config symlinks are operator-owned dotfile state.
    try:
        edit_path = spec.path.resolve(strict=False)
    except (OSError, RuntimeError):
        return StepResult(
            "mcp",
            spec.path,
            "error",
            note="config path could not be resolved",
        )
    if spec.format == "toml":
        if edit_path.exists() and not edit_path.is_file():
            return StepResult("mcp", spec.path, "error", note="config path is not a regular file")
        result = _apply_toml_mcp(edit_path, mode, command, catalog, dry_run)
        return StepResult(
            result.integration,
            spec.path,
            result.action,
            result.note,
        )

    if mode == "uninstall":
        managed = partial(is_managed_mcp_entry, spec.style)
        action = remove_json_member(
            edit_path,
            spec.key,
            SERVER_NAME,
            managed=managed,
            dry_run=dry_run,
        )
        if action == "lock-error":
            return StepResult("mcp", spec.path, "error", note=_LOCK_UNAVAILABLE_NOTE)
        if action == "skipped":
            note = "config is not strict JSON; remove the entry manually"
        elif action == "conflict":
            note = "an unmanaged phasesweep entry exists; it was left untouched"
        elif action == "error":
            note = "config could not be safely read or changed; remove the entry manually"
        else:
            note = None
        return StepResult("mcp", spec.path, action, note=note)
    assert catalog is not None
    if edit_path.exists() and not edit_path.is_file():
        return StepResult("mcp", spec.path, "error", note="config path is not a regular file")
    entry = mcp_entry(spec.style, command, catalog)
    managed = partial(is_managed_mcp_entry, spec.style)
    action = merge_json_member(
        edit_path,
        spec.key,
        SERVER_NAME,
        entry,
        managed=managed,
        dry_run=dry_run,
    )
    if action == "lock-error":
        return StepResult("mcp", spec.path, "error", note=_LOCK_UNAVAILABLE_NOTE)
    note = None
    if action in ("skipped", "conflict", "error"):
        if action == "skipped":
            reason = "config is not supported strict JSON (comments, duplicate keys, or numbers?)"
        elif action == "conflict":
            reason = "an unmanaged phasesweep entry already exists"
        else:
            reason = "config path or shape was unexpected"
        note = (
            f"{reason}; merge this manually:\n{manual_json_snippet(spec.key, SERVER_NAME, entry)}"
        )
    return StepResult("mcp", spec.path, action, note=note)


def _read_instruction_block(
    existing: str, valid_owner_ids: set[str]
) -> tuple[set[str], str] | None:
    """Parse the owner set and prompt body from a managed instructions block.

    :param str existing: Instructions text from the active locked transaction.
    :param set[str] valid_owner_ids: Agent ids allowed to own this exact path.
    :return tuple[set[str], str] | None: Owners and body, an empty pair when no
        block exists, or ``None`` when ownership metadata is unsafe to edit.
    :raises ValueError: If marker lines are incomplete, repeated, or out of order.
    """
    span = _marked_span(existing, start=MARKDOWN_START, end=MARKDOWN_END)
    if span is None:
        return set(), ""
    start_idx, end_idx = span

    body = existing[start_idx + len(MARKDOWN_START) : end_idx]
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    owner_line, separator, prompt = body.partition("\n")
    owner_line = owner_line.removesuffix("\r")
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
    try:
        edit_path = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return StepResult(
            "instructions",
            path,
            "error",
            note="instructions path could not be resolved",
        )
    valid_owner_ids: set[str] = set()
    for candidate_target in agent_targets(project):
        candidate_path = candidate_target.instructions_path
        if candidate_path is None:
            continue
        try:
            if candidate_path.resolve(strict=False) == edit_path:
                valid_owner_ids.add(candidate_target.id)
        except (OSError, RuntimeError):
            continue
    with _locked_editable_text(edit_path) as loaded:
        if isinstance(loaded, _EditLockUnavailable):
            return StepResult(
                "instructions",
                path,
                "error",
                note=_LOCK_UNAVAILABLE_NOTE,
            )
        if loaded is None:
            return StepResult(
                "instructions",
                path,
                "error",
                note="instructions path is not a readable regular UTF-8 file",
            )
        try:
            block = _read_instruction_block(loaded.text, valid_owner_ids)
        except ValueError as exc:
            return StepResult(
                "instructions",
                path,
                "error",
                note=f"instructions contain incomplete or repeated PhaseSweep markers ({exc})",
            )
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
                candidate = updated_marked_text(
                    loaded.text,
                    _owned_instruction_content(owners, installed_prompt),
                    start=MARKDOWN_START,
                    end=MARKDOWN_END,
                )
                if dry_run:
                    action: Action = "updated"
                else:
                    action = (
                        "updated"
                        if _atomic_write_text(edit_path, candidate, expected=loaded)
                        else "error"
                    )
                retained_note = (
                    f"retained for: {', '.join(sorted(owners))}"
                    if action != "error"
                    else "instructions changed before they could be replaced"
                )
                return StepResult("instructions", path, action, note=retained_note)
            removal_candidate = removed_marked_text(
                loaded.text,
                start=MARKDOWN_START,
                end=MARKDOWN_END,
            )
            assert removal_candidate is not None
            if dry_run:
                return StepResult("instructions", path, "removed")
            action = (
                "removed"
                if _atomic_write_text(edit_path, removal_candidate, expected=loaded)
                else "error"
            )
            removal_note = None if action == "removed" else "instructions changed before removal"
            return StepResult("instructions", path, action, note=removal_note)

        owners.add(target.id)
        candidate = updated_marked_text(
            loaded.text,
            _owned_instruction_content(owners, agent_prompt_text()),
            start=MARKDOWN_START,
            end=MARKDOWN_END,
        )
        if candidate == loaded.text:
            return StepResult("instructions", path, "unchanged")
        if dry_run:
            return StepResult(
                "instructions",
                path,
                "updated" if loaded.existed else "created",
            )
        action = (
            ("updated" if loaded.existed else "created")
            if _atomic_write_text(edit_path, candidate, expected=loaded)
            else "error"
        )
        install_note = (
            None if action != "error" else "instructions changed before they could be replaced"
        )
        return StepResult("instructions", path, action, note=install_note)


def _select_targets(
    project: Path, agent_ids: Sequence[str] | None, mode: Mode, yes: bool
) -> list[AgentTarget] | None:
    """Choose which agent targets to act on.

    :param Path project: Project root anchoring project-scoped paths.
    :param Sequence[str] | None agent_ids: Explicit target ids, or ``None``
        to select interactively from every supported client.
    :param Mode mode: ``install`` or ``uninstall`` (action shown above the menu).
    :param bool yes: Select every detected client without prompting.
    :return list[AgentTarget] | None: Selected targets, or ``None`` when
        nothing is selectable.
    """
    targets = agent_targets(project)
    if agent_ids is not None:
        by_id = {target.id: target for target in targets}
        return [by_id[agent_id] for agent_id in dict.fromkeys(agent_ids)]
    detected_by_id = {target.id: target.is_detected() for target in targets}
    detected = [target for target in targets if detected_by_id[target.id]]
    if yes:
        if detected:
            return detected
        click.echo(
            "no coding agents detected; pass --agent explicitly "
            f"(choices: {', '.join(t.id for t in targets)})",
            err=True,
        )
        return None

    ordered = sorted(targets, key=lambda target: not detected_by_id[target.id])
    defaults = [index for index, target in enumerate(ordered, start=1) if detected_by_id[target.id]]
    default_value = ",".join(str(index) for index in defaults) or "none"
    action = "install for" if mode == "install" else "uninstall from"
    click.echo(f"\nSelect coding agents to {action} (detected clients are preselected):")
    width = max(len(target.display_name) for target in ordered)
    for index, target in enumerate(ordered, start=1):
        is_detected = detected_by_id[target.id]
        marker = "x" if is_detected else " "
        status = "detected" if is_detected else "not detected"
        click.echo(f"  {index:>2}. [{marker}] {target.display_name:<{width}}  {status}")

    prompt = "Agent numbers (comma-separated, 'all', or 'none')"
    chosen: list[AgentTarget] = []
    while not chosen:
        raw = click.prompt(prompt, default=default_value).strip().lower()
        if raw == "none":
            break
        if raw == "all":
            chosen = ordered
            continue
        parts = [part.strip() for part in raw.split(",")]
        if any(not part.isdigit() for part in parts):
            click.echo(
                f"invalid selection {raw!r}; enter numbers 1-{len(ordered)}, 'all', or 'none'.",
                err=True,
            )
            continue
        selected = {int(part) for part in parts}
        if not selected or min(selected) < 1 or max(selected) > len(ordered):
            click.echo(
                f"invalid selection {raw!r}; enter numbers 1-{len(ordered)}, 'all', or 'none'.",
                err=True,
            )
            continue
        chosen = [target for index, target in enumerate(ordered, start=1) if index in selected]
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
            path: Path | None
            if integration == "mcp":
                path = target.mcp.path
                notice = target.mcp.notice
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
        target for target in targets if "mcp" in integrations and target.mcp.scope == "user"
    ]
    if mode == "install" and yes and not dry_run and user_scoped_targets and not allow_user_scope:
        names = ", ".join(target.display_name for target in user_scoped_targets)
        click.echo(
            "phasesweep mcp install: --yes cannot authorize user-scoped MCP config writes "
            f"for {names}. Review the plan, then re-run with --allow-user-scope; "
            "no client config was touched.",
            err=True,
        )
        return 2
    if not dry_run and not yes and not click.confirm("Proceed?", default=True):
        click.echo("cancelled; no client files were changed.")
        return 2

    command = ""
    if mode == "install" and "mcp" in integrations:
        try:
            command = resolve_server_command()
        except FileNotFoundError as exc:
            click.echo(f"phasesweep mcp install: {exc}; no client config was touched.", err=True)
            return 1
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
        click.echo("dry run complete; no client files were changed.")
        return 0
    if mode == "install":
        click.echo(
            "done. Restart your MCP client, then ask your agent to list phasesweep experiments."
        )
    else:
        click.echo("done. Restart your MCP client to drop the phasesweep server.")
    return 0
