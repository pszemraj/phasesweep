"""Plan-then-apply orchestration for ``phasesweep mcp install`` / ``uninstall``.

Selects agent targets (unattended by id, or interactively among detected
clients), prints exactly what will be written where, and applies the MCP
server entry and the marker-fenced instructions block per target. Every step
reports one of the edit :data:`~phasesweep.mcp.install.edits.Action` verdicts;
``skipped``/``error`` steps print the snippet to merge manually and make the
command exit nonzero so scripts notice.

:func:`check_install` is the read-only counterpart (review v0.5.15 / item G):
it never edits a client file, only inspects whichever phasesweep MCP entry is
already configured and reports whether its launcher executable and its
configured catalog path still resolve.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePath
from typing import Literal, TypeAlias

import click

from phasesweep.mcp import agent_prompt_text
from phasesweep.mcp.install.edits import (
    Action,
    _atomic_write_text,
    _EditLockUnavailable,
    _locked_editable_text,
    _marked_span,
    _read_editable_text,
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
    _entry_argv,
    _is_absolute_phasesweep_command,
    agent_targets,
    codex_toml_content,
    is_managed_mcp_entry,
    mcp_entry,
)
from phasesweep.mcp.registry import CatalogCheckReport
from phasesweep.runtime.json import strict_json_loads

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


def _which_on_absolute_path(command: str) -> str | None:
    """Look ``command`` up on the absolute ``PATH`` entries only.

    ``PATH`` may carry relative entries -- ``.``, a bare ``bin``, or an empty
    field, which POSIX reads as the current directory. Resolving through them
    would make the answer an accident of the directory the caller happened to
    run from, so they are dropped: a relative-only ``PATH`` resolves nothing
    here rather than falling back to the unfiltered ``PATH``.

    :param str command: Bare program name to look up.
    :return str | None: Absolute path to the resolved executable, or ``None``
        when no absolute ``PATH`` entry provides one.
    """
    entries = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and PurePath(entry).is_absolute()
    ]
    if not entries:
        return None
    return shutil.which(command, path=os.pathsep.join(entries))


def resolve_server_command() -> str:
    """Resolve the ``phasesweep-mcp`` executable clients should launch.

    Prefer the script beside the running interpreter so ``conda run`` and
    explicit environment executables cannot be redirected by an unrelated
    ``PATH`` entry. The ``PATH`` fallback consults only absolute ``PATH``
    entries (:func:`_which_on_absolute_path`), so the pinned command is never
    an artifact of the working directory the install ran from.

    :return str: Absolute path to an executable ``phasesweep-mcp`` script.
    :raises FileNotFoundError: If neither the active environment nor the
        absolute ``PATH`` entries contain a launchable script.
    """
    sibling = Path(sys.executable).parent / "phasesweep-mcp"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling.absolute())
    found = _which_on_absolute_path("phasesweep-mcp")
    if found:
        return str(Path(found).absolute())
    raise FileNotFoundError(
        "cannot find an executable phasesweep-mcp in the active Python environment or PATH"
    )


def _resolved_project_path(path: Path, project: Path) -> Path | None:
    """Resolve ``path`` once when it stays beneath the project root.

    :param Path path: Candidate path to resolve.
    :param Path project: Existing project root that must contain the candidate.
    :return Path | None: Resolved candidate, or ``None`` when resolution or containment fails.
    """
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(project.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


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
    :param str command: Absolute ``phasesweep-mcp`` executable path.
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
            write_result = _atomic_write_text(path, candidate, expected=loaded)
            action: Action = "removed" if write_result == "written" else "error"
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
        write_result = _atomic_write_text(path, candidate, expected=loaded)
        action = (
            ("updated" if loaded.existed else "created") if write_result == "written" else "error"
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
    if spec.scope == "project":
        edit_path = _resolved_project_path(spec.path, project)
        if edit_path is None:
            return StepResult(
                "mcp",
                spec.path,
                "error",
                note="refusing project config path that resolves outside the project",
            )
    else:
        # User config symlinks are operator-owned dotfile state.
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
        if action == "stale":
            return StepResult(
                "mcp",
                spec.path,
                "error",
                note="config changed before it could be replaced",
            )
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
    if action == "stale":
        return StepResult(
            "mcp",
            spec.path,
            "error",
            note=(
                "config changed before it could be replaced; merge this manually:\n"
                f"{manual_json_snippet(spec.key, SERVER_NAME, entry)}"
            ),
        )
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


def _valid_instruction_owner_ids(edit_path: Path, project: Path) -> set[str]:
    """Return agent ids whose instructions target resolves to one shared file.

    :param Path edit_path: Resolved instructions file being inspected or edited.
    :param Path project: Project root used to resolve all supported targets.
    :return set[str]: Valid owner ids for the shared instructions path.
    """
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
    return valid_owner_ids


def _instruction_plan_note(target: AgentTarget, project: Path, mode: Mode) -> str | None:
    """Describe existing shared ownership and prompt replacement before applying.

    :param AgentTarget target: Selected integration target.
    :param Path project: Project root anchoring the instructions path.
    :param Mode mode: Planned installer action.
    :return str | None: Shared-owner plan note, when a readable managed block exists.
    """
    path = target.instructions_path
    if path is None:
        return None
    edit_path = _resolved_project_path(path, project)
    if edit_path is None:
        return None
    snapshot = _read_editable_text(edit_path)
    if snapshot is None or not snapshot.existed:
        return None
    try:
        block = _read_instruction_block(
            snapshot.text,
            _valid_instruction_owner_ids(edit_path, project),
        )
    except ValueError:
        return None
    if block is None:
        return None
    owners, installed_prompt = block
    if not owners:
        return None
    owner_text = ", ".join(sorted(owners))
    if mode == "install" and installed_prompt != agent_prompt_text():
        return f"shared block prompt will update for existing owners: {owner_text}"
    return f"shared block currently owned by: {owner_text}"


def _apply_instructions(
    target: AgentTarget,
    mode: Mode,
    project: Path,
    dry_run: bool,
    dry_run_state: dict[Path, str] | None = None,
) -> StepResult:
    """Apply or remove the instructions marker block for one target.

    :param AgentTarget target: Client being configured.
    :param Mode mode: ``install`` or ``uninstall``.
    :param Path project: Project root used to contain project-scoped writes.
    :param bool dry_run: Compute the edit verdict without changing the instructions file.
    :param dict[Path, str] | None dry_run_state: Planned text from earlier dry-run
        steps, keyed by resolved instructions path.
    :return StepResult: Edit verdict for the instructions file.
    """
    path = target.instructions_path
    if path is None:
        return StepResult("instructions", None, None)
    edit_path = _resolved_project_path(path, project)
    if edit_path is None:
        return StepResult(
            "instructions",
            path,
            "error",
            note="refusing instructions path that resolves outside the project",
        )
    valid_owner_ids = _valid_instruction_owner_ids(edit_path, project)
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
        staged_text = (
            dry_run_state.get(edit_path) if dry_run and dry_run_state is not None else None
        )
        existing_text = staged_text if staged_text is not None else loaded.text
        existed = loaded.existed or staged_text is not None
        try:
            block = _read_instruction_block(existing_text, valid_owner_ids)
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
                    existing_text,
                    _owned_instruction_content(owners, installed_prompt),
                    start=MARKDOWN_START,
                    end=MARKDOWN_END,
                )
                if dry_run:
                    if dry_run_state is not None:
                        dry_run_state[edit_path] = candidate
                    action: Action = "updated"
                else:
                    write_result = _atomic_write_text(edit_path, candidate, expected=loaded)
                    action = "updated" if write_result == "written" else "error"
                retained_note = (
                    f"retained for: {', '.join(sorted(owners))}"
                    if action != "error"
                    else "instructions changed before they could be replaced"
                )
                return StepResult("instructions", path, action, note=retained_note)
            removal_candidate = removed_marked_text(
                existing_text,
                start=MARKDOWN_START,
                end=MARKDOWN_END,
            )
            assert removal_candidate is not None
            if dry_run:
                if dry_run_state is not None:
                    dry_run_state[edit_path] = removal_candidate
                return StepResult("instructions", path, "removed")
            write_result = _atomic_write_text(edit_path, removal_candidate, expected=loaded)
            action = "removed" if write_result == "written" else "error"
            removal_note = None if action == "removed" else "instructions changed before removal"
            return StepResult("instructions", path, action, note=removal_note)

        owners.add(target.id)
        candidate = updated_marked_text(
            existing_text,
            _owned_instruction_content(owners, agent_prompt_text()),
            start=MARKDOWN_START,
            end=MARKDOWN_END,
        )
        if candidate == existing_text:
            return StepResult("instructions", path, "unchanged")
        if dry_run:
            if dry_run_state is not None:
                dry_run_state[edit_path] = candidate
            return StepResult(
                "instructions",
                path,
                "updated" if existed else "created",
            )
        write_result = _atomic_write_text(edit_path, candidate, expected=loaded)
        action = (
            ("updated" if loaded.existed else "created") if write_result == "written" else "error"
        )
        install_note = (
            None if action != "error" else "instructions changed before they could be replaced"
        )
        return StepResult("instructions", path, action, note=install_note)


def _select_explicit_targets(
    targets: Sequence[AgentTarget],
    agent_ids: Sequence[str],
) -> list[AgentTarget] | None:
    """Resolve explicit agent ids once, preserving request order and reporting unknown ids.

    :param Sequence[AgentTarget] targets: Supported integration targets.
    :param Sequence[str] agent_ids: Explicit ids requested by the operator.
    :return list[AgentTarget] | None: Deduplicated targets, or ``None`` for unknown ids.
    """
    by_id = {target.id: target for target in targets}
    selected_ids = list(dict.fromkeys(agent_ids))
    unknown = [agent_id for agent_id in selected_ids if agent_id not in by_id]
    if unknown:
        click.echo(
            f"unknown coding agent id(s): {', '.join(unknown)} (choices: {', '.join(by_id)})",
            err=True,
        )
        return None
    return [by_id[agent_id] for agent_id in selected_ids]


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
        return _select_explicit_targets(targets, agent_ids)
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


def _mcp_plan_note(target: AgentTarget, command: str, catalog: Path) -> str:
    """Describe the MCP entry already on disk and what installing will do to it.

    Answers the only question the plan cannot leave to the post-apply verdicts:
    whether confirming rewrites an entry that is already there. Classification
    reuses :func:`_check_target_launcher`, the same read-only reader
    ``check-install`` and the post-install verification section use, so the
    pre-confirmation plan cannot disagree with them about the same on-disk
    state.

    :param AgentTarget target: Client whose MCP config the plan would edit.
    :param str command: Absolute ``phasesweep-mcp`` executable to be pinned.
    :param Path catalog: Absolute catalog path the installed entry will pass.
    :return str: Plan note, possibly spanning several lines.
    """
    check = _check_target_launcher(target)
    if check.status == "not-configured":
        return "no existing phasesweep entry; installing creates one"
    if check.status == "unmanaged":
        return (
            "an existing phasesweep entry is not a shape this installer owns; it will be left "
            "untouched and printed for manual merge"
        )
    if check.status == "unreadable" or check.executable is None:
        # No argv was recovered, so there is no existing invocation to describe.
        return (
            "the existing config could not be read as a phasesweep entry; it will be left "
            "untouched and the entry printed for manual merge"
        )
    current = shlex.join([check.executable, *check.args])
    planned = shlex.join([command, "--catalog", str(catalog)])
    if current == planned:
        return "an existing phasesweep entry already matches this plan; it stays unchanged"
    shape = (
        "an existing phasesweep entry"
        if _is_absolute_phasesweep_command(check.executable)
        else "a recognized legacy launcher entry"
    )
    return (
        f"{shape} WILL BE REWRITTEN to this environment's pinned executable\n"
        f"from: {current}\n"
        f"to:   {planned}"
    )


def _echo_plan_note(note: str) -> None:
    """Print one plan note under its integration line, indenting continuations.

    :param str note: Note text, possibly spanning several lines.
    """
    first, *rest = note.splitlines()
    click.echo(f"    {'':<13} note: {first}")
    for line in rest:
        click.echo(f"    {'':<13}       {line}")


def _print_plan(
    targets: Sequence[AgentTarget],
    integrations: tuple[Integration, ...],
    mode: Mode,
    dry_run: bool,
    *,
    project: Path,
    catalog: Path | None,
    command: str,
    catalog_report: CatalogCheckReport | None,
) -> None:
    """Print what will be written or removed before touching anything.

    MCP installs additionally state, per target, whether an entry is already
    configured and whether confirming rewrites it, so an operator never learns
    of a rewrite only from the verdicts printed after confirming.

    :param Sequence[AgentTarget] targets: Selected agent targets.
    :param tuple[Integration, ...] integrations: Integrations to apply.
    :param Mode mode: ``install`` or ``uninstall``.
    :param bool dry_run: Whether this plan will only compute edit verdicts.
    :param Path project: Project root anchoring shared instructions paths.
    :param Path | None catalog: Validated catalog used by MCP installs.
    :param str command: Resolved launcher executable for MCP installs.
    :param CatalogCheckReport | None catalog_report: Validated experiment permissions summary.
    """
    prefix = "Dry-run " if dry_run else ""
    click.echo(f"\n{prefix}{'install' if mode == 'install' else 'uninstall'} plan:")
    if mode == "install" and "mcp" in integrations:
        assert catalog is not None
        click.echo(f"  catalog      {catalog}")
        if catalog_report is not None:
            click.echo(f"  experiments  {len(catalog_report.entries)}")
            for entry in catalog_report.entries:
                actions = ", ".join(entry.actions) if entry.actions else "read-only"
                click.echo(f"    {entry.experiment_id}: {actions}")
        click.echo(f"  launcher     {command}")
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
            scope = " [user scope]" if integration == "mcp" and target.mcp.scope == "user" else ""
            click.echo(f"    {integration:<13} {path}{scope}")
            if notice and mode == "install":
                _echo_plan_note(notice)
            if integration == "mcp" and mode == "install" and catalog is not None:
                _echo_plan_note(_mcp_plan_note(target, command, catalog))
            if integration == "instructions":
                shared_note = _instruction_plan_note(target, project, mode)
                if shared_note:
                    _echo_plan_note(shared_note)
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
    catalog_report: CatalogCheckReport | None = None,
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
    :param CatalogCheckReport | None catalog_report: Validated entries summarized in the plan.
    :return int: ``0`` when every step succeeded, ``1`` when any step needs
        manual attention, ``2`` when nothing was selected or confirmed.
    :raises ValueError: If an MCP install is requested without a validated
        catalog path.
    """
    if mode == "install" and integration != "instructions" and catalog is None:
        raise ValueError("installing MCP entries requires a validated catalog path")
    integrations = _integrations(integration)
    command = ""
    if mode == "install" and "mcp" in integrations:
        try:
            command = resolve_server_command()
        except FileNotFoundError as exc:
            click.echo(f"phasesweep mcp install: {exc}; no client config was touched.", err=True)
            return 1
    targets = _select_targets(project, agent_ids, mode, yes)
    if targets is None:
        return 2
    _print_plan(
        targets,
        integrations,
        mode,
        dry_run,
        project=project,
        catalog=catalog,
        command=command,
        catalog_report=catalog_report,
    )
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

    attention = 0
    # Targets whose MCP step already reported a problem; verification re-reads the
    # same config and must not count that one problem a second time.
    mcp_needs_attention: set[str] = set()
    instruction_dry_run_state: dict[Path, str] | None = {} if dry_run else None
    for target in targets:
        click.echo(f"  {target.display_name}")
        for kind in integrations:
            if kind == "mcp":
                result = _apply_mcp(target, mode, command, catalog, project, dry_run)
            else:
                result = _apply_instructions(
                    target,
                    mode,
                    project,
                    dry_run,
                    dry_run_state=instruction_dry_run_state,
                )
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
                if kind == "mcp":
                    mcp_needs_attention.add(target.id)
    # Verification runs regardless of earlier per-step failures: a run where one
    # target failed is exactly where the remaining targets' launchers most need
    # checking. It is reported for every selected target, but only adds to
    # `attention` (and so to the exit code) for a problem no earlier step already
    # counted, using the same needs-attention predicate as `check_install` so both
    # commands reach the same verdict on the same on-disk state.
    if mode == "install" and not dry_run and "mcp" in integrations:
        click.echo("\nverification:")
        for target in targets:
            verification = _check_target_launcher(target)
            click.echo(f"  {target.display_name:<20} {verification.status}")
            if verification.detail:
                for line in verification.detail.splitlines():
                    click.echo(f"    {line}")
            if not verification.ok and target.id not in mcp_needs_attention:
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
        click.echo("done.")
        click.echo("Restart the selected client(s), then ask:")
        click.echo("List the available PhaseSweep experiments and their permitted actions.")
        click.echo("Do not launch anything.")
    else:
        click.echo("done. Restart your MCP client to drop the phasesweep server.")
    return 0


# --- check-install: read-only verification (review v0.5.15 / item G) ---

# User-facing status semantics live in docs/mcp_setup.md#verification-and-maintenance.
# `_CHECK_ATTENTION_STATUSES` is the implementation's needs-attention set.
CheckStatus: TypeAlias = Literal[
    "ok",
    "missing",
    "not-executable",
    "catalog-missing",
    "unmanaged",
    "not-configured",
    "unreadable",
]

_CHECK_ATTENTION_STATUSES: frozenset[str] = frozenset(
    {"missing", "not-executable", "catalog-missing", "unreadable"}
)

# A legacy pinned-launcher entry that resolves here still resolves its launcher from
# whatever PATH the client is started with, so `ok` alone would overstate the check.
_LEGACY_LAUNCHER_NOTE = (
    "recognized legacy launcher form, not the pinned absolute phasesweep-mcp executable; it "
    "resolved from this shell's PATH, but the client may launch with a different one; rerun "
    "`phasesweep mcp install` to pin this environment's executable"
)


@dataclass(frozen=True)
class LauncherCheck:
    """Verification outcome for one target's configured phasesweep MCP launcher."""

    config_path: Path | None
    executable: str | None
    args: tuple[str, ...]
    status: CheckStatus
    detail: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this check needs no further operator action.

        :return bool: True unless the launcher is missing, not executable, its
            catalog is missing, or the config is unreadable.
        """
        return self.status not in _CHECK_ATTENTION_STATUSES


def _probe_launcher_executable(command: str) -> tuple[CheckStatus, str | None]:
    """Probe whether one configured launcher executable actually resolves.

    A bare program name is resolved against the absolute ``PATH`` entries only,
    so the same entry cannot check out from one working directory and fail from
    another.

    :param str command: Configured launcher executable: an absolute path for
        entries this version writes, or a bare program name for the legacy
        pinned entries earlier versions wrote.
    :return tuple[CheckStatus, str | None]: ``("ok", None)``, or a status
        needing attention with actionable repair guidance.
    """
    path = Path(command)
    if not path.is_absolute():
        # Only the legacy pinned entry shape reaches here; it names a program the client
        # resolves from PATH, so resolve it the same way rather than against cwd. Relative
        # PATH entries are deliberately ignored, so the verdict never depends on where
        # check-install was run from.
        resolved = _which_on_absolute_path(command)
        if resolved is None:
            return "missing", (
                f"{command} is not on PATH; rerun `phasesweep mcp install` to replace this "
                "legacy entry with an absolute launcher path (--dry-run previews the repair)"
            )
        path = Path(resolved)
    if not path.is_file():
        return "missing", (
            f"{command} no longer exists; rerun `phasesweep mcp install` from the correct "
            "conda environment (--dry-run previews the repair)"
        )
    if not os.access(path, os.X_OK):
        return "not-executable", f"{command} exists but is not executable; check its permissions"
    return "ok", None


def _probe_configured_catalog(args: Sequence[str]) -> tuple[CheckStatus, str | None]:
    """Probe whether the catalog an entry passes to the server is still readable.

    A launcher that starts fine still fails at handshake time when its catalog
    was moved, renamed, or deleted, so ``check-install`` inspects the argument
    as well as the executable (review v0.5.16). The caller has already verified
    that every managed shape ends in ``--catalog <absolute path>``.

    :param Sequence[str] args: Configured launcher argv after the executable.
    :return tuple[CheckStatus, str | None]: ``("ok", None)``, or
        ``("catalog-missing", ...)`` with actionable repair guidance.
    """
    catalog = args[-1]
    path = Path(catalog)
    if not path.is_file() or not os.access(path, os.R_OK):
        return "catalog-missing", (
            f"the configured catalog {catalog} is not a readable file; rerun "
            "`phasesweep mcp install --catalog PATH` with the catalog's current location "
            "(--dry-run previews the repair)"
        )
    return "ok", None


def _check_target_launcher(target: AgentTarget) -> LauncherCheck:
    """Read one target's configured phasesweep MCP entry and probe it.

    Read-only counterpart to :func:`_apply_mcp`: recognizes every entry shape
    :func:`is_managed_mcp_entry` owns (including the legacy pinned-uvx entries
    earlier versions wrote), reports an entry this installer does not own as
    ``unmanaged`` without probing it, and never edits the client file. Probes
    both the launcher executable and the configured ``--catalog`` path, the
    executable first because it fails earlier at launch. A legacy entry that
    passes both probes stays ``ok`` but carries
    :data:`_LEGACY_LAUNCHER_NOTE`, because resolving its launcher here says
    nothing about the PATH the client will launch with.

    :param AgentTarget target: Client to inspect.
    :return LauncherCheck: Verification outcome for this target.
    """
    spec = target.mcp
    try:
        edit_path = spec.path.resolve(strict=False)
    except (OSError, RuntimeError):
        return LauncherCheck(
            spec.path,
            None,
            (),
            "unreadable",
            "config path could not be resolved",
        )

    snapshot = _read_editable_text(edit_path)
    if snapshot is None:
        return LauncherCheck(
            spec.path,
            None,
            (),
            "unreadable",
            "config path is not a readable regular UTF-8 file",
        )
    if not snapshot.existed or not snapshot.text.strip():
        return LauncherCheck(spec.path, None, (), "not-configured")

    if spec.format == "toml":
        try:
            parsed = tomllib.loads(snapshot.text)
        except tomllib.TOMLDecodeError as exc:
            return LauncherCheck(
                spec.path,
                None,
                (),
                "unreadable",
                f"config contains invalid TOML ({exc})",
            )
        entry = _toml_mcp_entry(parsed)
        if entry is None:
            return LauncherCheck(spec.path, None, (), "not-configured")
        if not is_managed_mcp_entry("stdio", entry) or not isinstance(entry, dict):
            return LauncherCheck(
                spec.path,
                None,
                (),
                "unmanaged",
                f"an unmanaged {_CODEX_TABLE_HEADER} table exists; not installer-verified",
            )
        argv = _entry_argv("stdio", entry)
        assert argv is not None  # is_managed_mcp_entry already validated this shape
        command, args = argv[0], argv[1:]
    else:
        try:
            data = strict_json_loads(snapshot.text, finite_floats=True)
        except ValueError:
            return LauncherCheck(
                spec.path,
                None,
                (),
                "unreadable",
                "config is not strict JSON",
            )
        if not isinstance(data, dict):
            return LauncherCheck(
                spec.path,
                None,
                (),
                "unreadable",
                "config top level is not a JSON object",
            )
        container = data.get(spec.key)
        member = container.get(SERVER_NAME) if isinstance(container, dict) else None
        if member is None:
            return LauncherCheck(spec.path, None, (), "not-configured")
        if not is_managed_mcp_entry(spec.style, member) or not isinstance(member, dict):
            return LauncherCheck(
                spec.path,
                None,
                (),
                "unmanaged",
                "an unmanaged phasesweep entry exists; not installer-verified",
            )
        argv = _entry_argv(spec.style, member)
        assert argv is not None  # is_managed_mcp_entry already validated this shape
        command, args = argv[0], argv[1:]

    status, detail = _probe_launcher_executable(command)
    if status == "ok":
        status, detail = _probe_configured_catalog(args)
        if status == "ok" and not _is_absolute_phasesweep_command(command):
            detail = _LEGACY_LAUNCHER_NOTE
    return LauncherCheck(spec.path, command, tuple(args), status, detail)


def check_install(project: Path, agent_ids: Sequence[str] | None = None) -> int:
    """Verify each target's configured phasesweep MCP launcher still resolves.

    Read-only counterpart to ``install``/``uninstall`` (review v0.5.15 / item
    G): for every selected target, inspects whatever phasesweep MCP entry is
    already on disk, reports
    whether it is installer-managed, whether its launcher executable resolves,
    and whether the ``--catalog`` path it passes is still a readable file, then
    prints repair guidance for anything broken. Never edits a client file or
    reaches the network.

    :param Path project: Project root anchoring project-scoped paths.
    :param Sequence[str] | None agent_ids: Explicit target ids, or ``None`` to
        check every supported target.
    :return int: ``0`` when every configured launcher resolves, ``1`` when at
        least one needs attention, ``2`` for an unknown agent id.
    """
    targets = agent_targets(project)
    if agent_ids is not None:
        selected = _select_explicit_targets(targets, agent_ids)
        if selected is None:
            return 2
        targets = selected

    click.echo("\ncheck-install report:")
    attention = 0
    configured = 0
    for target in targets:
        result = _check_target_launcher(target)
        click.echo(f"  {target.display_name}")
        if result.status == "not-configured":
            click.echo(f"    mcp           {result.status:<13}")
            continue
        configured += 1
        if result.status == "unmanaged":
            click.echo(f"    mcp           unmanaged     {result.config_path}")
        else:
            assert result.executable is not None
            invocation = shlex.join([result.executable, *result.args])
            click.echo(f"    mcp           {result.status:<13} {invocation}")
        if result.detail:
            for line in result.detail.splitlines():
                click.echo(f"      {line}")
        if not result.ok:
            attention += 1
    click.echo("")
    if attention:
        click.echo(
            f"{attention} configured launcher(s) need attention (see above).",
            err=True,
        )
        return 1
    if configured == 0:
        click.echo("no configured phasesweep MCP launchers found.")
    else:
        click.echo("every configured phasesweep MCP launcher resolves.")
    return 0
