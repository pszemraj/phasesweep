"""Coding-agent integration targets for the installer.

One :class:`AgentTarget` per supported client, resolved fresh per call so
tests can redirect ``HOME`` and the project directory. Scope policy: MCP
entries and instructions land in the project wherever the client reads them
there (a catalog belongs to one project); clients without a reliable
project-scope config fall back to their user config with a printed notice.

Client config schemas were verified against each client's documentation in
July 2026; quirks worth keeping in mind:

- VS Code keys its map ``servers`` (not ``mcpServers``) and requires
  ``"type": "stdio"``.
- opencode keys its map ``mcp`` and takes the command as one array including
  the arguments, with ``"type": "local"``.
- Codex reads project ``.codex/config.toml`` only in trusted projects, so the
  reliable target is the user-level ``~/.codex/config.toml``.
- Claude Desktop's Linux config path follows XDG convention but is not yet
  documented by the official Linux beta; detection gates on the directory
  actually existing.

Two launcher modes render the same entry shapes with a different executable
and leading argv (review v0.5.15 / item G): the default binds an absolute
``phasesweep-mcp`` path from the running environment; the opt-in ``uvx`` mode
instead writes a pinned ``uvx --from phasesweep[mcp]==<version>
phasesweep-mcp`` invocation that survives that environment being moved or
recreated, at the cost of requiring ``uvx`` on the client's ``PATH`` at
launch time. :func:`is_managed_mcp_entry` recognizes entries written in
either mode so uninstall and re-install stay reversible regardless of which
mode created the entry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SERVER_NAME = "phasesweep"
# PyPI distribution name backing the ``uvx`` pinned launcher (review v0.5.15 / item G).
# Coincides with SERVER_NAME today but is tracked separately since one names the MCP
# server entry and the other names a package on PyPI.
PACKAGE_NAME = "phasesweep"

MARKDOWN_START = "<!-- PHASESWEEP_START -->"
MARKDOWN_END = "<!-- PHASESWEEP_END -->"
TOML_START = "# PHASESWEEP_START (managed by `phasesweep mcp install`)"
TOML_END = "# PHASESWEEP_END"

Scope = Literal["project", "user"]
EntryStyle = Literal["stdio", "stdio-typed", "opencode"]
Launcher = Literal["path", "uvx"]

# Matches the pinned uvx requirement this installer writes, e.g. "phasesweep[mcp]==1.2.3".
_UVX_PIN_PATTERN = re.compile(
    rf"^{re.escape(PACKAGE_NAME)}\[mcp\]==(?P<version>[A-Za-z0-9][A-Za-z0-9.+_-]*)$"
)


@dataclass(frozen=True)
class McpSpec:
    """Where and how one client stores its MCP server map."""

    path: Path
    scope: Scope
    format: Literal["json", "toml"]
    key: str
    style: EntryStyle
    notice: str | None = None


@dataclass(frozen=True)
class AgentTarget:
    """One coding-agent client the installer can configure."""

    id: str
    display_name: str
    binary: str | None
    config_dir: Path | None
    mcp: McpSpec
    instructions_path: Path | None

    def is_detected(self) -> bool:
        """Whether the client appears installed on this machine.

        :return bool: True when the client binary is on ``PATH`` or its
            config directory exists.
        """
        if self.binary and shutil.which(self.binary):
            return True
        return bool(self.config_dir and self.config_dir.exists())


def mcp_entry(
    style: EntryStyle,
    command: str,
    catalog: Path,
    *,
    launcher_args: Sequence[str] = (),
) -> dict[str, object]:
    """Render one client's JSON server entry for the phasesweep server.

    :param EntryStyle style: Client entry dialect.
    :param str command: Launcher executable: an absolute ``phasesweep-mcp`` path
        (default mode), or ``"uvx"`` (pinned uvx launcher mode).
    :param Path catalog: Absolute catalog path passed as ``--catalog``.
    :param Sequence[str] launcher_args: Extra argv placed before ``--catalog``;
        empty for the default mode, the pinned uvx invocation otherwise.
    :return dict[str, object]: Entry value to store under the server name.
    """
    args = [*launcher_args, "--catalog", str(catalog)]
    if style == "opencode":
        return {
            "type": "local",
            "command": [command, *args],
            "enabled": True,
        }
    entry: dict[str, object] = {"command": command, "args": args}
    if style == "stdio-typed":
        return {"type": "stdio", **entry}
    return entry


def is_managed_mcp_entry(style: EntryStyle, value: object) -> bool:
    """Return whether a JSON member has exactly the shape this installer writes.

    Recognizes both launcher modes this installer can write: the default
    absolute-path launcher and the pinned ``uvx`` launcher.

    :param EntryStyle style: Client entry dialect expected for the member.
    :param object value: JSON member value to inspect.
    :return bool: True when the member is an installer-managed phasesweep entry.
    """
    if not isinstance(value, dict):
        return False
    if style == "opencode":
        if set(value) != {"type", "command", "enabled"}:
            return False
        command = value.get("command")
        return (
            value.get("type") == "local"
            and value.get("enabled") is True
            and isinstance(command, list)
            and _is_phasesweep_argv(command)
        )

    expected_keys = {"command", "args"}
    if style == "stdio-typed":
        expected_keys.add("type")
        if value.get("type") != "stdio":
            return False
    if set(value) != expected_keys:
        return False
    args = value.get("args")
    if not isinstance(args, list):
        return False
    return _is_phasesweep_argv([value.get("command"), *args])


def _is_phasesweep_argv(argv: object) -> bool:
    """Return whether a full command argv matches one recognized launcher shape.

    :param object argv: Candidate argv, executable first, as written by ``mcp_entry``.
    :return bool: True for the 3-element absolute-path launcher
        (``phasesweep-mcp --catalog PATH``) or the 6-element pinned uvx launcher
        (``uvx --from phasesweep[mcp]==VERSION phasesweep-mcp --catalog PATH``).
    """
    if not isinstance(argv, list):
        return False
    if len(argv) == 3:
        command, flag, catalog = argv
        return (
            _is_absolute_phasesweep_command(command)
            and flag == "--catalog"
            and _is_absolute_path(catalog)
        )
    if len(argv) == 6:
        command, from_flag, pin, entrypoint, flag, catalog = argv
        return (
            command == "uvx"
            and from_flag == "--from"
            and isinstance(pin, str)
            and _UVX_PIN_PATTERN.match(pin) is not None
            and entrypoint == "phasesweep-mcp"
            and flag == "--catalog"
            and _is_absolute_path(catalog)
        )
    return False


def _is_absolute_phasesweep_command(value: object) -> bool:
    """Return whether ``value`` is an absolute phasesweep MCP executable path.

    :param object value: Candidate command value.
    :return bool: True when the value is an absolute path named ``phasesweep-mcp``.
    """
    return (
        isinstance(value, str)
        and Path(value).is_absolute()
        and Path(value).name == "phasesweep-mcp"
    )


def _is_absolute_path(value: object) -> bool:
    """Return whether ``value`` is a non-empty absolute path string.

    :param object value: Candidate path value.
    :return bool: True when the value is a non-empty absolute path string.
    """
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def codex_toml_content(
    command: str,
    catalog: Path,
    *,
    launcher_args: Sequence[str] = (),
) -> str:
    """Render the Codex ``config.toml`` table body for the phasesweep server.

    :param str command: Launcher executable: an absolute ``phasesweep-mcp`` path
        (default mode), or ``"uvx"`` (pinned uvx launcher mode).
    :param Path catalog: Absolute catalog path passed as ``--catalog``.
    :param Sequence[str] launcher_args: Extra argv placed before ``--catalog``;
        empty for the default mode, the pinned uvx invocation otherwise.
    :return str: TOML table text placed between the installer markers.
    """
    # JSON and TOML basic strings share escaping for valid Unicode. Keeping
    # non-ASCII characters literal avoids JSON's non-BMP UTF-16 surrogate pairs,
    # which TOML rejects because each \u escape must be a Unicode scalar value.
    args = [*launcher_args, "--catalog", str(catalog)]
    args_literal = ", ".join(json.dumps(arg, ensure_ascii=False) for arg in args)
    return (
        f"[mcp_servers.{SERVER_NAME}]\n"
        f"command = {json.dumps(command, ensure_ascii=False)}\n"
        f"args = [{args_literal}]"
    )


def _xdg_config_home(home: Path) -> Path:
    """Return the XDG config base directory.

    :param Path home: Current home directory.
    :return Path: ``$XDG_CONFIG_HOME`` when set, else ``home/.config``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg) if xdg else home / ".config"


def _claude_desktop_config(home: Path) -> Path:
    """Return the Claude Desktop config path for this platform.

    :param Path home: Current home directory.
    :return Path: Platform-specific ``claude_desktop_config.json`` location.
    """
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return _xdg_config_home(home) / "Claude" / "claude_desktop_config.json"


_USER_SCOPE_NOTICE = (
    "user-scoped config: this client will see the phasesweep server from every project"
)


def agent_targets(project: Path) -> list[AgentTarget]:
    """Build the supported agent targets for one project directory.

    :param Path project: Project root anchoring project-scoped paths.
    :return list[AgentTarget]: All supported targets, detected or not.
    """
    home = Path.home()
    xdg = _xdg_config_home(home)
    opencode_dir = xdg / "opencode"
    return [
        AgentTarget(
            id="claude",
            display_name="Claude Code",
            binary="claude",
            config_dir=home / ".claude",
            mcp=McpSpec(project / ".mcp.json", "project", "json", "mcpServers", "stdio"),
            instructions_path=project / "CLAUDE.md",
        ),
        AgentTarget(
            id="claude-desktop",
            display_name="Claude Desktop",
            binary=None,
            config_dir=_claude_desktop_config(home).parent,
            mcp=McpSpec(
                _claude_desktop_config(home),
                "user",
                "json",
                "mcpServers",
                "stdio",
                notice=_USER_SCOPE_NOTICE,
            ),
            instructions_path=None,
        ),
        AgentTarget(
            id="codex",
            display_name="Codex",
            binary="codex",
            config_dir=home / ".codex",
            mcp=McpSpec(
                home / ".codex" / "config.toml",
                "user",
                "toml",
                "mcp_servers",
                "stdio",
                notice=_USER_SCOPE_NOTICE
                + " (Codex only reads project .codex/config.toml in trusted projects)",
            ),
            instructions_path=project / "AGENTS.md",
        ),
        AgentTarget(
            id="cursor",
            display_name="Cursor",
            binary="cursor",
            config_dir=home / ".cursor",
            mcp=McpSpec(project / ".cursor" / "mcp.json", "project", "json", "mcpServers", "stdio"),
            instructions_path=project / "AGENTS.md",
        ),
        AgentTarget(
            id="vscode",
            display_name="VS Code",
            binary="code",
            config_dir=None,
            mcp=McpSpec(
                project / ".vscode" / "mcp.json", "project", "json", "servers", "stdio-typed"
            ),
            instructions_path=project / ".github" / "copilot-instructions.md",
        ),
        AgentTarget(
            id="gemini",
            display_name="Gemini CLI",
            binary="gemini",
            config_dir=home / ".gemini",
            mcp=McpSpec(
                project / ".gemini" / "settings.json", "project", "json", "mcpServers", "stdio"
            ),
            instructions_path=project / "GEMINI.md",
        ),
        AgentTarget(
            id="opencode",
            display_name="opencode",
            binary="opencode",
            config_dir=opencode_dir,
            mcp=McpSpec(project / "opencode.json", "project", "json", "mcp", "opencode"),
            instructions_path=project / "AGENTS.md",
        ),
    ]


def agent_ids() -> tuple[str, ...]:
    """Return supported target identifiers in display order.

    :return tuple[str, ...]: Supported target identifiers.
    """
    return tuple(target.id for target in agent_targets(Path()))
