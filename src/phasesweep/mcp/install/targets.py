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
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SERVER_NAME = "phasesweep"

MARKDOWN_START = "<!-- PHASESWEEP_START -->"
MARKDOWN_END = "<!-- PHASESWEEP_END -->"
TOML_START = "# PHASESWEEP_START (managed by `phasesweep install`)"
TOML_END = "# PHASESWEEP_END"

Scope = Literal["project", "user"]
EntryStyle = Literal["stdio", "stdio-typed", "opencode"]

AGENT_IDS: tuple[str, ...] = (
    "claude",
    "claude-desktop",
    "codex",
    "cursor",
    "vscode",
    "gemini",
    "opencode",
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
    mcp: McpSpec | None
    instructions_path: Path | None
    instructions_scope: Scope = "project"

    def is_detected(self) -> bool:
        """Whether the client appears installed on this machine.

        :return bool: True when the client binary is on ``PATH`` or its
            config directory exists.
        """
        if self.binary and shutil.which(self.binary):
            return True
        return bool(self.config_dir and self.config_dir.exists())


def mcp_entry(style: EntryStyle, command: str, catalog: Path) -> dict[str, object]:
    """Render one client's JSON server entry for the phasesweep server.

    :param EntryStyle style: Client entry dialect.
    :param str command: Absolute ``phasesweep-mcp`` executable path.
    :param Path catalog: Absolute catalog path passed as ``--catalog``.
    :return dict[str, object]: Entry value to store under the server name.
    """
    if style == "opencode":
        return {
            "type": "local",
            "command": [command, "--catalog", str(catalog)],
            "enabled": True,
        }
    entry: dict[str, object] = {"command": command, "args": ["--catalog", str(catalog)]}
    if style == "stdio-typed":
        return {"type": "stdio", **entry}
    return entry


def codex_toml_content(command: str, catalog: Path) -> str:
    """Render the Codex ``config.toml`` table body for the phasesweep server.

    :param str command: Absolute ``phasesweep-mcp`` executable path.
    :param Path catalog: Absolute catalog path passed as ``--catalog``.
    :return str: TOML table text placed between the installer markers.
    """
    # JSON and TOML basic strings share escaping for valid Unicode. Keeping
    # non-ASCII characters literal avoids JSON's non-BMP UTF-16 surrogate pairs,
    # which TOML rejects because each \u escape must be a Unicode scalar value.
    return (
        f"[mcp_servers.{SERVER_NAME}]\n"
        f"command = {json.dumps(command, ensure_ascii=False)}\n"
        f'args = ["--catalog", {json.dumps(str(catalog), ensure_ascii=False)}]'
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
