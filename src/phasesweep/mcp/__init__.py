"""Optional MCP tools for operator-cataloged experiments.

Agents select experiment IDs and receive path-free results under catalog permissions.
Only :mod:`phasesweep.mcp.server` imports the MCP SDK.
"""

from __future__ import annotations

import functools
import importlib.resources

MCP_EXTRA_INSTALL_COMMAND = 'pip install "phasesweep[mcp]"'


@functools.cache
def agent_prompt_text() -> str:
    """Return the canonical packaged instructions for agents using the MCP server.

    :return str: Contents of the packaged ``agent_prompt.md`` resource.
    """
    text = (
        importlib.resources.files("phasesweep.mcp")
        .joinpath("agent_prompt.md")
        .read_text(encoding="utf-8")
    )
    return text.strip()
