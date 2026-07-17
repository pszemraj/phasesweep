"""Optional MCP server layer: a capability-narrowed broker over the engine.

The agent picks an experiment from a human-curated catalog by id and may only
launch a sweep, monitor it, and read the winning hyperparameters. It never
supplies, edits, or sees a ``trial_command``, ``env``, ``storage``, or
``workdir``. Only :mod:`phasesweep.mcp.server` imports the ``mcp`` SDK; every
other module here is SDK-free and unit-testable on its own.
"""

from __future__ import annotations

import functools
import importlib.resources


@functools.cache
def agent_prompt_text(*, strip: bool = False) -> str:
    """Return the canonical packaged instructions for agents using the MCP server.

    :param bool strip: Remove leading and trailing whitespace for MCP prompt responses.
    :return str: Contents of the packaged ``agent_prompt.md`` resource.
    """
    text = (
        importlib.resources.files("phasesweep.mcp")
        .joinpath("agent_prompt.md")
        .read_text(encoding="utf-8")
    )
    return text.strip() if strip else text
