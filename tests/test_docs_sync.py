"""The setup doc's embedded agent instructions must match the packaged MCP prompt."""

from __future__ import annotations

import re

from phasesweep.mcp.server import _run_and_monitor_prompt_text
from tests.conftest import REPO


def _embedded_instruction_block() -> str:
    doc = (REPO / "docs" / "mcp_setup.md").read_text(encoding="utf-8")
    _, marker, after = doc.partition("Instruct the agent")
    assert marker, "docs/mcp_setup.md lost its 'Instruct the agent' step"
    match = re.search(r"```text\n(.*?)```", after, re.DOTALL)
    assert match, "docs/mcp_setup.md step 5 lost its fenced instruction block"
    return match.group(1)


def _normalize(text: str) -> str:
    # The doc block renders inside a plain-text fence, so it drops the prompt's
    # inline-code backticks (quoting error strings instead) and uses an ASCII
    # hyphen where the prompt has an en dash. Erase exactly those presentation
    # differences; every remaining character must match.
    text = text.replace("`", "").replace('"', "").replace("–", "-")
    return " ".join(text.split())


def test_docs_instruction_block_matches_packaged_prompt() -> None:
    """docs/mcp_setup.md and agent_prompt.md can never drift apart."""
    assert _normalize(_embedded_instruction_block()) == _normalize(_run_and_monitor_prompt_text())
