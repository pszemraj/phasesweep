"""Client-config installer for ``phasesweep mcp install`` / ``phasesweep mcp uninstall``.

Operator-facing tooling that wires the phasesweep MCP server into coding-agent
configs (MCP server entries plus marker-fenced instructions blocks). Nothing
here runs on the agent side of the MCP trust boundary; printed output and
written config files may contain real paths because the operator owns both.
"""
