"""Client-config installer for ``phasesweep mcp install`` / ``phasesweep mcp uninstall``.

Operator-facing tooling that wires the phasesweep MCP server into coding-agent
configs (MCP server entries plus marker-fenced instructions blocks). Nothing
here runs on the agent side of the MCP trust boundary; printed output and
written config files may contain real paths because the operator owns both.

``install`` writes either an absolute-path launcher (default) or, with
``--launcher uvx``, a pinned ``uvx --from phasesweep[mcp]==<version>``
launcher that survives this environment being moved or recreated (review
v0.5.15 / item G). :func:`~phasesweep.mcp.install.installer.check_install` is
the read-only counterpart: it verifies each configured client's launcher
still resolves and never edits a client file.
"""
