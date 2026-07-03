"""Optional MCP server layer: a capability-narrowed broker over the engine.

The agent picks an experiment from a human-curated catalog by id and may only
launch a sweep, monitor it, and read the winning hyperparameters. It never
supplies, edits, or sees a ``trial_command``, ``env``, ``storage``, or
``workdir``. Only :mod:`phasesweep.mcp.server` imports the ``mcp`` SDK; every
other module here is SDK-free and unit-testable on its own.
"""
