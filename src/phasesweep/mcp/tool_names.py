"""MCP tool names: the identifiers agents call and the audit log records.

The FastMCP adapter registers each tool under one of these names, and the
launch and cancel implementations record the same name in their audit events.
"""

from __future__ import annotations

TOOL_INSPECT_EXPERIMENT = "inspect_experiment"
TOOL_GET_LATEST_RUN = "get_latest_run"
TOOL_GET_RUN_STATUS = "get_run_status"
TOOL_GET_RUN_RESULTS = "get_run_results"
TOOL_LAUNCH_RUN = "launch_run"
TOOL_CANCEL_RUN = "cancel_run"
TOOL_AWAIT_RUN = "await_run"
TOOL_LIST_EXPERIMENTS = "list_experiments"
