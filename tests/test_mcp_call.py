"""``phasesweep mcp call``: one tool of a catalog run from the shell through the server's own dispatch."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner, Result

from phasesweep.cli import cli as cli_main
from phasesweep.mcp import MCP_EXTRA_INSTALL_COMMAND
from phasesweep.mcp.tool_names import (
    TOOL_INSPECT_EXPERIMENT,
    TOOL_LAUNCH_RUN,
    TOOL_LIST_EXPERIMENTS,
)
from tests.mcp_helpers import make_mcp_app, mcp_experiment_config_text, write_mcp_config_catalog

if TYPE_CHECKING:
    from mcp.types import CallToolResult

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None, reason="the mcp extra is not installed"
)


def _catalog(tmp_path: Path) -> Path:
    """Write a read-only catalog of two experiments.

    :param Path tmp_path: Directory receiving the configs, catalog and state.
    :return Path: The written catalog.
    """
    return write_mcp_config_catalog(
        tmp_path,
        {
            "srv": mcp_experiment_config_text(tmp_path, name="srv"),
            "other": mcp_experiment_config_text(tmp_path, name="other"),
        },
    )


def _mcp_call(catalog: Path, tool: str, *arguments: str) -> Result:
    """Invoke ``phasesweep mcp call`` with stdout and stderr kept apart.

    :param Path catalog: Catalog to serve.
    :param str tool: Tool name.
    :param str arguments: Optional JSON argument text.
    :return Result: Click's result.
    """
    return CliRunner().invoke(
        cli_main, ["mcp", "call", tool, "--catalog", str(catalog), *arguments]
    )


def _client_result(catalog: Path, tool: str, arguments: dict[str, object]) -> CallToolResult:
    """Dispatch one tool call through the request handler a connected client reaches.

    :param Path catalog: Catalog to serve.
    :param str tool: Tool name.
    :param dict[str, object] arguments: Raw tool arguments, sent unvalidated.
    :return CallToolResult: The call's result, whose ``isError`` and ``content`` the caller inspects.
    """
    from mcp import types

    from phasesweep.mcp.server import build_server

    app, _registry, _store = make_mcp_app(catalog)
    handler = build_server(app)._mcp_server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name=tool, arguments=arguments)
    )
    result = asyncio.run(handler(request)).root
    assert isinstance(result, types.CallToolResult)
    return result


def test_mcp_call_prints_structured_result_as_json(tmp_path: Path) -> None:
    """The printed object is the structured content a client receives, for default and given arguments."""
    catalog = _catalog(tmp_path)

    result = _mcp_call(catalog, TOOL_LIST_EXPERIMENTS)

    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.startswith('{\n  "experiments": [')
    client = _client_result(catalog, TOOL_LIST_EXPERIMENTS, {})
    assert client.isError is False
    assert json.loads(result.stdout) == client.structuredContent
    assert [entry["id"] for entry in json.loads(result.stdout)["experiments"]] == ["srv", "other"]

    limited = _mcp_call(catalog, TOOL_LIST_EXPERIMENTS, '{"limit": 1}')

    assert limited.exit_code == 0
    payload = json.loads(limited.stdout)
    assert [entry["id"] for entry in payload["experiments"]] == ["srv"]
    assert payload["next_cursor"] is not None


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        pytest.param(
            TOOL_INSPECT_EXPERIMENT, {"experiment_id": "missing"}, id="unknown_experiment"
        ),
        pytest.param(TOOL_INSPECT_EXPERIMENT, {"experiment_id": 3}, id="wrong_argument_type"),
        pytest.param("no_such_tool", {}, id="unknown_tool"),
        pytest.param(TOOL_LAUNCH_RUN, {"experiment_id": "srv"}, id="refused_launch"),
    ],
)
def test_mcp_call_tool_error_matches_client_is_error_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, arguments: dict[str, object]
) -> None:
    """A tool error exits 1 with exactly the text a client receives in its ``isError`` result."""
    # A launch outside the initial PID namespace also warns; keep stderr to the error text.
    monkeypatch.setattr("phasesweep.mcp.server.in_initial_pid_namespace", lambda: True)
    catalog = _catalog(tmp_path)
    client = _client_result(catalog, tool, arguments)
    assert client.isError is True

    result = _mcp_call(catalog, tool, json.dumps(arguments))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.strip() == client.content[0].text


def test_mcp_call_rejects_undeclared_argument(tmp_path: Path) -> None:
    """The tool's strict input schema applies, so an undeclared key is refused rather than ignored."""
    result = _mcp_call(
        _catalog(tmp_path), TOOL_INSPECT_EXPERIMENT, '{"experiment_id": "srv", "x": 1}'
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Extra inputs are not permitted" in result.stderr


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("[1]", id="array"),
        pytest.param('"srv"', id="string"),
        pytest.param("3", id="number"),
        pytest.param("null", id="null"),
        pytest.param("{not json", id="malformed"),
        pytest.param('{"limit": 1, "limit": 2}', id="duplicate_key"),
        pytest.param('{"limit": NaN}', id="non_standard_constant"),
    ],
)
def test_mcp_call_rejects_non_object_json(tmp_path: Path, text: str) -> None:
    """Anything but a strict JSON object exits 2 before the catalog is loaded."""
    catalog = _catalog(tmp_path)

    result = _mcp_call(catalog, TOOL_LIST_EXPERIMENTS, text)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value for '[JSON_ARGS]'" in result.stderr
    assert not (tmp_path / "state").exists()


def _audit_records(tmp_path: Path) -> list[dict[str, object]]:
    """Read the audit log of the catalog written by ``_catalog``.

    :param Path tmp_path: Directory holding the catalog's ``state`` directory.
    :return list[dict[str, object]]: One decoded record per line, oldest first.
    """
    return [
        json.loads(line) for line in (tmp_path / "state" / "audit.jsonl").read_text().splitlines()
    ]


def test_mcp_call_audits_refused_launch(tmp_path: Path) -> None:
    """A launch the catalog forbids is refused and recorded like one from a connected client."""
    catalog = _catalog(tmp_path)

    result = _mcp_call(catalog, TOOL_LAUNCH_RUN, '{"experiment_id": "srv"}')

    assert result.exit_code == 1
    records = _audit_records(tmp_path)
    assert len(records) == 1
    assert records[0]["tool"] == TOOL_LAUNCH_RUN
    assert records[0]["outcome"] == "error"
    assert records[0]["error_type"] == "PermissionDeniedError"
    assert records[0]["args"] == {"experiment_id": "srv"}


def test_audit_records_tell_a_cli_call_from_a_served_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mcp call`` records ``cli`` and ``local-cli``; a served call keeps ``stdio`` and ``local-stdio``."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError

    from phasesweep.mcp.server import serve

    catalog = _catalog(tmp_path)

    def run_one_launch(self: FastMCP, transport: str = "stdio") -> None:
        """Stand in for the blocking stdio loop with one client call through the same dispatch."""
        with pytest.raises(ToolError):
            asyncio.run(self.call_tool(TOOL_LAUNCH_RUN, {"experiment_id": "srv"}))

    monkeypatch.setattr(FastMCP, "run", run_one_launch)

    assert serve(catalog) == 0
    assert _mcp_call(catalog, TOOL_LAUNCH_RUN, '{"experiment_id": "srv"}').exit_code == 1

    served, called = _audit_records(tmp_path)
    assert (served["transport"], served["actor"]) == ("stdio", "local-stdio")
    assert (called["transport"], called["actor"]) == ("cli", "local-cli")
    assert served["session_id"] != called["session_id"]


@pytest.mark.parametrize(
    ("tool", "initial", "warns"),
    [
        pytest.param(TOOL_LAUNCH_RUN, False, True, id="launch_in_nested_namespace"),
        pytest.param(TOOL_LAUNCH_RUN, True, False, id="launch_in_initial_namespace"),
        pytest.param(TOOL_LIST_EXPERIMENTS, False, False, id="read_in_nested_namespace"),
    ],
)
def test_mcp_call_warns_when_a_launch_would_not_outlive_its_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, initial: bool, warns: bool
) -> None:
    """Only a launch from outside the initial PID namespace warns; the read-only catalog refuses it unspawned."""
    monkeypatch.setattr("phasesweep.mcp.server.in_initial_pid_namespace", lambda: initial)
    arguments = '{"experiment_id": "srv"}' if tool == TOOL_LAUNCH_RUN else "{}"

    result = _mcp_call(_catalog(tmp_path), tool, arguments)

    assert ("not in the host's initial PID namespace" in result.stderr) is warns
    assert result.exit_code == (1 if tool == TOOL_LAUNCH_RUN else 0)


def test_mcp_call_requires_mcp_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the MCP SDK the call exits 2 with the install command, as ``serve`` does."""
    find_spec = importlib.util.find_spec

    def without_mcp(name: str, package: str | None = None) -> ModuleSpec | None:
        return None if name == "mcp" else find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", without_mcp)

    result = _mcp_call(_catalog(tmp_path), TOOL_LIST_EXPERIMENTS)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "MCP support is not installed" in result.stderr
    assert MCP_EXTRA_INSTALL_COMMAND in result.stderr
    assert not (tmp_path / "state").exists()


def test_mcp_call_catalog_error_exits_2(tmp_path: Path) -> None:
    """A catalog entry that fails startup validation exits 2, as ``serve`` does."""
    catalog = write_mcp_config_catalog(
        tmp_path, {"srv": mcp_experiment_config_text(tmp_path, with_storage=False)}
    )

    result = _mcp_call(catalog, TOOL_LIST_EXPERIMENTS)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith("phasesweep mcp: ")
