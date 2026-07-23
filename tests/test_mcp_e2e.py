"""End-to-end MCP flow: launch a real detached sweep, monitor it to completion,
read winners, and exercise the launch -> cancel -> relaunch cycle.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from phasesweep.mcp import agent_prompt_text
from phasesweep.mcp.server import (
    AWAIT_DEFAULT_TIMEOUT_SECONDS,
    AWAIT_MAX_TIMEOUT_SECONDS,
    AWAIT_MIN_TIMEOUT_SECONDS,
    CATALOG_RESOURCE_URI,
    DEFAULT_LIST_LIMIT,
    PROMPT_RUN_AND_MONITOR,
    TOOL_AWAIT_RUN,
    TOOL_CANCEL_SWEEP,
    TOOL_GET_LATEST_RUN,
    TOOL_GET_STATUS,
    TOOL_GET_WINNERS,
    TOOL_LAUNCH_SWEEP,
    TOOL_LIST_EXPERIMENTS,
    TOOL_VALIDATE_CONFIG,
)
from tests.conftest import copy_fake_train
from tests.mcp_helpers import (
    cancel_mcp_run_quietly,
    make_mcp_app,
    slow_mcp_config_text,
    wait_for_mcp_running_trial,
    write_mcp_config_catalog,
)

ALLOW_SIDE_EFFECTS = {"launch": True, "cancel": True, "from_phase": True}

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="detached runner + cancel rely on POSIX process groups + /proc liveness",
)


def _chained_config(tmp_path: Path) -> str:
    trainer = copy_fake_train(tmp_path)
    return f"""\
experiment: e2e_lm
storage: sqlite:///{tmp_path}/phases.db
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path}/runs
trial_command: "{sys.executable} {trainer} --out {{trial_dir}}/result.json {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
constraints:
  - name: param_bytes
    extractor: {{ type: json, path: result.json, key: param_bytes }}
    max: 16777216
phases:
  - name: depth
    n_trials: 2
    sampler: {{ type: grid }}
    search_space:
      n_layers: {{ type: categorical, choices: [4, 8] }}
  - name: lr
    inherits: [depth]
    n_trials: 3
    sampler: {{ type: tpe, seed: 0 }}
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
"""


def test_list_validate_launch_monitor_winners(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"e2e_lm": _chained_config(tmp_path)},
        allow=ALLOW_SIDE_EFFECTS,
    )
    app, _registry, store = make_mcp_app(catalog)

    # Catalog metadata is path-free and well-formed.
    summaries = app.list_experiments()["experiments"]
    assert summaries[0]["id"] == "e2e_lm"
    assert summaries[0]["phases"] == ["depth", "lr"]
    assert summaries[0]["capabilities"] == {
        "launch": True,
        "cancel": True,
        "resume_from_phase": True,
    }
    structure = app.validate("e2e_lm")
    assert structure["capabilities"] == summaries[0]["capabilities"]
    assert [p["name"] for p in structure["phases"]] == ["depth", "lr"]
    assert structure["phases"][0]["sampler"] == "grid"
    assert structure["phases"][1]["search_space"] == ["lr"]  # keys only

    run_id = app.launch("e2e_lm")["run_id"]
    try:
        while True:
            awaited = asyncio.run(app.await_run(run_id))
            state = awaited["run"]["state"]
            if state in {"succeeded", "failed", "cancelled"}:
                break
        log = store.log_path(run_id).read_text()
        assert state == "succeeded", f"run ended {state}; log:\n{log}"

        winners = app.winners(run_id=run_id)
        assert winners["result_source"] == "frozen_run_snapshot"
        assert winners["metric"]["objective_evidence"] == {
            "kind": "json_envelope",
            "attempt_bound": True,
            "checkpoint_bound": True,
            "evaluation_policy_bound": True,
        }
        assert winners["all_phases_have_winners"] is True
        assert winners["missing_phases"] == []
        phases = winners["phases"]
        assert [p["phase"] for p in phases] == ["depth", "lr"]
        for p in phases:
            assert p["winner_generation"] == "current_generation"
            assert isinstance(p["metric"], float)
            assert p["metric"] == p["metric"]  # not NaN
            assert "params" in p
            assert set(p["params"].values()) == {"<redacted>"}
            assert "effective_overrides" not in p
        # The chained phase reports only its sampled winner params; inherited
        # fixed/effective values stay out of MCP tool output.
        assert set(phases[1]["params"]) == {"lr"}
        for phase_status in awaited["phases"]:
            target = phase_status["target_terminal_trials"]
            assert phase_status["attempts_launched_this_run"] == target
            assert phase_status["terminal_trials_this_run"] == target
            assert phase_status["terminal_trials_before_run"] == 0
            assert phase_status["target_already_satisfied"] is False
    finally:
        cancel_mcp_run_quietly(app, run_id)


def test_launch_then_cancel_then_relaunch(tmp_path: Path) -> None:
    trainer = copy_fake_train(tmp_path)
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"slow": slow_mcp_config_text(tmp_path, trainer=trainer)},
        allow=ALLOW_SIDE_EFFECTS,
    )
    app, _registry, _store = make_mcp_app(catalog)

    run_id = app.launch("slow")["run_id"]
    second_run_id = None
    try:
        got = wait_for_mcp_running_trial(app, run_id, timeout=30)
        assert got == "running", f"expected a running trial, got {got}"
        # A second launch while one is live is refused.
        with pytest.raises(Exception, match="already has a running sweep"):
            app.launch("slow")

        result = app.cancel(run_id)
        assert result["state"] == "cancelled"
        assert result["cleanup_confirmed"] is True

        # Once the prior run is terminal the lock is released and relaunch works.
        second = app.launch("slow")
        second_run_id = second["run_id"]
        assert second["state"] == "running"
    finally:
        cancel_mcp_run_quietly(app, run_id)
        if second_run_id is not None:
            cancel_mcp_run_quietly(app, second_run_id)


def test_restarted_server_rediscovers_and_cancels_running_run(tmp_path: Path) -> None:
    trainer = copy_fake_train(tmp_path)
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"slow": slow_mcp_config_text(tmp_path, trainer=trainer)},
        allow=ALLOW_SIDE_EFFECTS,
    )
    app, _registry, _store = make_mcp_app(catalog)

    run_id = app.launch("slow")["run_id"]
    try:
        assert wait_for_mcp_running_trial(app, run_id, timeout=30) == "running"

        restarted_app, _restarted_registry, _restarted_store = make_mcp_app(catalog)
        status = restarted_app.status(run_id=run_id)
        assert status["run"]["state"] == "running"

        result = restarted_app.cancel(run_id)
        assert result["state"] == "cancelled"
        assert result["cleanup_confirmed"] is True
    finally:
        cancel_mcp_run_quietly(app, run_id)


def test_global_concurrency_cap_serializes_sweeps(tmp_path: Path) -> None:
    trainer = copy_fake_train(tmp_path)
    catalog = write_mcp_config_catalog(
        tmp_path,
        {
            "slowa": slow_mcp_config_text(tmp_path, trainer=trainer, name="slowa"),
            "slowb": slow_mcp_config_text(tmp_path, trainer=trainer, name="slowb"),
        },
        allow=ALLOW_SIDE_EFFECTS,
        filename="multi.catalog.yaml",
    )  # default max_concurrent_runs == 1
    app, _registry, _store = make_mcp_app(catalog)

    run_a = app.launch("slowa")["run_id"]
    run_b = None
    try:
        assert wait_for_mcp_running_trial(app, run_a, timeout=30) == "running"
        # A *different* experiment cannot start while one sweep is live (single-GPU cap).
        with pytest.raises(Exception, match="max_concurrent_runs=1") as exc_info:
            app.launch("slowb")
        assert run_a in str(exc_info.value)
        assert "phasesweep_await_run" in str(exc_info.value)
        # Freeing the slot lets the other experiment launch.
        assert app.cancel(run_a)["state"] == "cancelled"
        result = app.launch("slowb")
        run_b = result["run_id"]
        assert result["state"] == "running"
    finally:
        cancel_mcp_run_quietly(app, run_a)
        if run_b is not None:
            cancel_mcp_run_quietly(app, run_b)


def test_launch_refused_while_launch_lock_held(tmp_path: Path) -> None:
    # Holding the launch lock stands in for a concurrent launch mid-decision.
    # The cap check and spawn are serialized under it, so a second launch is
    # told to retry rather than racing past the cap. The refused path spawns
    # nothing (it raises before _spawn), so no real sweep is needed and the
    # test stays deterministic. White-box: reach for the store's lock directly.
    from phasesweep.runtime.files import try_lock_file, unlock_file

    trainer = copy_fake_train(tmp_path)
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"slow": slow_mcp_config_text(tmp_path, trainer=trainer)},
        allow=ALLOW_SIDE_EFFECTS,
    )
    app, _registry, store = make_mcp_app(catalog)

    held = try_lock_file(store._launch_lock_path)
    assert held is not None
    try:
        with pytest.raises(Exception, match="launch is in progress"):
            app.launch("slow")
    finally:
        unlock_file(held)

    # The lock is advisory and released on unlock: a fresh acquire succeeds.
    regrab = try_lock_file(store._launch_lock_path)
    assert regrab is not None
    unlock_file(regrab)


def test_status_and_cancel_error_paths(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(tmp_path, {"e2e_lm": _chained_config(tmp_path)})
    app, _registry, _store = make_mcp_app(catalog)

    with pytest.raises(Exception, match="unknown run id"):
        app.status(run_id="nope-123")
    with pytest.raises(Exception, match="unknown run id"):
        app.cancel("nope-123")
    with pytest.raises(Exception, match="exactly one of experiment_id or run_id"):
        app.status()
    with pytest.raises(Exception, match="exactly one of experiment_id or run_id"):
        app.status(experiment_id="e2e_lm", run_id="nope-123")
    # No run launched yet: experiment-level status reports no live run.
    assert app.status(experiment_id="e2e_lm")["run"] is None


def test_fastmcp_registers_eight_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from phasesweep.mcp.server import build_server

    catalog = write_mcp_config_catalog(tmp_path, {"e2e_lm": _chained_config(tmp_path)})
    app, _registry, _store = make_mcp_app(catalog)
    server = build_server(app)
    initialization = server._mcp_server.create_initialization_options()
    assert initialization.instructions == agent_prompt_text(strip=True)
    assert "unchanged relaunches do not need another validation call" in initialization.instructions
    assert "reason: recovery_required" in initialization.instructions
    assert "do not claim convergence, trends, robustness" in initialization.instructions
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        TOOL_LIST_EXPERIMENTS,
        TOOL_VALIDATE_CONFIG,
        TOOL_GET_LATEST_RUN,
        TOOL_GET_STATUS,
        TOOL_AWAIT_RUN,
        TOOL_GET_WINNERS,
        TOOL_LAUNCH_SWEEP,
        TOOL_CANCEL_SWEEP,
    }
    assert all(t.description for t in tools)
    assert all(t.annotations is not None for t in tools)
    assert all(t.outputSchema for t in tools)
    for tool_name in (
        TOOL_VALIDATE_CONFIG,
        TOOL_GET_LATEST_RUN,
        TOOL_GET_STATUS,
        TOOL_AWAIT_RUN,
        TOOL_GET_WINNERS,
        TOOL_LAUNCH_SWEEP,
        TOOL_CANCEL_SWEEP,
    ):
        assert server._tool_manager.get_tool(tool_name).is_async is True

    descriptions = {t.name: t.description for t in tools}
    assert (
        "unchanged relaunches do not need another validation call"
        in descriptions[TOOL_VALIDATE_CONFIG]
    )
    assert "independently re-checks config identity" in descriptions[TOOL_LAUNCH_SWEEP]
    assert "If found=false" in descriptions[TOOL_GET_LATEST_RUN]
    assert "run.recovery_required" in descriptions[TOOL_GET_STATUS]
    assert "reason is recovery_required" in descriptions[TOOL_AWAIT_RUN]
    assert "reason is terminal" in descriptions[TOOL_AWAIT_RUN]
    assert "reason is phase_completed" in descriptions[TOOL_AWAIT_RUN]
    assert "not search ranges or non-winning trial history" in descriptions[TOOL_GET_WINNERS]

    # The _safe_tool wrapper (functools.wraps + *args/**kwargs) must not erase
    # the parameter schema FastMCP derives from each signature, or the agent
    # could not call the tools. Lock the shapes in.
    schemas = {t.name: t.inputSchema for t in tools}
    assert all(schema.get("additionalProperties") is False for schema in schemas.values())
    assert sorted(schemas[TOOL_LAUNCH_SWEEP]["properties"]) == ["experiment_id", "from_phase"]
    assert schemas[TOOL_LAUNCH_SWEEP]["required"] == ["experiment_id"]
    assert (
        schemas[TOOL_LAUNCH_SWEEP]["properties"]["experiment_id"]["pattern"] == "^[A-Za-z0-9_-]+$"
    )
    assert schemas[TOOL_LAUNCH_SWEEP]["properties"]["experiment_id"]["description"]
    assert sorted(schemas[TOOL_GET_STATUS]["properties"]) == ["experiment_id", "run_id"]
    assert schemas[TOOL_GET_STATUS].get("required") is None  # both optional
    assert "oneOf" in schemas[TOOL_GET_STATUS]
    assert sorted(schemas[TOOL_GET_WINNERS]["properties"]) == ["experiment_id", "run_id"]
    assert schemas[TOOL_GET_WINNERS].get("required") is None  # both optional
    assert "oneOf" in schemas[TOOL_GET_WINNERS]
    assert schemas[TOOL_CANCEL_SWEEP]["required"] == ["run_id"]
    assert schemas[TOOL_AWAIT_RUN]["required"] == ["run_id"]
    assert sorted(schemas[TOOL_AWAIT_RUN]["properties"]) == ["run_id", "timeout_seconds"]
    timeout_schema = schemas[TOOL_AWAIT_RUN]["properties"]["timeout_seconds"]
    assert timeout_schema["default"] == AWAIT_DEFAULT_TIMEOUT_SECONDS
    assert timeout_schema["minimum"] == AWAIT_MIN_TIMEOUT_SECONDS
    assert timeout_schema["maximum"] == AWAIT_MAX_TIMEOUT_SECONDS
    assert schemas[TOOL_VALIDATE_CONFIG]["required"] == ["experiment_id"]
    assert schemas[TOOL_GET_LATEST_RUN]["required"] == ["experiment_id"]
    assert sorted(schemas[TOOL_LIST_EXPERIMENTS]["properties"]) == ["cursor", "limit"]
    assert schemas[TOOL_LIST_EXPERIMENTS].get("required") is None
    assert schemas[TOOL_LIST_EXPERIMENTS]["properties"]["limit"]["default"] == DEFAULT_LIST_LIMIT
    assert schemas[TOOL_LIST_EXPERIMENTS]["properties"]["limit"]["maximum"] == 100

    annotations = {t.name: t.annotations for t in tools}
    assert annotations[TOOL_LIST_EXPERIMENTS].readOnlyHint is True
    assert annotations[TOOL_AWAIT_RUN].readOnlyHint is True
    assert annotations[TOOL_LAUNCH_SWEEP].readOnlyHint is False
    assert annotations[TOOL_LAUNCH_SWEEP].destructiveHint is True
    assert annotations[TOOL_LAUNCH_SWEEP].openWorldHint is True
    assert annotations[TOOL_CANCEL_SWEEP].destructiveHint is True
    assert annotations[TOOL_CANCEL_SWEEP].idempotentHint is True

    output_schemas = {t.name: t.outputSchema for t in tools}
    assert "changed" in output_schemas[TOOL_AWAIT_RUN]["properties"]
    assert "reason" in output_schemas[TOOL_AWAIT_RUN]["properties"]
    assert "recovery_required" in output_schemas[TOOL_AWAIT_RUN]["properties"]["reason"]["enum"]
    assert "experiments" in output_schemas[TOOL_LIST_EXPERIMENTS]["properties"]
    assert "next_cursor" in output_schemas[TOOL_LIST_EXPERIMENTS]["properties"]
    assert "total_count" in output_schemas[TOOL_LIST_EXPERIMENTS]["properties"]
    assert "found" in output_schemas[TOOL_GET_LATEST_RUN]["properties"]
    assert "run" in output_schemas[TOOL_GET_LATEST_RUN]["properties"]
    assert "effective_overrides" not in json.dumps(output_schemas[TOOL_GET_WINNERS])
    assert "params" in json.dumps(output_schemas[TOOL_GET_WINNERS])

    resources = asyncio.run(server.list_resources())
    assert {str(resource.uri) for resource in resources} == {CATALOG_RESOURCE_URI}
    resource_payload = asyncio.run(server.read_resource(CATALOG_RESOURCE_URI))
    assert "e2e_lm" in str(resource_payload)
    assert "trial_command" not in str(resource_payload)

    prompts = asyncio.run(server.list_prompts())
    assert {prompt.name for prompt in prompts} == {PROMPT_RUN_AND_MONITOR}
    prompt = asyncio.run(server.get_prompt(PROMPT_RUN_AND_MONITOR, {}))
    assert "phasesweep_launch_sweep" in str(prompt)
    assert "target or label columns" in str(prompt)


@pytest.mark.parametrize("blocking_tool", [TOOL_CANCEL_SWEEP, TOOL_LAUNCH_SWEEP])
def test_fastmcp_blocking_tools_do_not_delay_concurrent_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocking_tool: str
) -> None:
    """The real SDK dispatch must keep await responsive during blocking work."""
    pytest.importorskip("mcp")
    from mcp import types

    from phasesweep.mcp.server import build_server
    from phasesweep.mcp.time import utc_now_iso

    catalog = write_mcp_config_catalog(tmp_path, {"e2e_lm": _chained_config(tmp_path)})
    app, _registry, _store = make_mcp_app(catalog)
    awaited = app.status(experiment_id="e2e_lm")
    awaited.update(
        {
            "run": {
                "run_id": "r1",
                "state": "running",
                "started_at": utc_now_iso(),
                "recovery_required": False,
            },
            "elapsed_seconds": 0,
            "changed": False,
            "reason": "timeout",
        }
    )

    async def quick_await(_run_id: str, timeout_seconds: int = 120) -> dict:
        del timeout_seconds
        await asyncio.sleep(0.02)
        return awaited

    def blocking_cancel(run_id: str) -> dict:
        time.sleep(0.3)
        return {
            "run_id": run_id,
            "state": "cancelled",
            "cleanup_confirmed": True,
            "recovery_required": False,
        }

    def blocking_launch(experiment_id: str, from_phase: str | None = None) -> dict:
        del from_phase
        time.sleep(0.3)
        return {"run_id": "r2", "experiment_id": experiment_id, "state": "running"}

    monkeypatch.setattr(app, "await_run", quick_await)
    if blocking_tool == TOOL_CANCEL_SWEEP:
        monkeypatch.setattr(app, "cancel", blocking_cancel)
        blocking_arguments = {"run_id": "r1"}
    else:
        monkeypatch.setattr(app, "launch", blocking_launch)
        blocking_arguments = {"experiment_id": "e2e_lm"}
    server = build_server(app)
    handler = server._mcp_server.request_handlers[types.CallToolRequest]
    await_request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name=TOOL_AWAIT_RUN,
            arguments={"run_id": "r1", "timeout_seconds": AWAIT_MIN_TIMEOUT_SECONDS},
        )
    )
    blocking_request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name=blocking_tool,
            arguments=blocking_arguments,
        )
    )

    async def dispatch_concurrently() -> tuple[float, object, object]:
        started = time.perf_counter()
        await_task = asyncio.create_task(handler(await_request))
        blocking_task = asyncio.create_task(handler(blocking_request))
        await_result = await await_task
        await_elapsed = time.perf_counter() - started
        blocking_result = await blocking_task
        return await_elapsed, await_result.root, blocking_result.root

    await_elapsed, await_result, blocking_result = asyncio.run(dispatch_concurrently())

    assert await_elapsed < 0.15
    assert await_result.isError is False
    assert blocking_result.isError is False


def test_fastmcp_tool_errors_are_is_error_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from mcp import types

    from phasesweep.mcp.server import build_server

    catalog = write_mcp_config_catalog(tmp_path, {"e2e_lm": _chained_config(tmp_path)})
    app, _registry, _store = make_mcp_app(catalog)
    server = build_server(app)
    handler = server._mcp_server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name=TOOL_VALIDATE_CONFIG,
            arguments={"experiment_id": "missing"},
        )
    )

    result = asyncio.run(handler(req)).root

    assert result.isError is True
    assert "unknown experiment id 'missing'" in result.content[0].text

    monkeypatch.setattr(
        app,
        "validate",
        lambda _experiment_id: (_ for _ in ()).throw(
            OSError("/tmp/SECRET_PATH/private-config.yaml")
        ),
    )
    leaked = asyncio.run(
        handler(
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name=TOOL_VALIDATE_CONFIG,
                    arguments={"experiment_id": "e2e_lm"},
                )
            )
        )
    ).root

    assert leaked.isError is True
    assert "internal server error" in leaked.content[0].text
    assert "SECRET_PATH" not in leaked.content[0].text


def test_fastmcp_rejects_extra_tool_arguments(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from mcp import types

    from phasesweep.mcp.server import build_server

    catalog = write_mcp_config_catalog(tmp_path, {"e2e_lm": _chained_config(tmp_path)})
    app, _registry, _store = make_mcp_app(catalog)
    server = build_server(app)
    handler = server._mcp_server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name=TOOL_VALIDATE_CONFIG,
            arguments={"experiment_id": "e2e_lm", "unexpected": True},
        )
    )

    result = asyncio.run(handler(req)).root

    assert result.isError is True
    assert "unexpected" in result.content[0].text
