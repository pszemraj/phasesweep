"""A running MCP server rereads an edited catalog and its configs before each call."""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import pytest

from phasesweep.mcp.errors import (
    CatalogReloadRefusedError,
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    PermissionDeniedError,
)
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle
from phasesweep.mcp.tool_names import TOOL_LAUNCH_RUN, TOOL_LIST_EXPERIMENTS
from phasesweep.mcp.tools import PhaseSweepMCP
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    call_mcp_tool,
    make_mcp_app,
    patch_popen_capture,
    side_effect_app,
    stage_live_run,
    write_mcp_catalog,
    write_srv_catalog,
    write_srv_config,
)

LOGGER = "phasesweep.mcp.server"


def _listed_ids(app: PhaseSweepMCP) -> list[str]:
    return [entry["id"] for entry in app.list_experiments()["experiments"]]


def _messages(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    return [record.getMessage() for record in caplog.records if needle in record.getMessage()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_added_entry_is_listed_and_inspected_without_a_restart(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A new entry is visible to the next call, through the tool adapter too, after one reload."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    config = write_srv_config(tmp_path)
    app, _registry, _store = make_mcp_app(write_mcp_catalog(tmp_path, {"srv": config}))
    assert _listed_ids(app) == ["srv"]

    write_mcp_catalog(tmp_path, {"srv": config, "other": write_srv_config(tmp_path, name="other")})

    listed = call_mcp_tool(app, TOOL_LIST_EXPERIMENTS, {})
    assert not listed.isError
    assert listed.structuredContent is not None
    assert [entry["id"] for entry in listed.structuredContent["experiments"]] == ["srv", "other"]
    assert app.validate("other")["experiment_id"] == "other"
    assert _listed_ids(app) == ["srv", "other"]
    assert len(_messages(caplog, "reloaded catalog")) == 1


def test_edited_config_is_inspected_and_launched_with_its_new_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch snapshots and pins the edited config, not the one loaded at startup."""
    config, app, registry, store = side_effect_app(tmp_path)
    captured = patch_popen_capture(monkeypatch)
    startup_sha256 = registry.get("srv").config_sha256
    config.write_text(config.read_text().replace("n_trials: 1", "n_trials: 3"))

    assert app.validate("srv")["phases"][0]["n_trials"] == 3
    result = app.launch("srv")

    handle = store.get(str(result["run_id"]))
    assert handle is not None
    assert handle.config_sha256 == _sha256(config) != startup_sha256
    assert store.config_snapshot_path(handle.run_id).read_bytes() == config.read_bytes()
    cmd = captured["cmd"]
    assert cmd[cmd.index("--config-sha256") + 1] == handle.config_sha256


def test_launch_refuses_a_config_edited_after_its_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit between the call's reread and the spawn refuses that launch; the next call loads it."""
    config, app, _registry, store = side_effect_app(tmp_path)
    captured = patch_popen_capture(monkeypatch)
    edited = config.read_text().replace("python train.py", "python changed.py")
    original_inventory = store.launch_inventory

    def edit_during_launch() -> tuple[list[RunHandle], set[str]]:
        config.write_text(edited)
        return original_inventory()

    monkeypatch.setattr(store, "launch_inventory", edit_during_launch)
    with pytest.raises(ConfigChangedError, match="retry this call once"):
        app.launch("srv")
    assert "cmd" not in captured
    assert store.list_handles() == []

    monkeypatch.setattr(store, "launch_inventory", original_inventory)
    handle = store.get(str(app.launch("srv")["run_id"]))
    assert handle is not None
    assert handle.config_sha256 == hashlib.sha256(edited.encode()).hexdigest()


def _break_catalog(tmp_path: Path, catalog: Path, config: Path, damage: str) -> None:
    """Apply one operator edit that fails catalog validation."""
    if damage == "catalog_yaml":
        catalog.write_text(catalog.read_text() + "  - id: [unterminated\n")
    elif damage == "invalid_config":
        config.write_text(config.read_text() + "unknown_top_level_key: 1\n")
    elif damage == "namespace_alias":
        write_mcp_catalog(
            tmp_path,
            {"srv": config, "alias": config},
            allow=ALLOW_SIDE_EFFECTS,
            filename=catalog.name,
        )
    else:
        write_mcp_catalog(
            tmp_path,
            {"srv": tmp_path / "missing.yaml"},
            allow=ALLOW_SIDE_EFFECTS,
            filename=catalog.name,
        )


@pytest.mark.parametrize(
    "damage", ["catalog_yaml", "invalid_config", "namespace_alias", "missing_config"]
)
def test_invalid_edit_keeps_reads_and_refuses_launch_until_fixed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, damage: str
) -> None:
    """Reads serve the last valid catalog, launch refuses path-free, and the refusal logs once."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    config, app, _registry, store = side_effect_app(tmp_path)
    catalog = tmp_path / "srv.catalog.yaml"
    original_catalog, original_config = catalog.read_bytes(), config.read_bytes()
    stage_live_run(store, "srv-live", config, "srv")

    _break_catalog(tmp_path, catalog, config, damage)

    assert _listed_ids(app) == ["srv"]
    catalog_status = app.list_experiments()["catalog_status"]
    assert catalog_status["stale"] is True
    assert catalog_status["reload_refusal"] == "invalid"
    assert "phasesweep mcp check" in catalog_status["remediation"]
    latest = app.latest_run("srv")
    assert latest["run"] is not None and latest["run"]["run_id"] == "srv-live"
    assert latest["catalog_status"] == catalog_status
    status = app.status(run_id="srv-live")
    assert status["run"]["state"] == "running"
    assert status["catalog_status"] == catalog_status
    if damage == "invalid_config":
        # The served entry no longer describes the file, so inspection refuses too.
        with pytest.raises(CatalogReloadRefusedError, match="no longer validates"):
            app.validate("srv")
        refused_inspection = call_mcp_tool(app, "inspect_experiment", {"experiment_id": "srv"})
        assert refused_inspection.isError
        assert "inspection of changed configs" in str(refused_inspection.content)
    else:
        assert app.validate("srv")["experiment_id"] == "srv"
    with pytest.raises(CatalogReloadRefusedError, match="no longer validates") as excinfo:
        app.launch("srv")
    refused = call_mcp_tool(app, TOOL_LAUNCH_RUN, {"experiment_id": "srv"})
    assert refused.isError
    refusal_text = " ".join(getattr(item, "text", "") for item in refused.content)
    assert "phasesweep mcp check" in refusal_text
    for message in (str(excinfo.value), refusal_text):
        assert str(tmp_path) not in message
    warnings = _messages(caplog, "catalog reload refused")
    assert len(warnings) == 1
    assert str(catalog) in warnings[0]

    catalog.write_bytes(original_catalog)
    config.write_bytes(original_config)
    with pytest.raises(ExperimentBusyError):
        app.launch("srv")
    assert app.list_experiments()["catalog_status"]["stale"] is False
    assert len(_messages(caplog, "again matches the loaded catalog")) == 1


def test_changed_state_dir_refuses_launch_and_keeps_the_original_store(tmp_path: Path) -> None:
    """A catalog naming another state_dir is not adopted, created or partly applied."""
    config, app, registry, store = side_effect_app(tmp_path)
    stage_live_run(store, "srv-live", config, "srv")
    catalog = tmp_path / "srv.catalog.yaml"
    moved = tmp_path / "moved-state"
    catalog.write_text(
        catalog.read_text().replace(f"state_dir: {tmp_path}/state", f"state_dir: {moved}")
        + f"  - id: other\n    config: {write_srv_config(tmp_path, name='other')}\n"
    )

    with pytest.raises(CatalogReloadRefusedError, match="restarts it") as excinfo:
        app.launch("srv")

    assert str(tmp_path) not in str(excinfo.value)
    assert not moved.exists()
    assert _listed_ids(app) == ["srv"]
    latest = app.latest_run("srv")
    assert latest["run"] is not None and latest["run"]["run_id"] == "srv-live"
    assert registry.state_dir == tmp_path / "state"


def test_granting_launch_takes_effect_on_the_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping allow.launch to true needs no restart."""
    config = write_srv_config(tmp_path)
    app, _registry, store = make_mcp_app(write_srv_catalog(tmp_path, config))
    with pytest.raises(PermissionDeniedError, match="applies on the next call"):
        app.launch("srv")

    write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    patch_popen_capture(monkeypatch)
    handle = store.get(str(app.launch("srv")["run_id"]))

    assert handle is not None
    assert handle.allow_cancel is True


def test_reloaded_max_concurrent_runs_applies_to_the_next_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {name: write_srv_config(tmp_path, name=name) for name in ("a", "b")}
    catalog = write_mcp_catalog(tmp_path, configs, allow=ALLOW_SIDE_EFFECTS, max_concurrent_runs=1)
    app, _registry, store = make_mcp_app(catalog)
    stage_live_run(store, "a-live", configs["a"], "a")
    patch_popen_capture(monkeypatch)
    with pytest.raises(ConcurrencyLimitError, match="max_concurrent_runs=1"):
        app.launch("b")

    write_mcp_catalog(tmp_path, configs, allow=ALLOW_SIDE_EFFECTS, max_concurrent_runs=2)

    assert app.launch("b")["experiment_id"] == "b"


def test_concurrent_calls_share_one_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calls racing on an edited catalog reload it once and never see a partial registry."""
    config = write_srv_config(tmp_path)
    app, _registry, _store = make_mcp_app(write_mcp_catalog(tmp_path, {"srv": config}))
    write_mcp_catalog(tmp_path, {"srv": config, "other": write_srv_config(tmp_path, name="other")})
    original_load = Registry.load
    loads: list[Path] = []

    def counting_load(catalog_path: Path, *, bound_state_dir: Path | None = None) -> Registry:
        loads.append(catalog_path)
        return original_load(catalog_path, bound_state_dir=bound_state_dir)

    monkeypatch.setattr(Registry, "load", staticmethod(counting_load))
    barrier = threading.Barrier(8)
    seen: list[list[str]] = []

    def list_once() -> None:
        barrier.wait()
        seen.append(_listed_ids(app))

    threads = [threading.Thread(target=list_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == [["srv", "other"]] * 8
    assert len(loads) == 1
