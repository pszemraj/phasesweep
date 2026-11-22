"""A running MCP server rereads an edited catalog and its configs before each call."""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import pytest

from phasesweep.mcp.errors import (
    CatalogEntryInvalidError,
    CatalogReloadRefusedError,
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    PermissionDeniedError,
    UnknownExperimentError,
)
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.tool_names import TOOL_LAUNCH_RUN, TOOL_LIST_EXPERIMENTS
from phasesweep.mcp.tools import PhaseSweepMCP
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    call_mcp_tool,
    make_mcp_app,
    patch_popen_capture,
    side_effect_app,
    stage_dead_run,
    stage_live_run,
    write_mcp_catalog,
    write_run_status,
    write_srv_catalog,
    write_srv_config,
)

LOGGER = "phasesweep.mcp.server"
#: The refusal of a catalog whose ``broken`` entry fails, which names it and every launch.
BROKEN_NAMED = (
    r"catalog entry 'broken' failed to load, so launches and inspection of a changed config "
    "are refused"
)


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
    with pytest.raises(ConfigChangedError, match="changed on disk during this call; retry once"):
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


#: How the refusal of each damaging edit begins: it names the entries that fail to load.
_REFUSED = ", so launches and inspection of a changed config are refused"
REFUSALS = {
    "catalog_yaml": f"the catalog file does not validate{_REFUSED}",
    "invalid_config": f"catalog entry 'srv' failed to load{_REFUSED}",
    "namespace_alias": f"catalog entry 'alias' failed to load{_REFUSED}",
    "missing_config": f"catalog entry 'srv' failed to load{_REFUSED}",
}


@pytest.mark.parametrize("damage", REFUSALS)
def test_invalid_edit_keeps_reads_and_refuses_launch_until_fixed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, damage: str
) -> None:
    """Reads serve the last valid catalog, launch refuses path-free naming the failed entries, and the refusal logs once."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    config, app, _registry, store = side_effect_app(tmp_path)
    catalog = tmp_path / "srv.catalog.yaml"
    original_catalog, original_config = catalog.read_bytes(), config.read_bytes()
    stage_live_run(store, "srv-live", config, "srv")

    _break_catalog(tmp_path, catalog, config, damage)

    assert _listed_ids(app) == ["srv"]
    listed = app.list_experiments()
    # A refused launch takes its from_phase option with it; cancel still covers the live run.
    assert listed["experiments"][0]["capabilities"] == {
        "launch": False,
        "cancel": True,
        "from_phase": False,
    }
    catalog_status = listed["catalog_status"]
    assert catalog_status["stale"] is True
    assert catalog_status["reload_refusal"] == "invalid"
    assert catalog_status["remediation"].startswith(REFUSALS[damage])
    assert "`phasesweep mcp check --catalog CATALOG`" in catalog_status["remediation"]
    latest = app.latest_run("srv")
    assert latest["run"] is not None and latest["run"]["run_id"] == "srv-live"
    assert latest["catalog_status"] == catalog_status
    status = app.status(run_id="srv-live")
    assert status["run"]["state"] == "running"
    assert status["catalog_status"] == catalog_status
    if damage == "invalid_config":
        # The served entry no longer describes the file, so inspection refuses too.
        with pytest.raises(CatalogReloadRefusedError, match=REFUSALS[damage]):
            app.validate("srv")
        refused_inspection = call_mcp_tool(app, "inspect_experiment", {"experiment_id": "srv"})
        assert refused_inspection.isError
        assert "inspection of a changed config" in str(refused_inspection.content)
    else:
        inspected = call_mcp_tool(app, "inspect_experiment", {"experiment_id": "srv"})
        assert not inspected.isError
        assert inspected.structuredContent["experiment_id"] == "srv"
        assert inspected.structuredContent["capabilities"] == {
            "launch": False,
            "cancel": True,
            "from_phase": False,
        }
    with pytest.raises(CatalogReloadRefusedError, match=REFUSALS[damage]) as excinfo:
        app.launch("srv")
    refused = call_mcp_tool(app, TOOL_LAUNCH_RUN, {"experiment_id": "srv"})
    assert refused.isError
    refusal_text = " ".join(getattr(item, "text", "") for item in refused.content)
    assert str(excinfo.value) in refusal_text
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
    assert app.list_experiments()["experiments"][0]["capabilities"]["launch"] is True
    assert len(_messages(caplog, "again matches the loaded catalog")) == 1


def test_startup_with_a_failed_entry_serves_the_rest_and_refuses_launch_until_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One entry failing at startup leaves the others readable, refuses launches, and clears with the fix."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    healthy = write_srv_config(tmp_path)
    failing = write_srv_config(tmp_path, name="broken")
    missing_cwd = tmp_path / "missing-cwd"
    catalog = write_mcp_catalog(
        tmp_path,
        {"srv": healthy, "broken": failing},
        allow=ALLOW_SIDE_EFFECTS,
        cwd={"broken": missing_cwd},
        filename="srv.catalog.yaml",
    )
    registry = Registry.load(catalog, collect_errors=True)
    store = RunStore(registry.state_dir)
    app = PhaseSweepMCP(registry, store)
    assert [failure.experiment_id for failure in registry.failures] == ["broken"]
    patch_popen_capture(monkeypatch)

    # Launch is the first call: an unchanged-looking registry would clear the refusal here.
    with pytest.raises(CatalogReloadRefusedError, match=BROKEN_NAMED) as refused:
        app.launch("srv")
    listed = app.list_experiments()
    assert _listed_ids(app) == ["srv"]
    assert listed["experiments"][0]["capabilities"]["launch"] is False
    assert listed["catalog_status"]["stale"] is True
    assert listed["catalog_status"]["reload_refusal"] == "invalid"
    assert listed["catalog_status"]["remediation"] == str(refused.value)
    with pytest.raises(CatalogEntryInvalidError, match="failed to load") as invalid:
        app.validate("broken")
    assert call_mcp_tool(app, "inspect_experiment", {"experiment_id": "broken"}).isError
    with pytest.raises(UnknownExperimentError) as unknown:
        app.validate("nope")
    assert type(unknown.value) is UnknownExperimentError
    for message in (str(refused.value), str(invalid.value)):
        assert str(tmp_path) not in message
    # The refusal the server started in carries the same text a reload would, so it logs once.
    assert _messages(caplog, "catalog reload refused") == []

    stage_dead_run(store, "srv-done", healthy, "srv", cleanup_uncertain=False)
    stage_dead_run(store, "broken-done", failing, "broken", cleanup_uncertain=False)
    for run_id in ("srv-done", "broken-done"):
        write_run_status(
            store,
            run_id,
            returncode=1,
            error_class="RuntimeError",
            result_snapshot_state="failed",
        )
    assert app.latest_run("broken")["run"]["run_id"] == "broken-done"
    assert app.latest_run("srv")["run"]["run_id"] == "srv-done"
    assert app.status(run_id="srv-done")["run"]["state"] == "failed"
    assert app.status(run_id="broken-done")["run"]["state"] == "failed"

    # Creating the directory changes no file the registry hashes; the next call still notices.
    missing_cwd.mkdir()
    assert app.list_experiments()["catalog_status"]["stale"] is False
    assert _listed_ids(app) == ["srv", "broken"]
    assert app.validate("broken")["experiment_id"] == "broken"
    assert app.launch("srv")["experiment_id"] == "srv"
    assert len(_messages(caplog, "reloaded catalog")) == 1


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
    assert app.list_experiments()["experiments"][0]["capabilities"]["launch"] is False
    latest = app.latest_run("srv")
    assert latest["run"] is not None and latest["run"]["run_id"] == "srv-live"
    assert registry.state_dir == tmp_path / "state"


def test_granting_launch_takes_effect_on_the_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping allow.launch to true needs no restart."""
    config = write_srv_config(tmp_path)
    app, _registry, store = make_mcp_app(write_srv_catalog(tmp_path, config))
    with pytest.raises(PermissionDeniedError, match=r"\(allow\.launch: true\)"):
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

    def counting_load(
        catalog_path: Path, *, bound_state_dir: Path | None = None, collect_errors: bool = False
    ) -> Registry:
        loads.append(catalog_path)
        return original_load(
            catalog_path, bound_state_dir=bound_state_dir, collect_errors=collect_errors
        )

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
