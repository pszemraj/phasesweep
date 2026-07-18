"""Catalog preflight: check_catalog and the ``mcp check`` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep.cli import main as cli_main
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.registry import Registry, check_catalog
from tests.mcp_helpers import mcp_experiment_config_text, write_mcp_config_catalog


def _relative_workdir_config(tmp_path: Path, name: str) -> str:
    return mcp_experiment_config_text(tmp_path, name=name).replace(
        f"workdir: {tmp_path}/runs/{name}", "workdir: runs/relative"
    )


def _relative_storage_config(tmp_path: Path, name: str) -> str:
    return mcp_experiment_config_text(tmp_path, name=name).replace(
        f"storage: sqlite:///{tmp_path}/{name}.db", "storage: sqlite:///relative.db"
    )


def test_check_catalog_reports_every_entry_ok(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {
            "alpha": mcp_experiment_config_text(tmp_path, name="alpha"),
            "beta": mcp_experiment_config_text(tmp_path, name="beta"),
        },
        allow={"launch": True, "cancel": True},
    )
    report = check_catalog(catalog)
    assert report.ok
    assert [entry.experiment_id for entry in report.entries] == ["alpha", "beta"]
    assert all(entry.actions == ("launch", "cancel") for entry in report.entries)
    assert not (tmp_path / "state").exists()
    # The shared code path means a green report implies a bootable server.
    Registry.load(catalog)
    assert (tmp_path / "state" / "runs").is_dir()
    assert (tmp_path / "state" / "logs").is_dir()


def test_check_catalog_read_only_entry_has_no_actions(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path, {"quiet": mcp_experiment_config_text(tmp_path, name="quiet")}
    )
    report = check_catalog(catalog)
    assert report.ok
    assert report.entries[0].actions == ()


def test_check_catalog_does_not_create_or_chmod_state_layout(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path, {"quiet": mcp_experiment_config_text(tmp_path, name="quiet")}
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o755)
    mode_before = state_dir.stat().st_mode & 0o777

    report = check_catalog(catalog)

    assert report.ok
    assert state_dir.stat().st_mode & 0o777 == mode_before
    assert not (state_dir / "runs").exists()
    assert not (state_dir / "logs").exists()


def test_check_catalog_collects_failures_past_the_first(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {
            "bad-workdir": _relative_workdir_config(tmp_path, "w"),
            "good": mcp_experiment_config_text(tmp_path, name="good"),
            "bad-storage": _relative_storage_config(tmp_path, "s"),
        },
    )
    report = check_catalog(catalog)
    verdicts = {entry.experiment_id: entry for entry in report.entries}
    assert not report.ok
    assert verdicts["good"].ok
    assert "absolute workdir" in (verdicts["bad-workdir"].error or "")
    assert "set workdir to an absolute path" in (verdicts["bad-workdir"].suggestion or "")
    assert "absolute sqlite storage path" in (verdicts["bad-storage"].error or "")
    assert "use an absolute path" in (verdicts["bad-storage"].suggestion or "")
    assert not (tmp_path / "state").exists()
    # Registry.load stays fail-fast on the same catalog.
    with pytest.raises(CatalogError, match="absolute workdir"):
        Registry.load(catalog)


def test_check_catalog_flags_duplicate_and_missing_config(tmp_path: Path) -> None:
    config = tmp_path / "one.yaml"
    config.write_text(mcp_experiment_config_text(tmp_path, name="one"))
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        f"""\
state_dir: {tmp_path}/state
experiments:
  - id: twin
    config: {config}
  - id: twin
    config: {config}
  - id: ghost
    config: {tmp_path}/missing.yaml
"""
    )
    report = check_catalog(catalog)
    assert not report.ok
    duplicate = report.entries[1]
    ghost = report.entries[2]
    assert report.entries[0].ok
    assert "duplicate catalog id" in (duplicate.error or "")
    assert "config not found" in (ghost.error or "")


def test_check_catalog_raises_on_catalog_level_error(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("experiments: []\n")
    with pytest.raises(CatalogError, match="catalog"):
        check_catalog(catalog)


def test_mcp_catalog_rejected_off_linux_before_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"tiny": mcp_experiment_config_text(tmp_path, name="tiny")},
    )
    monkeypatch.setattr("phasesweep.mcp.registry.sys.platform", "darwin")

    with pytest.raises(CatalogError, match="supported only on Linux"):
        check_catalog(catalog)
    with pytest.raises(CatalogError, match="supported only on Linux"):
        Registry.load(catalog)

    assert not (tmp_path / "state").exists()


def test_mcp_catalog_rejected_when_proc_identity_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"tiny": mcp_experiment_config_text(tmp_path, name="tiny")},
    )
    monkeypatch.setattr("phasesweep.mcp.registry.read_proc_starttime", lambda _pid: None)

    with pytest.raises(CatalogError, match="cannot read this process's Linux /proc start time"):
        check_catalog(catalog)

    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("blocked_path", ["state", "state/runs", "state/logs"])
def test_check_catalog_rejects_unusable_state_layout(tmp_path: Path, blocked_path: str) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"tiny": mcp_experiment_config_text(tmp_path, name="tiny")},
    )
    blocked = tmp_path / blocked_path
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("not a directory\n")

    with pytest.raises(CatalogError, match="state_dir is not usable") as exc_info:
        check_catalog(catalog)

    assert "writable directory path" in (exc_info.value.suggestion or "")
    with pytest.raises(CatalogError, match="state_dir is not usable"):
        Registry.load(catalog)


def test_check_catalog_rejects_existing_read_only_state_directory(tmp_path: Path) -> None:
    catalog = write_mcp_config_catalog(
        tmp_path,
        {"tiny": mcp_experiment_config_text(tmp_path, name="tiny")},
    )
    Registry.load(catalog)
    logs_dir = tmp_path / "state" / "logs"
    logs_dir.chmod(0o500)

    try:
        with pytest.raises(CatalogError, match="state_dir is not usable"):
            check_catalog(catalog)
    finally:
        logs_dir.chmod(0o700)


def test_mcp_check_cli_exit_codes_and_table(tmp_path: Path) -> None:
    runner = CliRunner()
    good = write_mcp_config_catalog(
        tmp_path,
        {"tiny": mcp_experiment_config_text(tmp_path, name="tiny")},
        allow={"launch": True},
        filename="good.yaml",
    )
    ok_result = runner.invoke(cli_main, ["mcp", "check", "--catalog", str(good)])
    assert ok_result.exit_code == 0
    assert "tiny" in ok_result.output
    assert "ok" in ok_result.output
    assert "(launch)" in ok_result.output
    assert (tmp_path / "state" / "runs").is_dir()
    assert (tmp_path / "state" / "logs").is_dir()

    bad = write_mcp_config_catalog(
        tmp_path,
        {
            "tiny2": mcp_experiment_config_text(tmp_path, name="tiny2"),
            "rel": _relative_workdir_config(tmp_path, "rel"),
        },
        filename="bad.yaml",
    )
    fail_result = runner.invoke(cli_main, ["mcp", "check", "--catalog", str(bad)])
    assert fail_result.exit_code == 2
    assert "FAIL" in fail_result.output
    assert "fix:" in fail_result.output
    # The good entry is still reported ok even though a sibling failed.
    assert "tiny2" in fail_result.output


def test_mcp_check_cli_catalog_level_failure_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    broken = tmp_path / "broken.yaml"
    broken.write_text("not: [valid catalog\n")
    result = runner.invoke(cli_main, ["mcp", "check", "--catalog", str(broken)])
    assert result.exit_code == 2
