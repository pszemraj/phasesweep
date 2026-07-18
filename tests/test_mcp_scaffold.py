"""Catalog scaffolding: derive_experiment_id, scaffold_catalog_text, and the
``mcp init-catalog`` CLI command (validate-before-write, refusal paths)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import phasesweep.cli as cli_module
from phasesweep.cli import main as cli_main
from phasesweep.mcp.errors import CatalogError
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.scaffold import derive_experiment_id, scaffold_catalog_text
from tests.mcp_helpers import mcp_experiment_config_text


def _write_config(tmp_path: Path, filename: str, *, name: str = "srv") -> Path:
    config = tmp_path / filename
    config.write_text(mcp_experiment_config_text(tmp_path, name=name))
    return config


def test_derive_experiment_id_sanitizes_stem(tmp_path: Path) -> None:
    assert derive_experiment_id(Path("my experiment!!.yaml")) == "my-experiment"
    assert derive_experiment_id(Path("tiny_lm-v2.yaml")) == "tiny_lm-v2"
    with pytest.raises(CatalogError, match="cannot derive"):
        derive_experiment_id(Path("!!!.yaml"))


def test_scaffold_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = tmp_path / "a" / "srv.yaml"
    second = tmp_path / "b" / "srv.yaml"
    with pytest.raises(CatalogError, match="both derive"):
        scaffold_catalog_text(tmp_path / "catalog.yaml", [first, second])


def test_init_catalog_writes_validated_read_only_catalog(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "srv  ok    (read-only)" in result.output
    assert f"wrote {output}" in result.output

    text = output.read_text()
    assert f'state_dir: "{tmp_path}/runs/.mcp"' in text
    assert 'config: "./srv.yaml"' in text
    assert "visible_params: none" in text
    assert "TODO" not in text
    assert '# description: "Human-curated one-line purpose shown to the agent"' in text
    assert "\n    allow:" not in text  # side effects stay commented out

    # The scaffold must boot the real server loader as-is, read-only.
    registry = Registry.load(output)
    entry = registry.get("srv")
    assert entry.allow_launch is False
    assert entry.allow_cancel is False
    assert entry.visible_params == "none"
    assert entry.description == ""


@pytest.mark.parametrize("filename", ["model # 1.yaml", "model: 1.yaml", "model\n1.yaml"])
def test_init_catalog_quotes_config_paths(tmp_path: Path, filename: str) -> None:
    config = _write_config(tmp_path, filename)
    output = tmp_path / "catalog.yaml"

    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    entry = Registry.load(output).get(derive_experiment_id(config))
    assert entry.config_path == config.resolve()


@pytest.mark.parametrize("filename", ["null.yaml", "true.yaml", "123.yaml"])
def test_init_catalog_quotes_implicit_yaml_scalar_ids(tmp_path: Path, filename: str) -> None:
    config = _write_config(tmp_path, filename)
    output = tmp_path / "catalog.yaml"

    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    experiment_id = config.stem
    assert f'- id: "{experiment_id}"' in output.read_text()
    assert Registry.load(output).get(experiment_id).id == experiment_id


def test_init_catalog_quotes_state_dir_with_yaml_punctuation(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    catalog_dir = tmp_path / "project # one"
    catalog_dir.mkdir()
    output = catalog_dir / "catalog.yaml"

    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert Registry.load(output).state_dir == (catalog_dir / "runs" / ".mcp").resolve()


def test_init_catalog_failure_writes_nothing(tmp_path: Path) -> None:
    config = tmp_path / "rel.yaml"
    config.write_text(
        mcp_experiment_config_text(tmp_path, name="rel").replace(
            f"workdir: {tmp_path}/runs/rel", "workdir: runs/relative"
        )
    )
    output = tmp_path / "catalog.yaml"
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 2
    assert "FAIL" in result.output
    assert "catalog destination was not written" in result.output
    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()
    assert not (tmp_path / "runs").exists()


def test_init_catalog_refuses_to_overwrite(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    output.write_text("operator-authored\n")
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "operator-authored\n"


def test_init_catalog_does_not_replace_destination_created_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    real_check_catalog = cli_module.check_catalog

    def check_after_late_arrival(staged: Path):
        report = real_check_catalog(staged)
        output.write_text("created concurrently\n")
        return report

    monkeypatch.setattr(cli_module, "check_catalog", check_after_late_arrival)
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "created concurrently\n"
    assert list(tmp_path.glob(".catalog.yaml.*.tmp")) == []


def test_init_catalog_requires_at_least_one_config(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli_main, ["mcp", "init-catalog", "-o", str(tmp_path / "c.yaml")])
    assert result.exit_code != 0
    assert "--from" in result.output


def test_init_catalog_multiple_configs(tmp_path: Path) -> None:
    alpha = _write_config(tmp_path, "alpha.yaml", name="alpha")
    beta = _write_config(tmp_path, "beta.yaml", name="beta")
    output = tmp_path / "catalog.yaml"
    result = CliRunner().invoke(
        cli_main,
        ["mcp", "init-catalog", "--from", str(alpha), "--from", str(beta), "-o", str(output)],
    )
    assert result.exit_code == 0, result.output
    registry = Registry.load(output)
    assert registry.get("alpha").id == "alpha"
    assert registry.get("beta").id == "beta"
