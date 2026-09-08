"""Catalog scaffolding and the ``mcp init-catalog`` CLI command."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from phasesweep.cli import cli as cli_main
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


def test_init_catalog_writes_read_only_catalog(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert f"wrote {output}" in result.output

    # The scaffold boots the real server loader as-is, read-only.
    registry = Registry.load(output)
    entry = registry.get("srv")
    digest = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    assert registry.state_dir == tmp_path / "state-home" / "phasesweep" / "mcp" / digest
    assert (registry.state_dir / "runs").is_dir()
    assert (registry.state_dir / "logs").is_dir()
    assert (registry.state_dir / "origin").read_text() == str(output.resolve()) + "\n"
    assert stat.S_IMODE(registry.state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((registry.state_dir / "origin").stat().st_mode) == 0o600
    assert not (tmp_path / "runs").exists()
    assert entry.config_path == config
    assert entry.cwd == output.parent.resolve()
    assert entry.allow_launch is False
    assert entry.allow_cancel is False
    assert entry.visible_params == "none"
    assert entry.description == ""


def test_init_catalog_pins_runner_cwd_to_catalog_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    config = _write_config(configs, "srv.yaml")
    output = project / "catalog.yaml"

    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert Registry.load(output).get("srv").cwd == project.resolve()
    text = output.read_text()
    assert 'cwd: "."' in text
    assert f"phasesweep mcp check --catalog {output}" in text


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


def test_init_catalog_quotes_state_dir_with_yaml_punctuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    catalog_dir = tmp_path / "project # one"
    catalog_dir.mkdir()
    output = catalog_dir / "catalog.yaml"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state # one"))

    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert Registry.load(output).state_dir.parent == tmp_path / "state # one" / "phasesweep" / "mcp"


@pytest.mark.parametrize("xdg", [None, "", "relative"])
def test_scaffold_state_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str | None
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if xdg is None:
        monkeypatch.delenv("XDG_STATE_HOME")
    else:
        monkeypatch.setenv("XDG_STATE_HOME", xdg)
    state = Path(yaml.safe_load(scaffold_catalog_text(tmp_path / "catalog.yaml", []))["state_dir"])
    assert state.parent == tmp_path / ".local" / "state" / "phasesweep" / "mcp"


def test_catalog_filenames_have_distinct_state(tmp_path: Path) -> None:
    states = [
        yaml.safe_load(scaffold_catalog_text(tmp_path / name, []))["state_dir"]
        for name in ("one.yaml", "two.yaml")
    ]
    assert states[0] != states[1]


def test_home_override_pins_scaffold_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "private-root"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("PHASESWEEP_HOME", str(root))
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    state = Registry.load(output).state_dir
    assert state.parent == root / "mcp"
    monkeypatch.setenv("PHASESWEEP_HOME", "relative-but-unused")
    assert Registry.load(output).state_dir == state
    assert (state / "origin").read_text() == str(output.resolve()) + "\n"


def test_scaffold_rejects_invalid_home_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHASESWEEP_HOME", "relative")
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 2
    assert "PHASESWEEP_HOME must be an absolute path" in result.output
    assert not output.exists()


def test_scaffold_refuses_unsafe_state(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"
    state = Path(yaml.safe_load(scaffold_catalog_text(output, [config]))["state_dir"])
    state.mkdir(parents=True)
    state.chmod(0o755)
    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )
    assert result.exit_code == 2
    assert not output.exists()
    assert stat.S_IMODE(state.stat().st_mode) == 0o755


def test_init_catalog_validates_before_publish(tmp_path: Path) -> None:
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
    assert "absolute workdir" in result.output
    assert not output.exists()
    assert not list(tmp_path.glob(".catalog.yaml.*.tmp"))
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


def test_init_catalog_losing_publish_race_preserves_other_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, "srv.yaml")
    output = tmp_path / "catalog.yaml"

    def lose_publish_race(_source: Path, destination: Path) -> None:
        destination.write_text("other process\n")
        raise FileExistsError(destination)

    monkeypatch.setattr("phasesweep.cli.os.link", lose_publish_race)

    result = CliRunner().invoke(
        cli_main, ["mcp", "init-catalog", "--from", str(config), "-o", str(output)]
    )

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "other process\n"
    assert not list(tmp_path.glob(".catalog.yaml.*.tmp"))


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
