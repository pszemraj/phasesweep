"""Catalog registry: the id -> config trust boundary and its startup validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.mcp.registry import Registry


def _write(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def _experiment_yaml(tmp_path: Path, *, name: str = "reg_ok", with_storage: bool = True) -> str:
    storage = f"storage: sqlite:///{tmp_path}/{name}.db" if with_storage else ""
    return f"""\
        experiment: {name}
        {storage}
        workdir: {tmp_path}/wd_{name}
        trial_command: "python train.py --out {{trial_dir}}/r.json {{overrides}}"
        metric:
          name: loss
          goal: minimize
          extractor: {{ type: json, path: r.json, key: loss }}
        phases:
          - name: warmup
            n_trials: 2
            search_space:
              lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
          - name: tune
            inherits: [warmup]
            n_trials: 3
            search_space:
              wd: {{ type: float, low: 0.0, high: 0.3 }}
    """


def _suite_yaml(tmp_path: Path) -> str:
    return f"""\
        suite: reg_suite
        defaults:
          workdir: {tmp_path}/runs
          trial_command: "echo {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: json, path: r.json, key: x }}
        studies:
          - name: a
            phases:
              - name: p
                n_trials: 1
                fixed_overrides: {{ score: 1.0 }}
                search_space: {{}}
    """


def _catalog(
    tmp_path: Path,
    config: Path,
    *,
    entry_id: str = "reg_ok",
    allow: dict | None = None,
    max_concurrent_runs: int | None = None,
) -> Path:
    lines = [f"state_dir: {tmp_path}/state"]
    if max_concurrent_runs is not None:
        lines.append(f"max_concurrent_runs: {max_concurrent_runs}")
    lines += ["experiments:", f"  - id: {entry_id}", f"    config: {config}"]
    if allow is not None:
        lines.append("    allow:")
        lines.extend(f"      {k}: {str(v).lower()}" for k, v in allow.items())
    path = tmp_path / "catalog.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_valid_catalog_loads_and_summaries_are_path_free(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    registry = Registry.load(_catalog(tmp_path, config))

    summaries = registry.summaries()
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["id"] == "reg_ok"
    assert summary["phases"] == ["warmup", "tune"]
    assert summary["metric"] == {"name": "loss", "goal": "minimize"}

    # The summary must carry no path, command, or storage URL.
    blob = str(summaries)
    for needle in ("train.py", "sqlite", str(config), str(tmp_path / "wd_reg_ok")):
        assert needle not in blob


def test_get_returns_registered_experiment_with_internal_fields(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    registry = Registry.load(_catalog(tmp_path, config))

    reg = registry.get("reg_ok")
    assert reg.config_path == config.resolve()
    assert len(reg.config_sha256) == 64
    assert reg.phase_names == ["warmup", "tune"]
    assert reg.allow_launch and reg.allow_cancel and reg.allow_from_phase
    assert reg.expose_trial_logs is False


def test_unknown_id_raises(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    registry = Registry.load(_catalog(tmp_path, config))
    with pytest.raises(UnknownExperimentError):
        registry.get("does-not-exist")


def test_invalid_config_raises_catalog_error(tmp_path: Path) -> None:
    # goal must be minimize/maximize; "sideways" is a Literal violation.
    bad = _experiment_yaml(tmp_path).replace("goal: minimize", "goal: sideways")
    config = _write(tmp_path / "exp.yaml", bad)
    with pytest.raises(CatalogError, match="invalid config"):
        Registry.load(_catalog(tmp_path, config))


def test_suite_config_rejected(tmp_path: Path) -> None:
    config = _write(tmp_path / "suite.yaml", _suite_yaml(tmp_path))
    with pytest.raises(CatalogError, match="suite"):
        Registry.load(_catalog(tmp_path, config))


def test_missing_storage_rejected(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path, with_storage=False))
    with pytest.raises(CatalogError, match="storage"):
        Registry.load(_catalog(tmp_path, config))


def test_config_not_found_rejected(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="not found"):
        Registry.load(_catalog(tmp_path, tmp_path / "nope.yaml"))


def test_unsafe_catalog_id_rejected(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    with pytest.raises(CatalogError):
        Registry.load(_catalog(tmp_path, config, entry_id="bad-id.evil"))


def test_max_concurrent_runs_defaults_to_one(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    assert Registry.load(_catalog(tmp_path, config)).max_concurrent_runs == 1


def test_max_concurrent_runs_override(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    reg = Registry.load(_catalog(tmp_path, config, max_concurrent_runs=3))
    assert reg.max_concurrent_runs == 3


def test_permission_flags_propagate(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    catalog = _catalog(
        tmp_path,
        config,
        allow={"launch": False, "cancel": False, "from_phase": False},
    )
    reg = Registry.load(catalog).get("reg_ok")
    assert reg.allow_launch is False
    assert reg.allow_cancel is False
    assert reg.allow_from_phase is False
