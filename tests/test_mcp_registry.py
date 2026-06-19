"""Catalog registry: the id -> config trust boundary and its startup validation."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.mcp.registry import Registry
from tests.mcp_helpers import write_mcp_catalog


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
    return write_mcp_catalog(
        tmp_path,
        {entry_id: config},
        allow=allow,
        max_concurrent_runs=max_concurrent_runs,
    )


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
    assert not reg.allow_launch
    assert not reg.allow_cancel
    assert not reg.allow_from_phase


def test_config_hash_and_model_come_from_same_startup_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path, name="mutated"))
    snapshot_bytes = textwrap.dedent(_experiment_yaml(tmp_path, name="snapshot")).lstrip().encode()
    original_read_bytes = Path.read_bytes
    calls = 0

    def read_bytes_once(path: Path) -> bytes:
        nonlocal calls
        if path == config:
            calls += 1
            return snapshot_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)

    reg = Registry.load(_catalog(tmp_path, config, entry_id="snapshot")).get("snapshot")

    assert calls == 1
    assert reg.experiment.experiment == "snapshot"
    assert reg.config_sha256 == hashlib.sha256(snapshot_bytes).hexdigest()


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


def test_malformed_config_yaml_raises_catalog_error(tmp_path: Path) -> None:
    config = _write(tmp_path / "broken.yaml", "experiment: [\n")
    with pytest.raises(CatalogError, match="invalid config"):
        Registry.load(_catalog(tmp_path, config))


def test_unknown_sampler_raises_catalog_error(tmp_path: Path) -> None:
    bad = _experiment_yaml(tmp_path).replace(
        "n_trials: 2\n            search_space:",
        "n_trials: 2\n            sampler: { type: nope }\n            search_space:",
        1,
    )
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


@pytest.mark.parametrize(
    "storage",
    ['"sqlite://"', '"sqlite:///:memory:"', '"sqlite+pysqlite:///:memory:"', '":memory:"'],
)
def test_in_memory_storage_urls_rejected(tmp_path: Path, storage: str) -> None:
    config = _write(
        tmp_path / "exp.yaml",
        _experiment_yaml(tmp_path).replace(f"sqlite:///{tmp_path}/reg_ok.db", storage),
    )
    with pytest.raises(CatalogError, match="storage must be persistent"):
        Registry.load(_catalog(tmp_path, config))


@pytest.mark.parametrize(
    "catalog_body",
    [
        """
        state_dir: {state}
        extra: true
        experiments:
          - id: reg_ok
            config: {config}
        """,
        """
        state_dir: {state}
        experiments:
          - id: reg_ok
            config: {config}
            cancle: false
        """,
        """
        state_dir: {state}
        experiments:
          - id: reg_ok
            config: {config}
            allow:
              from-phase: false
        """,
    ],
    ids=["top_level", "entry", "allow"],
)
def test_unknown_catalog_keys_rejected(tmp_path: Path, catalog_body: str) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    catalog = _write(
        tmp_path / "catalog.yaml",
        catalog_body.format(state=tmp_path / "state", config=config),
    )
    with pytest.raises(CatalogError, match="Extra inputs are not permitted"):
        Registry.load(catalog)


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
        allow={"launch": True, "cancel": True, "from_phase": True},
    )
    reg = Registry.load(catalog).get("reg_ok")
    assert reg.allow_launch is True
    assert reg.allow_cancel is True
    assert reg.allow_from_phase is True
