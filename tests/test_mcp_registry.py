"""Catalog registry: the id -> config trust boundary and its startup validation."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.mcp.registry import Registry
from tests.mcp_helpers import mcp_experiment_config_text, write_mcp_catalog

REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def _experiment_yaml(tmp_path: Path, *, name: str = "reg_ok", with_storage: bool = True) -> str:
    phases = """\
  - name: warmup
    n_trials: 2
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  - name: tune
    inherits: [warmup]
    n_trials: 3
    search_space:
      wd: { type: float, low: 0.0, high: 0.3 }
"""
    return mcp_experiment_config_text(tmp_path, name=name, phases=phases, with_storage=with_storage)


def _suite_yaml(tmp_path: Path) -> str:
    return f"""\
        suite: reg_suite
        defaults:
          workdir: {tmp_path}/runs
          trial_command: "echo {{overrides}}"
          metric:
            name: x
            goal: minimize
            extractor: {{ type: json_envelope, path: r.json, objective_name: x, split: test, policy: test }}
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
    cwd: dict[str, Path] | None = None,
    max_concurrent_runs: int | None = None,
) -> Path:
    return write_mcp_catalog(
        tmp_path,
        {entry_id: config},
        allow=allow,
        cwd=cwd,
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
    assert summary["metric"] == {
        "name": "loss",
        "goal": "minimize",
        "objective_evidence": {
            "kind": "json_envelope",
            "attempt_bound": True,
            "objective_name_bound": True,
            "split_bound": True,
            "evaluation_policy_bound": True,
            # mcp_experiment_config_text's json_envelope extractor declares
            # neither checkpoint nor expected_step, so neither is value-bound.
            "checkpoint_declared": False,
            "checkpoint_value_bound": False,
            "expected_step_declared": False,
            "expected_step_value_bound": False,
        },
    }
    assert summary["capabilities"] == {
        "launch": False,
        "cancel": False,
        "resume_from_phase": False,
    }

    # The summary must carry no path, command, or storage URL.
    blob = str(summaries)
    for needle in ("train.py", "sqlite", str(config), str(tmp_path / "runs" / "reg_ok")):
        assert needle not in blob


def test_get_returns_registered_experiment_with_internal_fields(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    registry = Registry.load(_catalog(tmp_path, config))

    reg = registry.get("reg_ok")
    assert reg.config_path == config.resolve()
    assert reg.cwd == config.resolve().parent
    assert len(reg.config_sha256) == 64
    assert reg.phase_names == ["warmup", "tune"]
    assert not reg.allow_launch
    assert not reg.allow_cancel
    assert not reg.allow_from_phase


@pytest.mark.parametrize(
    ("catalog_path", "experiment_id", "description_fragment"),
    [
        ("examples/catalog.yaml", "tiny-lm", "16 MB LM"),
        (
            "examples/tiny_decoder_enwik8/catalog.yaml",
            "tiny-decoder-enwik8-hparams",
            "1000 batches/trial",
        ),
    ],
)
def test_checked_in_catalog_loads(
    catalog_path: str,
    experiment_id: str,
    description_fragment: str,
) -> None:
    registry = Registry.load(REPO / catalog_path)

    entry = registry.get(experiment_id)
    assert entry.id == experiment_id
    assert description_fragment in entry.description
    assert "smoke" not in entry.experiment.experiment


def test_relative_state_dir_resolves_against_catalog_file(tmp_path: Path) -> None:
    _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    catalog_dir = tmp_path / "catalogs"
    catalog_dir.mkdir()
    catalog = _write(
        catalog_dir / "catalog.yaml",
        """\
        state_dir: .mcp
        experiments:
          - id: reg_ok
            config: ../exp.yaml
        """,
    )

    registry = Registry.load(catalog)

    assert registry.state_dir == (catalog_dir / ".mcp").resolve()


def test_catalog_cwd_resolves_against_catalog_file(tmp_path: Path) -> None:
    _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    run_cwd = tmp_path / "run-cwd"
    run_cwd.mkdir()
    catalog_dir = tmp_path / "catalogs"
    catalog_dir.mkdir()
    catalog = _write(
        catalog_dir / "catalog.yaml",
        """\
        state_dir: .mcp
        experiments:
          - id: reg_ok
            config: ../exp.yaml
            cwd: ../run-cwd
        """,
    )

    reg = Registry.load(catalog).get("reg_ok")

    assert reg.cwd == run_cwd.resolve()


def test_catalog_cwd_must_exist(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))

    with pytest.raises(CatalogError, match="cwd is not an existing directory"):
        Registry.load(_catalog(tmp_path, config, cwd={"reg_ok": tmp_path / "missing"}))


def test_catalog_visible_params_policy(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))

    all_policy = Registry.load(
        write_mcp_catalog(tmp_path, {"reg_ok": config}, visible_params={"reg_ok": "all"})
    ).get("reg_ok")
    allowlist_policy = Registry.load(
        write_mcp_catalog(
            tmp_path,
            {"reg_ok": config},
            visible_params={"reg_ok": ["lr", "dataset"]},
            filename="allowlist.catalog.yaml",
        )
    ).get("reg_ok")

    assert all_policy.visible_params == "all"
    assert allowlist_policy.visible_params == ["lr", "dataset"]


def test_catalog_rejects_invalid_visible_params_policy(tmp_path: Path) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))

    with pytest.raises(CatalogError, match="visible_params"):
        Registry.load(
            write_mcp_catalog(
                tmp_path,
                {"reg_ok": config},
                visible_params={"reg_ok": "sometimes"},
            )
        )


def test_relative_workdir_rejected_for_mcp(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "exp.yaml",
        _experiment_yaml(tmp_path).replace(f"workdir: {tmp_path}/runs/reg_ok", "workdir: runs"),
    )

    with pytest.raises(CatalogError, match="absolute workdir"):
        Registry.load(_catalog(tmp_path, config))


@pytest.mark.parametrize(
    "storage",
    [
        '"sqlite:///relative.db"',
        '"sqlite+pysqlite:///relative.db"',
        '"sqlite:///file:relative.db?mode=rwc&uri=true"',
        '"journal:///relative.journal"',
        '"journal://"',
        '"journal:///"',
    ],
)
def test_relative_file_storage_rejected_for_mcp(tmp_path: Path, storage: str) -> None:
    config = _write(
        tmp_path / "exp.yaml",
        _experiment_yaml(tmp_path).replace(f"sqlite:///{tmp_path}/reg_ok.db", storage),
    )

    with pytest.raises(CatalogError, match="absolute .*storage path"):
        Registry.load(_catalog(tmp_path, config))


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


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("goal: minimize", "goal: sideways"),
        (
            "n_trials: 2\n    search_space:",
            "n_trials: 2\n    sampler: { type: nope }\n    search_space:",
        ),
    ],
    ids=["invalid_goal", "unknown_sampler"],
)
def test_invalid_config_raises_catalog_error(tmp_path: Path, old: str, new: str) -> None:
    bad = _experiment_yaml(tmp_path).replace(old, new, 1)
    config = _write(tmp_path / "exp.yaml", bad)
    with pytest.raises(CatalogError, match="invalid config"):
        Registry.load(_catalog(tmp_path, config))


def test_malformed_config_yaml_raises_catalog_error(tmp_path: Path) -> None:
    config = _write(tmp_path / "broken.yaml", "experiment: [\n")
    with pytest.raises(CatalogError, match="invalid config"):
        Registry.load(_catalog(tmp_path, config))


@pytest.mark.parametrize(
    "catalog_body",
    [
        """
        state_dir: {state}
        state_dir: {state}/other
        experiments:
          - id: reg_ok
            config: {config}
        """,
        """
        state_dir: {state}
        experiments:
          - id: reg_ok
            config: {config}
            config: {config}
        """,
        """
        state_dir: {state}
        experiments:
          - id: reg_ok
            config: {config}
            allow:
              launch: false
              launch: true
        """,
    ],
    ids=["top_level", "entry", "allow"],
)
def test_duplicate_catalog_yaml_keys_rejected(tmp_path: Path, catalog_body: str) -> None:
    config = _write(tmp_path / "exp.yaml", _experiment_yaml(tmp_path))
    catalog = _write(
        tmp_path / "catalog.yaml",
        catalog_body.format(state=tmp_path / "state", config=config),
    )

    with pytest.raises(CatalogError, match="duplicate key"):
        Registry.load(catalog)


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
    [
        '"sqlite://"',
        '"sqlite:///:memory:"',
        '"sqlite+pysqlite:///:memory:"',
        '"sqlite:///file:memdb1?mode=memory&cache=shared&uri=true"',
        '"sqlite+pysqlite:///file:memdb1?mode=memory&cache=shared&uri=true"',
        '":memory:"',
    ],
)
def test_in_memory_storage_urls_rejected(tmp_path: Path, storage: str) -> None:
    config = _write(
        tmp_path / "exp.yaml",
        _experiment_yaml(tmp_path).replace(f"sqlite:///{tmp_path}/reg_ok.db", storage),
    )
    with pytest.raises(CatalogError, match="storage must be persistent"):
        Registry.load(_catalog(tmp_path, config))


@pytest.mark.parametrize(
    "storage",
    [
        '"postgresql://user:pass@example.com/phases"',
        '"postgresql+psycopg2://user:pass@example.com/phases"',
        '"mysql+pymysql://user:pass@example.com/phases"',
    ],
)
def test_external_rdb_storage_rejected_for_local_node_mcp(
    tmp_path: Path,
    storage: str,
) -> None:
    config = _write(
        tmp_path / "exp.yaml",
        _experiment_yaml(tmp_path)
        .replace(f"sqlite:///{tmp_path}/reg_ok.db", storage)
        # Acknowledge the config-level single-host gate so this test reaches
        # the MCP-specific local-node rejection, which has no such escape.
        .replace("experiment: reg_ok\n", "experiment: reg_ok\nallow_unsafe_multihost: true\n"),
    )

    with pytest.raises(CatalogError, match="local-node SQLite or JournalStorage"):
        Registry.load(_catalog(tmp_path, config))


def test_persistent_sqlite_uri_file_storage_allowed(tmp_path: Path) -> None:
    storage = f'"sqlite:///file:{tmp_path}/uri.db?mode=rwc&uri=true"'
    config = _write(
        tmp_path / "exp.yaml",
        _experiment_yaml(tmp_path).replace(f"sqlite:///{tmp_path}/reg_ok.db", storage),
    )

    registry = Registry.load(_catalog(tmp_path, config))

    assert registry.get("reg_ok").experiment.storage == storage.strip('"')


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
