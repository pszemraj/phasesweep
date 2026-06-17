"""Shared MCP test setup helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.server import PhaseSweepMCP


def write_mcp_catalog(
    tmp_path: Path,
    entries: Mapping[str, Path],
    *,
    allow: Mapping[str, bool] | None = None,
    max_concurrent_runs: int | None = None,
    filename: str = "catalog.yaml",
) -> Path:
    lines = [f"state_dir: {tmp_path}/state"]
    if max_concurrent_runs is not None:
        lines.append(f"max_concurrent_runs: {max_concurrent_runs}")
    lines.append("experiments:")
    for entry_id, config in entries.items():
        lines += [f"  - id: {entry_id}", f"    config: {config}"]
        if allow is not None:
            lines.append("    allow:")
            lines.extend(f"      {key}: {str(value).lower()}" for key, value in allow.items())
    catalog = tmp_path / filename
    catalog.write_text("\n".join(lines) + "\n")
    return catalog


def write_mcp_config_catalog(
    tmp_path: Path,
    configs: Mapping[str, str],
    *,
    max_concurrent_runs: int | None = None,
    filename: str = "catalog.yaml",
) -> Path:
    entries = {}
    for entry_id, body in configs.items():
        config = tmp_path / f"{entry_id}.yaml"
        config.write_text(body)
        entries[entry_id] = config
    return write_mcp_catalog(tmp_path, entries, max_concurrent_runs=max_concurrent_runs, filename=filename)


def make_mcp_app(catalog: Path) -> tuple[PhaseSweepMCP, Registry, RunStore]:
    registry = Registry.load(catalog)
    store = RunStore(registry.state_dir)
    return PhaseSweepMCP(registry, store), registry, store
