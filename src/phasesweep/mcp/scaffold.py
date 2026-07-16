"""Catalog scaffolding for ``phasesweep init-catalog``.

Builds an annotated, immediately-loadable MCP catalog from existing experiment
configs: absolute ``state_dir`` next to the catalog, one read-only entry per
config, ``visible_params: none``, and no ``allow`` block - side effects stay a
deliberate operator edit. Validation (``check_catalog``) happens in the CLI
before the file reaches its final name; this module only derives ids and
renders text. Operator-facing: rendered output contains real paths.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import yaml

from phasesweep.mcp.errors import CatalogError

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def derive_experiment_id(config: Path) -> str:
    """Derive an agent-visible experiment id from a config filename.

    :param Path config: Experiment config whose stem names the entry.
    :return str: Catalog-safe id (``[A-Za-z0-9_-]+``).
    :raises CatalogError: If nothing id-safe remains after sanitizing.
    """
    derived = _UNSAFE_ID_CHARS.sub("-", config.stem).strip("-")
    if not derived:
        raise CatalogError(
            f"cannot derive an experiment id from {config.name!r}",
            suggestion="rename the config file to letters, digits, '-' or '_'",
        )
    return derived


def _catalog_relative_config(config: Path, catalog_dir: Path) -> str:
    """Render a config path for the catalog, relative to the catalog when possible.

    :param Path config: Experiment config being cataloged.
    :param Path catalog_dir: Directory the catalog file will live in.
    :return str: ``./``-prefixed relative path for configs under the catalog
        directory, otherwise the absolute path (explicit beats a ``..`` chain
        that breaks when the catalog moves).
    """
    resolved = config.resolve()
    try:
        relative = resolved.relative_to(catalog_dir)
    except ValueError:
        return str(resolved)
    return f"./{relative}"


def _yaml_path_scalar(path: str | Path) -> str:
    """Render a filesystem path as one YAML-safe double-quoted scalar.

    :param str | Path path: Path value to serialize.
    :return str: YAML scalar that round-trips punctuation and line breaks.
    """
    return yaml.safe_dump(str(path), default_style='"', allow_unicode=True).strip()


def scaffold_catalog_text(output: Path, configs: Sequence[Path]) -> str:
    """Render an annotated catalog for ``configs``, addressed from ``output``.

    :param Path output: Catalog file the text will be written to; anchors the
        absolute ``state_dir`` and the relative ``config:`` paths.
    :param Sequence[Path] configs: Experiment configs, one catalog entry each.
    :return str: Complete catalog YAML with explanatory comments.
    :raises CatalogError: If two configs derive the same experiment id.
    """
    catalog_dir = output.parent.resolve()
    seen: dict[str, Path] = {}
    entries: list[str] = []
    for config in configs:
        experiment_id = derive_experiment_id(config)
        if experiment_id in seen:
            raise CatalogError(
                f"configs {seen[experiment_id]} and {config} both derive the "
                f"experiment id {experiment_id!r}",
                suggestion="rename one file, or edit the generated id afterwards",
            )
        seen[experiment_id] = config
        config_path = _yaml_path_scalar(_catalog_relative_config(config, catalog_dir))
        entries.append(
            f"""\
  - id: {experiment_id}          # the only token the agent ever sends
    config: {config_path}   # resolved relative to this catalog file
    description: "TODO: one line the agent sees in phasesweep_list_experiments"
    visible_params: none        # winner values return <redacted>; set all, or list the keys to expose
    # Side effects default to false. Uncomment deliberately to let the agent act:
    # allow:
    #   launch: true
    #   cancel: true
    #   from_phase: true
"""
        )
    state_dir = catalog_dir / "runs" / ".mcp"
    header = f"""\
# MCP catalog scaffolded by `phasesweep init-catalog`. Start the server with:
#   phasesweep mcp --catalog /absolute/path/to/catalog.yaml
#
# The agent only ever sends an experiment `id`. It cannot pass a path, author a
# config, or reach trial_command / env / storage / workdir. You curate which
# experiments exist by editing this file (same trust as the experiment YAML).

state_dir: {_yaml_path_scalar(state_dir)}   # run handles, logs, audit.jsonl (operator-owned)
max_concurrent_runs: 1          # sweeps at once across all experiments; 1 keeps a single GPU sane
experiments:
"""
    return header + "".join(entries)
