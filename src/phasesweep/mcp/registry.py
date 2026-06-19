"""Catalog parsing and the id -> config trust boundary.

The agent only ever names an experiment id. This module is the sole component
that maps an id to a config path, and it does so from a catalog file the
operator authored out of band (same trust as the experiment YAML). Paths are
resolved and frozen at load; configs are validated at load; anything invalid
fails server startup.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from phasesweep.config import Experiment, Suite
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.io import load_config_bytes
from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.runtime.files import file_url_path, storage_backend


class _CatalogModel(BaseModel):
    """Strict base for operator-authored catalog documents."""

    model_config = ConfigDict(extra="forbid")


class _Allow(_CatalogModel):
    """Per-experiment permission flags.

    Side effects are opt-in: omitting ``allow`` leaves the experiment
    read-only. Safety waivers are intentionally not expressible here - they are
    not config the agent may touch.
    """

    launch: bool = False
    cancel: bool = False
    from_phase: bool = False


class _Entry(_CatalogModel):
    """One catalog entry: an opaque id mapped to a local config path."""

    id: str
    config: Path
    description: str = ""
    allow: _Allow = Field(default_factory=_Allow)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        """Validate that the catalog id is safe for run ids and filenames.

        :param str value: Operator-authored catalog id.
        :return str: The validated id.
        """
        # The id appears in run ids and handle filenames, so keep it path-safe
        # even though the operator writes it.
        if not SAFE_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"catalog id {value!r} must match [A-Za-z0-9_-]+")
        return value


class _Catalog(_CatalogModel):
    """Top-level catalog file: a server state dir plus one or more entries."""

    state_dir: Path
    # Cap on simultaneously-running sweeps across ALL experiments. Defaults to 1
    # because the common deployment is a single GPU, where a second concurrent
    # sweep would contend for the device. Raise it on multi-GPU hosts.
    max_concurrent_runs: int = Field(default=1, ge=1)
    experiments: list[_Entry] = Field(min_length=1)


@dataclass(frozen=True)
class RegisteredExperiment:
    """A validated catalog entry.

    ``config_path`` is server-internal and is never returned to the agent.
    """

    id: str
    config_path: Path  # absolute, frozen at load
    config_sha256: str
    experiment: Experiment
    description: str
    allow_launch: bool
    allow_cancel: bool
    allow_from_phase: bool

    @property
    def phase_names(self) -> list[str]:
        """Declared phase names, in order.

        :return list[str]: Phase names exactly as declared by the experiment config.
        """
        return [phase.name for phase in self.experiment.phases]


def _storage_is_in_memory(storage: str | None) -> bool:
    """Return whether a storage URL resolves to an in-memory Optuna backend.

    :param str | None storage: Configured Optuna storage URL.
    :return bool: True when the storage cannot be monitored across processes.
    """
    if storage is None:
        return True
    if storage == ":memory:":
        return True
    if storage_backend(storage) != "sqlite":
        return False
    database = file_url_path(storage)
    query = storage.split("?", 1)[1].split("#", 1)[0] if "?" in storage else ""
    options = {key.lower(): value.lower() for key, value in parse_qsl(query)}
    return (
        database == ""
        or database == ":memory:"
        or database.startswith(":memory:?")
        or database.startswith("file::memory:")
        or (database.startswith("file:") and options.get("mode") == "memory")
    )


class Registry:
    """Immutable id -> RegisteredExperiment map plus the server state dir."""

    def __init__(
        self,
        state_dir: Path,
        items: dict[str, RegisteredExperiment],
        max_concurrent_runs: int = 1,
    ) -> None:
        """Create an immutable registry from already validated entries.

        :param Path state_dir: Directory used for MCP run handles and operator logs.
        :param dict[str, RegisteredExperiment] items: Validated catalog entries keyed by id.
        :param int max_concurrent_runs: Maximum live sweeps allowed across all entries.
        """
        self.state_dir = state_dir
        self.max_concurrent_runs = max_concurrent_runs
        self._items = items

    @classmethod
    def load(cls, catalog_path: Path) -> Registry:
        """Parse and validate a catalog file.

        Raises ``CatalogError`` on any problem so the server refuses to start
        with a bad catalog. Per entry: the config path exists, ``load_config``
        accepts it, it is an :class:`Experiment` (suites are out of scope for
        v1), and its storage is persistent (in-memory studies cannot be
        monitored across processes).

        Args:
            catalog_path: Path to the operator-authored catalog YAML.

        Returns:
            An immutable :class:`Registry`.

        """
        try:
            raw = yaml.safe_load(catalog_path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise CatalogError(f"cannot read catalog {catalog_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CatalogError(f"catalog {catalog_path}: top level must be a mapping")
        try:
            catalog = _Catalog.model_validate(raw)
        except ValidationError as exc:
            raise CatalogError(f"catalog {catalog_path}: {exc}") from exc

        base = catalog_path.resolve().parent
        items: dict[str, RegisteredExperiment] = {}
        for entry in catalog.experiments:
            if entry.id in items:
                raise CatalogError(f"duplicate catalog id {entry.id!r}")
            cfg_path = entry.config
            cfg_path = (cfg_path if cfg_path.is_absolute() else base / cfg_path).resolve()
            if not cfg_path.is_file():
                raise CatalogError(f"{entry.id!r}: config not found: {cfg_path}")
            try:
                config_bytes = cfg_path.read_bytes()
                config = load_config_bytes(config_bytes, source=cfg_path)
            except (ValueError, OSError, yaml.YAMLError) as exc:
                raise CatalogError(f"{entry.id!r}: invalid config {cfg_path}: {exc}") from exc
            if isinstance(config, Suite):
                raise CatalogError(
                    f"{entry.id!r}: suite configs are not supported by the MCP layer "
                    "in this version; register single-experiment configs"
                )
            if _storage_is_in_memory(config.storage):
                raise CatalogError(
                    f"{entry.id!r}: storage must be persistent; in-memory studies "
                    "cannot be monitored across processes"
                )
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            items[entry.id] = RegisteredExperiment(
                id=entry.id,
                config_path=cfg_path,
                config_sha256=config_sha256,
                experiment=config,
                description=entry.description,
                allow_launch=entry.allow.launch,
                allow_cancel=entry.allow.cancel,
                allow_from_phase=entry.allow.from_phase,
            )
        return cls(
            state_dir=catalog.state_dir.expanduser(),
            items=items,
            max_concurrent_runs=catalog.max_concurrent_runs,
        )

    def get(self, experiment_id: str) -> RegisteredExperiment:
        """Look up a registered experiment by id, or raise ``UnknownExperimentError``.

        :param str experiment_id: Agent-visible catalog id.
        :return RegisteredExperiment: Validated registry entry for the id.
        """
        try:
            return self._items[experiment_id]
        except KeyError:
            raise UnknownExperimentError(experiment_id) from None

    def summaries(self) -> list[dict[str, Any]]:
        """Path-free catalog listing for ``list_experiments``.

        :return list[dict[str, Any]]: Agent-visible summaries without paths, commands, storage URLs, or env values.
        """
        return [
            {
                "id": item.id,
                "description": item.description,
                "phases": item.phase_names,
                "metric": {
                    "name": item.experiment.metric.name,
                    "goal": item.experiment.metric.goal,
                },
            }
            for item in self._items.values()
        ]
