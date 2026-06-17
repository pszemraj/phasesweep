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

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from phasesweep.config import Experiment, Suite, load_config
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.runtime.files import file_url_path, storage_backend


class _CatalogModel(BaseModel):
    """Strict base for operator-authored catalog documents."""

    model_config = ConfigDict(extra="forbid")


class _Allow(_CatalogModel):
    """Per-experiment permission flags.

    Default-open for the read/launch verbs. Safety waivers are intentionally
    not expressible here - they are not config the agent may touch.
    """

    launch: bool = True
    cancel: bool = True
    from_phase: bool = True


class _Entry(_CatalogModel):
    """One catalog entry: an opaque id mapped to a local config path."""

    id: str
    config: Path
    description: str = ""
    allow: _Allow = Field(default_factory=_Allow)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
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
        """Declared phase names, in order."""
        return [phase.name for phase in self.experiment.phases]


def _storage_is_in_memory(storage: str | None) -> bool:
    """Return whether a storage URL resolves to an in-memory Optuna backend."""
    if storage is None:
        return True
    if storage == ":memory:":
        return True
    if storage_backend(storage) != "sqlite":
        return False
    database = file_url_path(storage)
    return (
        database == ""
        or database == ":memory:"
        or database.startswith(":memory:?")
        or database.startswith("file::memory:")
    )


class Registry:
    """Immutable id -> RegisteredExperiment map plus the server state dir."""

    def __init__(
        self,
        state_dir: Path,
        items: dict[str, RegisteredExperiment],
        max_concurrent_runs: int = 1,
    ) -> None:
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
                config = load_config(cfg_path)
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
            items[entry.id] = RegisteredExperiment(
                id=entry.id,
                config_path=cfg_path,
                config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
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
        """Look up a registered experiment by id, or raise ``UnknownExperimentError``."""
        try:
            return self._items[experiment_id]
        except KeyError:
            raise UnknownExperimentError(experiment_id) from None

    def summaries(self) -> list[dict[str, Any]]:
        """Path-free catalog listing for ``list_experiments``."""
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
