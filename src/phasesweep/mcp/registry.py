"""Catalog parsing and the id -> config trust boundary.

The agent only ever names an experiment id. This module is the sole component
that maps an id to a config path, and it does so from a catalog file the
operator authored out of band (same trust as the experiment YAML). Paths are
resolved and frozen at load; configs are validated at load; anything invalid
fails server startup.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from phasesweep.config import Experiment, Suite
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.io import _load_yaml_mapping_from_text, load_config_bytes
from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.mcp.runs import RunStore
from phasesweep.runtime.files import (
    file_url_path,
    sqlite_uri_filename_path,
    storage_backend,
    storage_is_in_memory,
)
from phasesweep.runtime.process import read_proc_starttime

VisibleParamsPolicy: TypeAlias = Literal["none", "all"] | list[str]


def _require_linux_mcp_host() -> None:
    """Require Linux process identity semantics for autonomous MCP control.

    :raises CatalogError: If the host cannot provide Linux ``/proc`` process
        start times used to prevent PID-reuse mistakes during cancellation and
        crash recovery.
    """
    if not sys.platform.startswith("linux"):
        raise CatalogError(
            "the phasesweep MCP broker is supported only on Linux because safe "
            "cancellation and crash recovery require /proc process identities",
            suggestion="run the MCP broker on Linux; the core phasesweep CLI remains POSIX-oriented",
        )
    if read_proc_starttime(os.getpid()) is None:
        raise CatalogError(
            "the phasesweep MCP broker cannot read this process's Linux /proc start time, "
            "which is required for PID-reuse-safe cancellation and crash recovery",
            suggestion="mount /proc with process stat access for the MCP server process",
        )


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
    cwd: Path | None = Field(
        default=None,
        description="Directory used as the detached runner working directory.",
    )
    description: str = Field(default="", max_length=500)
    allow: _Allow = Field(default_factory=_Allow)
    visible_params: VisibleParamsPolicy = Field(
        default="none",
        description=(
            "Sampled winner param values exposed to agents: 'none', 'all', or an allowlist."
        ),
    )

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

    @field_validator("visible_params")
    @classmethod
    def _valid_visible_params(cls, value: VisibleParamsPolicy) -> VisibleParamsPolicy:
        """Validate sampled-parameter visibility policy.

        :param VisibleParamsPolicy value: Operator-authored ``visible_params`` setting.
        :return VisibleParamsPolicy: ``value`` unchanged when it is ``"none"`` or
            ``"all"``; otherwise the key list with entries stripped and
            duplicates removed.
        """
        if isinstance(value, str):
            if value not in {"none", "all"}:
                raise ValueError("visible_params must be 'none', 'all', or a list of keys")
            return value
        normalized = [key.strip() for key in value]
        bad = [key for key in normalized if not key]
        if bad:
            raise ValueError("visible_params keys must be non-empty")
        return list(dict.fromkeys(normalized))


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
    cwd: Path  # absolute, frozen at load
    config_sha256: str
    experiment: Experiment
    description: str
    allow_launch: bool
    allow_cancel: bool
    allow_from_phase: bool
    visible_params: VisibleParamsPolicy

    @property
    def phase_names(self) -> list[str]:
        """Declared phase names, in order.

        :return list[str]: Phase names exactly as declared by the experiment config.
        """
        return [phase.name for phase in self.experiment.phases]

    @property
    def metric_payload(self) -> dict[str, str]:
        """Return the agent-visible optimization metric descriptor."""
        return {
            "name": self.experiment.metric.name,
            "goal": self.experiment.metric.goal,
        }

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return the agent-visible catalog permissions for this experiment."""
        return {
            "launch": self.allow_launch,
            "cancel": self.allow_cancel,
            "resume_from_phase": self.allow_from_phase,
        }


def _resolve_catalog_relative_path(base: Path, path: Path) -> Path:
    """Resolve an operator path relative to the catalog file when not absolute.

    :param Path base: Directory containing the catalog file.
    :param Path path: Operator-authored path from the catalog.
    :return Path: Absolute, resolved filesystem path.
    """
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _resolve_existing_dir(base: Path, path: Path, *, label: str) -> Path:
    """Resolve an operator path and require that it names an existing directory.

    :param Path base: Directory containing the catalog file.
    :param Path path: Operator-authored path from the catalog, resolved relative to ``base``.
    :param str label: Human-readable name for ``path``, used in the error message
        when the resolved path is not an existing directory.
    :return Path: Absolute, resolved path to the existing directory.
    """
    resolved = _resolve_catalog_relative_path(base, path)
    if not resolved.is_dir():
        raise CatalogError(f"{label} is not an existing directory: {resolved}")
    return resolved


def _prepare_state_dir(base: Path, path: Path) -> Path:
    """Resolve and initialize the run-store directories used at server startup.

    :param Path base: Directory containing the catalog file.
    :param Path path: Operator-authored ``state_dir`` path.
    :return Path: Absolute initialized state directory.
    :raises CatalogError: If the run store cannot create or secure its directories.
    """
    resolved = _resolve_catalog_relative_path(base, path)
    try:
        RunStore(resolved)
        for directory in (resolved, resolved / "runs", resolved / "logs"):
            with tempfile.NamedTemporaryFile(dir=directory):
                pass
    except OSError as exc:
        raise CatalogError(
            f"state_dir is not usable: {resolved}: {exc}",
            suggestion="set state_dir to a writable directory path (not a file)",
        ) from exc
    return resolved


def _check_state_dir(base: Path, path: Path) -> Path:
    """Validate the run-store layout without creating or chmodding any path.

    :param Path base: Directory containing the catalog file.
    :param Path path: Operator-authored ``state_dir`` path.
    :return Path: Absolute validated state directory path.
    :raises CatalogError: If an existing component has the wrong shape or the
        nearest existing directory cannot create the missing layout.
    """
    resolved = _resolve_catalog_relative_path(base, path)
    for candidate in (resolved, resolved / "runs", resolved / "logs"):
        if candidate.exists():
            if not candidate.is_dir() or not os.access(candidate, os.W_OK | os.X_OK):
                raise CatalogError(
                    f"state_dir is not usable: {resolved}",
                    suggestion="set state_dir to a writable directory path (not a file)",
                )
            continue
        parent = candidate.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise CatalogError(
                f"state_dir is not usable: {resolved}",
                suggestion="set state_dir to a writable directory path (not a file)",
            )
    return resolved


def _require_mcp_stable_paths(
    experiment_id: str, experiment: Experiment, *, config_dir: Path
) -> None:
    """Reject MCP configs whose filesystem targets depend on server CWD.

    :param str experiment_id: Catalog id being validated, used in operator-facing errors.
    :param Experiment experiment: Parsed experiment config registered for MCP access.
    :param Path config_dir: Directory of the experiment config, used to compute
        concrete fix suggestions for ``phasesweep mcp-check``.
    """
    storage = experiment.storage
    if storage is None or storage_is_in_memory(storage):
        raise CatalogError(
            f"{experiment_id!r}: storage must be persistent; "
            "in-memory storage cannot be monitored or resumed across processes",
            suggestion=_suggest_storage(config_dir),
        )

    workdir = Path(experiment.workdir).expanduser()
    if not workdir.is_absolute():
        raise CatalogError(
            f"{experiment_id!r}: MCP experiments must use an absolute workdir; "
            "relative workdir values depend on the server launch directory and "
            "break restart/recovery semantics",
            suggestion=f"set workdir to an absolute path, e.g. {(config_dir / workdir).resolve()}",
        )

    backend = storage_backend(storage)
    if backend not in {"sqlite", "journal"}:
        raise CatalogError(
            f"{experiment_id!r}: MCP experiments currently support only local-node "
            "SQLite or JournalStorage file-backed Optuna storage; external RDB "
            "storage is out of scope until multi-host cleanup semantics are supported",
            suggestion=_suggest_storage(config_dir),
        )
    raw_path = sqlite_uri_filename_path(storage) if backend == "sqlite" else None
    raw_path = file_url_path(storage) if raw_path is None else raw_path
    if raw_path == "":
        raise CatalogError(
            f"{experiment_id!r}: MCP experiments must use a non-empty absolute "
            f"{backend} storage path; empty file-backed storage URLs cannot be "
            "monitored across detached processes",
            suggestion=_suggest_storage(config_dir),
        )
    if not Path(raw_path).expanduser().is_absolute():
        raise CatalogError(
            f"{experiment_id!r}: MCP experiments must use an absolute {backend} "
            "storage path; relative storage URLs depend on the server launch "
            "directory and can point at a different Optuna study after restart",
            suggestion=f"use an absolute path, e.g. {(config_dir / raw_path).resolve()}",
        )


def _suggest_storage(config_dir: Path) -> str:
    """Suggest a valid MCP storage URL near the config.

    :param Path config_dir: Directory of the experiment config.
    :return str: Operator-facing suggestion for ``phasesweep mcp-check``.
    """
    return f"use a persistent local URL, e.g. storage: sqlite:///{config_dir / 'optuna.db'}"


def _parse_catalog(catalog_path: Path) -> tuple[_Catalog, Path]:
    """Read and schema-validate a catalog document.

    :param Path catalog_path: Path to the operator-authored catalog YAML.
    :return tuple[_Catalog, Path]: Parsed catalog and its base directory for
        resolving relative entry paths.
    """
    try:
        raw = _load_yaml_mapping_from_text(catalog_path.read_text(), catalog_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CatalogError(f"cannot read catalog {catalog_path}: {exc}") from exc
    try:
        catalog = _Catalog.model_validate(raw)
    except ValidationError as exc:
        raise CatalogError(f"catalog {catalog_path}: {exc}") from exc
    return catalog, catalog_path.resolve().parent


def _load_entry(base: Path, entry: _Entry) -> RegisteredExperiment:
    """Validate one catalog entry with the exact rules server startup applies.

    :param Path base: Directory containing the catalog file.
    :param _Entry entry: Schema-validated catalog entry to load.
    :return RegisteredExperiment: Frozen entry with resolved paths and config hash.
    """
    cfg_path = _resolve_catalog_relative_path(base, entry.config)
    if not cfg_path.is_file():
        raise CatalogError(f"{entry.id!r}: config not found: {cfg_path}")
    cwd = _resolve_existing_dir(
        base,
        entry.cwd if entry.cwd is not None else cfg_path.parent,
        label=f"{entry.id!r}: cwd",
    )
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
    _require_mcp_stable_paths(entry.id, config, config_dir=cfg_path.parent)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    return RegisteredExperiment(
        id=entry.id,
        config_path=cfg_path,
        cwd=cwd,
        config_sha256=config_sha256,
        experiment=config,
        description=entry.description,
        allow_launch=entry.allow.launch,
        allow_cancel=entry.allow.cancel,
        allow_from_phase=entry.allow.from_phase,
        visible_params=entry.visible_params,
    )


@dataclass(frozen=True)
class CatalogCheckEntry:
    """Operator-facing validation verdict for one catalog entry."""

    experiment_id: str
    error: str | None = None
    suggestion: str | None = None
    actions: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the entry would load at server startup.

        :return bool: True when the entry passed every validation rule.
        """
        return self.error is None


@dataclass(frozen=True)
class CatalogCheckReport:
    """Operator-facing validation verdicts for a whole catalog."""

    entries: tuple[CatalogCheckEntry, ...]

    @property
    def ok(self) -> bool:
        """Whether the whole catalog would load at server startup.

        :return bool: True when every entry passed validation.
        """
        return all(entry.ok for entry in self.entries)


def check_catalog(catalog_path: Path) -> CatalogCheckReport:
    """Validate every catalog entry, collecting per-entry verdicts.

    Runs the same validation as :meth:`Registry.load` (shared per-entry code
    path) but does not stop at the first failure, so ``phasesweep mcp-check``
    can report a full ok/FAIL table. This check is observational: it validates
    the state layout without creating directories or changing permissions.
    Catalog-level problems (unreadable file, schema errors) still raise
    :class:`CatalogError`.

    :param Path catalog_path: Path to the operator-authored catalog YAML.
    :return CatalogCheckReport: One verdict per catalog entry, in catalog order.
    """
    _require_linux_mcp_host()
    catalog, base = _parse_catalog(catalog_path)
    seen: set[str] = set()
    entries: list[CatalogCheckEntry] = []
    for entry in catalog.experiments:
        if entry.id in seen:
            entries.append(CatalogCheckEntry(entry.id, error=f"duplicate catalog id {entry.id!r}"))
            continue
        seen.add(entry.id)
        try:
            registered = _load_entry(base, entry)
        except CatalogError as exc:
            entries.append(CatalogCheckEntry(entry.id, error=str(exc), suggestion=exc.suggestion))
            continue
        actions = tuple(
            action
            for action, allowed in (
                ("launch", registered.allow_launch),
                ("cancel", registered.allow_cancel),
                ("from_phase", registered.allow_from_phase),
            )
            if allowed
        )
        entries.append(CatalogCheckEntry(entry.id, actions=actions))
    report = CatalogCheckReport(entries=tuple(entries))
    if report.ok:
        _check_state_dir(base, catalog.state_dir)
    return report


def prepare_catalog_state(catalog_path: Path) -> Path:
    """Create and secure the run-store directories declared by a catalog.

    Callers must validate the catalog before invoking this mutating step.

    :param Path catalog_path: Path to the operator-authored catalog YAML.
    :return Path: Absolute initialized state directory.
    """
    _require_linux_mcp_host()
    catalog, base = _parse_catalog(catalog_path)
    return _prepare_state_dir(base, catalog.state_dir)


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
        v1), and its storage is a persistent local SQLite/Journal file (the MCP
        layer is local-node only in this version).

        Args:
            catalog_path: Path to the operator-authored catalog YAML.

        Returns:
            An immutable :class:`Registry`.

        """
        _require_linux_mcp_host()
        catalog, base = _parse_catalog(catalog_path)
        items: dict[str, RegisteredExperiment] = {}
        for entry in catalog.experiments:
            if entry.id in items:
                raise CatalogError(f"duplicate catalog id {entry.id!r}")
            items[entry.id] = _load_entry(base, entry)
        return cls(
            state_dir=_prepare_state_dir(base, catalog.state_dir),
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
                "metric": item.metric_payload,
                "capabilities": item.capabilities,
            }
            for item in self._items.values()
        ]
