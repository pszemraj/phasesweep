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
import shlex
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from phasesweep.config import Experiment, Suite
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.io import _load_yaml_mapping_from_text, load_config_bytes
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.engine.paths import _experiment_dir
from phasesweep.mcp.errors import CatalogError, UnknownExperimentError
from phasesweep.mcp.runs import RunStore
from phasesweep.runtime.files import (
    UnsafePrivatePathError,
    canonical_storage_identity,
    file_url_path,
    sqlite_uri_filename_path,
    storage_backend,
    storage_is_in_memory,
)
from phasesweep.runtime.process import read_proc_starttime

VisibleParamsPolicy: TypeAlias = Literal["none", "all"] | list[str]


def require_linux_mcp_host() -> None:
    """Require Linux process identity semantics for autonomous MCP control.

    :raises CatalogError: If the host cannot provide Linux ``/proc`` process
        start times used to prevent PID-reuse mistakes during cancellation and
        crash recovery.
    """
    if not sys.platform.startswith("linux"):
        raise CatalogError(
            "the phasesweep MCP server is supported only on Linux because safe "
            "cancellation and crash recovery require /proc process identities",
            suggestion="run the MCP server on Linux; the core phasesweep CLI remains POSIX-oriented",
        )
    if read_proc_starttime(os.getpid()) is None:
        raise CatalogError(
            "the phasesweep MCP server cannot read this process's Linux /proc start time, "
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
        :raises ValueError: If the id is not ``[A-Za-z0-9_-]+``.
        """
        # The id appears in run ids and handle filenames, so keep it path-safe
        # even though the operator writes it.
        if not SAFE_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"catalog id {value!r} must match [A-Za-z0-9_-]+")
        return value

    @field_validator("visible_params")
    @classmethod
    def _valid_visible_params(cls, value: VisibleParamsPolicy) -> VisibleParamsPolicy:
        """Validate and normalize a sampled-parameter visibility policy.

        :param VisibleParamsPolicy value: Policy string or parameter-name allowlist.
        :return VisibleParamsPolicy: A valid policy string or deduplicated, stripped allowlist.
        :raises ValueError: If a string policy is neither ``'none'`` nor ``'all'``,
            or an allowlist contains a blank key.
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
    def metric_payload(self) -> dict[str, Any]:
        """Return the agent-visible optimization metric descriptor."""
        return _metric_semantics_payload(self.experiment.metric)

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
    :param Path path: Operator-authored directory path.
    :param str label: Operator-facing name for the path in validation errors.
    :return Path: Absolute, resolved existing directory path.
    :raises CatalogError: If the resolved path is not an existing directory.
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
    except (OSError, UnsafePrivatePathError) as exc:
        raise CatalogError(
            f"state_dir is not usable: {resolved}: {exc}",
            suggestion=_state_dir_remediation(resolved),
        ) from exc
    return resolved


def _state_dir_remediation(state_dir: Path) -> str:
    """Return an exact mode fix when an existing private directory is repairable.

    Checks ``state_dir`` and its ``runs``/``logs`` subdirectories for one this
    process owns whose mode is not ``0700``, and if found, suggests the exact
    ``chmod`` command to fix it. Falls back to a generic suggestion when none
    of those directories exist, none are owned by this process, or the
    problem is something other than permissions (e.g. a file or symlink in
    place of a directory).

    :param Path state_dir: Operator-configured MCP state directory root.
    :return str: Operator-facing suggestion for ``phasesweep mcp check``.
    """
    for directory in (state_dir, state_dir / "runs", state_dir / "logs"):
        try:
            info = os.stat(directory, follow_symlinks=False)
        except OSError:
            continue
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and mode != 0o700:
            return (
                f"run chmod 700 {shlex.quote(str(directory))}, then retry "
                f"(found uid {info.st_uid} and mode {mode:04o})"
            )
    return "set state_dir to a writable directory path (owner-only; not a file or symlink)"


def _require_mcp_stable_paths(
    experiment_id: str, experiment: Experiment, *, config_dir: Path
) -> None:
    """Reject MCP configs whose filesystem targets depend on server CWD.

    :param str experiment_id: Catalog id being validated, used in operator-facing errors.
    :param Experiment experiment: Parsed experiment config registered for MCP access.
    :param Path config_dir: Directory of the experiment config, used to compute
        concrete fix suggestions for ``phasesweep mcp check``.
    :raises CatalogError: If storage is absent or in-memory, ``workdir`` or
        ``execution.cwd`` is relative, the storage backend is not local SQLite
        or JournalStorage, or its file path is empty or relative.
    """
    storage = experiment.resolved_storage
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

    execution_cwd = experiment.execution.cwd
    if execution_cwd is not None and not Path(execution_cwd).expanduser().is_absolute():
        # Same stability rule as workdir (review v0.5.17 / blocker 4): a
        # relative trainer cwd would resolve differently for a CLI operator
        # and the detached MCP runner, splitting one config into two
        # execution contexts.
        raise CatalogError(
            f"{experiment_id!r}: MCP experiments must use an absolute execution.cwd; "
            "relative values depend on the invoking process's directory and would "
            "give CLI and MCP invocations different trainer working directories",
            suggestion=(
                "set execution.cwd to an absolute path, e.g. "
                f"{(config_dir / Path(execution_cwd).expanduser()).resolve()}"
            ),
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
    :return str: Operator-facing suggestion for ``phasesweep mcp check``.
    """
    return f"use a persistent local URL, e.g. storage: sqlite:///{config_dir / 'optuna.db'}"


def _parse_catalog(catalog_path: Path) -> tuple[_Catalog, Path]:
    """Read and schema-validate a catalog document.

    :param Path catalog_path: Path to the operator-authored catalog YAML.
    :return tuple[_Catalog, Path]: Parsed catalog and its base directory for
        resolving relative entry paths.
    :raises CatalogError: If the file cannot be read, is not a YAML mapping, or
        fails catalog schema validation.
    """
    try:
        raw = _load_yaml_mapping_from_text(catalog_path.read_text(), catalog_path)
    except (OSError, ValueError) as exc:
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
    :raises CatalogError: If the config path or ``cwd`` does not exist, the
        config cannot be parsed, it is a suite rather than an experiment, or it
        fails the MCP path-stability rules.
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
    except (ValueError, OSError) as exc:
        raise CatalogError(f"{entry.id!r}: invalid config {cfg_path}: {exc}") from exc
    if isinstance(config, Suite):
        raise CatalogError(
            f"{entry.id!r}: suite configs are not supported by the MCP layer "
            "in this version; register single-experiment configs"
        )
    _require_mcp_stable_paths(entry.id, config, config_dir=cfg_path.parent)
    if config.execution.cwd is None:
        # The detached runner enters the catalog entry's cwd before invoking
        # the engine. Materialize that effective trainer cwd in the in-memory
        # registry config so server-side fingerprints and drift comparisons
        # describe the same execution context as the runner.
        config = config.model_copy(
            update={"execution": config.execution.model_copy(update={"cwd": str(cwd)})}
        )
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


def _claim_catalog_entry_identity(
    loaded: RegisteredExperiment,
    namespace_owners: dict[str, str],
    storage_owners: dict[str, str],
) -> None:
    """Claim one entry's engine namespaces, rejecting aliases across catalog ids.

    :param RegisteredExperiment loaded: Validated catalog entry to claim.
    :param dict[str, str] namespace_owners: Catalog ids keyed by output namespace.
    :param dict[str, str] storage_owners: Catalog ids keyed by Optuna study namespace.
    :raises CatalogError: If another catalog id already governs either namespace.
    """
    namespace = str(_experiment_dir(loaded.experiment).resolve())
    other = namespace_owners.get(namespace)
    if other is not None:
        raise CatalogError(
            f"catalog ids {other!r} and {loaded.id!r} resolve to the same "
            f"experiment output namespace ({namespace}); one engine "
            "experiment must be governed by exactly one catalog entry."
        )
    namespace_owners[namespace] = loaded.id

    storage_identity = canonical_storage_identity(loaded.experiment.resolved_storage)
    if storage_identity is None:
        return
    storage_key = f"{storage_identity}::{loaded.experiment.experiment}"
    other = storage_owners.get(storage_key)
    if other is not None:
        raise CatalogError(
            f"catalog ids {other!r} and {loaded.id!r} resolve to the same "
            f"Optuna study namespace ({loaded.experiment.experiment!r} on "
            f"{storage_identity}); one engine experiment must be governed "
            "by exactly one catalog entry."
        )
    storage_owners[storage_key] = loaded.id


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


def _load_catalog_entries(
    catalog: _Catalog,
    base: Path,
    *,
    collect_errors: bool,
) -> tuple[dict[str, RegisteredExperiment], tuple[CatalogCheckEntry, ...]]:
    """Validate catalog entries under the check or startup error policy.

    :param _Catalog catalog: Parsed catalog whose entries should be loaded.
    :param Path base: Catalog directory used to resolve relative paths.
    :param bool collect_errors: Collect one verdict per entry when true; raise
        the first :class:`CatalogError` when false.
    :return tuple: Successfully loaded entries keyed by id and any collected
        check verdicts, in catalog order.
    :raises CatalogError: The first entry error when ``collect_errors`` is false.
    """
    items: dict[str, RegisteredExperiment] = {}
    verdicts: list[CatalogCheckEntry] = []
    seen: set[str] = set()
    # Two catalog ids must never govern one engine experiment: the MCP busy
    # guard keys on the id string while the engine's locks key on the output
    # namespace and storage identity.
    namespace_owners: dict[str, str] = {}
    storage_owners: dict[str, str] = {}
    for entry in catalog.experiments:
        try:
            if entry.id in seen:
                raise CatalogError(f"duplicate catalog id {entry.id!r}")
            seen.add(entry.id)
            registered = _load_entry(base, entry)
            _claim_catalog_entry_identity(registered, namespace_owners, storage_owners)
        except CatalogError as exc:
            if not collect_errors:
                raise
            verdicts.append(CatalogCheckEntry(entry.id, error=str(exc), suggestion=exc.suggestion))
            continue

        items[entry.id] = registered
        if collect_errors:
            actions = tuple(
                action
                for action, allowed in (
                    ("launch", registered.allow_launch),
                    ("cancel", registered.allow_cancel),
                    ("from_phase", registered.allow_from_phase),
                )
                if allowed
            )
            verdicts.append(CatalogCheckEntry(entry.id, actions=actions))
    return items, tuple(verdicts)


def check_catalog(catalog_path: Path) -> CatalogCheckReport:
    """Validate every catalog entry, collecting per-entry verdicts.

    Runs the same validation as :meth:`Registry.load` (shared per-entry code
    path) but does not stop at the first failure, so ``phasesweep mcp check``
    can report a full ok/FAIL table. Once every entry passes, it also runs the
    exact state-directory provisioning and write probe used at server startup.
    Catalog-level problems (unreadable file, schema errors, or unusable state)
    still raise :class:`CatalogError`.

    :param Path catalog_path: Path to the operator-authored catalog YAML.
    :return CatalogCheckReport: One verdict per catalog entry, in catalog order.
    """
    require_linux_mcp_host()
    catalog, base = _parse_catalog(catalog_path)
    _, verdicts = _load_catalog_entries(catalog, base, collect_errors=True)
    report = CatalogCheckReport(entries=verdicts)
    if report.ok:
        _prepare_state_dir(base, catalog.state_dir)
    return report


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

        Every problem is reported as ``CatalogError`` so the server refuses to
        start with a bad catalog. Per entry: the config path exists,
        ``load_config`` accepts it, it is an :class:`Experiment` (suites are out
        of scope for v1), and its storage is a persistent local SQLite/Journal
        file (the MCP layer is local-node only in this version).

        :param Path catalog_path: Path to the operator-authored catalog YAML.
        :return Registry: Immutable registry of validated catalog entries.
        :raises CatalogError: If the host is not a supported Linux MCP host, the
            catalog cannot be parsed, an id is duplicated, an entry fails
            validation, two entries claim one engine namespace, or the state
            directory cannot be prepared.
        """
        require_linux_mcp_host()
        catalog, base = _parse_catalog(catalog_path)
        items, _ = _load_catalog_entries(catalog, base, collect_errors=False)
        return cls(
            state_dir=_prepare_state_dir(base, catalog.state_dir),
            items=items,
            max_concurrent_runs=catalog.max_concurrent_runs,
        )

    def get(self, experiment_id: str) -> RegisteredExperiment:
        """Look up a registered experiment by id, or raise ``UnknownExperimentError``.

        :param str experiment_id: Agent-visible catalog id.
        :return RegisteredExperiment: Validated registry entry for the id.
        :raises UnknownExperimentError: If no catalog entry has that id.
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
