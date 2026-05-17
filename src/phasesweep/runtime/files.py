"""Filesystem, lock, timeout, and storage-URL runtime helpers."""

from __future__ import annotations

import contextlib
import os
import queue
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any, TypeVar

T = TypeVar("T")


POSIX_RUNTIME_ERROR = (
    "phasesweep execution currently requires a POSIX platform. It relies on "
    "fcntl.flock host locks and POSIX process groups for safe subprocess cleanup; "
    "Windows support needs a separate locking and process-tree implementation."
)


def require_posix_runtime() -> None:
    """Raise a clear error when execution is attempted on an unsupported platform."""
    if not _supports_posix_runtime_features():
        raise RuntimeError(POSIX_RUNTIME_ERROR)


def _supports_posix_runtime_features(
    *,
    os_name: str | None = None,
    has_killpg: bool | None = None,
    has_fcntl: bool | None = None,
) -> bool:
    """Return whether the runtime has the Unix features phasesweep needs.

    :param str | None os_name: Optional `os.name` override for tests.
    :param bool | None has_killpg: Optional `os.killpg` availability override.
    :param bool | None has_fcntl: Optional `fcntl` import availability override.
    :return bool: Whether execution can rely on POSIX process groups and `flock`.
    """
    if os_name is None:
        os_name = os.name
    if has_killpg is None:
        has_killpg = hasattr(os, "killpg")
    if has_fcntl is None:
        has_fcntl = _fcntl_available()
    return os_name == "posix" and has_killpg and has_fcntl


def _fcntl_available() -> bool:
    """Return whether the Unix ``fcntl`` module can be imported.

    :return bool: `True` when `fcntl` is importable in the current interpreter.
    """
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return False
    return True


def call_with_timeout(fn: Callable[[], T], *, timeout: float) -> T:
    """Run a blocking function in a daemon thread and bound caller wait time."""
    q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            q.put((True, fn()))
        except Exception as exc:  # noqa: BLE001
            q.put((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, timeout))
    if thread.is_alive():
        raise TimeoutError(f"call exceeded {timeout:g}s")
    ok, value = q.get_nowait()
    if ok:
        return value
    raise value


def lock_dir() -> Path:
    """Return the shared same-host phasesweep lock directory."""
    path = Path(tempfile.gettempdir()) / "phasesweep-locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def try_lock_file(path: Path) -> IO[str] | None:
    """Open ``path`` without truncating and take an exclusive flock, or return ``None``."""
    require_posix_runtime()
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the file if needed, but preserve holder diagnostics until we own it.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    handle = os.fdopen(fd, "r+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def unlock_file(handle: IO[str]) -> None:
    """Release and close a handle returned by :func:`try_lock_file`."""
    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(handle, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        handle.close()


@contextlib.contextmanager
def exclusive_lock(path: Path, *, busy_message: str) -> Iterator[None]:
    """Hold a non-blocking exclusive flock for the context duration."""
    handle = try_lock_file(path)
    if handle is None:
        raise RuntimeError(busy_message)
    try:
        yield
    finally:
        unlock_file(handle)


def storage_backend(storage: str | None) -> str | None:
    """Return the logical backend name for an Optuna storage URL.

    Args:
        storage: An Optuna storage URL (e.g. ``"sqlite:///x.db"``,
            ``"journal:///x.journal"``, ``"postgresql+psycopg2://..."``),
            or ``None`` for in-memory storage.

    Returns:
        The dialect-collapsed scheme (``"sqlite"``, ``"journal"``,
        ``"postgresql"``, ...), or ``None`` if ``storage`` is ``None``.

    Examples:
        >>> storage_backend("sqlite:///x.db")
        'sqlite'
        >>> storage_backend("sqlite+pysqlite:///x.db")
        'sqlite'
        >>> storage_backend("postgresql+psycopg2://user@host/db")
        'postgresql'
        >>> storage_backend("journal:///x.journal")
        'journal'
        >>> storage_backend(None) is None
        True

    """
    if storage is None:
        return None
    scheme = storage.split(":", 1)[0].lower()
    return scheme.split("+", 1)[0]


def file_url_path(storage: str) -> str:
    """Return the filesystem path component of a phasesweep file-style URL.

    SQLAlchemy's URL grammar for file-based backends uses three slashes for
    relative paths and four for absolute POSIX paths (the fourth slash is the
    root ``/``). We must preserve that distinction; ``lstrip("/")`` would
    destroy absolute paths (review v0.5.9 / blocker 1).

    Supported forms::

        sqlite:///relative.db             -> relative.db
        sqlite:////tmp/absolute.db        -> /tmp/absolute.db
        sqlite+pysqlite:///relative.db    -> relative.db
        sqlite+pysqlite:////tmp/x.db      -> /tmp/x.db
        sqlite://                         -> ""
        sqlite:///:memory:                -> :memory:
        journal:///relative.journal       -> relative.journal
        journal:////tmp/absolute.journal  -> /tmp/absolute.journal

    Args:
        storage: A file-style storage URL whose scheme is already known to be
            file-based (``sqlite``, ``journal``).

    Returns:
        The bare filesystem path (or sentinel like ``":memory:"``), without
        the scheme or leading slashes that belong to URL grammar.

    """
    rest = storage.split(":", 1)[1]

    if rest.startswith("////"):
        # POSIX absolute file path. The fourth slash IS the root ``/``.
        return "/" + rest[4:]

    if rest.startswith("///"):
        # Relative file path (or :memory: sentinel) under SQLAlchemy grammar.
        return rest[3:]

    if rest.startswith("//"):
        # Handles bare ``sqlite://`` (in-memory shorthand).
        return rest[2:]

    return rest


def canonical_storage_identity(storage: str | None) -> str | None:
    """Stable same-host identity string for a storage URL.

    File-based backends (SQLite, JournalStorage) are resolved to absolute
    paths so two configs that differ only in relative vs. absolute spelling
    still collide on the lock file. SQLite URLs additionally fold their
    SQLAlchemy dialect (``sqlite+pysqlite:///`` etc.) onto the canonical
    ``sqlite:///`` prefix so dialect choice never splits the lock.

    Args:
        storage: An Optuna storage URL, or ``None`` for in-memory storage.

    Returns:
        The canonical identity string used to derive the same-host storage
        lock path, or ``None`` for in-memory storage (no shared backend to
        collide on). Non-file RDB URLs are returned unchanged.

    """
    if storage is None:
        return None

    backend = storage_backend(storage)

    if backend == "sqlite":
        database = file_url_path(storage)
        if database in ("", ":memory:"):
            return "sqlite:///:memory:"
        return "sqlite:///" + str(Path(database).expanduser().resolve())

    if backend == "journal":
        path = file_url_path(storage)
        return "journal:///" + str(Path(path).expanduser().resolve())

    # RDB URLs (postgres, mysql, ...) are passed through.
    return storage
