"""Filesystem, lock, timeout, and storage-URL runtime helpers."""

from __future__ import annotations

import contextlib
import os
import queue
import stat
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any, TypeVar
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

T = TypeVar("T")


POSIX_RUNTIME_ERROR = (
    "phasesweep execution currently requires a POSIX platform. It relies on "
    "fcntl.flock host locks and POSIX process groups for safe subprocess cleanup; "
    "Windows support needs a separate locking and process-tree implementation."
)
_LOCK_DIR_ENV = "PHASESWEEP_LOCK_DIR"
_DEFAULT_LOCK_DIR = Path("/var/tmp") / "phasesweep-locks"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


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
    """Run a blocking function in a daemon thread and bound caller wait time.

    :param Callable[[], T] fn: Zero-argument function to execute.
    :param float timeout: Maximum number of seconds to wait for completion.
    :raises TimeoutError: If ``fn`` does not complete before ``timeout`` elapses.
    :return T: Value returned by ``fn``.
    """
    q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        """Execute ``fn`` and store either its value or raised exception."""
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
    """Return the shared same-host phasesweep lock directory.

    ``PHASESWEEP_LOCK_DIR`` can override the location for schedulers,
    containers, and managed hosts. The default deliberately avoids
    ``tempfile.gettempdir()`` because job schedulers commonly set per-job
    ``TMPDIR`` values, which would split the lock namespace and break
    cross-process GPU/output/storage exclusion.

    :return Path: Directory used for host-local lock files.
    """
    override = os.environ.get(_LOCK_DIR_ENV)
    path = Path(override) if override else _DEFAULT_LOCK_DIR
    created = False
    try:
        path.mkdir(parents=True, exist_ok=False)
        created = True
    except FileExistsError:
        if not path.is_dir():
            raise

    if override is None or created:
        with contextlib.suppress(OSError):
            path.chmod(0o1777)
    return path


def try_lock_file(path: Path) -> IO[str] | None:
    """Open ``path`` without truncating and take an exclusive flock, or return ``None``.

    :param Path path: Lock file path to open or create.
    :return IO[str] | None: Open lock handle when acquired, otherwise ``None``.
    """
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
    """Hold a non-blocking exclusive flock for the context duration.

    :param Path path: Lock file path to hold during the context.
    :param str busy_message: Error message used when the lock is already held.
    :raises RuntimeError: If the lock cannot be acquired immediately.
    :return Iterator[None]: Context manager iterator for the held lock.
    """
    handle = try_lock_file(path)
    if handle is None:
        raise RuntimeError(busy_message)
    try:
        yield
    finally:
        unlock_file(handle)


def fsync_directory(path: Path) -> None:
    """Best-effort fsync for a directory after an atomic replace."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_private_dir(path: Path) -> None:
    """Create ``path`` and remove group/other access from an existing directory.

    ``mkdir(mode=...)`` is still filtered through the process umask. MCP state
    directories hold operator-only logs, config snapshots, and status files, so
    creation and reuse both converge to a private mode and fail if that cannot
    be enforced.

    :param Path path: Directory that must be accessible only by the owner.
    :raises PermissionError: If group/other access remains after chmod.
    """
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        path.chmod(mode & ~0o077)
    final_mode = stat.S_IMODE(path.stat().st_mode)
    if final_mode & 0o077:
        raise PermissionError(f"{path} is accessible by group or other users")


def open_private_text(path: Path, mode: str = "w") -> IO[str]:
    """Open a UTF-8 text file with owner-only permissions.

    :param Path path: File to open.
    :param str mode: Either ``"w"`` or ``"a"``.
    :return IO[str]: Open text handle.
    :raises ValueError: If ``mode`` is unsupported.
    """
    if mode not in {"w", "a"}:
        raise ValueError(f"unsupported private text mode: {mode!r}")
    ensure_private_dir(path.parent)
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if mode == "w" else os.O_APPEND
    fd = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        return os.fdopen(fd, mode, encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


@contextlib.contextmanager
def private_atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[IO[str]]:
    """Atomically replace a private UTF-8 text file.

    :param Path path: Destination path to replace.
    :param str | None newline: Newline handling passed to ``open``.
    :return Iterator[IO[str]]: Writable text handle.
    """
    ensure_private_dir(path.parent)
    tmp_path: Path | None = None
    replaced = False
    try:
        fd, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            tmp_path = Path(name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        assert tmp_path is not None
        os.replace(tmp_path, path)
        os.chmod(path, PRIVATE_FILE_MODE)
        replaced = True
        fsync_directory(path.parent)
    finally:
        if tmp_path is not None and not replaced:
            tmp_path.unlink(missing_ok=True)


def private_atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a private UTF-8 text file.

    :param Path path: Destination path to replace; the parent directory is
        created and locked down to owner-only access if needed.
    :param str text: Text to write using UTF-8 encoding.
    """
    with private_atomic_text_writer(path) as handle:
        handle.write(text)


@contextlib.contextmanager
def private_atomic_bytes_writer(path: Path) -> Iterator[IO[bytes]]:
    """Atomically replace a private bytes file.

    :param Path path: Destination path to replace.
    :return Iterator[IO[bytes]]: Writable binary handle.
    """
    ensure_private_dir(path.parent)
    tmp_path: Path | None = None
    replaced = False
    try:
        fd, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            tmp_path = Path(name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        assert tmp_path is not None
        os.replace(tmp_path, path)
        os.chmod(path, PRIVATE_FILE_MODE)
        replaced = True
        fsync_directory(path.parent)
    finally:
        if tmp_path is not None and not replaced:
            tmp_path.unlink(missing_ok=True)


def private_atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a private bytes file.

    :param Path path: Destination path to replace; the parent directory is
        created and locked down to owner-only access if needed.
    :param bytes data: Bytes to write.
    """
    with private_atomic_bytes_writer(path) as handle:
        handle.write(data)


@contextlib.contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[IO[str]]:
    """Write text through a same-directory temp file and atomically replace ``path``.

    :param Path path: Destination path that should be replaced atomically.
    :param str | None newline: Newline handling passed to ``NamedTemporaryFile``.
    :return Iterator[IO[str]]: Writable text handle yielded for the caller to populate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline=newline,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        assert tmp_path is not None
        os.replace(tmp_path, path)
        replaced = True
        fsync_directory(path.parent)
    finally:
        if tmp_path is not None and not replaced:
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with UTF-8 text.

    :param Path path: Destination path to replace.
    :param str text: Text to write using UTF-8 encoding.
    """
    with atomic_text_writer(path) as handle:
        handle.write(text)


@contextlib.contextmanager
def atomic_bytes_writer(path: Path) -> Iterator[IO[bytes]]:
    """Write bytes through a same-directory temp file and atomically replace ``path``.

    :param Path path: Destination path to replace when the context exits successfully.
    :return Iterator[IO[bytes]]: Writable binary file handle for the temporary file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        assert tmp_path is not None
        os.replace(tmp_path, path)
        replaced = True
        fsync_directory(path.parent)
    finally:
        if tmp_path is not None and not replaced:
            tmp_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with bytes.

    :param Path path: Destination path to replace.
    :param bytes data: Bytes to write.
    """
    with atomic_bytes_writer(path) as handle:
        handle.write(data)


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
        sqlite:///relative.db?timeout=30  -> relative.db
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
        path = "/" + rest[4:]
    elif rest.startswith("///"):
        # Relative file path (or :memory: sentinel) under SQLAlchemy grammar.
        path = rest[3:]
    elif rest.startswith("//"):
        # Handles bare ``sqlite://`` (in-memory shorthand).
        path = rest[2:]
    else:
        path = rest

    path = path.split("?", 1)[0]
    return path.split("#", 1)[0]


def _url_query_pairs(storage: str) -> list[tuple[str, str]]:
    """Return URL query pairs from a storage URL.

    :param str storage: Storage URL whose query string should be parsed.
    :return list[tuple[str, str]]: Ordered query key/value pairs, preserving blank values.
    """
    query = storage.split("?", 1)[1].split("#", 1)[0] if "?" in storage else ""
    return parse_qsl(query, keep_blank_values=True)


def storage_url_query_options(storage: str) -> dict[str, str]:
    """Return lower-cased URL query options for storage policy checks.

    :param str storage: Storage URL whose query string should be inspected.
    :return dict[str, str]: Query parameters with lower-cased keys and values.
    """
    return {key.lower(): value.lower() for key, value in _url_query_pairs(storage)}


def _truthy_url_option(value: str | None) -> bool:
    """Return whether a URL query value opts into a boolean behavior.

    :param str | None value: Query value to interpret as a boolean opt-in.
    :return bool: True when the value is a recognized truthy token.
    """
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _sqlite_uri_filename_enabled(storage: str, database: str | None = None) -> bool:
    """Return whether SQLAlchemy will treat a SQLite ``file:`` path as a URI.

    :param str storage: SQLite storage URL to inspect.
    :param str | None database: Optional already-parsed database filename from the storage URL.
    :return bool: True when the database uses a ``file:`` filename and the URL sets ``uri=true``.
    """
    database = file_url_path(storage) if database is None else database
    if not database.startswith("file:"):
        return False
    return _truthy_url_option(storage_url_query_options(storage).get("uri"))


def sqlite_uri_filename_path(storage: str) -> str | None:
    """Return the local filesystem path named by a SQLite URI filename.

    SQLAlchemy only treats a SQLite ``file:`` database string as a URI filename
    when ``uri=true`` is present in the URL query. Without that flag,
    ``sqlite:///file:literal.db`` is a literal filename and must keep the
    ``file:`` prefix.

    :param str storage: SQLite storage URL.
    :return str | None: Decoded local filesystem path from the URI filename, or
        ``None`` when the storage URL is not a local SQLite URI filename.
    """
    database = file_url_path(storage)
    if not _sqlite_uri_filename_enabled(storage, database):
        return None

    parsed = urlsplit(database)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in {"", "localhost"}:
        return None
    return unquote(parsed.path)


def storage_is_in_memory(storage: str | None) -> bool:
    """Return whether ``storage`` names an in-memory Optuna backend.

    :param str | None storage: Optuna storage URL, SQLite sentinel, or ``None``.
    :return bool: ``True`` when the storage has no durable file or external backend.
    """
    if storage is None:
        return True
    if storage == ":memory:":
        return True
    if storage_backend(storage) != "sqlite":
        return False

    database = file_url_path(storage)
    if database in {"", ":memory:"}:
        return True
    if not _sqlite_uri_filename_enabled(storage, database):
        return False

    options = storage_url_query_options(storage)
    uri_path = sqlite_uri_filename_path(storage)
    return (
        uri_path in {"", ":memory:"}
        or database.startswith("file::memory:")
        or options.get("mode") == "memory"
    )


def sqlite_readonly_uri(storage: str) -> str | None:
    """Build a ``sqlite3.connect(..., uri=True)`` URI for read-only status reads.

    The returned URI opens the configured persistent SQLite database in
    ``mode=ro`` so status polling cannot create a missing DB or schema. SQLite
    URI filenames such as ``sqlite:///file:/tmp/x.db?mode=rwc&uri=true`` are
    preserved as URI filenames with the write mode replaced by ``mode=ro``.

    :param str storage: SQLite storage URL.
    :return str | None: Read-only SQLite URI, or ``None`` for in-memory storage.
    """
    if storage_is_in_memory(storage):
        return None

    database = file_url_path(storage)
    if _sqlite_uri_filename_enabled(storage, database):
        params = [
            (key, value)
            for key, value in _url_query_pairs(storage)
            if key.lower() not in {"mode", "uri"}
        ]
        params.append(("mode", "ro"))
        return f"{database}?{urlencode(params)}"

    path = Path(database).expanduser().resolve()
    return f"file:{quote(str(path), safe='/')}?mode=ro"


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
        uri_path = sqlite_uri_filename_path(storage)
        if _sqlite_uri_filename_enabled(storage, database):
            if storage_is_in_memory(storage):
                return "sqlite:///:memory:"
            if uri_path is None:
                params = [
                    (key, value)
                    for key, value in _url_query_pairs(storage)
                    if key.lower() not in {"mode", "uri"}
                ]
                query = urlencode(params)
                return f"sqlite-uri:{database}" + (f"?{query}" if query else "")
            database = uri_path
        if database in ("", ":memory:"):
            return "sqlite:///:memory:"
        return "sqlite:///" + str(Path(database).expanduser().resolve())

    if backend == "journal":
        path = file_url_path(storage)
        return "journal:///" + str(Path(path).expanduser().resolve())

    # RDB URLs (postgres, mysql, ...) are passed through.
    return storage
