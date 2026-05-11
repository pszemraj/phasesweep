"""Shared parsing utilities for Optuna storage URLs.

Two callers need the same notion of "what backend is this URL?":

* :func:`phasesweep.config._validate_storage_policy` — must reject SQLite under
  ``n_jobs > 1`` regardless of which SQLAlchemy dialect was spelled
  (``sqlite:///``, ``sqlite+pysqlite:///``, ``sqlite+pysqlcipher:///``, ...).
* :func:`phasesweep.orchestrator._canonical_storage_identity` — must produce
  the same lock-identity string for two URLs that point at the same SQLite
  file even if their dialect differs, so the same-host lock collides.

Pre-v0.5.8 each caller hand-rolled ``startswith("sqlite:///")`` and missed
driver-qualified URLs (review v0.5.7 / blocker 1). Centralizing the parser
removes the duplication and the bypass.

The implementation deliberately avoids importing SQLAlchemy directly: scheme
parsing is trivial, and we want config validation to keep working even if
SQLAlchemy is later trimmed from the dependency tree.
"""

from __future__ import annotations

from pathlib import Path


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


def _file_url_path(storage: str) -> str:
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
        database = _file_url_path(storage)
        if database in ("", ":memory:"):
            return "sqlite:///:memory:"
        return "sqlite:///" + str(Path(database).expanduser().resolve())

    if backend == "journal":
        path = _file_url_path(storage)
        return "journal:///" + str(Path(path).expanduser().resolve())

    # RDB URLs (postgres, mysql, ...) are passed through.
    return storage
