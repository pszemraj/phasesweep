"""Filesystem, lock, timeout, and storage-URL runtime helpers."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

log = logging.getLogger("phasesweep.runtime.files")

POSIX_RUNTIME_ERROR = (
    "phasesweep execution currently requires a POSIX platform. It relies on "
    "fcntl.flock host locks and POSIX process groups for safe subprocess cleanup; "
    "Windows support needs a separate locking and process-tree implementation."
)
_LOCK_DIR_ENV = "PHASESWEEP_LOCK_DIR"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SHARED_DIR_MODE = 0o3770
SHARED_FILE_MODE = 0o660


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without holding all bytes in memory.

    :param Path path: File to hash.
    :return str: 64-character hexadecimal digest.
    :raises OSError: The file cannot be read.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class UnsafeLockPathError(RuntimeError):
    """Raised when a lock directory or file is not safe to trust."""


class UnsafePrivatePathError(RuntimeError):
    """Raised when a private directory or file is not safe to mutate."""


class PlatformCapabilityError(RuntimeError):
    """Raised when the host lacks a capability required for safe operation."""


@dataclass(frozen=True)
class _LockPolicy:
    """Ownership and mode expected for one lock namespace."""

    shared: bool
    uid: int
    gid: int
    file_mode: int


def require_posix_runtime() -> None:
    """Raise a clear error when execution is attempted on an unsupported platform.

    :raises PlatformCapabilityError: The host is not POSIX or lacks
        ``os.killpg``/``fcntl``, so process groups and ``flock`` are
        unavailable.
    """
    if not _supports_posix_runtime_features():
        raise PlatformCapabilityError(POSIX_RUNTIME_ERROR)


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


def lock_dir() -> Path:
    """Return the validated same-host phasesweep lock directory.

    The default is private to the current user. ``PHASESWEEP_LOCK_DIR`` selects
    an existing operator-provisioned directory and is never created or chmodded
    by phasesweep.

    :return Path: Directory used for host-local lock files.
    :raises UnsafeLockPathError: ``PHASESWEEP_LOCK_DIR`` is relative, missing,
        contains a symlinked component, or fails the private/shared
        ownership-and-mode check; or the default lock directory under the
        user's cache cannot be created as owner-only.
    """
    override = os.environ.get(_LOCK_DIR_ENV)
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise UnsafeLockPathError(f"{_LOCK_DIR_ENV} must be an absolute path: {path}")
        _lock_policy(path)
        return path

    path = Path.home() / ".cache" / "phasesweep" / "locks"
    try:
        ensure_private_dir(path)
    except UnsafePrivatePathError as exc:
        raise UnsafeLockPathError(f"Default lock directory {path} is unsafe.") from exc
    return path


def _lock_policy(path: Path) -> _LockPolicy:
    """Open, validate, and classify a lock directory as private or shared.

    Walks ``path`` with :func:`open_directory_fd` (no ``O_NOFOLLOW`` bypass,
    directory not created) and hands the resulting descriptor to
    :func:`_lock_policy_for_info` for the ownership/mode check; the
    descriptor is always closed before returning.

    :param Path path: Lock directory to validate.
    :return _LockPolicy: Sharing policy (private vs. group-shared) with the
        owner uid/gid and lock-file mode this directory requires.
    :raises UnsafeLockPathError: If ``path`` does not exist, contains a
        symlinked component, or fails the private/shared ownership-and-mode
        check.
    """
    try:
        fd = open_directory_fd(path, create=False, private_final=False)
    except FileNotFoundError as exc:
        raise UnsafeLockPathError(
            f"Lock directory {path} does not exist; provision it before setting {_LOCK_DIR_ENV}."
        ) from exc
    except UnsafePrivatePathError as exc:
        raise UnsafeLockPathError(f"Lock directory {path} has an unsafe path component.") from exc
    try:
        return _lock_policy_for_info(path, os.fstat(fd))
    finally:
        os.close(fd)


def _lock_policy_for_info(path: Path, info: os.stat_result) -> _LockPolicy:
    """Classify an already-opened lock directory's ownership and mode.

    Accepts either an owner-only directory (mode ``0700``, owned by the
    current effective uid) or an administrator-owned shared directory (mode
    ``03770``, owned by uid 0, with a gid in the caller's current group
    set). Any other owner/mode combination — including a non-directory,
    which indicates the path resolved to a symlink or other non-directory
    entry — is refused.

    :param Path path: Lock directory the ``info`` was captured from, used
        only for error messages.
    :param os.stat_result info: ``stat`` result of the already-opened
        directory descriptor.
    :return _LockPolicy: Sharing policy: ``shared=False`` with owner-uid
        file mode ``0600`` for a private directory, or ``shared=True`` with
        group file mode ``0660`` for a shared directory.
    :raises UnsafeLockPathError: If ``info`` is not a real directory or its
        owner/mode does not match either the private or shared policy.
    """
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeLockPathError(f"Lock directory {path} must be a real directory, not a symlink.")

    mode = stat.S_IMODE(info.st_mode)
    euid = os.geteuid()
    if info.st_uid == euid and mode == PRIVATE_DIR_MODE:
        return _LockPolicy(False, info.st_uid, info.st_gid, PRIVATE_FILE_MODE)

    groups = {os.getegid(), *os.getgroups()}
    if info.st_uid == 0 and info.st_gid in groups and mode == SHARED_DIR_MODE:
        return _LockPolicy(True, info.st_uid, info.st_gid, SHARED_FILE_MODE)

    raise UnsafeLockPathError(
        f"Unsafe lock directory {path}: expected owner-only mode 0700 owned by uid {euid}, "
        "or an administrator-owned shared directory with mode 03770 and an accessible group."
    )


def open_lock_file(path: Path) -> IO[str]:
    """Open or create a validated regular lock file inside a validated lock directory.

    Requires a POSIX runtime and ``O_NOFOLLOW``. Walks and validates
    ``path.parent`` as a private-or-shared lock directory (see
    :func:`_lock_policy_for_info`), then opens ``path`` relative to that
    directory descriptor with ``O_NOFOLLOW`` so a symlinked lock path is
    refused rather than followed. If the file does not exist it is created
    exclusively at the policy's file mode; if it already exists, it is
    opened as-is and re-validated: it must be a regular file with a single
    hard link, owned by the expected uid (private policy) or gid (shared
    policy), and already at the policy's file mode. The parent directory
    descriptor is always closed before returning; the returned file handle
    is closed automatically if any validation step after opening fails.

    :param Path path: Lock file path to open or create.
    :return IO[str]: Text-mode (``"r+"``, UTF-8) handle open on the
        validated lock file.
    :raises PlatformCapabilityError: If the platform lacks ``O_NOFOLLOW``.
    :raises UnsafeLockPathError: If the parent directory is missing or unsafe,
        the leaf name is unsafe, or the file fails the regular-file/ownership/
        mode checks.
    """
    from phasesweep.runtime.process import defer_shutdown_signals

    with defer_shutdown_signals():
        return _open_lock_file(path)


def _open_lock_file(path: Path) -> IO[str]:
    """Open and validate a lock file while shutdown is deferred.

    Unlocked core of :func:`open_lock_file`, called from within its
    ``defer_shutdown_signals`` context.

    :param Path path: Lock file path to open or create.
    :return IO[str]: Text-mode (``"r+"``, UTF-8) handle open on the
        validated lock file.
    :raises PlatformCapabilityError: If the platform lacks ``O_NOFOLLOW``.
    :raises UnsafeLockPathError: If the parent directory is missing or unsafe,
        the leaf name is unsafe, or the file fails the regular-file/ownership/
        mode checks.
    """
    require_posix_runtime()
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise PlatformCapabilityError(
            "This platform cannot safely open lock files without following symlinks."
        )
    try:
        parent_fd = open_directory_fd(path.parent, create=False, private_final=False)
    except (FileNotFoundError, UnsafePrivatePathError) as exc:
        raise UnsafeLockPathError(f"Lock directory {path.parent} is not safe to open.") from exc
    try:
        policy = _lock_policy_for_info(path.parent, os.fstat(parent_fd))
        try:
            leaf = leaf_name(path)
        except UnsafePrivatePathError as exc:
            raise UnsafeLockPathError(f"Lock path {path} has no safe filename.") from exc
        flags = os.O_RDWR | os.O_CLOEXEC | nofollow
        created = False
        try:
            fd = os.open(
                leaf,
                flags | os.O_CREAT | os.O_EXCL,
                policy.file_mode,
                dir_fd=parent_fd,
            )
            created = True
        except FileExistsError:
            try:
                fd = os.open(leaf, flags, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise UnsafeLockPathError(f"Lock path {path} must not be a symlink.") from exc
                raise
    finally:
        os.close(parent_fd)

    try:
        info = os.fstat(fd)
        mode = stat.S_IMODE(info.st_mode)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise UnsafeLockPathError(f"Lock path {path} must be a regular file with one link.")
        if policy.shared:
            if info.st_gid != policy.gid:
                raise UnsafeLockPathError(
                    f"Shared lock {path} has gid {info.st_gid}; expected {policy.gid}."
                )
        elif info.st_uid != policy.uid:
            raise UnsafeLockPathError(
                f"Private lock {path} is owned by uid {info.st_uid}; expected {policy.uid}."
            )
        if created:
            os.fchmod(fd, policy.file_mode)
            mode = policy.file_mode
        if mode != policy.file_mode:
            raise UnsafeLockPathError(
                f"Lock file {path} has mode {mode:04o}; expected {policy.file_mode:04o}."
            )
        return os.fdopen(fd, "r+", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


def try_lock_file(path: Path) -> IO[str] | None:
    """Open ``path`` without truncating and take an exclusive flock, or return ``None``.

    :param Path path: Lock file path to open or create.
    :return IO[str] | None: Open lock handle when acquired, otherwise ``None``.
    """
    require_posix_runtime()
    import fcntl

    handle = open_lock_file(path)
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
    """Best-effort fsync for a directory after an atomic replace or create.

    Never raises. By the time this runs, the caller's rename or exclusive
    create has already changed the destination, so a directory-durability
    failure here must not be reported as a failed commit: a caller reacting
    to the exception would wrongly reclassify an already-authoritative write
    (e.g. a committed last-success pointer marked ``publication_failed``,
    review v0.5.16 / blocker 1). The failure is logged as a durability
    warning instead — the commit stands; only its crash-durability is
    uncertain.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        log.warning(
            "Directory fsync failed after an atomic write in %s; the write is "
            "committed but its durability across a crash is uncertain.",
            path,
            exc_info=True,
        )
    finally:
        os.close(fd)


def nofollow_flag() -> int:
    """Return ``O_NOFOLLOW`` or fail when safe private traversal is unavailable.

    :return int: The platform's ``os.O_NOFOLLOW`` open flag.
    :raises PlatformCapabilityError: The platform does not define
        ``O_NOFOLLOW``, so private files cannot be opened without following
        symlinks.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise PlatformCapabilityError(
            "This platform cannot safely access private files without following symlinks."
        )
    return nofollow


def absolute_path(path: Path) -> Path:
    """Return a lexical absolute path without resolving symlinks.

    Anchors a relative path at the current working directory, then collapses
    ``.`` and ``..`` components purely by name, matching
    :func:`posixpath.normpath`. This must not touch the filesystem: callers
    such as :func:`open_directory_fd` walk the resulting components with
    ``O_NOFOLLOW`` specifically to detect symlinks, so resolving them here
    would defeat that check.

    :param Path path: Candidate path, absolute or relative to the current
        working directory.
    :return Path: Absolute path with ``.`` and ``..`` components collapsed
        lexically; a ``..`` above the root stays at the root.
    """
    anchored = path if path.is_absolute() else Path.cwd() / path
    collapsed: list[str] = []
    for part in anchored.parts[1:]:
        if part == ".":
            continue
        if part == "..":
            if collapsed:
                collapsed.pop()
            continue
        collapsed.append(part)
    return Path(anchored.parts[0], *collapsed)


def leaf_name(path: Path) -> str:
    """Return the final path component, refusing names with no fixed identity.

    :param Path path: Path whose final component is extracted.
    :return str: The final path component (``path.name``).
    :raises UnsafePrivatePathError: If the final component is empty, ``"."``,
        or ``".."`` — none of which name a single, unambiguous filesystem
        entry safe to open relative to a parent directory descriptor.
    """
    leaf = path.name
    if leaf in {"", ".", ".."}:
        raise UnsafePrivatePathError(f"Path {path} has no safe final component.")
    return leaf


def _validate_private_dir_info(path: Path, info: os.stat_result) -> None:
    """Validate that an opened directory is owner-only, without modifying it.

    :param Path path: Directory the ``info`` was captured from, used only
        for error messages.
    :param os.stat_result info: ``stat`` result of the already-opened
        directory descriptor.
    :raises UnsafePrivatePathError: If the entry is not a directory, or is
        not owned by the current effective uid with mode ``0700``.
    """
    mode = stat.S_IMODE(info.st_mode)
    euid = os.geteuid()
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafePrivatePathError(f"Private directory {path} is not a directory.")
    if info.st_uid != euid or mode != PRIVATE_DIR_MODE:
        raise UnsafePrivatePathError(
            f"Private directory {path} must be owned by uid {euid} with mode 0700; "
            f"found uid {info.st_uid} and mode {mode:04o}."
        )


def open_directory_fd(
    path: Path,
    *,
    create: bool,
    private_final: bool,
    umask_created_dirs: bool = False,
) -> int:
    """Open a directory by walking every path component relative to its parent, refusing symlinks.

    Starts from an ``O_DIRECTORY`` descriptor on ``/`` and, for each
    component, ``lstat``s it relative to the currently-held parent
    descriptor, opens it with ``O_NOFOLLOW`` relative to that same
    descriptor, then compares the pre-open ``lstat`` and post-open
    ``fstat`` device/inode to detect a symlink swapped in between
    (TOCTOU). A component that is not a real directory — including a
    symlink — raises, as does a component that changed identity mid-open.

    A missing component is created only when ``create`` is true. By default,
    it is forced to owner-only mode ``0700`` and validated accordingly;
    ``umask_created_dirs=True`` instead requests mode ``0777`` and leaves the
    resulting permissions to the process umask for non-private config paths.
    When ``create`` is false, the underlying ``FileNotFoundError`` propagates
    uncaught so callers can distinguish "does not exist" from "unsafe".
    Components created this way are always
    validated private unless ``umask_created_dirs`` is true. Pre-existing intermediate
    components are only checked to be real directories and are **not** required
    to be owner-only — only the final component is validated against the private
    owner/mode policy, and only when ``private_final`` is true.

    The walk runs with shutdown signals deferred (see
    :func:`phasesweep.runtime.process.defer_shutdown_signals`), so
    ``PhaseSweepShutdown`` cannot land between the per-component descriptor
    handoffs and strand an intermediate descriptor. A shutdown that arrives
    mid-walk is serviced when the walk finishes: the just-opened final
    descriptor is closed here before the shutdown propagates, so the caller
    never owns a descriptor it does not receive.

    :param Path path: Directory to open, resolved lexically (not through
        the filesystem) before walking.
    :param bool create: Whether to ``mkdir`` any missing path component instead
        of failing on the first missing one.
    :param bool private_final: Whether the last path component must pass
        :func:`_validate_private_dir_info` (owner-only, mode ``0700``) even
        when it already existed.
    :param bool umask_created_dirs: Create missing components with mode ``0777``
        governed by the process umask instead of forcing private mode ``0700``.
    :return int: Open, ``O_NOFOLLOW``-validated file descriptor for the
        final directory; ownership transfers to the caller, who must close
        it.
    :raises FileNotFoundError: If a component is missing and ``create`` is
        false.
    :raises UnsafePrivatePathError: If the path is the filesystem root, a
        component is not a real directory, a component changed between its
        pre-open stat and the open, or the final component fails the
        private policy when ``private_final`` is true.
    """
    from phasesweep.runtime.process import defer_shutdown_signals

    result_fd = -1
    try:
        with defer_shutdown_signals():
            result_fd = _walk_directory_fd(
                path,
                create=create,
                private_final=private_final,
                umask_created_dirs=umask_created_dirs,
            )
    except BaseException:
        if result_fd >= 0:
            os.close(result_fd)
        raise
    return result_fd


def _walk_directory_fd(
    path: Path,
    *,
    create: bool,
    private_final: bool,
    umask_created_dirs: bool,
) -> int:
    """Perform the component walk for :func:`open_directory_fd`.

    :param Path path: Directory to open.
    :param bool create: Whether to create missing components.
    :param bool private_final: Whether the final component must be private.
    :param bool umask_created_dirs: Whether the umask governs created dirs.
    :return int: Open descriptor for the final directory; caller closes it.
    :raises UnsafePrivatePathError: ``path`` is the filesystem root, a
        component is not a real directory (symlink or other entry), a component
        was swapped while it was being opened, or the final directory fails the
        owner-only mode/ownership check.
    :raises FileNotFoundError: A component is missing and ``create`` is false.
    :raises PlatformCapabilityError: The platform does not provide
        ``O_NOFOLLOW``.
    """
    absolute = absolute_path(path)
    parts = absolute.parts[1:]
    if not parts:
        raise UnsafePrivatePathError("The filesystem root cannot be a private directory.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow_flag()
    current_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for index, component in enumerate(parts):
            created = False
            component_path = Path(*absolute.parts[: index + 2])
            try:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    mode = 0o777 if umask_created_dirs else PRIVATE_DIR_MODE
                    os.mkdir(component, mode, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                try:
                    before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                except OSError as exc:
                    raise UnsafePrivatePathError(
                        f"Private path component {component_path} is not a real directory."
                    ) from exc
            except OSError as exc:
                raise UnsafePrivatePathError(
                    f"Private path component {component_path} is not a real directory."
                ) from exc
            if not stat.S_ISDIR(before.st_mode):
                raise UnsafePrivatePathError(
                    f"Private path component {component_path} is not a real directory."
                )
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise UnsafePrivatePathError(
                    f"Private path component {component_path} is not a real directory."
                ) from exc
            after = os.fstat(next_fd)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(next_fd)
                raise UnsafePrivatePathError(
                    f"Private path component {component_path} changed while it was opened."
                )
            # Shutdown signals are deferred for the whole walk (see
            # ``open_directory_fd``), so ``PhaseSweepShutdown`` cannot land
            # between these statements. Historically it could: closing before
            # the store left ``current_fd`` naming a closed descriptor and
            # turned a shutdown into ``OSError: [Errno 9] Bad file
            # descriptor`` in the ``finally`` below, while storing before the
            # close leaked ``previous_fd`` on the reverse boundary. The
            # assign-first order stays so an exception raised by ``os.close``
            # itself still cannot double-close.
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
            if created and not umask_created_dirs:
                os.fchmod(current_fd, PRIVATE_DIR_MODE)
                _validate_private_dir_info(absolute, os.fstat(current_fd))
            if private_final and index == len(parts) - 1:
                _validate_private_dir_info(absolute, os.fstat(current_fd))
        result = current_fd
        current_fd = -1
        return result
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def ensure_private_dir(path: Path) -> None:
    """Create or validate an owner-only directory without following links.

    :param Path path: Directory that must be accessible only by the owner.
    :raises UnsafePrivatePathError: If a component is a link or the final mode/owner is unsafe.
    """
    fd = open_directory_fd(path, create=True, private_final=True)
    os.close(fd)


def validate_private_dir(path: Path) -> None:
    """Validate an existing owner-only directory without changing it."""
    fd = open_directory_fd(path, create=False, private_final=True)
    os.close(fd)


def _validate_private_file_info(path: Path, info: os.stat_result) -> None:
    """Validate that an opened file is a private, unshared regular file.

    :param Path path: File the ``info`` was captured from, used only for
        error messages.
    :param os.stat_result info: ``stat`` result of the already-opened file
        descriptor.
    :raises UnsafePrivatePathError: If the entry is not a regular file, has
        more than one hard link (so another path could reach the same
        inode), or is not owned by the current effective uid with mode
        ``0600``.
    """
    mode = stat.S_IMODE(info.st_mode)
    euid = os.geteuid()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise UnsafePrivatePathError(f"Private file {path} must be one regular file.")
    if info.st_uid != euid or mode != PRIVATE_FILE_MODE:
        raise UnsafePrivatePathError(
            f"Private file {path} must be owned by uid {euid} with mode 0600; "
            f"found uid {info.st_uid} and mode {mode:04o}."
        )


def _validate_private_destination(parent_fd: int, leaf: str, path: Path) -> None:
    """Validate an atomic-replace destination if it exists, without following it.

    Called immediately before :func:`os.replace` so a destination that is a
    symlink, a hardlinked file, or has the wrong owner/mode is refused
    instead of being silently overwritten. A missing destination is not an
    error — :func:`os.replace` is expected to create it.

    :param int parent_fd: Open descriptor on the destination's parent
        directory; not closed or otherwise consumed by this function.
    :param str leaf: Final path component of the destination, resolved
        relative to ``parent_fd``.
    :param Path path: Full destination path, used only for error messages.
    :raises UnsafePrivatePathError: If the destination exists and is not a
        private, unshared regular file (see
        :func:`_validate_private_file_info`).
    """
    try:
        info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _validate_private_file_info(path, info)


def read_private_text_at(parent_fd: int, leaf: str, path: Path) -> str:
    """Read one private UTF-8 file relative to an already-open directory.

    :param int parent_fd: Validated directory descriptor that anchors the read.
    :param str leaf: Single filename to open relative to ``parent_fd``.
    :param Path path: Display path used in validation errors.
    :return str: Complete UTF-8 contents of the validated owner-only file.
    :raises UnsafePrivatePathError: The name is unsafe or the entry is not a
        private, unshared regular file.
    """
    if leaf != leaf_name(Path(leaf)):
        raise UnsafePrivatePathError(f"Private file {path} has an unsafe filename.")
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_CLOEXEC | nofollow_flag(), dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise UnsafePrivatePathError(f"Private file {path} must not be a symlink.") from exc
        raise
    try:
        _validate_private_file_info(path, os.fstat(fd))
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def open_private_text(path: Path, mode: str = "w") -> IO[str]:
    """Open a UTF-8 text file with owner-only permissions.

    :param Path path: File to open.
    :param str mode: ``"w"``, ``"a"``, or exclusive-create ``"x"``.
    :return IO[str]: Open text handle.
    :raises ValueError: If ``mode`` is unsupported.
    :raises UnsafePrivatePathError: If the file or its parent path is unsafe.
    """
    from phasesweep.runtime.process import defer_shutdown_signals

    if mode not in {"w", "a", "x"}:
        raise ValueError(f"unsupported private text mode: {mode!r}")
    with defer_shutdown_signals():
        return _open_private_text(path, mode)


def _open_private_text(path: Path, mode: str) -> IO[str]:
    """Open and validate a private text file while shutdown is deferred.

    Unlocked core of :func:`open_private_text`, called from within its
    ``defer_shutdown_signals`` context with an already-validated ``mode``.

    :param Path path: File to open.
    :param str mode: ``"w"``, ``"a"``, or exclusive-create ``"x"``.
    :return IO[str]: Open text handle.
    :raises UnsafePrivatePathError: If the file or its parent path is unsafe.
    """
    parent_fd = open_directory_fd(path.parent, create=True, private_final=True)
    leaf = leaf_name(path)
    flags = os.O_WRONLY | os.O_CLOEXEC | nofollow_flag()
    if mode == "a":
        flags |= os.O_APPEND
    created = False
    try:
        if mode == "x":
            fd = os.open(
                leaf,
                flags | os.O_CREAT | os.O_EXCL,
                PRIVATE_FILE_MODE,
                dir_fd=parent_fd,
            )
            created = True
        else:
            try:
                fd = os.open(
                    leaf,
                    flags | os.O_CREAT | os.O_EXCL,
                    PRIVATE_FILE_MODE,
                    dir_fd=parent_fd,
                )
                created = True
            except FileExistsError:
                try:
                    fd = os.open(leaf, flags, dir_fd=parent_fd)
                except OSError as exc:
                    if exc.errno == errno.ELOOP:
                        raise UnsafePrivatePathError(
                            f"Private file {path} must not be a symlink."
                        ) from exc
                    raise
        try:
            if created:
                os.fchmod(fd, PRIVATE_FILE_MODE)
            _validate_private_file_info(path, os.fstat(fd))
            if mode == "w":
                os.ftruncate(fd, 0)
            return os.fdopen(fd, "w" if mode == "x" else mode, encoding="utf-8")
        except Exception:
            os.close(fd)
            if created:
                with contextlib.suppress(OSError):
                    os.unlink(leaf, dir_fd=parent_fd)
            raise
    finally:
        os.close(parent_fd)


def _new_private_temp_fd(parent_fd: int, leaf: str) -> tuple[int, str]:
    """Create a uniquely-named, owner-only temporary file next to a destination leaf.

    Uses ``O_CREAT | O_EXCL`` with ``O_NOFOLLOW`` relative to ``parent_fd``
    so the temporary file can never collide with an existing path or be a
    followed symlink, then ``fchmod``s it to the private file mode
    (belt-and-suspenders against umask). Retries with a fresh random suffix
    on a name collision.

    :param int parent_fd: Open descriptor on the destination directory the
        temporary file is created inside; not closed or otherwise consumed
        by this function.
    :param str leaf: Final path component of the eventual destination, used
        only to build a recognizable temporary filename.
    :return tuple[int, str]: The open file descriptor (ownership transfers
        to the caller, who must close it) and the temporary file's name,
        relative to ``parent_fd``.
    :raises FileExistsError: If 10 consecutive random names all collide.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | nofollow_flag()
    for _ in range(10):
        temporary = f".{leaf}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(temporary, flags, PRIVATE_FILE_MODE, dir_fd=parent_fd)
        except FileExistsError:
            continue
        os.fchmod(fd, PRIVATE_FILE_MODE)
        return fd, temporary
    raise FileExistsError(f"Unable to create a temporary file for {leaf!r}.")


@contextlib.contextmanager
def _private_atomic_writer(
    path: Path,
    *,
    newline: str | None = None,
    require_private_dir: bool = True,
) -> Iterator[IO[str]]:
    """Write to a private temporary file, then atomically replace a validated destination.

    Opens (creating if needed) the private destination directory, creates a
    uniquely-named owner-only temporary file inside it, and yields a handle
    to the caller to populate. On a clean exit from the ``with`` block, the
    handle is flushed and ``fsync``ed, the destination is validated if it
    already exists (refusing anything but a private, unshared regular file;
    see :func:`_validate_private_destination`), and the temporary file is
    renamed onto ``path`` with :func:`os.replace` — atomic because both
    names resolve relative to the same open parent directory descriptor.
    The parent directory is then best-effort ``fsync``ed. If the caller's
    block raises, or if any step before the rename fails, the temporary
    file is unlinked and ``path`` is left untouched.

    :param Path path: Destination path to replace.
    :param str | None newline: Newline handling passed to the text-mode
        ``open`` call.
    :param bool require_private_dir: Whether the destination directory must
        itself be owner-only. ``True`` for the hardened namespaces (locks, MCP
        ``state_dir``). ``False`` for an owner-only FILE inside an
        operator-trusted directory — the experiment artifact tree, whose
        directories are deliberately not owner-only (see the trust-boundary
        note in ``docs/runtime.md``). The file is created ``0600`` and an
        existing destination is still refused unless it is a private, unshared
        regular file, either way.
    :return Iterator[IO[str]]: Writable text handle on the temporary file,
        open for the caller to populate before the atomic replace.
    """
    parent_fd = open_directory_fd(
        path.parent,
        create=True,
        private_final=require_private_dir,
        umask_created_dirs=not require_private_dir,
    )
    leaf = leaf_name(path)
    fd = -1
    temporary: str | None = None
    replaced = False
    try:
        fd, temporary = _new_private_temp_fd(parent_fd, leaf)
        # Keep the raw descriptor as the sole owner until the outer ``finally``.
        # A shutdown between wrapping it and invalidating ``fd`` would otherwise
        # leave both the stream and cleanup path closing the same descriptor.
        stream = os.fdopen(fd, "w", encoding="utf-8", newline=newline, closefd=False)
        with stream as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        _validate_private_destination(parent_fd, leaf, path)
        os.replace(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        replaced = True
        try:
            os.fsync(parent_fd)
        except OSError:
            log.warning(
                "Directory fsync failed after a private atomic write in %s; the write is "
                "committed but its durability across a crash is uncertain.",
                path.parent,
                exc_info=True,
            )
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None and not replaced:
            with contextlib.suppress(OSError):
                os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def private_atomic_write_text(path: Path, text: str, *, require_private_dir: bool = True) -> None:
    """Atomically replace a private UTF-8 text file.

    :param Path path: Destination path to replace.
    :param str text: Text to write.
    :param bool require_private_dir: Whether the destination directory must be
        owner-only too; pass ``False`` to write an owner-only file into an
        operator-trusted directory such as a trial directory.
    """
    with _private_atomic_writer(path, require_private_dir=require_private_dir) as handle:
        handle.write(text)


def _new_shared_temp_fd(directory: Path, leaf: str) -> tuple[int, Path]:
    """Create a uniquely-named, umask-governed temporary file next to a destination leaf.

    Opens with ``O_CREAT | O_EXCL`` and a requested mode of ``0o666`` so the
    kernel applies the process umask at creation time. That is the only
    race-free way to honor the umask here: ``os.umask`` has no getter, so a
    read-modify-restore around the create would be visible to every other
    thread in the ``n_jobs > 1`` trial pool.

    :param Path directory: Existing directory the temporary file is created in.
    :param str leaf: Final path component of the eventual destination, used
        only to build a recognizable temporary filename.
    :return tuple[int, Path]: The open file descriptor (ownership transfers to
        the caller, who must close it) and the temporary file's path.
    :raises FileExistsError: If 10 consecutive random names all collide.
    :raises OSError: If the temporary file cannot be created.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    for _ in range(10):
        temporary = directory / f".{leaf}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(temporary, flags, 0o666), temporary
        except FileExistsError:
            continue
    raise FileExistsError(f"Unable to create a temporary file for {leaf!r}.")


@contextlib.contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[IO[str]]:
    """Write text through a same-directory temp file and atomically replace ``path``.

    File modes follow the ``workdir`` trust boundary documented in
    ``docs/runtime.md``: the experiment artifact tree is deliberately *not*
    owner-only, because operators and tooling need ordinary access to winners,
    summaries, and evidence. A fresh destination is therefore created ``0o666``
    masked by the process umask (``0o644`` under the usual ``0o022``), and a
    rewrite keeps whatever mode the destination already carried.

    ``tempfile.NamedTemporaryFile`` is deliberately not used for the staging
    file: it hardcodes ``0o600`` as a security primitive, and ``os.replace``
    would carry that mode onto every published artifact — leaving ``winner.yaml``
    and ``process_identity.json`` unreadable to a second operator or to the
    stale reaper running as another user, while sibling files written by plain
    ``open`` stayed ``0o644``. The private, genuinely owner-only counterpart is
    :func:`_private_atomic_writer`.

    :param Path path: Destination path that should be replaced atomically.
    :param str | None newline: Newline handling passed to the text-mode wrapper.
    :return Iterator[IO[str]]: Writable text handle yielded for the caller to populate.
    :raises FileExistsError: If no unique temporary name can be created.
    :raises OSError: If the temporary file cannot be created, written, or renamed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_info = path.lstat()
    except FileNotFoundError:
        target_mode = None
    else:
        target_mode = (
            stat.S_IMODE(target_info.st_mode) if stat.S_ISREG(target_info.st_mode) else None
        )
    fd = -1
    tmp_path: Path | None = None
    replaced = False
    try:
        fd, tmp_path = _new_shared_temp_fd(path.parent, path.name)
        if target_mode is not None:
            # Preserve the destination's mode before the rename; a chmod after
            # os.replace would leave a window where the artifact is published
            # with the wrong permissions.
            os.fchmod(fd, target_mode)
        # Keep the raw descriptor as the sole owner until the outer ``finally``
        # so a shutdown between wrapping it and invalidating ``fd`` cannot leave
        # both the stream and the cleanup path closing the same descriptor.
        stream = os.fdopen(fd, "w", encoding="utf-8", newline=newline, closefd=False)
        with stream as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        replaced = True
        fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_path is not None and not replaced:
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with UTF-8 text.

    :param Path path: Destination path to replace.
    :param str text: Text to write using UTF-8 encoding.
    """
    with atomic_text_writer(path) as handle:
        handle.write(text)


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


def sqlite_database_path(storage: str) -> Path | None:
    """Return the filesystem path of a file-backed SQLite storage URL.

    Lets callers distinguish "the database file does not exist" (so no study
    can exist in it) from "the file exists but cannot be read right now"
    (locked, corrupt, permission-denied), which must not be collapsed into
    absence on paths that authorize mutation.

    :param str storage: SQLite storage URL.
    :return Path | None: Concrete database file path, or ``None`` for
        in-memory storage.
    """
    if storage_is_in_memory(storage):
        return None
    database = file_url_path(storage)
    if _sqlite_uri_filename_enabled(storage, database):
        uri_path = sqlite_uri_filename_path(storage)
        return Path(uri_path).expanduser() if uri_path is not None else None
    return Path(database).expanduser()


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


def storage_recovery_locator(storage: str | None) -> str | None:
    """Freeze a storage URL for later recovery from a different working directory.

    Relative SQLite and journal paths are invocation-relative. Active-attempt
    recovery can happen after the caller changes directories, so the durable
    locator must resolve those paths while the attempt is registered. RDB URLs
    are returned unchanged because their credentials and connection options are
    operationally significant; callers persisting the result must use private
    storage.

    :param str | None storage: Configured Optuna storage URL.
    :return str | None: Operationally equivalent locator with file paths made
        absolute, or ``None`` for in-memory storage.
    """
    if storage_is_in_memory(storage):
        return None
    assert storage is not None
    backend = storage_backend(storage)
    if backend == "journal":
        path = Path(file_url_path(storage)).expanduser().resolve()
        return "journal:///" + str(path)
    if backend != "sqlite":
        return storage

    from sqlalchemy.engine.url import make_url

    url = make_url(storage)
    database = url.database or ""
    if _sqlite_uri_filename_enabled(storage, database):
        uri_path = sqlite_uri_filename_path(storage)
        if uri_path is None:
            # A non-local ``file:`` authority is not cwd-relative.
            return storage
        frozen_database = "file:" + str(Path(uri_path).expanduser().resolve())
    else:
        frozen_database = str(Path(database).expanduser().resolve())
    return url.set(database=frozen_database).render_as_string(hide_password=False)


# Default TCP ports per RDB dialect family, so `host/db` and `host:5432/db`
# share one lock identity. Families absent here keep no port segment when the
# URL omits the port.
_RDB_DEFAULT_PORTS = {
    "postgresql": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "mssql": 1433,
    "oracle": 1521,
}

# Query options that configure *how* we connect, not *what* we connect to.
# Excluding them fails safe: an excluded identity-bearing option can only
# over-collide (a spurious "already running" error), whereas keeping a
# connection-only option would split the lock for one shared database.
_RDB_CONNECTION_ONLY_OPTIONS = frozenset(
    {
        "application_name",
        "charset",
        "connect_timeout",
        "read_timeout",
        "sslcert",
        "sslkey",
        "sslmode",
        "sslrootcert",
        "target_session_attrs",
        "write_timeout",
    }
)
_RDB_CONNECTION_ONLY_OPTION_PREFIXES = ("keepalives",)


def _is_connection_only_rdb_option(key: str) -> bool:
    """Return whether an RDB URL query key configures the connection, not the target.

    :param str key: Query parameter name from an RDB storage URL.
    :return bool: True when the option must be excluded from lock identity.
    """
    lowered = key.lower()
    return lowered in _RDB_CONNECTION_ONLY_OPTIONS or lowered.startswith(
        _RDB_CONNECTION_ONLY_OPTION_PREFIXES
    )


def _rdb_identity_query_pairs(query: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return the identity-bearing query pairs of an RDB URL, in stable order.

    Connection-only options are dropped; everything else is kept, because a
    parameter such as libpq's ``host=/var/run/postgresql`` (unix socket) or a
    schema selector really does name a different target.

    :param Mapping[str, Any] query: SQLAlchemy ``URL.query`` mapping; a value may
        be a tuple when the URL repeats a key.
    :return list[tuple[str, str]]: Key/value pairs sorted deterministically.
    """
    pairs: list[tuple[str, str]] = []
    for key, value in query.items():
        if _is_connection_only_rdb_option(key):
            continue
        values = value if isinstance(value, tuple | list) else (value,)
        pairs.extend((key, str(item)) for item in values)
    return sorted(pairs, key=lambda pair: (pair[0].lower(), pair[0], pair[1]))


def _canonical_rdb_identity(storage: str) -> str:
    """Canonicalize a non-file RDB URL into a same-host lock identity.

    Equivalent spellings of one database must produce one identity, or the
    same-host experiment lock silently splits and two orchestrators write the
    same study (review v0.5.17 / blocker 5). Normalized away: the password
    (rotating a credential must not orphan a live lock), the driver
    (``postgresql``/``postgresql+psycopg``/``postgresql+psycopg2`` collide on
    the base dialect), host case, a port left implicit when it equals the
    family default, query-parameter order, and connection-only query options.

    Emitted form (each component percent-encoded so no separator is
    ambiguous)::

        rdb://<family>://<user>@<host>[:<port>]/<database>[?<sorted-query>]

    :param str storage: A non-file Optuna storage URL (``postgresql://...``, ...).
    :return str: The canonical identity, or ``storage`` unchanged when
        SQLAlchemy cannot parse it — lock-path derivation must never crash.
    """
    # Local import: SQLAlchemy ships with Optuna, but this module is on the
    # `phasesweep --help` path and importing it costs ~80ms we only owe for
    # the rare RDB storage URL.
    from sqlalchemy.engine.url import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        url = make_url(storage)
    except (ArgumentError, ValueError):
        return storage

    family = url.drivername.split("+", 1)[0].lower()
    port = url.port if url.port is not None else _RDB_DEFAULT_PORTS.get(family)
    port_segment = "" if port is None else f":{port}"
    user = quote(url.username or "", safe="")
    host = quote((url.host or "").lower(), safe="")
    database = quote(url.database or "", safe="")
    query = urlencode(_rdb_identity_query_pairs(url.query))
    query_segment = f"?{query}" if query else ""
    return f"rdb://{family}://{user}@{host}{port_segment}/{database}{query_segment}"


def canonical_storage_identity(storage: str | None) -> str | None:
    """Stable same-host identity string for a storage URL.

    File-based backends (SQLite, JournalStorage) are resolved to absolute
    paths so two configs that differ only in relative vs. absolute spelling
    still collide on the lock file. SQLite URLs additionally fold their
    SQLAlchemy dialect (``sqlite+pysqlite:///`` etc.) onto the canonical
    ``sqlite:///`` prefix so dialect choice never splits the lock.

    Non-file RDB URLs (``postgresql://``, ``mysql://``, ...) are canonicalized
    by :func:`_canonical_rdb_identity`: password, driver suffix, host case,
    implicit vs. explicit default port, query order, and connection-only query
    options are all normalized away. What is *not* resolved: DNS aliases and
    ``CNAME``s, load-balancer or pgbouncer endpoints, ``PGSERVICE`` service
    files, ``~/.pg_service.conf`` or environment-supplied defaults, and
    ``localhost`` vs. ``127.0.0.1`` vs. a unix socket. Those all reach the same
    database under different identities, so operators relying on
    ``allow_external_rdb_single_host: true`` must still spell the storage URL
    the same way in every config that shares one database.

    Args:
        storage: An Optuna storage URL, or ``None`` for in-memory storage.

    Returns:
        The canonical identity string used to derive the same-host storage
        lock path, or ``None`` for in-memory storage (no shared backend to
        collide on). Storage strings SQLAlchemy cannot parse are returned
        unchanged.

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

    # RDB URLs (postgresql, mysql, ...): canonicalize so equivalent spellings of
    # one database share a lock instead of splitting it (review v0.5.17 / blocker 5).
    return _canonical_rdb_identity(storage)
