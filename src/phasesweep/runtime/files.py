"""Filesystem, lock, timeout, and storage-URL runtime helpers."""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import stat
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

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


class UnsafeLockPathError(RuntimeError):
    """Raised when a lock directory or file is not safe to trust."""


class UnsafePrivatePathError(RuntimeError):
    """Raised when a private directory or file is not safe to mutate."""


@dataclass(frozen=True)
class _LockPolicy:
    """Ownership and mode expected for one lock namespace."""

    shared: bool
    uid: int
    gid: int
    file_mode: int


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


def lock_dir() -> Path:
    """Return the validated same-host phasesweep lock directory.

    The default is private to the current user. ``PHASESWEEP_LOCK_DIR`` selects
    an existing operator-provisioned directory and is never created or chmodded
    by phasesweep.

    :return Path: Directory used for host-local lock files.
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
    _lock_policy(path)
    return path


def _lock_policy(path: Path) -> _LockPolicy:
    """Open, validate, and classify a lock directory as private or shared.

    Walks ``path`` with :func:`_open_directory_fd` (no ``O_NOFOLLOW`` bypass,
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
        fd = _open_directory_fd(path, create=False, private_final=False)
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
    :raises UnsafeLockPathError: If the platform lacks ``O_NOFOLLOW``, the
        parent directory is missing or unsafe, the leaf name is unsafe, or
        the file fails the regular-file/ownership/mode checks.
    """
    require_posix_runtime()
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise UnsafeLockPathError("This platform cannot safely open lock files without symlinks.")
    try:
        parent_fd = _open_directory_fd(path.parent, create=False, private_final=False)
    except (FileNotFoundError, UnsafePrivatePathError) as exc:
        raise UnsafeLockPathError(f"Lock directory {path.parent} is not safe to open.") from exc
    try:
        policy = _lock_policy_for_info(path.parent, os.fstat(parent_fd))
        try:
            leaf = _leaf_name(path)
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
    """Best-effort fsync for a directory after an atomic replace."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _nofollow_flag() -> int:
    """Return ``O_NOFOLLOW`` or fail when safe private traversal is unavailable."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise UnsafePrivatePathError(
            "This platform cannot safely access private files without following symlinks."
        )
    return nofollow


def _absolute_path(path: Path) -> Path:
    """Return a lexical absolute path without resolving symlinks.

    Anchors a relative path at the current working directory, then collapses
    ``.`` and ``..`` components purely by name, matching
    :func:`posixpath.normpath`. This must not touch the filesystem: callers
    such as :func:`_open_directory_fd` walk the resulting components with
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


def _leaf_name(path: Path) -> str:
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


def _open_directory_fd(
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
    absolute = _absolute_path(path)
    parts = absolute.parts[1:]
    if not parts:
        raise UnsafePrivatePathError("The filesystem root cannot be a private directory.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | _nofollow_flag()
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
            os.close(current_fd)
            current_fd = next_fd
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
    fd = _open_directory_fd(path, create=True, private_final=True)
    os.close(fd)


def validate_private_dir(path: Path) -> None:
    """Validate an existing owner-only directory without changing it."""
    fd = _open_directory_fd(path, create=False, private_final=True)
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


def open_private_text(path: Path, mode: str = "w") -> IO[str]:
    """Open a UTF-8 text file with owner-only permissions.

    :param Path path: File to open.
    :param str mode: ``"w"``, ``"a"``, or exclusive-create ``"x"``.
    :return IO[str]: Open text handle.
    :raises ValueError: If ``mode`` is unsupported.
    :raises UnsafePrivatePathError: If the file or its parent path is unsafe.
    """
    if mode not in {"w", "a", "x"}:
        raise ValueError(f"unsupported private text mode: {mode!r}")
    parent_fd = _open_directory_fd(path.parent, create=True, private_final=True)
    leaf = _leaf_name(path)
    flags = os.O_WRONLY | os.O_CLOEXEC | _nofollow_flag()
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | _nofollow_flag()
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
    binary: bool,
    newline: str | None = None,
) -> Iterator[IO[Any]]:
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
    :param bool binary: Whether to open the temporary file in binary
        (``"wb"``) mode instead of UTF-8 text (``"w"``).
    :param str | None newline: Newline handling passed to the text-mode
        ``open`` call; ignored when ``binary`` is true.
    :return Iterator[IO[Any]]: Writable handle (binary or text per
        ``binary``) on the temporary file, open for the caller to populate
        before the atomic replace.
    """
    parent_fd = _open_directory_fd(path.parent, create=True, private_final=True)
    leaf = _leaf_name(path)
    fd = -1
    temporary: str | None = None
    replaced = False
    try:
        fd, temporary = _new_private_temp_fd(parent_fd, leaf)
        stream: IO[Any]
        if binary:
            stream = os.fdopen(fd, "wb")
        else:
            stream = os.fdopen(fd, "w", encoding="utf-8", newline=newline)
        fd = -1
        with stream as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        _validate_private_destination(parent_fd, leaf, path)
        os.replace(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        replaced = True
        with contextlib.suppress(OSError):
            os.fsync(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None and not replaced:
            with contextlib.suppress(OSError):
                os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


@contextlib.contextmanager
def private_atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[IO[str]]:
    """Atomically replace a private UTF-8 text file.

    :param Path path: Destination path to replace.
    :param str | None newline: Newline handling passed to ``open``.
    :return Iterator[IO[str]]: Writable text handle.
    """
    with _private_atomic_writer(path, binary=False, newline=newline) as handle:
        yield cast(IO[str], handle)


def private_atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a private UTF-8 text file.

    :param Path path: Destination path to replace.
    :param str text: Text to write.
    """
    with private_atomic_text_writer(path) as handle:
        handle.write(text)


@contextlib.contextmanager
def private_atomic_bytes_writer(path: Path) -> Iterator[IO[bytes]]:
    """Atomically replace a private bytes file.

    :param Path path: Destination path to replace.
    :return Iterator[IO[bytes]]: Writable binary handle.
    """
    with _private_atomic_writer(path, binary=True) as handle:
        yield cast(IO[bytes], handle)


def private_atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a private bytes file.

    :param Path path: Destination path to replace.
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
