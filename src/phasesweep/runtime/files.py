"""Filesystem, lock, timeout, and storage-URL runtime helpers."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import secrets
import stat
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from phasesweep.errors import OperatorAction, PhaseSweepError

log = logging.getLogger("phasesweep.runtime.files")

SUPPORTED_RUNTIME_ERROR = (
    "phasesweep runs on Linux and macOS, and this host reports {platform!r}. It requires "
    "kernel process identities, fcntl.flock host locks and POSIX process groups."
)
_LOCK_DIR_ENV = "PHASESWEEP_LOCK_DIR"
_LOCK_ACCESS_GUIDANCE = (
    "Use the same writable PHASESWEEP_LOCK_DIR for every shell, MCP server, and sandbox "
    "launching work on this host; provision that directory before setting it."
)
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SHARED_DIR_MODE = 0o3770
SHARED_FILE_MODE = 0o660


def cache_dir() -> Path:
    """Return the disposable PhaseSweep cache namespace.

    :return Path: Absolute XDG cache root plus phasesweep, defaulting below Path.home().
    """
    return xdg_home("XDG_CACHE_HOME", Path.home() / ".cache") / "phasesweep"


def artifact_root_path(workdir: str, experiment: str) -> Path:
    """Resolve the workdir before appending the experiment namespace.

    :param str workdir: Configured parent directory of experiment artifacts.
    :param str experiment: Validated experiment directory name.
    :return Path: Absolute artifact root with any experiment leaf symlink unresolved.
    """
    return Path(workdir).expanduser().resolve() / experiment


def ensure_artifact_dir(path: Path) -> None:
    """Keep artifacts out of Git while allowing ripgrep to search their contents.

    :param Path path: Managed experiment directory, not its shared workdir.
    """
    path.mkdir(parents=True, exist_ok=True)
    for name, rule in ((".gitignore", "*\n"), (".ignore", "!**\n")):
        try:
            with (path / name).open("x", encoding="utf-8") as handle:
                handle.write(rule)
        except FileExistsError:
            pass


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


class UnsafeLockPathError(PhaseSweepError):
    """Raised when a lock directory or file is inaccessible or unsafe to trust."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class UnsafePrivatePathError(PhaseSweepError):
    """Raised when a private directory or file is not safe to mutate."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PlatformCapabilityError(PhaseSweepError):
    """Raised when the host lacks a capability required for safe operation."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


@dataclass(frozen=True)
class _LockPolicy:
    """Ownership and mode expected for one lock namespace."""

    shared: bool
    uid: int
    gid: int
    file_mode: int


def require_supported_runtime() -> None:
    """Require Linux or macOS process and filesystem primitives.

    This is the filesystem platform check. Everything below it may use POSIX
    primitives (``fcntl``, ``os.killpg``, ``os.O_NOFOLLOW``, ``signal.pthread_sigmask``)
    without a per-call fallback; ``fcntl`` is imported after the gate so that
    importing this module still works elsewhere.

    :raises PlatformCapabilityError: The host is neither Linux nor macOS.
    """
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise PlatformCapabilityError(SUPPORTED_RUNTIME_ERROR.format(platform=sys.platform))


def xdg_home(variable: str, fallback: Path) -> Path:
    """Return an absolute XDG base, falling back for unset or relative values.

    :param str variable: XDG environment variable name.
    :param Path fallback: Default absolute base directory.
    :return Path: Selected base without creating it.
    """
    value = os.environ.get(variable)
    path = Path(value) if value else fallback
    return path if path.is_absolute() else fallback


def phasesweep_home() -> Path | None:
    """Return the validated private root override, creating a missing one owner-only.

    A missing directory, and any missing parent of it, is created with mode
    ``0700``. An existing directory must already be owner-only, and no
    component of the path may be a symlink.

    :return Path | None: Private root, or None when unset.
    :raises UnsafePrivatePathError: The override is relative, has a symlinked
        component, or names an existing directory that is not owner-only.
    """
    override = os.environ.get("PHASESWEEP_HOME")
    if not override:
        return None
    path = Path(override)
    if not path.is_absolute():
        raise UnsafePrivatePathError(
            f"PHASESWEEP_HOME must be an absolute path: {path}", action=OperatorAction.FIX_CONFIG
        )
    ensure_private_dir(path)
    return path


def lock_dir() -> Path:
    """Return the validated same-host phasesweep lock directory.

    The default is ``.cache/phasesweep/locks`` under the effective user's
    account home, independent of environment-based state/cache placement.
    ``PHASESWEEP_LOCK_DIR`` selects an existing operator-provisioned directory
    and is never created or chmodded by phasesweep.
    macOS's root-owned ``/var`` and ``/tmp`` aliases resolve before the
    no-symlink walk; operator-controlled path components still cannot be links.

    :return Path: Directory used for host-local lock files.
    :raises UnsafeLockPathError: ``PHASESWEEP_LOCK_DIR`` is relative, missing,
        contains a symlinked component, or fails the private/shared
        ownership-and-mode check; the OS account has no absolute home; or
        the default lock directory fails the owner-only checks; or the OS refuses access.
    :raises PlatformCapabilityError: The host is neither Linux nor macOS.
    """
    override = os.environ.get(_LOCK_DIR_ENV)
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise UnsafeLockPathError(
                f"{_LOCK_DIR_ENV} must be an absolute path: {path}",
                action=OperatorAction.FIX_CONFIG,
            )
        try:
            if sys.platform == "darwin" and len(path.parts) > 1 and path.parts[1] in {"var", "tmp"}:
                system_root = Path(path.anchor, path.parts[1])
                if system_root.is_symlink() and system_root.lstat().st_uid == 0:
                    path = system_root.resolve(strict=True).joinpath(*path.parts[2:])
            _lock_policy(path)
        except OSError as exc:
            raise UnsafeLockPathError(
                f"Cannot access lock directory {path}: {exc}. {_LOCK_ACCESS_GUIDANCE}",
                action=OperatorAction.PROVISION_LOCK_DIR,
            ) from exc
        return path

    require_supported_runtime()
    import pwd

    # Shells and services for one account must contend even when HOME or XDG
    # settings differ. Only the explicit lock override may move this namespace.
    try:
        home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except KeyError as exc:
        raise UnsafeLockPathError(
            f"No OS account home exists for uid {os.geteuid()}; provision an absolute "
            f"lock directory and set {_LOCK_DIR_ENV}.",
            action=OperatorAction.FIX_CONFIG,
        ) from exc
    if not home.is_absolute():
        raise UnsafeLockPathError(
            f"The OS account home is not absolute; provision an absolute lock directory "
            f"and set {_LOCK_DIR_ENV}.",
            action=OperatorAction.FIX_CONFIG,
        )
    path = home / ".cache" / "phasesweep" / "locks"
    try:
        ensure_private_dir(path)
    except UnsafePrivatePathError as exc:
        raise UnsafeLockPathError(f"Default lock directory is unsafe: {exc}") from exc
    except OSError as exc:
        raise UnsafeLockPathError(
            f"Cannot create or access default lock directory {path}: {exc}. {_LOCK_ACCESS_GUIDANCE}",
            action=OperatorAction.PROVISION_LOCK_DIR,
        ) from exc
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
            f"Lock directory {path} does not exist; provision it before setting {_LOCK_DIR_ENV}.",
            action=OperatorAction.PROVISION_LOCK_DIR,
        ) from exc
    except UnsafePrivatePathError as exc:
        raise UnsafeLockPathError(f"Lock directory {path} has an unsafe path component.") from exc
    try:
        return _lock_policy_for_info(path, os.fstat(fd))
    finally:
        os.close(fd)


def _lock_policy_for_info(path: Path, info: os.stat_result) -> _LockPolicy:
    """Determine the private or shared lock-directory policy.

    Private directories require mode ``0700`` and the current UID; shared directories
    require root ownership, mode ``03770`` and membership in the owning group.

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
    """Open a lock file relative to its validated directory descriptor.

    Refuse symlinks and validate ownership, mode and single-link status against the
    directory policy. Close the descriptor if validation fails.

    :param Path path: Lock file path to open or create.
    :return IO[str]: Text-mode (``"r+"``, UTF-8) handle open on the
        validated lock file.
    :raises PlatformCapabilityError: The host is neither Linux nor macOS.
    :raises UnsafeLockPathError: If the parent directory is missing or unsafe,
        the leaf name is unsafe, or the file fails the regular-file/ownership/
        mode checks, or the OS refuses access.
    """
    from phasesweep.runtime.shutdown import defer_shutdown_signals

    with defer_shutdown_signals():
        try:
            return _open_lock_file(path)
        except OSError as exc:
            raise UnsafeLockPathError(
                f"Cannot open lock file {path}: {exc}. {_LOCK_ACCESS_GUIDANCE}",
                action=OperatorAction.PROVISION_LOCK_DIR,
            ) from exc


def _open_lock_file(path: Path) -> IO[str]:
    """Open and validate a lock file while shutdown is deferred.

    Unlocked core of :func:`open_lock_file`, called from within its
    ``defer_shutdown_signals`` context.

    :param Path path: Lock file path to open or create.
    :return IO[str]: Text-mode (``"r+"``, UTF-8) handle open on the
        validated lock file.
    :raises PlatformCapabilityError: The host is neither Linux nor macOS.
    :raises UnsafeLockPathError: If the parent directory is missing or unsafe,
        the leaf name is unsafe, or the file fails the regular-file/ownership/
        mode checks.
    """
    require_supported_runtime()
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
        flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
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
    handle = open_lock_file(path)
    if try_flock(handle):
        return handle
    handle.close()
    return None


def try_flock(handle: IO[str]) -> bool:
    """Take an exclusive nonblocking flock on an already-open file.

    :param IO[str] handle: File descriptor whose inode all contenders open.
    :return bool: Whether the lock was acquired; the caller owns the handle.
    """
    require_supported_runtime()
    import fcntl

    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def unlock_file(handle: IO[str]) -> None:
    """Release a flock and close its file handle."""
    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(handle, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        handle.close()


def fsync_directory(path: Path) -> None:
    """Best-effort fsync a directory after publication.

    :param Path path: Directory whose entries should be durable.
    :raises OSError: Descriptor closing fails.
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


def _validate_private_owner_mode(
    path: Path,
    info: os.stat_result,
    *,
    kind: str,
    expected_mode: int,
) -> None:
    """Validate the owner and exact permission mode of an opened private entry.

    :param Path path: Display path used in an unsafe-entry diagnostic.
    :param os.stat_result info: Metadata read from the already-open entry.
    :param str kind: Human-readable entry kind used in the diagnostic.
    :param int expected_mode: Required exact permission bits.
    :raises UnsafePrivatePathError: If ownership or permissions are unsafe.
    """
    mode = stat.S_IMODE(info.st_mode)
    euid = os.geteuid()
    if info.st_uid != euid or mode != expected_mode:
        raise UnsafePrivatePathError(
            f"Private {kind} {path} must be owned by uid {euid} with mode "
            f"{expected_mode:04o}; found uid {info.st_uid} and mode {mode:04o}."
        )


def _validate_private_dir_info(path: Path, info: os.stat_result) -> None:
    """Validate that an opened directory is owner-only, without modifying it.

    :param Path path: Directory the ``info`` was captured from, used only
        for error messages.
    :param os.stat_result info: ``stat`` result of the already-opened
        directory descriptor.
    :raises UnsafePrivatePathError: If the entry is not a directory, or is
        not owned by the current effective uid with mode ``0700``.
    """
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafePrivatePathError(f"Private directory {path} is not a directory.")
    _validate_private_owner_mode(
        path,
        info,
        kind="directory",
        expected_mode=PRIVATE_DIR_MODE,
    )


def open_directory_fd(
    path: Path,
    *,
    create: bool,
    private_final: bool,
    umask_created_dirs: bool = False,
) -> int:
    """Open a directory through a descriptor-relative walk that refuses symlinks.

    Each component must remain the same real directory between stat and open.
    Created components are private unless ``umask_created_dirs`` is set; existing
    intermediate components need not be private. Only ``private_final`` validates
    the final directory's private policy. Signals are deferred through descriptor
    handoffs; the final descriptor is closed if shutdown interrupts its return.

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
    from phasesweep.runtime.shutdown import defer_shutdown_signals

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
    """
    absolute = absolute_path(path)
    parts = absolute.parts[1:]
    if not parts:
        raise UnsafePrivatePathError("The filesystem root cannot be a private directory.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
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
            # Signals remain deferred through this ownership transfer. Assign first so a
            # close failure cannot make finally close the old descriptor again.
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
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise UnsafePrivatePathError(f"Private file {path} must be one regular file.")
    _validate_private_owner_mode(
        path,
        info,
        kind="file",
        expected_mode=PRIVATE_FILE_MODE,
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
        fd = os.open(leaf, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
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


def read_private_text(path: Path, *, require_private_dir: bool = True) -> str:
    """Read a private UTF-8 file through its validated parent directory.

    :param Path path: Private file to read.
    :param bool require_private_dir: Require owner-only parent permissions too.
    :return str: Complete file contents.
    """
    directory_fd = open_directory_fd(path.parent, create=False, private_final=require_private_dir)
    try:
        return read_private_text_at(directory_fd, path.name, path)
    finally:
        os.close(directory_fd)


def open_private_text(path: Path, mode: str = "w") -> IO[str]:
    """Open a UTF-8 text file with owner-only permissions.

    :param Path path: File to open.
    :param str mode: ``"w"``, ``"a"``, or exclusive-create ``"x"``.
    :return IO[str]: Open text handle.
    :raises ValueError: If ``mode`` is unsupported.
    :raises UnsafePrivatePathError: If the file or its parent path is unsafe.
    """
    from phasesweep.runtime.shutdown import defer_shutdown_signals

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
    flags = os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW
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


def _new_exclusive_temp_fd(
    path: Path, mode: int, *, directory_fd: int | None = None
) -> tuple[int, Path]:
    """Create a random exclusive staging file beside a destination.

    :param Path path: Destination used to name the staging file.
    :param int mode: Requested creation mode, subject to the process umask.
    :param int | None directory_fd: Validated parent descriptor for private paths.
    :return tuple[int, Path]: Open descriptor and staging path.
    :raises FileExistsError: If 10 consecutive random names all collide.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    leaf = path.name if directory_fd is None else leaf_name(path)
    if directory_fd is not None:
        flags |= os.O_NOFOLLOW
    for _ in range(10):
        temporary = path.with_name(f".{leaf}.{secrets.token_hex(8)}.tmp")
        try:
            target = temporary if directory_fd is None else temporary.name
            return os.open(target, flags, mode, dir_fd=directory_fd), temporary
        except FileExistsError:
            continue
    raise FileExistsError(f"Unable to create a temporary file for {leaf!r}.")


@contextlib.contextmanager
def _staged_text(
    path: Path,
    *,
    directory_fd: int | None = None,
    mode: int = 0o666,
    file_mode: int | None = None,
    newline: str | None = None,
    collision_message: str | None = None,
) -> Iterator[tuple[IO[str], Path]]:
    """Write and fsync staging bytes, retaining the file only on success.

    :param Path path: Destination used to name the staging file.
    :param int | None directory_fd: Validated parent descriptor for private paths.
    :param int mode: Requested creation mode, subject to the process umask.
    :param int | None file_mode: Exact staging mode applied before writing.
    :param str | None newline: Text stream newline handling.
    :param str | None collision_message: Caller diagnostic for exhausted staging names.
    :return Iterator[tuple[IO[str], Path]]: Writable stream and its staging path.
    """
    fd = -1
    temporary: Path | None = None
    try:
        try:
            try:
                fd, temporary = _new_exclusive_temp_fd(path, mode, directory_fd=directory_fd)
            except FileExistsError as exc:
                if collision_message is not None:
                    raise FileExistsError(collision_message) from exc
                raise
            if file_mode is not None:
                os.fchmod(fd, file_mode)
            # The raw descriptor is the sole owner even if shutdown interrupts fdopen.
            with os.fdopen(fd, "w", encoding="utf-8", newline=newline, closefd=False) as handle:
                yield handle, temporary
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    except BaseException:
        if temporary is not None:
            with contextlib.suppress(OSError):
                target = temporary if directory_fd is None else temporary.name
                os.unlink(target, dir_fd=directory_fd)
        raise


@contextlib.contextmanager
def _private_atomic_writer(
    path: Path,
    *,
    require_private_dir: bool = True,
    strict_dir_fsync: bool = False,
) -> Iterator[IO[str]]:
    """Stage private UTF-8 text and atomically replace a validated destination.

    The file remains ``0600`` regardless of directory policy. Replacement is relative
    to the validated parent descriptor; prepublication failures remove staging without
    changing the destination. Directory fsync follows the selected failure policy.

    :param Path path: Destination path to replace.
    :param bool require_private_dir: Whether the destination directory must
        itself be owner-only, as in lock and MCP state namespaces. ``False``
        permits the operator-trusted experiment artifact tree. In either case,
        an existing destination must be a private, unshared regular file.
    :param bool strict_dir_fsync: Raise on directory-fsync failure after replacement.
    :raises OSError: Staging or replacement failed, or strict directory fsync failed.
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
    temporary: Path | None = None
    replaced = False
    try:
        with _staged_text(
            path, directory_fd=parent_fd, mode=PRIVATE_FILE_MODE, file_mode=PRIVATE_FILE_MODE
        ) as (handle, temporary):
            yield handle
        _validate_private_destination(parent_fd, leaf, path)
        os.replace(temporary.name, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        replaced = True
        try:
            os.fsync(parent_fd)
        except OSError:
            if strict_dir_fsync:
                raise
            log.warning(
                "Directory fsync failed after a private atomic write in %s; the write is "
                "committed but its durability across a crash is uncertain.",
                path.parent,
                exc_info=True,
            )
    finally:
        if temporary is not None and not replaced:
            with contextlib.suppress(OSError):
                os.unlink(temporary.name, dir_fd=parent_fd)
        os.close(parent_fd)


def private_atomic_write_text(
    path: Path, text: str, *, require_private_dir: bool = True, strict_dir_fsync: bool = False
) -> None:
    """Atomically replace a private UTF-8 text file.

    :param Path path: Destination path to replace.
    :param str text: Text to write.
    :param bool require_private_dir: Whether the destination directory must be
        owner-only too; pass ``False`` to write an owner-only file into an
        operator-trusted directory such as a trial directory.
    :param bool strict_dir_fsync: Raise on directory-fsync failure after replacement.
    :raises OSError: Writing failed; a strict directory-fsync failure leaves the replacement published.
    """
    with _private_atomic_writer(
        path, require_private_dir=require_private_dir, strict_dir_fsync=strict_dir_fsync
    ) as handle:
        handle.write(text)


@contextlib.contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[IO[str]]:
    """Write text through a same-directory stage and atomically replace ``path``.

    Create missing parent directories. New files use ``0666`` masked by umask;
    replacements preserve the destination's mode.

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
    tmp_path: Path | None = None
    replaced = False
    try:
        with _staged_text(path, file_mode=target_mode, newline=newline) as (handle, tmp_path):
            yield handle
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
    :raises OSError: Parent creation, staging, replacement, or durability fails.
    """
    with atomic_text_writer(path) as handle:
        handle.write(text)


def atomic_create_text(
    path: Path,
    text: str,
    *,
    validate: Callable[[Path], object] | None = None,
) -> bool:
    """Stage, optionally validate, and exclusively publish UTF-8 text.

    A hard link publishes the complete fsynced stage, which is then removed.
    Files use ``0666`` masked by umask; directory fsync is best-effort.

    :param Path path: Destination that must not already exist.
    :param str text: Complete UTF-8 text to publish.
    :param Callable[[Path], object] | None validate: Optional callable run on
        the staged file before publication; may raise to refuse the write.
    :return bool: True when published, or False if another writer already
        holds ``path``.
    :raises FileExistsError: If ten randomized staging names beside ``path``
        all collide.
    :raises OSError: If the staged file cannot be written, or ``validate``
        or the hard link otherwise fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with _staged_text(
            path, collision_message=f"cannot create a staging file beside {path}"
        ) as (handle, staged):
            handle.write(text)
        if validate is not None:
            validate(staged)
        try:
            os.link(staged, path)
        except FileExistsError:
            return False
        fsync_directory(path.parent)
    finally:
        if staged is not None:
            with contextlib.suppress(OSError):
                staged.unlink()
    return True


def private_atomic_create_text(path: Path, text: str) -> None:
    """Exclusively publish a private file, rolling back any durability failure.

    :param Path path: Destination inside a validated private directory.
    :param str text: Complete UTF-8 contents to publish.
    :raises FileExistsError: The destination already exists.
    :raises OSError: Staging, publication, or directory fsync fails.
    """
    parent_fd = open_directory_fd(path.parent, create=True, private_final=True)
    temporary: Path | None = None
    published = False
    try:
        with _staged_text(
            path, directory_fd=parent_fd, mode=PRIVATE_FILE_MODE, file_mode=PRIVATE_FILE_MODE
        ) as (handle, temporary):
            handle.write(text)
        os.link(
            temporary.name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if published:
            with contextlib.suppress(OSError):
                os.unlink(path.name, dir_fd=parent_fd)
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary.name, dir_fd=parent_fd)
        with contextlib.suppress(OSError):
            os.fsync(parent_fd)
        raise
    finally:
        os.close(parent_fd)


def local_storage_url(path: Path) -> str:
    """Encode an absolute journal filename as a local URI URL.

    :param Path path: Journal filename.
    :return str: Absolute ``journal:///`` URL preserving reserved characters
        in the filename.
    """
    return f"journal:///file:{quote(str(path.resolve()), safe='/')}?uri=true"


def file_url_path(storage: str) -> str:
    """Return the filesystem path component of a phasesweep file-style URL.

    This file-style URL grammar uses three slashes for relative paths and
    four for absolute POSIX paths (the fourth slash is the root ``/``). We
    must preserve that distinction; ``lstrip("/")`` would destroy absolute
    paths.

    Supported forms::

        journal:///relative.journal         -> relative.journal
        journal:///relative.journal?x=1     -> relative.journal
        journal:////tmp/absolute.journal    -> /tmp/absolute.journal
        journal:///file:/tmp/a%3Fb?uri=true -> /tmp/a?b

    :param str storage: A file-style storage URL whose scheme is already known to be file-based
        (``journal``).
    :return str: The bare filesystem path, without the scheme or leading slashes that belong
        to URL grammar.
    """
    scheme, rest = storage.split(":", 1)

    if rest.startswith("////"):
        # POSIX absolute file path. The fourth slash IS the root ``/``.
        path = "/" + rest[4:]
    elif rest.startswith("///"):
        # Relative file path.
        path = rest[3:]
    elif rest.startswith("//"):
        path = rest[2:]
    else:
        path = rest

    path = path.split("?", 1)[0]
    path = path.split("#", 1)[0]
    query = storage.split("?", 1)[1].split("#", 1)[0] if "?" in storage else ""
    options = {
        key.lower(): value.lower() for key, value in parse_qsl(query, keep_blank_values=True)
    }
    # Auto journal storage uses an escaped file: URI convention. Ordinary
    # explicit journal paths keep their literal spelling.
    if (
        scheme.lower() == "journal"
        and path.startswith("file:")
        and options.get("uri") in {"1", "true", "yes", "on"}
    ):
        return unquote(urlsplit(path).path)
    return path


def journal_file_path(storage: str) -> Path:
    """Resolve a file-backed journal URL to its canonical path.

    Anchor a relative path at the current working directory and resolve symlinks so
    storage and repair locks address the same journal.

    :param str storage: Journal storage URL, including escaped ``file:`` forms.
    :raises ValueError: ``storage`` does not name a journal file.
    :return Path: The journal file, with ``~`` expanded and symlinks resolved.
    """
    backend = storage.split(":", 1)[0].lower()
    if backend != "journal":
        raise ValueError(f"Unsupported local storage backend: {backend!r}.")
    return Path(file_url_path(storage)).expanduser().resolve()


def storage_recovery_locator(storage: str | None) -> str | None:
    """Freeze a storage URL for later recovery from a different working directory.

    Relative journal paths are invocation-relative. Active-attempt recovery
    can happen after the caller changes directories, so the durable locator
    must resolve those paths while the attempt is registered.

    :param str | None storage: Configured Optuna storage URL.
    :return str | None: Operationally equivalent locator with file paths made
        absolute, or ``None`` for in-memory storage.
    :raises ValueError: ``storage`` does not name a journal file.
    """
    if storage is None:
        return None
    return local_storage_url(journal_file_path(storage))


def canonical_storage_identity(storage: str | None) -> str | None:
    """Stable same-host identity string for a storage URL.

    JournalStorage is resolved to an absolute path so two configs that differ
    only in relative vs. absolute spelling still collide on the lock file.

    :param str | None storage: An Optuna storage URL, or ``None`` for in-memory
        storage.
    :return str | None: The canonical identity string used to derive the
        same-host storage lock path, or ``None`` for in-memory storage (no
        shared backend to collide on).
    :raises ValueError: ``storage`` does not name a journal file.
    """
    if storage is None:
        return None
    return "journal:///" + str(journal_file_path(storage))
