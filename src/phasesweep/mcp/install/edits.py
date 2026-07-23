"""File-edit primitives for installer integrations.

Two edit families share one safety contract: modify only installer-managed
content, refuse same-name conflicts, and serialize each target's complete
read-modify-write transaction:

- JSON member edits: parse the whole document, change one member under one
  container key, and re-serialize with the file's detected indentation and
  newline style. Key order is retained, but whitespace and finite-number spellings may change.
  Duplicate keys, non-finite/overflowing numbers, comments, and JSON5 are
  never modified; the caller gets ``"skipped"`` and prints a manual snippet.
- Marker-fenced text blocks: replace-or-append a block between start/end
  marker lines, leaving every byte outside the markers alone. Removal restores
  the original bytes (including newline style) and leaves the file in place;
  file creation provenance is not persisted, so emptiness is not treated as
  proof that the installer owns the whole file.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, TypeAlias

from phasesweep.runtime.files import (
    UnsafeLockPathError,
    UnsafePrivatePathError,
    _absolute_path,
    _leaf_name,
    _nofollow_flag,
    _open_directory_fd,
    lock_dir,
    open_lock_file,
)
from phasesweep.runtime.json import strict_json_loads

Action: TypeAlias = Literal[
    "created",
    "updated",
    "unchanged",
    "skipped",
    "conflict",
    "error",
    "lock-error",
    "removed",
    "not-found",
]
MemberPredicate: TypeAlias = Callable[[object], bool]

_INDENT_PATTERN = re.compile(r"^([ \t]+)\S", re.MULTILINE)


@dataclass(frozen=True)
class _TextSnapshot:
    """One stable read used to detect changes before an installer commit."""

    existed: bool
    text: str
    raw: bytes
    stat_token: tuple[int, int, int, int, int, int] | None

    @property
    def mode(self) -> int | None:
        """Return the original POSIX mode when the target existed."""
        if self.stat_token is None:
            return None
        return stat.S_IMODE(self.stat_token[2])


class _EditLockUnavailable:
    """Sentinel distinguishing lock failure from an unreadable edit target."""


_EDIT_LOCK_UNAVAILABLE = _EditLockUnavailable()


def _stat_token(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return identity and mutation fields for one open regular file.

    :param os.stat_result value: Metadata captured from the open file descriptor.
    :return tuple[int, int, int, int, int, int]: Stable comparison token.
    """
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_editable_text(path: Path) -> _TextSnapshot | None:
    """Read a stable regular non-symlink file, or describe a missing target.

    :param Path path: Candidate config path.
    :return _TextSnapshot | None: Stable raw/text snapshot for an editable
        target, or ``None`` for a symlink, non-file, invalid UTF-8, file that
        changed while being read, or another unreadable path.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        if path.is_symlink() or path.exists():
            return None
        return _TextSnapshot(False, "", b"", None)
    except OSError:
        return None

    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError:
        return None
    if _stat_token(before) != _stat_token(after):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return None
    return _TextSnapshot(True, text, raw, _stat_token(after))


def _edit_lock_path(path: Path) -> Path:
    """Return the persistent host-local sidecar lock for ``path``.

    The shared lock directory avoids leaving lock artifacts in user projects.
    The digest is only a bounded filename for the absolute target path; it is
    not an integrity check.

    :param Path path: User config path whose transactions must serialize.
    :return Path: Persistent lock file in the shared host-local lock directory.
    """
    absolute = str(_absolute_path(path))
    digest = hashlib.sha256(os.fsencode(absolute)).hexdigest()
    return lock_dir() / f"installer-{digest}.lock"


@contextlib.contextmanager
def _locked_editable_text(
    path: Path,
) -> Iterator[_TextSnapshot | _EditLockUnavailable | None]:
    """Serialize one target's read-modify-write transaction and read it once.

    :param Path path: Config or instructions file being edited.
    :return Iterator: Stable snapshot while the scoped lock is held,
        ``_EDIT_LOCK_UNAVAILABLE`` when locking fails, or ``None`` when the
        edit target is unsafe or unreadable.
    """
    handle: IO[str] | None = None
    try:
        handle = open_lock_file(_edit_lock_path(path))
        fcntl.flock(handle, fcntl.LOCK_EX)
    except (OSError, UnsafeLockPathError):
        if handle is not None:
            handle.close()
        yield _EDIT_LOCK_UNAVAILABLE
        return

    try:
        yield _read_editable_text(path)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()


def _read_editable_text_at(parent_fd: int, leaf: str) -> _TextSnapshot | None:
    """Read an editable leaf relative to one already-validated parent directory.

    :param int parent_fd: Open descriptor for the target's parent directory.
    :param str leaf: Target filename relative to ``parent_fd``.
    :return _TextSnapshot | None: Stable snapshot, missing-file description,
        or ``None`` when the leaf is unsafe or unreadable.
    """
    flags = os.O_RDONLY | os.O_CLOEXEC | _nofollow_flag() | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(leaf, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return _TextSnapshot(False, "", b"", None)
    except OSError:
        return None

    try:
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError:
        return None
    if _stat_token(before) != _stat_token(after):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return None
    return _TextSnapshot(True, text, raw, _stat_token(after))


def _snapshot_matches(parent_fd: int, leaf: str, expected: _TextSnapshot) -> bool:
    """Return whether one dir-fd-relative target still matches ``expected``.

    :param int parent_fd: Open descriptor for the target's parent directory.
    :param str leaf: Target filename relative to ``parent_fd``.
    :param _TextSnapshot expected: Snapshot captured at transaction start.
    :return bool: Whether identity, metadata, and raw bytes still match.
    """
    current = _read_editable_text_at(parent_fd, leaf)
    if current is None:
        return False
    if current.existed != expected.existed:
        return False
    if not expected.existed:
        return True
    return current.stat_token == expected.stat_token and current.raw == expected.raw


def _new_temporary_fd(parent_fd: int, leaf: str, mode: int) -> tuple[int, str]:
    """Create an umask-governed temporary file relative to ``parent_fd``.

    :param int parent_fd: Open descriptor for the target's parent directory.
    :param str leaf: Destination filename used to make the temporary name recognizable.
    :param int mode: Requested creation mode, subject to the process umask.
    :return tuple[int, str]: Open descriptor and temporary filename.
    :raises FileExistsError: If ten random temporary names all collide.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | _nofollow_flag()
    for _ in range(10):
        temporary = f".{leaf}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(temporary, flags, mode, dir_fd=parent_fd), temporary
        except FileExistsError:
            continue
    raise FileExistsError(f"Unable to create a temporary file for {leaf!r}.")


def _atomic_write_text(path: Path, text: str, *, expected: _TextSnapshot) -> bool:
    """Atomically replace ``path`` with UTF-8 ``text`` in the same directory.

    Existing permissions are retained. The replacement is refused when the
    target no longer matches the transaction's original snapshot. A failed or
    refused write leaves the then-current file untouched and removes the
    temporary file.

    :param Path path: Destination config path.
    :param str text: Complete replacement contents.
    :param _TextSnapshot expected: Snapshot that must still match before replace.
    :return bool: Whether the replacement completed successfully.
    """
    parent_fd = -1
    temporary: str | None = None
    try:
        parent_fd = _open_directory_fd(
            path.parent,
            create=True,
            private_final=False,
            umask_created_dirs=True,
        )
        leaf = _leaf_name(path)
        create_mode = expected.mode if expected.mode is not None else 0o666
        fd, temporary = _new_temporary_fd(parent_fd, leaf, create_mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            if expected.mode is not None:
                os.fchmod(handle.fileno(), expected.mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if not _snapshot_matches(parent_fd, leaf, expected):
            return False
        os.replace(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temporary = None
        os.fsync(parent_fd)
        return True
    except (OSError, UnsafePrivatePathError):
        return False
    finally:
        if temporary is not None and parent_fd >= 0:
            with contextlib.suppress(OSError):
                os.unlink(temporary, dir_fd=parent_fd)
        if parent_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(parent_fd)


def _detected_indent(text: str) -> str:
    """Return the indentation unit used by a JSON document.

    :param str text: Raw JSON text.
    :return str: The first line's leading whitespace, or two spaces when the
        document has no indented line to learn from.
    """
    match = _INDENT_PATTERN.search(text)
    return match.group(1) if match else "  "


def _detected_newline(text: str) -> str:
    """Return the first newline convention used by a text document."""
    newline = text.find("\n")
    return "\r\n" if newline > 0 and text[newline - 1] == "\r" else "\n"


def _dump_json(
    data: dict[str, object],
    *,
    indent: str,
    newline: str,
    trailing_newline: bool,
) -> str:
    """Serialize a JSON document preserving the source file's surface style.

    :param dict[str, object] data: Document to serialize.
    :param str indent: Indentation unit to apply.
    :param str newline: Newline convention to apply.
    :param bool trailing_newline: Whether the output should end with a newline.
    :return str: Serialized JSON text.
    """
    text = json.dumps(data, indent=indent, allow_nan=False, ensure_ascii=False).replace(
        "\n", newline
    )
    return text + newline if trailing_newline else text


def manual_json_snippet(container_key: str, member_key: str, entry: dict[str, object]) -> str:
    """Render the JSON snippet an operator must merge by hand after a skip.

    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name within the container.
    :param dict[str, object] entry: Server entry value.
    :return str: Pretty-printed snippet of the member inside its container.
    """
    return json.dumps({container_key: {member_key: entry}}, indent=2, allow_nan=False)


def _edit_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    *,
    operation: Literal["merge", "remove"],
    entry: dict[str, object] | None,
    managed: MemberPredicate,
    dry_run: bool = False,
) -> Action:
    """Apply one managed JSON member mutation through a locked transaction.

    :param Path path: Client configuration path.
    :param str container_key: Top-level server-map key.
    :param str member_key: Managed server name.
    :param Literal operation: Mutation to apply.
    :param dict[str, object] | None entry: Entry used by a merge.
    :param MemberPredicate managed: Predicate recognizing an owned existing entry.
    :param bool dry_run: Return the action without writing.
    :return Action: Edit result.
    """
    with _locked_editable_text(path) as loaded:
        if isinstance(loaded, _EditLockUnavailable):
            return "lock-error"
        if loaded is None:
            return "error"
        text = loaded.text
        if operation == "remove" and not loaded.existed:
            return "not-found"
        if operation == "merge" and not text.strip():
            assert entry is not None
            if dry_run:
                return "updated" if loaded.existed else "created"
            try:
                updated = _dump_json(
                    {container_key: {member_key: entry}},
                    indent="  ",
                    newline="\n",
                    trailing_newline=True,
                )
            except (TypeError, ValueError):
                return "error"
            written = _atomic_write_text(path, updated, expected=loaded)
            return ("updated" if loaded.existed else "created") if written else "error"

        try:
            data = strict_json_loads(text, finite_floats=True) if text.strip() else {}
        except ValueError:
            return "skipped"
        if not isinstance(data, dict):
            return "error"
        if operation == "remove" and container_key not in data:
            return "not-found"
        container = data.setdefault(container_key, {})
        if not isinstance(container, dict):
            return "error"
        if operation == "merge":
            assert entry is not None
            current = container.get(member_key)
            if member_key in container and current == entry:
                return "unchanged"
            if member_key in container and not managed(current):
                return "conflict"
            container[member_key] = entry
            action: Action = "updated"
        else:
            if member_key not in container:
                return "not-found"
            if not managed(container[member_key]):
                return "conflict"
            del container[member_key]
            action = "removed"
        if dry_run:
            return action
        try:
            updated = _dump_json(
                data,
                indent=_detected_indent(text),
                newline=_detected_newline(text),
                trailing_newline=text.endswith("\n"),
            )
        except (TypeError, ValueError):
            return "error"
        written = _atomic_write_text(path, updated, expected=loaded)
        return action if written else "error"


def merge_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    entry: dict[str, object],
    *,
    managed: MemberPredicate,
    dry_run: bool = False,
) -> Action:
    """Add or update one installer-managed JSON member.

    :param Path path: Client configuration path.
    :param str container_key: Top-level server-map key.
    :param str member_key: Managed server name.
    :param dict[str, object] entry: Entry to install.
    :param MemberPredicate managed: Predicate recognizing an owned existing entry.
    :param bool dry_run: Return the action without writing.
    :return Action: Edit result.
    """
    return _edit_json_member(
        path,
        container_key,
        member_key,
        operation="merge",
        entry=entry,
        managed=managed,
        dry_run=dry_run,
    )


def remove_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    *,
    managed: MemberPredicate,
    dry_run: bool = False,
) -> Action:
    """Remove one installer-managed JSON member.

    :param Path path: Client configuration path.
    :param str container_key: Top-level server-map key.
    :param str member_key: Managed server name.
    :param MemberPredicate managed: Predicate recognizing an owned existing entry.
    :param bool dry_run: Return the action without writing.
    :return Action: Edit result.
    """
    return _edit_json_member(
        path,
        container_key,
        member_key,
        operation="remove",
        entry=None,
        managed=managed,
        dry_run=dry_run,
    )


def render_marked_block(content: str, *, start: str, end: str) -> str:
    """Render a marker-fenced block ready to append to a file.

    :param str content: Block body; surrounding whitespace is normalized.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return str: ``start`` line, body, ``end`` line, trailing newline.
    """
    return f"{start}\n{content.strip()}\n{end}\n"


def _marked_span(existing: str, *, start: str, end: str) -> tuple[int, int] | None:
    """Locate one well-formed marker pair in ``existing``.

    :param str existing: Text that may contain the marker pair.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return tuple[int, int] | None: Marker indices, or ``None`` when both are absent.
    :raises ValueError: If markers are incomplete, repeated, or out of order.
    """
    start_matches = list(re.finditer(rf"(?m)^{re.escape(start)}(?=\r?$)", existing))
    end_matches = list(re.finditer(rf"(?m)^{re.escape(end)}(?=\r?$)", existing))
    if not start_matches and not end_matches:
        return None
    if len(start_matches) != 1 or len(end_matches) != 1:
        raise ValueError("existing marker block is incomplete or repeated")
    start_idx = start_matches[0].start()
    end_idx = end_matches[0].start()
    if end_idx <= start_idx:
        raise ValueError("existing marker block is out of order")
    return start_idx, end_idx


def updated_marked_text(existing: str, content: str, *, start: str, end: str) -> str:
    """Return ``existing`` with one marker-fenced block replaced or appended.

    :param str existing: Existing file text.
    :param str content: Block body to place between the markers.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return str: Complete updated file text.
    :raises ValueError: If the existing text has an unmatched marker.
    """
    span = _marked_span(existing, start=start, end=end)
    if span is not None:
        start_idx, end_idx = span
        marker_end = start_idx + len(start)
        newline = "\r\n" if existing[marker_end : marker_end + 2] == "\r\n" else "\n"
        block = render_marked_block(content, start=start, end=end)
        if newline == "\r\n":
            block = block.replace("\r\n", "\n").replace("\n", "\r\n")
        return existing[:start_idx] + block.removesuffix(newline) + existing[end_idx + len(end) :]
    newline = _detected_newline(existing)
    block = render_marked_block(content, start=start, end=end)
    if newline == "\r\n":
        block = block.replace("\r\n", "\n").replace("\n", "\r\n")
    separator = "" if not existing else newline
    return existing + separator + block


def removed_marked_text(existing: str, *, start: str, end: str) -> str | None:
    """Return ``existing`` without one marker block, or ``None`` when absent.

    :param str existing: Existing file text.
    :param str start: Exact standalone start marker line.
    :param str end: Exact standalone end marker line.
    :return str | None: Remaining text, or ``None`` when no marker pair exists.
    :raises ValueError: If marker lines are incomplete, repeated, or out of order.
    """
    span = _marked_span(existing, start=start, end=end)
    if span is None:
        return None
    start_idx, end_idx = span

    cut_end = end_idx + len(end)
    if existing[cut_end : cut_end + 2] == "\r\n":
        cut_end += 2
    elif existing[cut_end : cut_end + 1] == "\n":
        cut_end += 1
    before = existing[:start_idx]
    marker_end = start_idx + len(start)
    separator = "\r\n" if existing[marker_end : marker_end + 2] == "\r\n" else "\n"
    if before.endswith(separator):
        # Drop the separator appended at install time. Matching the managed
        # block's newline style also handles a file later normalized to CRLF.
        before = before[: -len(separator)]
    return before + existing[cut_end:]
