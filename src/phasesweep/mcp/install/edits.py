"""File-edit primitives for installer integrations.

Two edit families share one safety contract: modify only installer-managed
content, refuse same-name conflicts, and serialize each target's complete
read-modify-write transaction:

- JSON member edits: parse the whole document, change one member under one
  container key, and re-serialize with the file's detected indentation. Key
  order is retained, but whitespace and finite-number spellings may change.
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
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, TypeAlias

from phasesweep.runtime.files import lock_dir

Action: TypeAlias = Literal[
    "created",
    "updated",
    "unchanged",
    "skipped",
    "conflict",
    "error",
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
        if os.path.lexists(path):
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
    absolute = os.path.abspath(os.fspath(path))
    digest = hashlib.sha256(os.fsencode(absolute)).hexdigest()
    return lock_dir() / f"installer-{digest}.lock"


@contextlib.contextmanager
def _locked_editable_text(path: Path) -> Iterator[_TextSnapshot | None]:
    """Serialize one target's read-modify-write transaction and read it once.

    :param Path path: Config or instructions file being edited.
    :return Iterator[_TextSnapshot | None]: Stable snapshot while the scoped
        lock is held, or ``None`` when locking or reading is unsafe.
    """
    fd: int | None = None
    handle: IO[str] | None = None
    try:
        fd = os.open(_edit_lock_path(path), os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(fd, "a+", encoding="utf-8")
        fd = None
        fcntl.flock(handle, fcntl.LOCK_EX)
    except OSError:
        if handle is not None:
            handle.close()
        elif fd is not None:
            os.close(fd)
        yield None
        return

    try:
        yield _read_editable_text(path)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()


def _snapshot_matches(path: Path, expected: _TextSnapshot) -> bool:
    """Return whether ``path`` still exactly matches ``expected``.

    :param Path path: Target to compare immediately before replacement.
    :param _TextSnapshot expected: Snapshot captured at transaction start.
    :return bool: Whether identity, metadata, and raw bytes still match.
    """
    current = _read_editable_text(path)
    if current is None:
        return False
    if current.existed != expected.existed:
        return False
    if not expected.existed:
        return True
    return current.stat_token == expected.stat_token and current.raw == expected.raw


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
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if expected.mode is not None:
            os.chmod(temporary, expected.mode)
        if not _snapshot_matches(path, expected):
            return False
        os.replace(temporary, path)
        temporary = None
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


def _detected_indent(text: str) -> str:
    """Return the indentation unit used by a JSON document.

    :param str text: Raw JSON text.
    :return str: The first line's leading whitespace, or two spaces when the
        document has no indented line to learn from.
    """
    match = _INDENT_PATTERN.search(text)
    return match.group(1) if match else "  "


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate member names.

    :param list[tuple[str, object]] pairs: Parsed object members in source order.
    :raises ValueError: If a member name occurs more than once.
    :return dict[str, object]: Ordered unique member mapping.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    """Parse one JSON float and reject values outside finite float range.

    :param str value: Raw JSON numeric token.
    :raises ValueError: If the token overflows to a non-finite Python float.
    :return float: Finite parsed value.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _reject_json_constant(value: str) -> object:
    """Reject Python JSON extensions such as ``NaN`` and ``Infinity``.

    :param str value: Non-standard constant token accepted by Python's parser.
    :raises ValueError: Always; strict JSON has no such constants.
    :return object: This function never returns.
    """
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_strict_json(text: str) -> object:
    """Parse strict JSON with unique keys and finite numeric values.

    :param str text: Complete JSON document.
    :raises ValueError: If syntax, key uniqueness, or numeric finiteness is invalid.
    :return object: Parsed JSON value.
    """
    return json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_float=_finite_json_float,
        parse_constant=_reject_json_constant,
    )


def _dump_json(data: dict[str, object], *, indent: str, trailing_newline: bool) -> str:
    """Serialize a JSON document preserving the source file's surface style.

    :param dict[str, object] data: Document to serialize.
    :param str indent: Indentation unit to apply.
    :param bool trailing_newline: Whether the output should end with a newline.
    :return str: Serialized JSON text.
    """
    text = json.dumps(data, indent=indent, allow_nan=False)
    return text + "\n" if trailing_newline else text


def manual_json_snippet(container_key: str, member_key: str, entry: dict[str, object]) -> str:
    """Render the JSON snippet an operator must merge by hand after a skip.

    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name within the container.
    :param dict[str, object] entry: Server entry value.
    :return str: Pretty-printed snippet of the member inside its container.
    """
    return json.dumps({container_key: {member_key: entry}}, indent=2, allow_nan=False)


def merge_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    entry: dict[str, object],
    *,
    managed: MemberPredicate | None = None,
    dry_run: bool = False,
) -> Action:
    """Add or update ``container_key.member_key = entry`` in a JSON config file.

    Missing or empty files are created fresh. Existing files are parsed as
    strict JSON and rewritten with their detected indentation; key order and
    supported values survive at Python's JSON precision. Comment-bearing
    configs (``.jsonc``/JSON5) fail strict parsing and are left untouched.

    :param Path path: Client config file to edit.
    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name to add or update within the container.
    :param dict[str, object] entry: Server entry value to store.
    :param MemberPredicate | None managed: Optional predicate identifying an
        existing member as installer-managed and therefore safe to update.
    :param bool dry_run: Compute the action without writing the file.
    :return Action: ``created``/``updated``/``unchanged`` on success,
        ``skipped`` when the file is not strict JSON, ``error`` when the
        document or container is not an object, or ``conflict`` when a
        differing unmanaged member already exists.
    """
    with _locked_editable_text(path) as loaded:
        if loaded is None:
            return "error"
        text = loaded.text
        if not text.strip():
            if dry_run:
                return "updated" if loaded.existed else "created"
            try:
                updated = _dump_json(
                    {container_key: {member_key: entry}}, indent="  ", trailing_newline=True
                )
            except (TypeError, ValueError):
                return "error"
            written = _atomic_write_text(path, updated, expected=loaded)
            return ("updated" if loaded.existed else "created") if written else "error"

        try:
            data = _load_strict_json(text)
        except ValueError:
            return "skipped"
        if not isinstance(data, dict):
            return "error"
        container = data.setdefault(container_key, {})
        if not isinstance(container, dict):
            return "error"
        current = container.get(member_key)
        if member_key in container and current == entry:
            return "unchanged"
        if member_key in container and (managed is None or not managed(current)):
            return "conflict"
        container[member_key] = entry
        if dry_run:
            return "updated"
        try:
            updated = _dump_json(
                data,
                indent=_detected_indent(text),
                trailing_newline=text.endswith("\n"),
            )
        except (TypeError, ValueError):
            return "error"
        written = _atomic_write_text(path, updated, expected=loaded)
        return "updated" if written else "error"


def remove_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    *,
    managed: MemberPredicate | None = None,
    dry_run: bool = False,
) -> Action:
    """Remove ``container_key.member_key`` from a JSON config file.

    The containing object and config file remain even when empty because the
    installer does not persist enough provenance to prove it created them.

    :param Path path: Client config file to edit.
    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name to remove from the container.
    :param MemberPredicate | None managed: Optional predicate the existing
        member must satisfy before removal.
    :param bool dry_run: Compute the action without writing the file.
    :return Action: ``removed`` on success, ``not-found`` when the file or
        member is absent, ``skipped`` when the file is not strict JSON,
        ``error`` when the document or container is not an object, or
        ``conflict`` when the member is not installer-managed.
    """
    with _locked_editable_text(path) as loaded:
        if loaded is None:
            return "error"
        text = loaded.text
        if not loaded.existed:
            return "not-found"
        try:
            data = _load_strict_json(text) if text.strip() else {}
        except ValueError:
            return "skipped"
        if not isinstance(data, dict):
            return "error"
        if container_key not in data:
            return "not-found"
        container = data[container_key]
        if not isinstance(container, dict):
            return "error"
        if member_key not in container:
            return "not-found"
        if managed is not None and not managed(container[member_key]):
            return "conflict"
        del container[member_key]
        if dry_run:
            return "removed"
        try:
            updated = _dump_json(
                data,
                indent=_detected_indent(text),
                trailing_newline=text.endswith("\n"),
            )
        except (TypeError, ValueError):
            return "error"
        written = _atomic_write_text(path, updated, expected=loaded)
        return "removed" if written else "error"


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
    block = render_marked_block(content, start=start, end=end)
    separator = "" if not existing else "\n"
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


def replace_or_append_marked(
    path: Path,
    content: str,
    *,
    start: str,
    end: str,
    dry_run: bool = False,
) -> Action:
    """Install a marker-fenced block into a text file, idempotently.

    An existing fenced block is replaced in place (bytes outside the markers
    untouched); otherwise the block is appended after a newline separator.

    :param Path path: Text file to edit (created if missing).
    :param str content: Block body to place between the markers.
    :param str start: Start marker line.
    :param str end: End marker line.
    :param bool dry_run: Compute the action without writing the file.
    :return Action: ``created``/``updated``/``unchanged``.
    """
    with _locked_editable_text(path) as loaded:
        if loaded is None:
            return "error"
        try:
            updated = updated_marked_text(loaded.text, content, start=start, end=end)
        except ValueError:
            return "error"
        if updated == loaded.text:
            return "unchanged"
        if dry_run:
            return "created" if not loaded.existed else "updated"
        if not _atomic_write_text(path, updated, expected=loaded):
            return "error"
        return "created" if not loaded.existed else "updated"


def remove_marked(
    path: Path,
    *,
    start: str,
    end: str,
    dry_run: bool = False,
) -> Action:
    """Remove a marker-fenced block, undoing :func:`replace_or_append_marked`.

    The pre-install bytes come back exactly, including newline style and final
    newline state. The file remains even when empty because emptiness does not
    prove that the installer created the file.

    :param Path path: Text file to edit.
    :param str start: Start marker line.
    :param str end: End marker line.
    :param bool dry_run: Compute the action without writing the file.
    :return Action: ``removed`` on success, ``not-found`` when the file or a
        well-formed marker pair is absent.
    """
    with _locked_editable_text(path) as loaded:
        if loaded is None:
            return "error"
        if not loaded.existed:
            return "not-found"
        try:
            remainder = removed_marked_text(loaded.text, start=start, end=end)
        except ValueError:
            return "error"
        if remainder is None:
            return "not-found"
        if dry_run:
            return "removed"
        return "removed" if _atomic_write_text(path, remainder, expected=loaded) else "error"
