"""File-edit primitives for installer integrations.

Two edit families with one safety contract - touch only what phasesweep owns,
and make uninstall the inverse of install:

- JSON member edits: parse the whole document, change exactly one member under
  one container key, and re-serialize with the file's own indentation. Files
  that do not parse as strict JSON (comments, JSON5) are never modified; the
  caller gets ``"skipped"`` and prints a manual snippet instead.
- Marker-fenced text blocks: replace-or-append a block between start/end
  marker lines, leaving every byte outside the markers alone. Removal restores
  the original file byte-identically (including its final-newline state) and
  unlinks a file that becomes empty.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

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


def _read_editable_text(path: Path) -> tuple[bool, str] | None:
    """Read a regular non-symlink file, or describe a missing target.

    :param Path path: Candidate config path.
    :return tuple[bool, str] | None: ``(exists, text)`` for an editable target,
        or ``None`` for a symlink, non-file, or unreadable path.
    """
    if path.is_symlink() or (path.exists() and not path.is_file()):
        return None
    existed = path.exists()
    if not existed:
        return False, ""
    try:
        return True, path.read_text(encoding="utf-8")
    except OSError:
        return None


def _atomic_write_text(path: Path, text: str) -> bool:
    """Atomically replace ``path`` with UTF-8 ``text`` in the same directory.

    Existing permissions are retained. A failed write or replace leaves the
    original file untouched and removes the temporary file.

    :param Path path: Destination config path.
    :param str text: Complete replacement contents.
    :return bool: Whether the replacement completed successfully.
    """
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
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


def _dump_json(data: dict[str, object], *, indent: str, trailing_newline: bool) -> str:
    """Serialize a JSON document preserving the source file's surface style.

    :param dict[str, object] data: Document to serialize.
    :param str indent: Indentation unit to apply.
    :param bool trailing_newline: Whether the output should end with a newline.
    :return str: Serialized JSON text.
    """
    text = json.dumps(data, indent=indent)
    return text + "\n" if trailing_newline else text


def manual_json_snippet(container_key: str, member_key: str, entry: dict[str, object]) -> str:
    """Render the JSON snippet an operator must merge by hand after a skip.

    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name within the container.
    :param dict[str, object] entry: Server entry value.
    :return str: Pretty-printed snippet of the member inside its container.
    """
    return json.dumps({container_key: {member_key: entry}}, indent=2)


def merge_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    entry: dict[str, object],
    *,
    managed: MemberPredicate | None = None,
) -> Action:
    """Add or update ``container_key.member_key = entry`` in a JSON config file.

    Missing or empty files are created fresh. Existing files are parsed as
    strict JSON and rewritten with their detected indentation; key order and
    all unrelated data survive the round trip. Comment-bearing configs
    (``.jsonc``/JSON5) fail strict parsing and are left untouched.

    :param Path path: Client config file to edit.
    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name to add or update within the container.
    :param dict[str, object] entry: Server entry value to store.
    :param MemberPredicate | None managed: Optional predicate identifying an
        existing member as installer-managed and therefore safe to update.
    :return Action: ``created``/``updated``/``unchanged`` on success,
        ``skipped`` when the file is not strict JSON, ``error`` when the
        document or container is not an object, or ``conflict`` when a
        differing unmanaged member already exists.
    """
    loaded = _read_editable_text(path)
    if loaded is None:
        return "error"
    existed, text = loaded
    if not text.strip():
        written = _atomic_write_text(
            path,
            _dump_json({container_key: {member_key: entry}}, indent="  ", trailing_newline=True),
        )
        return ("updated" if existed else "created") if written else "error"

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
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
    written = _atomic_write_text(
        path,
        _dump_json(data, indent=_detected_indent(text), trailing_newline=text.endswith("\n")),
    )
    return "updated" if written else "error"


def remove_json_member(
    path: Path,
    container_key: str,
    member_key: str,
    *,
    managed: MemberPredicate | None = None,
) -> Action:
    """Remove ``container_key.member_key`` from a JSON config file.

    Prunes the container when it becomes empty and unlinks the file when the
    whole document becomes empty, so a file install created disappears again.

    :param Path path: Client config file to edit.
    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name to remove from the container.
    :param MemberPredicate | None managed: Optional predicate the existing
        member must satisfy before removal.
    :return Action: ``removed`` on success, ``not-found`` when the file or
        member is absent, ``skipped`` when the file is not strict JSON,
        ``error`` when the document or container is not an object, or
        ``conflict`` when the member is not installer-managed.
    """
    loaded = _read_editable_text(path)
    if loaded is None:
        return "error"
    existed, text = loaded
    if not existed:
        return "not-found"
    try:
        data = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
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
    if not container:
        del data[container_key]
    if not data:
        try:
            path.unlink()
        except OSError:
            return "error"
        else:
            return "removed"
    written = _atomic_write_text(
        path,
        _dump_json(data, indent=_detected_indent(text), trailing_newline=text.endswith("\n")),
    )
    return "removed" if written else "error"


def render_marked_block(content: str, *, start: str, end: str) -> str:
    """Render a marker-fenced block ready to append to a file.

    :param str content: Block body; surrounding whitespace is normalized.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return str: ``start`` line, body, ``end`` line, trailing newline.
    """
    return f"{start}\n{content.strip()}\n{end}\n"


def updated_marked_text(existing: str, content: str, *, start: str, end: str) -> str:
    """Return ``existing`` with one marker-fenced block replaced or appended.

    :param str existing: Existing file text.
    :param str content: Block body to place between the markers.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return str: Complete updated file text.
    :raises ValueError: If the existing text has an unmatched marker.
    """
    block = render_marked_block(content, start=start, end=end)
    start_idx = existing.find(start)
    end_idx = existing.find(end)
    if (start_idx == -1) != (end_idx == -1) or (start_idx != -1 and end_idx <= start_idx):
        raise ValueError("existing marker block is incomplete or out of order")
    if start_idx != -1:
        return existing[:start_idx] + block.rstrip("\n") + existing[end_idx + len(end) :]
    separator = "" if not existing else "\n"
    return existing + separator + block


def replace_or_append_marked(path: Path, content: str, *, start: str, end: str) -> Action:
    """Install a marker-fenced block into a text file, idempotently.

    An existing fenced block is replaced in place (bytes outside the markers
    untouched); otherwise the block is appended after a newline separator.

    :param Path path: Text file to edit (created if missing).
    :param str content: Block body to place between the markers.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return Action: ``created``/``updated``/``unchanged``.
    """
    loaded = _read_editable_text(path)
    if loaded is None:
        return "error"
    existed, existing = loaded
    try:
        updated = updated_marked_text(existing, content, start=start, end=end)
    except ValueError:
        return "error"
    if updated == existing:
        return "unchanged"
    if not _atomic_write_text(path, updated):
        return "error"
    return "created" if not existed else "updated"


def remove_marked(path: Path, *, start: str, end: str) -> Action:
    """Remove a marker-fenced block, undoing :func:`replace_or_append_marked`.

    The pre-install bytes come back exactly, including whether the original
    ended with a newline. A file left empty (or whitespace-only) is unlinked.

    :param Path path: Text file to edit.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return Action: ``removed`` on success, ``not-found`` when the file or a
        well-formed marker pair is absent.
    """
    loaded = _read_editable_text(path)
    if loaded is None:
        return "error"
    existed, existing = loaded
    if not existed:
        return "not-found"
    start_idx = existing.find(start)
    end_idx = existing.find(end)
    if start_idx == -1 and end_idx == -1:
        return "not-found"
    if start_idx == -1 or end_idx <= start_idx:
        return "error"

    cut_end = end_idx + len(end)
    if existing[cut_end : cut_end + 1] == "\n":
        cut_end += 1
    before = existing[:start_idx]
    if before.endswith("\n"):
        # Drop the separator newline appended at install time. This restores
        # both newline-terminated and non-newline-terminated originals exactly.
        before = before[:-1]
    remainder = before + existing[cut_end:]
    if not remainder.strip():
        try:
            path.unlink()
        except OSError:
            return "error"
        else:
            return "removed"
    return "removed" if _atomic_write_text(path, remainder) else "error"
