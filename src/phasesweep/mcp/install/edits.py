"""File-edit primitives for installer integrations.

Two edit families with one safety contract - touch only what phasesweep owns,
and make uninstall the inverse of install:

- JSON member edits: parse the whole document, change exactly one member under
  one container key, and re-serialize with the file's own indentation. Files
  that do not parse as strict JSON (comments, JSON5) are never modified; the
  caller gets ``"skipped"`` and prints a manual snippet instead.
- Marker-fenced text blocks: replace-or-append a block between start/end
  marker lines, leaving every byte outside the markers alone. Removal restores
  a newline-terminated file byte-identically (including the separator install
  added) and unlinks a file that becomes empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, TypeAlias

Action: TypeAlias = Literal[
    "created", "updated", "unchanged", "skipped", "error", "removed", "not-found"
]

_INDENT_PATTERN = re.compile(r"^([ \t]+)\S", re.MULTILINE)


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
    path: Path, container_key: str, member_key: str, entry: dict[str, object]
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
    :return Action: ``created``/``updated``/``unchanged`` on success,
        ``skipped`` when the file is not strict JSON, ``error`` when the
        document or container is not an object.
    """
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not text.strip():
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _dump_json({container_key: {member_key: entry}}, indent="  ", trailing_newline=True),
            encoding="utf-8",
        )
        return "updated" if existed else "created"

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "skipped"
    if not isinstance(data, dict):
        return "error"
    container = data.setdefault(container_key, {})
    if not isinstance(container, dict):
        return "error"
    if container.get(member_key) == entry:
        return "unchanged"
    container[member_key] = entry
    path.write_text(
        _dump_json(data, indent=_detected_indent(text), trailing_newline=text.endswith("\n")),
        encoding="utf-8",
    )
    return "updated"


def remove_json_member(path: Path, container_key: str, member_key: str) -> Action:
    """Remove ``container_key.member_key`` from a JSON config file.

    Prunes the container when it becomes empty and unlinks the file when the
    whole document becomes empty, so a file install created disappears again.

    :param Path path: Client config file to edit.
    :param str container_key: Top-level key holding the client's server map.
    :param str member_key: Server name to remove from the container.
    :return Action: ``removed`` on success, ``not-found`` when the file or
        member is absent, ``skipped`` when the file is not strict JSON,
        ``error`` when the document is not an object.
    """
    if not path.exists():
        return "not-found"
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return "skipped"
    if not isinstance(data, dict):
        return "error"
    container = data.get(container_key)
    if not isinstance(container, dict) or member_key not in container:
        return "not-found"
    del container[member_key]
    if not container:
        del data[container_key]
    if not data:
        path.unlink()
        return "removed"
    path.write_text(
        _dump_json(data, indent=_detected_indent(text), trailing_newline=text.endswith("\n")),
        encoding="utf-8",
    )
    return "removed"


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
    separator = "" if not existing else ("\n" if existing.endswith("\n") else "\n\n")
    return existing + separator + block


def replace_or_append_marked(path: Path, content: str, *, start: str, end: str) -> Action:
    """Install a marker-fenced block into a text file, idempotently.

    An existing fenced block is replaced in place (bytes outside the markers
    untouched); otherwise the block is appended after a blank-line separator.

    :param Path path: Text file to edit (created if missing).
    :param str content: Block body to place between the markers.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return Action: ``created``/``updated``/``unchanged``.
    """
    existed = path.exists()
    existing = path.read_text(encoding="utf-8") if existed else ""
    try:
        updated = updated_marked_text(existing, content, start=start, end=end)
    except ValueError:
        return "error"
    if updated == existing:
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return "created" if not existed else "updated"


def remove_marked(path: Path, *, start: str, end: str) -> Action:
    """Remove a marker-fenced block, undoing :func:`replace_or_append_marked`.

    For a newline-terminated file the pre-install bytes come back exactly,
    including the blank-line separator install added. A file left empty (or
    whitespace-only) is unlinked.

    :param Path path: Text file to edit.
    :param str start: Start marker line.
    :param str end: End marker line.
    :return Action: ``removed`` on success, ``not-found`` when the file or a
        well-formed marker pair is absent.
    """
    if not path.exists():
        return "not-found"
    existing = path.read_text(encoding="utf-8")
    start_idx = existing.find(start)
    end_idx = existing.find(end)
    if start_idx == -1 or end_idx <= start_idx:
        return "not-found"

    cut_end = end_idx + len(end)
    if existing[cut_end : cut_end + 1] == "\n":
        cut_end += 1
    before = existing[:start_idx]
    if before.endswith("\n\n"):
        # Drop the separator newline appended at install time.
        before = before[:-1]
    remainder = before + existing[cut_end:]
    if not remainder.strip():
        path.unlink()
        return "removed"
    path.write_text(remainder, encoding="utf-8")
    return "removed"
