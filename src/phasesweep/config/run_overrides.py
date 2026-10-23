"""Apply ``run --set KEY=VALUE`` assignments to a raw config mapping before validation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import yaml
from pydantic import ValidationError

from phasesweep.config.io import load_yaml_document
from phasesweep.runtime.json import JsonObject, JsonValue, json_value

SET_SYNTAX = (
    "Accepted syntax: dot-separated names (env.NOTE); a list element by zero-based index, "
    "as .N or [N] (phases.0, phases[0]); a phase by name, as [NAME] (phases[lr]); "
    "a mapping key taken verbatim, as [KEY] (search_space[model.n_layers]), "
    "where \\] and \\\\ stand for a literal ] and \\ inside the brackets."
)

_INDEX = re.compile(r"[0-9]+")
_SHOWN_LIMIT = 100
_NAME_END = ".[]="

Location = tuple[str | int, ...]


class SetPathError(ValueError):
    """A ``--set`` assignment that cannot be parsed or cannot address the config."""


@dataclass(frozen=True)
class CreatedKey:
    """A mapping key an assignment added to the config.

    :param str setting: The assignment as the user wrote it.
    :param str written: The key path up to and including the new key, as written.
    :param Location location: The same position as a validation error ``loc``.
    """

    setting: str
    written: str
    location: Location


@dataclass(frozen=True)
class _Segment:
    """One step of a ``--set`` key: a dotted name or a bracketed selector.

    :param str name: The name, index text or bracket contents with escapes resolved.
    :param bool bracketed: Whether it was written as ``[...]``.
    :param str text: The step as written, brackets included.
    :param int end: Offset in the key just past this step.
    """

    name: str
    bracketed: bool
    text: str
    end: int


def apply_set(payload: JsonObject, setting: str) -> CreatedKey | None:
    """Assign one ``KEY=VALUE`` setting into ``payload`` in place.

    Mapping keys the path names that the payload lacks are created, so optional
    blocks can be filled in; the first one created is returned so a later validation
    error there can name this setting. List positions are never created.

    :param JsonObject payload: Raw config mapping to edit.
    :param str setting: ``KEY=VALUE`` with a YAML ``VALUE``.
    :raises SetPathError: The key is malformed or does not address the payload, or the
        value is not valid YAML.
    :return CreatedKey | None: The mapping key this assignment created, if any.
    """
    segments, raw = _parse_key(setting)
    try:
        value = json_value(load_yaml_document(raw))
    except yaml.MarkedYAMLError as exc:
        # Its text quotes the offending line of the value, which may be a secret.
        problem = ", ".join(part for part in (exc.context, exc.problem) if part)
        mark = exc.problem_mark
        where = "" if mark is None else f" at column {mark.column + 1} of the value"
        raise SetPathError(
            f"{_shown(setting)}: the value is not valid YAML: {problem}{where}"
        ) from None
    except (ValueError, yaml.YAMLError) as exc:
        raise SetPathError(f"{_shown(setting)}: the value is not valid YAML: {exc}") from exc
    key = setting[: segments[-1].end]
    container: JsonValue = payload
    location: list[str | int] = []
    created: CreatedKey | None = None
    before = 0
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        prefix = key[:before]
        if isinstance(container, dict):
            name = segment.name
            if name not in container:
                container[name] = {}
                if created is None:
                    created = CreatedKey(setting, key[: segment.end], (*location, name))
            if last:
                container[name] = value
            else:
                location.append(name)
                container = container[name]
        elif isinstance(container, list):
            position = _list_position(setting, segment, prefix, container)
            if last:
                container[position] = value
            else:
                location.append(position)
                container = container[position]
        else:
            raise _path_error(
                setting,
                f"segment {segment.text!r} cannot descend into the {_kind(container)} value "
                f"at {prefix!r}",
            )
        before = segment.end
    return created


def created_key_notes(exc: ValidationError, created: Sequence[CreatedKey]) -> list[str]:
    """Name the assignments that created a key where validation failed.

    :param ValidationError exc: Validation failure of the derived config.
    :param Sequence[CreatedKey] created: Keys the run's assignments created.
    :return list[str]: One note per assignment whose new key holds or contains an error.
    """
    notes: list[str] = []
    for error in exc.errors(include_input=False):
        for key in created:
            if _at_or_below(error["loc"], key.location):
                note = (
                    f"note: --set {_shown(key.setting)} created {key.written!r} because the "
                    "config had no such key, and validation failed at or below it. "
                    "Check the spelling of each segment."
                )
                if note not in notes:
                    notes.append(note)
    return notes


def _at_or_below(loc: Location, location: Location) -> bool:
    """Report whether an error ``loc`` is the given position or lies beneath it.

    Pydantic inserts union tags (such as ``int`` for a search-space entry) into a
    ``loc``; text elements that do not match are skipped once the path has started.
    """
    matched = 0
    for part in loc:
        if matched == len(location):
            return True
        if part == location[matched]:
            matched += 1
        elif matched == 0 or not isinstance(part, str):
            return False
    return matched == len(location)


def _parse_key(setting: str) -> tuple[list[_Segment], str]:
    """Split ``KEY=VALUE`` into key segments and the raw value text."""
    segments: list[_Segment] = []
    pos = 0
    expect_name = True
    while pos < len(setting):
        char = setting[pos]
        if char == "=":
            break
        if char == "[":
            if expect_name:
                raise _path_error(setting, f"the '[' at offset {pos} has no name before it")
            end, name = _read_bracket(setting, pos)
            segments.append(_Segment(name, True, setting[pos:end], end))
            pos = end
        elif char == ".":
            if expect_name:
                raise _path_error(setting, f"empty path segment before the '.' at offset {pos}")
            expect_name = True
            pos += 1
        elif char == "]":
            raise _path_error(setting, f"unmatched ']' at offset {pos}")
        else:
            if not expect_name:
                raise _path_error(setting, f"{char!r} at offset {pos} must follow a '.'")
            end = pos
            while end < len(setting) and setting[end] not in _NAME_END:
                end += 1
            segments.append(_Segment(setting[pos:end], False, setting[pos:end], end))
            expect_name = False
            pos = end
    else:
        raise _path_error(setting, "expected KEY=VALUE")
    if not segments:
        raise _path_error(setting, "expected KEY=VALUE with a nonempty KEY")
    if expect_name:
        raise _path_error(setting, "empty path segment: the key ends with a '.'")
    return segments, setting[pos + 1 :]


def _read_bracket(setting: str, start: int) -> tuple[int, str]:
    """Read ``[...]`` from ``start`` and return the offset past ``]`` and its contents."""
    chars: list[str] = []
    pos = start + 1
    while pos < len(setting):
        char = setting[pos]
        if char == "\\" and setting[pos + 1 : pos + 2] in ("\\", "]"):
            chars.append(setting[pos + 1])
            pos += 2
        elif char == "]":
            return pos + 1, "".join(chars)
        else:
            chars.append(char)
            pos += 1
    raise _path_error(setting, f"the '[' at offset {start} has no closing ']'")


def _list_position(setting: str, segment: _Segment, prefix: str, items: list[JsonValue]) -> int:
    """Resolve an index or, for a bracketed name, the element whose ``name`` matches."""
    if _INDEX.fullmatch(segment.name):
        position = int(segment.name)
        if position >= len(items):
            size = f"has {len(items)} elements" if items else "is empty"
            raise _path_error(
                setting,
                f"segment {segment.text!r}: index {position} is out of range; {prefix!r} {size}",
            )
        return position
    if not segment.bracketed:
        raise _path_error(
            setting,
            f"segment {segment.text!r} is not a list index after {prefix!r}; "
            f"select an element by its name with [NAME]",
        )
    names = [item.get("name") if isinstance(item, dict) else None for item in items]
    matches = [i for i, name in enumerate(names) if name == segment.name]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise _path_error(
            setting,
            f"segment {segment.text!r}: {len(matches)} elements of {prefix!r} are named "
            f"{segment.name!r}; use an index",
        )
    known = ", ".join(name for name in names if isinstance(name, str)) or "none"
    raise _path_error(
        setting,
        f"segment {segment.text!r}: no element of {prefix!r} is named {segment.name!r}; "
        f"names: {known}",
    )


def _path_error(setting: str, problem: str) -> SetPathError:
    """Build a path error that quotes the assignment and repeats the accepted syntax."""
    return SetPathError(f"{_shown(setting)}: {problem}. {SET_SYNTAX}")


def _shown(setting: str) -> str:
    """Quote an assignment's key for a message, shortening a long one.

    The value is never shown: like the config itself, it can hold a credential. The key
    ends at the first ``=`` outside ``[...]``, or at the first ``=`` when a ``[`` never
    closes.
    """
    key = setting.partition("=")[0]
    pos, inside = 0, False
    while pos < len(setting):
        char = setting[pos]
        if inside and char == "\\":
            pos += 1
        elif char == "[" or char == "]":
            inside = char == "["
        elif char == "=" and not inside:
            key = setting[:pos]
            break
        pos += 1
    if len(key) > _SHOWN_LIMIT:
        key = f"{key[:_SHOWN_LIMIT]}..."
    return repr(key)


def _kind(value: JsonValue) -> str:
    """Name a scalar's JSON type for a message."""
    return "null" if value is None else type(value).__name__
