"""Query the packaged configuration reference without importing tracker SDKs.

Parse config_reference.yaml into entries for CLI help, typo suggestions,
validation-error hints, and header notes.
"""

from __future__ import annotations

import difflib
import functools
import importlib.resources
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources.abc import Traversable

_SECTION = re.compile(r"# -{10} (.+?) -{10}")
_ENTRY = re.compile(r"( *)(- )?([\w.-]+): (.+)")
_VARIANT_SIGNATURE = re.compile(r'"([^"\n]+)" \(required\)')
_BRACKETS = re.compile(r"\[[^\]]*\]")
_TAG = re.compile(r"\[type=[^\]]*\]")
_TAG_NAME = re.compile(r"\[type=([^\]]*)\]")
_MAP_KEY = "<key>"


@dataclass(frozen=True)
class ReferenceEntry:
    """One documented config key.

    :param str path: Documented path, with ``[]`` after list keys, ``[type=...]`` after
        a discriminated key whose variant it belongs to, and ``<key>`` for user-chosen map keys.
    :param str signature: The ``type = default`` or ``type (required)`` text after the key.
    :param tuple[str, ...] comments: The comment lines above the entry, without ``#``.
    :param str section: Title of the reference section the entry sits in.
    """

    path: str
    signature: str
    comments: tuple[str, ...]
    section: str


def plain_path(path: str) -> str:
    """Normalize a key as a user types it to the dotted form lookups compare.

    Bracket groups (``[]``, ``[0]``, ``[type=json]``) and all-digit segments are
    dropped, so ``phases[].sampler``, ``phases.sampler`` and ``phases.0.sampler``
    are one key.

    :param str path: A documented path or a user-typed key.
    :return str: Dotted path without list markers, variant tags or indexes.
    """
    stripped = _BRACKETS.sub("", path.strip())
    return ".".join(part for part in stripped.split(".") if part and not part.isdigit())


def _segments(path: str) -> tuple[str, ...]:
    """Split a key into its plain segments.

    :param str path: A documented path or a user-typed key.
    :return tuple[str, ...]: The segments of :func:`plain_path`.
    """
    plain = plain_path(path)
    return tuple(plain.split(".")) if plain else ()


def _display_path(path: str) -> str:
    """Drop variant tags from a documented path and keep its list markers.

    :param str path: A documented path such as ``phases[].gates[][type=sha256].path``.
    :return str: The path as a user types it, such as ``phases[].gates[].path``.
    """
    return _TAG.sub("", path)


def _parse(text: str) -> tuple[ReferenceEntry, ...]:
    """Parse reference text into entries, in file order.

    :param str text: Contents of ``config_reference.yaml``.
    :raises ValueError: If a line is neither blank, a comment nor an entry, or a path repeats.
    :return tuple[ReferenceEntry, ...]: One entry per documented key and variant.
    """
    entries: list[ReferenceEntry] = []
    seen: set[str] = set()
    stack: list[tuple[int, str]] = []
    comments: list[str] = []
    section = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            comments.clear()
            continue
        if stripped.startswith("#"):
            title = _SECTION.fullmatch(stripped)
            if title is not None:
                section = title[1]
                comments.clear()
            else:
                comments.append(stripped.removeprefix("#").strip())
            continue
        match = _ENTRY.fullmatch(line)
        if match is None:
            raise ValueError(f"Unrecognized reference entry: {line}")
        indent = len(match[1]) + (2 if match[2] else 0)
        key, signature = match[3], match[4]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack and stack[-1][1] == "search_space":
            key = _MAP_KEY
        tag = _VARIANT_SIGNATURE.fullmatch(signature) if key == "type" else None
        if tag is not None:
            parent_indent, parent = stack[-1]
            stack[-1] = (parent_indent, parent.split("[type=", 1)[0] + f"[type={tag[1]}]")
        path = ".".join([*(part for _, part in stack), key])
        if path in seen:
            raise ValueError(f"Duplicate documented path: {path}")
        seen.add(path)
        entries.append(ReferenceEntry(path, signature, tuple(comments), section))
        stack.append((indent, key + ("[]" if signature.startswith("list[") else "")))
        comments.clear()
    return tuple(entries)


def _reference_resource() -> Traversable:
    """Locate the packaged reference file.

    :return Traversable: The resource ``phasesweep/config_reference.yaml``.
    """
    return importlib.resources.files("phasesweep").joinpath("config_reference.yaml")


def _parse_notes(text: str) -> tuple[str, ...]:
    """Read the second comment block of the reference header as general config notes.

    :param str text: Contents of ``config_reference.yaml``.
    :raises ValueError: If the header holds fewer than two comment blocks.
    :return tuple[str, ...]: The notes, one comment line each, without ``#``.
    """
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            blocks.append([])
        elif stripped.startswith("#") and _SECTION.fullmatch(stripped) is None:
            blocks[-1].append(stripped.removeprefix("#").strip())
        else:
            break
    header = [block for block in blocks if block]
    if len(header) < 2:
        raise ValueError("The reference header has no general notes block.")
    return tuple(header[1])


@functools.cache
def reference_notes() -> tuple[str, ...]:
    """Return the packaged reference's general notes on paths, YAML typing and environment.

    :raises ValueError: If the packaged reference header has no notes block.
    :return tuple[str, ...]: The notes, one line each, in file order.
    """
    return _parse_notes(_reference_resource().read_text(encoding="utf-8"))


@functools.cache
def reference_entries() -> tuple[ReferenceEntry, ...]:
    """Return every documented key of the packaged reference, in file order.

    :raises ValueError: If the packaged reference does not parse.
    :return tuple[ReferenceEntry, ...]: One entry per documented key and variant.
    """
    return _parse(_reference_resource().read_text(encoding="utf-8"))


def top_level_entries() -> tuple[ReferenceEntry, ...]:
    """Return the entries of the top-level config keys, in file order.

    :return tuple[ReferenceEntry, ...]: One entry per key of the experiment mapping.
    """
    return tuple(entry for entry in reference_entries() if "." not in entry.path)


def sub_key_count(entry: ReferenceEntry) -> int:
    """Count the documented entries below a key, one per variant of a discriminated key.

    :param ReferenceEntry entry: A documented key.
    :return int: Number of entries below it, zero for a leaf.
    """
    return len(lookup(entry.path)) - 1


def first_sentence(entry: ReferenceEntry, limit: int | None = 80) -> str:
    """Return the first sentence of an entry's description, shortened to fit a listing.

    :param ReferenceEntry entry: A documented key.
    :param int | None limit: Longest result in characters; a longer sentence is cut at a word
        boundary and ends in ``...``. None keeps the whole sentence.
    :return str: The first sentence, or an empty string when the entry has no comments.
    """
    if not entry.comments:
        return ""
    head = re.match(r"(.+?[.!?])(?:\s|$)", entry.comments[0])
    sentence = head[1] if head else entry.comments[0]
    if limit is None or len(sentence) <= limit:
        return sentence
    return sentence[: limit - 3].rsplit(" ", 1)[0].rstrip(",;:") + "..."


@functools.cache
def _display_by_segments() -> dict[tuple[str, ...], str]:
    """Map each documented plain path to the display path of its first entry.

    :return dict[tuple[str, ...], str]: Display path keyed by plain segments.
    """
    display: dict[tuple[str, ...], str] = {}
    for entry in reference_entries():
        display.setdefault(_segments(entry.path), _display_path(entry.path))
    return display


@functools.cache
def _variant_tags() -> dict[tuple[str, ...], frozenset[str]]:
    """Collect the discriminator values the reference documents below each key.

    :return dict[tuple[str, ...], frozenset[str]]: Tag names such as ``json_envelope`` or
        ``sha256``, keyed by the plain segments of the key they select a variant of.
    """
    tags: dict[tuple[str, ...], set[str]] = {}
    for entry in reference_entries():
        for marker in _TAG_NAME.finditer(entry.path):
            owner = _segments(entry.path[: marker.start()])
            tags.setdefault(owner, set()).add(marker[1])
    return {owner: frozenset(names) for owner, names in tags.items()}


def _matches(entry_segments: tuple[str, ...], query: tuple[str, ...]) -> bool:
    """Tell whether a query is a segment prefix of an entry, with ``<key>`` as a wildcard.

    :param tuple[str, ...] entry_segments: Plain segments of a documented path.
    :param tuple[str, ...] query: Plain segments the user asked about.
    :return bool: True when the entry is the queried key or sits below it.
    """
    if len(entry_segments) < len(query):
        return False
    pairs = zip(entry_segments, query, strict=False)
    return all(part in (wanted, _MAP_KEY) for part, wanted in pairs)


def lookup(key: str) -> tuple[ReferenceEntry, ...]:
    """Return the entries for a key and everything documented below it.

    A key that several discriminator variants share, such as ``metric.extractor.path``,
    returns one entry per variant, each labelled by its ``[type=...]`` path. A
    ``<key>`` segment in the reference matches a dotted trainer key in the query. An
    empty key matches every entry.

    :param str key: A key as the reference or a user spells it.
    :return tuple[ReferenceEntry, ...]: Matching entries in file order, empty when none match.
    """
    query = _segments(key)
    entries = reference_entries()
    paths = [_segments(entry.path) for entry in entries]
    owners = {path[: path.index(_MAP_KEY)] for path in paths if _MAP_KEY in path}
    for owner in owners:
        if query[: len(owner)] != owner or len(query) <= len(owner):
            continue
        suffixes = [path[len(owner) + 1 :] for path in paths if path[: len(owner)] == owner]
        end = next(
            (
                index
                for index in range(len(owner) + 1, len(query))
                if any(suffix and _matches(suffix, query[index:]) for suffix in suffixes)
            ),
            len(query),
        )
        query = (*owner, _MAP_KEY, *query[end:])
        break
    return tuple(entry for entry, path in zip(entries, paths, strict=True) if _matches(path, query))


def closest(key: str, limit: int = 3) -> tuple[str, ...]:
    """Suggest documented keys for a key that matched nothing.

    Keys that end with the query's segments come first, so a bare ``inherit_env``
    finds ``execution.inherit_env``. Only when none does, misspelled whole paths
    and misspelled final segments are tried.

    :param str key: The unmatched key as the user typed it.
    :param int limit: Maximum number of suggestions.
    :return tuple[str, ...]: Distinct documented keys without variant tags, best first.
    """
    query = _segments(key)
    if not query:
        return ()
    by_segments = _display_by_segments()
    ranked = [
        segments
        for segments in by_segments
        if len(segments) >= len(query) and segments[-len(query) :] == query
    ]
    if not ranked:
        joined = {".".join(segments): segments for segments in by_segments}
        ranked.extend(
            joined[name]
            for name in difflib.get_close_matches(
                ".".join(query), list(joined), n=limit, cutoff=0.6
            )
        )
        leaves: dict[str, list[tuple[str, ...]]] = {}
        for segments in by_segments:
            leaves.setdefault(segments[-1], []).append(segments)
        for leaf in difflib.get_close_matches(query[-1], list(leaves), n=limit, cutoff=0.75):
            ranked.extend(leaves[leaf])
    suggestions: list[str] = []
    for segments in ranked:
        if (display := by_segments[segments]) not in suggestions:
            suggestions.append(display)
    return tuple(suggestions[:limit])


def lookup_or_suggest(key: str) -> tuple[tuple[ReferenceEntry, ...], tuple[str, ...]]:
    """Find entries for a supplied key, suggesting alternatives only on a miss.

    :param str key: User-typed key; an empty normalized path matches nothing.
    :return tuple[tuple[ReferenceEntry, ...], tuple[str, ...]]: Entries and suggestions.
    """
    entries = lookup(key) if plain_path(key) else ()
    return entries, () if entries else closest(key)


def loc_to_path(loc: Sequence[int | str]) -> str | None:
    """Map a validation error location to the deepest documented key it points into.

    List indexes and discriminator tags are dropped, and a segment below a map of
    user-chosen keys becomes ``<key>``. The walk stops at the first segment the
    reference does not document, so an error inside a free-form map points at the map.

    :param Sequence[int | str] loc: Pydantic ``loc`` of one validation error.
    :return str | None: The documented key as a user types it, or None when no prefix is documented.
    """
    by_segments = _display_by_segments()
    tags = _variant_tags()
    prefix: tuple[str, ...] = ()
    tagged_at: tuple[str, ...] | None = None
    for part in loc:
        if isinstance(part, int):
            continue
        if prefix != tagged_at and part in tags.get(prefix, ()):
            tagged_at = prefix
            continue
        if (candidate := (*prefix, part)) in by_segments or (
            candidate := (*prefix, _MAP_KEY)
        ) in by_segments:
            prefix = candidate
        else:
            break
    return by_segments.get(prefix)
