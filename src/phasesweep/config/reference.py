"""Query the packaged configuration reference without importing tracker SDKs.

Parse config_reference.yaml into entries for CLI help, typo suggestions,
validation-error hints, and header notes. Shared blocks, unions of blocks and
``<<:`` merges are expanded at parse time, so every concrete key path is one
entry, and the section rules and shared-block descriptions that govern an entry
are kept for ``explain``.
"""

from __future__ import annotations

import difflib
import functools
import importlib.resources
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources.abc import Traversable
from typing import Literal

from phasesweep._metadata import __version__

_SECTION = re.compile(r"# -{10} (.+?) -{10}")
_ENTRY = re.compile(r"( *)(- )?([\w.-]+): (.+)")
_BLOCK = re.compile(r"([A-Z][A-Za-z0-9]*):")
_BLOCK_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")
_MERGE = re.compile(r"( *)<<: (\S+)")
_QUOTED = re.compile(r'"[^"\n]*"')
_TYPE_PART = re.compile(r"(.+?)(?: = .*| \(required\))")
_VARIANT_SIGNATURE = re.compile(r'"([^"\n]+)" \(required\)')
_BRACKETS = re.compile(r"\[[^\]]*\]")
_TAG = re.compile(r"\[type=[^\]]*\]")
_TAG_NAME = re.compile(r"\[type=([^\]]*)\]")
_MAP_KEY = "<key>"
_DOC_URL = re.compile(r"https://github\.com/pszemraj/phasesweep/blob/[^/]+/")


@dataclass(frozen=True)
class ReferenceEntry:
    """One documented config key.

    :param str path: Documented path, with ``[]`` after list keys, ``[type=...]`` after
        a discriminated key whose variant it belongs to, and ``<key>`` for user-chosen map keys.
    :param str signature: The ``type = default`` or ``type (required)`` text after the key.
    :param tuple[str, ...] comments: The comment lines above the entry, without ``#``.
    :param str section: Title of the reference section the entry sits in; an entry expanded
        from a shared block takes the section of the key that uses the block.
    :param tuple[str, ...] blocks: Shared blocks the entry was expanded from, outermost first;
        empty for a key written out at its own path.
    """

    path: str
    signature: str
    comments: tuple[str, ...]
    section: str
    blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceRules:
    """Comment lines that govern a group of entries rather than one key.

    :param Literal["section", "block"] kind: ``section`` for the rule block at the head of a
        reference section, ``block`` for the opening comment of a shared block.
    :param str name: Section title or shared-block name.
    :param tuple[str, ...] lines: The comment lines, without ``#``.
    """

    kind: Literal["section", "block"]
    name: str
    lines: tuple[str, ...]

    @property
    def title(self) -> str:
        """Name the group as ``explain`` heads it.

        :return str: ``Section <title>`` or ``Shared block <name>``.
        """
        return f"Section {self.name}" if self.kind == "section" else f"Shared block {self.name}"


@dataclass(frozen=True)
class _Block:
    """One shared block definition, before expansion.

    :param tuple[str, ...] opening: Comment lines directly above the block name, without ``#``.
    :param tuple[str, ...] body: The block's lines, dedented so its fields start at column 0.
    """

    opening: tuple[str, ...]
    body: tuple[str, ...]

    @property
    def is_variant(self) -> bool:
        """Tell whether the block is a discriminator variant, whose first field is ``type``.

        :return bool: True when the first key line of the body is ``type: "<tag>" (required)``.
        """
        first = next((line for line in self.body if line and not line.startswith("#")), "")
        match = _ENTRY.fullmatch(first)
        return (
            match is not None
            and match[3] == "type"
            and bool(_VARIANT_SIGNATURE.fullmatch(match[4]))
        )


@dataclass(frozen=True)
class _Reference:
    """A parsed reference: expanded entries plus the rule lines that govern them.

    :param tuple[ReferenceEntry, ...] entries: One entry per documented key and variant.
    :param dict[str, tuple[str, ...]] section_rules: Rule lines at the head of each section.
    :param dict[str, tuple[str, ...]] block_rules: Opening comment of each shared block.
    """

    entries: tuple[ReferenceEntry, ...]
    section_rules: dict[str, tuple[str, ...]]
    block_rules: dict[str, tuple[str, ...]]

    def rules_for(self, entries: Iterable[ReferenceEntry]) -> tuple[ReferenceRules, ...]:
        """Collect the section rules and shared-block descriptions that govern some entries.

        :param Iterable[ReferenceEntry] entries: Entries of this reference, in output order.
        :return tuple[ReferenceRules, ...]: Each group with lines once, in order of first use:
            an entry's section first, then its shared blocks from the outermost.
        """
        groups: list[ReferenceRules] = []
        seen: set[tuple[str, str]] = set()
        for entry in entries:
            candidates: list[ReferenceRules] = [
                ReferenceRules("section", entry.section, self.section_rules.get(entry.section, ()))
            ]
            candidates += [
                ReferenceRules("block", name, self.block_rules.get(name, ()))
                for name in entry.blocks
            ]
            for group in candidates:
                if group.lines and (group.kind, group.name) not in seen:
                    seen.add((group.kind, group.name))
                    groups.append(group)
        return tuple(groups)


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


def _indent(line: str) -> int:
    """Count the leading spaces of a line.

    :param str line: One reference line.
    :return int: Number of leading spaces.
    """
    return len(line) - len(line.lstrip(" "))


def _split_blocks(lines: list[str]) -> tuple[list[str], dict[str, _Block]]:
    """Take the shared-block definitions out of the reference lines.

    A definition is a column-0 line ``Name:`` with a TitleCase name and nothing after
    the colon. The comment lines directly above it are its opening comment, and the
    indented lines below it, up to the next non-blank column-0 line, are its body.

    :param list[str] lines: Reference lines.
    :raises ValueError: If a block name repeats or a block has no fields.
    :return tuple[list[str], dict[str, _Block]]: The remaining lines, and each block by name.
    """
    rest: list[str] = []
    blocks: dict[str, _Block] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        head = _BLOCK.fullmatch(line)
        if head is None:
            rest.append(line)
            index += 1
            continue
        name = head[1]
        if name in blocks:
            raise ValueError(f"Duplicate shared block: {name}")
        opening: list[str] = []
        while rest and rest[-1].startswith("#") and _SECTION.fullmatch(rest[-1]) is None:
            opening.insert(0, rest.pop().removeprefix("#").strip())
        index += 1
        body: list[str] = []
        while index < len(lines) and (not lines[index].strip() or lines[index].startswith(" ")):
            body.append(lines[index])
            index += 1
        while body and not body[-1].strip():
            rest.append(body.pop())
        if not body:
            raise ValueError(f"Shared block {name} has no fields")
        depth = _indent(next(line for line in body if line.strip()))
        blocks[name] = _Block(
            tuple(opening), tuple(line[depth:] if line.strip() else "" for line in body)
        )
    return rest, blocks


def _used_blocks(signature: str, blocks: dict[str, _Block]) -> list[str]:
    """Return the shared blocks a signature names as its type alternatives.

    :param str signature: Signature text after the key.
    :param dict[str, _Block] blocks: Shared blocks by name.
    :raises ValueError: If a TitleCase type names no block, a block name appears other than as
        a whole alternative, or a union names a block that is not a discriminator variant.
    :return list[str]: Block names in signature order, empty when the type names none.
    """
    head = _TYPE_PART.fullmatch(signature)
    type_part = head[1] if head else signature
    alternatives = [part.strip() for part in type_part.split(" | ")]
    names = [part for part in alternatives if part in blocks]
    tokens = set(_BLOCK_TOKEN.findall(_QUOTED.sub("", type_part)))
    if unknown := sorted(tokens - blocks.keys()):
        raise ValueError(f"Unknown shared block {', '.join(unknown)}: {signature}")
    if tokens - set(names):
        raise ValueError(f"Shared block used inside another type: {signature}")
    if len(names) > 1 and not all(blocks[name].is_variant for name in names):
        raise ValueError(f"A union of shared blocks needs a type field in each: {signature}")
    return names


def _expand(
    lines: Sequence[str],
    blocks: dict[str, _Block],
    chain: tuple[str, ...],
    used: set[str],
) -> list[tuple[str, tuple[str, ...]]]:
    """Inline shared blocks at their use sites.

    A key whose signature names blocks gets each block's fields as its children, and a
    ``<<: Name`` line becomes the block's fields at its own indent.

    :param Sequence[str] lines: Reference lines without block definitions.
    :param dict[str, _Block] blocks: Shared blocks by name.
    :param tuple[str, ...] chain: Blocks being expanded around these lines, outermost first.
    :param set[str] used: Collects every block name that is expanded.
    :raises ValueError: On an unknown or recursive block, a comment directly above a merge,
        or keys written below a key whose signature names a block.
    :return list[tuple[str, tuple[str, ...]]]: Each expanded line with the blocks it came from.
    """
    out: list[tuple[str, tuple[str, ...]]] = []
    closed_at: int | None = None
    for line in lines:
        if line.strip() and closed_at is not None:
            if _indent(line) > closed_at:
                raise ValueError(f"Keys below a shared-block use site: {line.strip()}")
            closed_at = None
        merge = _MERGE.fullmatch(line)
        match = _ENTRY.fullmatch(line)
        if merge is not None:
            names = [merge[2]]
            if out and out[-1][0].lstrip().startswith("#"):
                raise ValueError(f"Comment above a merge, which has no entry: {line.strip()}")
            depth = len(merge[1])
        elif match is not None and (names := _used_blocks(match[4], blocks)):
            out.append((line, chain))
            closed_at = len(match[1]) + (2 if match[2] else 0)
            depth = closed_at + 2
        else:
            out.append((line, chain))
            continue
        for name in names:
            if name not in blocks:
                raise ValueError(f"Unknown shared block: {name}")
            if name in chain:
                raise ValueError(f"Shared block {name} includes itself")
        for name in names:
            used.add(name)
            body = [" " * depth + text if text else "" for text in blocks[name].body]
            out.extend(_expand(body, blocks, (*chain, name), used))
    return out


def _parse(text: str) -> _Reference:
    """Parse reference text into expanded entries and the rule lines that govern them.

    A comment block that directly follows a section banner and ends at a blank line is
    that section's rules. Comment blocks before the first banner are the header and are
    read by :func:`_parse_notes` instead.

    :param str text: Contents of ``config_reference.yaml``.
    :raises ValueError: If a line is neither blank, a comment nor an entry, a path repeats,
        a comment block belongs to neither a key nor a section head, or a shared block is
        unknown, unused or malformed.
    :return _Reference: Entries in file order, with each block's fields at its use sites.
    """
    rest, blocks = _split_blocks(text.splitlines())
    used: set[str] = set()
    lines = _expand(rest, blocks, (), used)
    if unused := sorted(blocks.keys() - used):
        raise ValueError(f"Shared blocks never used: {', '.join(unused)}")
    entries: list[ReferenceEntry] = []
    section_rules: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    stack: list[tuple[int, str]] = []
    comments: list[str] = []
    section = ""
    keyed = False
    for line, chain in lines:
        stripped = line.lstrip()
        if not stripped:
            if comments and section and not keyed:
                section_rules[section] = (*section_rules.get(section, ()), *comments)
            elif comments and keyed:
                raise ValueError(f"Comment block belongs to no key or section head: {comments[0]}")
            comments.clear()
            continue
        if stripped.startswith("#"):
            title = _SECTION.fullmatch(stripped)
            if title is not None:
                if comments and (keyed or section):
                    raise ValueError(f"Comment block belongs to no key: {comments[0]}")
                section = title[1]
                keyed = False
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
        entries.append(ReferenceEntry(path, signature, tuple(comments), section, chain))
        stack.append((indent, key + ("[]" if signature.startswith("list[") else "")))
        comments.clear()
        keyed = True
    if comments:
        raise ValueError(f"Comment block belongs to no key: {comments[0]}")
    if stray := sorted(section_rules.keys() - {entry.section for entry in entries}):
        raise ValueError(f"Section rules with no keys below them: {', '.join(stray)}")
    block_rules = {name: block.opening for name, block in blocks.items()}
    return _Reference(tuple(entries), section_rules, block_rules)


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
    """Return the packaged reference's general notes on the config file, paths and YAML typing.

    :raises ValueError: If the packaged reference header has no notes block.
    :return tuple[str, ...]: The notes, one line each, in file order.
    """
    return _parse_notes(_reference_text())


def _editable_install() -> bool:
    """Return whether installed metadata identifies an editable checkout.

    :return bool: Whether the package was installed from an editable directory.
    """
    try:
        direct_url = distribution("phasesweep").read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    if direct_url is None:
        return False
    metadata = json.loads(direct_url)
    if not isinstance(metadata, dict):
        return False
    directory = metadata.get("dir_info")
    return isinstance(directory, dict) and directory.get("editable") is True


def pin_documentation_links(text: str) -> str:
    """Pin release documentation; use main for editable and development installs.

    :param str text: Packaged text containing project documentation URLs.
    :return str: Text with release-tag URLs, or main when no published revision is known.
    """
    released = re.fullmatch(r"\d+\.\d+\.\d+", __version__) is not None
    ref = f"v{__version__}" if released and not _editable_install() else "main"
    return _DOC_URL.sub(
        f"https://github.com/pszemraj/phasesweep/blob/{ref}/",
        text,
    )


def _reference_text() -> str:
    """Read the reference with documentation links selected for this installation.

    :return str: Reference text with release-tag or main documentation URLs.
    """
    return pin_documentation_links(_reference_resource().read_text(encoding="utf-8"))


@functools.cache
def _packaged() -> _Reference:
    """Parse the packaged reference once.

    :raises ValueError: If the packaged reference does not parse.
    :return _Reference: The packaged reference with its shared blocks expanded.
    """
    return _parse(_reference_text())


def reference_entries() -> tuple[ReferenceEntry, ...]:
    """Return every documented key of the packaged reference, in file order.

    Shared blocks are expanded, so a field documented once in a block has one entry at
    each path that uses the block.

    :raises ValueError: If the packaged reference does not parse.
    :return tuple[ReferenceEntry, ...]: One entry per documented key and variant.
    """
    return _packaged().entries


def rules_for(entries: Iterable[ReferenceEntry]) -> tuple[ReferenceRules, ...]:
    """Return the section rules and shared-block descriptions that govern packaged entries.

    :param Iterable[ReferenceEntry] entries: Entries of the packaged reference, in output order.
    :return tuple[ReferenceRules, ...]: Each group with lines once, in order of first use.
    """
    return _packaged().rules_for(entries)


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
