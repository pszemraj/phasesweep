"""Guard documented config paths/signatures and load the starting configurations.

These checks detect inventory, type and default drift. They do not prove the
reference's behavioral comments, bounds, precedence or cross-field rules. The
one comment check is structural: every ``Conflicts with`` and ``Overrides``
line must have its counterpart on the key it names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import phasesweep.config as config_models
from phasesweep.config import Experiment, load_experiment
from phasesweep.config.io import load_experiment_bytes
from phasesweep.mcp.registry import _Allow, _Catalog, _Entry
from phasesweep.runtime.json import JsonObject

REPO = Path(__file__).resolve().parent.parent


def _type_signature(node: JsonObject) -> str:
    """Render schema types in the reference's user-facing notation.

    :param JsonObject node: JSON-schema node of one field.
    :return str: The type as the reference spells it.
    """
    if "$ref" in node:
        return "object"
    if "const" in node:
        return json.dumps(node["const"])
    if "enum" in node:
        return " | ".join(json.dumps(value) for value in node["enum"])
    if "discriminator" in node:
        return " | ".join(node["discriminator"]["mapping"])
    for union in ("anyOf", "oneOf"):
        if union in node:
            return " | ".join(_type_signature(item) for item in node[union])
    kind = node.get("type")
    if kind == "array":
        return f"list[{_type_signature(node['items'])}]"
    if kind == "object":
        if "properties" in node:
            return "object"
        values = node.get("additionalProperties")
        value_type = _type_signature(values) if isinstance(values, dict) else "any"
        return f"map[string, {value_type}]"
    return {
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "string": "string",
        "null": "null",
    }.get(kind, "any")


def _schema_inventory() -> dict[str, str]:
    """Inventory all model fields and discriminated variants, including map values."""
    schema = Experiment.model_json_schema()
    inventory: dict[str, str] = {}

    def children(node: JsonObject, path: str) -> None:
        """Descend into the model, variants, list items or map values below one field.

        :param JsonObject node: JSON-schema node of the field.
        :param str path: Inventory path of the field.
        """
        if "$ref" in node:
            model(schema["$defs"][node["$ref"].rsplit("/", 1)[1]], path)
        elif "discriminator" in node:
            for reference in node["discriminator"]["mapping"].values():
                children({"$ref": reference}, path)
        elif "anyOf" in node:
            for option in node["anyOf"]:
                if "$ref" in option:
                    children(option, path)
        elif node.get("type") == "array":
            children(node["items"], path + "[]")
        elif isinstance(node.get("additionalProperties"), dict):
            values = node["additionalProperties"]
            if "discriminator" in values:
                inventory[path + ".<key>"] = _type_signature(values) + " (required)"
                children(values, path + ".<key>")

    def model(node: JsonObject, prefix: str) -> None:
        """Record every field of one model schema under a path prefix.

        :param JsonObject node: JSON-schema node of the model.
        :param str prefix: Inventory path of the model, with its variant tag if any.
        """
        name = node["title"]
        model_type = getattr(config_models, name)
        tag = node["properties"].get("type", {}).get("const")
        if tag is not None:
            prefix += f"[type={tag}]"
        for key, value in node["properties"].items():
            field = model_type.model_fields[key]
            if field.is_required():
                suffix = " (required)"
            else:
                default = field.get_default(call_default_factory=True)
                # A default model block is spelled {} with its defaults nested below.
                if hasattr(default, "model_dump"):
                    default = {}
                suffix = " = " + json.dumps(default)
            kind = _type_signature(value)
            if name == "CategoricalParam" and key == "choices":
                # The schema uses Any; its validator narrows the scalar domain.
                kind = "list[null | bool | int | float | string]"
            path = f"{prefix}.{key}" if prefix else key
            inventory[path] = kind + suffix
            children(value, path)

    model(schema, "")
    return inventory


def _reference_entries() -> dict[str, tuple[str, list[str]]]:
    """Read indented signatures and their comment lines, keeping every discriminator alternative."""
    inventory: dict[str, tuple[str, list[str]]] = {}
    stack: list[tuple[int, str]] = []
    comments: list[str] = []
    for line in (REPO / "docs/config_reference.yaml").read_text().splitlines():
        stripped = line.lstrip()
        if not stripped:
            comments.clear()
            continue
        if stripped.startswith("#"):
            comments.append(stripped.removeprefix("#").strip())
            continue
        match = re.fullmatch(r"( *)(- )?([\w.-]+): (.+)", line)
        assert match is not None, f"Unrecognized reference entry: {line}"
        indent = len(match[1]) + (2 if match[2] else 0)
        key, signature = match[3], match[4]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack and stack[-1][1] == "search_space":
            key = "<key>"
        tag = re.fullmatch(r'"([^"\n]+)" \(required\)', signature) if key == "type" else None
        if tag is not None:
            parent_indent, parent = stack[-1]
            stack[-1] = (parent_indent, parent.split("[type=", 1)[0] + f"[type={tag[1]}]")
        path = ".".join([*(part for _, part in stack), key])
        assert path not in inventory, f"Duplicate documented path: {path}"
        assert comments, f"Description missing at {path}"
        assert not comments[-1].startswith("Variant "), f"Description missing at {path}"
        assert re.search(r" = .+| \(required\)$| \(optional\)$", signature), path
        if signature.endswith(" (optional)"):
            assert any(comment.startswith("Unset:") for comment in comments), path
        inventory[path] = (signature, comments.copy())
        stack.append((indent, key + ("[]" if signature.startswith("list[") else "")))
        comments.clear()
    return inventory


def _reference_inventory() -> dict[str, str]:
    """Map each documented path, with its discriminator alternative, to its signature."""
    return {path: signature for path, (signature, _) in _reference_entries().items()}


def test_reference_matches_every_config_path_variant_type_and_default() -> None:
    """Fail for missing/extra keys, variant drift, or changed defaults/types."""
    expected = _schema_inventory()
    actual = _reference_inventory()
    assert actual.keys() == expected.keys(), {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
    }
    assert actual == expected


_TAG = re.compile(r"(\[type=[^\]]+\])")


def _plain_path_and_tags(path: str) -> tuple[str, dict[str, str]]:
    """Split an inventory path into the path a user types and its variant tags.

    :param str path: Inventory path such as ``phases[].gates[type=sha256].path``.
    :return tuple[str, dict[str, str]]: The path without ``[type=...]`` markers,
        and the variant tag chosen at each plain prefix along it.
    """
    plain = ""
    tags: dict[str, str] = {}
    for segment in _TAG.split(path):
        if segment.startswith("[type="):
            tags[plain] = segment[len("[type=") : -1]
        else:
            plain += segment
    return plain, tags


def _labeled_paths(path: str, comments: list[str], label: str, known: set[str]) -> set[str]:
    """Return the key paths named by one entry's lines for a pairing label.

    A labeled line is ``<label>: <path>[, <path>]...`` followed by nothing, a
    parenthesis, or a semicolon and an explanation. Every listed item must be a
    documented key path, so a line that breaks the convention fails loudly
    instead of being skipped.

    :param str path: Inventory path of the entry, for diagnostics.
    :param list[str] comments: Comment lines above the entry.
    :param str label: ``Conflicts with``, ``Overrides`` or ``Overridden by``.
    :param set[str] known: Every documented path as a user types it.
    :return set[str]: Plain paths named by the entry's lines for this label.
    """
    named: set[str] = set()
    for comment in comments:
        if not comment.startswith(f"{label}: "):
            continue
        head = re.split(r" \(|;", comment[len(label) + 2 :], maxsplit=1)[0].rstrip(".")
        for item in (part.strip() for part in head.split(",")):
            assert item in known, f"{path}: '{label}:' names {item!r}, not a documented key path"
            named.add(item)
    return named


def test_conflict_and_override_labels_are_paired_on_both_keys() -> None:
    """Fail when one side of a Conflicts with / Overrides / Overridden by pair is missing.

    Each pairing label lists key paths first, then an optional explanation after
    a parenthesis or semicolon. Where a path exists in several discriminator
    variants, the counterpart is looked up in the variants that agree with the
    labeled entry on every shared variant choice.
    """
    entries = _reference_entries()
    parsed = {path: _plain_path_and_tags(path) for path in entries}
    known = {plain for plain, _ in parsed.values()}
    labels = ("Conflicts with", "Overrides", "Overridden by")
    named = {
        label: {
            path: _labeled_paths(path, comments, label, known)
            for path, (_, comments) in entries.items()
        }
        for label in labels
    }
    assert sum(len(targets) for targets in named["Conflicts with"].values()), "no Conflicts with"
    assert sum(len(targets) for targets in named["Overrides"].values()), "no Overrides"

    def counterparts(owner: str, target: str) -> list[str]:
        """Return entries at ``target`` whose variant tags agree with ``owner``'s.

        :param str owner: Inventory path of the entry that carries the label.
        :param str target: Plain key path the label names.
        :return list[str]: Inventory paths of the candidate counterparts.
        """
        owner_tags = parsed[owner][1]
        return [
            path
            for path, (plain, tags) in parsed.items()
            if plain == target and all(owner_tags.get(k, v) == v for k, v in tags.items())
        ]

    unpaired: list[str] = []
    for label, reverse in (
        ("Conflicts with", "Conflicts with"),
        ("Overrides", "Overridden by"),
        ("Overridden by", "Overrides"),
    ):
        for owner, targets in named[label].items():
            for target in targets:
                if not any(
                    parsed[owner][0] in named[reverse][other]
                    for other in counterparts(owner, target)
                ):
                    unpaired.append(f"{owner}: '{label}: {target}' lacks '{reverse}: ...' there")
    assert not unpaired, unpaired


def test_minimal_starting_example_loads_through_real_loader() -> None:
    """Validate the checked-in starting file without launching its trainer."""
    experiment = load_experiment(REPO / "examples/example.yaml")
    assert experiment.experiment == "first_trial"
    assert len(experiment.phases) == 1
    assert experiment.phases[0].n_trials == 1
    assert experiment.phases[0].gpu_policy == "none"
    assert experiment.phases[0].gates == []
    assert experiment.metric.extractor.type == "json_envelope"
    assert experiment.storage is None
    assert experiment.mlflow is None


def test_configuration_guide_complete_yaml_examples_load_through_real_loader() -> None:
    """Validate complete guide examples while leaving partial illustrations alone."""
    blocks = re.findall(r"```yaml\n(.*?)\n```", (REPO / "docs/config.md").read_text(), re.DOTALL)
    complete = [block for block in blocks if re.search(r"^experiment:", block, re.MULTILINE)]
    assert complete, "The configuration guide must retain a complete experiment example"
    for index, block in enumerate(complete):
        load_experiment_bytes(block.encode("utf-8"), source=f"docs/config.md YAML example {index}")


def test_mcp_catalog_table_matches_all_catalog_paths() -> None:
    """Keep the separate catalog field table complete in both directions."""
    expected = {
        *_Catalog.model_fields,
        *(f"experiments[].{key}" for key in _Entry.model_fields),
        *(f"experiments[].allow.{key}" for key in _Allow.model_fields),
    }
    catalog_section = (REPO / "docs/mcp.md").read_text().split("## The catalog\n", 1)[1]
    catalog_section = catalog_section.split("\n## ", 1)[0]
    paths = re.findall(r"^\| `([^`]+)` \|", catalog_section, re.MULTILINE)
    assert len(paths) == len(set(paths)), "Duplicate catalog table paths"
    actual = set(paths)
    assert actual == expected, {
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
    }
