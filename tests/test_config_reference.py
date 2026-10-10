"""Keep the narrative signature reference aligned with the primary config schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import phasesweep.config as config_models
from phasesweep.config import Experiment, load_experiment

REPO = Path(__file__).resolve().parent.parent


def _type_signature(node: dict[str, Any]) -> str:
    """Render schema types in the reference's user-facing notation."""
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

    def children(node: dict[str, Any], path: str) -> None:
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

    def model(node: dict[str, Any], prefix: str) -> None:
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


def _reference_inventory() -> dict[str, str]:
    """Read indented signatures while retaining every discriminator alternative."""
    inventory: dict[str, str] = {}
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
        inventory[path] = signature
        stack.append((indent, key + ("[]" if signature.startswith("list[") else "")))
        comments.clear()
    return inventory


def test_reference_matches_every_config_path_variant_type_and_default() -> None:
    """Fail for missing/extra keys, variant drift, or changed defaults/types."""
    expected = _schema_inventory()
    actual = _reference_inventory()
    assert actual.keys() == expected.keys(), {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
    }
    assert actual == expected


def test_minimal_starting_example_loads_through_real_loader() -> None:
    """Validate the checked-in starting file without launching its trainer."""
    experiment = load_experiment(REPO / "examples/example.yaml")
    assert experiment.experiment == "first_trial"
    assert len(experiment.phases) == 1
    assert experiment.phases[0].n_trials == 1
    assert experiment.storage is None
    assert experiment.mlflow is None
