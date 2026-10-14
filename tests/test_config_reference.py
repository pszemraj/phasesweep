"""Guard documented config paths/signatures and load the starting configurations.

These checks detect inventory, type and default drift. They do not prove the
reference's behavioral comments, bounds, precedence or cross-field rules. The
one comment check is structural: every ``Conflicts with`` and ``Overrides``
line must have its counterpart on the key it names.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import phasesweep.config as config_models
from phasesweep.config import Experiment, load_experiment
from phasesweep.config.io import load_experiment_bytes
from phasesweep.config.reference import (
    ReferenceEntry,
    _parse,
    closest,
    loc_to_path,
    lookup,
    plain_path,
    reference_entries,
)
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
    """Read the packaged reference and check every entry's description and signature shape.

    :return dict[str, tuple[str, list[str]]]: Signature and comment lines per documented path,
        keeping every discriminator alternative.
    """
    inventory: dict[str, tuple[str, list[str]]] = {}
    for entry in reference_entries():
        comments = list(entry.comments)
        assert comments, f"Description missing at {entry.path}"
        assert not comments[-1].startswith("Variant "), f"Description missing at {entry.path}"
        assert re.search(r" = .+| \(required\)$| \(optional\)$", entry.signature), entry.path
        if entry.signature.endswith(" (optional)"):
            assert any(comment.startswith("Unset:") for comment in comments), entry.path
        inventory[entry.path] = (entry.signature, comments)
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


def test_reference_entries_record_their_section_and_are_immutable() -> None:
    """Every entry names the section marker above it, and entries cannot be edited."""
    entries = reference_entries()
    assert entries[0].path == "experiment"
    assert entries[0].section == "Required experiment fields"
    assert {entry.section for entry in entries} == {
        "Required experiment fields",
        "Trainer inputs",
        "Storage and provenance",
        "Constraints",
        "Observations",
        "MLflow target and ownership",
        "Execution environment",
    }
    by_path = {entry.path: entry for entry in entries}
    assert by_path["execution.inherit_env"].section == "Execution environment"
    assert isinstance(by_path["trial_command"].comments, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entries[0].path = "other"


def test_reference_parser_rejects_unrecognized_lines_and_duplicate_paths() -> None:
    """The parser fails loudly instead of skipping a line it cannot place."""
    with pytest.raises(ValueError, match="Unrecognized reference entry"):
        _parse("# about\nnot an entry\n")
    with pytest.raises(ValueError, match="Duplicate documented path: name"):
        _parse("# one\nname: string (required)\n\n# two\nname: string (required)\n")
    parsed = _parse("# ---------- First ----------\n\n# one\nname: string (required)\n")
    assert parsed == (ReferenceEntry("name", "string (required)", ("one",), "First"),)


@pytest.mark.parametrize(
    "typed",
    [
        "phases[].sampler.type",
        "phases.sampler.type",
        "phases.0.sampler.type",
        "phases[0].sampler.type",
        "  phases[].sampler.type  ",
    ],
)
def test_plain_path_accepts_every_spelling_of_one_key(typed: str) -> None:
    """Brackets, indexes and variant tags never change which key is meant."""
    assert plain_path(typed) == "phases.sampler.type"


def test_plain_path_drops_variant_tags_and_list_markers() -> None:
    """A documented inventory path reduces to the dotted form lookups compare."""
    assert plain_path("phases[].gates[][type=sha256].path") == "phases.gates.path"
    assert plain_path("phases[].search_space.<key>[type=float].low") == (
        "phases.search_space.<key>.low"
    )


def test_lookup_returns_the_entry_and_everything_below_it() -> None:
    """A key matches itself and its descendants, in every spelling."""
    sampler = [entry.path for entry in lookup("phases[].sampler")]
    assert sampler[0] == "phases[].sampler"
    assert "phases[].sampler.type" in sampler
    assert all(path.startswith("phases[].sampler") for path in sampler)
    assert [entry.path for entry in lookup("phases.0.sampler")] == sampler
    assert [entry.path for entry in lookup("phases.sampler")] == sampler
    assert [entry.path for entry in lookup("inherit_env")] == []
    exact = lookup("execution.inherit_env")
    assert [entry.path for entry in exact] == ["execution.inherit_env"]
    assert exact[0].signature.startswith('"all" | "none" | list[string]')


def test_lookup_labels_each_variant_of_a_shared_key() -> None:
    """A key several discriminator variants share returns one labelled entry per variant."""
    paths = [entry.path for entry in lookup("metric.extractor.path")]
    assert "metric.extractor[type=json].path" in paths
    assert "metric.extractor[type=json_envelope].path" in paths
    assert len(paths) == len(set(paths)) >= 2
    low = [entry.path for entry in lookup("phases.search_space.lr.low")]
    assert low == [
        "phases[].search_space.<key>[type=float].low",
        "phases[].search_space.<key>[type=int].low",
    ]


def test_lookup_of_an_unknown_key_is_empty_and_an_empty_key_is_everything() -> None:
    """Misses return nothing so callers can fall back to suggestions."""
    assert lookup("phases.sampler.typ") == ()
    assert lookup("") == reference_entries()


def test_closest_suggests_the_documented_key_for_a_partial_or_misspelled_one() -> None:
    """A bare leaf finds its parent path, and a typo finds the key it misspells."""
    assert closest("inherit_env") == ("execution.inherit_env",)
    assert closest("sampler.type") == ("phases[].sampler.type",)
    assert closest("phases.sampler.typ")[0] == "phases[].sampler.type"
    assert closest("trial_comand")[0] == "trial_command"
    assert closest("zzzzzz") == ()
    assert closest("") == ()
    assert len(closest("type")) == 3
    assert all("[type=" not in suggestion for suggestion in closest("type"))


def _error_locs(change: Callable[[dict], None]) -> list[tuple[int | str, ...]]:
    """Validate the minimal starting example after a change and return its error locations.

    :param Callable[[dict], None] change: Edit applied to the loaded example mapping.
    :return list[tuple[int | str, ...]]: The ``loc`` of every validation error.
    """
    data = yaml.safe_load((REPO / "examples/example.yaml").read_text())
    change(data)
    with pytest.raises(ValidationError) as caught:
        Experiment.model_validate(data)
    return [error["loc"] for error in caught.value.errors(include_input=False)]


def test_loc_to_path_maps_variant_and_search_key_segments() -> None:
    """Indexes and variant tags drop out, and a user-chosen search key maps to ``<key>``."""

    def bad_sampler(data: dict) -> None:
        data["phases"][0]["sampler"] = {"type": "bogus"}

    def bad_extractor_path(data: dict) -> None:
        data["metric"]["extractor"]["path"] = 3

    def bad_search_bound(data: dict) -> None:
        data["phases"][0]["search_space"] = {"lr": {"type": "float", "low": "x", "high": 1}}

    def bad_gate_digest(data: dict) -> None:
        data["phases"][0]["gates"] = [{"type": "sha256", "path": "a"}]

    def bad_inherit_env(data: dict) -> None:
        data["execution"] = {"inherit_env": 3}

    def bad_env_value(data: dict) -> None:
        data["env"] = {"A": 1}

    cases = [
        (bad_sampler, "phases[].sampler.type"),
        (bad_extractor_path, "metric.extractor.path"),
        (bad_search_bound, "phases[].search_space.<key>.low"),
        (bad_gate_digest, "phases[].gates[].sha256"),
        (bad_inherit_env, "execution.inherit_env"),
        (bad_env_value, "env"),
    ]
    for change, expected in cases:
        locs = _error_locs(change)
        assert {loc_to_path(loc) for loc in locs} == {expected}, locs


def test_loc_to_path_has_no_key_for_unknown_or_model_level_locations() -> None:
    """An empty location or an undocumented top-level key yields no hint."""
    assert loc_to_path(()) is None
    assert loc_to_path(("bogus",)) is None
    assert loc_to_path(("trainer_config", "a", "b")) == "trainer_config"
    assert loc_to_path(("phases", 2, "n_trials")) == "phases[].n_trials"
