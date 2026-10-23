"""Guard documented config paths/signatures, the reference parser and the starting configurations.

Shared blocks are expanded before the comparison with the schema, so every key
path a block covers is checked at each place that uses the block.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import re

import pytest
import yaml
from pydantic import ValidationError

import phasesweep.config.models as config_models
import phasesweep.config.search as search_models
import phasesweep.evidence.models as evidence_models
import phasesweep.tracking.models as tracking_models
from phasesweep.config.io import describe_validation_error, load_experiment, load_experiment_bytes
from phasesweep.config.models import Experiment
from phasesweep.config.reference import (
    ReferenceEntry,
    _parse,
    _parse_notes,
    _reference_resource,
    closest,
    first_sentence,
    loc_to_path,
    lookup,
    plain_path,
    reference_entries,
    reference_notes,
    sub_key_count,
    top_level_entries,
)
from phasesweep.mcp.registry import _Allow, _Catalog, _Entry
from phasesweep.runtime.json import JsonObject
from tests.conftest import REPO


def _type_signature(node: JsonObject) -> str:
    """Render schema types in the reference's user-facing notation.

    :param JsonObject node: JSON-schema node of one field.
    :return str: The type as the reference spells it.
    """
    if "$ref" in node:
        return "any" if node["$ref"].endswith("/JsonValue") else "object"
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
        model_type = next(
            getattr(module, name)
            for module in (config_models, search_models, evidence_models, tracking_models)
            if hasattr(module, name)
        )
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


_BLOCK_NAME = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")
_VARIANT_TYPE = re.compile(r"\[type=([^\]]+)\]\.type$")


def _reference_inventory() -> dict[str, str]:
    """Map each documented path, with its discriminator alternative, to its signature.

    A signature that names shared blocks, such as ``JsonExtractor | WandbExtractor``, is
    rewritten to the variant tags those blocks declare in their ``type`` field, which is how
    the schema spells the union.

    :return dict[str, str]: Signature per documented path, every shared block expanded.
    """
    entries = reference_entries()
    tags = {
        entry.blocks[-1]: match[1]
        for entry in entries
        if entry.blocks and (match := _VARIANT_TYPE.search(entry.path))
    }
    inventory: dict[str, str] = {}
    for entry in entries:
        assert entry.comments, f"Description missing at {entry.path}"
        assert re.search(r" = .+| \(required\)$", entry.signature), entry.path
        inventory[entry.path] = _BLOCK_NAME.sub(lambda name: tags[name[0]], entry.signature)
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


def test_config_guide_index_lists_every_top_level_key() -> None:
    """The guide's reference index names each top-level key under the section it sits in."""
    guide = (REPO / "docs/config.md").read_text()
    rows = re.findall(
        r"^\| ([^|`]+?) \| ((?:`[^`]+`(?:, )?)+) \| `phasesweep explain ([^`]+)` \|$",
        guide,
        re.MULTILINE,
    )
    documented = {section: re.findall(r"`([^`]+)`", keys) for section, keys, _ in rows}
    assert len(documented) == len(rows), "Duplicate index section"
    expected: dict[str, list[str]] = {}
    for entry in top_level_entries():
        expected.setdefault(entry.section, []).append(entry.path)
    assert documented == expected
    for section, _, command_key in rows:
        assert command_key in documented[section], (section, command_key)


def test_top_level_entries_are_the_undotted_paths_in_file_order() -> None:
    """Top-level keys are exactly the entries whose path has no dot."""
    paths = [entry.path for entry in top_level_entries()]
    assert paths[:4] == ["experiment", "trial_command", "metric", "phases"]
    assert paths == [entry.path for entry in reference_entries() if "." not in entry.path]


def test_sub_key_count_counts_entries_below_a_key() -> None:
    """A leaf has none, and a container counts every entry below it."""
    by_path = {entry.path: entry for entry in reference_entries()}
    assert sub_key_count(by_path["storage"]) == 0
    assert sub_key_count(by_path["execution"]) == len(
        [path for path in by_path if path.startswith("execution.")]
    )
    assert sub_key_count(by_path["phases"]) > 40


def test_first_sentence_stops_at_the_first_full_stop_and_shortens_long_ones() -> None:
    """A listing row shows one sentence, cut at a word boundary when it is long."""
    storage = next(entry for entry in reference_entries() if entry.path == "storage")
    sentence = first_sentence(storage, None)
    assert sentence
    assert storage.comments[0].startswith(sentence)
    long = ReferenceEntry("k", "string", ("Word " * 40 + "end. Second.",), "S")
    assert first_sentence(long, 30) == "Word Word Word Word Word..."
    assert first_sentence(long, None) == "Word " * 40 + "end."
    assert first_sentence(ReferenceEntry("k", "string", (), "S")) == ""


def test_mcp_catalog_table_matches_all_catalog_paths() -> None:
    """Keep the separate catalog field table complete in both directions."""
    expected = {
        *_Catalog.model_fields,
        *(f"experiments[].{key}" for key in _Entry.model_fields),
        *(f"experiments[].allow.{key}" for key in _Allow.model_fields),
    }
    # The catalog is the only three-column table whose first column is a code-spelled key.
    paths = re.findall(
        r"^\| `([^`]+)` \| [^|]+ \| [^|]+ \|$",
        (REPO / "docs/mcp.md").read_text(),
        re.MULTILINE,
    )
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
    markers = re.findall(
        r"^ *# -{10} (.+?) -{10}$",
        _reference_resource().read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    # Each marker, nested ones included, heads one unbroken run of entries, in file order.
    # The last marker holds the shared blocks, whose fields take the section of each use site.
    assert markers[-1] == "Shared blocks"
    assert [section for section, _ in itertools.groupby(e.section for e in entries)] == markers[:-1]
    assert len(set(markers)) == len(markers) > 1
    wandb_entity = "metric.extractor[type=wandb].entity"
    assert {e.path: e.section for e in entries}[wandb_entity] == "Objective"
    by_path = {entry.path: entry for entry in entries}
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
    assert parsed.entries == (ReferenceEntry("name", "string (required)", ("one",), "First"),)


def test_reference_notes_are_the_second_header_block_and_never_reach_an_entry() -> None:
    """The first header block is the title, the block after its blank line is the notes."""
    text = (
        "# about this file\n# parsed by x\n\n# note one\n# note two\n\n"
        "# ---------- First ----------\n\n# one\nname: string (required)\n"
    )
    assert _parse_notes(text) == ("note one", "note two")
    assert _parse(text).entries == (ReferenceEntry("name", "string (required)", ("one",), "First"),)
    with pytest.raises(ValueError, match="no general notes"):
        _parse_notes("# about this file\n\n# ---------- First ----------\n\n# one\nname: string\n")


_SAMPLE = """\
# sample reference

# note one

# ---------- Objective ----------
# A bad read fails the trial.

# Scalar source.
extractor: JsonX | WandbX (required)

# Upper bound.
max: float | null = null

# ---------- Gates ----------

# Hard checks.
gates: list[sha | wb] = []
  # Hash gate.
  - type: "sha" (required)
    # File to hash.
    path: string (required)
  # W&B gate.
  - type: "wb" (required)
    <<: Target
    # Keys that must be present.
    keys: list[string] (required)

# ---------- Shared blocks ----------

# Read one number from a JSON file.
JsonX:
  # Selects the JSON reader.
  type: "json" (required)

  # JSON file.
  path: string (required)

# Read a W&B summary.
WandbX:
  # Selects the W&B reader.
  type: "wandb" (required)
  <<: Target
  # Summary key.
  metric_key: string (required)

# The W&B run every consumer in a phase reads; all of them agree on these fields.
Target:
  # W&B entity.
  entity: string (required)
  # Seconds to wait.
  timeout_seconds: float = 120.0
"""


def test_parser_expands_a_union_of_shared_blocks_into_variant_paths() -> None:
    """A use site naming blocks gets each block's fields below it, tagged by the block's type."""
    entries = {entry.path: entry for entry in _parse(_SAMPLE).entries}
    assert entries["extractor"] == ReferenceEntry(
        "extractor", "JsonX | WandbX (required)", ("Scalar source.",), "Objective"
    )
    assert entries["extractor[type=json].path"] == ReferenceEntry(
        "extractor[type=json].path", "string (required)", ("JSON file.",), "Objective", ("JsonX",)
    )
    assert entries["extractor[type=wandb].type"].comments == ("Selects the W&B reader.",)
    assert entries["max"].section == "Objective"
    assert entries["max"].blocks == ()


def test_parser_expands_merges_at_every_use_site() -> None:
    """A ``<<:`` merge inlines the block's fields, inside another block and inside a list variant."""
    entries = {entry.path: entry for entry in _parse(_SAMPLE).entries}
    nested = entries["extractor[type=wandb].timeout_seconds"]
    assert nested.signature == "float = 120.0"
    assert nested.comments == ("Seconds to wait.",)
    assert nested.blocks == ("WandbX", "Target")
    gate = entries["gates[][type=wb].entity"]
    assert gate.section == "Gates"
    assert gate.blocks == ("Target",)
    assert list(entries)[-3:] == [
        "gates[][type=wb].entity",
        "gates[][type=wb].timeout_seconds",
        "gates[][type=wb].keys",
    ]
    assert not any(path.startswith(("JsonX", "WandbX", "Target")) for path in entries)


def test_parser_keeps_section_rules_and_block_descriptions_in_order_of_use() -> None:
    """Rules come once per group: the entry's section, then its blocks from the outermost."""
    reference = _parse(_SAMPLE)
    assert reference.section_rules == {"Objective": ("A bad read fails the trial.",)}
    wandb = [entry for entry in reference.entries if entry.path.startswith("extractor[type=wandb]")]
    assert [(group.title, group.lines) for group in reference.rules_for(wandb)] == [
        ("Section Objective", ("A bad read fails the trial.",)),
        ("Shared block WandbX", ("Read a W&B summary.",)),
        (
            "Shared block Target",
            ("The W&B run every consumer in a phase reads; all of them agree on these fields.",),
        ),
    ]
    gates = [entry for entry in reference.entries if entry.path.startswith("gates")]
    assert [group.title for group in reference.rules_for(gates)] == ["Shared block Target"]


@pytest.mark.parametrize(
    "text,message",
    [
        ("# ---------- S ----------\n\n# x\nkey: Missing | null = null\n", "Unrecognized|Unknown"),
        ("# x\nkey: string = ''\n\n# y\nUnused:\n  # z\n  a: int = 1\n", "never used"),
        (
            "# x\nkey: A | B (required)\n\n# a\nA:\n  # f\n  f: int = 1\n\n# b\nB:\n  # g\n  g: int = 1\n",
            "type field",
        ),
        ("# x\nkey: list[A] = []\n\n# a\nA:\n  # f\n  f: int = 1\n", "inside another type"),
        (
            "# x\nkey: A (required)\n  # y\n  extra: int = 1\n\n# a\nA:\n  # f\n  f: int = 1\n",
            "below a shared-block use site",
        ),
        (
            "# x\nkey: object = {}\n  # y\n  <<: A\n\n# a\nA:\n  # f\n  f: int = 1\n",
            "above a merge",
        ),
        ("# ---------- S ----------\n\n# x\nkey: int = 1\n# loose\n\n# y\nb: int = 1\n", "no key"),
        (
            "# ---------- S ----------\n# rule\n\n# ---------- T ----------\n\n# x\nk: int = 1\n",
            "no keys",
        ),
    ],
    ids=[
        "unknown-block",
        "unused-block",
        "union-without-type",
        "block-in-list",
        "keys-below-use-site",
        "comment-above-merge",
        "orphan-comment",
        "rules-without-keys",
    ],
)
def test_parser_rejects_malformed_blocks_and_comments_it_cannot_place(
    text: str, message: str
) -> None:
    """Nothing in the file is silently dropped: a line the parser cannot place is an error."""
    with pytest.raises(ValueError, match=message):
        _parse(text)


def test_packaged_reference_notes_are_neither_the_title_block_nor_an_entry_comment() -> None:
    """The packaged notes are the header's second block, apart from the title and every entry."""
    first_block = _reference_resource().read_text(encoding="utf-8").split("\n\n", 1)[0]
    title_lines = {line.removeprefix("#").strip() for line in first_block.splitlines()}
    notes = set(reference_notes())
    assert notes
    assert title_lines
    assert not notes & title_lines
    assert not notes & {comment for entry in reference_entries() for comment in entry.comments}


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
    """A partial path or bare key finds the full documented path, and a typo finds the key it misspells."""
    assert closest("inherit_env") == ("execution.inherit_env",)
    assert closest("sampler.type") == ("phases[].sampler.type",)
    assert closest("phases.sampler.typ")[0] == "phases[].sampler.type"
    assert closest("trial_comand")[0] == "trial_command"
    assert closest("zzzzzz") == ()
    assert closest("") == ()
    assert len(closest("type")) == 3
    assert all("[type=" not in suggestion for suggestion in closest("type"))
    # Fields expanded from shared blocks are suggested at every place that uses the block.
    assert set(closest("extractor.entiy")) >= {
        "metric.extractor.entity",
        "constraints[].extractor.entity",
    }
    assert "phases[].gates[].entity" in closest("gates.entiy")


@pytest.mark.parametrize(
    "path,value,expected",
    [
        (("phases", 0, "sampler"), {"type": "bogus"}, "phases[].sampler.type"),
        (("metric", "extractor", "path"), 3, "metric.extractor.path"),
        (
            ("phases", 0, "search_space"),
            {"lr": {"type": "float", "low": "x", "high": 1}},
            "phases[].search_space.<key>.low",
        ),
        (("phases", 0, "gates"), [{"type": "sha256", "path": "a"}], "phases[].gates[].sha256"),
        (("execution",), {"inherit_env": 3}, "execution.inherit_env"),
        (("env",), {"A": 1}, "env"),
        (
            ("constraints",),
            [{"name": "c", "max": 1, "extractor": {"type": "json", "path": "r.json", "key": 5}}],
            "constraints[].extractor.key",
        ),
        (
            ("phases", 0, "gates"),
            [
                {
                    "type": "wandb_summary_required",
                    "entity": "e",
                    "project": "p",
                    "keys": ["k"],
                    "poll_seconds": "soon",
                }
            ],
            "phases[].gates[].poll_seconds",
        ),
    ],
    ids=[
        "sampler",
        "extractor-path",
        "search-bound",
        "gate-digest",
        "inherit-env",
        "env-value",
        "block-field",
        "merged-block-field",
    ],
)
def test_loc_to_path_maps_variant_and_search_key_segments(path, value, expected) -> None:
    """Indexes and variant tags drop out, and a user-chosen search key maps to ``<key>``."""
    data = yaml.safe_load((REPO / "examples/example.yaml").read_text())
    target = data
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValidationError) as caught:
        Experiment.model_validate(data)
    locs = [error["loc"] for error in caught.value.errors(include_input=False)]
    assert {loc_to_path(loc) for loc in locs} == {expected}, locs


def test_describe_validation_error_adds_distinct_quoted_hints() -> None:
    """Errors under one key give one hint, and a path with brackets is shell-quoted."""
    data = yaml.safe_load((REPO / "examples/example.yaml").read_text())
    data["env"] = {"A": 1, "B": 2}
    data["phases"][0]["sampler"] = {"type": "bogus"}
    with pytest.raises(ValidationError) as caught:
        Experiment.model_validate(data)
    assert len(caught.value.errors()) == 3
    assert describe_validation_error(caught.value) == (
        f"{caught.value}\nsee: phasesweep explain 'phases[].sampler.type'"
        "\nsee: phasesweep explain env"
    )


def test_describe_validation_error_adds_no_hint_without_a_location() -> None:
    """An experiment-level validator reports an empty location, so there is no key to explain."""
    data = yaml.safe_load((REPO / "examples/example.yaml").read_text())
    data["phases"][0]["inherits"] = ["baseline"]
    with pytest.raises(ValidationError) as caught:
        Experiment.model_validate(data)
    assert [error["loc"] for error in caught.value.errors()] == [()]
    assert describe_validation_error(caught.value) == str(caught.value)


def test_loc_to_path_has_no_key_for_unknown_or_model_level_locations() -> None:
    """An empty location or an undocumented top-level key yields no hint."""
    assert loc_to_path(()) is None
    assert loc_to_path(("bogus",)) is None
    assert loc_to_path(("trainer_config", "a", "b")) == "trainer_config"
    assert loc_to_path(("phases", 2, "n_trials")) == "phases[].n_trials"
