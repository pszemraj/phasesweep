"""``phasesweep explain``: the configuration reference one key at a time."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

import phasesweep.config.reference as reference
from phasesweep.cli import cli as cli_main
from phasesweep.config.reference import (
    reference_entries,
    reference_notes,
    rules_for,
    top_level_entries,
)
from phasesweep.mcp.tools import PhaseSweepMCP

_SAMPLE = """\
# sample reference

# note one

# ---------- Objective ----------
# A bad read fails the trial.

# Scalar source.
extractor: JsonX | WandbX (required)

# ---------- Gates ----------

# Hard checks.
gates: list[wb] = []
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

# The W&B run every consumer in a phase reads.
Target:
  # Seconds to wait.
  timeout_seconds: float = 120.0
"""


@pytest.fixture
def sample_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve a small reference with section rules and shared blocks in place of the packaged one.

    :param Path tmp_path: Directory for the sample file.
    :param pytest.MonkeyPatch monkeypatch: Replaces the packaged resource lookup.
    :return Iterator[None]: Yields while the sample is served, then restores the packaged one.
    """
    path = tmp_path / "config_reference.yaml"
    path.write_text(_SAMPLE, encoding="utf-8")
    caches = (
        reference._packaged,
        reference.reference_notes,
        reference._display_by_segments,
        reference._variant_tags,
    )
    monkeypatch.setattr(reference, "_reference_resource", lambda: path)
    for cache in caches:
        cache.cache_clear()
    yield
    monkeypatch.undo()
    for cache in caches:
        cache.cache_clear()


def _explain(*args: str) -> tuple[int, str, str]:
    """Run ``phasesweep explain`` and capture its streams.

    :param str args: Arguments after the command name.
    :return tuple[int, str, str]: Exit code, stdout and stderr.
    """
    runner = CliRunner()
    if hasattr(runner, "mix_stderr"):
        runner.mix_stderr = False
    result = runner.invoke(cli_main, ["explain", *args], terminal_width=120)
    return result.exit_code, result.stdout, result.stderr


@pytest.mark.parametrize(
    ("version", "editable", "ref"),
    [
        ("0.3.11", False, "v0.3.11"),
        ("0.3.12.dev4+g123abcd.d20261023", False, "main"),
        ("0.3.13.dev0", False, "main"),
        ("0.0.0+unknown", False, "main"),
        ("0.3.11", True, "main"),
    ],
)
def test_explain_links_pin_installed_release_or_source_revision(
    monkeypatch: pytest.MonkeyPatch, version: str, editable: bool, ref: str
) -> None:
    """Release help pins its tag; development and editable installs use main."""
    with monkeypatch.context() as patch:
        patch.setattr(reference, "__version__", version)
        patch.setattr(reference, "_editable_install", lambda: editable)
        reference._packaged.cache_clear()
        try:
            code, out, err = _explain("metric.extractor")
            assert code == 0 and not err
            assert f"https://github.com/pszemraj/phasesweep/blob/{ref}/docs/mlflow.md#" in out
            assert "See: docs/" not in out
        finally:
            reference._packaged.cache_clear()


@pytest.mark.parametrize("editable", [False, True])
def test_documentation_links_read_editable_install_metadata(
    monkeypatch: pytest.MonkeyPatch, editable: bool
) -> None:
    """Editable metadata selects main even when the installed version is a release."""

    class InstalledDistribution:
        def read_text(self, filename: str) -> str:
            assert filename == "direct_url.json"
            return '{"dir_info": {"editable": ' + str(editable).lower() + "}}"

    monkeypatch.setattr(reference, "distribution", lambda _name: InstalledDistribution())
    monkeypatch.setattr(reference, "__version__", "0.3.12")
    ref = "main" if editable else "v0.3.12"
    assert (
        reference.pin_documentation_links(
            "https://github.com/pszemraj/phasesweep/blob/main/docs/mlflow.md"
        )
        == f"https://github.com/pszemraj/phasesweep/blob/{ref}/docs/mlflow.md"
    )


def test_explain_mlflow_mode_gives_the_native_environment_contract() -> None:
    """Managed-mode help keeps every log in the supplied native run."""
    code, out, err = _explain("mlflow.run_mode")
    assert code == 0 and not err
    assert "first mlflow.start_run() or mlflow.log_* resumes it" in out
    assert "mlflow.active_run().info.run_id" in out
    assert "mlflow.start_run(run_id=...)" in out
    assert "Do not call mlflow.set_experiment() with another name" in out


def test_explain_wandb_gives_the_native_environment_contract() -> None:
    """W&B help names identity overrides and premature finish as contract violations."""
    code, out, err = _explain("metric.extractor")
    assert code == 0 and not err
    assert "Pass no project, entity or id to wandb.init()" in out
    assert "wandb.finish() only after the final log" in out


def test_explain_environment_points_to_queryable_tracker_keys() -> None:
    """Environment help points to config keys that explain accepts."""
    code, out, err = _explain("env")
    assert code == 0 and not err
    assert "WandbTarget" not in out
    assert "metric.extractor.entity" in out
    assert _explain("metric.extractor.entity")[0] == 0


def test_explain_without_a_key_lists_each_section_and_top_level_key() -> None:
    """Every top-level key appears once under its section, with signature, summary and count."""
    code, out, err = _explain()
    assert code == 0
    assert err == ""
    lines = out.splitlines()
    for entry in top_level_entries():
        rows = [line for line in lines if line.startswith(f"  {entry.path} ")]
        assert len(rows) == 1, entry.path
        assert entry.signature in rows[0]
        assert lines.index(entry.section) < lines.index(rows[0])
    phases = next(line for line in lines if line.startswith("  phases "))
    assert "list[object] (required)" in phases
    assert "Ordered list of phase objects" in phases
    assert phases.endswith("sub-keys)")
    storage = next(line for line in lines if line.startswith("  storage "))
    assert "sub-key" not in storage
    assert "phasesweep explain 'phases[].sampler'" in out
    assert out.rstrip("\n").endswith("phasesweep explain --notes")
    assert "config_reference.yaml" not in out


def test_explain_notes_prints_the_general_notes_and_not_the_title() -> None:
    """``--notes`` prints the header block that applies to every key, one line per note."""
    code, out, err = _explain("--notes")
    assert code == 0
    assert err == ""
    assert out.splitlines() == list(reference_notes())
    assert "Paths: a relative workdir" in out
    assert "YAML scalars follow PyYAML" in out
    assert "docs/runtime.md#environment-variables" in out
    for title_line in ("does not load", "signature form", "parsed by", "tests/"):
        assert title_line not in out


def test_explain_notes_refuses_a_key() -> None:
    """``--notes`` and a KEY are different requests, so combining them is a usage error."""
    code, out, err = _explain("--notes", "phases")
    assert code == 2
    assert out == ""
    assert "--notes takes no KEY" in err


def test_explain_listing_stays_within_one_screen_of_summary() -> None:
    """The listing is an index, so no row carries a whole paragraph."""
    _, out, _ = _explain()
    assert len(out.splitlines()) < 40
    assert max(len(line) for line in out.splitlines() if line.startswith("  ")) < 170


def test_explain_a_subtree_prints_each_entry_with_indented_comments() -> None:
    """The key and every key below it print as ``path: signature`` plus indented comments."""
    code, out, err = _explain("phases[].sampler")
    assert code == 0
    assert err == ""
    blocks = out.rstrip("\n").split("\n\n")
    expected = [entry for entry in reference_entries() if entry.path.startswith("phases[].sampler")]
    heads = [
        "\n".join([f"{rules.title}:", *(f"  {line}" for line in rules.lines)])
        for rules in rules_for(expected)
    ]
    assert heads and heads[0].startswith("Section Phases:\n")
    assert blocks[: len(heads)] == heads
    blocks = blocks[len(heads) :]
    assert len(blocks) == len(expected) >= 2
    first = blocks[0].splitlines()
    assert first[0] == f"phases[].sampler: {expected[0].signature}"
    assert first[1:] == [f"  {comment}" for comment in expected[0].comments]
    assert blocks[1].startswith("phases[].sampler.type: ")


def test_explain_labels_discriminator_variants() -> None:
    """Entries that exist in several variants carry their ``[type=...]`` label."""
    code, out, _ = _explain("metric.extractor.path")
    assert code == 0
    heads = [line for line in out.splitlines() if not line.startswith(" ") and line]
    assert any(head.startswith("metric.extractor[type=json].path: ") for head in heads)
    assert any(head.startswith("metric.extractor[type=json_envelope].path: ") for head in heads)
    assert len(heads) == len(set(heads))


@pytest.mark.parametrize(
    "spelling",
    ["phases[].repeats", "phases.repeats", "phases.0.repeats", "phases[0].repeats"],
)
def test_explain_accepts_bracket_dotted_and_indexed_spellings(spelling: str) -> None:
    """One key, four spellings, one answer."""
    reference = _explain("phases[].repeats")
    assert reference[0] == 0
    assert "phases[].repeats.values: " in reference[1]
    assert _explain(spelling) == reference


def test_explain_a_leaf_prints_only_that_entry() -> None:
    """A key without children prints its section rules and then one entry block."""
    code, out, _ = _explain("execution.inherit_env")
    assert code == 0
    blocks = out.rstrip("\n").split("\n\n")
    assert blocks[0].startswith("Section Execution environment:\n")
    assert [block for block in blocks if not block.startswith("Section ")] == [blocks[-1]]
    assert blocks[-1].startswith("execution.inherit_env: ")


def test_explain_resolves_a_dotted_search_parameter_name() -> None:
    """A dotted trainer key resolves to its search-parameter reference."""
    code, out, err = _explain("phases[].search_space.optimizer.lr")

    assert code == 0
    assert not err
    assert "choices:" in out
    assert "low:" in out


def test_explain_phases_is_a_compact_index_to_its_children() -> None:
    """The broad phases query stays usable and points to detailed child queries."""
    code, out, err = _explain("phases")

    assert code == 0
    assert not err
    assert "phases[].n_trials" in out
    assert "phases[].search_space" in out
    assert "phasesweep explain" in out
    assert len(out) < 8000


def test_explain_unknown_key_exits_two_with_closest_matches() -> None:
    """A miss goes to stderr, names the closest documented keys and prints nothing else."""
    code, out, err = _explain("inherit_env")
    assert code == 2
    assert out == ""
    assert (
        err
        == "phasesweep explain: no config key matches 'inherit_env'. Closest: execution.inherit_env.\n"
    )
    code, out, err = _explain("phases.sampler.typ")
    assert code == 2
    assert out == ""
    assert err.startswith(
        "phasesweep explain: no config key matches 'phases.sampler.typ'. Closest: "
    )
    assert "phases[].sampler.type" in err


@pytest.mark.parametrize("key", ["zzzzzz", ".", "[]", " "])
def test_explain_unknown_key_without_neighbours_points_at_the_listing(key: str) -> None:
    """When nothing is close, or the key has no segments, the error points at the listing."""
    code, out, err = _explain(key)
    assert code == 2
    assert out == ""
    assert "no config key matches" in err
    assert "Run `phasesweep explain` to list the top-level keys." in err


@pytest.mark.usefixtures("sample_reference")
def test_explain_prints_section_rules_and_block_descriptions_before_the_entries() -> None:
    """A key's section rules and the blocks its entries came from head the output, once each."""
    code, out, err = _explain("extractor.timeout_seconds")
    assert code == 0
    assert err == ""
    assert out == (
        "Section Objective:\n  A bad read fails the trial.\n\n"
        "Shared block WandbX:\n  Read a W&B summary.\n\n"
        "Shared block Target:\n  The W&B run every consumer in a phase reads.\n\n"
        "extractor[type=wandb].timeout_seconds: float = 120.0\n  Seconds to wait.\n"
    )
    code, out, _ = _explain("gates[].timeout_seconds")
    assert code == 0
    assert out.startswith("Shared block Target:\n")
    assert "Section" not in out
    assert PhaseSweepMCP.explain_config("extractor.timeout_seconds")["rules"] == [
        {"kind": "section", "name": "Objective", "description": "A bad read fails the trial."},
        {"kind": "block", "name": "WandbX", "description": "Read a W&B summary."},
        {
            "kind": "block",
            "name": "Target",
            "description": "The W&B run every consumer in a phase reads.",
        },
    ]
    assert PhaseSweepMCP.explain_config(None)["rules"] == []


@pytest.mark.usefixtures("sample_reference")
def test_typo_suggestions_and_validation_hints_reach_shared_block_fields() -> None:
    """Expanded block fields are ordinary keys for suggestions and error locations."""
    assert set(reference.closest("timeout_secnds")) == {
        "extractor.timeout_seconds",
        "gates[].timeout_seconds",
    }
    assert reference.loc_to_path(("extractor", "wandb", "timeout_seconds")) == (
        "extractor.timeout_seconds"
    )
    assert reference.loc_to_path(("gates", 0, "wb", "timeout_seconds")) == "gates[].timeout_seconds"


def test_explain_prints_the_packaged_section_rules_and_shared_block_descriptions() -> None:
    """A field merged from a shared block shows its use site's section and the block's description."""
    code, out, err = _explain("constraints[].extractor.timeout_seconds")
    assert code == 0
    assert err == ""
    blocks = out.rstrip("\n").split("\n\n")
    assert blocks[0].startswith("Section Constraints:\n")
    assert blocks[1].startswith("Shared block WandbTarget:\n")
    assert blocks[2].startswith(
        "constraints[].extractor[type=wandb].timeout_seconds: float = 120.0\n"
    )
    assert len(blocks) == 3
    rules = PhaseSweepMCP.explain_config("phases[].gates[].entity")["rules"]
    assert [(group["kind"], group["name"]) for group in rules] == [
        ("section", "Phase gates"),
        ("block", "WandbTarget"),
    ]


def test_explain_mlflow_target_uses_a_queryable_wandb_url_reference() -> None:
    """The normalization reference names a real extractor config key."""
    code, out, err = _explain("mlflow.tracking_uri")
    assert code == 0 and not err
    assert "WandbTarget" not in out
    assert "metric.extractor.base_url" in out
    assert _explain("metric.extractor.base_url")[0] == 0
