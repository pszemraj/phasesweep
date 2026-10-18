"""``phasesweep explain``: the configuration reference one key at a time."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from phasesweep.cli import cli as cli_main
from phasesweep.config.reference import reference_entries, reference_notes, top_level_entries


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


def test_explain_notes_prints_the_general_notes_and_no_maintainer_lines() -> None:
    """``--notes`` prints the header block that applies to every key, one line per note."""
    code, out, err = _explain("--notes")
    assert code == 0
    assert err == ""
    assert out.splitlines() == list(reference_notes())
    assert "Paths: a relative workdir" in out
    assert "YAML scalars follow PyYAML" in out
    assert "PHASESWEEP_LOCK_DIR" in out
    for maintainer_line in ("does not load as YAML", "maintained by hand", "parsed by", "tests/"):
        assert maintainer_line not in out


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
    """A key without children prints one block."""
    code, out, _ = _explain("execution.inherit_env")
    assert code == 0
    assert out.startswith("execution.inherit_env: ")
    assert "\n\n" not in out


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
