"""Terminal primitives: colour decision, styling, width, wrapping, path display and plurals."""

from __future__ import annotations

import fcntl
import io
import os
import struct
import sys
import termios
from pathlib import Path

import pytest

from phasesweep.runtime.terminal import (
    directory_lines,
    display_path,
    hang,
    paint,
    paint_word,
    plural,
    terminal_width,
    use_color,
    wrap,
    wrap_labeled,
)


class _Stream:
    """A stream that only knows whether it is a terminal."""

    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _no_inherited_terminal_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without the developer's ``NO_COLOR`` or ``COLUMNS``."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLUMNS", raising=False)


@pytest.mark.parametrize("no_color", [None, "1", ""], ids=["unset", "set", "empty"])
@pytest.mark.parametrize("stderr_tty", [True, False], ids=["stderr-tty", "stderr-redirected"])
@pytest.mark.parametrize("stdout_tty", [True, False], ids=["stdout-tty", "stdout-redirected"])
def test_color_is_decided_per_stream_and_disabled_by_any_no_color(
    monkeypatch: pytest.MonkeyPatch,
    stdout_tty: bool,
    stderr_tty: bool,
    no_color: str | None,
) -> None:
    """Colour needs a terminal and no ``NO_COLOR``, an empty one included, for each stream alone."""
    if no_color is not None:
        monkeypatch.setenv("NO_COLOR", no_color)
    monkeypatch.setattr(sys, "stdout", _Stream(tty=stdout_tty))
    monkeypatch.setattr(sys, "stderr", _Stream(tty=stderr_tty))

    assert use_color(sys.stdout) is (stdout_tty and no_color is None)
    assert use_color(sys.stderr) is (stderr_tty and no_color is None)


def test_paint_returns_plain_text_without_color_and_escapes_with_it() -> None:
    """Escapes wrap the whole text and reset after it; empty text and no colour stay untouched."""
    assert paint("winner", "winner", color=False) == "winner"
    assert paint("winner", "winner", color=True) == "\x1b[1;32mwinner\x1b[0m"
    assert paint("x", "error", color=True) == "\x1b[31mx\x1b[0m"
    assert paint("x", "warning", color=True) == "\x1b[33mx\x1b[0m"
    assert paint("x", "label", color=True) == "\x1b[1mx\x1b[0m"
    assert paint("x", "path", color=True) == "\x1b[2mx\x1b[0m"
    assert paint("", "winner", color=True) == ""


def test_paint_word_styles_the_first_match_from_the_start_column_only() -> None:
    """The phase name is skipped by the start column, a missing word and no colour change nothing."""
    line = "depth  depth FAIL FAIL"

    assert paint_word(line, "depth", "label", start=2, color=True) == (
        "depth  \x1b[1mdepth\x1b[0m FAIL FAIL"
    )
    assert paint_word(line, "FAIL", "error", color=True) == "depth  depth \x1b[31mFAIL\x1b[0m FAIL"
    assert paint_word(line, "absent", "error", color=True) == line
    assert paint_word(line, "FAIL", "error", color=False) == line


def test_hang_puts_the_lead_on_the_first_line_and_blanks_under_it() -> None:
    """A styled lead hangs later lines by its visible width, and no lines stays empty."""
    assert hang("lead  ", ["one", "two", "three"]) == ["lead  one", "      two", "      three"]
    styled = paint("log:", "label", color=True) + " "
    assert hang(styled, ["a", "b"], visible=5) == [f"{styled}a", "     b"]
    assert hang("lead", []) == []


def test_terminal_width_prefers_columns_then_the_stream_then_eighty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``COLUMNS`` wins, an unusable value is skipped, and a stream without a size reads 80."""
    redirected = io.StringIO()
    assert terminal_width(redirected) == 80
    monkeypatch.setenv("COLUMNS", "132")
    assert terminal_width(redirected) == 132
    for unusable in ("", "0", "wide", "-5"):
        monkeypatch.setenv("COLUMNS", unusable)
        assert terminal_width(redirected) == 80


def test_terminal_width_reads_the_size_of_the_stream_itself() -> None:
    """A pseudo-terminal's own column count is used when ``COLUMNS`` is absent."""
    master, slave = os.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 67, 0, 0))
        with os.fdopen(slave, "w", closefd=False) as stream:
            assert terminal_width(stream) == 67
    finally:
        os.close(slave)
        os.close(master)


def test_wrap_hangs_continuation_lines_and_keeps_every_word() -> None:
    """Every line fits, the first carries the indent, the rest the hanging indent."""
    text = "trainer exited with code 3 after the evaluation step failed to produce a metric value"
    lines = wrap(text, 40, indent="reason: ", subsequent_indent="        ")

    assert lines == [
        "reason: trainer exited with code 3 after",
        "        the evaluation step failed to",
        "        produce a metric value",
    ]
    assert all(len(line) <= 40 for line in lines)
    assert " ".join(line.strip() for line in lines).removeprefix("reason: ") == text


def test_wrap_collapses_newlines_and_keeps_a_word_longer_than_the_width_whole() -> None:
    """Newlines become spaces, and a path wider than the line gets a line of its own, uncut."""
    long_word = "/data/" + "x" * 50
    lines = wrap(f"saved to\n{long_word} done", 30, indent="# ", subsequent_indent="# ")

    assert lines == ["# saved to", f"# {long_word}", "# done"]


def test_wrap_never_cuts_a_word_at_a_hyphen_or_a_slash() -> None:
    """A hyphenated name or a path that does not fit the line moves down whole."""
    name = "trial_00003__generation_88e54b25a7bb4d6c__attempt_dc15a62b79fb4dc1-and-more"
    lines = wrap(f"reason: JSON file not found: {name} (the trainer writes it)", 40)

    assert lines == [
        "reason: JSON file not found:",
        name,
        "(the trainer writes it)",
    ]


def test_wrap_labeled_hangs_every_later_line_and_paragraph_under_the_text() -> None:
    """The label leads the first line only, and a path wider than the line stays whole."""
    path = "/data/" + "x" * 40
    lines = wrap_labeled("fix:", f"Open {path} and rerun the failed trial.\nThen check it.", 30)

    assert lines == [
        "fix: Open",
        f"     {path}",
        "     and rerun the failed",
        "     trial.",
        "     Then check it.",
    ]
    assert wrap_labeled("fix:", "", 30) == ["fix:"]


def test_directory_lines_name_a_shared_parent_once_and_fit_or_keep_whole() -> None:
    """Directories in one parent print it once; a width breaks at slashes, none keeps one line."""
    parent = "/data/sweeps/phasesweep_starter/runs/phasesweep_starter/depth"
    members = [f"{parent}/trial_0000{number}__attempt_{'f' * 40}" for number in (4, 5)]

    assert directory_lines(members, width=80, color=False) == [
        f"{parent}/",
        f"  {members[0].rsplit('/', 1)[1]}",
        f"  {members[1].rsplit('/', 1)[1]}",
    ]
    assert directory_lines(members, width=40, color=False)[:2] == [
        "/data/sweeps/phasesweep_starter/runs/",
        "phasesweep_starter/depth/",
    ]
    assert directory_lines(members[:1], width=None, color=False) == members[:1]
    assert directory_lines(members, width=None, color=False)[0] == f"{parent}/"
    assert directory_lines([], width=80, color=False) == []


def test_directory_lines_print_each_path_in_full_when_nothing_is_shared() -> None:
    """Different parents, one directory, and bare names have no parent line to share."""
    assert directory_lines(["a/trial_1", "b/trial_2"], width=40, color=False) == [
        "a/trial_1",
        "b/trial_2",
    ]
    assert directory_lines(["/a/trial_1"], width=40, color=False) == ["/a/trial_1"]
    assert directory_lines(["trial_1", "trial_2"], width=40, color=False) == ["trial_1", "trial_2"]


def test_directory_lines_dim_each_path_and_nothing_else() -> None:
    """Colour wraps the text of every line, and the indent before a member stays plain."""
    assert directory_lines(["/a/trial_1", "/a/trial_2"], width=40, color=True) == [
        "\x1b[2m/a/\x1b[0m",
        "  \x1b[2mtrial_1\x1b[0m",
        "  \x1b[2mtrial_2\x1b[0m",
    ]


def test_wrap_keeps_a_minimum_text_width_on_a_very_narrow_terminal() -> None:
    """A width smaller than the indent still wraps rather than raising."""
    lines = wrap(
        "a short sentence that needs wrapping", 5, indent="    ", subsequent_indent="        "
    )

    assert lines
    assert all(len(line) <= 28 for line in lines)
    assert wrap("", 80, indent="# ") == []


def test_display_path_prefers_the_working_directory_then_home_then_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shortest applicable spelling wins, in that order."""
    work = tmp_path / "work"
    home = tmp_path / "home"
    work.mkdir()
    home.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))

    assert display_path(work / "runs" / "p" / "trials.csv") == "runs/p/trials.csv"
    assert display_path(Path("runs/p/trials.csv")) == "runs/p/trials.csv"
    assert display_path(work) == "."
    assert display_path(home / "projects" / "a.yaml") == "~/projects/a.yaml"
    assert display_path(home) == "~"
    assert display_path(Path("/opt/elsewhere/trials.csv")) == "/opt/elsewhere/trials.csv"
    assert display_path(tmp_path / "other" / "a.yaml") == str(tmp_path / "other" / "a.yaml")


def test_display_path_checks_the_working_directory_before_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A working directory under home shows paths relative to itself, not as ``~/...``."""
    home = tmp_path / "home"
    work = home / "work"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))

    assert display_path(work / "a.csv") == "a.csv"
    assert display_path(home / "b.csv") == "~/b.csv"


@pytest.mark.parametrize(
    ("count", "text"),
    [(0, "0 phases"), (1, "1 phase"), (2, "2 phases"), (1234, "1,234 phases")],
)
def test_plural_uses_the_singular_only_for_one(count: int, text: str) -> None:
    """One takes the bare noun and every other count the regular plural."""
    assert plural(count, "phase") == text
