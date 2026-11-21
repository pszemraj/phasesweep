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
    display_path,
    paint,
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
