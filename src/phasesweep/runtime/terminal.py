"""Terminal primitives for human-readable CLI output: styling, width, wrapping and display text.

Standard library only. Every renderer takes its width and colour decision as plain values, so a
renderer's text is a pure function of its inputs and tests pin it without a terminal.
"""

from __future__ import annotations

import os
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, TextIO

#: Name of the logger whose records are a run's report: the lines a person reads on the console.
REPORT_LOGGER = "phasesweep.report"

Style = Literal["winner", "error", "warning", "label", "path"]

#: SGR parameters of each style: the winner mark is bold green, errors red, warnings yellow,
#: labels bold and paths dim. Every style keeps its meaning readable in words without colour.
_SGR: dict[Style, str] = {
    "winner": "1;32",
    "error": "31",
    "warning": "33",
    "label": "1",
    "path": "2",
}

#: Indent of the lines a block draws under its first, and of a table's rows.
INDENT = "  "

#: Separator of the facts on one line, and of the columns that follow a label.
GAP = "  "

_DEFAULT_WIDTH = 80

#: Columns kept for text after an indent, so a very narrow terminal wraps instead of failing.
MIN_TEXT_WIDTH = 20


class _SupportsIsatty(Protocol):
    """The one stream method the colour decision reads."""

    def isatty(self) -> bool:
        """Report whether the stream is attached to a terminal."""
        ...


def use_color(stream: _SupportsIsatty) -> bool:
    """Decide whether styled text may be written to ``stream``.

    Styling needs a terminal and no ``NO_COLOR`` variable in the environment; an empty
    ``NO_COLOR`` disables it too. Stdout and stderr are decided separately.

    :param _SupportsIsatty stream: Stream the text is about to be written to.
    :return bool: ``True`` when ANSI styling is appropriate for that stream.
    """
    return stream.isatty() and "NO_COLOR" not in os.environ


def paint(text: str, style: Style, *, color: bool) -> str:
    """Wrap ``text`` in the escape sequences of ``style``, or return it unchanged without colour.

    Callers measure and pad the plain text first, so escape bytes never count toward a width.

    :param str text: Visible text to style.
    :param Style style: Named style to apply.
    :param bool color: The result of :func:`use_color` for the destination stream.
    :return str: The styled text, or ``text`` itself when ``color`` is off or ``text`` is empty.
    """
    if not color or not text:
        return text
    return f"\x1b[{_SGR[style]}m{text}\x1b[0m"


def paint_word(line: str, word: str, style: Style, *, color: bool, start: int = 0) -> str:
    """Style the first ``word`` found at or after column ``start`` of an already laid-out line.

    :param str line: Plain text of one line.
    :param str word: Text to style.
    :param Style style: Style to apply.
    :param bool color: Whether styling is on.
    :param int start: Column to search from, so a phase name never matches, defaults to 0.
    :return str: The line with ``word`` styled, or unchanged when absent or colour is off.
    """
    index = line.find(word, start)
    if not color or index < 0:
        return line
    return line[:index] + paint(word, style, color=True) + line[index + len(word) :]


def terminal_width(stream: TextIO) -> int:
    """Return the column count to fit prose to: ``COLUMNS``, then the stream's own size, then 80.

    :param TextIO stream: Stream the text is about to be written to.
    :return int: Positive number of columns.
    """
    columns = os.environ.get("COLUMNS", "")
    if columns.isdecimal() and int(columns) > 0:
        return int(columns)
    try:
        size = os.get_terminal_size(stream.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return _DEFAULT_WIDTH
    return size if size > 0 else _DEFAULT_WIDTH


def wrap(text: str, width: int, *, indent: str = "", subsequent_indent: str = "") -> list[str]:
    """Wrap prose to ``width`` columns at spaces, never cutting a word.

    Runs of whitespace, newlines included, collapse to one space. A word wider than the line,
    such as a path, goes whole on a line of its own and overruns the width, so a person can copy
    it. A terminal too narrow for the indents still gets at least twenty columns of text.

    :param str text: Prose to wrap.
    :param int width: Maximum visible width of each line, indent included, except for a word wider
        than that.
    :param str indent: Prefix of the first line.
    :param str subsequent_indent: Prefix of every later line, the hanging indent.
    :return list[str]: The wrapped lines, empty when ``text`` has no words.
    """
    limit = max(width, len(indent) + MIN_TEXT_WIDTH, len(subsequent_indent) + MIN_TEXT_WIDTH)
    return textwrap.wrap(
        " ".join(text.split()),
        limit,
        initial_indent=indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def wrap_labeled(label: str, text: str, width: int, *, indent: str = "") -> list[str]:
    """Wrap ``text`` after ``label``, every later line hanging under the text.

    A line break in ``text`` starts a new paragraph, which hangs under the text too. A word
    wider than the line is kept whole, as :func:`wrap` does.

    :param str label: Leading text of the first line, such as ``fix:``.
    :param str text: Prose that follows the label.
    :param int width: Maximum visible width of each line, indent included, except for a word wider
        than that.
    :param str indent: Prefix of every line, so the label and the text it hangs under both sit
        past it.
    :return list[str]: The lines; the indent and label alone when ``text`` has no words.
    """
    hanging = indent + " " * (len(label) + 1)
    paragraphs = text.splitlines() or [""]
    first = wrap(paragraphs[0], width, indent=f"{indent}{label} ", subsequent_indent=hanging)
    lines = first or [indent + label]
    for paragraph in paragraphs[1:]:
        lines.extend(wrap(paragraph, width, indent=hanging, subsequent_indent=hanging) or [""])
    return lines


def hang(lead: str, lines: Sequence[str], *, visible: int | None = None) -> list[str]:
    """Put ``lead`` before the first of ``lines`` and blanks as wide as it before each later one.

    :param str lead: Text before the first line, possibly styled.
    :param Sequence[str] lines: The lines, with nothing reserved in front of them for the lead.
    :param int | None visible: Columns ``lead`` takes on screen when its escape sequences are
        not counted, defaults to its length.
    :return list[str]: The lines, every later one hanging under the first.
    """
    pad = " " * (len(lead) if visible is None else visible)
    return [(lead if index == 0 else pad) + line for index, line in enumerate(lines)]


def display_path(path: Path) -> str:
    """Show ``path`` relative to the working directory, else under ``~``, else in full.

    :param Path path: A local path, absolute or relative to the working directory.
    :return str: The shortest of those three spellings that applies, with ``/`` separators.
    """
    absolute = path if path.is_absolute() else path.resolve()
    for base, prefix in ((Path.cwd(), ""), (Path.home(), "~")):
        if absolute == base:
            return prefix or "."
        if absolute.is_relative_to(base):
            relative = absolute.relative_to(base).as_posix()
            return f"{prefix}/{relative}" if prefix else relative
    return absolute.as_posix()


def plural(count: int, noun: str) -> str:
    """Pair ``count`` with ``noun`` in the singular for one and the regular plural otherwise.

    :param int count: How many there are.
    :param str noun: Singular noun that takes a plain ``s``.
    :return str: ``1 trial``, ``2 trials``, ``0 trials``, with a thousands separator on large counts.
    """
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def significant(value: float, digits: int) -> str:
    """Format a number to ``digits`` significant digits, never in positive-exponent form.

    A value too large for ``digits`` would read as ``8.8e+06``, so it prints in full plain
    notation instead. The exact value stays in ``trials.csv`` and ``winner.yaml``.

    :param float value: The number to show.
    :param int digits: Significant digits kept for a value that fits them.
    :return str: Terminal text for the number.
    """
    text = f"{value:.{digits}g}"
    if "e+" not in text:
        return text
    return format(Decimal(repr(value)).normalize(), "f")


def objective_text(value: float) -> str:
    """Format an objective, or a difference between objectives, to six significant digits.

    :param float value: The objective or gap to show.
    :return str: Terminal text for the number.
    """
    return significant(value, 6)


def param_text(value: object) -> str:
    """Format one sampled param for display, floats to four significant digits.

    :param object value: The recorded param value.
    :return str: Terminal text for the value.
    """
    return significant(value, 4) if isinstance(value, float) else str(value)


def pack(items: Sequence[str], width: int, *, indent: str = "", separator: str = GAP) -> list[str]:
    """Fill lines of ``width`` columns with whole items joined by ``separator``.

    An item is never split, so ``key=value`` pairs stay readable; one wider than the line gets a
    line of its own. A line that ends before the last item keeps the separator's trailing
    punctuation, so a comma list reads on after each break.

    :param Sequence[str] items: Texts to place, in order.
    :param int width: Maximum visible width of each line, indent included.
    :param str indent: Prefix of every line.
    :param str separator: Text between two items on one line.
    :return list[str]: The lines, empty when there are no items.
    """
    lines: list[str] = []
    current = ""
    punctuation = separator.rstrip()
    for item in items:
        if current and len(indent + current + separator + item + punctuation) > width:
            lines.append(indent + current + punctuation)
            current = item
        else:
            current = f"{current}{separator}{item}" if current else item
    if current:
        lines.append(indent + current)
    return lines


def break_path(text: str, width: int, *, indent: str = "") -> list[str]:
    """Break a path or URL after a ``/`` so every component stays whole and readable.

    :param str text: The path, absolute or relative.
    :param int width: Maximum visible width of each line, indent included.
    :param str indent: Prefix of every line.
    :return list[str]: The lines, each but the last ending in ``/``.
    """
    parts = text.split("/")
    if text.startswith("/") and len(parts) > 1:
        parts = ["/" + parts[1], *parts[2:]]
    return pack(parts, width, indent=indent, separator="/")


def directory_lines(directories: Sequence[str], *, width: int | None, color: bool) -> list[str]:
    """Lay out directory paths dim, each on a line of its own, with a shared parent named once.

    Several directories in one parent print that parent on a line ending in a slash, then each
    directory's own name two columns in; otherwise each prints in full. A path is never cut: with
    a ``width`` it breaks after slashes to fit and keeps each component whole, and without one it
    is a single line however long, so a person can copy it.

    :param Sequence[str] directories: Display paths, as :func:`display_path` spells them, in order.
    :param int | None width: Columns each line must fit, or ``None`` to keep each path whole.
    :param bool color: Whether to style the paths.
    :return list[str]: The lines, without any indent in front of the first column.
    """

    def drawn(text: str, *, indent: str = "") -> list[str]:
        """Break ``text`` to fit when a width is given and style it."""
        fitted = [text] if width is None else break_path(text, max(width - len(indent), 1))
        return [indent + paint(line, "path", color=color) for line in fitted]

    parents = {PurePosixPath(directory).parent for directory in directories}
    if len(directories) < 2 or len(parents) != 1 or parents == {PurePosixPath(".")}:
        return [line for directory in directories for line in drawn(directory)]
    shared = parents.pop().as_posix().rstrip("/") + "/"
    return [
        *drawn(shared),
        *(
            line
            for directory in directories
            for line in drawn(PurePosixPath(directory).name, indent=INDENT)
        ),
    ]


class ReportBlock(ABC):
    """A log message that draws itself on the console and reads as one line in a log file.

    A block is logged on the ``phasesweep.report`` logger. ``str()`` is the plain one-line text
    that ``run.log``, a test's captured records and the MCP runner's log receive, and
    :meth:`render` is what the console handler prints after the time of day.
    """

    @abstractmethod
    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the block for a terminal.

        :param int width: Columns available to each line; the console takes the time prefix off.
        :param bool color: Whether to style the text, from :func:`use_color` on the stream.
        :return list[str]: The lines, without line breaks and without the time prefix.
        """

    @abstractmethod
    def __str__(self) -> str:
        """Return the block as one plain line of text.

        :return str: Text without line breaks or styling.
        """


@dataclass(frozen=True)
class ReportText(ReportBlock):
    """A report line of prose: wrapped to the terminal on the console, as written in a log file."""

    text: str

    def render(self, width: int, *, color: bool) -> list[str]:
        """Wrap the text to the terminal, any later line under the first.

        :param int width: Columns available to each line.
        :param bool color: Unused; the line carries no styling.
        :return list[str]: The wrapped lines.
        """
        return wrap(self.text, width)

    def __str__(self) -> str:
        """Return the text as written.

        :return str: The one-line text.
        """
        return self.text
