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
from pathlib import Path
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
_MIN_TEXT_WIDTH = 20


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
    limit = max(width, len(indent) + _MIN_TEXT_WIDTH, len(subsequent_indent) + _MIN_TEXT_WIDTH)
    return textwrap.wrap(
        " ".join(text.split()),
        limit,
        initial_indent=indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


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


def param_text(value: object) -> str:
    """Format one sampled param for display, floats to four significant digits.

    :param object value: The recorded param value.
    :return str: Terminal text for the value.
    """
    return significant(value, 4) if isinstance(value, float) else str(value)


def pack(items: Sequence[str], width: int, *, indent: str = "", separator: str = "  ") -> list[str]:
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
