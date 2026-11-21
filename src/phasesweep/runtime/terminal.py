"""Terminal primitives for human-readable CLI output: styling, width, wrapping and display text.

Standard library only. Every renderer takes its width and colour decision as plain values, so a
renderer's text is a pure function of its inputs and tests pin it without a terminal.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Literal, Protocol, TextIO

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
    """Wrap prose to ``width`` columns, breaking an overlong word rather than overrunning.

    Runs of whitespace, newlines included, collapse to one space. A terminal too narrow for the
    indents still gets at least twenty columns of text.

    :param str text: Prose to wrap.
    :param int width: Maximum visible width of each line, indent included.
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
