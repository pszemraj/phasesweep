"""The console half of a run's logging: the report and its warnings on stderr, nothing else.

``run.log`` receives every record with its logger name and level. The console receives only what a
person needs while the sweep runs, drawn for the terminal, with no traceback and no logger name.
A dry run adds the package's INFO records, which are its launch contract.
"""

from __future__ import annotations

import logging
import sys
import time

from phasesweep.runtime.terminal import (
    GAP,
    REPORT_LOGGER,
    ReportBlock,
    Style,
    hang,
    paint_word,
    terminal_width,
    use_color,
    wrap,
)

_PACKAGE_LOGGER = "phasesweep"


class ConsoleHandler(logging.Handler):
    """Print a run's report lines and its warnings on stderr, and drop every other record.

    A record from the report logger prints as the time and its text, any later line indented
    under the first. A record at WARNING or above from any other logger prints as
    ``warning:`` or ``error:`` and its message, wrapped to the terminal with a hanging indent.
    Neither shows a traceback. A dry run is the launch contract, so there an INFO record from
    any ``phasesweep`` logger also prints, as the time and its text like a report line; a flush-left
    line wider than the terminal wraps under its first, and an indented one stays whole. Each
    record is one write under the handler's lock, so lines from parallel trials never
    interleave inside a record. The stream, its width and its colour decision are read at each
    emit, so a swapped ``sys.stderr`` is honoured.
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        """Choose whether INFO records of the package print too.

        :param bool dry_run: Print INFO records from every ``phasesweep`` logger, not only the
            report logger's.
        """
        super().__init__()
        self._dry_run = dry_run

    def emit(self, record: logging.LogRecord) -> None:
        """Write one record to stderr if the console shows it.

        :param logging.LogRecord record: The record being handled, under the handler's lock.
        """
        if not self._shows(record):
            return
        try:
            stream = sys.stderr
            lines = self._lines(record, terminal_width(stream), use_color(stream))
            stream.write("\n".join(lines) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def _shows(self, record: logging.LogRecord) -> bool:
        """Say whether the console prints ``record``.

        :param logging.LogRecord record: The record being handled.
        :return bool: ``True`` for a report line, a warning or error, and in a dry run an INFO
            record of the package.
        """
        if record.name == REPORT_LOGGER or record.levelno >= logging.WARNING:
            return True
        return (
            self._dry_run
            and record.levelno >= logging.INFO
            and (record.name == _PACKAGE_LOGGER or record.name.startswith(f"{_PACKAGE_LOGGER}."))
        )

    def _lines(self, record: logging.LogRecord, width: int, color: bool) -> list[str]:
        """Draw one record as the console prints it.

        :param logging.LogRecord record: The record to draw.
        :param int width: Terminal width in columns.
        :param bool color: Whether the stream takes styling.
        :return list[str]: The lines to print, the time before the first.
        """
        block = record.msg
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created)) + GAP
        room = width - len(stamp)
        if isinstance(block, ReportBlock):
            body = block.render(room, color=color)
        elif record.name == REPORT_LOGGER or record.levelno < logging.WARNING:
            body = _plain_lines(record.getMessage(), room)
        else:
            body = _diagnostic(record, room, color)
        return [line.rstrip() for line in hang(stamp, body)]


def _plain_lines(message: str, width: int) -> list[str]:
    """Draw a record's own lines, wrapping each flush-left one that is wider than ``width``.

    A wrapped line breaks at spaces and hangs under its first line, which the handler's time
    prefix already indents; a word wider than the line, such as a path, stays whole. A line that
    starts with whitespace holds text to copy, such as the example command, so it stays whole.

    :param str message: The formatted record text.
    :param int width: Columns available after the time.
    :return list[str]: The lines to print, at least one.
    """
    lines: list[str] = []
    for line in message.splitlines() or [""]:
        if len(line) <= width or line[0].isspace():
            lines.append(line)
        else:
            lines.extend(wrap(line, width))
    return lines


def _diagnostic(record: logging.LogRecord, width: int, color: bool) -> list[str]:
    """Draw a warning or error: its label, then its message wrapped under a hanging indent.

    :param logging.LogRecord record: A record at WARNING or above.
    :param int width: Columns available after the time.
    :param bool color: Whether to style the label.
    :return list[str]: The wrapped lines; a message line break starts a new wrapped paragraph.
    """
    label = "error" if record.levelno >= logging.ERROR else "warning"
    style: Style = "error" if label == "error" else "warning"
    paragraphs = record.getMessage().splitlines() or [""]
    lines = wrap(f"{label}: {paragraphs[0]}", width)
    for paragraph in paragraphs[1:]:
        lines.extend(wrap(paragraph, width) or [""])
    lines[0] = paint_word(lines[0], f"{label}:", style, color=color)
    return lines
