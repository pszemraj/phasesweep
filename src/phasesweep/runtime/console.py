"""The console half of a run's logging: the report and its warnings on stderr, nothing else.

``run.log`` receives every record with its logger name and level. The console receives only what a
person needs while the sweep runs, drawn for the terminal, with no traceback and no logger name.
"""

from __future__ import annotations

import logging
import sys
import time

from phasesweep.runtime.terminal import (
    REPORT_LOGGER,
    ReportBlock,
    Style,
    paint,
    terminal_width,
    use_color,
    wrap,
)


class ConsoleHandler(logging.Handler):
    """Print a run's report lines and its warnings on stderr, and drop every other record.

    A record from the report logger prints as the time and its text, any later line indented
    under the first. A record at WARNING or above from any other logger prints as
    ``warning:`` or ``error:`` and its message, wrapped to the terminal with a hanging indent.
    Neither shows a traceback. Each record is one write under the handler's lock, so lines from
    parallel trials never interleave inside a record. The stream, its width and its colour
    decision are read at each emit, so a swapped ``sys.stderr`` is honoured.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write one record to stderr if the console shows it.

        :param logging.LogRecord record: The record being handled, under the handler's lock.
        """
        if record.name != REPORT_LOGGER and record.levelno < logging.WARNING:
            return
        try:
            stream = sys.stderr
            lines = self._lines(record, terminal_width(stream), use_color(stream))
            stream.write("\n".join(lines) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def _lines(self, record: logging.LogRecord, width: int, color: bool) -> list[str]:
        """Draw one record as the console prints it.

        :param logging.LogRecord record: The record to draw.
        :param int width: Terminal width in columns.
        :param bool color: Whether the stream takes styling.
        :return list[str]: The lines to print, the time before the first of a stamped record.
        """
        block = record.msg
        if isinstance(block, ReportBlock) and not block.timestamped:
            return block.render(width, color=color)
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created)) + "  "
        room = width - len(stamp)
        if isinstance(block, ReportBlock):
            body = block.render(room, color=color)
        elif record.name == REPORT_LOGGER:
            body = record.getMessage().splitlines() or [""]
        else:
            body = _diagnostic(record, room, color)
        return [
            ((stamp if index == 0 else " " * len(stamp)) + line).rstrip()
            for index, line in enumerate(body)
        ]


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
    lines[0] = paint(f"{label}:", style, color=color) + lines[0].removeprefix(f"{label}:")
    return lines
