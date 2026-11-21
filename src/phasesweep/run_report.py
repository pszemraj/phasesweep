"""The report of a ``phasesweep run``: the facts a person reads on the console, as log messages.

Each block is one fact of the run, held as plain values and logged on ``phasesweep.report``.
``str()`` is the one-line text that ``run.log`` and captured records carry, and ``render`` draws
the same facts for a terminal of a given width, styled only when the caller says so. Nothing here
touches a stream or a clock, so a block's text is a pure function of its fields.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.state import Winner
from phasesweep.runtime.terminal import (
    REPORT_LOGGER,
    ReportBlock,
    Style,
    pack,
    paint,
    param_text,
    plural,
    significant,
    wrap,
)

report = logging.getLogger(REPORT_LOGGER)

#: Separator of the facts on one line, and of the columns that follow a phase name.
_GAP = "  "

#: How the console styles a trial state that needs attention; other states print plain.
_STATE_STYLES: dict[str, Style] = {"FAIL": "error", "INFEASIBLE": "error", "INTERRUPTED": "warning"}


def phase_label_width(experiment: Experiment) -> int:
    """Return the width of the phase-name column every progress line shares.

    :param Experiment experiment: The experiment whose phases the report names.
    :return int: The longest configured phase name.
    """
    return max(len(phase.name) for phase in experiment.phases)


def _lay_out(
    label: str, label_width: int, head: Sequence[str], tail: Sequence[str], width: int
) -> list[str]:
    """Lay out one labelled fact: the head first, the tail beside it or packed below.

    The tail joins the first line when the whole line fits ``width``; otherwise it moves to
    lines under the text after the label. A head that alone overruns the line is packed onto
    several, so a fact is never split.

    :param str label: Text of the label column, a phase name.
    :param int label_width: Width the label column is padded to.
    :param Sequence[str] head: Facts that come first, on the label's line where they fit.
    :param Sequence[str] tail: Facts that move below when the first line would overrun.
    :param int width: Columns available to each line.
    :return list[str]: The lines, the label column blank after the first.
    """
    lead = f"{label:<{label_width}}{_GAP}"
    pad = " " * len(lead)
    if len(lead + _GAP.join([*head, *tail])) <= width:
        return [(lead + _GAP.join([*head, *tail])).rstrip()]
    if len(lead + _GAP.join(head)) <= width:
        return [lead + _GAP.join(head), *pack(tail, width, indent=pad)]
    lines = [*pack(head, width, indent=pad), *pack(tail, width, indent=pad)]
    lines[0] = lead + lines[0].removeprefix(pad)
    return lines


def _lay_out_clauses(label: str, label_width: int, clauses: Sequence[str], width: int) -> list[str]:
    """Lay out one labelled sentence, broken between its comma-separated clauses.

    :param str label: Text of the label column, a phase name.
    :param int label_width: Width the label column is padded to.
    :param Sequence[str] clauses: The sentence's clauses, in order.
    :param int width: Columns available to each line.
    :return list[str]: The lines, the label column blank after the first.
    """
    lead = f"{label:<{label_width}}{_GAP}"
    pad = " " * len(lead)
    lines = pack(clauses, width, indent=pad, separator=", ")
    lines[0] = lead + lines[0].removeprefix(pad)
    return lines


def _paint_word(line: str, word: str, style: Style, *, start: int, color: bool) -> str:
    """Style the first ``word`` found at or after column ``start`` of an already laid-out line.

    :param str line: Plain text of one line.
    :param str word: Text to style.
    :param Style style: Style to apply.
    :param int start: Column to search from, so a phase name never matches.
    :param bool color: Whether styling is on.
    :return str: The line with ``word`` styled, or unchanged when absent or colour is off.
    """
    index = line.find(word, start)
    if not color or index < 0:
        return line
    return line[:index] + paint(word, style, color=True) + line[index + len(word) :]


def _reason_lines(reason: str, label_width: int, width: int) -> list[str]:
    """Wrap a trial's cause under the text of its line.

    :param str reason: The cause, first line of the recorded outcome.
    :param int label_width: Width of the label column the text is indented past.
    :param int width: Columns available to each line.
    :return list[str]: The wrapped lines, each indented past the label column.
    """
    pad = " " * (label_width + len(_GAP))
    return wrap(f"reason: {reason}", width, indent=pad, subsequent_indent=pad)


@dataclass(frozen=True)
class PhaseStart(ReportBlock):
    """The line that opens a phase: how much is left to run, or that it replays a result."""

    phase: str
    label_width: int
    remaining: int
    finished: int
    reruns: int
    n_jobs: int

    def _clauses(self) -> list[str]:
        """Say what the phase is about to do, one clause per fact.

        :return list[str]: The to-run count and parallelism, with finished and rerun counts when
            nonzero, or that the phase replays its recorded result.
        """
        if self.remaining == 0:
            return [
                "nothing to run",
                f"{plural(self.finished, 'trial')} already finished",
                "replaying the recorded result",
            ]
        clauses = [f"{plural(self.remaining, 'trial')} to run"]
        if self.reruns:
            clauses.append(
                "1 of them a rerun" if self.reruns == 1 else f"{self.reruns} of them reruns"
            )
        clauses.append(f"{self.n_jobs} at a time")
        if self.finished:
            clauses.append(f"{self.finished} already finished")
        return clauses

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the phase name and its plan, broken between clauses under the text.

        :param int width: Columns available to each line.
        :param bool color: Unused; the line carries no styling.
        :return list[str]: The lines of the plan.
        """
        return _lay_out_clauses(self.phase, self.label_width, self._clauses(), width)

    def __str__(self) -> str:
        """Return the phase name and its plan on one line.

        :return str: The one-line text.
        """
        return f"{self.phase}{_GAP}{', '.join(self._clauses())}"


@dataclass(frozen=True)
class TrialLine(ReportBlock):
    """One finished trial: its state, objective, time, place in the phase and configuration."""

    phase: str
    label_width: int
    number: int
    state: str
    objective: str | None
    duration: str | None
    ordinal: str
    params: tuple[str, ...]
    notes: tuple[str, ...]
    reason: str | None

    def _head(self) -> list[str]:
        """Collect the facts that always share the first line.

        :return list[str]: Trial number, state, objective, duration and place in the phase.
        """
        facts = [f"trial {self.number}", self.state, self.objective, self.duration, self.ordinal]
        return [fact for fact in facts if fact is not None]

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the trial with its configuration beside it or below, and its cause under it.

        :param int width: Columns available to each line.
        :param bool color: Whether to style a failed, infeasible or interrupted state.
        :return list[str]: The lines of the trial.
        """
        lines = _lay_out(
            self.phase, self.label_width, self._head(), [*self.params, *self.notes], width
        )
        style = _STATE_STYLES.get(self.state)
        if style is not None:
            lines[0] = _paint_word(
                lines[0], self.state, style, start=self.label_width + len(_GAP), color=color
            )
        if self.reason:
            lines.extend(_reason_lines(self.reason, self.label_width, width))
        return lines

    def __str__(self) -> str:
        """Return the trial's facts on one line, its cause last.

        :return str: The one-line text.
        """
        facts = [self.phase, *self._head(), *self.params, *self.notes]
        if self.reason:
            facts.append(f"reason: {self.reason}")
        return _GAP.join(facts)


@dataclass(frozen=True)
class RerunLine(ReportBlock):
    """A trial the run schedules again, and the cause of the trial it repeats."""

    phase: str
    label_width: int
    number: int
    source: int
    reason: str | None

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the rerun and the cause of its source under it.

        :param int width: Columns available to each line.
        :param bool color: Unused; the line carries no styling.
        :return list[str]: The lines of the rerun.
        """
        lines = _lay_out(
            self.phase,
            self.label_width,
            [f"trial {self.number}", f"RERUN of trial {self.source}"],
            [],
            width,
        )
        if self.reason:
            lines.extend(_reason_lines(self.reason, self.label_width, width))
        return lines

    def __str__(self) -> str:
        """Return the rerun and its cause on one line.

        :return str: The one-line text.
        """
        facts = [self.phase, f"trial {self.number}", f"RERUN of trial {self.source}"]
        if self.reason:
            facts.append(f"reason: {self.reason}")
        return _GAP.join(facts)


@dataclass(frozen=True)
class SkippedLine(ReportBlock):
    """A phase a resumed run does not execute, and where its winner comes from."""

    phase: str
    label_width: int
    why: str

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the phase name and why it is skipped.

        :param int width: Columns available to each line.
        :param bool color: Unused; the line carries no styling.
        :return list[str]: The wrapped lines.
        """
        lead = f"{self.phase:<{self.label_width}}{_GAP}"
        return wrap(f"skipped: {self.why}", width, indent=lead, subsequent_indent=" " * len(lead))

    def __str__(self) -> str:
        """Return the phase name and why it is skipped on one line.

        :return str: The one-line text.
        """
        return f"{self.phase}{_GAP}skipped: {self.why}"


def winner_facts(winner: Winner, metric_name: str) -> tuple[list[str], list[str]]:
    """Split a phase winner into the facts that stay with its line and those that may move below.

    The winner's trial, objective and an accepted partial result make the head. The sampled
    configuration and, for a repeats phase, the reducer, spread and member trials make the tail.

    :param Winner winner: The phase's selected winner.
    :param str metric_name: Name the objective is shown under.
    :return tuple[list[str], list[str]]: The head and the tail.
    """
    head = [f"trial {winner.trial_number}", f"{metric_name} {significant(winner.metric, 6)}"]
    completion = winner.completion
    if completion.get("incomplete"):
        head += [
            f"INCOMPLETE {completion['finished_trials']}/{completion['requested_trials']} finished",
            f"timeout scope {completion['timeout_scope']}",
        ]
    tail = [f"{key}={param_text(value)}" for key, value in winner.params.items()]
    group = winner.repeats
    if group is not None:
        spread = group["spread"]
        tail += [
            f"{group['reduce']} of {spread['n']} repeats of {group['key']}",
            f"sd {significant(spread['sd'], 4)}",
            f"range {significant(spread['min'], 6)}..{significant(spread['max'], 6)}",
            f"members {', '.join(str(member['trial_number']) for member in group['members'])}",
        ]
    return head, tail


@dataclass(frozen=True)
class WinnerLine(ReportBlock):
    """The winner a phase closes with: its trial, objective, configuration and any spread."""

    phase: str
    label_width: int
    head: tuple[str, ...]
    tail: tuple[str, ...]

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the winner, its configuration beside it or below.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the word ``winner``.
        :return list[str]: The lines of the winner.
        """
        lines = _lay_out(
            self.phase,
            self.label_width,
            [f"winner: {self.head[0]}", *self.head[1:]],
            self.tail,
            width,
        )
        lines[0] = _paint_word(
            lines[0], "winner", "winner", start=self.label_width + len(_GAP), color=color
        )
        return lines

    def __str__(self) -> str:
        """Return the winner's facts on one line.

        :return str: The one-line text.
        """
        return _GAP.join([self.phase, f"winner: {self.head[0]}", *self.head[1:], *self.tail])


def phase_start(
    experiment: Experiment, phase: Phase, *, finished: int, remaining: int, reruns: int
) -> PhaseStart:
    """Build the line that opens a phase of a live run.

    :param Experiment experiment: The experiment, for the width of the phase-name column.
    :param Phase phase: The phase about to run or replay.
    :param int finished: Trials that count toward its target already.
    :param int remaining: Trials still needed to reach ``phase.n_trials``.
    :param int reruns: Reruns among those, for interrupted trials and retried grid cells.
    :return PhaseStart: The block to log.
    """
    return PhaseStart(
        phase.name, phase_label_width(experiment), remaining, finished, reruns, phase.n_jobs
    )


def trial_line(
    experiment: Experiment,
    phase: Phase,
    *,
    number: int,
    state: str,
    value: float | None,
    seconds: float | None,
    finished: int,
    params: dict[str, object],
    repeat: tuple[str, object] | None,
    rerun_of: object,
    reason: str | None,
) -> TrialLine:
    """Build the line for one finished trial from its recorded values.

    :param Experiment experiment: The experiment, for the objective's name and the label width.
    :param Phase phase: The trial's phase.
    :param int number: Trial number within the phase.
    :param str state: The state the trial is counted under.
    :param float | None value: The objective, ``None`` when the trial produced none.
    :param float | None seconds: Trainer wall-clock seconds, ``None`` when none were recorded.
    :param int finished: The trial's place in the phase's completion order.
    :param dict[str, object] params: Sampled configuration, repeat key excluded.
    :param tuple[str, object] | None repeat: Repeat key and value this trial ran, if any.
    :param object rerun_of: Trial number this one reruns, ``None`` for a first attempt.
    :param str | None reason: First line of the recorded cause, if any.
    :return TrialLine: The block to log.
    """
    notes = [] if repeat is None else [f"repeat {repeat[0]}={param_text(repeat[1])}"]
    if rerun_of is not None:
        notes.append(f"rerun of {rerun_of}")
    return TrialLine(
        phase=phase.name,
        label_width=phase_label_width(experiment),
        number=number,
        state=state,
        objective=None if value is None else f"{experiment.metric.name} {significant(value, 6)}",
        duration=None if seconds is None else f"{seconds:.1f}s",
        ordinal=f"{finished}/{phase.n_trials}",
        params=tuple(f"{key}={param_text(item)}" for key, item in params.items()),
        notes=tuple(notes),
        reason=reason,
    )


def rerun_line(
    experiment: Experiment, phase: Phase, *, number: int, source: int, reason: str | None
) -> RerunLine:
    """Build the line for a trial scheduled again.

    :param Experiment experiment: The experiment, for the label width.
    :param Phase phase: The rerun's phase.
    :param int number: Trial number of the rerun.
    :param int source: Trial number it repeats.
    :param str | None reason: First line of the source's recorded cause, if any.
    :return RerunLine: The block to log.
    """
    return RerunLine(phase.name, phase_label_width(experiment), number, source, reason)


def skipped_line(experiment: Experiment, phase: Phase, why: str) -> SkippedLine:
    """Build the line for a phase a resumed run does not execute.

    :param Experiment experiment: The experiment, for the label width.
    :param Phase phase: The skipped phase.
    :param str why: Where its winner comes from, or that it has none.
    :return SkippedLine: The block to log.
    """
    return SkippedLine(phase.name, phase_label_width(experiment), why)


def winner_line(experiment: Experiment, phase: Phase, winner: Winner) -> WinnerLine:
    """Build the line that closes a phase with its winner.

    :param Experiment experiment: The experiment, for the objective's name and the label width.
    :param Phase phase: The phase the winner belongs to.
    :param Winner winner: The selected winner.
    :return WinnerLine: The block to log.
    """
    head, tail = winner_facts(winner, experiment.metric.name)
    return WinnerLine(phase.name, phase_label_width(experiment), tuple(head), tuple(tail))
