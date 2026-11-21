"""The report of a ``phasesweep run``: the facts a person reads on the console, as blocks.

Each block is one fact of the run, held as plain values. The progress blocks (phase start, trial,
rerun, skipped phase and winner) are logged on ``phasesweep.report``: ``str()`` is the one-line
text that ``run.log`` and captured records carry, and ``render`` draws the same facts for a
terminal of a given width, styled only when the caller says so. The banner, the closing summary,
the replay sentence and the log line are written to stderr by the CLI and reach no log file, so
they have ``render`` alone. Nothing here touches a stream or a clock, so a block's text is a pure function
of its fields.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import phase_draws
from phasesweep.engine.state import PhaseCounts, Winner
from phasesweep.evidence.models import WandbExtractor
from phasesweep.runtime.terminal import (
    GAP,
    INDENT,
    REPORT_LOGGER,
    ReportBlock,
    Style,
    break_path,
    display_path,
    pack,
    paint,
    param_text,
    plural,
    significant,
    wrap,
)

report = logging.getLogger(REPORT_LOGGER)

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
    lead = f"{label:<{label_width}}{GAP}"
    pad = " " * len(lead)
    if len(lead + GAP.join([*head, *tail])) <= width:
        return [(lead + GAP.join([*head, *tail])).rstrip()]
    if len(lead + GAP.join(head)) <= width:
        return [lead + GAP.join(head), *pack(tail, width, indent=pad)]
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
    lead = f"{label:<{label_width}}{GAP}"
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
    pad = " " * (label_width + len(GAP))
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
        return f"{self.phase}{GAP}{', '.join(self._clauses())}"


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
                lines[0], self.state, style, start=self.label_width + len(GAP), color=color
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
        return GAP.join(facts)


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
        return GAP.join(facts)


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
        lead = f"{self.phase:<{self.label_width}}{GAP}"
        return wrap(f"skipped: {self.why}", width, indent=lead, subsequent_indent=" " * len(lead))

    def __str__(self) -> str:
        """Return the phase name and why it is skipped on one line.

        :return str: The one-line text.
        """
        return f"{self.phase}{GAP}skipped: {self.why}"


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
            lines[0], "winner", "winner", start=self.label_width + len(GAP), color=color
        )
        return lines

    def __str__(self) -> str:
        """Return the winner's facts on one line.

        :return str: The one-line text.
        """
        return GAP.join([self.phase, f"winner: {self.head[0]}", *self.head[1:], *self.tail])


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


#: Width of the label column in the banner, after its two-column indent.
_BANNER_LABEL_WIDTH = 10


@dataclass(frozen=True)
class BannerRow:
    """One labelled value of the banner, drawn dim and broken at slashes when it is a path."""

    label: str
    text: str
    is_path: bool = False


@dataclass(frozen=True)
class BannerPhase:
    """One phase of the banner: its name and the facts of what it will run."""

    name: str
    facts: tuple[str, ...]


def _label_column(label: str, *, color: bool) -> str:
    """Pad a banner label to its column and style it.

    :param str label: The label text.
    :param bool color: Whether to style the label.
    :return str: The indent, the label padded to the column and its styling.
    """
    return INDENT + paint(label, "label", color=color) + " " * (_BANNER_LABEL_WIDTH - len(label))


def _value_lines(row: BannerRow, room: int, *, color: bool) -> list[str]:
    """Fit one banner value to ``room`` columns, a path broken at slashes and prose wrapped.

    :param BannerRow row: The row whose value is drawn.
    :param int room: Columns left after the label column.
    :param bool color: Whether to style a path.
    :return list[str]: The value's lines, without the label column.
    """
    if len(row.text) <= room:
        lines = [row.text]
    elif row.is_path:
        lines = break_path(row.text, room)
    else:
        lines = wrap(row.text, room)
    return [paint(line, "path", color=color) if row.is_path else line for line in lines]


@dataclass(frozen=True)
class Banner:
    """What a run is about to do: the experiment, its phases and where everything goes."""

    experiment: str
    head: tuple[BannerRow, ...]
    phases: tuple[BannerPhase, ...]
    tail: tuple[BannerRow, ...]

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the banner as labelled rows under its title.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the labels and paths.
        :return list[str]: The lines, ending with a blank line that separates the progress.
        """
        indent = len(INDENT) + _BANNER_LABEL_WIDTH
        room = max(width - indent, 20)
        lines = [f"{paint('phasesweep run', 'label', color=color)}{GAP}{self.experiment}", ""]
        for row in self.head:
            lines.extend(self._row(row, room, color=color))
        label = _label_column("phases", color=color)
        name_width = max((len(phase.name) for phase in self.phases), default=0)
        for number, phase in enumerate(self.phases):
            drawn = _lay_out(phase.name, name_width, phase.facts[:1], phase.facts[1:], room)
            lines.extend(
                (label if number == 0 and index == 0 else " " * indent) + line
                for index, line in enumerate(drawn)
            )
        for row in self.tail:
            lines.extend(self._row(row, room, color=color))
        return [*lines, ""]

    @staticmethod
    def _row(row: BannerRow, room: int, *, color: bool) -> list[str]:
        """Draw one labelled value, its label on the first line only.

        :param BannerRow row: The row to draw.
        :param int room: Columns left after the label column.
        :param bool color: Whether to style the label and a path.
        :return list[str]: The row's lines.
        """
        values = _value_lines(row, room, color=color)
        pad = " " * (len(INDENT) + _BANNER_LABEL_WIDTH)
        return [
            (_label_column(row.label, color=color) if index == 0 else pad) + value
            for index, value in enumerate(values)
        ]


def _gpu_fact(phase: Phase) -> str:
    """Name a phase's GPU policy with the device list the config sets, if any.

    :param Phase phase: The phase.
    :return str: For example ``gpu none`` or ``gpu single_per_trial ids 0,1``.
    """
    if phase.gpu_ids is not None:
        return f"gpu {phase.gpu_policy} ids {','.join(str(index) for index in phase.gpu_ids)}"
    if phase.gpu_devices is not None:
        return f"gpu {phase.gpu_policy} devices {','.join(phase.gpu_devices)}"
    return f"gpu {phase.gpu_policy}"


def _trackers(experiment: Experiment) -> str:
    """Name the services a run reads results from, MLflow and W&B, or ``none``.

    Only the target is named, never a credential or an environment value.

    :param Experiment experiment: The experiment.
    :return str: The services, comma-separated, or ``none``.
    """
    names = []
    if experiment.mlflow is not None:
        names.append(f"mlflow {experiment.mlflow.tracking_uri}")
    extractors = [
        experiment.metric.extractor,
        *(item.extractor for item in [*experiment.constraints, *experiment.observe]),
    ]
    for extractor in extractors:
        if isinstance(extractor, WandbExtractor):
            name = f"wandb {extractor.entity}/{extractor.project}"
            if name not in names:
                names.append(name)
    return ", ".join(names) or "none"


def banner(
    experiment: Experiment,
    *,
    config_path: str,
    workdir: str,
    log_path: str,
    from_phase: str | None,
) -> Banner:
    """Build the banner a live run opens with.

    :param Experiment experiment: The experiment about to run.
    :param str config_path: The config file, as the console shows it.
    :param str workdir: The directory the run writes its artifacts under, as shown.
    :param str log_path: Absolute path of ``run.log``.
    :param str | None from_phase: The phase ``--from-phase`` starts at, if given.
    :return Banner: The block to log.
    """
    storage = "none (in memory)" if experiment.storage is None else experiment.storage
    tail = [BannerRow("from", from_phase)] if from_phase is not None else []
    tail += [
        BannerRow("storage", storage),
        BannerRow("tracker", _trackers(experiment)),
        BannerRow("workdir", workdir, is_path=True),
        BannerRow("log", log_path, is_path=True),
    ]
    phases = tuple(
        BannerPhase(
            phase.name,
            (
                plural(phase.n_trials, "trial"),
                phase.sampler.type if phase_draws(phase) else "fixed",
                f"{phase.n_jobs} at a time",
                _gpu_fact(phase),
            ),
        )
        for phase in experiment.phases
    )
    return Banner(
        experiment.experiment,
        (BannerRow("config", config_path, is_path=True),),
        phases,
        tuple(tail),
    )


def duration_text(seconds: float) -> str:
    """Format an elapsed time: tenths of a second under a minute, then minutes and hours.

    :param float seconds: Elapsed wall-clock seconds.
    :return str: For example ``3.1s``, ``2m05s`` or ``1h02m05s``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{rest:02d}s"
    return f"{minutes}m{rest:02d}s"


@dataclass(frozen=True)
class SummaryPhase:
    """One phase of the closing summary: how its trials ended and what won."""

    name: str
    counts: tuple[str, ...]
    winner: tuple[str, ...]
    #: Directory of each trial the winner stands on, as the console shows it.
    winner_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Summary:
    """What a finished run did and where it went: counts, winners, files and the next command."""

    headline: str
    phases: tuple[SummaryPhase, ...]
    written_root: str
    written: tuple[str, ...]
    next_command: str

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the summary in sections separated by blank lines.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the labels, the winner mark and the paths.
        :return list[str]: The lines, starting with a blank line that separates the progress.
        """
        name_width = max((len(phase.name) for phase in self.phases), default=0)
        pad = INDENT + " " * (name_width + len(GAP))
        lines = ["", *self._headline_lines(width, color=color), ""]
        for phase in self.phases:
            drawn_counts = _lay_out_clauses(
                phase.name, name_width, phase.counts, width - len(INDENT)
            )
            lines.extend(INDENT + line for line in drawn_counts)
            if phase.winner:
                lead = f"{pad}winner{GAP}"
                drawn = pack(phase.winner, width, indent=" " * len(lead))
                drawn[0] = lead + drawn[0].removeprefix(" " * len(lead))
                lines.append(_paint_word(drawn[0], "winner", "winner", start=len(pad), color=color))
                lines.extend(drawn[1:])
                for directory in phase.winner_dirs:
                    lines.extend(
                        pad + paint(line.removeprefix(pad), "path", color=color)
                        for line in break_path(directory, width, indent=pad)
                    )
        if self.written:
            lines.append("")
            lines.extend(self._written_lines(width, color=color))
        # A command is pasted, so it stays on one line whatever the width, as the log path does.
        following = f"{paint('Next:', 'label', color=color)} {self.next_command}"
        return [*lines, "", following]

    def _headline_lines(self, width: int, *, color: bool) -> list[str]:
        """Wrap the headline, styling its opening words.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the opening words.
        :return list[str]: The headline's lines.
        """
        lines = wrap(self.headline, width)
        opening = self.headline.split(":")[0]
        lines[0] = paint(opening, "label", color=color) + lines[0].removeprefix(opening)
        return lines

    def _written_lines(self, width: int, *, color: bool) -> list[str]:
        """Draw the files the run wrote, relative to the directory they are under.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the label and the paths.
        :return list[str]: The section's lines.
        """
        head = f"Written under {self.written_root}"
        if len(head) > width:
            drawn = [
                "Written under",
                *break_path(self.written_root, width, indent=INDENT),
            ]
        else:
            drawn = [head]
        drawn[0] = paint("Written under", "label", color=color) + drawn[0].removeprefix(
            "Written under"
        )
        return [
            *drawn,
            *(INDENT + paint(path, "path", color=color) for path in self.written),
        ]


@dataclass(frozen=True)
class NothingToRun:
    """The one line a run closes with when every trial was already finished and nothing changed."""

    elapsed: str
    trials: int
    generation_id: str

    def _text(self) -> str:
        """Say that the run launched nothing and wrote nothing.

        :return str: The sentence.
        """
        return (
            f"Finished in {self.elapsed}: nothing to run, {plural(self.trials, 'trial')} already "
            f"finished, and generation {self.generation_id} already publishes these winners."
        )

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the sentence wrapped to the terminal.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the opening words.
        :return list[str]: A blank line, then the sentence.
        """
        lines = wrap(self._text(), width)
        opening = f"Finished in {self.elapsed}"
        lines[0] = paint(opening, "label", color=color) + lines[0].removeprefix(opening)
        return ["", *lines]


@dataclass(frozen=True)
class LogLine:
    """The last line of a run: where ``run.log`` is, its path broken at slashes to fit."""

    path: str

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the label and the path, any later line indented under the path.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the label and the path.
        :return list[str]: The lines of the log line.
        """
        label = "log:"
        pad = " " * (len(label) + 1)
        values = _value_lines(
            BannerRow("log", self.path, is_path=True), max(width - len(pad), 20), color=color
        )
        return [
            (paint(label, "label", color=color) + " " if index == 0 else pad) + value
            for index, value in enumerate(values)
        ]


def _count_clauses(counts: PhaseCounts) -> tuple[str, ...]:
    """List a phase's nonzero trial counts, such as ``3 complete`` and ``1 failed``.

    :param PhaseCounts counts: The phase's counts.
    :return tuple[str, ...]: One clause per nonzero count, or ``no trials`` when all are zero.
    """
    named = [
        (counts.complete, "complete"),
        (counts.failed, "failed"),
        (counts.infeasible, "infeasible"),
        (counts.interrupted, "interrupted"),
    ]
    return tuple(f"{count} {name}" for count, name in named if count) or ("no trials",)


def summary(
    experiment: Experiment,
    *,
    elapsed: float,
    counts: Mapping[str, PhaseCounts],
    winners: Mapping[str, Winner],
    from_phase: str | None,
    written_root: str,
    written: Sequence[str],
    next_command: str,
) -> Summary:
    """Build the summary a successful run closes with.

    :param Experiment experiment: The experiment that ran.
    :param float elapsed: Wall-clock seconds the engine took.
    :param Mapping[str, PhaseCounts] counts: Trial counts of each phase the run executed.
    :param Mapping[str, Winner] winners: Every phase's winner.
    :param str | None from_phase: The phase ``--from-phase`` started at; earlier phases are
        reported as skipped.
    :param str written_root: The directory the written files are under, as shown.
    :param Sequence[str] written: The files this run wrote, relative to ``written_root``.
    :param str next_command: The command that reads the result.
    :return Summary: The block to log.
    """
    names = [phase.name for phase in experiment.phases]
    skipped = set(names[: names.index(from_phase)]) if from_phase in names else set()
    phases = []
    for phase in experiment.phases:
        winner = winners.get(phase.name)
        if phase.name in skipped:
            clauses: tuple[str, ...] = ("skipped",)
        elif phase.name in counts:
            clauses = _count_clauses(counts[phase.name])
        else:
            clauses = ("trial counts unavailable",)
        facts: tuple[str, ...] = ()
        directories: tuple[str, ...] = ()
        if winner is not None:
            head, tail = winner_facts(winner, experiment.metric.name)
            facts = (*head, *tail)
            if phase.name in counts:
                directories = tuple(display_path(path) for path in counts[phase.name].winner_dirs)
        phases.append(SummaryPhase(phase.name, clauses, facts, directories))
    trials = sum(item.total for item in counts.values())
    earlier = sum(item.earlier for item in counts.values())
    headline = (
        f"Finished in {duration_text(elapsed)}: {plural(len(counts), 'phase')}, "
        f"{plural(trials, 'trial')}, {earlier} from a previous run"
    )
    return Summary(headline, tuple(phases), written_root, tuple(written), next_command)


def nothing_to_run(
    *, elapsed: float, counts: Mapping[str, PhaseCounts], generation_id: str
) -> NothingToRun:
    """Build the closing line of a run that replayed a published result unchanged.

    :param float elapsed: Wall-clock seconds the engine took.
    :param Mapping[str, PhaseCounts] counts: Trial counts of each phase the run replayed.
    :param str generation_id: The generation that already publishes the winners.
    :return NothingToRun: The block to log.
    """
    trials = sum(item.total for item in counts.values())
    return NothingToRun(duration_text(elapsed), trials, generation_id)
