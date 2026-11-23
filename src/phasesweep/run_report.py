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
from phasesweep.engine.confound import _flagged_checks
from phasesweep.engine.state import PhaseCounts, Winner
from phasesweep.evidence.models import WandbExtractor
from phasesweep.runtime.terminal import (
    GAP,
    INDENT,
    MIN_TEXT_WIDTH,
    REPORT_LOGGER,
    ReportBlock,
    Style,
    break_path,
    directory_lines,
    display_path,
    hang,
    objective_text,
    pack,
    paint,
    paint_word,
    param_text,
    plural,
    significant,
    wrap,
    wrap_labeled,
)

report = logging.getLogger(REPORT_LOGGER)

#: The word that marks a winner's line, and so the width of the column its facts hang under.
_WINNER_WORD = "winner"

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
    if len(lead + GAP.join([*head, *tail])) <= width:
        return [(lead + GAP.join([*head, *tail])).rstrip()]
    room = width - len(lead)
    return hang(lead, [*pack(head, room), *pack(tail, room)])


def _lay_out_clauses(label: str, label_width: int, clauses: Sequence[str], width: int) -> list[str]:
    """Lay out one labelled sentence, broken between its comma-separated clauses.

    :param str label: Text of the label column, a phase name.
    :param int label_width: Width the label column is padded to.
    :param Sequence[str] clauses: The sentence's clauses, in order.
    :param int width: Columns available to each line.
    :return list[str]: The lines, the label column blank after the first.
    """
    lead = f"{label:<{label_width}}{GAP}"
    return hang(lead, pack(clauses, width - len(lead), separator=", "))


def rerun_clause(reruns: int) -> str:
    """Say how many of a phase's trials rerun an interrupted trial or retry a failed grid cell.

    :param int reruns: Reruns among the trials to run, at least one.
    :return str: ``1 of them a rerun``, or ``3 of them reruns``.
    """
    return f"{reruns:,} of them {'a rerun' if reruns == 1 else 'reruns'}"


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
            clauses.append(rerun_clause(self.reruns))
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
class FactLine(ReportBlock):
    """A trial, a rerun or a phase winner: facts beside a phase name, and the cause under them."""

    phase: str
    label_width: int
    #: Facts that stay with the phase name's line.
    head: tuple[str, ...]
    #: Facts that move below the head when the whole line would overrun.
    tail: tuple[str, ...]
    reason: str | None
    #: A word of the first line and the style the console gives it, such as a failed trial's state.
    marked: tuple[str, Style] | None = None

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the facts beside the phase name or below it, and the cause under them.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the marked word.
        :return list[str]: The lines of the block.
        """
        lines = _lay_out(self.phase, self.label_width, self.head, self.tail, width)
        if self.marked is not None:
            word, style = self.marked
            lines[0] = paint_word(
                lines[0], word, style, start=self.label_width + len(GAP), color=color
            )
        if self.reason:
            pad = " " * (self.label_width + len(GAP))
            lines.extend(wrap_labeled("reason:", self.reason, width, indent=pad))
        return lines

    def __str__(self) -> str:
        """Return the facts on one line, the cause last.

        :return str: The one-line text.
        """
        facts = [self.phase, *self.head, *self.tail]
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
        return hang(lead, wrap_labeled("skipped:", self.why, width - len(lead)))

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
    head = [f"trial {winner.trial_number}", f"{metric_name} {objective_text(winner.metric)}"]
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
            f"range {objective_text(spread['min'])}..{objective_text(spread['max'])}",
            f"members {', '.join(str(member['trial_number']) for member in group['members'])}",
        ]
    return head, tail


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
    recovered: bool = False,
) -> FactLine:
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
    :param bool recovered: Whether a later run's recovery settled the trial, not its own run.
    :return FactLine: The block to log.
    """
    facts = [
        f"trial {number}",
        state,
        None if value is None else f"{experiment.metric.name} {objective_text(value)}",
        None if seconds is None else f"{seconds:.1f}s",
        f"{finished}/{phase.n_trials}",
    ]
    notes = [] if repeat is None else [f"repeat {repeat[0]}={param_text(repeat[1])}"]
    if rerun_of is not None:
        notes.append(f"rerun of {rerun_of}")
    if recovered:
        notes.append("recovered")
    style = _STATE_STYLES.get(state)
    return FactLine(
        phase.name,
        phase_label_width(experiment),
        head=tuple(fact for fact in facts if fact is not None),
        tail=(*(f"{key}={param_text(item)}" for key, item in params.items()), *notes),
        reason=reason,
        marked=None if style is None else (state, style),
    )


def trial_start_line(
    experiment: Experiment,
    phase: Phase,
    *,
    number: int,
    gpu: str | None,
    params: dict[str, object],
    repeat: tuple[str, object] | None,
    resumes: object,
) -> FactLine:
    """Build the line for a trial whose trainer is about to start.

    :param Experiment experiment: The experiment, for the label width.
    :param Phase phase: The trial's phase.
    :param int number: Trial number within the phase.
    :param str | None gpu: The devices the trainer sees, ``None`` or empty when it leases none.
    :param dict[str, object] params: Sampled configuration, repeat key excluded.
    :param tuple[str, object] | None repeat: Repeat key and value this trial runs, if any.
    :param object resumes: The interrupted trial this rerun continues in its directory, ``None``
        for a trial that starts in a directory of its own.
    :return FactLine: The block to log.
    """
    head = [f"trial {number}", "RUNNING", *([f"gpu {gpu}"] if gpu else [])]
    notes = [] if repeat is None else [f"repeat {repeat[0]}={param_text(repeat[1])}"]
    if resumes is not None:
        notes.append(f"resumes trial {resumes}")
    return FactLine(
        phase.name,
        phase_label_width(experiment),
        head=tuple(head),
        tail=(*(f"{key}={param_text(item)}" for key, item in params.items()), *notes),
        reason=None,
    )


def rerun_line(
    experiment: Experiment, phase: Phase, *, number: int, source: int, reason: str | None
) -> FactLine:
    """Build the line for a trial scheduled again.

    :param Experiment experiment: The experiment, for the label width.
    :param Phase phase: The rerun's phase.
    :param int number: Trial number of the rerun.
    :param int source: Trial number it repeats.
    :param str | None reason: First line of the source's recorded cause, if any.
    :return FactLine: The block to log.
    """
    head = (f"trial {number}", f"RERUN of trial {source}")
    return FactLine(phase.name, phase_label_width(experiment), head, (), reason)


def skipped_line(experiment: Experiment, phase: Phase, why: str) -> SkippedLine:
    """Build the line for a phase a resumed run does not execute.

    :param Experiment experiment: The experiment, for the label width.
    :param Phase phase: The skipped phase.
    :param str why: Where its winner comes from, or that it has none.
    :return SkippedLine: The block to log.
    """
    return SkippedLine(phase.name, phase_label_width(experiment), why)


def winner_line(experiment: Experiment, phase: Phase, winner: Winner) -> FactLine:
    """Build the line that closes a phase with its winner.

    :param Experiment experiment: The experiment, for the objective's name and the label width.
    :param Phase phase: The phase the winner belongs to.
    :param Winner winner: The selected winner.
    :return FactLine: The block to log.
    """
    head, tail = winner_facts(winner, experiment.metric.name)
    return FactLine(
        phase.name,
        phase_label_width(experiment),
        head=(f"{_WINNER_WORD}: {head[0]}", *head[1:]),
        tail=tuple(tail),
        reason=None,
        marked=(_WINNER_WORD, "winner"),
    )


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


def _value_lines(text: str, room: int, *, is_path: bool, color: bool) -> list[str]:
    """Fit one banner value to ``room`` columns, a path broken at slashes and prose wrapped.

    :param str text: The value to draw.
    :param int room: Columns left after the label column.
    :param bool is_path: Whether the value is a path, which is dim and breaks only at slashes.
    :param bool color: Whether to style a path.
    :return list[str]: The value's lines, without the label column.
    """
    if len(text) <= room:
        lines = [text]
    elif is_path:
        lines = break_path(text, room)
    else:
        lines = wrap(text, room)
    return [paint(line, "path", color=color) if is_path else line for line in lines]


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
        room = max(width - indent, MIN_TEXT_WIDTH)
        lines = [f"{paint('phasesweep run', 'label', color=color)}{GAP}{self.experiment}", ""]
        for row in self.head:
            lines.extend(self._row(row, room, color=color))
        name_width = max((len(phase.name) for phase in self.phases), default=0)
        plans = [
            line
            for phase in self.phases
            for line in _lay_out(phase.name, name_width, phase.facts[:1], phase.facts[1:], room)
        ]
        lines.extend(hang(_label_column("phases", color=color), plans, visible=indent))
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
        return hang(
            _label_column(row.label, color=color),
            _value_lines(row.text, room, is_path=row.is_path, color=color),
            visible=len(INDENT) + _BANNER_LABEL_WIDTH,
        )


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
    #: Confound checks the winner's selection flagged, in report order.
    flagged: tuple[str, ...] = ()


def _next_line(command: str, *, color: bool) -> str:
    """Draw the command a run closes by pointing to.

    A command is pasted, so it stays on one line whatever the width.

    :param str command: The command that reads the result.
    :param bool color: Whether to style the label.
    :return str: The ``Next:`` line.
    """
    return f"{paint('Next:', 'label', color=color)} {command}"


@dataclass(frozen=True)
class Summary:
    """What a finished run did and where it went: counts, winners, files and the next command."""

    headline: str
    phases: tuple[SummaryPhase, ...]
    written_root: str
    written: tuple[str, ...]
    next_command: str
    #: The ``show-winners`` command, as typed, that a flagged winner's line points to.
    show_winners_command: str

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
                drawn = [
                    pad + line
                    for line in _lay_out(
                        _WINNER_WORD, len(_WINNER_WORD), phase.winner, (), width - len(pad)
                    )
                ]
                drawn[0] = paint_word(drawn[0], _WINNER_WORD, "winner", start=len(pad), color=color)
                lines.extend(drawn)
                lines.extend(self._flagged_lines(phase, pad, width, color=color))
                lines.extend(
                    pad + line
                    for line in directory_lines(
                        phase.winner_dirs, width=width - len(pad), color=color
                    )
                )
        if self.written:
            lines.append("")
            lines.extend(self._written_lines(width, color=color))
        return [*lines, "", _next_line(self.next_command, color=color)]

    def _flagged_lines(
        self, phase: SummaryPhase, pad: str, width: int, *, color: bool
    ) -> list[str]:
        """Draw the confound checks a winner's selection flagged, under its facts.

        The one warning printed as the phase ended scrolls away in a long sweep, so the summary
        names the checks again and gives the command that prints the numbers and remedies
        behind them, on one line because it is pasted.

        :param SummaryPhase phase: The phase whose winner is drawn.
        :param str pad: Indent of the winner's label column.
        :param int width: Columns available to each line.
        :param bool color: Whether to style the labels.
        :return list[str]: The lines under the winner's text column, empty when none was flagged.
        """
        if not phase.flagged:
            return []
        indent = pad + " " * len(f"{_WINNER_WORD}{GAP}")
        lines = wrap_labeled("flagged:", ", ".join(phase.flagged), width - len(indent))
        lines[0] = paint_word(lines[0], "flagged:", "warning", color=color)
        details = f"{paint('details:', 'label', color=color)} {self.show_winners_command}"
        return [indent + line for line in (*lines, details)]

    def _headline_lines(self, width: int, *, color: bool) -> list[str]:
        """Wrap the headline, styling its opening words.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the opening words.
        :return list[str]: The headline's lines.
        """
        lines = wrap(self.headline, width)
        lines[0] = paint_word(lines[0], self.headline.split(":")[0], "label", color=color)
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
        drawn[0] = paint_word(drawn[0], "Written under", "label", color=color)
        return [
            *drawn,
            *(INDENT + paint(path, "path", color=color) for path in self.written),
        ]


@dataclass(frozen=True)
class NothingToRun:
    """The sentence a run closes with when every trial was already finished and nothing changed."""

    elapsed: str
    trials: int
    generation_id: str
    next_command: str

    def _text(self) -> str:
        """Say that the run launched nothing and wrote nothing.

        :return str: The sentence.
        """
        return (
            f"Finished in {self.elapsed}: nothing to run, {plural(self.trials, 'trial')} already "
            f"finished, and generation {self.generation_id} already publishes these winners."
        )

    def render(self, width: int, *, color: bool) -> list[str]:
        """Draw the sentence wrapped to the terminal, then the command that reads the result.

        :param int width: Columns available to each line.
        :param bool color: Whether to style the opening words and the label.
        :return list[str]: A blank line, the sentence, a blank line and the ``Next:`` line.
        """
        lines = wrap(self._text(), width)
        lines[0] = paint_word(lines[0], f"Finished in {self.elapsed}", "label", color=color)
        return ["", *lines, "", _next_line(self.next_command, color=color)]


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
        taken = len(label) + 1
        values = _value_lines(
            self.path, max(width - taken, MIN_TEXT_WIDTH), is_path=True, color=color
        )
        return hang(paint(label, "label", color=color) + " ", values, visible=taken)


def _count_clauses(counts: PhaseCounts) -> tuple[str, ...]:
    """List a phase's nonzero trial counts, such as ``3 complete`` and ``1 failed``.

    The clauses add up to the trials that count toward ``n_trials``. Interrupted trials never
    count, so they follow in parentheses on the last clause, as the ``status --trials`` heading
    names them: ``3 complete, 1 failed (1 interrupted)``.

    :param PhaseCounts counts: The phase's counts.
    :return tuple[str, ...]: One clause per nonzero counted state, or ``no trials`` when there
        are none, the last clause carrying any interrupted trials.
    """
    named = [
        (counts.complete, "complete"),
        (counts.failed, "failed"),
        (counts.infeasible, "infeasible"),
    ]
    clauses = [f"{count} {name}" for count, name in named if count] or ["no trials"]
    if counts.interrupted:
        clauses[-1] += f" ({counts.interrupted} interrupted)"
    return tuple(clauses)


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
    show_winners_command: str,
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
    :param str show_winners_command: The ``show-winners`` command a flagged winner points to.
    :return Summary: The block to log.
    """
    reached = {phase.name for _index, phase in experiment.phases_from(from_phase)}
    phases = []
    for phase in experiment.phases:
        winner = winners.get(phase.name)
        if phase.name not in reached:
            clauses: tuple[str, ...] = ("skipped",)
        elif phase.name in counts:
            clauses = _count_clauses(counts[phase.name])
        else:
            clauses = ("trial counts unavailable",)
        facts: tuple[str, ...] = ()
        directories: tuple[str, ...] = ()
        flagged: tuple[str, ...] = ()
        if winner is not None:
            head, tail = winner_facts(winner, experiment.metric.name)
            facts = (*head, *tail)
            if winner.confound is not None:
                flagged = tuple(_flagged_checks(winner.confound))
            if phase.name in counts:
                directories = tuple(display_path(path) for path in counts[phase.name].winner_dirs)
        phases.append(SummaryPhase(phase.name, clauses, facts, directories, flagged))
    headline = f"Finished in {duration_text(elapsed)}: {run_totals(counts)}"
    return Summary(
        headline, tuple(phases), written_root, tuple(written), next_command, show_winners_command
    )


def _trial_total(counts: Mapping[str, PhaseCounts]) -> int:
    """Sum the trials of every phase that count toward its ``n_trials``.

    :param Mapping[str, PhaseCounts] counts: Trial counts of each phase.
    :return int: The counted trials of all the phases; interrupted ones are not among them.
    """
    return sum(item.counted for item in counts.values())


def run_totals(counts: Mapping[str, PhaseCounts]) -> str:
    """Count the phases and trials a run covered, as the summary headline and ``run.log`` say it.

    The trials are those that count toward each phase's ``n_trials``, as the ``status --trials``
    headings count them, and the previous run's share is the part of them it recorded.

    :param Mapping[str, PhaseCounts] counts: Trial counts of each phase the run executed.
    :return str: For example ``2 phases, 6 trials, 4 from a previous run``.
    """
    earlier = sum(item.earlier for item in counts.values())
    return (
        f"{plural(len(counts), 'phase')}, {plural(_trial_total(counts), 'trial')}, "
        f"{earlier} from a previous run"
    )


def nothing_to_run(
    *, elapsed: float, counts: Mapping[str, PhaseCounts], generation_id: str, next_command: str
) -> NothingToRun:
    """Build the closing block of a run that replayed a published result unchanged.

    :param float elapsed: Wall-clock seconds the engine took.
    :param Mapping[str, PhaseCounts] counts: Trial counts of each phase the run replayed.
    :param str generation_id: The generation that already publishes the winners.
    :param str next_command: The command that reads the result.
    :return NothingToRun: The block to log.
    """
    return NothingToRun(duration_text(elapsed), _trial_total(counts), generation_id, next_command)
