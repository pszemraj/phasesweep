"""Plain-text trial tables for ``phasesweep status --trials``.

Each phase prints one aligned table, one row per trial, with a column per swept parameter and the
winner named in its own leading column. Tables print at their natural width and are never
reflowed; only prose (headings, notes and a trial's failure reason) wraps to the terminal width.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import CategoricalParam, SearchParam, sampler_search_space
from phasesweep.engine.evidence import pruned_trials_note
from phasesweep.engine.paths import TRIALS_CSV_FILENAME
from phasesweep.engine.read import (
    _REPEAT_GROUP_COLUMN,
    _RERUN_OF_COLUMN,
    RepeatGroupView,
    TrialTable,
    TrialTables,
)
from phasesweep.engine.reread import PendingRereadNotice
from phasesweep.engine.state import OBSERVED_PREFIX
from phasesweep.errors import operator_step
from phasesweep.runtime.json import JsonObject, JsonValue, json_integer
from phasesweep.runtime.terminal import (
    GAP,
    INDENT,
    display_path,
    objective_text,
    paint,
    param_text,
    plural,
    wrap,
    wrap_labeled,
)

#: The word that marks a winning row, and so the width of the leading marker column.
_WINNER_WORD = "winner"
_REPEAT_GROUPS_HEADING = "Repeat groups (statistics include infeasible measured members):"
_REPEAT_GROUPS_NOTE = "Group numbers can change as members arrive."
#: How a winner's generation is labelled when its recorded config does not match the current one,
#: in the words ``show-winners`` uses for the same publication.
_HISTORICAL = "does not match the current config"

#: How a value sorts within one parameter column: listed choices, then unlisted values, then
#: missing ones; the second and third fields order values within the first kind.
_SortKey = tuple[int, float, str]


@dataclass(frozen=True)
class _Column:
    """One table column: its header, one text per row, and whether it aligns as a number."""

    header: str
    cells: tuple[str, ...]
    numeric: bool


def _is_number(value: JsonValue) -> TypeGuard[int | float]:
    """Say whether a JSON value is an int or a float, booleans excluded.

    :param JsonValue value: Value to test.
    :return TypeGuard[int | float]: ``True`` for an int or float.
    """
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number_text(value: JsonValue) -> str:
    """Format a measured value to six significant digits, ``-`` when there is none.

    :param JsonValue value: The row's value, ``None`` when no number was recorded.
    :return str: Terminal text for the cell.
    """
    if not _is_number(value):
        return "-"
    return objective_text(float(value))


def _duration_text(value: JsonValue) -> str:
    """Format a trainer runtime in seconds to one decimal place, ``-`` when it never ran.

    :param JsonValue value: Recorded wall-clock seconds, ``None`` for a trial without a trainer.
    :return str: Terminal text for the cell.
    """
    return f"{value:.1f}s" if _is_number(value) else "-"


def _reason(row: JsonObject) -> str | None:
    """Read a trial row's recorded cause, if the table carries one.

    :param JsonObject row: A trial-table row.
    :return str | None: The failure, interruption or infeasibility cause, or ``None``.
    """
    value = row.get("reason")
    return value if isinstance(value, str) else None


def _optional_text(value: JsonValue) -> str:
    """Format an identifier cell, ``-`` when the row has none.

    :param JsonValue value: A trial or group number, or ``None``.
    :return str: Terminal text for the cell.
    """
    return "-" if value is None else str(value)


def _params(row: JsonObject) -> JsonObject:
    """Read a trial row's sampled params.

    :param JsonObject row: A trial-table row.
    :raises TypeError: The row's params are not a mapping.
    :return JsonObject: The params by key.
    """
    value = row["params"]
    if not isinstance(value, dict):
        raise TypeError("Trial table column 'params' must hold a mapping.")
    return value


def _param_keys(
    space: dict[str, SearchParam], recorded: Sequence[JsonObject], *, leave_out: str | None = None
) -> list[str]:
    """List the parameter columns: the search space in config order, then other recorded keys.

    :param dict[str, SearchParam] space: The phase's sampler search space, repeat key included.
    :param Sequence[JsonObject] recorded: Params of every row to show.
    :param str | None leave_out: A key that gets no column, the repeat key of a group table.
    :return list[str]: Keys in display order.
    """
    keys = dict.fromkeys(key for key in space if key != leave_out)
    for params in recorded:
        keys.update(dict.fromkeys(key for key in params if key != leave_out))
    return list(keys)


def _sort_key(
    params: JsonObject, keys: Sequence[str], space: dict[str, SearchParam]
) -> list[_SortKey]:
    """Order a row by its parameter values, left to right.

    A categorical value sorts by its place in the configured choices, a number by value and a
    value outside the choices after the listed ones; a missing value sorts last.

    :param JsonObject params: The row's sampled params.
    :param Sequence[str] keys: Parameter columns in display order.
    :param dict[str, SearchParam] space: The phase's sampler search space.
    :return list[_SortKey]: One sort key per parameter column.
    """
    ordered: list[_SortKey] = []
    for key in keys:
        if key not in params:
            ordered.append((2, 0.0, ""))
            continue
        value = params[key]
        param = space.get(key)
        if isinstance(param, CategoricalParam):
            listed = value in param.choices
            ordered.append((0, float(param.choices.index(value)), "") if listed else (1, 0.0, ""))
        elif _is_number(value):
            ordered.append((0, float(value), ""))
        else:
            ordered.append((0, 0.0, str(value)))
    return ordered


def _param_columns(keys: Sequence[str], params: Sequence[JsonObject]) -> list[_Column]:
    """Build one column per parameter key, numbers right-aligned and anything else left.

    :param Sequence[str] keys: Parameter columns in display order.
    :param Sequence[JsonObject] params: Sampled params of each row, in row order.
    :return list[_Column]: A column per key, ``-`` where a row lacks the key.
    """
    columns: list[_Column] = []
    for key in keys:
        present = [row[key] for row in params if key in row]
        columns.append(
            _Column(
                header=key,
                cells=tuple(param_text(row[key]) if key in row else "-" for row in params),
                numeric=bool(present) and all(_is_number(value) for value in present),
            )
        )
    return columns


def _table_lines(
    columns: Sequence[_Column],
    winners: Sequence[bool],
    *,
    width: int,
    color: bool,
    reasons: Sequence[str | None] = (),
) -> list[str]:
    """Render columns as an aligned table with a leading winner column.

    Cells are padded as plain text before any styling, so escapes never count toward a column.
    A winning row is bold with the word ``winner`` in green. A row's reason prints under it,
    indented to the ``trial`` column and wrapped to ``width`` with a hanging indent.

    :param Sequence[_Column] columns: Columns after the winner column.
    :param Sequence[bool] winners: Whether each row is a winner.
    :param int width: Terminal width the reason lines wrap to.
    :param bool color: Whether to style the winner rows.
    :param Sequence[str | None] reasons: Each row's recorded cause, ``None`` for none; may be empty.
    :return list[str]: The header line, then each row followed by its reason lines.
    """
    sizes = [max([len(column.header), *map(len, column.cells)]) for column in columns]

    def join(texts: Sequence[str]) -> str:
        """Join one row's padded cells, numbers right-aligned and text left-aligned.

        :param Sequence[str] texts: One text per column.
        :return str: The cells separated by two spaces.
        """
        return GAP.join(
            text.rjust(size) if column.numeric else text.ljust(size)
            for column, text, size in zip(columns, texts, sizes, strict=True)
        )

    blank = " " * len(_WINNER_WORD)
    lines = [f"{INDENT}{blank}{GAP}{join([column.header for column in columns])}".rstrip()]
    reason_indent = " " * (len(INDENT) + len(_WINNER_WORD) + len(GAP))
    for index, winner in enumerate(winners):
        body = f"{GAP}{join([column.cells[index] for column in columns])}".rstrip()
        if winner:
            lines.append(
                INDENT
                + paint(_WINNER_WORD, "winner", color=color)
                + paint(body, "label", color=color)
            )
        else:
            lines.append(f"{INDENT}{blank}{body}")
        reason = reasons[index] if reasons else None
        if reason:
            lines.extend(wrap_labeled("reason:", reason, width, indent=reason_indent))
    return lines


def _trial_lines(
    table: TrialTable, phase: Phase, metric_name: str, *, width: int, color: bool
) -> list[str]:
    """Render one phase's trials sorted by their parameter values.

    :param TrialTable table: The phase's rows and the optional columns they carry.
    :param Phase phase: The configured phase, for its search space.
    :param str metric_name: Header of the objective column.
    :param int width: Terminal width the reason lines wrap to.
    :param bool color: Whether to style the winner rows.
    :return list[str]: The table, with a trial's reason under its row.
    """
    space = sampler_search_space(phase)
    recorded = {json_integer(row, "trial"): _params(row) for row in table.rows}
    keys = _param_keys(space, list(recorded.values()))
    rows = sorted(
        table.rows,
        key=lambda row: (
            _sort_key(recorded[json_integer(row, "trial")], keys, space),
            json_integer(row, "trial"),
        ),
    )
    params = [recorded[json_integer(row, "trial")] for row in rows]
    columns = [
        _Column("trial", tuple(str(row["trial"]) for row in rows), numeric=True),
        _Column(metric_name, tuple(_number_text(row["value"]) for row in rows), numeric=True),
        *_param_columns(keys, params),
    ]
    if _REPEAT_GROUP_COLUMN in table.columns:
        cells = tuple(_optional_text(row[_REPEAT_GROUP_COLUMN]) for row in rows)
        columns.append(_Column("group", cells, numeric=True))
    columns.extend(
        _Column(name, tuple(_number_text(row[name]) for row in rows), numeric=True)
        for name in table.columns
        if name.startswith(OBSERVED_PREFIX)
    )
    columns.append(_Column("state", tuple(str(row["state"]) for row in rows), numeric=False))
    columns.append(
        _Column("time", tuple(_duration_text(row["duration_s"]) for row in rows), numeric=True)
    )
    if _RERUN_OF_COLUMN in table.columns:
        cells = tuple(_optional_text(row[_RERUN_OF_COLUMN]) for row in rows)
        columns.append(_Column("rerun of", cells, numeric=True))
    return [
        *_table_lines(
            columns,
            [row["winner"] == "*" for row in rows],
            width=width,
            color=color,
            reasons=[_reason(row) for row in rows],
        ),
        *_winner_dir_lines(rows, width=width, color=color),
    ]


def _winner_dir_lines(rows: Sequence[JsonObject], *, width: int, color: bool) -> list[str]:
    """Name the directory of each winning trial under the table, in trial order.

    A path is never wrapped, as the table is not: one that does not fit its line after the
    label prints whole on a line of its own. A winning trial that recorded no directory
    gets no line.

    :param Sequence[JsonObject] rows: The phase's trial rows.
    :param int width: Terminal width a label and its path must fit to share a line.
    :param bool color: Whether to style the paths.
    :return list[str]: One line per winning trial with a recorded directory, two when it is wide.
    """
    lines: list[str] = []
    for row in sorted(rows, key=lambda row: json_integer(row, "trial")):
        directory = row["trial_dir"]
        if row["winner"] != "*" or not isinstance(directory, str):
            continue
        label = f"{INDENT}winner trial {row['trial']} directory:"
        shown = display_path(Path(directory))
        drawn = paint(shown, "path", color=color)
        if len(label) + 1 + len(shown) <= width:
            lines.append(f"{label} {drawn}")
        else:
            lines.extend([label, f"{INDENT}{INDENT}{drawn}"])
    return lines


def _group_lines(
    groups: Sequence[RepeatGroupView], phase: Phase, *, width: int, color: bool
) -> list[str]:
    """Render a repeats phase's groups with the trial table's marker, params and sort.

    :param Sequence[RepeatGroupView] groups: The phase's repeat groups.
    :param Phase phase: The configured phase, for its search space and repeat key.
    :param int width: Terminal width the explanatory lines wrap to.
    :param bool color: Whether to style the winner rows.
    :return list[str]: Two explanatory lines, then the table.
    """
    space = sampler_search_space(phase)
    leave_out = None if phase.repeats is None else phase.repeats.key
    keys = _param_keys(space, [group.params for group in groups], leave_out=leave_out)
    ordered = sorted(groups, key=lambda group: (_sort_key(group.params, keys, space), group.group))
    columns = [
        _Column("group", tuple(str(group.group) for group in ordered), numeric=True),
        *_param_columns(keys, [group.params for group in ordered]),
        _Column(
            "n/expected",
            tuple(f"{group.n}/{group.expected}" for group in ordered),
            numeric=True,
        ),
        _Column("reduce", tuple(group.reduce for group in ordered), numeric=False),
    ]
    for name, values in (
        ("aggregate", [group.aggregate for group in ordered]),
        ("mean", [group.mean for group in ordered]),
        ("sd", [group.sd for group in ordered]),
        ("min", [group.min for group in ordered]),
        ("max", [group.max for group in ordered]),
    ):
        cells = tuple(_number_text(value) for value in values)
        columns.append(_Column(name, cells, numeric=True))
    eligible = tuple("yes" if group.eligible else "no" for group in ordered)
    columns.append(_Column("eligible", eligible, numeric=False))
    for name in sorted({name for group in groups for name in group.observed}):
        means = tuple(_number_text(group.observed.get(name)) for group in ordered)
        counts = tuple(str(group.observed_n.get(name, 0)) for group in ordered)
        columns.append(_Column(f"{OBSERVED_PREFIX}{name}", means, numeric=True))
        columns.append(_Column(f"observed_n:{name}", counts, numeric=True))
    return [
        "",
        *wrap(_REPEAT_GROUPS_HEADING, width),
        *wrap(_REPEAT_GROUPS_NOTE, width),
        *_table_lines(columns, [group.winner for group in ordered], width=width, color=color),
    ]


def _source_text(tables: TrialTables, generation: str, published: bool) -> str:
    """Name a generation the winner marks come from, its publication state and config match.

    A generation whose recorded config does not match the current one marks the winner that
    config selected, which a row's current state (``INFEASIBLE`` after a bound edit) can
    contradict; ``show-winners`` labels the same publication historical, so this says it too.

    :param TrialTables tables: The tables, for which generations are historical.
    :param str generation: The generation id.
    :param bool published: Whether it is the validated publication.
    :return str: Such as ``published generation g1 (does not match the current config)``.
    """
    text = f"{'published' if published else 'unpublished'} generation {generation}"
    if generation in tables.historical_generations:
        text = f"{text} ({_HISTORICAL})"
    return text


def _generation_line(tables: TrialTables) -> str:
    """Say which generation the winner marks come from, once for the whole listing.

    :param TrialTables tables: The tables one publication resolution produced.
    :return str: The common generation and its publication state, a pointer to the per-phase
        lines when the winners differ, or that no winner is marked without a publication.
    """
    sources = set(tables.winner_generations.values())
    if len(sources) == 1:
        generation, published = next(iter(sources))
        return _source_text(tables, generation, published)
    if len(sources) > 1:
        return "winners come from different generations; each phase names its own"
    represented = tables.represented_generation_id or tables.published_generation_id
    if represented is None:
        return "no generation is published, so no winner is marked"
    state = "published" if represented == tables.published_generation_id else "unpublished"
    return f"{state} generation {represented}"


def _direction(goal: str) -> str:
    """Say which values are better under a metric goal.

    :param str goal: ``minimize`` or ``maximize``.
    :return str: ``lower`` for ``minimize``, else ``higher``.
    """
    return "lower" if goal == "minimize" else "higher"


def _phase_lines(
    table: TrialTable,
    phase: Phase,
    experiment: Experiment,
    tables: TrialTables,
    *,
    csv_root: Path | None,
    width: int,
    color: bool,
    rerun_command: str | None,
) -> list[str]:
    """Render one phase: its heading, any notes under it, its trials and its repeat groups.

    The heading reads ``finished of target trials``, then in parentheses the rows that do not
    count toward the target, by state, then the configured metric name and the direction of its
    goal, which addresses the study the rows come from. A phase a run would refuse prints that
    refusal and its fix before any other note. Each search-space growth the study recorded then
    gets a note naming the draw position it took effect at and the values it added, and finished
    trials whose directory is gone get one naming them.

    :param TrialTable table: The phase's trials.
    :param Phase phase: The configured phase.
    :param Experiment experiment: The experiment, for the configured metric's name and goal.
    :param TrialTables tables: All tables, for the winner generation of each phase.
    :param Path | None csv_root: Artifact directory of an experiment that keeps no ledger.
    :param int width: Terminal width the prose wraps to.
    :param bool color: Whether to style the winner rows.
    :param str | None rerun_command: Shell-quoted run command for a phase waiting to re-read.
    :return list[str]: The phase's lines, without a leading blank line.
    """
    notes: list[str] = []
    if not table.available:
        count = "trial data unavailable"
    elif csv_root is not None:
        count = "no trial ledger (storage: null)"
        snapshot = csv_root / table.phase / TRIALS_CSV_FILENAME
        if snapshot.is_file():
            notes.append(f"the trials of this phase's last run are in {display_path(snapshot)}")
    elif not table.rows:
        count = "no trials yet"
    else:
        count = f"{table.finished} of {plural(phase.n_trials, 'trial')}"
        if table.uncounted:
            others = ", ".join(f"{rows} {state.lower()}" for state, rows in table.uncounted)
            count = f"{count} ({others})"
    lines = wrap(
        f"{table.phase}: {count}, {experiment.metric.name} "
        f"({_direction(experiment.metric.goal)} is better)",
        width,
        subsequent_indent=INDENT,
    )
    if table.refusal is not None:
        lines.extend(wrap(str(table.refusal), width, indent=INDENT, subsequent_indent=INDENT))
        lines.extend(wrap_labeled("fix:", operator_step(table.refusal), width, indent=INDENT))
        if isinstance(table.refusal, PendingRereadNotice) and rerun_command is not None:
            lines.append(f"{INDENT}{rerun_command}")
    for position, gains in table.growth:
        lines.extend(
            wrap(
                f"search_space grew at draw position {position}: {gains}",
                width,
                indent=INDENT,
                subsequent_indent=INDENT * 2,
            )
        )
    if table.pruned:
        notes.append(pruned_trials_note(table.pruned))
    sources = set(tables.winner_generations.values())
    source = tables.winner_generations.get(table.phase)
    if len(sources) > 1 and source is not None:
        generation, published = source
        state = "published" if published else "unpublished"
        if generation in tables.historical_generations:
            state = f"{state}, {_HISTORICAL}"
        notes.append(f"winner from generation {generation} ({state})")
    for note in notes:
        lines.extend(wrap(note, width, indent=INDENT, subsequent_indent=INDENT))
    if table.rows:
        lines.extend(_trial_lines(table, phase, experiment.metric.name, width=width, color=color))
    if table.groups:
        lines.extend(_group_lines(table.groups, phase, width=width, color=color))
    return lines


def render_trial_tables(
    tables: TrialTables,
    experiment: Experiment,
    *,
    csv_root: Path | None,
    width: int,
    color: bool,
    rerun_command: str | None = None,
) -> str:
    """Render every phase's trial table under a header that names where the winner marks come from.

    :param TrialTables tables: The tables one publication resolution produced.
    :param Experiment experiment: The config the tables were read with, for each phase's search
        space and the metric's name and direction.
    :param Path | None csv_root: The artifact directory of an experiment that keeps no ledger,
        where each phase's ``trials.csv`` is the only record of its trials; ``None`` when a
        ledger holds them.
    :param int width: Terminal width that prose wraps to; tables keep their natural width.
    :param bool color: Whether to style the winner rows, from ``use_color`` on the output stream.
    :param str | None rerun_command: Shell-quoted run command for a phase waiting to re-read.
    :return str: The experiment name, the generation line, then each phase's heading and table.
    """
    phases = {phase.name: phase for phase in experiment.phases}
    lines = [tables.experiment, *wrap(_generation_line(tables), width)]
    for table in tables.phases:
        lines.append("")
        lines.extend(
            _phase_lines(
                table,
                phases[table.phase],
                experiment,
                tables,
                csv_root=csv_root,
                width=width,
                color=color,
                rerun_command=rerun_command,
            )
        )
    return "\n".join(lines)


def unsettled_running_comment(running: int, width: int | None = None) -> str:
    """Say that trials recorded RUNNING belong to no live run and what a rerun does with them.

    Such a trial was cut off, or finished with a result its run could not read. A rerun
    settles both, and stops a trainer that outlived its run, so the comment says to let
    one finish first.

    :param int running: How many trials the ledger records as RUNNING while no run holds the locks.
    :param int | None width: Terminal width to wrap to, ``# ``-prefixing each line; ``None`` keeps
        the comment on one line.
    :return str: The ``#`` comment.
    """
    verb, pronoun = ("is", "it") if running == 1 else ("are", "them")
    text = (
        f"No run is active; {plural(running, 'trial')} {verb} recorded RUNNING. Rerun the same "
        f"config to settle {pronoun}. A trainer still running is stopped, so let it exit first "
        "to keep its result."
    )
    if width is None:
        return f"# {text}"
    return "\n".join(wrap(text, width, indent="# ", subsequent_indent="# "))
