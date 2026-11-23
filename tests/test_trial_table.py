"""Exact plain-text output of ``status --trials`` tables, built from trial tables directly."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, FloatParam, IntParam, Repeats, Sampler
from phasesweep.engine.read import RepeatGroupView, TrialTable, TrialTables, read_trial_tables
from phasesweep.evidence.models import LogRegexExtractor
from phasesweep.runtime.json import JsonObject, JsonValue
from phasesweep.trial_table import render_trial_tables, unsettled_running_comment
from tests.conftest import (
    ATTEMPT_TRIAL_NAME,
    SHORT_TRIAL_NAME,
    invoke_cli_boundary,
    make_experiment,
    terminal_stream,
    write_constant_trainer,
)

GENERATION = "6413d5c1edd5475998cfadd4d3b59980"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _metric(goal: str = "minimize") -> Metric:
    """Build the starter's ``eval_loss`` metric."""
    return Metric(
        name="eval_loss",
        goal="maximize" if goal == "maximize" else "minimize",
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
    )


def _phase(
    name: str, *, n_trials: int = 4, **space: CategoricalParam | FloatParam | IntParam
) -> Phase:
    """Build a phase of ``n_trials`` whose search space is ``space``, in the given order."""
    return Phase(
        name=name,
        n_trials=n_trials,
        sampler=Sampler(type="random", seed=0),
        search_space=dict(space),
    )


def _row(
    trial: int,
    value: float | None,
    params: JsonObject,
    *,
    state: str = "COMPLETE",
    duration: float | None = 0.5,
    winner: bool = False,
    **extra: JsonValue,
) -> JsonObject:
    """Build one trial row; ``extra`` carries the optional columns the row has."""
    return {
        "trial": trial,
        "state": state,
        "value": value,
        "params": params,
        "duration_s": duration,
        **extra,
        "winner": "*" if winner else "",
    }


def _table(
    phase: str,
    rows: list[JsonObject],
    *,
    groups: tuple[RepeatGroupView, ...] = (),
    available: bool = True,
    finished: int | None = None,
    uncounted: tuple[tuple[str, int], ...] = (),
    refusal: tuple[str, str] | None = None,
    recorded_metric: tuple[str, str] | None = None,
    growth: tuple[tuple[int, str], ...] = (),
) -> TrialTable:
    """Build a table whose columns are the core ones plus every optional key any row carries.

    Every row counts toward the target unless ``finished`` and ``uncounted`` say otherwise.
    """
    core = ("trial", "state", "value", "params", "duration_s", "trial_dir")
    optional = {key for row in rows for key in row} - {*core, "winner"}
    ordered = [name for name in ("repeat_group", "rerun_of") if name in optional]
    ordered += sorted(name for name in optional if name.startswith("observed:"))
    ordered += ["reason"] if "reason" in optional else []
    columns = (*core, *ordered, "winner")
    return TrialTable(
        phase=phase,
        available=available,
        columns=columns,
        rows=tuple({name: row.get(name) for name in columns} for row in rows),
        finished=len(rows) if finished is None else finished,
        uncounted=uncounted,
        groups=groups,
        refusal=refusal,
        recorded_metric=recorded_metric,
        growth=growth,
    )


def _tables(
    *tables: TrialTable,
    represented: str | None = GENERATION,
    published: str | None = GENERATION,
    winners: dict[str, tuple[str, bool]] | None = None,
    name: str = "phasesweep_starter",
) -> TrialTables:
    """Wrap tables with the publication facts the header reads."""
    if winners is None:
        winners = {t.phase: (GENERATION, True) for t in tables if any(r["winner"] for r in t.rows)}
    return TrialTables(
        experiment=name,
        published_generation_id=published,
        publication_integrity="ok",
        publication_error=None,
        phases=tables,
        represented_generation_id=represented,
        winner_generations=winners,
    )


def _render(
    tables: TrialTables,
    *phases: Phase,
    width: int = 80,
    color: bool = False,
    csv_root: Path | None = None,
    goal: str = "minimize",
) -> str:
    """Render ``tables`` for an experiment with these phases and the starter's metric."""
    experiment: Experiment = make_experiment(
        experiment=tables.experiment, phases=list(phases), metric=_metric(goal)
    )
    return render_trial_tables(tables, experiment, csv_root=csv_root, width=width, color=color)


def _starter() -> tuple[TrialTables, tuple[Phase, Phase]]:
    """Build the packaged starter's two phases as ``status --trials`` reads them."""
    depth = _phase(
        "depth",
        n_trials=2,
        **{"model.n_layers": CategoricalParam(type="categorical", choices=[6, 8])},
    )
    rate = _phase(
        "learning_rate",
        n_trials=2,
        **{"optimizer.lr": CategoricalParam(type="categorical", choices=[0.0001, 0.0003])},
    )
    tables = _tables(
        _table(
            "depth",
            [
                _row(0, 0.3, {"model.n_layers": 8}, winner=True),
                _row(1, 0.5, {"model.n_layers": 6}),
            ],
        ),
        _table(
            "learning_rate",
            [
                _row(0, 0.3, {"optimizer.lr": 0.0003}, winner=True),
                _row(1, 0.413822345853, {"optimizer.lr": 0.0001}),
            ],
        ),
    )
    return tables, (depth, rate)


def test_starter_tables_name_the_winner_in_a_column_and_fit_eighty_columns() -> None:
    """Both phases print one generation line once, a labelled objective and the winner word."""
    tables, phases = _starter()

    rendered = _render(tables, *phases)

    assert rendered == "\n".join(
        [
            "phasesweep_starter",
            f"published generation {GENERATION}",
            "",
            "depth: 2 of 2 trials, eval_loss (lower is better)",
            "          trial  eval_loss  model.n_layers  state     time",
            "              1        0.5               6  COMPLETE  0.5s",
            "  winner      0        0.3               8  COMPLETE  0.5s",
            "",
            "learning_rate: 2 of 2 trials, eval_loss (lower is better)",
            "          trial  eval_loss  optimizer.lr  state     time",
            "              1   0.413822        0.0001  COMPLETE  0.5s",
            "  winner      0        0.3        0.0003  COMPLETE  0.5s",
        ]
    )
    assert all(len(line) <= 80 for line in rendered.splitlines())


def test_a_maximized_metric_says_higher_is_better() -> None:
    """The heading names the direction from the metric's goal."""
    tables, phases = _starter()

    rendered = _render(tables, *phases, goal="maximize")

    assert "depth: 2 of 2 trials, eval_loss (higher is better)" in rendered.splitlines()


def test_colour_styles_only_the_winner_row_and_leaves_the_text_otherwise_identical() -> None:
    """On a terminal the winner row is bold with a green word; nothing else carries an escape."""
    tables, phases = _starter()

    plain = _render(tables, *phases)
    styled = _render(tables, *phases, color=True)

    assert ANSI.sub("", styled) == plain
    escaped = [line for line in styled.splitlines() if "\x1b" in line]
    assert escaped == [
        "  \x1b[1;32mwinner\x1b[0m\x1b[1m      0        0.3               8  COMPLETE  0.5s\x1b[0m",
        "  \x1b[1;32mwinner\x1b[0m\x1b[1m      0        0.3        0.0003  COMPLETE  0.5s\x1b[0m",
    ]
    assert "\x1b" not in plain


def test_a_phase_without_trials_says_so() -> None:
    """An empty phase prints its heading with the note in place of a count, and no table."""
    phase = _phase("depth", **{"model.n_layers": IntParam(type="int", low=1, high=9)})
    tables = _tables(_table("depth", []), represented=None, published=None)

    assert _render(tables, phase) == "\n".join(
        [
            "phasesweep_starter",
            "no generation is published, so no winner is marked",
            "",
            "depth: no trials yet, eval_loss (lower is better)",
        ]
    )


def test_a_recorded_refusal_and_its_fix_print_under_the_heading() -> None:
    """A phase a run would refuse says why and what to do under its heading, wrapped to the width."""
    phase = _phase("search", n_trials=8, x=IntParam(type="int", low=0, high=9))
    refusal = (
        "Phase 'search' previously aborted at its accepted n_trials=8: 3 consecutive "
        "failed/infeasible trials reached max_consecutive_failures=3.",
        "Fix the cause, then raise phases[search].n_trials above 8 and rerun.",
    )
    rows = [_row(index, None, {"x": index}, state="FAIL", duration=0.1) for index in range(3)]
    tables = _tables(
        _table("search", rows, refusal=refusal), represented=None, published=None, winners={}
    )

    lines = _render(tables, phase).splitlines()

    assert lines[3:8] == [
        "search: 3 of 8 trials, eval_loss (lower is better)",
        "  Phase 'search' previously aborted at its accepted n_trials=8: 3 consecutive",
        "  failed/infeasible trials reached max_consecutive_failures=3.",
        "  fix: Fix the cause, then raise phases[search].n_trials above 8 and rerun.",
        "          trial  eval_loss  x  state  time",
    ]


def test_each_search_space_growth_prints_under_the_heading_after_the_refusal() -> None:
    """A grown phase names each growth's first draw position and added values, after any refusal and its fix."""
    phase = _phase("search", n_trials=16, lr=CategoricalParam(type="categorical", choices=[0.001]))
    refusal = ("Phase 'search' previously aborted.", "Raise phases[search].n_trials and rerun.")
    rows = [_row(index, 0.5, {"lr": 0.001}) for index in range(2)]
    growth = ((12, "lr +0.003, +0.01"), (14, "lr +0.03"))
    tables = _tables(
        _table("search", rows, refusal=refusal, growth=growth),
        represented=None,
        published=None,
        winners={},
    )

    lines = _render(tables, phase).splitlines()

    assert lines[3:8] == [
        "search: 2 of 16 trials, eval_loss (lower is better)",
        "  Phase 'search' previously aborted.",
        "  fix: Raise phases[search].n_trials and rerun.",
        "  search_space grew at draw position 12: lr +0.003, +0.01",
        "  search_space grew at draw position 14: lr +0.03",
    ]


def test_the_table_is_headed_with_the_recorded_metric_and_names_a_changed_config_metric() -> None:
    """Values print under the metric the study recorded; a config naming another says so under the heading."""
    phase = _phase("search", n_trials=8, x=IntParam(type="int", low=0, high=9))
    rows = [_row(index, 0.5, {"x": index}) for index in range(6)]
    tables = _tables(
        _table("search", rows, recorded_metric=("loss", "minimize")),
        represented=None,
        published=None,
        winners={},
    )

    lines = _render(tables, phase, goal="maximize").splitlines()

    assert lines[3:6] == [
        "search: 6 of 8 trials, loss (lower is better)",
        "  the config names eval_loss (higher is better); a run refuses this study",
        "          trial  loss  x  state     time",
    ]
    # An unchanged metric prints no note, so a clean listing is the one the config alone gives.
    unchanged = _table("search", rows, recorded_metric=("eval_loss", "minimize"))
    clean = _render(_tables(unchanged, represented=None, published=None, winners={}), phase)
    assert "the config names" not in clean


def test_the_recorded_metric_comes_from_the_study_the_trials_were_read_from(
    tmp_path: Path,
) -> None:
    """``read_trial_tables`` carries the study's recorded name and goal, not the edited config's."""
    experiment = make_experiment(persistent=tmp_path, trainer=write_constant_trainer(tmp_path))
    run_experiment(experiment)
    edited = experiment.model_copy(
        update={
            "metric": experiment.metric.model_copy(update={"name": "renamed", "goal": "maximize"})
        }
    )

    (table,) = read_trial_tables(edited).phases

    assert table.recorded_metric == (experiment.metric.name, experiment.metric.goal)
    assert table.recorded_metric != (edited.metric.name, edited.metric.goal)


def test_unreadable_trial_data_is_reported_without_a_table() -> None:
    """A phase whose trials could not be read says so under its heading."""
    phase = _phase("depth", **{"model.n_layers": IntParam(type="int", low=1, high=9)})
    tables = _tables(_table("depth", [], available=False), winners={})

    assert _render(tables, phase).splitlines()[3:] == [
        "depth: trial data unavailable, eval_loss (lower is better)"
    ]


def test_storage_null_names_the_last_runs_csv_when_one_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a ledger the heading says so and the note points at the phase's trials.csv."""
    monkeypatch.chdir(tmp_path)
    phase = _phase("depth", **{"model.n_layers": IntParam(type="int", low=1, high=9)})
    tables = _tables(_table("depth", []), represented=None, published=None)
    (tmp_path / "runs" / "depth").mkdir(parents=True)

    without = _render(tables, phase, csv_root=tmp_path / "runs")
    (tmp_path / "runs" / "depth" / "trials.csv").write_text("number\n", encoding="utf-8")
    with_csv = _render(tables, phase, csv_root=tmp_path / "runs")

    heading = "depth: no trial ledger (storage: null), eval_loss (lower is better)"
    assert without.splitlines()[3:] == [heading]
    assert with_csv.splitlines()[3:] == [
        heading,
        "  the trials of this phase's last run are in runs/depth/trials.csv",
    ]


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            80,
            ["learning_rate: no trial ledger (storage: null), eval_loss (lower is better)"],
            id="80",
        ),
        pytest.param(
            60,
            [
                "learning_rate: no trial ledger (storage: null), eval_loss",
                "  (lower is better)",
            ],
            id="60",
        ),
    ],
)
def test_storage_null_heading_wraps_to_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, expected: list[str]
) -> None:
    """The heading keeps the phase name, the explanation and the objective, and wraps under an indent."""
    monkeypatch.chdir(tmp_path)
    phase = _phase(
        "learning_rate", **{"optimizer.lr": FloatParam(type="float", low=1e-4, high=1e-3)}
    )
    tables = _tables(_table("learning_rate", []), represented=None, published=None)

    rendered = _render(tables, phase, width=width, csv_root=tmp_path / "runs")

    assert rendered.splitlines()[3:] == expected
    assert all(len(line) <= width for line in rendered.splitlines())


def test_repeat_groups_mark_every_member_of_the_winning_group() -> None:
    """A repeats phase marks each member row, then prints groups with the same columns and sort."""
    phase = Phase(
        name="r",
        n_trials=6,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
        repeats=Repeats(key="seed", values=[0, 1, 2]),
    )
    members = [
        _row(trial, value, {"x": x, "seed": seed}, repeat_group=group, winner=x == 3)
        for trial, (x, seed, value, group) in enumerate(
            [(3, 0, 0.5, 2), (3, 1, 0.7, 2), (3, 2, 0.9, 2), (1, 0, 1.0, 5), (1, 1, 2.0, 5)]
        )
    ]
    groups = (
        RepeatGroupView(
            group=2, params={"x": 3}, n=3, expected=3, reduce="mean", aggregate=0.7, mean=0.7,
            sd=0.2, min=0.5, max=0.9, eligible=True, winner=True,
            observed={"acc": 0.8}, observed_n={"acc": 3},
        ),
        RepeatGroupView(
            group=5, params={"x": 1}, n=2, expected=3, reduce="mean", aggregate=None, mean=1.5,
            sd=0.7071, min=1.0, max=2.0, eligible=False, winner=False,
            observed={}, observed_n={},
        ),
    )  # fmt: skip
    tables = _tables(_table("r", members, groups=groups))

    assert _render(tables, phase, width=60) == "\n".join(
        [
            "phasesweep_starter",
            f"published generation {GENERATION}",
            "",
            "r: 5 of 6 trials, eval_loss (lower is better)",
            "          trial  eval_loss  x  seed  group  state     time",
            "              3          1  1     0      5  COMPLETE  0.5s",
            "              4          2  1     1      5  COMPLETE  0.5s",
            "  winner      0        0.5  3     0      2  COMPLETE  0.5s",
            "  winner      1        0.7  3     1      2  COMPLETE  0.5s",
            "  winner      2        0.9  3     2      2  COMPLETE  0.5s",
            "",
            "Repeat groups (statistics include infeasible measured",
            "members):",
            "Group numbers can change as members arrive.",
            "          group  x  n/expected  reduce  aggregate  mean      sd"
            "  min  max  eligible  observed:acc  observed_n:acc",
            "              5  1         2/3  mean            -   1.5  0.7071    1    2  no"
            "                   -               0",
            "  winner      2  3         3/3  mean          0.7   0.7     0.2  0.5  0.9  yes"
            "                0.8               3",
        ]
    )


def test_rows_with_different_key_sets_show_a_dash_and_sort_by_value() -> None:
    """Config-order columns come first, a recorded extra key follows, a missing key reads ``-``."""
    phase = _phase(
        "p", a=IntParam(type="int", low=0, high=9), c=IntParam(type="int", low=0, high=9)
    )
    tables = _tables(
        _table(
            "p",
            [
                _row(0, 1.0, {"a": 5, "extra": "z"}),
                _row(1, 2.0, {"c": 2, "a": 2}),
                _row(2, 3.0, {"a": 2}),
                _row(3, 4.0, {"a": 2, "c": 1}),
            ],
        ),
        winners={},
    )

    assert _render(tables, phase).splitlines()[3:] == [
        "p: 4 of 4 trials, eval_loss (lower is better)",
        "          trial  eval_loss  a  c  extra  state     time",
        "              3          4  2  1  -      COMPLETE  0.5s",
        "              1          2  2  2  -      COMPLETE  0.5s",
        "              2          3  2  -  -      COMPLETE  0.5s",
        "              0          1  5  -  z      COMPLETE  0.5s",
    ]


def test_a_categorical_parameter_sorts_by_configured_choice_order() -> None:
    """Listed choices keep their configured order, unlisted values follow, missing ones are last."""
    phase = _phase("p", act=CategoricalParam(type="categorical", choices=["silu", "relu", "gelu"]))
    tables = _tables(
        _table(
            "p",
            [
                _row(0, 1.0, {"act": "gelu"}),
                _row(1, 1.0, {}),
                _row(2, 1.0, {"act": "swish"}),
                _row(3, 1.0, {"act": "silu"}),
                _row(4, 1.0, {"act": "relu"}),
                _row(5, 1.0, {"act": "silu"}),
            ],
        ),
        winners={},
    )

    assert _render(tables, phase).splitlines()[4:] == [
        "          trial  eval_loss  act    state     time",
        "              3          1  silu   COMPLETE  0.5s",
        "              5          1  silu   COMPLETE  0.5s",
        "              4          1  relu   COMPLETE  0.5s",
        "              0          1  gelu   COMPLETE  0.5s",
        "              2          1  swish  COMPLETE  0.5s",
        "              1          1  -      COMPLETE  0.5s",
    ]


def test_numeric_parameters_sort_by_value_and_ties_by_trial_number() -> None:
    """A float parameter sorts ascending whatever the draw order, equal values by trial number."""
    phase = _phase("p", lr=FloatParam(type="float", low=1e-5, high=1e-1, log=True))
    tables = _tables(
        _table(
            "p",
            [
                _row(0, 0.9, {"lr": 0.001}),
                _row(1, 0.8, {"lr": 0.0001}),
                _row(2, 0.7, {"lr": 0.03162277}),
                _row(3, 0.6, {"lr": 0.0001}),
            ],
        ),
        winners={},
    )

    assert _render(tables, phase).splitlines()[4:] == [
        "          trial  eval_loss       lr  state     time",
        "              1        0.8   0.0001  COMPLETE  0.5s",
        "              3        0.6   0.0001  COMPLETE  0.5s",
        "              0        0.9    0.001  COMPLETE  0.5s",
        "              2        0.7  0.03162  COMPLETE  0.5s",
    ]


def test_a_rerun_and_an_observation_get_their_own_columns() -> None:
    """``rerun of`` and ``observed:<name>`` appear only when a trial carries them."""
    phase = _phase("p", x=IntParam(type="int", low=0, high=9))
    tables = _tables(
        _table(
            "p",
            [
                _row(
                    0, None, {"x": 3}, state="INTERRUPTED", duration=None, **{"observed:acc": None}
                ),
                _row(1, 0.4, {"x": 3}, rerun_of=0, **{"observed:acc": 0.91234567}),
            ],
            finished=1,
            uncounted=(("INTERRUPTED", 1),),
        ),
        winners={},
    )

    assert _render(tables, phase).splitlines()[4:] == [
        "          trial  eval_loss  x  observed:acc  state        time  rerun of",
        "              0          -  3             -  INTERRUPTED     -         -",
        "              1        0.4  3      0.912346  COMPLETE     0.5s         0",
    ]


def test_a_large_measured_value_prints_in_full_instead_of_exponent_form() -> None:
    """A value past six significant digits keeps plain notation, as float params do."""
    phase = _phase("p", x=IntParam(type="int", low=0, high=9))
    tables = _tables(
        _table(
            "p",
            [_row(0, 12345678.5, {"x": 3}, **{"observed:param_bytes": 8.8e6})],
        ),
        winners={},
    )

    assert _render(tables, phase).splitlines()[4:] == [
        "          trial   eval_loss  x  observed:param_bytes  state     time",
        "              0  12345678.5  3               8800000  COMPLETE  0.5s",
    ]


_REASON = (
    "trainer exited with code 3 after the evaluation step failed to produce a metric "
    "named 'eval_loss' in /data/runs/depth/trial_1/result.json"
)


def _failed_trial_lines(reason: str, width: int) -> list[str]:
    """Render a phase whose second trial failed with ``reason``; return the lines after its row."""
    phase = _phase("p", x=IntParam(type="int", low=0, high=9))
    tables = _tables(
        _table(
            "p",
            [
                _row(1, 0.4, {"x": 5}),
                _row(2, None, {"x": 7}, state="FAIL", reason=reason),
            ],
        ),
        winners={},
    )
    lines = _render(tables, phase, width=width).splitlines()
    return lines[lines.index("              2          -  7  FAIL      0.5s") + 1 :]


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (
            80,
            [
                "          reason: trainer exited with code 3 after the evaluation step failed to",
                "                  produce a metric named 'eval_loss' in",
                "                  /data/runs/depth/trial_1/result.json",
            ],
        ),
        (
            60,
            [
                "          reason: trainer exited with code 3 after the",
                "                  evaluation step failed to produce a metric",
                "                  named 'eval_loss' in",
                "                  /data/runs/depth/trial_1/result.json",
            ],
        ),
    ],
)
def test_a_failure_reason_wraps_under_its_row_with_a_hanging_indent(
    width: int, expected: list[str]
) -> None:
    """The whole reason prints under its row, wrapped to the width with a hanging indent."""
    lines = _failed_trial_lines(_REASON, width)

    assert lines == expected
    assert all(len(line) <= width for line in lines)


@pytest.mark.parametrize("width", [80, 60])
def test_a_path_longer_than_the_line_goes_whole_on_a_line_of_its_own(width: int) -> None:
    """A word wider than the room is never cut, so the reason's path can be copied from one line."""
    path = "/data/" + "/".join(f"directory_{index}" for index in range(8)) + "/result.json"

    lines = _failed_trial_lines(f"could not read {path} before it exits", width)

    assert lines == [
        "          reason: could not read",
        f"                  {path}",
        "                  before it exits",
    ]


def test_winners_from_different_generations_name_each_phases_own() -> None:
    """When winner generations differ the header says so and each phase names its source."""
    first = _phase("a", n_trials=1, x=IntParam(type="int", low=0, high=9))
    second = _phase("b", n_trials=1, x=IntParam(type="int", low=0, high=9))
    tables = _tables(
        _table("a", [_row(0, 0.2, {"x": 1}, winner=True)]),
        _table("b", [_row(0, 0.3, {"x": 2}, winner=True)]),
        winners={"a": (GENERATION, True), "b": ("0123456789abcdef0123456789abcdef", False)},
    )

    assert _render(tables, first, second) == "\n".join(
        [
            "phasesweep_starter",
            "winners come from different generations; each phase names its own",
            "",
            "a: 1 of 1 trial, eval_loss (lower is better)",
            f"  winner from generation {GENERATION} (published)",
            "          trial  eval_loss  x  state     time",
            "  winner      0        0.2  1  COMPLETE  0.5s",
            "",
            "b: 1 of 1 trial, eval_loss (lower is better)",
            "  winner from generation 0123456789abcdef0123456789abcdef (unpublished)",
            "          trial  eval_loss  x  state     time",
            "  winner      0        0.3  2  COMPLETE  0.5s",
        ]
    )


@pytest.mark.parametrize(
    ("represented", "published", "winners", "line"),
    [
        (None, None, {}, "no generation is published, so no winner is marked"),
        ("g2", "g1", {}, "unpublished generation g2"),
        ("g1", "g1", {}, "published generation g1"),
        ("g2", "g1", {"p": ("g2", False)}, "unpublished generation g2"),
        ("g1", "g1", {"p": ("g1", True)}, "published generation g1"),
    ],
)
def test_the_generation_line_covers_every_publication_state(
    represented: str | None,
    published: str | None,
    winners: dict[str, tuple[str, bool]],
    line: str,
) -> None:
    """The one generation line says which generation the marks come from, or that none is marked."""
    phase = _phase("p", x=IntParam(type="int", low=0, high=9))
    tables = _tables(
        _table("p", [_row(0, 0.5, {"x": 1}, winner=bool(winners))]),
        represented=represented,
        published=published,
        winners=winners,
    )

    assert _render(tables, phase).splitlines()[:2] == ["phasesweep_starter", line]


@pytest.mark.parametrize("width", [80, 60])
def test_the_unsettled_running_comment_wraps_as_comment_lines(width: int) -> None:
    """Wrapped lines each keep the ``# `` prefix and the words are the one-line comment's."""
    single = unsettled_running_comment(2)

    wrapped = unsettled_running_comment(2, width)

    assert "\n" not in single and single.startswith(
        "# No run is active; 2 trials are recorded RUNNING."
    )
    lines = wrapped.splitlines()
    assert len(lines) > 1 and all(line.startswith("# ") and len(line) <= width for line in lines)
    assert " ".join(line.removeprefix("# ") for line in lines) == single.removeprefix("# ")


SHORT_DIR = f"runs/depth/{SHORT_TRIAL_NAME}"
ATTEMPT_DIR = f"runs/phasesweep_starter/depth/{ATTEMPT_TRIAL_NAME}"


def _winner_tables(trial_dir: str | None, *, winner: bool = True) -> tuple[TrialTables, Phase]:
    """Build a two-trial phase whose trial 1 carries ``trial_dir`` and is the winner when asked."""
    phase = _phase("depth", n_trials=2, x=IntParam(type="int", low=0, high=9))
    rows = [
        _row(0, 0.5, {"x": 2}, trial_dir="runs/depth/trial_00000"),
        _row(1, 0.3, {"x": 4}, winner=winner, trial_dir=trial_dir),
    ]
    return _tables(_table("depth", rows), winners={"depth": (GENERATION, True)}), phase


def test_the_winning_trials_directory_prints_under_its_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that fits shares a line with its label, relative to the working directory."""
    monkeypatch.chdir(tmp_path)
    tables, phase = _winner_tables(str(tmp_path / SHORT_DIR))

    assert _render(tables, phase).splitlines()[3:] == [
        "depth: 2 of 2 trials, eval_loss (lower is better)",
        "          trial  eval_loss  x  state     time",
        "              0        0.5  2  COMPLETE  0.5s",
        "  winner      1        0.3  4  COMPLETE  0.5s",
        f"  winner trial 1 directory: {SHORT_DIR}",
    ]


@pytest.mark.parametrize("width", [80, 60])
def test_a_directory_too_wide_for_its_line_prints_whole_on_a_line_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """A real attempt directory is wider than the terminal, so its label stands alone above it."""
    monkeypatch.chdir(tmp_path)
    tables, phase = _winner_tables(str(tmp_path / ATTEMPT_DIR))

    lines = _render(tables, phase, width=width).splitlines()

    assert lines[3:] == [
        "depth: 2 of 2 trials, eval_loss (lower is better)",
        "          trial  eval_loss  x  state     time",
        "              0        0.5  2  COMPLETE  0.5s",
        "  winner      1        0.3  4  COMPLETE  0.5s",
        "  winner trial 1 directory:",
        f"    {ATTEMPT_DIR}",
    ]
    assert len(lines[-1]) > width


def test_a_directory_outside_the_working_directory_prints_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path under neither the working directory nor home keeps its full spelling."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    tables, phase = _winner_tables("/data/sweeps/depth/trial_00001")

    assert _render(tables, phase).splitlines()[-1] == (
        "  winner trial 1 directory: /data/sweeps/depth/trial_00001"
    )


def test_a_phase_without_a_winner_or_a_recorded_directory_prints_no_directory_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No winner mark means no line, and a winner that recorded no directory gets none either."""
    monkeypatch.chdir(tmp_path)
    unmarked, phase = _winner_tables(SHORT_DIR, winner=False)
    unrecorded, _ = _winner_tables(None)

    assert "directory" not in _render(unmarked, phase)
    assert "directory" not in _render(unrecorded, phase)


def test_the_directory_is_dim_on_a_terminal_and_nothing_else_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Colour wraps only the path of the directory line."""
    monkeypatch.chdir(tmp_path)
    tables, phase = _winner_tables(str(tmp_path / SHORT_DIR))

    styled = _render(tables, phase, color=True)

    assert ANSI.sub("", styled) == _render(tables, phase)
    assert styled.splitlines()[-1] == f"  winner trial 1 directory: \x1b[2m{SHORT_DIR}\x1b[0m"


@pytest.mark.parametrize("width", [80, 60])
def test_a_repeats_winner_names_each_winning_trials_directory_before_the_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """Every member of the winning group gets a line, in trial order, above the group table."""
    monkeypatch.chdir(tmp_path)
    phase = Phase(
        name="r",
        n_trials=4,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=10)},
        repeats=Repeats(key="seed", values=[0, 1]),
    )
    rows = [
        _row(
            number,
            value,
            {"x": x, "seed": seed},
            repeat_group=group,
            winner=x == 3,
            trial_dir=str(tmp_path / f"runs/r/trial_{number:05d}"),
        )
        for number, (x, seed, value, group) in enumerate(
            [(3, 0, 0.5, 1), (3, 1, 0.7, 1), (1, 0, 1.0, 3), (1, 1, 2.0, 3)]
        )
    ]
    groups = (
        RepeatGroupView(
            group=1, params={"x": 3}, n=2, expected=2, reduce="mean", aggregate=0.6, mean=0.6,
            sd=0.1, min=0.5, max=0.7, eligible=True, winner=True, observed={}, observed_n={},
        ),
    )  # fmt: skip
    tables = _tables(_table("r", rows, groups=groups))

    lines = _render(tables, phase, width=width).splitlines()

    start = lines.index("  winner trial 0 directory: runs/r/trial_00000")
    assert lines[start : start + 3] == [
        "  winner trial 0 directory: runs/r/trial_00000",
        "  winner trial 1 directory: runs/r/trial_00001",
        "",
    ]
    assert lines[start + 3].startswith("Repeat groups")


@pytest.mark.integration
def test_status_trials_styles_the_winner_on_a_terminal_unless_no_color_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escapes reach a terminal only when ``NO_COLOR`` is absent, and an empty value counts as set."""
    config = tmp_path / "starter.yaml"
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0

    shown: dict[str | None, str] = {}
    for no_color in (None, "1", ""):
        if no_color is None:
            monkeypatch.delenv("NO_COLOR", raising=False)
        else:
            monkeypatch.setenv("NO_COLOR", no_color)
        with terminal_stream("stdout", 100) as out:
            assert invoke_cli_boundary(["status", str(config), "--trials"], monkeypatch) == 0
        shown[no_color] = out.text

    assert "\x1b[1;32mwinner\x1b[0m" in shown[None]
    assert "\x1b" not in shown["1"]
    assert "\x1b" not in shown[""]
    assert ANSI.sub("", shown[None]) == shown["1"] == shown[""]
