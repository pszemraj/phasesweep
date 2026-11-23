"""The closing summary of a run: exact text of its blocks at 80 and 60 columns."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from phasesweep.engine.repeats import RepeatsRecord, spread
from phasesweep.engine.state import PhaseCounts, Winner
from phasesweep.run_report import Summary, nothing_to_run, run_totals, summary
from tests.conftest import (
    ATTEMPT_TRIAL_NAME,
    SHORT_TRIAL_NAME,
    grid_experiment,
    single_trial_confound,
)


def _counts(**counts: int) -> PhaseCounts:
    """Build phase counts from the state names a test names."""
    return PhaseCounts(**counts)


STARTER_WINNERS = {
    "depth": Winner(
        trial_number=0, params={"model.n_layers": 8}, effective_overrides={}, metric=0.3
    ),
    "learning_rate": Winner(
        trial_number=0, params={"optimizer.lr": 0.0003}, effective_overrides={}, metric=0.3
    ),
}


def test_summary_for_a_fresh_run() -> None:
    """The headline counts phases and trials, each phase shows its counts and winner."""
    block = summary(
        grid_experiment(),
        elapsed=3.05,
        counts={"depth": _counts(complete=2), "learning_rate": _counts(complete=2)},
        winners=STARTER_WINNERS,
        from_phase=None,
        written_root="runs/phasesweep_starter",
        written=[
            "generations/6413d5c1edd5475998cfadd4d3b59980/summary.yaml",
            "depth/trials.csv",
            "learning_rate/trials.csv",
        ],
        next_command="phasesweep status starter.yaml --trials",
        show_winners_command="phasesweep show-winners starter.yaml",
    )
    assert block.render(80, color=False) == [
        "",
        "Finished in 3.0s: 2 phases, 4 trials, 0 from a previous run",
        "",
        "  depth          2 complete",
        "                 winner  trial 0  eval_loss 0.3  model.n_layers=8",
        "  learning_rate  2 complete",
        "                 winner  trial 0  eval_loss 0.3  optimizer.lr=0.0003",
        "",
        "Written under runs/phasesweep_starter",
        "  generations/6413d5c1edd5475998cfadd4d3b59980/summary.yaml",
        "  depth/trials.csv",
        "  learning_rate/trials.csv",
        "",
        "Next: phasesweep status starter.yaml --trials",
    ]


def test_summary_of_a_resumed_run_with_failures_and_a_skipped_phase() -> None:
    """Zero counts are omitted, earlier trials count in the headline, skipped phases say so."""
    block = summary(
        grid_experiment(),
        elapsed=125.4,
        counts={
            "learning_rate": _counts(complete=3, infeasible=1, failed=2, interrupted=1, earlier=4)
        },
        winners=STARTER_WINNERS,
        from_phase="learning_rate",
        written_root="runs/phasesweep_starter",
        written=["learning_rate/trials.csv"],
        next_command="phasesweep status 'my config.yaml' --set workdir=/data/x --trials",
        show_winners_command="phasesweep show-winners 'my config.yaml' --set workdir=/data/x",
    )
    assert block.render(60, color=False) == [
        "",
        "Finished in 2m05s: 1 phase, 6 trials, 4 from a previous run",
        "",
        "  depth          skipped",
        "                 winner  trial 0  eval_loss 0.3",
        "                         model.n_layers=8",
        "  learning_rate  3 complete, 2 failed,",
        "                 1 infeasible (1 interrupted)",
        "                 winner  trial 0  eval_loss 0.3",
        "                         optimizer.lr=0.0003",
        "",
        "Written under runs/phasesweep_starter",
        "  learning_rate/trials.csv",
        "",
        # A command is pasted, so it is never broken, even past the width.
        "Next: phasesweep status 'my config.yaml' --set workdir=/data/x --trials",
    ]
    assert all(len(line) <= 60 for line in block.render(60, color=False)[:-1])


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            80,
            [
                "",
                "Finished in 2.1s: 2 phases, 8 trials, 3 from a previous run",
                "",
                "  depth          3 complete, 1 failed (1 interrupted)",
                "                 winner  trial 0  eval_loss 0.3  model.n_layers=8",
                "  learning_rate  4 complete (2 interrupted)",
                "                 winner  trial 0  eval_loss 0.3  optimizer.lr=0.0003",
                "",
                "Next: phasesweep status starter.yaml --trials",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "",
                "Finished in 2.1s: 2 phases, 8 trials, 3 from a previous run",
                "",
                "  depth          3 complete, 1 failed (1 interrupted)",
                "                 winner  trial 0  eval_loss 0.3",
                "                         model.n_layers=8",
                "  learning_rate  4 complete (2 interrupted)",
                "                 winner  trial 0  eval_loss 0.3",
                "                         optimizer.lr=0.0003",
                "",
                "Next: phasesweep status starter.yaml --trials",
            ],
            id="60",
        ),
    ],
)
def test_summary_counts_the_trials_toward_n_trials_and_names_interrupted_ones_apart(
    width: int, expected: list[str]
) -> None:
    """Both reports of one study agree: the status heading's count first, interrupted rows in parentheses."""
    counts = {
        "depth": _counts(complete=3, failed=1, interrupted=1, earlier=3),
        "learning_rate": _counts(complete=4, interrupted=2),
    }
    block = summary(
        grid_experiment(),
        elapsed=2.1,
        counts=counts,
        winners=STARTER_WINNERS,
        from_phase=None,
        written_root="runs/phasesweep_starter",
        written=[],
        next_command="phasesweep status starter.yaml --trials",
        show_winners_command="phasesweep show-winners starter.yaml",
    )

    assert block.render(width, color=False) == expected
    assert all(len(line) <= width for line in block.render(width, color=False))
    assert run_totals(counts) == "2 phases, 8 trials, 3 from a previous run"
    replay = nothing_to_run(
        elapsed=0.4, counts=counts, generation_id="g" * 32, next_command="phasesweep status c.yaml"
    )
    assert replay.render(200, color=False)[1].startswith(
        "Finished in 0.4s: nothing to run, 8 trials already finished"
    )


def _flagged_block(*names: str) -> dict[str, object]:
    """Build a one-trial confound block whose checks ``names`` are flagged."""
    block = single_trial_confound(0)
    checks = block["checks"]
    for name in names:
        checks[name]["verdict"] = "heterogeneous"
    return block


def _flagged_summary(depth: tuple[str, ...], learning_rate: tuple[str, ...]) -> Summary:
    """Build the starter's summary whose winners carry blocks that flag the named checks."""
    winners = {
        name: Winner(
            trial_number=0,
            params=STARTER_WINNERS[name].params,
            effective_overrides={},
            metric=0.3,
            confound=_flagged_block(*flagged),
        )
        for name, flagged in (("depth", depth), ("learning_rate", learning_rate))
    }
    return summary(
        grid_experiment(),
        elapsed=3.0,
        counts={"depth": _counts(complete=2), "learning_rate": _counts(complete=2)},
        winners=winners,
        from_phase=None,
        written_root="runs/x",
        written=[],
        next_command="phasesweep status c.yaml --trials",
        show_winners_command="phasesweep show-winners c.yaml",
    )


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            80,
            [
                "  depth          2 complete",
                "                 winner  trial 0  eval_loss 0.3  model.n_layers=8",
                "                         flagged: survivorship",
                "                         details: phasesweep show-winners c.yaml",
                "  learning_rate  2 complete",
                "                 winner  trial 0  eval_loss 0.3  optimizer.lr=0.0003",
                "                         flagged: evaluation_point, survivorship, tie",
                "                         details: phasesweep show-winners c.yaml",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "  depth          2 complete",
                "                 winner  trial 0  eval_loss 0.3",
                "                         model.n_layers=8",
                "                         flagged: survivorship",
                "                         details: phasesweep show-winners c.yaml",
                "  learning_rate  2 complete",
                "                 winner  trial 0  eval_loss 0.3",
                "                         optimizer.lr=0.0003",
                "                         flagged: evaluation_point,",
                "                                  survivorship, tie",
                "                         details: phasesweep show-winners c.yaml",
            ],
            id="60",
        ),
    ],
)
def test_summary_names_the_checks_a_winners_selection_flagged(
    width: int, expected: list[str]
) -> None:
    """A flagged winner's lines name each check, then the show-winners command, never broken."""
    block = _flagged_summary(("survivorship",), ("evaluation_point", "survivorship", "tie"))
    lines = block.render(width, color=False)

    assert lines[3:-2] == expected
    # A command is pasted, so the details line is the one line that may pass the width.
    assert all(len(line) <= width for line in lines if "details:" not in line)


def test_summary_adds_no_flag_line_for_a_winner_without_flagged_checks() -> None:
    """Clean checks, or no confound block at all, leave the winner's lines as they were."""
    clean = _flagged_summary((), ())
    assert not any("flagged" in line for line in clean.render(80, color=False))
    assert not any(
        "flagged" in line
        for line in _summary_with_dirs(()).render(80, color=False)  # winners without a block
    )


def test_summary_paints_only_the_flag_label_yellow_on_a_terminal() -> None:
    """The label is the warning style; redirected output carries no escape and the same text."""
    block = _flagged_summary(("survivorship",), ())
    styled = block.render(80, color=True)

    assert "                         \x1b[33mflagged:\x1b[0m survivorship" in styled
    assert re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(styled)) == "\n".join(
        block.render(80, color=False)
    )


def test_a_phase_that_only_holds_interrupted_trials_still_says_so() -> None:
    """With nothing counted, the clause reads ``no trials`` and the interrupted ones follow it."""
    block = summary(
        grid_experiment(),
        elapsed=2.1,
        counts={"depth": _counts(interrupted=1)},
        winners={},
        from_phase=None,
        written_root="runs/phasesweep_starter",
        written=[],
        next_command="phasesweep status starter.yaml --trials",
        show_winners_command="phasesweep show-winners starter.yaml",
    )

    assert "  depth          no trials (1 interrupted)" in block.render(80, color=False)


def test_summary_omits_the_written_section_when_nothing_was_written() -> None:
    """A run that wrote no file this time lists none."""
    block = summary(
        grid_experiment(),
        elapsed=0.2,
        counts={"depth": _counts(complete=2), "learning_rate": _counts(complete=2, earlier=2)},
        winners=STARTER_WINNERS,
        from_phase=None,
        written_root="runs/phasesweep_starter",
        written=[],
        next_command="phasesweep status starter.yaml --trials",
        show_winners_command="phasesweep show-winners starter.yaml",
    )
    assert "Written under" not in "\n".join(block.render(80, color=False))


SHORT_TRIAL_DIR = Path("/data/runs/depth") / SHORT_TRIAL_NAME
ATTEMPT_TRIAL_DIR = (
    Path("/data/sweeps/phasesweep_starter/runs/phasesweep_starter/depth") / ATTEMPT_TRIAL_NAME
)


def _summary_with_dirs(
    winner_dirs: tuple[Path, ...], *, winners: dict[str, Winner] | None = None
) -> Summary:
    """Build the starter's summary whose ``depth`` winner stands on trials in ``winner_dirs``."""
    return summary(
        grid_experiment(),
        elapsed=3.05,
        counts={
            "depth": PhaseCounts(complete=2, winner_dirs=winner_dirs),
            "learning_rate": PhaseCounts(complete=2),
        },
        winners=STARTER_WINNERS if winners is None else winners,
        from_phase=None,
        written_root="runs/x",
        written=[],
        next_command="phasesweep status c.yaml --trials",
        show_winners_command="phasesweep show-winners c.yaml",
    )


@pytest.mark.parametrize(
    ("width", "directory", "expected"),
    [
        pytest.param(
            80,
            SHORT_TRIAL_DIR,
            ["                 /data/runs/depth/trial_00001"],
            id="short-80",
        ),
        pytest.param(
            60,
            SHORT_TRIAL_DIR,
            [
                "                 winner  trial 0  eval_loss 0.3",
                "                         model.n_layers=8",
                "                 /data/runs/depth/trial_00001",
            ],
            id="short-60",
        ),
        pytest.param(
            80,
            ATTEMPT_TRIAL_DIR,
            [
                "                 /data/sweeps/phasesweep_starter/runs/phasesweep_starter/depth/",
                f"                 {ATTEMPT_TRIAL_NAME}",
            ],
            id="long-80",
        ),
        pytest.param(
            60,
            ATTEMPT_TRIAL_DIR,
            [
                "                 winner  trial 0  eval_loss 0.3",
                "                         model.n_layers=8",
                "                 /data/sweeps/phasesweep_starter/runs/",
                "                 phasesweep_starter/depth/",
                f"                 {ATTEMPT_TRIAL_NAME}",
            ],
            id="long-60",
        ),
    ],
)
def test_summary_names_the_winners_directory_under_its_winner_line(
    width: int, directory: Path, expected: list[str]
) -> None:
    """The directory sits under the winner line, broken at slashes, its last component whole."""
    lines = _summary_with_dirs((directory,)).render(width, color=False)

    start = lines.index("  depth          2 complete")
    block = lines[start + 1 : lines.index("  learning_rate  2 complete")]
    assert block[-len(expected) :] == expected
    assert block[0].startswith("                 winner  trial 0")
    # Only a path component wider than the terminal may overrun, and it keeps a line to itself.
    overrunning = [line for line in block if len(line) > width]
    assert overrunning == ([expected[-1]] if directory == ATTEMPT_TRIAL_DIR else [])


_REPEATS_PARENT = Path("/data/sweeps/phasesweep_starter/runs/phasesweep_starter/depth")
_REPEATS_MEMBER_DIRS = tuple(
    _REPEATS_PARENT / f"trial_{number:05d}__generation_6413d5c1edd5475998cfadd4d3b59980"
    f"__attempt_{number:x}f1e2d3c4b5a69788796a5b4c3d2e1f0"
    for number in (4, 5)
)


@pytest.mark.parametrize(
    ("width", "directories", "expected"),
    [
        pytest.param(
            80,
            (Path("/data/runs/depth/trial_00004"), Path("/data/runs/depth/trial_00005")),
            [
                "                 /data/runs/depth/",
                "                   trial_00004",
                "                   trial_00005",
            ],
            id="short-80",
        ),
        pytest.param(
            80,
            _REPEATS_MEMBER_DIRS,
            [
                "                 /data/sweeps/phasesweep_starter/runs/phasesweep_starter/depth/",
                f"                   {_REPEATS_MEMBER_DIRS[0].name}",
                f"                   {_REPEATS_MEMBER_DIRS[1].name}",
            ],
            id="attempt-80",
        ),
        pytest.param(
            60,
            _REPEATS_MEMBER_DIRS,
            [
                "                 /data/sweeps/phasesweep_starter/runs/",
                "                 phasesweep_starter/depth/",
                f"                   {_REPEATS_MEMBER_DIRS[0].name}",
                f"                   {_REPEATS_MEMBER_DIRS[1].name}",
            ],
            id="attempt-60",
        ),
    ],
)
def test_summary_names_every_winning_trials_directory_of_a_repeats_phase(
    width: int, directories: tuple[Path, ...], expected: list[str]
) -> None:
    """A repeats winner stands on each member: the shared parent prints once, then one line each."""
    repeats: RepeatsRecord = {
        "key": "seed",
        "reduce": "mean",
        "spread": spread([0.2, 0.4]),
        "members": [
            {
                "trial_number": number,
                "repeat": number - 4,
                "metric": metric,
                "generation_id": "g",
                "attempt_id": f"attempt-{number}",
                "objective_provenance": {},
                "trainer_input": {},
                "constraints": {},
                "observed": {},
            }
            for number, metric in ((4, 0.2), (5, 0.4))
        ],
    }
    repeated = Winner(
        trial_number=5,
        params={"model.n_layers": 8},
        effective_overrides={},
        metric=0.3,
        repeats=repeats,
    )
    lines = _summary_with_dirs(directories, winners={**STARTER_WINNERS, "depth": repeated}).render(
        width, color=False
    )

    start = lines.index("  learning_rate  2 complete")
    assert lines[start - len(expected) : start] == expected
    # A member's own name is the only text that may overrun, and each keeps a line to itself.
    names = {path.name for path in directories}
    assert [line for line in lines if len(line) > width] == [
        line for line in expected if line.strip() in names and len(line) > width
    ]


def test_summary_prints_no_directory_without_a_winner_or_a_recorded_directory() -> None:
    """A phase with no winner, and a winner whose trial recorded no directory, add no line."""
    without_dirs = _summary_with_dirs(()).render(80, color=False)
    without_winner = _summary_with_dirs(
        (SHORT_TRIAL_DIR,), winners={"learning_rate": STARTER_WINNERS["learning_rate"]}
    ).render(80, color=False)

    assert not any("/data" in line for line in without_dirs)
    assert not any("/data" in line for line in without_winner)
    assert "  depth          2 complete" in without_winner
    assert "                 winner  trial 0  eval_loss 0.3  model.n_layers=8" not in without_winner


def test_summary_dims_the_winners_directory_on_a_terminal() -> None:
    """Colour wraps the path alone and leaves its indentation plain."""
    lines = _summary_with_dirs((SHORT_TRIAL_DIR,)).render(80, color=True)

    assert f"                 \x1b[2m{SHORT_TRIAL_DIR}\x1b[0m" in lines


@pytest.mark.parametrize("width", [80, 60])
def test_nothing_to_run_is_one_sentence_and_a_next_line(width: int) -> None:
    """A replay says it launched nothing and which generation holds the result, then where to look."""
    block = nothing_to_run(
        elapsed=0.42,
        counts={"depth": _counts(complete=2), "learning_rate": _counts(complete=2)},
        generation_id="6413d5c1edd5475998cfadd4d3b59980",
        next_command="phasesweep status starter.yaml --trials",
    )
    sentence = {
        80: [
            "Finished in 0.4s: nothing to run, 4 trials already finished, and generation",
            "6413d5c1edd5475998cfadd4d3b59980 already publishes these winners.",
        ],
        60: [
            "Finished in 0.4s: nothing to run, 4 trials already finished,",
            "and generation 6413d5c1edd5475998cfadd4d3b59980 already",
            "publishes these winners.",
        ],
    }[width]
    assert block.render(width, color=False) == [
        "",
        *sentence,
        "",
        "Next: phasesweep status starter.yaml --trials",
    ]
    assert block.render(width, color=True)[-1] == (
        "\x1b[1mNext:\x1b[0m phasesweep status starter.yaml --trials"
    )


def test_banner_and_summary_style_only_labels_paths_and_the_winner_mark() -> None:
    """Colour marks the labels bold, paths dim and ``winner`` bold green, and changes no width."""
    shown = summary(
        grid_experiment(),
        elapsed=3.0,
        counts={"depth": _counts(complete=2), "learning_rate": _counts(complete=2)},
        winners=STARTER_WINNERS,
        from_phase=None,
        written_root="runs/x",
        written=["depth/trials.csv"],
        next_command="phasesweep status c.yaml --trials",
        show_winners_command="phasesweep show-winners c.yaml",
    ).render(80, color=True)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(shown))
    assert plain.splitlines() == summary(
        grid_experiment(),
        elapsed=3.0,
        counts={"depth": _counts(complete=2), "learning_rate": _counts(complete=2)},
        winners=STARTER_WINNERS,
        from_phase=None,
        written_root="runs/x",
        written=["depth/trials.csv"],
        next_command="phasesweep status c.yaml --trials",
        show_winners_command="phasesweep show-winners c.yaml",
    ).render(80, color=False)
    text = "\n".join(shown)
    assert "\x1b[1;32mwinner\x1b[0m" in text
    assert "\x1b[1mFinished in 3.0s\x1b[0m" in text
    assert "\x1b[1mWritten under\x1b[0m" in text
    assert "\x1b[2mdepth/trials.csv\x1b[0m" in text
    assert "\x1b[1mNext:\x1b[0m" in text
