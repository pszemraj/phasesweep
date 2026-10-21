"""Run summaries preserve the selected winner's reporting and comparison details."""

from __future__ import annotations

from phasesweep.engine.repeats import RepeatsRecord, spread
from phasesweep.engine.selection import _winner_summary_item
from phasesweep.engine.state import Winner, WinnerSource


def test_summary_retains_repeat_observations_and_confound_details() -> None:
    """A compact summary still carries the data needed to interpret the winning metric."""
    repeats: RepeatsRecord = {
        "key": "seed",
        "reduce": "mean",
        "spread": spread([0.4, 0.6]),
        "members": [
            {
                "trial_number": number,
                "repeat": number,
                "metric": metric,
                "generation_id": "generation-a",
                "attempt_id": f"attempt-{number}",
                "objective_provenance": {},
                "trainer_input": {},
                "constraints": {},
                "observed": {"heldout": metric + 0.1},
            }
            for number, metric in enumerate([0.4, 0.6])
        ],
    }
    winner = Winner(
        trial_number=1,
        params={"x": 1},
        effective_overrides={"x": 1},
        metric=0.5,
        source=WinnerSource("phase_trial", "p", 1, "generation-a", "attempt-1"),
        repeats=repeats,
        observed={"heldout": 0.6},
        confound={"checks": {"tie": {"verdict": "n/a", "reason": "one group"}}},
    )

    summary = _winner_summary_item("p", winner)

    assert summary["repeats"] == winner.repeats
    assert summary["observed"] == winner.observed
    assert summary["confound"] == winner.confound
