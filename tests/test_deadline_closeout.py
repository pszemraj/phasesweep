"""The close of a run whose wallclock budget ran out: process cleanup only, with a finished trainer's evidence read left to the next invocation."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.evidence.evaluation import TrialContext, run_extractor
from phasesweep.evidence.models import ObjectiveExtractor
from phasesweep.runtime.json import JsonObject
from tests.conftest import make_experiment, phase_trials, write_flaky_trainer

_RUN_BUDGET_SECONDS = 2.0
_READ_SECONDS = 3.0
"""How long one evidence read blocks unless a deadline cuts it, as a slow tracker's would."""


@pytest.mark.integration
def test_a_run_budget_spent_on_an_evidence_read_leaves_the_read_to_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run deadline that cuts a finished trainer's evidence read ends the run near its budget with the trial pending and no second read; the next run scores it without launching a trainer."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0)
    read_deadlines: list[float | None] = []

    def blocking_read(
        ctx: TrialContext,
        cfg: ObjectiveExtractor,
        *,
        deadline: float | None = None,
        provenance: JsonObject | None = None,
    ) -> float:
        read_deadlines.append(deadline)
        wait = _READ_SECONDS
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.monotonic()) + 0.05)
        time.sleep(wait)
        return run_extractor(ctx, cfg, deadline=deadline, provenance=provenance)

    monkeypatch.setattr("phasesweep.engine.trial.run_extractor", blocking_read)
    budgeted = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=1).model_copy(
        update={"timeout_seconds_per_run": _RUN_BUDGET_SECONDS}
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="while reading the evidence of trial 0, which finished"):
        run_experiment(budgeted)
    elapsed = time.monotonic() - started

    assert elapsed < _RUN_BUDGET_SECONDS + _READ_SECONDS / 2, elapsed
    assert len(read_deadlines) == 1 and read_deadlines[0] is not None
    assert [t.state.name for t in phase_trials(budgeted)] == ["RUNNING"]

    unbudgeted = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=1)
    winners = run_experiment(unbudgeted)
    assert winners["p"].trial_number == 0
    assert [t.state.name for t in phase_trials(unbudgeted)] == ["COMPLETE"]
    assert read_deadlines[1:] == [None]
    assert len(calls.read_text().splitlines()) == 1
