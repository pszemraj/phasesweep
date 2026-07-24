"""Publication-transaction fault injection (review v0.5.14 / blockers 1, 2, 5).

The last-success pointer is the single final authoritative commit: every
injected failure must leave either the previous publication authoritative or
a fully valid ``complete`` generation published — never a pointer to a
failed, missing, or mismatched generation, and never a committed success
downgraded by later bookkeeping. Suite manifests must name exactly the
component generation that produced their winners, and terminal-state
persistence failures must never replace the primary exception.
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path

import pytest
import yaml

import phasesweep.engine.run as engine_run
from phasesweep import load_config, run_experiment
from phasesweep.config import IntParam, Phase, Sampler, Suite
from phasesweep.engine import NoFeasibleTrialError, read_winner
from phasesweep.engine.run import run_suite
from phasesweep.engine.state import (
    _generation_path,
    _generation_record_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _last_successful_generation_path,
)
from phasesweep.runtime.process import PhaseSweepShutdown, ShutdownCleanupReport
from tests.conftest import make_experiment, write_trainer, write_yaml

_TRAINER_BODY = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out")
parser.add_argument("--x", type=int, default=0)
args, _ = parser.parse_known_args()
print(f"x={args.x}")
"""


def _stored_experiment(tmp_path: Path, *, n_trials: int = 1):
    trainer = write_trainer(tmp_path / "trainer.py", _TRAINER_BODY)
    return make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=[
            Phase(
                name="p",
                n_trials=n_trials,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            )
        ],
    )


def test_projection_failure_before_pointer_keeps_prior_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A compatibility-projection failure leaves the previous pointer authoritative."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated projection failure")

    monkeypatch.setattr(engine_run, "_copy_yaml_projection", fail_projection)
    with (
        caplog.at_level(logging.WARNING, logger="phasesweep.engine.run"),
        pytest.raises(OSError, match="simulated projection failure"),
    ):
        run_experiment(experiment)

    second_generation = yaml.safe_load(_generation_path(experiment).read_text())["generation_id"]
    assert second_generation != first_generation
    # The interrupted generation's terminal record stays complete: the failure
    # handler cannot downgrade a terminal state (monotonic lifecycle).
    second_record = yaml.safe_load(
        _generation_record_path(experiment, second_generation).read_text()
    )
    assert second_record["state"] == "complete"
    assert any("Refusing to rewrite terminal generation" in r.message for r in caplog.records)
    # The pointer never advanced: the previous publication stays authoritative.
    assert _last_successful_generation_id(experiment) == first_generation
    published = read_winner(experiment, "p")
    assert published is not None
    assert published.generation_id == first_generation


def test_current_pointer_projection_failure_after_commit_is_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing fallible after the pointer commit can fail a published run."""
    experiment = _stored_experiment(tmp_path)
    original = engine_run._write_generation_state

    def flaky_state(experiment_arg, **kwargs: object):
        if kwargs.get("state") == "complete" and kwargs.get("publish_current"):
            raise OSError("simulated current-pointer projection failure")
        return original(experiment_arg, **kwargs)

    monkeypatch.setattr(engine_run, "_write_generation_state", flaky_state)
    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    published_generation = _last_successful_generation_id(experiment)
    assert published_generation is not None
    record = yaml.safe_load(_generation_record_path(experiment, published_generation).read_text())
    assert record["state"] == "complete"


def test_terminal_generation_states_are_monotonic(tmp_path: Path) -> None:
    """A published complete generation can never be rewritten as failed."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    engine_run._write_generation_state(
        experiment,
        generation_id=generation_id,
        state="failed",
        from_phase=None,
        publish_current=True,
        error_class="OSError",
    )

    record = yaml.safe_load(_generation_record_path(experiment, generation_id).read_text())
    assert record["state"] == "complete"
    assert _last_successful_generation_id(experiment) == generation_id


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param("experiment: t\ngeneration_id: ../evil\n", id="traversal-id"),
        pytest.param("experiment: other\ngeneration_id: {gid}\n", id="wrong-experiment"),
        pytest.param("experiment: t\ngeneration_id: no-such-generation\n", id="missing-record"),
    ],
)
def test_pointer_validation_fails_closed(tmp_path: Path, tamper: str) -> None:
    """An invalid pointer is treated as nothing-published, not trusted for reads."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    _last_successful_generation_path(experiment).write_text(tamper.format(gid=generation_id))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_pointer_to_non_complete_generation_is_not_authoritative(tmp_path: Path) -> None:
    """A pointer whose target record is not complete fails closed."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    record_path = _generation_record_path(experiment, generation_id)
    record = yaml.safe_load(record_path.read_text())
    record["state"] = "failed"
    record_path.write_text(yaml.safe_dump(record))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_suite_manifest_names_the_generation_that_produced_its_winners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interleaved external top-up cannot corrupt suite component provenance."""
    trainer = write_trainer(tmp_path / "trainer.py", _TRAINER_BODY)
    config = load_config(
        write_yaml(
            tmp_path,
            f"""
            suite: provenance_suite
            defaults:
              workdir: {tmp_path}/runs
              storage: sqlite:///{tmp_path}/suite.db
              provenance: {{revision: test-v1}}
              trial_command: "python {trainer} --out {{trial_dir}}/r.json {{overrides}}"
              metric:
                name: x
                goal: minimize
                extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            studies:
              - name: one
                phases:
                  - name: p
                    n_trials: 1
                    sampler: {{ type: random, seed: 0 }}
                    search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
            """,
        )
    )
    assert isinstance(config, Suite)
    component = config.experiment_for_study(config.studies[0])
    top_up = component.model_copy(
        update={"phases": [component.phases[0].model_copy(update={"n_trials": 2})]}
    )
    original_promotion = engine_run._apply_study_promotion

    def interleave(**kwargs: object):
        # Publish a newer component generation in the gap between the
        # component run returning and the suite recording provenance.
        run_experiment(top_up)
        return original_promotion(**kwargs)

    monkeypatch.setattr(engine_run, "_apply_study_promotion", interleave)
    results = run_suite(config)

    summary_path = tmp_path / "runs" / "provenance_suite" / "suite_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text())
    recorded = summary["studies"][0]["experiment_generation_id"]
    current = _last_successful_generation_id(component)

    # The pointer moved on to the top-up's generation, but the manifest still
    # names the generation whose immutable winners equal the suite's results.
    assert recorded != current
    assert recorded == results["one"]["p"].generation_id
    assert _generation_winner_path(component, recorded, "p").is_file()


def test_suite_state_write_failure_preserves_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed terminal-state write cannot replace SIGTERM cancellation."""
    config = load_config(
        write_yaml(
            tmp_path,
            f"""
            suite: masked_suite
            defaults:
              workdir: {tmp_path}/runs
              trial_command: "echo {{overrides}}"
              metric:
                name: x
                goal: minimize
                extractor: {{ type: log_regex, pattern: 'x=(?P<value>[0-9.eE+-]+)' }}
            studies:
              - name: one
                phases:
                  - name: p
                    n_trials: 1
                    search_space: {{}}
            """,
        )
    )
    assert isinstance(config, Suite)
    shutdown = PhaseSweepShutdown(
        signal.SIGTERM,
        ShutdownCleanupReport(signum=signal.SIGTERM, cleanup_confirmed=True, child_pgids=()),
    )

    def cancel(*_args: object, **_kwargs: object):
        raise shutdown

    original_state = engine_run._write_suite_generation_state

    def flaky_state(suite_arg, **kwargs: object):
        if kwargs.get("state") == "failed":
            raise OSError("simulated suite state persistence failure")
        return original_state(suite_arg, **kwargs)

    monkeypatch.setattr(engine_run, "_run_experiment_outcome", cancel)
    monkeypatch.setattr(engine_run, "_write_suite_generation_state", flaky_state)

    with (
        caplog.at_level(logging.ERROR, logger="phasesweep.engine.run"),
        pytest.raises(PhaseSweepShutdown) as exc_info,
    ):
        run_suite(config)

    assert exc_info.value is shutdown
    assert exc_info.value.signum == signal.SIGTERM
    assert exc_info.value.code == 128 + signal.SIGTERM
    assert any("failed to persist terminal failure state" in r.message for r in caplog.records)


def test_experiment_state_write_failure_preserves_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a control-flow exception during persistence cannot mask the failure."""
    trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        max_consecutive_failures=1,
    )
    original = engine_run._write_generation_state

    def flaky_state(experiment_arg, **kwargs: object):
        if kwargs.get("state") == "failed":
            raise SystemExit("simulated persistence interruption")
        return original(experiment_arg, **kwargs)

    monkeypatch.setattr(engine_run, "_write_generation_state", flaky_state)

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(experiment)
