"""Publication-transaction fault injection (review v0.5.15 / blocker 3).

The last-success pointer is the single final authoritative commit: every
injected failure must leave either the previous publication authoritative or
a fully valid published generation -- never a pointer to a failed, missing,
or mismatched generation, a current pointer stuck at a non-terminal state, or
a committed success downgraded by later bookkeeping. The per-generation
lifecycle record is write-once and purely informational (written *after* the
pointer commits); pointer validation instead reads back the generation's own
immutable summary artifact. Suite manifests must name exactly the component
generation that produced their winners, and terminal-state persistence
failures must never replace the primary exception.
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
from phasesweep.engine import NoFeasibleTrialError, TerminalReport, read_status, read_winner
from phasesweep.engine.run import run_suite
from phasesweep.engine.state import (
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _last_successful_generation_path,
    _last_successful_suite_generation_id,
    _suite_generation_record_path,
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


def _current_pointer_state(experiment) -> str | None:
    """Read the current-pointer file's own ``state`` label."""
    payload = yaml.safe_load(_generation_path(experiment).read_text())
    return payload.get("state") if isinstance(payload, dict) else None


def _record_state(experiment, generation_id: str) -> str | None:
    """Read one generation's immutable record ``state`` label."""
    payload = yaml.safe_load(_generation_record_path(experiment, generation_id).read_text())
    return payload.get("state") if isinstance(payload, dict) else None


# --------------------------------------------------------------------------
# Pre-commit validation failure (step 2): prior publication stays authoritative.
# --------------------------------------------------------------------------


def test_precommit_validation_failure_keeps_prior_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-commit validation failure leaves the prior publication authoritative."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated pre-commit validation failure")

    monkeypatch.setattr(engine_run, "_validate_generation_publishable", fail_validation)

    captured: list[str] = []

    def capture(report: TerminalReport) -> None:
        captured.append(report.generation_id)

    with pytest.raises(RuntimeError, match="simulated pre-commit validation failure"):
        run_experiment(experiment, terminal_callback=capture)

    assert len(captured) == 1
    second_generation = captured[0]
    assert second_generation != first_generation

    # The failed generation's own record and current pointer both name the
    # specific publication_failed outcome, not a generic failure.
    assert _record_state(experiment, second_generation) == "publication_failed"
    assert _current_pointer_state(experiment) == "publication_failed"
    payload = yaml.safe_load(_generation_record_path(experiment, second_generation).read_text())
    assert payload["error_class"] == "RuntimeError"

    # The prior publication is untouched: the pointer never advanced.
    assert _last_successful_generation_id(experiment) == first_generation
    published = read_winner(experiment, "p")
    assert published is not None
    assert published.generation_id == first_generation


def test_pointer_commit_failure_keeps_prior_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure writing the last-success pointer itself is the same failure shape as validation."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    pointer_path = _last_successful_generation_path(experiment)
    original_write = engine_run._write_yaml_atomic

    def flaky_write(path: Path, payload: object) -> None:
        if path == pointer_path:
            raise OSError("simulated pointer commit failure")
        return original_write(path, payload)

    monkeypatch.setattr(engine_run, "_write_yaml_atomic", flaky_write)

    with pytest.raises(OSError, match="simulated pointer commit failure"):
        run_experiment(experiment)

    second_generation = yaml.safe_load(_generation_path(experiment).read_text())["generation_id"]
    assert second_generation != first_generation
    assert _record_state(experiment, second_generation) == "publication_failed"
    assert _current_pointer_state(experiment) == "publication_failed"
    assert _last_successful_generation_id(experiment) == first_generation


# --------------------------------------------------------------------------
# Post-commit failures (steps 4 and 5): the run must still succeed.
# --------------------------------------------------------------------------


def test_record_write_failure_after_commit_leaves_run_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure writing the immutable record after the pointer commit must not fail the run.

    The pointer is validated against the generation's own summary artifact,
    not the record, so publication is unaffected even though the record
    itself never gets written.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    def fail_record_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated record write failure")

    monkeypatch.setattr(engine_run, "_write_generation_record_once", fail_record_write)

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.run"):
        winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    second_generation = _last_successful_generation_id(experiment)
    assert second_generation is not None
    assert second_generation != first_generation
    assert any(
        "failed to write the immutable generation record" in r.message for r in caplog.records
    )
    # The record itself never got created; publication still succeeded.
    assert not _generation_record_path(experiment, second_generation).is_file()


def test_cache_projection_failure_after_commit_leaves_run_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A legacy compatibility-cache projection failure must not fail the run.

    Flips the pre-v0.5.15 behavior (projection failures used to precede and
    block the pointer commit): projections are now a post-commit, best-effort
    cache, and reads never depend on them once any generation has published.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated projection failure")

    monkeypatch.setattr(engine_run, "_copy_yaml_projection", fail_projection)

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.run"):
        winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    second_generation = _last_successful_generation_id(experiment)
    assert second_generation is not None
    assert second_generation != first_generation
    assert _record_state(experiment, second_generation) == "published"
    assert any(
        "failed to refresh the current-generation pointer or compatibility caches" in r.message
        for r in caplog.records
    )
    # Reads are unaffected: once a generation has published, they resolve the
    # generation-scoped artifact directly rather than the stale legacy cache.
    # The winning trial itself still belongs to the first generation (the
    # target trial count was already satisfied, so no new trial ran); what
    # matters is that the read resolves via the new last-success pointer
    # rather than a stale root-level copy.
    published_winner_path = _generation_winner_path(experiment, second_generation, "p")
    assert published_winner_path.is_file()
    published = read_winner(experiment, "p")
    assert published is not None
    assert published.generation_id == first_generation


# --------------------------------------------------------------------------
# No failure path leaves the current pointer non-terminal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inject",
    ["precommit_validation", "pointer_commit", "ordinary_execution"],
)
def test_no_failure_path_leaves_current_pointer_non_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inject: str,
) -> None:
    """Every injected failure drives the current pointer to a terminal state."""
    experiment = _stored_experiment(tmp_path)

    if inject == "precommit_validation":

        def fail(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(engine_run, "_validate_generation_publishable", fail)
        expected_exc = RuntimeError
    elif inject == "pointer_commit":
        pointer_path = _last_successful_generation_path(experiment)
        original_write = engine_run._write_yaml_atomic

        def flaky_write(path: Path, payload: object) -> None:
            if path == pointer_path:
                raise OSError("simulated failure")
            return original_write(path, payload)

        monkeypatch.setattr(engine_run, "_write_yaml_atomic", flaky_write)
        expected_exc = OSError
    else:
        trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
        experiment = make_experiment(
            workdir=tmp_path / "runs",
            trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
            n_trials=1,
            max_consecutive_failures=1,
        )
        expected_exc = NoFeasibleTrialError

    with pytest.raises(expected_exc):
        run_experiment(experiment)

    assert _current_pointer_state(experiment) in engine_run._TERMINAL_GENERATION_STATES


# --------------------------------------------------------------------------
# The immutable record is write-once.
# --------------------------------------------------------------------------


def test_generation_record_is_write_once(tmp_path: Path) -> None:
    """A published generation's record can never be rewritten, even to the same state."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    first_content = _generation_record_path(experiment, generation_id).read_bytes()

    engine_run._write_generation_state(
        experiment,
        generation_id=generation_id,
        state="failed",
        from_phase=None,
        publish_current=True,
        error_class="OSError",
    )

    # The record is untouched -- refused, not downgraded.
    assert _generation_record_path(experiment, generation_id).read_bytes() == first_content
    assert _record_state(experiment, generation_id) == "published"
    # Pointer validation reads the summary, not the record, so publication
    # still resolves correctly regardless of this direct (out-of-band) call.
    assert _last_successful_generation_id(experiment) == generation_id

    # A second attempt at the SAME state is refused too -- not "monotonic",
    # truly write-once.
    engine_run._write_generation_state(
        experiment,
        generation_id=generation_id,
        state="published",
        from_phase=None,
        publish_current=False,
    )
    assert _generation_record_path(experiment, generation_id).read_bytes() == first_content


def test_successful_publication_never_logs_a_record_refusal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The happy path never exercises the write-once refusal branch.

    The post-commit current-pointer refresh passes ``write_record=False``, so
    a normal successful publication must not log the "Refusing to rewrite"
    warning -- that message has to stay a real anomaly signal, not routine
    noise on every publication.
    """
    experiment = _stored_experiment(tmp_path)
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.run"):
        run_experiment(experiment)
        run_experiment(experiment)  # republish onto the same storage
    assert not [r for r in caplog.records if "Refusing to rewrite" in r.message]


# --------------------------------------------------------------------------
# Pointer validation moved from record state to artifact identity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param("experiment: t\ngeneration_id: ../evil\n", id="traversal-id"),
        pytest.param("experiment: other\ngeneration_id: {gid}\n", id="wrong-experiment"),
        pytest.param("experiment: t\ngeneration_id: no-such-generation\n", id="missing-generation"),
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


def test_pointer_to_generation_with_tampered_summary_is_not_authoritative(tmp_path: Path) -> None:
    """A pointer whose target's own summary no longer names it fails closed.

    Pointer validation now reads back the generation's immutable *summary*,
    not the (post-commit, best-effort) lifecycle record -- so tampering the
    record no longer has any effect on publication status; tampering the
    summary does.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["generation_id"] = "not-this-generation"
    summary_path.write_text(yaml.safe_dump(summary))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_pointer_to_generation_with_missing_summary_is_not_authoritative(tmp_path: Path) -> None:
    """A pointer whose target has no summary at all fails closed."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    _generation_summary_path(experiment, generation_id).unlink()

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_tampering_the_record_state_does_not_affect_publication_status(tmp_path: Path) -> None:
    """The record is informational only; publication status ignores its state entirely."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    record_path = _generation_record_path(experiment, generation_id)
    record = yaml.safe_load(record_path.read_text())
    record["state"] = "failed"
    record_path.write_text(yaml.safe_dump(record))

    # Unlike the pre-v0.5.15 design, this has no effect: the record is not consulted.
    assert _last_successful_generation_id(experiment) == generation_id
    assert read_winner(experiment, "p") is not None


# --------------------------------------------------------------------------
# Identity fields (read_status): pinned reads of a failed-publication
# generation, and single-capture consistency.
# --------------------------------------------------------------------------


def test_pinned_read_of_failed_publication_generation_reports_truthful_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned read of a failed-publication generation is honest about its status."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = _last_successful_generation_id(experiment)
    assert first_generation is not None

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated pre-commit validation failure")

    monkeypatch.setattr(engine_run, "_validate_generation_publishable", fail_validation)

    captured: list[str] = []

    def capture(report: TerminalReport) -> None:
        captured.append(report.generation_id)

    with pytest.raises(RuntimeError):
        run_experiment(experiment, terminal_callback=capture)

    failed_generation = captured[0]
    assert failed_generation != first_generation

    status = read_status(experiment, generation_id=failed_generation)
    assert status["represented_generation_id"] == failed_generation
    assert status["is_published"] is False
    assert status["published_generation_id"] == first_generation
    assert status["current_generation_id"] == failed_generation
    # The generation's own (unpublished) winner is still readable pinned.
    assert status["phases"][0]["winner_present"] is True


def test_read_status_single_captures_the_published_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer swap mid-read cannot mix identities across one status object.

    Monkeypatches the published-pointer resolver to return a different id on
    each call. If ``read_status`` (or anything it calls) re-resolved the
    pointer per field, this would produce an internally inconsistent payload
    (e.g. ``published_generation_id`` naming one generation while
    ``winner_present``/``summary_present`` reflect another). Single capture
    means every field in one call is consistent with whichever id the *first*
    (and only) resolution returned.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    real_generation = _last_successful_generation_id(experiment)
    assert real_generation is not None

    calls: list[int] = []

    def swapping_resolver(_experiment: object) -> str | None:
        calls.append(1)
        return real_generation if len(calls) == 1 else "generation-does-not-exist"

    monkeypatch.setattr("phasesweep.engine.read._last_successful_generation_id", swapping_resolver)

    status = read_status(experiment)

    assert len(calls) == 1
    assert status["published_generation_id"] == real_generation
    assert status["represented_generation_id"] == real_generation
    assert status["is_published"] is True
    assert status["summary_present"] is True
    assert status["phases"][0]["winner_present"] is True


# --------------------------------------------------------------------------
# Suite equivalents for the core cases.
# --------------------------------------------------------------------------


def _stored_suite_config(tmp_path: Path) -> Suite:
    trainer = write_trainer(tmp_path / "trainer.py", _TRAINER_BODY)
    config = load_config(
        write_yaml(
            tmp_path,
            f"""
            suite: pub_suite
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
    return config


def test_suite_precommit_validation_failure_keeps_prior_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suite mirror: a pre-commit validation failure keeps the prior suite publication."""
    suite = _stored_suite_config(tmp_path)
    run_suite(suite)
    first_generation = _last_successful_suite_generation_id(suite)
    assert first_generation is not None

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated suite validation failure")

    monkeypatch.setattr(engine_run, "_validate_suite_generation_publishable", fail_validation)

    with pytest.raises(RuntimeError, match="simulated suite validation failure"):
        run_suite(suite)

    second_generation = yaml.safe_load(
        (tmp_path / "runs" / "pub_suite" / "suite_generation.yaml").read_text()
    )["suite_generation_id"]
    assert second_generation != first_generation
    record = yaml.safe_load(_suite_generation_record_path(suite, second_generation).read_text())
    assert record["state"] == "publication_failed"
    assert _last_successful_suite_generation_id(suite) == first_generation


def test_suite_cache_projection_failure_after_commit_leaves_run_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Suite mirror: a post-commit projection failure must not fail the suite run."""
    suite = _stored_suite_config(tmp_path)
    run_suite(suite)
    first_generation = _last_successful_suite_generation_id(suite)
    assert first_generation is not None

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated suite projection failure")

    monkeypatch.setattr(engine_run, "_copy_yaml_projection", fail_projection)

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.run"):
        results = run_suite(suite)

    assert set(results) == {"one"}
    second_generation = _last_successful_suite_generation_id(suite)
    assert second_generation is not None
    assert second_generation != first_generation
    record = yaml.safe_load(_suite_generation_record_path(suite, second_generation).read_text())
    assert record["state"] == "published"
    assert any(
        "failed to refresh the current suite-generation pointer or compatibility cache" in r.message
        for r in caplog.records
    )


def test_suite_generation_record_is_write_once(tmp_path: Path) -> None:
    """Suite mirror: a published suite generation's record can never be rewritten."""
    suite = _stored_suite_config(tmp_path)
    run_suite(suite)
    generation_id = _last_successful_suite_generation_id(suite)
    assert generation_id is not None
    record_path = _suite_generation_record_path(suite, generation_id)
    first_content = record_path.read_bytes()

    engine_run._write_suite_generation_state(
        suite,
        generation_id=generation_id,
        state="failed",
        started_at="2020-01-01T00:00:00Z",
        ended_at="2020-01-01T00:01:00Z",
        error_class="OSError",
    )

    assert record_path.read_bytes() == first_content
    assert _last_successful_suite_generation_id(suite) == generation_id


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
