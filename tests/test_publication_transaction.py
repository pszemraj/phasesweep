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

import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

import phasesweep.engine.run as engine_run
from phasesweep import load_config, run_experiment
from phasesweep.config import IntParam, Phase, Sampler, Suite
from phasesweep.engine import (
    NoFeasibleTrialError,
    TerminalReport,
    generation_id_source,
    read_status,
    read_winner,
)
from phasesweep.engine.run import run_suite
from phasesweep.engine.state import (
    PublicationPointer,
    _generation_dir,
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _last_successful_generation_path,
    _last_successful_suite_generation_id,
    _resolve_publication_pointer,
    _resolve_suite_publication_pointer,
    _suite_generation_path,
    _suite_generation_record_path,
    _suite_generation_summary_path,
    _suite_summary_path,
)
from phasesweep.runtime import process as runtime_process
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


# Provenance files frozen into every generation namespace at claim time
# (review v0.5.18 / finding F6). Referenced by literal name on purpose: these
# names are the documented on-disk contract in docs/runtime.md, so renaming
# them must fail here rather than silently move an operator's evidence.
_CONFIG_SNAPSHOT_NAME = "config.snapshot.yaml"
_REPRODUCIBILITY_NAME = "reproducibility.json"
_SENTINEL_SECRET = "s3cr3t-sentinel-must-not-be-published"


def _stored_experiment(tmp_path: Path, *, n_trials: int = 1, env: dict[str, str] | None = None):
    trainer = write_trainer(tmp_path / "trainer.py", _TRAINER_BODY)
    return make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        env=env,
        phases=[
            Phase(
                name="p",
                n_trials=n_trials,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
                fixed_overrides={"batch_size": 8},
            )
        ],
    )


@contextlib.contextmanager
def _umask(mask: int) -> Iterator[None]:
    """Pin the process umask for one test so mode assertions are deterministic."""
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _provenance_paths(experiment, generation_id: str) -> tuple[Path, Path]:  # noqa: ANN001
    """Return one generation's ``(config.snapshot.yaml, reproducibility.json)`` paths."""
    generation_dir = _generation_dir(experiment, generation_id)
    return (
        generation_dir / _CONFIG_SNAPSHOT_NAME,
        generation_dir / _REPRODUCIBILITY_NAME,
    )


def _current_pointer_state(experiment) -> str | None:
    """Read the current-pointer file's own ``state`` label."""
    payload = yaml.safe_load(_generation_path(experiment).read_text())
    return payload.get("state") if isinstance(payload, dict) else None


def _record_state(experiment, generation_id: str) -> str | None:
    """Read one generation's immutable record ``state`` label."""
    payload = yaml.safe_load(_generation_record_path(experiment, generation_id).read_text())
    return payload.get("state") if isinstance(payload, dict) else None


def _suite_record_state(suite: Suite, generation_id: str) -> str | None:
    """Read one suite generation's immutable record ``state`` label."""
    payload = yaml.safe_load(_suite_generation_record_path(suite, generation_id).read_text())
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


def test_directory_fsync_failure_after_pointer_rename_still_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory-fsync failure after the pointer rename cannot fail the publication.

    ``os.replace`` is the commit; the directory fsync after it is durability
    bookkeeping. Pre-fix, an fsync error surfaced as an exception from the
    pointer write and the already-committed publication was rewritten as
    ``publication_failed`` (review v0.5.16 / blocker 1, window 1).
    """
    experiment = _stored_experiment(tmp_path)

    real_fsync = os.fsync

    def flaky_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("simulated directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr("phasesweep.runtime.files.os.fsync", flaky_fsync)

    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    assert _record_state(experiment, generation_id) == "published"
    assert _current_pointer_state(experiment) == "published"


def test_control_flow_exception_from_postcommit_record_write_cannot_downgrade_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KeyboardInterrupt escaping a post-commit step must not reclassify the publication.

    ``_log_on_failure`` used to catch only ``Exception``; a control-flow
    exception from the record write propagated into the outer terminal
    handler and rewrote the committed publication as ``failed`` (review
    v0.5.16 / blocker 1, window 2).
    """
    experiment = _stored_experiment(tmp_path)

    def interrupt_record_write(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(engine_run, "_write_generation_record_once", interrupt_record_write)

    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    # The record write itself was interrupted, but the current pointer still
    # reached "published" via step 5 and nothing rewrote the outcome.
    assert _current_pointer_state(experiment) == "published"


def test_terminal_callback_control_flow_exception_cannot_replace_success(
    tmp_path: Path,
) -> None:
    """A diagnostic callback raising KeyboardInterrupt cannot fail a published run."""
    experiment = _stored_experiment(tmp_path)

    def interrupting_callback(_report: TerminalReport) -> None:
        raise KeyboardInterrupt("diagnostic consumer interrupted")

    winners = run_experiment(experiment, terminal_callback=interrupting_callback)

    assert set(winners) == {"p"}
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    assert _record_state(experiment, generation_id) == "published"


def test_terminal_callback_control_flow_exception_cannot_replace_failure(
    tmp_path: Path,
) -> None:
    """A diagnostic callback raising KeyboardInterrupt cannot mask the primary error."""
    trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        max_consecutive_failures=1,
    )

    def interrupting_callback(_report: TerminalReport) -> None:
        raise KeyboardInterrupt("diagnostic consumer interrupted")

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment, terminal_callback=interrupting_callback)


def test_shutdown_signal_during_publication_is_absorbed_until_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown signal racing the publication transaction loses to the commit.

    The signal lands mid-transaction (during pre-commit validation); the
    publication must still commit, the run must return success, and the
    shutdown must be delivered at the next checkpoint — never rewritten into
    a ``failed``/``publication_failed`` state for the committed generation
    (review v0.5.16 / blocker 1, window 2).
    """
    experiment = _stored_experiment(tmp_path)

    original_validate = engine_run._validate_generation_publishable

    def validate_then_signal(*args: object, **kwargs: object) -> None:
        original_validate(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(engine_run, "_validate_generation_publishable", validate_then_signal)

    try:
        winners = run_experiment(experiment)

        assert set(winners) == {"p"}
        generation_id = _last_successful_generation_id(experiment)
        assert generation_id is not None
        assert _record_state(experiment, generation_id) == "published"
        assert _current_pointer_state(experiment) == "published"

        # The absorbed shutdown is still honored before any new work starts.
        with pytest.raises(PhaseSweepShutdown) as exc_info:
            runtime_process.service_pending_shutdown()
        assert exc_info.value.signum == signal.SIGTERM
    finally:
        runtime_process._deferred_shutdown_signum = None


def test_shutdown_absorbed_during_component_publication_stops_suite_before_next_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown that loses to a component's publication still stops the suite.

    The committed component publication wins its own race, but the absorbed
    shutdown must be serviced before the next study starts — the suite must
    not keep launching new work after the operator asked it to stop.
    """
    trainer = write_trainer(tmp_path / "trainer.py", _TRAINER_BODY)
    config = load_config(
        write_yaml(
            tmp_path,
            f"""
            suite: absorb_suite
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
              - name: two
                phases:
                  - name: p
                    n_trials: 1
                    sampler: {{ type: random, seed: 0 }}
                    search_space: {{ x: {{ type: int, low: 0, high: 10 }} }}
            """,
        )
    )
    assert isinstance(config, Suite)

    original_validate = engine_run._validate_generation_publishable
    signalled = {"done": False}

    def validate_then_signal(*args: object, **kwargs: object) -> None:
        original_validate(*args, **kwargs)
        if not signalled["done"]:
            signalled["done"] = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(engine_run, "_validate_generation_publishable", validate_then_signal)

    try:
        with pytest.raises(PhaseSweepShutdown) as exc_info:
            run_suite(config)
        assert exc_info.value.signum == signal.SIGTERM

        # Study one's component publication committed and survived.
        component_one = config.experiment_for_study(config.studies[0])
        assert _last_successful_generation_id(component_one) is not None
        # Study two never started: no generation namespace was ever claimed.
        component_two = config.experiment_for_study(config.studies[1])
        assert _last_successful_generation_id(component_two) is None
        assert not _generation_path(component_two).is_file()
    finally:
        runtime_process._deferred_shutdown_signum = None


# --------------------------------------------------------------------------
# No failure path leaves the current pointer non-terminal.
# --------------------------------------------------------------------------


def test_execution_failure_leaves_current_pointer_terminal(tmp_path: Path) -> None:
    """An ordinary execution failure drives the current pointer to a terminal state."""
    trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        max_consecutive_failures=1,
    )

    with pytest.raises(NoFeasibleTrialError):
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
# The generation manifest (summary schema v2) is validated as a complete
# result graph before the pointer may commit, and again on reads.
# --------------------------------------------------------------------------


def test_publication_refuses_missing_winner_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summary that claims a winner whose artifact is gone must not publish.

    Review v0.5.16 / blocker 3 reproduction: the old validator looped the
    current config and silently ``continue``-d past a missing winner file, so
    deleting the artifact immediately before validation still published an
    officially successful generation with no winner.
    """
    experiment = _stored_experiment(tmp_path)

    original = engine_run._validate_generation_publishable

    def delete_winner_then_validate(exp, generation_id: str) -> None:  # noqa: ANN001
        _generation_winner_path(exp, generation_id, "p").unlink()
        original(exp, generation_id)

    monkeypatch.setattr(engine_run, "_validate_generation_publishable", delete_winner_then_validate)

    with pytest.raises(RuntimeError, match="missing or unreadable"):
        run_experiment(experiment)

    assert _last_successful_generation_id(experiment) is None
    assert _current_pointer_state(experiment) == "publication_failed"


def test_publication_refuses_tampered_winner_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A winner artifact whose content no longer hashes to the manifest must not publish."""
    experiment = _stored_experiment(tmp_path)

    original = engine_run._validate_generation_publishable

    def tamper_winner_then_validate(exp, generation_id: str) -> None:  # noqa: ANN001
        winner_path = _generation_winner_path(exp, generation_id, "p")
        winner = yaml.safe_load(winner_path.read_text())
        winner["metric"]["x"] = -999.0
        winner_path.write_text(yaml.safe_dump(winner, sort_keys=False))
        original(exp, generation_id)

    monkeypatch.setattr(engine_run, "_validate_generation_publishable", tamper_winner_then_validate)

    with pytest.raises(RuntimeError, match="does not match its recorded hash"):
        run_experiment(experiment)

    assert _last_successful_generation_id(experiment) is None
    assert _current_pointer_state(experiment) == "publication_failed"


def test_read_side_rejects_generation_with_altered_winner_artifact(tmp_path: Path) -> None:
    """A warmed reader notices later winner corruption and fails closed."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner = yaml.safe_load(winner_path.read_text())
    winner["metric"]["x"] = -999.0
    winner_path.write_text(yaml.safe_dump(winner, sort_keys=False))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


@pytest.mark.parametrize("identity_field", ["generation_id", "attempt_id"])
def test_manifest_rejects_winner_source_identity_disagreement(
    tmp_path: Path,
    identity_field: str,
) -> None:
    """The hash-covered winner's two provenance copies must identify one attempt."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner = yaml.safe_load(winner_path.read_text())
    winner["winner_source"][identity_field] = f"different-{identity_field}"
    winner_path.write_text(yaml.safe_dump(winner, sort_keys=False))

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    artifact = next(
        item for item in summary["artifacts"] if item["kind"] == "winner" and item["phase"] == "p"
    )
    artifact["sha256"] = hashlib.sha256(winner_path.read_bytes()).hexdigest()
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_read_side_accepts_legacy_summary_without_manifest(tmp_path: Path) -> None:
    """A pre-manifest summary (no schema_version) keeps the identity-only gate."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    for key in ("schema_version", "artifacts", "config_fingerprint", "phase_plan"):
        summary.pop(key, None)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    assert _last_successful_generation_id(experiment) == generation_id
    assert read_winner(experiment, "p") is not None


def test_suite_publication_refuses_broken_component_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suite must not publish when a recorded component's manifest no longer validates."""
    suite = _stored_suite_config(tmp_path)
    component = suite.experiment_for_study(suite.studies[0])

    original = engine_run._validate_suite_generation_publishable

    def break_component_then_validate(suite_arg, generation_id: str) -> None:  # noqa: ANN001
        component_generation = _last_successful_generation_id(component)
        assert component_generation is not None
        _generation_winner_path(component, component_generation, "p").unlink()
        original(suite_arg, generation_id)

    monkeypatch.setattr(
        engine_run, "_validate_suite_generation_publishable", break_component_then_validate
    )

    with pytest.raises(RuntimeError, match="component manifest is invalid"):
        run_suite(suite)

    assert _last_successful_suite_generation_id(suite) is None


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
    # The pointer file itself is still there, so this is corruption of a
    # recorded publication -- never an experiment that has published nothing
    # (review v0.5.18 / finding F4).
    assert _resolve_publication_pointer(experiment).state == "failed"


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
# Publication integrity tri-state (review v0.5.18 / finding F4). "Nothing
# published" and "the recorded publication no longer validates" are different
# facts and must never collapse into the same answer: reporting corruption as
# a fresh tree invites a re-run, which advances the pointer and leaves the
# corruption flagged nowhere.
# --------------------------------------------------------------------------


def test_publication_pointer_reports_ok_for_a_healthy_publication(tmp_path: Path) -> None:
    """A valid publication resolves ``ok`` with its generation id and no error."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    assert _resolve_publication_pointer(experiment) == PublicationPointer(
        state="ok", generation_id=generation_id, error=None
    )

    status = read_status(experiment)
    assert status["publication_integrity"] == "ok"
    assert "publication_error" not in status
    assert status["published_generation_id"] == generation_id


def test_publication_pointer_reports_absent_before_anything_publishes(tmp_path: Path) -> None:
    """A tree that never published is healthy, not corrupt."""
    experiment = _stored_experiment(tmp_path)

    assert _resolve_publication_pointer(experiment) == PublicationPointer(
        state="absent", generation_id=None, error=None
    )

    status = read_status(experiment)
    assert status["publication_integrity"] == "absent"
    assert "publication_error" not in status
    assert status["published_generation_id"] is None
    assert status["is_published"] is False


def test_corrupt_publication_is_reported_as_failed_not_absent(tmp_path: Path) -> None:
    """The F4 reproduction: a tampered winner must not read as "nothing published"."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None
    assert "does not match its recorded hash" in pointer.error

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in status["publication_error"]
    # Nothing is fabricated: the no-publication facts stay exactly as they read
    # today, only the integrity verdict is new.
    assert status["published_generation_id"] is None
    assert status["represented_generation_id"] is None
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False
    # The boolean-blind wrapper keeps its fail-closed contract for path callers.
    assert _last_successful_generation_id(experiment) is None


@pytest.mark.parametrize("filename", [_CONFIG_SNAPSHOT_NAME, _REPRODUCIBILITY_NAME])
def test_tampered_provenance_file_is_reported_as_a_failed_publication(
    tmp_path: Path,
    filename: str,
) -> None:
    """The F6 provenance files feed the same tri-state as any other artifact."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    target = _generation_dir(experiment, generation_id) / filename
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None
    assert "does not match its recorded hash" in pointer.error

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in status["publication_error"]


def test_deleted_pointer_with_an_intact_namespace_reports_absent(tmp_path: Path) -> None:
    """The pointer is the publication authority; an orphaned namespace is not one."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    _last_successful_generation_path(experiment).unlink()
    assert _generation_summary_path(experiment, generation_id).is_file()

    assert _resolve_publication_pointer(experiment) == PublicationPointer(
        state="absent", generation_id=None, error=None
    )
    assert read_status(experiment)["publication_integrity"] == "absent"


def test_pointer_to_a_deleted_generation_namespace_reports_failed(tmp_path: Path) -> None:
    """A pointer whose whole target namespace is gone is corruption, not a fresh tree."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    shutil.rmtree(_generation_dir(experiment, generation_id))

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert status["publication_error"] == pointer.error


def test_resume_path_still_raises_the_manifest_error(tmp_path: Path) -> None:
    """The tri-state must not soften the raise the resume/rebind path depends on."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    with pytest.raises(RuntimeError, match="does not match its recorded hash"):
        _last_successful_generation_id(experiment, raise_on_manifest_error=True)


def test_suite_publication_pointer_reports_a_tampered_component_as_failed(
    tmp_path: Path,
) -> None:
    """The suite pointer gets the same tri-state as the experiment pointer."""
    suite = _stored_suite_config(tmp_path)
    run_suite(suite)
    generation_id = _last_successful_suite_generation_id(suite)
    assert generation_id is not None

    assert _resolve_suite_publication_pointer(suite) == PublicationPointer(
        state="ok", generation_id=generation_id, error=None
    )

    summary = yaml.safe_load(_suite_generation_summary_path(suite, generation_id).read_text())
    component_path = Path(summary["studies"][0]["component_summary_path"])
    component_path.write_text(component_path.read_text() + "\n# tampered\n")

    pointer = _resolve_suite_publication_pointer(suite)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None
    assert _last_successful_suite_generation_id(suite) is None


def test_suite_publication_pointer_reports_absent_before_anything_publishes(
    tmp_path: Path,
) -> None:
    """A suite that never published reports absent, like the experiment pointer."""
    suite = _stored_suite_config(tmp_path)

    assert _resolve_suite_publication_pointer(suite) == PublicationPointer(
        state="absent", generation_id=None, error=None
    )


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

    def swapping_resolver(_experiment: object) -> PublicationPointer:
        calls.append(1)
        return PublicationPointer(
            state="ok",
            generation_id=real_generation if len(calls) == 1 else "generation-does-not-exist",
            error=None,
        )

    monkeypatch.setattr("phasesweep.engine.read._resolve_publication_pointer", swapping_resolver)

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

    second_generation = yaml.safe_load(_suite_generation_path(suite).read_text())[
        "suite_generation_id"
    ]
    assert second_generation != first_generation
    assert _suite_record_state(suite, second_generation) == "publication_failed"
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
    assert _suite_record_state(suite, second_generation) == "published"
    assert any(
        "failed to refresh the current suite-generation pointer or compatibility cache" in r.message
        for r in caplog.records
    )


def test_suite_summary_winner_facts_are_anchored_to_component_artifacts(
    tmp_path: Path,
) -> None:
    """An edited suite summary or component summary fails read-side integrity
    validation instead of presenting altered results as published (review
    v0.5.17 gap hunt): the suite summary's winner facts are anchored to the
    hash-covered component summaries it recorded at publication. The
    experiment path has enforced this since the v0.5.16 manifest work; the
    suite path previously trusted the summary text on identity alone."""
    suite = _stored_suite_config(tmp_path)
    run_suite(suite)
    generation_id = _last_successful_suite_generation_id(suite)
    assert generation_id is not None

    summary_path = _suite_generation_summary_path(suite, generation_id)
    original = summary_path.read_text()
    summary = yaml.safe_load(original)
    study = summary["studies"][0]
    exposed = [item for item in study["phases"] if item.get("exposed")]
    assert exposed, "test setup: suite must expose at least one winner"

    # Spoof the published metric value in the suite summary itself.
    exposed[0]["metric"] = exposed[0]["metric"] + 1.0
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    assert _last_successful_suite_generation_id(suite) is None

    # Parameters and effective overrides are published winner facts too.
    summary = yaml.safe_load(original)
    study = summary["studies"][0]
    exposed = [item for item in study["phases"] if item.get("exposed")]
    exposed[0]["params"]["x"] = exposed[0]["params"]["x"] + 1
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    assert _last_successful_suite_generation_id(suite) is None

    # Restore, then tamper the hash-anchored component summary instead.
    summary_path.write_text(original)
    assert _last_successful_suite_generation_id(suite) == generation_id
    component_path = Path(study["component_summary_path"])
    component_original = component_path.read_text()
    component_path.write_text(component_original + "\n# tampered\n")
    assert _last_successful_suite_generation_id(suite) is None

    # Restoring both artifacts restores the published result.
    component_path.write_text(component_original)
    assert _last_successful_suite_generation_id(suite) == generation_id


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

    summary = yaml.safe_load(_suite_summary_path(config).read_text())
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


# --------------------------------------------------------------------------
# Generation provenance: the configuration and identity that produced a
# published result are frozen into its namespace (review v0.5.18 / finding F6).
# --------------------------------------------------------------------------


def test_generation_namespace_freezes_the_config_that_produced_it(tmp_path: Path) -> None:
    """A published generation keeps the canonical config that produced it, owner-only.

    Before this, a generation kept only the config *fingerprint*: once the
    operator edited or lost the YAML, the digest could prove a mismatch but
    could not reconstruct the search spaces, fixed overrides, contracts, env,
    or trial command behind a published winner.
    """
    experiment = _stored_experiment(tmp_path, env={"TRAINER_TOKEN": _SENTINEL_SECRET})
    with _umask(0o022):
        run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot_path, _ = _provenance_paths(experiment, generation_id)

    # The snapshot may hold secrets (``env:``), so it is owner-only even though
    # its directory is deliberately operator-readable.
    assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o600

    snapshot = yaml.safe_load(snapshot_path.read_text())
    assert snapshot == experiment.model_dump(mode="json")

    # Spot-check the values an operator actually needs to reproduce the run.
    assert snapshot["trial_command"] == experiment.trial_command
    assert snapshot["env"] == {"TRAINER_TOKEN": _SENTINEL_SECRET}
    phase = snapshot["phases"][0]
    assert phase["fixed_overrides"] == {"batch_size": 8}
    assert phase["search_space"]["x"]["type"] == "int"
    assert phase["search_space"]["x"]["low"] == 0
    assert phase["search_space"]["x"]["high"] == 10


def test_generation_reproducibility_record_is_shareable_digests_only(tmp_path: Path) -> None:
    """The readable provenance record carries identity and digests, never config values."""
    experiment = _stored_experiment(tmp_path, env={"TRAINER_TOKEN": _SENTINEL_SECRET})
    with _umask(0o022):
        run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot_path, repro_path = _provenance_paths(experiment, generation_id)

    # Ordinary umask-governed artifact: safe to read and share, unlike the snapshot.
    assert stat.S_IMODE(repro_path.stat().st_mode) == 0o644

    raw = repro_path.read_text()
    record = json.loads(raw)
    summary = yaml.safe_load(_generation_summary_path(experiment, generation_id).read_text())

    assert record["experiment"] == experiment.experiment
    assert record["generation_id"] == generation_id
    # A direct-caller run mints its own id; only a launcher-granted identity
    # records "caller" (PR #5 review / P2 missing-handle authority).
    assert record["generation_id_source"] == "engine"
    assert generation_id_source(experiment, generation_id) == "engine"
    assert record["phasesweep_version"] == summary["phasesweep_version"]
    assert record["config_fingerprint"] == summary["config_fingerprint"]
    assert record["provenance"] == experiment.provenance
    assert record["schema_versions"]["generation_summary"] == summary["schema_version"]
    assert [item["name"] for item in record["phase_config_fingerprints"]] == [
        phase.name for phase in experiment.phases
    ]
    assert all(len(item["sha256"]) == 64 for item in record["phase_config_fingerprints"]), record[
        "phase_config_fingerprints"
    ]
    assert record["config_snapshot"] == {
        "path": _CONFIG_SNAPSHOT_NAME,
        "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }

    # Digests, never values: nothing from the snapshot's contents leaks here,
    # least of all a configured env secret.
    assert _SENTINEL_SECRET not in raw
    assert "TRAINER_TOKEN" not in raw
    assert experiment.trial_command not in raw


def test_caller_granted_generation_id_is_recorded_durably(tmp_path: Path) -> None:
    """A launcher-granted identity survives in the artifact tree itself.

    The launcher (the MCP server) freezes the run's authority in its own state
    dir; this record is what lets readers detect that such frozen authority
    exists even after that state dir is deleted or replaced (PR #5 review /
    P2 missing-handle authority). The reader answers ``None`` -- never a
    guess -- for ids without a valid record, so legacy trees keep their
    current-policy behavior.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment, generation_id="launcher-granted-1")

    record = json.loads(
        (_generation_dir(experiment, "launcher-granted-1") / _REPRODUCIBILITY_NAME).read_text()
    )
    assert record["generation_id_source"] == "caller"
    assert generation_id_source(experiment, "launcher-granted-1") == "caller"

    # Unanswerable cases collapse to None: no such generation, and a record
    # predating the field (schema version 1).
    assert generation_id_source(experiment, "never-claimed") is None
    del record["generation_id_source"]
    record_path = _generation_dir(experiment, "launcher-granted-1") / _REPRODUCIBILITY_NAME
    record_path.write_text(json.dumps(record))
    assert generation_id_source(experiment, "launcher-granted-1") is None


def test_generation_manifest_covers_the_provenance_files(tmp_path: Path) -> None:
    """Both provenance files are manifest-listed with their content hashes.

    Without this the publication validator would reject every new generation:
    the namespace may hold nothing the manifest does not list.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot_path, repro_path = _provenance_paths(experiment, generation_id)

    summary = yaml.safe_load(_generation_summary_path(experiment, generation_id).read_text())
    entries = {item["kind"]: item for item in summary["artifacts"] if "path" in item}
    assert entries["config_snapshot"] == {
        "kind": "config_snapshot",
        "path": _CONFIG_SNAPSHOT_NAME,
        "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    assert entries["reproducibility"] == {
        "kind": "reproducibility",
        "path": _REPRODUCIBILITY_NAME,
        "sha256": hashlib.sha256(repro_path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("filename", [_CONFIG_SNAPSHOT_NAME, _REPRODUCIBILITY_NAME])
def test_read_side_rejects_generation_with_tampered_provenance_file(
    tmp_path: Path,
    filename: str,
) -> None:
    """Editing or deleting either provenance file invalidates the publication.

    Same failure shape as tampering a winner: the manifest hash no longer
    matches, so the pointer target is not trusted.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    target = _generation_dir(experiment, generation_id) / filename
    original = target.read_bytes()

    target.write_bytes(original + b"\n# tampered\n")
    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None

    target.write_bytes(original)
    assert _last_successful_generation_id(experiment) == generation_id
    assert read_winner(experiment, "p") is not None

    target.unlink()
    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root reads any mode, so no PermissionError can be provoked",
)
@pytest.mark.parametrize("filename", [_CONFIG_SNAPSHOT_NAME, _REPRODUCIBILITY_NAME])
def test_unreadable_provenance_file_reports_permission_denied_not_corruption(
    tmp_path: Path,
    filename: str,
) -> None:
    """A provenance file this user may not read fails closed with its own reason.

    Re-review v0.5.19 / observation N1: ``config.snapshot.yaml`` is owner-only,
    so a second operator reading a perfectly healthy tree got the generic
    "missing or unreadable" verdict and the "inspect or restore the generation
    namespace" remedy -- for a permission bit. The verdict must stay ``failed``
    (an unvalidatable tree is not a published one), but the reason must name
    the permission denial and point at the publishing user.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    target = _generation_dir(experiment, generation_id) / filename
    original_mode = stat.S_IMODE(target.stat().st_mode)
    target.chmod(0o000)
    try:
        status = read_status(experiment)
        assert status["publication_integrity"] == "failed"
        assert "permission denied" in status["publication_error"]
        assert "only the publishing user" in status["publication_error"]
        assert "missing or unreadable" not in status["publication_error"]
        assert status["published_generation_id"] is None
    finally:
        target.chmod(original_mode)

    # Restoring the mode restores the publication: nothing was ever corrupt.
    assert _last_successful_generation_id(experiment) == generation_id


def test_failed_generation_still_retains_its_provenance_files(tmp_path: Path) -> None:
    """A generation that fails mid-run still records what configuration ran.

    The files are written at claim time precisely so a post-mortem of a failed
    generation can name its search spaces and trial command.
    """
    trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        max_consecutive_failures=1,
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    failed_generation = yaml.safe_load(_generation_path(experiment).read_text())["generation_id"]
    snapshot_path, repro_path = _provenance_paths(experiment, failed_generation)
    assert snapshot_path.is_file()
    assert repro_path.is_file()
    snapshot = yaml.safe_load(snapshot_path.read_text())
    assert snapshot == experiment.model_dump(mode="json")
    assert json.loads(repro_path.read_text())["generation_id"] == failed_generation


def test_generation_published_before_provenance_files_existed_stays_valid(
    tmp_path: Path,
) -> None:
    """A pre-F6 namespace -- no provenance files, no manifest entries -- still reads valid.

    Replayed the way the other legacy-compat cases in this module are: publish
    normally, then rewrite the namespace into the older shape. Dropping only
    the manifest entries must fail (the namespace would hold unlisted
    artifacts); dropping the files too is exactly the historical layout and
    must publish-read cleanly.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot_path, repro_path = _provenance_paths(experiment, generation_id)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["artifacts"] = [
        item for item in summary["artifacts"] if item["kind"] in ("winner", "promotion")
    ]
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    # Files present but unlisted: the manifest no longer covers the namespace.
    assert _last_successful_generation_id(experiment) is None

    snapshot_path.unlink()
    repro_path.unlink()

    # Neither listed nor present -- the pre-F6 layout, still fully valid.
    assert _last_successful_generation_id(experiment) == generation_id
    assert read_winner(experiment, "p") is not None


def test_partially_dropped_provenance_record_is_not_a_legacy_namespace(tmp_path: Path) -> None:
    """Half a provenance record is an edit, not a historical layout.

    The two files are written together at claim time, so a namespace that
    keeps one and drops the other must fail rather than fall through the
    pre-F6 compatibility path.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    _, repro_path = _provenance_paths(experiment, generation_id)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["artifacts"] = [
        item for item in summary["artifacts"] if item.get("kind") != "reproducibility"
    ]
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    repro_path.unlink()

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {**payload, "generation_id": "not-this-generation"},
            id="reproducibility-names-another-generation",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {
                **payload,
                "config_snapshot": {**payload["config_snapshot"], "sha256": "0" * 64},
            },
            id="reproducibility-unanchors-the-snapshot",
        ),
        pytest.param(
            _CONFIG_SNAPSHOT_NAME,
            lambda payload: {**payload, "experiment": "some-other-experiment"},
            id="snapshot-names-another-experiment",
        ),
    ],
)
def test_rehashed_provenance_edit_still_fails_the_manifest_cross_checks(
    tmp_path: Path,
    filename: str,
    mutate,  # noqa: ANN001
) -> None:
    """Re-hashing an edited provenance file into the manifest does not launder it.

    Same escalation the winner artifacts get: an operator who updates the
    recorded hash after editing still trips the summary cross-checks, so the
    published record cannot be made to describe a different config or
    generation.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    target = _generation_dir(experiment, generation_id) / filename
    if filename == _REPRODUCIBILITY_NAME:
        target.write_text(json.dumps(mutate(json.loads(target.read_text()))))
    else:
        target.write_text(yaml.safe_dump(mutate(yaml.safe_load(target.read_text()))))

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    entry = next(item for item in summary["artifacts"] if item.get("path") == filename)
    entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


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
