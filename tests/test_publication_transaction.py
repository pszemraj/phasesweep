"""Publication-transaction fault injection (review v0.5.15 / blocker 3).

The last-success pointer is the single final authoritative commit: every
injected failure must leave either the previous publication authoritative or
a fully valid published generation -- never a pointer to a failed, missing,
or mismatched generation, a current pointer stuck at a non-terminal state, or
a committed success downgraded by later bookkeeping. The per-generation
lifecycle record is write-once and purely informational (written *after* the
pointer commits); pointer validation instead reads back the generation's own
immutable summary artifact, and terminal-state persistence failures must
never replace the primary exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.generation as generation_ops
import phasesweep.engine.publication_validation as validation_ops
from phasesweep import run_experiment
from phasesweep.config import Experiment, IntParam, Phase, Sampler
from phasesweep.engine import (
    NoFeasibleTrialError,
    PublicationCommitError,
    PublicationIntegrityError,
    TerminalReport,
    Winner,
    generation_id_source,
    read_status,
    read_winner,
)
from phasesweep.engine.paths import (
    _generation_dir,
    _generation_path,
    _generation_record_path,
    _generation_summary_path,
    _generation_winner_path,
    _generations_dir,
    _last_successful_generation_path,
)
from phasesweep.engine.publication import (
    PublicationPointer,
    _last_successful_generation_id,
    _resolve_publication_pointer,
    _unresolvable_pointer,
)
from phasesweep.runtime import shutdown as runtime_shutdown
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import (
    make_experiment,
    patch_directory_fsync_failure,
    patch_path_method_failure,
    requires_nonroot,
    temporary_umask,
    write_param_echo_trainer,
    write_trainer,
)
from tests.ledger_fixtures import materialize, reanchor_summary_pointer


def _fail_terminal_generation_state(original: Callable, error: BaseException):
    def flaky_state(owner, **kwargs: object):
        if kwargs.get("state") == "failed":
            raise error
        return original(owner, **kwargs)

    return flaky_state


# Provenance files frozen into every generation namespace at claim time
# (review v0.5.18 / finding F6). Referenced by literal name on purpose: these
# names are the documented on-disk interface in docs/runtime.md, so renaming
# them must fail here rather than silently move an operator's evidence.
_CONFIG_SNAPSHOT_NAME = "config.snapshot.yaml"
_REPRODUCIBILITY_NAME = "reproducibility.json"
_SENTINEL_SECRET = "s3cr3t-sentinel-must-not-be-published"


def _tamper_winner_artifact(path: Path) -> None:
    """Change a winner payload without updating its recorded manifest hash."""
    winner = yaml.safe_load(path.read_text())
    winner["metric"]["x"] = -999.0
    path.write_text(yaml.safe_dump(winner, sort_keys=False))


def _stored_experiment(tmp_path: Path, *, env: dict[str, str] | None = None):
    trainer = write_param_echo_trainer(tmp_path)
    return make_experiment(
        persistent=tmp_path, trainer=trainer, env=env, n_trials=1, fixed_overrides={"batch_size": 8}
    )


def _golden_publication(tmp_path: Path) -> tuple[Experiment, str]:
    """Copy the golden one-generation publication for a test that needs a tree, not a sweep.

    :param Path tmp_path: Per-test temporary directory.
    :return tuple[Experiment, str]: The experiment reading the copy and its
        published generation id.
    """
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    return experiment, generation_id


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


# --------------------------------------------------------------------------
# Pre-commit validation failure (step 2): prior publication stays authoritative.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary_text", "match"),
    [
        pytest.param(None, "could not be read back", id="missing"),
        pytest.param(
            "experiment: foreign\ngeneration_id: generation-test\n",
            "failed publication validation",
            id="wrong-owner",
        ),
    ],
)
def test_summary_readback_refusals_are_publication_commit_errors(
    tmp_path: Path,
    summary_text: str | None,
    match: str,
) -> None:
    """A new generation's bad summary is an expected failed commit, not a bug."""
    summary_path = tmp_path / "summary.yaml"
    if summary_text is not None:
        summary_path.write_text(summary_text)

    with pytest.raises(PublicationCommitError, match=match):
        generation_ops._validate_publishable_summary(
            summary_path=summary_path,
            owner_key="experiment",
            owner_value="expected",
            id_key="generation_id",
            id_value="generation-test",
            label="Generation",
        )


@pytest.mark.integration
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

    monkeypatch.setattr(generation_ops, "_validate_generation_publishable", fail_validation)

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


@pytest.mark.integration
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
    original_write = artifact_io._write_yaml_atomic

    def flaky_write(path: Path, payload: object) -> None:
        if path == pointer_path:
            raise OSError("simulated pointer commit failure")
        return original_write(path, payload)

    monkeypatch.setattr(artifact_io, "_write_yaml_atomic", flaky_write)

    with pytest.raises(OSError, match="simulated pointer commit failure"):
        run_experiment(experiment)

    second_generation = yaml.safe_load(_generation_path(experiment).read_text())["generation_id"]
    assert second_generation != first_generation
    assert _record_state(experiment, second_generation) == "publication_failed"
    assert _current_pointer_state(experiment) == "publication_failed"
    assert _last_successful_generation_id(experiment) == first_generation


@pytest.mark.integration
def test_required_publication_sidecar_failure_prevents_pointer_commit(tmp_path: Path) -> None:
    """A detached-run snapshot that cannot become durable must block publication."""
    experiment = _stored_experiment(tmp_path)
    prepared_generation: str | None = None

    class FailingHook:
        def prepare(
            self,
            *,
            experiment: Experiment,
            generation_id: str,
            winners: Mapping[str, Winner],
            summary: Mapping[str, object],
        ) -> None:
            nonlocal prepared_generation
            prepared_generation = generation_id
            assert set(winners) == {"p"}
            assert summary["generation_id"] == generation_id
            assert _last_successful_generation_id(experiment) is None
            raise OSError("simulated detached snapshot persistence failure")

        def committed(self, *, generation_id: str) -> None:
            raise AssertionError(f"uncommitted generation was notified: {generation_id}")

    with pytest.raises(OSError, match="detached snapshot persistence failure"):
        run_experiment(experiment, publication_hook=FailingHook())

    assert prepared_generation is not None
    assert _last_successful_generation_id(experiment) is None
    assert _record_state(experiment, prepared_generation) == "publication_failed"


@pytest.mark.integration
def test_publication_sidecar_is_notified_after_pointer_commit(tmp_path: Path) -> None:
    """Postcommit notification observes the pointer and cannot downgrade success."""
    experiment = _stored_experiment(tmp_path)
    events: list[str] = []

    class ObservingHook:
        def prepare(
            self,
            *,
            experiment: Experiment,
            generation_id: str,
            winners: Mapping[str, Winner],
            summary: Mapping[str, object],
        ) -> None:
            assert set(winners) == {"p"}
            assert summary["generation_id"] == generation_id
            assert _last_successful_generation_id(experiment) is None
            events.append(f"prepared:{generation_id}")

        def committed(self, *, generation_id: str) -> None:
            assert _last_successful_generation_id(experiment) == generation_id
            events.append(f"committed:{generation_id}")
            raise KeyboardInterrupt("simulated postcommit interruption")

    winners = run_experiment(experiment, publication_hook=ObservingHook())

    assert set(winners) == {"p"}
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    assert events == [f"prepared:{generation_id}", f"committed:{generation_id}"]
    assert _record_state(experiment, generation_id) == "published"


# --------------------------------------------------------------------------
# Post-commit failures (steps 4 and 5): the run must still succeed.
# --------------------------------------------------------------------------


@pytest.mark.integration
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

    monkeypatch.setattr(generation_ops, "_write_generation_record_once", fail_record_write)

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.generation"):
        winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    second_generation = _last_successful_generation_id(experiment)
    assert second_generation is not None
    assert second_generation != first_generation
    assert any(
        r.name == "phasesweep.engine.generation"
        and "failed to write the immutable generation record" in r.message
        for r in caplog.records
    )
    # The record itself never got created; publication still succeeded.
    assert not _generation_record_path(experiment, second_generation).is_file()


@pytest.mark.integration
def test_cache_projection_failure_after_commit_leaves_run_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A convenience root-projection failure must not fail the run.

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

    monkeypatch.setattr(generation_ops, "_copy_yaml_projection", fail_projection)

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.generation"):
        winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    second_generation = _last_successful_generation_id(experiment)
    assert second_generation is not None
    assert second_generation != first_generation
    assert _record_state(experiment, second_generation) == "published"
    assert any(
        "failed to refresh the current-generation pointer or convenience projections" in r.message
        for r in caplog.records
    )
    # Reads are unaffected: once a generation has published, they resolve the
    # generation-scoped artifact directly rather than a stale root projection.
    # The winning trial itself still belongs to the first generation (the
    # target trial count was already satisfied, so no new trial ran); what
    # matters is that the read resolves via the new last-success pointer
    # rather than a stale root-level copy.
    published_winner_path = _generation_winner_path(experiment, second_generation, "p")
    assert published_winner_path.is_file()
    published = read_winner(experiment, "p")
    assert published is not None
    assert published.generation_id == first_generation


@pytest.mark.integration
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

    patch_directory_fsync_failure(monkeypatch, "simulated directory fsync failure")

    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    assert _record_state(experiment, generation_id) == "published"
    assert _current_pointer_state(experiment) == "published"


@pytest.mark.integration
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

    monkeypatch.setattr(generation_ops, "_write_generation_record_once", interrupt_record_write)

    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    # The record write itself was interrupted, but the current pointer still
    # reached "published" via step 5 and nothing rewrote the outcome.
    assert _current_pointer_state(experiment) == "published"


@pytest.mark.integration
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
        workdir=tmp_path / "runs", trainer=trainer, n_trials=1, max_consecutive_failures=1
    )

    def interrupting_callback(_report: TerminalReport) -> None:
        raise KeyboardInterrupt("diagnostic consumer interrupted")

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment, terminal_callback=interrupting_callback)


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_shutdown_signal_during_publication_is_absorbed_until_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown signal racing the publication transaction loses to the commit.

    The signal lands mid-transaction (during pre-commit validation); the
    publication must still commit and the shutdown must be delivered at the
    next checkpoint — never rewritten into
    a ``failed``/``publication_failed`` state for the committed generation
    (review v0.5.16 / blocker 1, window 2).
    """
    experiment = _stored_experiment(tmp_path)

    original_validate = generation_ops._validate_generation_publishable

    def validate_then_signal(*args: object, **kwargs: object) -> tuple[dict[str, Any], bytes]:
        validated = original_validate(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return validated

    monkeypatch.setattr(generation_ops, "_validate_generation_publishable", validate_then_signal)

    try:
        with pytest.raises(PhaseSweepShutdown) as exc_info:
            run_experiment(experiment)

        assert exc_info.value.signum == signal.SIGTERM
        assert exc_info.value.published_result_committed is True
        generation_id = _last_successful_generation_id(experiment)
        assert generation_id is not None
        assert _record_state(experiment, generation_id) == "published"
        assert _current_pointer_state(experiment) == "published"
        assert runtime_shutdown.service_pending_shutdown() is None
    finally:
        runtime_shutdown._deferred_shutdown_signum = None


# --------------------------------------------------------------------------
# No failure path leaves the current pointer non-terminal.
# --------------------------------------------------------------------------


def test_execution_failure_leaves_current_pointer_terminal(tmp_path: Path) -> None:
    """An ordinary execution failure drives the current pointer to a terminal state."""
    trainer = write_trainer(tmp_path / "failing.py", "raise SystemExit(1)")
    experiment = make_experiment(
        workdir=tmp_path / "runs", trainer=trainer, n_trials=1, max_consecutive_failures=1
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    assert _current_pointer_state(experiment) in generation_ops._TERMINAL_GENERATION_STATES


# --------------------------------------------------------------------------
# The immutable record is write-once.
# --------------------------------------------------------------------------


def test_generation_record_is_write_once(tmp_path: Path) -> None:
    """A published generation's record can never be rewritten, even to the same state."""
    experiment, generation_id = _golden_publication(tmp_path)
    first_content = _generation_record_path(experiment, generation_id).read_bytes()

    generation_ops._write_generation_state(
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
    generation_ops._write_generation_state(
        experiment,
        generation_id=generation_id,
        state="published",
        from_phase=None,
        publish_current=False,
    )
    assert _generation_record_path(experiment, generation_id).read_bytes() == first_content


@pytest.mark.integration
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
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.generation"):
        run_experiment(experiment)
        run_experiment(experiment)  # republish onto the same storage
    assert not [r for r in caplog.records if "Refusing to rewrite" in r.message]


# --------------------------------------------------------------------------
# The generation manifest (summary schema v2) is validated as a complete
# result graph before the pointer may commit, and again on reads.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate_winner", "error_match"),
    [
        pytest.param(Path.unlink, "missing or unreadable", id="missing"),
        pytest.param(_tamper_winner_artifact, "does not match its recorded hash", id="tampered"),
    ],
)
@pytest.mark.integration
def test_publication_refuses_invalid_winner_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_winner: Callable[[Path], None],
    error_match: str,
) -> None:
    """A missing or modified winner artifact must not publish.

    Review v0.5.16 / blocker 3 reproduction: the old validator looped the
    current config and silently ``continue``-d past an invalid winner file, so
    deleting or modifying the artifact immediately before validation still
    published an officially successful generation with no verified winner.
    """
    experiment = _stored_experiment(tmp_path)

    original = generation_ops._validate_generation_publishable

    def delete_winner_then_validate(exp, generation_id: str) -> None:  # noqa: ANN001
        mutate_winner(_generation_winner_path(exp, generation_id, "p"))
        original(exp, generation_id)

    monkeypatch.setattr(
        generation_ops, "_validate_generation_publishable", delete_winner_then_validate
    )

    with pytest.raises(RuntimeError, match=error_match):
        run_experiment(experiment)

    assert _last_successful_generation_id(experiment) is None
    assert _current_pointer_state(experiment) == "publication_failed"


@pytest.mark.parametrize("identity_field", ["generation_id", "attempt_id"])
def test_manifest_rejects_winner_source_identity_disagreement(
    tmp_path: Path,
    identity_field: str,
) -> None:
    """The hash-covered winner's two provenance copies must identify one attempt."""
    experiment, generation_id = _golden_publication(tmp_path)

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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "unknown_source"),
        ("phase", "../other"),
        ("phase", "other"),
        ("trial_number", 999),
        ("study", 999),
        ("phase_fingerprint", 999),
    ],
)
def test_manifest_rejects_malformed_reanchored_winner_provenance(
    tmp_path: Path, field: str, value: object
) -> None:
    experiment, generation_id = _golden_publication(tmp_path)
    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner = yaml.safe_load(winner_path.read_text())
    if field == "phase_fingerprint":
        winner[field] = value
    else:
        winner["winner_source"][field] = value
    winner_path.write_text(yaml.safe_dump(winner, sort_keys=False))
    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    artifact = next(
        item for item in summary["artifacts"] if item["kind"] == "winner" and item["phase"] == "p"
    )
    artifact["sha256"] = hashlib.sha256(winner_path.read_bytes()).hexdigest()
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)

    assert _resolve_publication_pointer(experiment).state == "failed"
    assert read_status(experiment)["publication_integrity"] == "failed"
    assert read_winner(experiment, "p") is None


@pytest.mark.parametrize("removed_artifact", ["promotion_decisions", "promotion.yaml"])
def test_manifest_rejects_removed_promotion_artifacts(
    tmp_path: Path,
    removed_artifact: str,
) -> None:
    """Current-format publications cannot silently adopt removed promotion state."""
    experiment, generation_id = _golden_publication(tmp_path)

    if removed_artifact == "promotion_decisions":
        summary_path = _generation_summary_path(experiment, generation_id)
        summary = yaml.safe_load(summary_path.read_text())
        summary[removed_artifact] = []
        summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
        reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)
    else:
        promotion_path = (
            _generation_dir(experiment, generation_id) / "phases" / "p" / removed_artifact
        )
        promotion_path.write_text("removed: true\n")

    assert _resolve_publication_pointer(experiment).state == "failed"
    assert read_status(experiment)["publication_integrity"] == "failed"
    assert read_winner(experiment, "p") is None


def test_load_winner_rejects_linked_winner_with_current_summary(tmp_path: Path) -> None:
    """Strict resume rejects a linked winner from a current-format publication."""
    experiment, generation_id = _golden_publication(tmp_path)

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    preserved_winner_path = winner_path.with_name("winner.original.yaml")
    winner_path.rename(preserved_winner_path)
    winner_path.symlink_to(preserved_winner_path.name)

    with pytest.raises(PublicationIntegrityError, match="missing or unreadable"):
        artifact_io._load_winner(experiment, experiment.phases[0], {})


# --------------------------------------------------------------------------
# Pointer validation moved from record state to artifact identity.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param("experiment: t\ngeneration_id: ../evil\n", id="traversal-id"),
        pytest.param("experiment: other\ngeneration_id: {gid}\n", id="wrong-experiment"),
    ],
)
def test_pointer_validation_fails_closed(tmp_path: Path, tamper: str) -> None:
    """An invalid pointer is treated as nothing-published, not trusted for reads."""
    experiment, generation_id = _golden_publication(tmp_path)

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
    experiment, generation_id = _golden_publication(tmp_path)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["generation_id"] = "not-this-generation"
    summary_path.write_text(yaml.safe_dump(summary))

    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_tampering_the_record_state_does_not_affect_publication_status(tmp_path: Path) -> None:
    """The record is informational only; publication status ignores its state entirely."""
    experiment, generation_id = _golden_publication(tmp_path)

    record_path = _generation_record_path(experiment, generation_id)
    record = yaml.safe_load(record_path.read_text())
    record["state"] = "failed"
    record_path.write_text(yaml.safe_dump(record))

    # Unlike the pre-v0.5.15 design, this has no effect: the record is not consulted.
    assert _last_successful_generation_id(experiment) == generation_id
    assert read_winner(experiment, "p") is not None


# --------------------------------------------------------------------------
# Publication integrity states (review v0.5.18 / finding F4). "Nothing
# published" and "the recorded publication no longer validates" are different
# facts and must never collapse into the same answer: reporting corruption as
# a fresh tree invites a re-run, which advances the pointer and leaves the
# corruption flagged nowhere.
# --------------------------------------------------------------------------


def test_publication_pointer_reports_ok_for_a_healthy_publication(tmp_path: Path) -> None:
    """A valid publication resolves ``ok`` with its generation id and no error."""
    experiment, generation_id = _golden_publication(tmp_path)

    assert _resolve_publication_pointer(experiment) == PublicationPointer(
        state="ok", generation_id=generation_id, error=None
    )

    status = read_status(experiment)
    assert status["publication_integrity"] == "ok"
    assert "publication_error" not in status
    assert status["published_generation_id"] == generation_id


def test_experiment_pointer_anchors_the_exact_summary_bytes(tmp_path: Path) -> None:
    """The commit record stores the byte length and SHA-256 of the validated summary."""
    experiment, generation_id = _golden_publication(tmp_path)

    summary = _generation_summary_path(experiment, generation_id).read_bytes()
    pointer = yaml.safe_load(_last_successful_generation_path(experiment).read_text())
    assert pointer["summary_size_bytes"] == len(summary)
    assert pointer["summary_sha256"] == hashlib.sha256(summary).hexdigest()


def test_summary_digest_is_checked_before_the_summary_is_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer-backed summary with wrong bytes is rejected before YAML sees them."""
    experiment, generation_id = _golden_publication(tmp_path)

    summary_path = _generation_summary_path(experiment, generation_id)
    original = summary_path.read_bytes()
    tampered = original.replace(b"experiment:", b"experimenT:", 1)
    assert len(tampered) == len(original)
    summary_path.write_bytes(tampered)

    parsed_summary_bytes = False
    original_safe_load = validation_ops.yaml.safe_load

    def track_safe_load(stream):  # noqa: ANN001, ANN202
        nonlocal parsed_summary_bytes
        if isinstance(stream, bytes):
            parsed_summary_bytes = True
        return original_safe_load(stream)

    monkeypatch.setattr(validation_ops.yaml, "safe_load", track_safe_load)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "digest" in pointer.error
    assert parsed_summary_bytes is False


@pytest.mark.parametrize(
    "tamper",
    ["phase_plan", "metric_goal", "objective_evidence", "config_fingerprint", "completion"],
)
def test_summary_semantics_are_pointer_anchored_and_cross_checked(
    tmp_path: Path,
    tamper: str,
) -> None:
    """Historical plan, metric/evidence, and config identity cannot self-authenticate."""
    experiment, generation_id = _golden_publication(tmp_path)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    if tamper == "phase_plan":
        summary["phase_plan"][0]["name"] = "q"
        semantic_error = "phase plan"
    elif tamper == "metric_goal":
        summary["metric"]["goal"] = "maximize"
        semantic_error = "metric semantics"
    elif tamper == "objective_evidence":
        evidence = summary["metric"]["objective_evidence"]
        flag = next(key for key, value in evidence.items() if type(value) is bool)
        evidence[flag] = not evidence[flag]
        semantic_error = "metric semantics"
    elif tamper == "config_fingerprint":
        summary["config_fingerprint"] = "0" * 64
        semantic_error = "config fingerprint"
    else:
        summary["phases"][0]["completion"]["finished_trials"] = 0
        summary["phases"][0]["completion"]["completed_trials"] = 0
        semantic_error = "completion"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "summary" in pointer.error.lower()

    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)
    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert semantic_error in pointer.error


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


def test_unreadable_pointer_is_permission_denied_not_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = tmp_path / "last_successful_generation.yaml"
    patch_path_method_failure(
        monkeypatch,
        pointer,
        "lstat",
        PermissionError("permission denied"),
    )

    verdict = _unresolvable_pointer(pointer, "experiment 'x'")

    assert verdict.state == "permission_denied"
    assert verdict.error is not None
    assert "permission denied" in verdict.error


def test_corrupt_publication_is_reported_as_failed_not_absent(tmp_path: Path) -> None:
    """The F4 reproduction: a tampered winner must not read as "nothing published"."""
    experiment, generation_id = _golden_publication(tmp_path)

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
    # The boolean-blind wrapper remains fail-closed for path callers.
    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None

    pointer_path = _last_successful_generation_path(experiment)
    pointer_bytes = pointer_path.read_bytes()
    generations_before = set(_generation_dir(experiment, generation_id).parent.iterdir())
    with pytest.raises(PublicationIntegrityError, match="does not match its recorded hash"):
        run_experiment(experiment)
    assert pointer_path.read_bytes() == pointer_bytes
    assert set(_generation_dir(experiment, generation_id).parent.iterdir()) == generations_before


@pytest.mark.parametrize("delete_current_pointer", [False, True])
def test_deleted_pointer_with_an_intact_namespace_reports_absent(
    tmp_path: Path, delete_current_pointer: bool
) -> None:
    """The pointer is the publication authority; an orphaned namespace is not one."""
    experiment, generation_id = _golden_publication(tmp_path)

    _last_successful_generation_path(experiment).unlink()
    if delete_current_pointer:
        _generation_path(experiment).unlink()
    assert _generation_summary_path(experiment, generation_id).is_file()

    assert _resolve_publication_pointer(experiment) == PublicationPointer(
        state="absent", generation_id=None, error=None
    )
    status = read_status(experiment)
    assert status["publication_integrity"] == "absent"
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False
    assert read_winner(experiment, "p") is None


def test_pointer_to_a_deleted_generation_namespace_reports_failed(tmp_path: Path) -> None:
    """A pointer whose whole target namespace is gone is corruption, not a fresh tree."""
    experiment, generation_id = _golden_publication(tmp_path)

    shutil.rmtree(_generation_dir(experiment, generation_id))

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert status["publication_error"] == pointer.error
    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


def test_resume_path_still_raises_the_manifest_error(tmp_path: Path) -> None:
    """The reporting verdict must not soften the raise the resume path depends on."""
    experiment, generation_id = _golden_publication(tmp_path)

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    with pytest.raises(PublicationIntegrityError, match="does not match its recorded hash"):
        _last_successful_generation_id(experiment, raise_on_manifest_error=True)


def test_dangling_last_success_pointer_is_corrupt_and_blocks_rerun(tmp_path: Path) -> None:
    owner, _ = _golden_publication(tmp_path)
    pointer_path = _last_successful_generation_path(owner)
    current_path = _generation_path(owner)
    current = current_path.read_bytes()
    pointer_path.unlink()
    pointer_path.symlink_to("missing-target.yaml")
    assert pointer_path.is_symlink() and not pointer_path.exists()

    assert _resolve_publication_pointer(owner).state == "failed"
    with pytest.raises(PublicationIntegrityError):
        run_experiment(owner)

    assert pointer_path.is_symlink()
    assert current_path.read_bytes() == current


def test_surviving_pointer_prevents_root_projection_fallback_after_generation_loss(
    tmp_path: Path,
) -> None:
    experiment, _ = _golden_publication(tmp_path)
    _generation_path(experiment).unlink()
    generations = _generations_dir(experiment)
    generations.rename(generations.with_name(f"saved-{generations.name}"))

    assert _resolve_publication_pointer(experiment).state == "failed"
    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False
    assert read_winner(experiment, "p") is None


def test_dangling_generation_root_does_not_enable_projected_winner(tmp_path: Path) -> None:
    experiment, _ = _golden_publication(tmp_path)
    assert read_winner(experiment, "p") is not None
    _last_successful_generation_path(experiment).unlink()
    _generation_path(experiment).unlink()
    generations = _generations_dir(experiment)
    generations.rename(generations.with_name("saved-generations"))
    generations.symlink_to("missing-generations", target_is_directory=True)
    assert generations.is_symlink() and not generations.exists()

    status = read_status(experiment)

    assert status["publication_integrity"] == "absent"
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False
    assert read_winner(experiment, "p") is None


# --------------------------------------------------------------------------
# Identity fields (read_status): pinned reads of a failed-publication
# generation, and single-capture consistency.
# --------------------------------------------------------------------------


@pytest.mark.integration
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

    monkeypatch.setattr(generation_ops, "_validate_generation_publishable", fail_validation)

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
    experiment, real_generation = _golden_publication(tmp_path)

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
# Generation provenance: the configuration and identity that produced a
# published result are frozen into its namespace (review v0.5.18 / finding F6).
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_generation_namespace_freezes_the_config_that_produced_it(tmp_path: Path) -> None:
    """A published generation keeps the canonical config that produced it, owner-only.

    Before this, a generation kept only the config *fingerprint*: once the
    operator edited or lost the YAML, the digest could prove a mismatch but
    could not reconstruct the search spaces, fixed overrides, environment, or
    or trial command behind a published winner.
    """
    experiment = _stored_experiment(tmp_path, env={"TRAINER_TOKEN": _SENTINEL_SECRET})
    with temporary_umask(0o022):
        run_experiment(experiment)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    snapshot_path, _ = _provenance_paths(experiment, generation_id)

    # The snapshot may hold secrets (``env:``), so it is owner-only even though
    # its directory is deliberately operator-readable.
    assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o600

    snapshot = yaml.safe_load(snapshot_path.read_text())
    expected_snapshot = experiment.model_dump(mode="json")
    expected_snapshot["execution"]["cwd"] = str(Path.cwd().resolve())
    assert snapshot == expected_snapshot

    # Spot-check the values an operator actually needs to reproduce the run.
    assert snapshot["trial_command"] == experiment.trial_command
    assert snapshot["env"] == {"TRAINER_TOKEN": _SENTINEL_SECRET}
    phase = snapshot["phases"][0]
    assert phase["fixed_overrides"] == {"batch_size": 8}
    assert phase["search_space"]["x"]["type"] == "int"
    assert phase["search_space"]["x"]["low"] == 0
    assert phase["search_space"]["x"]["high"] == 10


@pytest.mark.integration
def test_generation_reproducibility_record_is_shareable_digests_only(tmp_path: Path) -> None:
    """The readable provenance record carries identity and digests, never config values."""
    experiment = _stored_experiment(tmp_path, env={"TRAINER_TOKEN": _SENTINEL_SECRET})
    experiment = experiment.model_copy(
        update={
            "phases": [
                *experiment.phases,
                Phase(
                    name="q",
                    n_trials=1,
                    sampler=Sampler(type="random", seed=1),
                    search_space={"y": IntParam(type="int", low=0, high=10)},
                ),
            ]
        }
    )
    with temporary_umask(0o022):
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
    phase_fingerprints = record["phase_config_fingerprints"]
    assert [item["name"] for item in phase_fingerprints] == [
        phase.name for phase in experiment.phases
    ]
    assert all(len(item["sha256"]) == 64 for item in phase_fingerprints), phase_fingerprints
    assert len({item["sha256"] for item in phase_fingerprints}) == len(experiment.phases)
    assert record["config_snapshot"] == {
        "path": _CONFIG_SNAPSHOT_NAME,
        "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }

    # Digests, never values: nothing from the snapshot's contents leaks here,
    # least of all a configured env secret.
    assert _SENTINEL_SECRET not in raw
    assert "TRAINER_TOKEN" not in raw
    assert experiment.trial_command not in raw


@pytest.mark.integration
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


def test_generation_id_source_rejects_linked_provenance_record(tmp_path: Path) -> None:
    """A source claim is not trusted through a linked reproducibility record."""
    experiment, generation_id = _golden_publication(tmp_path)

    record_path = _generation_dir(experiment, generation_id) / _REPRODUCIBILITY_NAME
    preserved_record_path = record_path.with_name("reproducibility.original.json")
    record_path.rename(preserved_record_path)
    record_path.symlink_to(preserved_record_path.name)

    assert generation_id_source(experiment, generation_id) is None


def test_generation_manifest_covers_the_provenance_files(tmp_path: Path) -> None:
    """Both provenance files are manifest-listed with their content hashes.

    Without this the publication validator would reject every new generation:
    the namespace may hold nothing the manifest does not list.
    """
    experiment, generation_id = _golden_publication(tmp_path)
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
    experiment, generation_id = _golden_publication(tmp_path)

    target = _generation_dir(experiment, generation_id) / filename
    original = target.read_bytes()

    target.write_bytes(original + b"\n# tampered\n")
    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None
    assert "does not match its recorded hash" in pointer.error
    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in status["publication_error"]
    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None

    target.write_bytes(original)
    assert _last_successful_generation_id(experiment) == generation_id
    assert read_winner(experiment, "p") is not None

    target.unlink()
    assert _last_successful_generation_id(experiment) is None
    assert read_winner(experiment, "p") is None


@requires_nonroot
@pytest.mark.parametrize("filename", [_CONFIG_SNAPSHOT_NAME, _REPRODUCIBILITY_NAME])
def test_unreadable_provenance_file_reports_permission_denied_not_corruption(
    tmp_path: Path,
    filename: str,
) -> None:
    """A provenance file this user may not read fails closed with its own reason.

    Re-review v0.5.19 / observation N1: ``config.snapshot.yaml`` is owner-only,
    so a second operator reading a perfectly healthy tree got the generic
    "missing or unreadable" verdict and the "inspect or restore the generation
    namespace" remedy -- for a permission bit. The verdict is
    ``permission_denied`` (an unvalidatable tree is still not exposed as a
    published one), and the reason points at the publishing user.
    """
    experiment, generation_id = _golden_publication(tmp_path)

    target = _generation_dir(experiment, generation_id) / filename
    original_mode = stat.S_IMODE(target.stat().st_mode)
    target.chmod(0o000)
    try:
        status = read_status(experiment)
        assert status["publication_integrity"] == "permission_denied"
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
        workdir=tmp_path / "runs", trainer=trainer, n_trials=1, max_consecutive_failures=1
    )

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    failed_generation = yaml.safe_load(_generation_path(experiment).read_text())["generation_id"]
    snapshot_path, repro_path = _provenance_paths(experiment, failed_generation)
    assert snapshot_path.is_file()
    assert repro_path.is_file()
    snapshot = yaml.safe_load(snapshot_path.read_text())
    expected_snapshot = experiment.model_dump(mode="json")
    expected_snapshot["execution"]["cwd"] = str(Path.cwd().resolve())
    assert snapshot == expected_snapshot
    assert json.loads(repro_path.read_text())["generation_id"] == failed_generation


def test_partially_dropped_provenance_record_is_current_format_tampering(tmp_path: Path) -> None:
    """Half a provenance record is a current-format tamper, not valid state.

    The two files are written together at claim time, so a namespace that
    keeps one and drops the other must fail closed.
    """
    experiment, generation_id = _golden_publication(tmp_path)
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
    experiment, generation_id = _golden_publication(tmp_path)

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
        workdir=tmp_path / "runs", trainer=trainer, n_trials=1, max_consecutive_failures=1
    )
    original = generation_ops._write_generation_state

    monkeypatch.setattr(
        generation_ops,
        "_write_generation_state",
        _fail_terminal_generation_state(original, SystemExit("simulated persistence interruption")),
    )

    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(experiment)
