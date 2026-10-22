"""Publication-transaction fault injection.

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
from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.generation as generation_ops
import phasesweep.engine.publication_validation as validation_ops
from phasesweep import run_experiment
from phasesweep.cli import cli as cli_main
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import (
    PublicationAccessError,
    PublicationCommitError,
    PublicationIntegrityError,
)
from phasesweep.engine.paths import (
    _experiment_dir,
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
    _resolve_publication_pointer,
    _unresolvable_pointer,
)
from phasesweep.engine.read import read_result, read_status
from phasesweep.engine.resume import _preflight_skipped_winners, _published_for_resume
from phasesweep.engine.run import TerminalReport
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import Winner
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.runtime import shutdown as runtime_shutdown
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import (
    make_experiment,
    patch_directory_fsync_failure,
    patch_path_method_failure,
    requires_nonroot,
    temporary_umask,
    write_constant_trainer,
    write_experiment_config,
    write_param_echo_trainer,
    write_trainer,
)
from tests.ledger_fixtures import (
    golden_publication,
    published_generation_id,
    reanchor_summary_pointer,
    reseal_winner,
)


def _fail_terminal_generation_state(original: Callable, error: BaseException):
    def flaky_state(owner, **kwargs: object):
        if kwargs.get("state") == "failed":
            raise error
        return original(owner, **kwargs)

    return flaky_state


# Provenance files frozen into every generation namespace at claim time.
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


def _topped_up(experiment: Experiment) -> Experiment:
    """Return ``experiment`` with phase ``p`` asking for one more trial than its first run.

    A run of it launches a trial when the study holds one, and otherwise replays winners
    that differ from the one-trial publication, so either way it publishes.

    :param Experiment experiment: The one-trial experiment from :func:`_stored_experiment`.
    :return Experiment: The same experiment at ``n_trials=2``.
    """
    return experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )


def _provenance_paths(experiment, generation_id: str) -> tuple[Path, Path]:
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
            experiment_name="expected",
            generation_id="generation-test",
        )


def _fail_precommit_validation(monkeypatch: pytest.MonkeyPatch, experiment: Experiment) -> None:
    """Make the next publication's pre-commit validation raise ``RuntimeError``."""
    del experiment

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated pre-commit validation failure")

    monkeypatch.setattr(generation_ops, "_validate_generation_publishable", fail_validation)


def _fail_pointer_commit(monkeypatch: pytest.MonkeyPatch, experiment: Experiment) -> None:
    """Make writing the last-success pointer raise ``OSError``."""
    pointer_path = _last_successful_generation_path(experiment)
    original_write = artifact_io._write_yaml_atomic

    def flaky_write(path: Path, payload: object) -> None:
        if path == pointer_path:
            raise OSError("simulated pointer commit failure")
        return original_write(path, payload)

    monkeypatch.setattr(artifact_io, "_write_yaml_atomic", flaky_write)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("inject_failure", "error_type", "error_match"),
    [
        pytest.param(
            _fail_precommit_validation,
            RuntimeError,
            "simulated pre-commit validation failure",
            id="precommit-validation",
        ),
        # A failure writing the last-success pointer itself is the same failure shape as validation.
        pytest.param(
            _fail_pointer_commit,
            OSError,
            "simulated pointer commit failure",
            id="pointer-commit",
        ),
    ],
)
def test_publication_commit_failure_keeps_prior_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inject_failure: Callable[[pytest.MonkeyPatch, Experiment], None],
    error_type: type[Exception],
    error_match: str,
) -> None:
    """A failure before the pointer commits leaves the prior publication authoritative."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = published_generation_id(experiment)
    assert first_generation is not None

    inject_failure(monkeypatch, experiment)

    captured: list[str] = []

    def capture(report: TerminalReport) -> None:
        captured.append(report.generation_id)

    with pytest.raises(error_type, match=error_match):
        run_experiment(_topped_up(experiment), terminal_callback=capture)

    assert len(captured) == 1
    second_generation = captured[0]
    assert second_generation != first_generation
    assert (
        yaml.safe_load(_generation_path(experiment).read_text())["generation_id"]
        == second_generation
    )

    # The failed generation's own record and current pointer both name the
    # specific publication_failed outcome, not a generic failure.
    assert _record_state(experiment, second_generation) == "publication_failed"
    assert _current_pointer_state(experiment) == "publication_failed"
    payload = yaml.safe_load(_generation_record_path(experiment, second_generation).read_text())
    assert payload["error_class"] == error_type.__name__

    # The prior publication is untouched: the pointer never advanced.
    assert published_generation_id(experiment) == first_generation
    _, (published,) = read_result(experiment, phase_names=["p"])
    assert published is not None
    assert published.source is not None
    assert published.source.generation_id == first_generation


@pytest.mark.integration
@pytest.mark.parametrize(
    "prepare_fails", [True, False], ids=["prepare-fails", "committed-interrupts"]
)
def test_publication_sidecar_observes_the_pointer_commit_boundary(
    tmp_path: Path, prepare_fails: bool
) -> None:
    """Preparation failures block publication; committed notification cannot downgrade it."""
    experiment = _stored_experiment(tmp_path)
    prepared_generation: str | None = None
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
            nonlocal prepared_generation
            prepared_generation = generation_id
            assert set(winners) == {"p"}
            assert summary["generation_id"] == generation_id
            assert published_generation_id(experiment) is None
            events.append(f"prepared:{generation_id}")
            if prepare_fails:
                raise OSError("simulated detached snapshot persistence failure")

        def committed(self, *, generation_id: str) -> None:
            assert not prepare_fails, f"uncommitted generation was notified: {generation_id}"
            assert published_generation_id(experiment) == generation_id
            events.append(f"committed:{generation_id}")
            raise KeyboardInterrupt("simulated postcommit interruption")

    if prepare_fails:
        with pytest.raises(OSError, match="detached snapshot persistence failure"):
            run_experiment(experiment, publication_hook=ObservingHook())
        assert prepared_generation is not None
        assert published_generation_id(experiment) is None
        assert _record_state(experiment, prepared_generation) == "publication_failed"
    else:
        winners = run_experiment(experiment, publication_hook=ObservingHook())
        assert set(winners) == {"p"}
        generation_id = published_generation_id(experiment)
        assert generation_id is not None
        assert events == [f"prepared:{generation_id}", f"committed:{generation_id}"]
        assert _record_state(experiment, generation_id) == "published"


# --------------------------------------------------------------------------
# Post-commit failures (steps 4 and 5): the run must still succeed.
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "fault",
    [
        pytest.param("atomic_create_text", id="immutable-record-write"),
        pytest.param("_copy_yaml_projection", id="convenience-projection"),
    ],
)
def test_postcommit_write_failure_leaves_run_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fault: str,
) -> None:
    """Postcommit bookkeeping cannot downgrade an authoritative publication."""
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment)
    first_generation = published_generation_id(experiment)
    assert first_generation is not None
    record_write = fault == "atomic_create_text"

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated projection failure")

    if record_write:
        _lose_record_writes(monkeypatch)
    else:
        monkeypatch.setattr(generation_ops, fault, fail_write)

    with caplog.at_level(logging.ERROR, logger="phasesweep.engine.generation"):
        winners = run_experiment(_topped_up(experiment))

    assert set(winners) == {"p"}
    second_generation = published_generation_id(experiment)
    assert second_generation is not None
    assert second_generation != first_generation
    if record_write:
        assert any(
            r.name == "phasesweep.engine.generation"
            and "failed to write the immutable generation record" in r.message
            for r in caplog.records
        )
        assert not _generation_record_path(experiment, second_generation).is_file()
    else:
        assert _record_state(experiment, second_generation) == "published"
        assert any(
            "failed to refresh the current-generation pointer or convenience projections"
            in r.message
            for r in caplog.records
        )
        # The seeded second trial measures better, so the new publication selected its own winner.
        published_winner_path = _generation_winner_path(experiment, second_generation, "p")
        assert published_winner_path.is_file()
        _, (published,) = read_result(experiment, phase_names=["p"])
        assert published is not None
        assert published.source is not None
        assert published.source.generation_id == second_generation


@pytest.mark.integration
def test_directory_fsync_failure_after_pointer_rename_still_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory-fsync failure after the pointer rename cannot fail the publication.

    ``os.replace`` is the commit; the directory fsync after it is durability
    bookkeeping. An fsync error must not surface as an exception from the
    pointer write, or the already-committed publication would be rewritten as
    ``publication_failed``.
    """
    experiment = _stored_experiment(tmp_path)

    patch_directory_fsync_failure(monkeypatch, "simulated directory fsync failure")

    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    assert _record_state(experiment, generation_id) == "published"
    assert _current_pointer_state(experiment) == "published"


@pytest.mark.integration
def test_control_flow_exception_from_postcommit_record_write_cannot_downgrade_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KeyboardInterrupt escaping a post-commit step must not reclassify the publication.

    ``_log_on_failure`` catches ``BaseException``; a control-flow exception
    from the record write would otherwise propagate into the outer terminal
    handler and rewrite the committed publication as ``failed``.
    """
    experiment = _stored_experiment(tmp_path)

    original = generation_ops.atomic_create_text

    def interrupt_record_write(path: Path, text: str) -> bool:
        if path.name == "generation.yaml":
            raise KeyboardInterrupt
        return original(path, text)

    monkeypatch.setattr(generation_ops, "atomic_create_text", interrupt_record_write)

    winners = run_experiment(experiment)

    assert set(winners) == {"p"}
    generation_id = published_generation_id(experiment)
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
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    assert _record_state(experiment, generation_id) == "published"


@pytest.mark.integration
@pytest.mark.signals_own_pid
def test_shutdown_signal_during_publication_is_absorbed_until_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown signal racing the publication transaction loses to the commit.

    The signal lands mid-transaction (during pre-commit validation); the
    publication must still commit and the shutdown must be delivered at the
    next checkpoint — never rewritten into a ``failed``/``publication_failed``
    state for the committed generation.
    """
    experiment = _stored_experiment(tmp_path)

    original_validate = generation_ops._validate_generation_publishable

    def validate_then_signal(
        *args: object, **kwargs: object
    ) -> tuple[validation_ops.ValidatedPublication, bytes]:
        validated = original_validate(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return validated

    monkeypatch.setattr(generation_ops, "_validate_generation_publishable", validate_then_signal)

    try:
        with pytest.raises(PhaseSweepShutdown) as exc_info:
            run_experiment(experiment)

        assert exc_info.value.signum == signal.SIGTERM
        assert exc_info.value.published_result_committed is True
        generation_id = published_generation_id(experiment)
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
    experiment, generation_id = golden_publication(tmp_path)
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
    assert published_generation_id(experiment) == generation_id

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


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(b"\xff", id="invalid-utf8"),
        pytest.param(b"state: published\nupdated_at: 2026-99-99\n", id="invalid-timestamp"),
    ],
)
def test_record_refusal_logs_a_malformed_existing_record(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    contents: bytes,
) -> None:
    """A malformed existing record is refused and logged with no recorded state."""
    experiment, generation_id = golden_publication(tmp_path)
    record_path = _generation_record_path(experiment, generation_id)
    record_path.write_bytes(contents)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.generation"):
        generation_ops._write_generation_state(
            experiment,
            generation_id=generation_id,
            state="failed",
            from_phase=None,
            publish_current=False,
        )

    assert record_path.read_bytes() == contents
    refusals = [r for r in caplog.records if "Refusing to rewrite" in r.message]
    assert len(refusals) == 1
    assert "existing state None" in refusals[0].getMessage()


@pytest.mark.integration
def test_successful_publication_never_logs_a_record_refusal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The happy path never exercises the write-once refusal branch.

    The post-commit current-pointer refresh passes ``write_record=False``, so
    a normal successful publication must not log the "Refusing to rewrite"
    warning -- that message has to stay a real anomaly signal, not routine
    noise on every publication. The second publication replaces a pointer
    whose target already has its record, which must stay byte for byte as
    written.
    """
    experiment = _stored_experiment(tmp_path)
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.generation"):
        run_experiment(experiment)
        first_generation = published_generation_id(experiment)
        assert first_generation is not None
        first_record = _generation_record_path(experiment, first_generation).read_bytes()
        run_experiment(_topped_up(experiment))
    assert published_generation_id(experiment) != first_generation
    assert _generation_record_path(experiment, first_generation).read_bytes() == first_record
    assert not [r for r in caplog.records if "Refusing to rewrite" in r.message]


# --------------------------------------------------------------------------
# The pointer never stops naming a generation that has no record: it is the
# only durable proof of a commit whose controller died before the record write.
# --------------------------------------------------------------------------


def _lose_record_writes(monkeypatch: pytest.MonkeyPatch, *, only: Path | None = None) -> None:
    """Make writing an immutable record raise ``OSError``, for one record path or every one."""
    original = generation_ops.atomic_create_text

    def flaky(path: Path, text: str) -> bool:
        if path.name == "generation.yaml" and (only is None or path == only):
            raise OSError("simulated record write failure")
        return original(path, text)

    monkeypatch.setattr(generation_ops, "atomic_create_text", flaky)


def _publish_without_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Experiment, str]:
    """Commit a publication whose record write never happened, as after a post-commit crash.

    :param Path tmp_path: Per-test temporary directory.
    :param pytest.MonkeyPatch monkeypatch: Source of the crash patch, scoped to this call.
    :return tuple[Experiment, str]: The experiment and the generation the pointer now names.
    """
    experiment = _stored_experiment(tmp_path)
    with monkeypatch.context() as crash:
        _lose_record_writes(crash)
        run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    assert not _generation_record_path(experiment, generation_id).exists()
    return experiment, generation_id


@pytest.mark.integration
def test_next_publication_records_the_generation_whose_record_was_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the pointer first writes the ``published`` record of the generation it leaves."""
    experiment, first_generation = _publish_without_record(tmp_path, monkeypatch)

    run_experiment(_topped_up(experiment))

    second_generation = published_generation_id(experiment)
    assert second_generation is not None
    assert second_generation != first_generation
    record = yaml.safe_load(_generation_record_path(experiment, first_generation).read_text())
    assert record["state"] == "published"
    assert record["generation_id"] == first_generation
    assert record["experiment"] == experiment.experiment
    # The original invocation's resume point is not recoverable.
    assert record["from_phase"] is None
    assert record["error_class"] is None
    assert _record_state(experiment, second_generation) == "published"


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(None, id="published"),
        pytest.param(b"state: failed\n", id="other-state"),
        pytest.param(b"\xff", id="malformed"),
    ],
)
def test_replacing_the_pointer_never_rewrites_an_existing_record(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    contents: bytes | None,
) -> None:
    """A record of any state or condition is left as written, without a refusal warning."""
    experiment, generation_id = golden_publication(tmp_path)
    record_path = _generation_record_path(experiment, generation_id)
    if contents is not None:
        record_path.write_bytes(contents)
    before = record_path.read_bytes()

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.generation"):
        generation_ops._record_superseded_publication(experiment, superseding_id="next")

    assert record_path.read_bytes() == before
    assert not caplog.records


@pytest.mark.integration
@pytest.mark.parametrize(
    "inject_failure",
    [
        pytest.param(_fail_precommit_validation, id="precommit-validation"),
        pytest.param(_fail_pointer_commit, id="pointer-commit"),
    ],
)
def test_failed_publication_is_never_recorded_as_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inject_failure: Callable[[pytest.MonkeyPatch, Experiment], None],
) -> None:
    """A generation the pointer never named has nothing to reconcile into ``published``."""
    experiment, first_generation = _publish_without_record(tmp_path, monkeypatch)
    captured: list[str] = []

    topped_up = _topped_up(experiment)
    with monkeypatch.context() as failing:
        inject_failure(failing, experiment)
        with pytest.raises((RuntimeError, OSError)):
            run_experiment(topped_up, terminal_callback=lambda r: captured.append(r.generation_id))

    [failed_generation] = captured
    assert published_generation_id(experiment) == first_generation
    assert _record_state(experiment, failed_generation) == "publication_failed"

    run_experiment(topped_up)

    assert published_generation_id(experiment) not in {first_generation, failed_generation}
    assert _record_state(experiment, first_generation) == "published"
    assert _record_state(experiment, failed_generation) == "publication_failed"


@pytest.mark.integration
def test_unwritable_record_for_the_superseded_generation_keeps_the_prior_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the proof cannot be preserved the pointer does not move, and a later publication retries."""
    experiment, first_generation = _publish_without_record(tmp_path, monkeypatch)
    first_record = _generation_record_path(experiment, first_generation)
    captured: list[str] = []
    topped_up = _topped_up(experiment)

    with monkeypatch.context() as failing:
        _lose_record_writes(failing, only=first_record)
        with pytest.raises(OSError, match="simulated record write failure"):
            run_experiment(topped_up, terminal_callback=lambda r: captured.append(r.generation_id))

    [failed_generation] = captured
    assert published_generation_id(experiment) == first_generation
    assert not first_record.exists()
    assert _record_state(experiment, failed_generation) == "publication_failed"
    assert _current_pointer_state(experiment) == "publication_failed"
    failed = yaml.safe_load(_generation_record_path(experiment, failed_generation).read_text())
    assert failed["error_class"] == "OSError"

    run_experiment(topped_up)

    assert published_generation_id(experiment) not in {first_generation, failed_generation}
    assert _record_state(experiment, first_generation) == "published"


# --------------------------------------------------------------------------
# The generation manifest is validated as a complete
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

    The validator must not loop the current config and silently ``continue``
    past an invalid winner file: deleting or modifying the artifact immediately
    before validation would otherwise publish an officially successful
    generation with no verified winner.
    """
    experiment = _stored_experiment(tmp_path)

    original = generation_ops._validate_generation_publishable

    def delete_winner_then_validate(exp, generation_id: str) -> None:
        mutate_winner(_generation_winner_path(exp, generation_id, "p"))
        original(exp, generation_id)

    monkeypatch.setattr(
        generation_ops, "_validate_generation_publishable", delete_winner_then_validate
    )

    with pytest.raises(RuntimeError, match=error_match):
        run_experiment(experiment)

    assert published_generation_id(experiment) is None
    assert _current_pointer_state(experiment) == "publication_failed"


@pytest.mark.parametrize(
    ("field", "value", "mirror_summary", "error_match"),
    [
        pytest.param(
            "kind", "unknown_source", True, "invalid winner_source kind", id="unknown-kind"
        ),
        pytest.param("phase", "../other", True, "winner_source phase name", id="unsafe-phase"),
        pytest.param("phase", "other", True, "names phase 'other', expected 'p'", id="other-phase"),
        pytest.param(
            "trial_number",
            999,
            True,
            "winner_source that disagrees with its trial_number",
            id="trial-number",
        ),
        pytest.param(
            "trial_number", True, True, "invalid winner_source trial_number", id="bool-trial"
        ),
        pytest.param(
            "trial_number", 0.0, True, "invalid winner_source trial_number", id="float-trial"
        ),
        pytest.param(
            "trial_number", "0", True, "invalid winner_source trial_number", id="string-trial"
        ),
        pytest.param("study", 999, True, "missing [], unexpected ['study']", id="extra-key"),
        pytest.param(
            "phase_fingerprint",
            999,
            True,
            "has no valid phase_fingerprint",
            id="phase-fingerprint",
        ),
        # The hash-covered winner's two provenance copies must identify one attempt.
        pytest.param(
            "generation_id",
            "different-generation_id",
            True,
            "winner_source that disagrees with its generation_id",
            id="source-generation-id",
        ),
        pytest.param(
            "attempt_id",
            "different-attempt_id",
            True,
            "winner_source that disagrees with its attempt_id",
            id="source-attempt-id",
        ),
        # Only the winner's copy changes, so the summary's own copy disagrees with it.
        pytest.param(
            "generation_id",
            "different-generation_id",
            False,
            "disagrees with the summary winner_source",
            id="summary-copy-generation-id",
        ),
        pytest.param(
            "attempt_id",
            "different-attempt_id",
            False,
            "disagrees with the summary winner_source",
            id="summary-copy-attempt-id",
        ),
    ],
)
def test_manifest_rejects_malformed_reanchored_winner_provenance(
    tmp_path: Path, field: str, value: object, mirror_summary: bool, error_match: str
) -> None:
    """Each malformed field is refused by its own check, not by an earlier cross-check.

    With ``mirror_summary`` the summary's per-phase ``winner_source`` copy agrees
    with the mutated winner, so the summary cross-check passes and only the
    winner's payload validation can refuse it.
    """
    experiment, generation_id = golden_publication(tmp_path)
    winner = yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())
    if field == "phase_fingerprint":
        winner[field] = value
    else:
        winner["winner_source"][field] = value
    reseal_winner(experiment, generation_id, winner, mirror_summary=mirror_summary)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert error_match in pointer.error
    assert read_status(experiment)["publication_integrity"] == "failed"
    assert read_result(experiment, phase_names=["p"])[1] == []


def test_load_winner_rejects_linked_winner_with_current_summary(tmp_path: Path) -> None:
    """Strict resume rejects a linked winner from a current-format publication."""
    experiment, generation_id = golden_publication(tmp_path)

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    preserved_winner_path = winner_path.with_name("winner.original.yaml")
    winner_path.rename(preserved_winner_path)
    winner_path.symlink_to(preserved_winner_path.name)

    with pytest.raises(PublicationIntegrityError, match="missing or unreadable"):
        artifact_io._load_winner(
            experiment,
            experiment.phases[0],
            {},
            publication=_published_for_resume(experiment),
        )


@pytest.mark.parametrize(
    "phase_plan",
    [
        pytest.param([{"name": "../p", "comment": None}], id="unsafe-name"),
        pytest.param([{"name": "p", "comment": 3}], id="non-string-comment"),
        pytest.param(["p"], id="bare-name"),
        pytest.param([], id="empty"),
        pytest.param(None, id="absent"),
    ],
)
def test_manifest_rejects_a_malformed_phase_plan(tmp_path: Path, phase_plan: object) -> None:
    """Plan names become path components, so one bad entry fails the publication."""
    experiment, generation_id = golden_publication(tmp_path)
    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["phase_plan"] = phase_plan
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)

    assert _resolve_publication_pointer(experiment).state == "failed"
    assert read_status(experiment)["publication_integrity"] == "failed"


def test_manifest_rejects_a_duplicate_winner_entry_naming_its_phase(tmp_path: Path) -> None:
    """A phase listed twice in the artifact manifest fails the publication and names the phase."""
    experiment, generation_id = golden_publication(tmp_path)
    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    winner_entry = next(item for item in summary["artifacts"] if item.get("kind") == "winner")
    summary["artifacts"].append(dict(winner_entry))
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert f"duplicate artifact entry for winner {winner_entry['phase']!r}" in pointer.error


def test_published_reads_consume_the_bytes_validation_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A winner swapped after validation read it never reaches a consumer.

    Each validation first restores the bytes the manifest hashed, then swaps
    the file once it returns, so a consumer that reopened the winner by path
    would see the swapped metric.
    """
    experiment, generation_id = golden_publication(tmp_path)
    winner_path = _generation_winner_path(experiment, generation_id, "p")
    validated_bytes = winner_path.read_bytes()
    swapped = yaml.safe_load(validated_bytes)
    validated_metric = swapped["metric"][experiment.metric.name]
    swapped["metric"][experiment.metric.name] = validated_metric + 1.0
    original_validate = validation_ops._validate_generation_manifest

    def validate_then_swap(*args: Any, **kwargs: Any) -> validation_ops.ValidatedPublication:
        winner_path.write_bytes(validated_bytes)
        validated = original_validate(*args, **kwargs)
        winner_path.write_text(yaml.safe_dump(swapped, sort_keys=False))
        return validated

    monkeypatch.setattr(validation_ops, "_validate_generation_manifest", validate_then_swap)

    _, (winner,) = read_result(experiment, phase_names=["p"])
    assert winner is not None
    assert winner.metric == validated_metric
    _status, winners = read_result(experiment)
    assert [view.metric for view in winners] == [validated_metric]
    snapshot = capture_result_snapshot(experiment)
    assert [frozen["metric"] for frozen in snapshot["winners"]] == [validated_metric]
    resumed = artifact_io._load_winner(
        experiment,
        experiment.phases[0],
        {},
        publication=_published_for_resume(experiment),
    )
    assert resumed.metric == validated_metric
    config_path = tmp_path / "experiment.yaml"
    write_experiment_config(config_path, experiment)
    result = CliRunner().invoke(cli_main, ["show-winners", str(config_path)])
    assert result.exit_code == 0, result.output
    assert validated_bytes.decode("utf-8") in result.output


# --------------------------------------------------------------------------
# Pointer validation keys on artifact identity, not record state.
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
    experiment, generation_id = golden_publication(tmp_path)

    _last_successful_generation_path(experiment).write_text(tamper.format(gid=generation_id))

    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []
    # The pointer file itself is still there, so this is corruption of a
    # recorded publication -- never an experiment that has published nothing.
    assert _resolve_publication_pointer(experiment).state == "failed"


def test_pointer_to_generation_with_tampered_summary_is_not_authoritative(tmp_path: Path) -> None:
    """A pointer whose target's own summary no longer names it fails closed.

    Pointer validation reads back the generation's immutable *summary*, not
    the (post-commit, best-effort) lifecycle record -- so tampering the record
    has no effect on publication status; tampering the summary does.
    """
    experiment, generation_id = golden_publication(tmp_path)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["generation_id"] = "not-this-generation"
    summary_path.write_text(yaml.safe_dump(summary))
    # Re-anchored, so the pointer's byte anchors pass and only the summary's own identity is refused.
    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None
    assert "does not name experiment" in pointer.error
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []


def test_tampering_the_record_state_does_not_affect_publication_status(tmp_path: Path) -> None:
    """The record is informational only; publication status ignores its state entirely."""
    experiment, generation_id = golden_publication(tmp_path)

    record_path = _generation_record_path(experiment, generation_id)
    record = yaml.safe_load(record_path.read_text())
    record["state"] = "failed"
    record_path.write_text(yaml.safe_dump(record))

    assert published_generation_id(experiment) == generation_id
    assert read_result(experiment, phase_names=["p"])[1] != []


# --------------------------------------------------------------------------
# Publication integrity states: nothing published and a recorded publication
# that no longer validates are distinct facts.
# --------------------------------------------------------------------------


def test_publication_pointer_reports_ok_for_a_healthy_publication(tmp_path: Path) -> None:
    """A valid publication resolves ``ok`` with its generation id and no error."""
    experiment, generation_id = golden_publication(tmp_path)

    assert _resolve_publication_pointer(experiment) == PublicationPointer(
        state="ok", generation_id=generation_id, error=None
    )

    status = read_status(experiment)
    assert status["publication_integrity"] == "ok"
    assert "publication_error" not in status
    assert status["published_generation_id"] == generation_id


def test_experiment_pointer_anchors_the_exact_summary_bytes(tmp_path: Path) -> None:
    """The commit record stores the byte length and SHA-256 of the validated summary."""
    experiment, generation_id = golden_publication(tmp_path)

    summary = _generation_summary_path(experiment, generation_id).read_bytes()
    pointer = yaml.safe_load(_last_successful_generation_path(experiment).read_text())
    assert pointer["summary_size_bytes"] == len(summary)
    assert pointer["summary_sha256"] == hashlib.sha256(summary).hexdigest()


def test_summary_digest_is_checked_before_the_summary_is_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer-backed summary with wrong bytes is rejected before YAML sees them."""
    experiment, generation_id = golden_publication(tmp_path)

    summary_path = _generation_summary_path(experiment, generation_id)
    original = summary_path.read_bytes()
    tampered = original.replace(b"experiment:", b"experimenT:", 1)
    assert len(tampered) == len(original)
    summary_path.write_bytes(tampered)

    parsed_summary_bytes = False
    original_safe_load = validation_ops.yaml.safe_load

    def track_safe_load(stream):
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
    [
        "schema_version",
        "phase_plan",
        "metric_goal",
        "objective_evidence",
        "config_fingerprint",
        "completion",
    ],
)
def test_summary_semantics_are_pointer_anchored_and_cross_checked(
    tmp_path: Path,
    tamper: str,
) -> None:
    """Historical plan, metric/evidence, and config identity cannot self-authenticate."""
    experiment, generation_id = golden_publication(tmp_path)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    if tamper == "schema_version":
        summary["schema_version"] = 3
        semantic_error = "unsupported summary schema_version 3"
    elif tamper == "phase_plan":
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
    experiment, generation_id = golden_publication(tmp_path)

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
    assert status["published_generation_id"] is None
    assert status["represented_generation_id"] is None
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []

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
    experiment, generation_id = golden_publication(tmp_path)

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
    assert read_result(experiment, phase_names=["p"])[1] == []


@pytest.mark.parametrize("damage", ["deleted-target", "missing-generation-root"])
def test_generation_loss_with_a_surviving_pointer_reports_failed(
    tmp_path: Path, damage: str
) -> None:
    """Losing published generation artifacts cannot authorize convenience-winner fallback."""
    experiment, generation_id = golden_publication(tmp_path)
    if damage == "deleted-target":
        shutil.rmtree(_generation_dir(experiment, generation_id))
    else:
        _generation_path(experiment).unlink()
        generations = _generations_dir(experiment)
        generations.rename(generations.with_name(f"saved-{generations.name}"))

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.generation_id == generation_id
    assert pointer.error is not None

    status = read_status(experiment)
    assert status["publication_integrity"] == "failed"
    assert status["publication_error"] == pointer.error
    assert status["is_published"] is False
    assert status["phases"][0]["winner_present"] is False
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []


def test_resume_path_still_raises_the_manifest_error(tmp_path: Path) -> None:
    """The reporting verdict must not soften the raise the resume path depends on."""
    experiment, generation_id = golden_publication(tmp_path)

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    with pytest.raises(PublicationIntegrityError, match="does not match its recorded hash"):
        published_generation_id(experiment, raise_on_manifest_error=True)


def test_from_phase_preflight_validates_the_published_manifest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping several phases costs one manifest validation, not one per phase."""
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trainer=trainer,
        phases=[
            Phase(
                name=name,
                inherits=[previous] if previous is not None else [],
                n_trials=1,
                sampler=Sampler(type="random", seed=seed),
                search_space={parameter: IntParam(type="int", low=0, high=4)},
            )
            for seed, (name, previous, parameter) in enumerate(
                [("a", None, "x"), ("b", "a", "y"), ("c", "b", "z"), ("d", "c", "w")]
            )
        ],
    )
    run_experiment(experiment)

    validate = Mock(wraps=validation_ops._validate_generation_manifest)
    monkeypatch.setattr(validation_ops, "_validate_generation_manifest", validate)

    winners = _preflight_skipped_winners(experiment, from_phase="d", run_deadline=None)

    assert set(winners) == {"a", "b", "c"}
    validate.assert_called_once()


def test_dangling_last_success_pointer_is_corrupt_and_blocks_rerun(tmp_path: Path) -> None:
    owner, _ = golden_publication(tmp_path)
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


def _memory_publication(tmp_path: Path) -> tuple[Experiment, str]:
    """Publish one generation from a ``storage: null`` run.

    :param Path tmp_path: Per-test temporary directory.
    :return tuple[Experiment, str]: The in-memory experiment and its published generation id.
    """
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(workdir=tmp_path / "runs", trainer=trainer, n_trials=1)
    assert experiment.resolved_storage is None
    run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    return experiment, generation_id


def _assert_memory_launches_refuse_before_any_write(
    experiment: Experiment, error: type[Exception], match: str
) -> None:
    """Require a ``storage: null`` run, dry run, and resume to refuse without writing.

    :param Experiment experiment: In-memory experiment whose publication was made unusable.
    :param type[Exception] error: Refusal every launch must raise.
    :param str match: Pattern the refusal's message must contain.
    """
    root = _experiment_dir(experiment)
    pointer_path = _last_successful_generation_path(experiment)
    pointer_mode = stat.S_IMODE(pointer_path.lstat().st_mode)
    before = {path: path.lstat().st_mtime_ns for path in [root, *root.rglob("*")]}
    with pytest.raises(error, match=match):
        run_experiment(experiment)
    with pytest.raises(error, match=match):
        run_experiment(experiment, dry_run=True)
    with pytest.raises(error, match=match):
        run_experiment(_with_later_phase(experiment), from_phase="q")
    assert stat.S_IMODE(pointer_path.lstat().st_mode) == pointer_mode
    assert {path: path.lstat().st_mtime_ns for path in [root, *root.rglob("*")]} == before


@pytest.mark.parametrize("damage", ["edited_winner", "malformed_pointer"])
def test_memory_run_refuses_a_corrupt_publication_before_any_write(
    tmp_path: Path, damage: str
) -> None:
    """``storage: null`` has no ledger to check, yet a corrupt publication still blocks a rerun."""
    experiment, generation_id = _memory_publication(tmp_path)
    if damage == "edited_winner":
        _tamper_winner_artifact(_generation_winner_path(experiment, generation_id, "p"))
    else:
        _last_successful_generation_path(experiment).write_text("not a pointer\n")
    assert _resolve_publication_pointer(experiment).state == "failed"

    _assert_memory_launches_refuse_before_any_write(
        experiment, PublicationIntegrityError, "no longer validates"
    )


@requires_nonroot
@pytest.mark.parametrize("unreadable", ["winner", "pointer"])
def test_memory_run_refuses_an_unreadable_publication_before_any_write(
    tmp_path: Path, unreadable: str
) -> None:
    """``storage: null`` refuses a publication it may not read, rather than publishing past it."""
    experiment, generation_id = _memory_publication(tmp_path)
    path = (
        _generation_winner_path(experiment, generation_id, "p")
        if unreadable == "winner"
        else _last_successful_generation_path(experiment)
    )
    original_mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(0o000)
    try:
        assert _resolve_publication_pointer(experiment).state == "permission_denied"
        _assert_memory_launches_refuse_before_any_write(
            experiment, PublicationAccessError, "cannot be validated as the current user"
        )
    finally:
        path.chmod(original_mode)
    assert published_generation_id(experiment) == generation_id


def _with_later_phase(experiment: Experiment) -> Experiment:
    """Append a phase ``q`` after the published phase, so a resume can start from it."""
    later = Phase(
        name="q",
        inherits=["p"],
        n_trials=1,
        sampler=Sampler(type="random", seed=1),
        search_space={"b": IntParam(type="int", low=0, high=10)},
    )
    return experiment.model_copy(update={"phases": [*experiment.phases, later]})


@pytest.mark.parametrize(
    ("tamper", "match"),
    [("append", "byte length"), ("rewrite", "digest")],
)
def test_summary_anchor_mismatch_raises_under_the_resume_flag(
    tmp_path: Path, tamper: str, match: str
) -> None:
    """A summary that disagrees with its pointer is a refusal, not a verdict, for a resume."""
    experiment, generation_id = golden_publication(tmp_path)
    summary_path = _generation_summary_path(experiment, generation_id)
    content = summary_path.read_bytes()
    summary_path.write_bytes(content + b"\n" if tamper == "append" else content[:-2] + b"#\n")

    verdict = _resolve_publication_pointer(experiment)
    assert verdict.state == "failed" and verdict.error is not None and match in verdict.error
    with pytest.raises(PublicationIntegrityError, match=match):
        _resolve_publication_pointer(experiment, raise_on_manifest_error=True)
    with pytest.raises(PublicationIntegrityError, match=match):
        _published_for_resume(experiment)


@pytest.mark.parametrize("damage", ["dangling_pointer", "malformed_pointer", "missing_summary"])
def test_unusable_publication_is_refused_for_resume_never_read_as_absent(
    tmp_path: Path, damage: str
) -> None:
    """A recorded publication that cannot be used refuses a resume instead of answering "no winner yet"."""
    experiment, generation_id = golden_publication(tmp_path)
    pointer_path = _last_successful_generation_path(experiment)
    if damage == "dangling_pointer":
        pointer_path.unlink()
        pointer_path.symlink_to("missing-target.yaml")
    elif damage == "malformed_pointer":
        pointer_path.write_text("not a pointer\n")
    else:
        _generation_summary_path(experiment, generation_id).unlink()
    assert _resolve_publication_pointer(experiment, raise_on_manifest_error=True).state == "failed"

    with pytest.raises(PublicationIntegrityError):
        _published_for_resume(experiment)
    with pytest.raises(PublicationIntegrityError):
        run_experiment(_with_later_phase(experiment), from_phase="q", dry_run=True)


def test_absent_publication_is_still_nothing_published_for_resume(tmp_path: Path) -> None:
    """Only a tree that never published reads as absent."""
    experiment, _ = golden_publication(tmp_path)
    _last_successful_generation_path(experiment).unlink()

    assert _published_for_resume(experiment) is None


@requires_nonroot
def test_unreadable_pointer_is_an_access_refusal_for_resume(tmp_path: Path) -> None:
    """A pointer this user may not read is a permission refusal, not an absent publication."""
    experiment, _ = golden_publication(tmp_path)
    pointer_path = _last_successful_generation_path(experiment)
    original_mode = stat.S_IMODE(pointer_path.stat().st_mode)
    pointer_path.chmod(0o000)
    try:
        assert _resolve_publication_pointer(experiment).state == "permission_denied"
        with pytest.raises(PublicationAccessError, match="permission denied"):
            _published_for_resume(experiment)
        with pytest.raises(PublicationAccessError):
            run_experiment(_with_later_phase(experiment), from_phase="q", dry_run=True)
    finally:
        pointer_path.chmod(original_mode)


def test_dangling_generation_root_does_not_enable_projected_winner(tmp_path: Path) -> None:
    experiment, _ = golden_publication(tmp_path)
    assert read_result(experiment, phase_names=["p"])[1] != []
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
    assert read_result(experiment, phase_names=["p"])[1] == []


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
    first_generation = published_generation_id(experiment)
    assert first_generation is not None

    def fail_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated pre-commit validation failure")

    monkeypatch.setattr(generation_ops, "_validate_generation_publishable", fail_validation)

    captured: list[str] = []

    def capture(report: TerminalReport) -> None:
        captured.append(report.generation_id)

    with pytest.raises(RuntimeError):
        run_experiment(_topped_up(experiment), terminal_callback=capture)

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
    experiment, real_generation = golden_publication(tmp_path)

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
# published result are frozen into its namespace.
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_generation_namespace_freezes_the_config_that_produced_it(tmp_path: Path) -> None:
    """A published generation keeps the canonical config that produced it, owner-only.

    The config *fingerprint* alone is not enough: once the operator edited or
    lost the YAML, the digest could prove a mismatch but could not reconstruct
    the search spaces, fixed overrides, environment, or trial command behind a
    published winner.
    """
    experiment = _stored_experiment(tmp_path, env={"TRAINER_TOKEN": _SENTINEL_SECRET})
    with temporary_umask(0o022):
        run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    snapshot_path, _ = _provenance_paths(experiment, generation_id)

    # The snapshot may hold secrets (``env:``), so it is owner-only even though
    # its directory is deliberately operator-readable.
    assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o600

    snapshot = yaml.safe_load(snapshot_path.read_text())
    expected_snapshot = experiment.model_dump(mode="json")
    expected_snapshot["execution"]["cwd"] = str(Path.cwd().resolve())
    assert snapshot == expected_snapshot

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
    generation_id = published_generation_id(experiment)
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
    # records "caller".
    assert record["generation_id_source"] == "engine"
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
    exists even after that state dir is deleted or replaced.
    """
    experiment = _stored_experiment(tmp_path)
    run_experiment(experiment, generation_id="launcher-granted-1")

    record = json.loads(
        (_generation_dir(experiment, "launcher-granted-1") / _REPRODUCIBILITY_NAME).read_text()
    )
    assert record["generation_id_source"] == "caller"


def test_generation_manifest_covers_the_provenance_files(tmp_path: Path) -> None:
    """Both provenance files are manifest-listed with their content hashes.

    Without this the publication validator would reject every new generation:
    the namespace may hold nothing the manifest does not list.
    """
    experiment, generation_id = golden_publication(tmp_path)
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
    experiment, generation_id = golden_publication(tmp_path)

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
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []

    target.write_bytes(original)
    assert published_generation_id(experiment) == generation_id
    assert read_result(experiment, phase_names=["p"])[1] != []

    target.unlink()
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []


@requires_nonroot
@pytest.mark.parametrize("filename", [_CONFIG_SNAPSHOT_NAME, _REPRODUCIBILITY_NAME])
def test_unreadable_provenance_file_reports_permission_denied_not_corruption(
    tmp_path: Path,
    filename: str,
) -> None:
    """A provenance file this user may not read reports ``permission_denied`` and names the publishing user."""
    experiment, generation_id = golden_publication(tmp_path)

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
    assert published_generation_id(experiment) == generation_id


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
    experiment, generation_id = golden_publication(tmp_path)
    _, repro_path = _provenance_paths(experiment, generation_id)

    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    summary["artifacts"] = [
        item for item in summary["artifacts"] if item.get("kind") != "reproducibility"
    ]
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)
    repro_path.unlink()

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "does not list the complete generation provenance record" in pointer.error
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []


@pytest.mark.parametrize(
    ("filename", "mutate", "error_match"),
    [
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {**payload, "generation_id": "not-this-generation"},
            "reproducibility artifact names a different generation",
            id="reproducibility-names-another-generation",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {
                **payload,
                "config_snapshot": {**payload["config_snapshot"], "sha256": "0" * 64},
            },
            "reproducibility artifact does not anchor the config snapshot it published",
            id="reproducibility-unanchors-the-snapshot",
        ),
        pytest.param(
            _CONFIG_SNAPSHOT_NAME,
            lambda payload: {**payload, "experiment": "some-other-experiment"},
            "config_snapshot artifact names a different experiment",
            id="snapshot-names-another-experiment",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {
                key: value for key, value in payload.items() if key != "code_revision"
            },
            "reproducibility artifact has a malformed code_revision (the member is missing)",
            id="reproducibility-drops-the-code-revision",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {**payload, "code_revision": "a" * 40},
            "reproducibility artifact has a malformed code_revision "
            "(Expected a JSON object, not str.)",
            id="reproducibility-records-a-bare-commit",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {**payload, "code_revision": {"commit": "a" * 40}},
            "reproducibility artifact has a malformed code_revision "
            "(it must be null or hold exactly commit and dirty)",
            id="reproducibility-omits-the-dirty-flag",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {**payload, "code_revision": {"commit": "abc123", "dirty": False}},
            "reproducibility artifact has a malformed code_revision "
            "(commit must be a full lowercase hex git commit id)",
            id="reproducibility-records-a-short-commit",
        ),
        pytest.param(
            _REPRODUCIBILITY_NAME,
            lambda payload: {**payload, "code_revision": {"commit": "a" * 40, "dirty": "no"}},
            "reproducibility artifact has a malformed code_revision (dirty must be a boolean)",
            id="reproducibility-records-a-non-boolean-dirty",
        ),
    ],
)
def test_rehashed_provenance_edit_still_fails_the_manifest_cross_checks(
    tmp_path: Path,
    filename: str,
    mutate,
    error_match: str,
) -> None:
    """Re-hashing an edited provenance file into the manifest does not launder it.

    Same escalation the winner artifacts get: an operator who updates the
    recorded hash after editing still trips the summary cross-checks, so the
    published record cannot be made to describe a different config or
    generation.
    """
    experiment, generation_id = golden_publication(tmp_path)

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
    # Re-anchored, so the pointer's byte anchors pass and the cross-check itself refuses it.
    reanchor_summary_pointer(_last_successful_generation_path(experiment), summary_path)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert error_match in pointer.error
    assert published_generation_id(experiment) is None
    assert read_result(experiment, phase_names=["p"])[1] == []


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
