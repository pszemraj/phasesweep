"""PhaseSweep scores a trial only from files that resolve inside its own attempt directory."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import optuna
import pytest

import phasesweep.engine.evidence as evidence_ops
from phasesweep import run_experiment
from phasesweep.config.models import Metric, Observation
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.paths import _last_successful_generation_path
from phasesweep.engine.selection import PhaseTrialsFailedError, select_winner
from phasesweep.engine.state import FAILURE_REASON_ATTR, TRIAL_DIR_ATTR
from phasesweep.engine.trial import _observe
from phasesweep.evidence.evaluation import ExtractorError, evaluate_gates, run_extractor
from phasesweep.evidence.models import (
    ArtifactSizeGate,
    Gate,
    JsonEnvelopeExtractor,
    JsonEqualsGate,
    JsonExtractor,
    JsonScalarBoundGate,
    LogRegexExtractor,
    ObjectiveExtractor,
    Sha256Gate,
)
from tests.conftest import (
    make_experiment,
    make_trial_context,
    phase_study,
    phase_trials,
    write_constant_trainer,
    write_trainer,
)


def _refusal(relative: str, attempt: Path, target: Path) -> str:
    """Return the refusal text for one evidence file that resolves outside its attempt.

    :param str relative: Configured trial-relative path.
    :param Path attempt: The attempt directory the path was read from.
    :param Path target: Where the path really resolves.
    :return str: The full refusal text.
    """
    return (
        f"`{relative}` in attempt directory {attempt} resolves to {target.resolve()}, outside "
        "the attempt. Have the trainer write it under PHASESWEEP_TRIAL_DIR."
    )


def _envelope(value: float) -> str:
    """Return a complete envelope for the attempt :func:`make_trial_context` builds.

    :param float value: Objective value it reports.
    :return str: The envelope's JSON text.
    """
    return json.dumps(
        {
            "schema_version": 1,
            "status": "complete",
            "generation_id": "generation-test",
            "attempt_id": "attempt-test",
            "overrides_sha256": "a" * 64,
            "objective": {"name": "loss", "split": "validation", "value": value},
            "evaluation": {"policy": "final", "checkpoint": "final.pt", "step": 10},
        }
    )


_ENVELOPE = JsonEnvelopeExtractor(
    type="json_envelope",
    path="result.json",
    objective_name="loss",
    split="validation",
    policy="final",
)


@pytest.mark.integration
def test_trials_that_link_one_shared_result_both_fail_and_publish_no_winner(
    tmp_path: Path,
) -> None:
    """Two trials whose result.json links name one overwritten file are refused, not tied."""
    shared = tmp_path / "shared.json"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, os, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--score", type=float)
        score = parser.parse_known_args()[0].score
        shared = Path({str(shared)!r})
        result = Path(os.environ["PHASESWEEP_TRIAL_DIR"]) / "result.json"
        high, low = shared.with_name("high"), shared.with_name("low")
        def wait_for(marker):
            deadline = time.monotonic() + 30
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
        if score == 0.8:
            shared.write_text(json.dumps({{"score": 0.8}}))
            result.symlink_to(shared)
            high.touch()
            wait_for(low)
        else:
            wait_for(high)
            shared.write_text(json.dumps({{"score": 0.3}}))
            result.symlink_to(shared)
            low.touch()
        """,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        metric=Metric(
            name="score",
            goal="maximize",
            extractor=JsonExtractor(type="json", path="result.json", key="score"),
        ),
        n_trials=2,
        n_jobs=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        sampler=Sampler(type="grid"),
        search_space={"score": CategoricalParam(type="categorical", choices=[0.8, 0.3])},
    )

    with pytest.raises(PhaseTrialsFailedError):
        run_experiment(experiment)

    trials = phase_trials(experiment)
    assert [trial.state for trial in trials] == [optuna.trial.TrialState.FAIL] * 2
    for trial in trials:
        attempt = Path(trial.user_attrs[TRIAL_DIR_ATTR])
        assert trial.user_attrs[FAILURE_REASON_ATTR] == "metric extractor: " + _refusal(
            "result.json", attempt, shared
        )
    assert not _last_successful_generation_path(experiment).exists()


@pytest.mark.parametrize(
    ("layout", "cfg", "expected"),
    [
        pytest.param(
            "plain", JsonExtractor(type="json", path="result.json", key="x"), 0.5, id="json"
        ),
        pytest.param(
            "nested-link",
            JsonExtractor(type="json", path="latest/result.json", key="x"),
            0.5,
            id="json-through-an-in-attempt-directory-link",
        ),
        pytest.param("envelope-link", _ENVELOPE, 0.25, id="envelope-through-an-in-attempt-link"),
        pytest.param(
            "log",
            LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)"),
            0.75,
            id="stdout-log",
        ),
    ],
)
def test_files_inside_the_attempt_still_read(
    tmp_path: Path, layout: str, cfg: ObjectiveExtractor, expected: float
) -> None:
    """A regular file, or a link that stays inside the attempt, reads as before."""
    if layout == "plain":
        (tmp_path / "result.json").write_text('{"x": 0.5}')
    elif layout == "nested-link":
        (tmp_path / "checkpoint-500").mkdir()
        (tmp_path / "checkpoint-500" / "result.json").write_text('{"x": 0.5}')
        (tmp_path / "latest").symlink_to("checkpoint-500")
    elif layout == "envelope-link":
        (tmp_path / "eval").mkdir()
        (tmp_path / "eval" / "final.json").write_text(_envelope(0.25))
        (tmp_path / "result.json").symlink_to(tmp_path / "eval" / "final.json")
    else:
        (tmp_path / "stdout.log").write_text("step 1\nx=0.75\n")

    assert run_extractor(make_trial_context(tmp_path), cfg) == expected


_OUTSIDE_EXTRACTORS = [
    pytest.param(JsonExtractor(type="json", path="result.json", key="x"), id="json"),
    pytest.param(_ENVELOPE, id="json_envelope"),
    pytest.param(
        LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.]+)"), id="log_regex"
    ),
]


def _link_outside(tmp_path: Path, relative: str, content: str) -> tuple[Path, Path]:
    """Make ``relative`` inside a fresh attempt a link to a readable file outside it.

    :param Path tmp_path: Test directory holding the attempt and the outside file.
    :param str relative: Path inside the attempt that becomes the link.
    :param str content: Valid evidence for the reader under test.
    :return tuple[Path, Path]: The attempt directory and the outside file.
    """
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    outside = tmp_path / "outside" / Path(relative).name
    outside.parent.mkdir()
    outside.write_text(content)
    (attempt / relative).symlink_to(outside)
    return attempt, outside


@pytest.mark.parametrize("cfg", _OUTSIDE_EXTRACTORS)
def test_every_extractor_refuses_a_file_outside_the_attempt(
    tmp_path: Path, cfg: ObjectiveExtractor
) -> None:
    """Objective and constraint extractors fail the trial with the refusal as its reason."""
    relative = "stdout.log" if isinstance(cfg, LogRegexExtractor) else "result.json"
    content = {
        "json": '{"x": 0.5}',
        "json_envelope": _envelope(0.25),
        "log_regex": "x=0.75\n",
    }[cfg.type]
    attempt, outside = _link_outside(tmp_path, relative, content)

    with pytest.raises(ExtractorError) as refused:
        run_extractor(make_trial_context(attempt), cfg)

    assert str(refused.value) == _refusal(relative, attempt, outside)


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(
            JsonEqualsGate(type="json_equals", path="gate.json", key="ok", value=True),
            id="json_equals",
        ),
        pytest.param(
            JsonScalarBoundGate(type="json_scalar_bound", path="gate.json", key="n", max=10),
            id="json_scalar_bound",
        ),
        pytest.param(
            ArtifactSizeGate(
                type="artifact_size", source="json", path="gate.json", key="n", max_bytes=10
            ),
            id="artifact_size-json",
        ),
        pytest.param(Sha256Gate(type="sha256", path="gate.json", sha256="0" * 64), id="sha256"),
    ],
)
def test_every_value_gate_refuses_a_file_outside_the_attempt(tmp_path: Path, gate: Gate) -> None:
    """A gate that reads a value or a digest fails on a file that resolves outside the attempt."""
    attempt, outside = _link_outside(tmp_path, "gate.json", '{"ok": true, "n": 3}')

    (result,) = evaluate_gates(make_trial_context(attempt), [gate])

    assert result.passed is False
    assert _refusal("gate.json", attempt, outside) in result.detail


def test_presence_and_size_gates_keep_following_links(tmp_path: Path) -> None:
    """A size check reads no value, so a file gate still measures what its link names."""
    attempt, _outside = _link_outside(tmp_path, "model.bin", "payload")

    (result,) = evaluate_gates(
        make_trial_context(attempt),
        [ArtifactSizeGate(type="artifact_size", source="file", path="model.bin", max_bytes=64)],
    )

    assert result.passed is True


def test_an_observation_outside_the_attempt_is_omitted_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A report-only value outside the attempt is left out, as any unreadable observation is."""
    attempt, outside = _link_outside(tmp_path, "result.json", '{"x": 0.5}')
    observation = Observation(
        name="seen", extractor=JsonExtractor(type="json", path="result.json", key="x")
    )

    with caplog.at_level(logging.WARNING):
        assert _observe(make_trial_context(attempt), [observation], deadline=None) == {}

    assert "observation seen not recorded: " + _refusal("result.json", attempt, outside) in (
        caplog.text
    )


@pytest.mark.integration
def test_a_trainer_that_replaces_its_stdout_log_fails_the_trial(tmp_path: Path) -> None:
    """The log PhaseSweep captures follows the same rule when the trainer swaps it for a link."""
    outside = tmp_path / "elsewhere.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import os
        from pathlib import Path
        log = Path(os.environ["PHASESWEEP_TRIAL_DIR"]) / "stdout.log"
        Path({str(outside)!r}).write_text("x=0.1\\n")
        log.unlink()
        log.symlink_to({str(outside)!r})
        """,
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, n_trials=1)

    with pytest.raises(PhaseTrialsFailedError):
        run_experiment(experiment)

    (trial,) = phase_trials(experiment)
    attempt = Path(trial.user_attrs[TRIAL_DIR_ATTR])
    assert trial.user_attrs[FAILURE_REASON_ATTR] == "metric extractor: " + _refusal(
        "stdout.log", attempt, outside
    )


@pytest.mark.parametrize("filename", ["r.json", "overrides_resolved.json"])
def test_verification_refuses_evidence_replaced_by_an_outside_link_with_identical_bytes(
    tmp_path: Path, filename: str
) -> None:
    """Selection preflight and winner verification refuse a recorded file that now resolves outside."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        metric=Metric(extractor=JsonExtractor(type="json", path="r.json", key="x")),
        n_trials=1,
    )
    run_experiment(experiment)
    (trial,) = phase_trials(experiment)
    attempt = Path(trial.user_attrs[TRIAL_DIR_ATTR])
    outside = tmp_path / "outside" / filename
    outside.parent.mkdir()
    shutil.copyfile(attempt / filename, outside)
    (attempt / filename).unlink()
    (attempt / filename).symlink_to(outside)
    study = phase_study(experiment)
    refusal = _refusal(filename, attempt, outside)

    with pytest.raises(TrialEvidenceMissingError) as selection:
        evidence_ops._validate_selection_evidence(experiment, {"p": study})
    with pytest.raises(TrialEvidenceMissingError) as winner:
        evidence_ops._verify_winner_evidence(
            experiment, experiment.phases[0], select_winner(study, experiment, phase_name="p")
        )
    with pytest.raises(TrialEvidenceMissingError) as resumed:
        run_experiment(experiment)

    for refused in (selection, winner, resumed):
        assert str(refused.value).endswith("but " + refusal)
