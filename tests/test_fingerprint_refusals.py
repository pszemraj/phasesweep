"""A refused phase fingerprint names the keys that changed, from the payload the study stored."""

from __future__ import annotations

import re
from pathlib import Path

import optuna
import pytest

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import ExecutionContext, Experiment
from phasesweep.config.search import IntParam
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.fingerprints import (
    _phase_fingerprint,
    _phase_fingerprint_payload,
    _render_changed_values,
    _semantic_payload_digest,
    _verify_fingerprint,
)
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.state import PHASE_FINGERPRINT_ATTR, PHASE_FINGERPRINT_PAYLOAD_ATTR
from phasesweep.errors import NEW_STUDY_STEP, OperatorAction
from tests.conftest import make_experiment
from tests.experiments import anchored_phase, project_directory, write_anchored_config

_CWD_SENTENCE = "The trainer working directory follows the directory phasesweep runs from"


def _stamped_study(experiment: Experiment, *, populated: bool = True) -> optuna.Study:
    """Return an in-memory study bound to ``experiment``'s first phase, with one trial by default."""
    study = optuna.create_study()
    _verify_fingerprint(study, experiment, experiment.phases[0], {})
    if populated:
        study.optimize(lambda trial: 0.0, n_trials=1)
    return study


def _refusal(study: optuna.Study, experiment: Experiment) -> StudyFingerprintMismatchError:
    """Verify ``experiment`` against ``study`` and return the refusal it raises."""
    with pytest.raises(StudyFingerprintMismatchError) as caught:
        _verify_fingerprint(study, experiment, experiment.phases[0], {})
    return caught.value


def test_refusal_names_the_changed_search_space_bound(tmp_path: Path) -> None:
    """Widening one bound refuses the study and names that key with both values."""
    project = project_directory(tmp_path)
    config = write_anchored_config(
        project / "exp.yaml",
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "."},
        phases=[anchored_phase(3)],
    )
    run_experiment(load_experiment(config))
    write_anchored_config(
        config,
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "."},
        phases=[anchored_phase(4)],
    )

    with pytest.raises(StudyFingerprintMismatchError, match="different phase config") as caught:
        run_experiment(load_experiment(config))

    assert "Changed: phase.search_space.x.high: 3 -> 4." in str(caught.value)
    assert "execution.cwd" not in str(caught.value)
    assert caught.value.action is OperatorAction.FRESH_NAMESPACE


def test_refusal_names_the_trainer_cwd_when_the_same_config_runs_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset ``execution.cwd`` follows the invocation directory, and the refusal says so."""
    project = project_directory(tmp_path)
    config = write_anchored_config(
        project / "exp.yaml",
        workdir="runs",
        storage="journal:///study.journal",
        phases=[anchored_phase(3)],
    )
    first_cwd = (tmp_path / "first").resolve()
    second_cwd = (tmp_path / "second").resolve()
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    run_experiment(load_experiment(config))
    monkeypatch.chdir(second_cwd)
    with pytest.raises(StudyFingerprintMismatchError, match="different phase config") as caught:
        run_experiment(load_experiment(config))

    message = _operational_error_text(caught.value)
    # No ";" before the closing quote: the invocation directory is the only change.
    assert re.search(r'Changed: execution\.cwd: [^;]*first" -> [^;]*second"\.\nfix:', message)
    assert str(caught.value) == str(caught.value).rstrip()
    assert (
        f"{_CWD_SENTENCE} when execution.cwd is unset; run from the original directory or set "
        "execution.cwd in the config (relative paths resolve against the config file)."
    ) in message
    # Running from the original directory keeps the study, so a new name is not offered.
    assert "new `experiment` name" not in message
    assert caught.value.action is OperatorAction.FIX_CONFIG


def test_stored_payload_hashes_to_the_stored_fingerprint(tmp_path: Path) -> None:
    """The payload a study keeps is what its fingerprint digests, after a storage round trip."""
    experiment = make_experiment()
    phase = experiment.phases[0]
    storage_url = f"journal:///{tmp_path / 'study.journal'}"
    optuna.create_study(study_name="t::p", storage=_resolve_storage(storage_url))
    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(storage_url))

    fingerprint = _verify_fingerprint(study, experiment, phase, {})

    stored = optuna.load_study(study_name="t::p", storage=_resolve_storage(storage_url)).user_attrs
    assert stored[PHASE_FINGERPRINT_ATTR] == fingerprint
    assert _semantic_payload_digest(stored[PHASE_FINGERPRINT_PAYLOAD_ATTR]) == fingerprint
    assert stored[PHASE_FINGERPRINT_PAYLOAD_ATTR] == _phase_fingerprint_payload(
        experiment, phase, {}
    )
    assert fingerprint == _phase_fingerprint(experiment, phase, {})


def test_rebinding_an_empty_study_replaces_the_stored_payload() -> None:
    """A study with no trials rebinds to the new config, payload and fingerprint together."""
    narrow = make_experiment(search_space={"x": IntParam(type="int", low=0, high=10)})
    wide = make_experiment(search_space={"x": IntParam(type="int", low=0, high=11)})
    study = _stamped_study(narrow, populated=False)

    fingerprint = _verify_fingerprint(study, wide, wide.phases[0], {})

    assert study.user_attrs[PHASE_FINGERPRINT_ATTR] == fingerprint
    assert _semantic_payload_digest(study.user_attrs[PHASE_FINGERPRINT_PAYLOAD_ATTR]) == fingerprint


def test_refusal_lists_ten_changed_keys_then_counts_the_rest() -> None:
    """A wholesale change stays one readable sentence."""
    stored = make_experiment(env={f"K{index:02d}": "1" for index in range(12)})
    study = _stamped_study(stored)

    message = str(_refusal(study, make_experiment()))

    assert message.count(" -> <absent>") == 10
    assert 'Changed: env.K00: "1" -> <absent>; env.K01: "1" -> <absent>;' in message
    assert message.endswith('env.K09: "1" -> <absent>; and 2 more.')
    assert "env.K10" not in message


def test_refusal_gives_the_step_that_runs_the_changed_config() -> None:
    """A changed input leaves one way forward, so the message names it beside the routing."""
    study = _stamped_study(make_experiment(provenance={"revision": "toy-v1"}))

    error = _refusal(study, make_experiment(provenance={"revision": "toy-v2"}))

    assert 'Changed: provenance.revision: "toy-v1" -> "toy-v2".' in str(error)
    assert str(error) == str(error).rstrip()
    assert error.remediation == NEW_STUDY_STEP
    assert _operational_error_text(error).endswith(NEW_STUDY_STEP)
    assert error.action is OperatorAction.FRESH_NAMESPACE


def test_refusal_for_a_configured_cwd_omits_the_unset_cwd_advice() -> None:
    """A configured ``execution.cwd`` that changed is named, without the unset-cwd advice."""
    study = _stamped_study(make_experiment(execution=ExecutionContext(cwd="/srv/a")))

    error = _refusal(study, make_experiment(execution=ExecutionContext(cwd="/srv/b")))

    assert 'Changed: execution.cwd: "/srv/a" -> "/srv/b".' in str(error)
    assert str(error) == str(error).rstrip()
    assert _CWD_SENTENCE not in str(error)
    assert error.action is OperatorAction.FRESH_NAMESPACE


def test_refusal_without_a_stored_payload_keeps_the_hash_only_message() -> None:
    """A stored digest with no payload beside it still refuses, naming only the hashes."""
    experiment = make_experiment()
    study = optuna.create_study()
    study.set_user_attr(PHASE_FINGERPRINT_ATTR, "0" * 64)
    study.optimize(lambda trial: 0.0, n_trials=1)

    error = _refusal(study, experiment)
    message = str(error)

    fingerprint = _phase_fingerprint(experiment, experiment.phases[0], {})
    assert message == (
        f"Study {study.study_name!r} was created with a different phase config "
        f"(fingerprint {'0' * 64} != {fingerprint}). "
    )
    assert error.action is OperatorAction.FRESH_NAMESPACE
    assert error.remediation == NEW_STUDY_STEP


def test_long_values_are_windowed_around_their_first_difference() -> None:
    """Two long values that differ late still show where, and every value stays short."""
    old, new = _render_changed_values("x" * 100 + "a", "x" * 100 + "b")
    assert (old, new) == ("..." + "x" * 24 + 'a"', "..." + "x" * 24 + 'b"')

    old, new = _render_changed_values("x" * 200, "y" * 200)
    assert old == '"' + "x" * 79 + "..."
    assert new == '"' + "y" * 79 + "..."

    assert _render_changed_values({"low": 0}, None) == ('{"low":0}', "null")
