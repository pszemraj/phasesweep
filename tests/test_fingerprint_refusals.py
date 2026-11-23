"""A refused phase fingerprint or artifact-root binding names what changed."""

from __future__ import annotations

import json
import re
from pathlib import Path

import optuna
import pytest

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import ExecutionContext, Experiment, Phase
from phasesweep.config.reference import lookup
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.artifact_roots import (
    _check_artifact_root_binding,
    _check_study_artifact_root,
)
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
)
from phasesweep.engine.fingerprints import (
    _fresh_identity,
    _phase_fingerprint,
    _phase_fingerprint_payload,
    _render_changed_values,
    _semantic_payload_digest,
    _stamp_identity,
    _verify_fingerprint,
)
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _artifact_root_binding_path, _experiment_dir
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR, Winner, _recorded_identity
from phasesweep.errors import (
    FRESH_STUDY,
    NEW_STUDY_STEP,
    OperatorAction,
    rename_step,
)
from tests.conftest import make_experiment, phase_trials, write_constant_trainer
from tests.experiments import (
    anchored_phase,
    phase_payload,
    project_directory,
    write_anchored_config,
)
from tests.ledger_fixtures import materialize

_CWD_SENTENCE = "execution.cwd is unset, so the trainer runs where phasesweep is started"


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


@pytest.mark.integration
def test_refusal_names_the_changed_search_space_bound(tmp_path: Path) -> None:
    """Narrowing one bound refuses the study, names that key with both values and what it dropped, and offers the phase rename that then runs."""
    project = project_directory(tmp_path)
    config = write_anchored_config(
        project / "exp.yaml",
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "."},
        phases=[anchored_phase(4)],
    )
    run_experiment(load_experiment(config))
    write_anchored_config(
        config,
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "."},
        phases=[anchored_phase(3)],
    )

    with pytest.raises(StudyFingerprintMismatchError, match="different config") as caught:
        run_experiment(load_experiment(config))

    assert str(caught.value).endswith(
        "Changed: phase.search_space.x.high: 4 -> 3. search_space 'x' dropped high 4; added "
        "values alone keep the study and run only the new ones."
    )
    assert "execution.cwd" not in str(caught.value)
    assert caught.value.action is OperatorAction.FRESH_NAMESPACE
    assert caught.value.remediation == (
        "Put back x high 4, or give `p` a new `name`, which runs it again from scratch."
    )

    write_anchored_config(
        config,
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "."},
        phases=[{**anchored_phase(3), "name": "narrower"}],
    )
    assert set(run_experiment(load_experiment(config))) == {"narrower"}


@pytest.mark.integration
def test_refusal_for_an_inherited_phase_names_the_dependents_to_rename(tmp_path: Path) -> None:
    """A changed phase that another inherits reruns by renaming both, and the fix line names the dependent."""
    project = project_directory(tmp_path)
    fields: dict[str, object] = {
        "workdir": "runs",
        "storage": "journal:///study.journal",
        "execution": {"cwd": "."},
    }
    child = phase_payload(
        name="q", inherits=["p"], search_space={"y": {"type": "int", "low": 0, "high": 3}}
    )
    config = write_anchored_config(
        project / "exp.yaml", phases=[anchored_phase(4), child], **fields
    )
    run_experiment(load_experiment(config))
    write_anchored_config(config, phases=[anchored_phase(3), child], **fields)

    with pytest.raises(StudyFingerprintMismatchError, match="::p'") as caught:
        run_experiment(load_experiment(config))

    assert caught.value.remediation == (
        "Put back x high 4, or give `p` and `q` new names and update the `inherits` "
        "that name them, which runs them again from scratch."
    )

    renamed = [
        {**anchored_phase(3), "name": "narrower"},
        {**child, "name": "after", "inherits": ["narrower"]},
    ]
    write_anchored_config(config, phases=renamed, **fields)
    assert set(run_experiment(load_experiment(config))) == {"narrower", "after"}


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
    with pytest.raises(StudyFingerprintMismatchError, match="different config") as caught:
        run_experiment(load_experiment(config))

    message = _operational_error_text(caught.value)
    # No ";" before the closing quote: the invocation directory is the only change.
    assert re.search(r'Changed: execution\.cwd: [^;]*first" -> [^;]*second"\.\nfix:', message)
    assert str(caught.value) == str(caught.value).rstrip()
    assert (
        f"fix: {_CWD_SENTENCE}: run from the original directory, or set execution.cwd to it."
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

    stored = _recorded_identity(
        optuna.load_study(study_name="t::p", storage=_resolve_storage(storage_url))
    )
    assert stored is not None
    assert stored["fingerprint"] == fingerprint
    assert _semantic_payload_digest(stored["payload"]) == fingerprint
    assert stored["payload"] == _phase_fingerprint_payload(experiment, phase, {})
    assert fingerprint == _phase_fingerprint(experiment, phase, {})


def test_rebinding_an_empty_study_replaces_the_stored_payload() -> None:
    """A study with no trials rebinds to the new config, payload and fingerprint together."""
    narrow = make_experiment(search_space={"x": IntParam(type="int", low=0, high=10)})
    wide = make_experiment(search_space={"x": IntParam(type="int", low=0, high=11)})
    study = _stamped_study(narrow, populated=False)

    fingerprint = _verify_fingerprint(study, wide, wide.phases[0], {})

    stored = _recorded_identity(study)
    assert stored is not None
    assert stored["fingerprint"] == fingerprint
    assert _semantic_payload_digest(stored["payload"]) == fingerprint
    assert stored["search_spaces"] == [
        {
            "search_space": {"x": {"type": "int", "low": 0, "high": 11, "log": False, "step": 1}},
            "first_position": 0,
        }
    ]


def test_refusal_lists_ten_changed_keys_then_counts_the_rest() -> None:
    """A wholesale change stays one readable sentence."""
    stored = make_experiment(env={f"K{index:02d}": "1" for index in range(12)})
    study = _stamped_study(stored)

    message = str(_refusal(study, make_experiment()))

    assert message.count(" -> <absent>") == 10
    assert 'Changed: env.K00: "1" -> <absent>; env.K01: "1" -> <absent>;' in message
    assert message.endswith('env.K09: "1" -> <absent>; and 2 more.')
    assert "env.K10" not in message


def test_refusal_leads_with_restoring_the_recorded_value_it_shows() -> None:
    """The message shows the old value, so restoring it comes before starting a fresh study."""
    study = _stamped_study(make_experiment(provenance={"revision": "toy-v1"}))

    error = _refusal(study, make_experiment(provenance={"revision": "toy-v2"}))

    assert 'Changed: provenance.revision: "toy-v1" -> "toy-v2".' in str(error)
    assert str(error) == str(error).rstrip()
    assert error.remediation == f"Restore the recorded value(s), or {FRESH_STUDY}."
    assert _operational_error_text(error).endswith(error.remediation)
    assert error.action is OperatorAction.FRESH_NAMESPACE


def test_refusal_for_a_configured_cwd_omits_the_unset_cwd_advice() -> None:
    """A configured ``execution.cwd`` that changed is named, without the unset-cwd advice."""
    study = _stamped_study(make_experiment(execution=ExecutionContext(cwd="/srv/a")))

    error = _refusal(study, make_experiment(execution=ExecutionContext(cwd="/srv/b")))

    assert 'Changed: execution.cwd: "/srv/a" -> "/srv/b".' in str(error)
    assert str(error) == str(error).rstrip()
    assert _CWD_SENTENCE not in str(error)
    assert error.action is OperatorAction.FRESH_NAMESPACE


def test_refusal_of_a_digest_its_payload_does_not_explain_names_only_the_hashes() -> None:
    """A recorded digest that differs from its own unchanged payload refuses, naming only the hashes."""
    experiment = make_experiment()
    study = optuna.create_study()
    _stamp_identity(
        study, {**_fresh_identity(experiment, experiment.phases[0], {}), "fingerprint": "0" * 64}
    )
    study.optimize(lambda trial: 0.0, n_trials=1)

    error = _refusal(study, experiment)
    message = str(error)

    fingerprint = _phase_fingerprint(experiment, experiment.phases[0], {})
    assert message == (
        f"Study {study.study_name!r} was created with a different config "
        f"(fingerprint {'0' * 12}... != {fingerprint[:12]}...)."
    )
    assert error.action is OperatorAction.FRESH_NAMESPACE
    assert error.remediation == NEW_STUDY_STEP


def _bound_tree(tmp_path: Path, **recorded: object) -> tuple[Experiment, Path]:
    """Materialize a bound tree, overwrite fields of its binding, and return both."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    binding_path = _artifact_root_binding_path(experiment)
    binding = json.loads(binding_path.read_text())
    binding.update(recorded)
    binding_path.write_text(json.dumps(binding) + "\n")
    return experiment, binding_path


def test_a_tree_at_another_path_is_refused_as_created_elsewhere(tmp_path: Path) -> None:
    """A moved or copied tree names the path it was created at, and the fix moves it back."""
    experiment, _ = _bound_tree(tmp_path, artifact_root="/old/place/runs/t")
    root = str(_experiment_dir(experiment).resolve())

    with pytest.raises(ArtifactRootConflictError) as caught:
        _check_artifact_root_binding(experiment)

    assert str(caught.value) == (
        f"The artifact tree at {root!r} was created at '/old/place/runs/t'."
    )
    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert caught.value.remediation == f"Use the tree at the recorded path, or {FRESH_STUDY}."


def test_a_binding_for_another_experiment_names_both_experiments(tmp_path: Path) -> None:
    """The experiment the tree was created for is shown beside the configured one."""
    experiment, _ = _bound_tree(tmp_path, experiment="renamed")

    with pytest.raises(ArtifactRootConflictError) as caught:
        _check_artifact_root_binding(experiment)

    assert f"was created for experiment 'renamed', not {experiment.experiment!r}." in str(
        caught.value
    )
    assert "storage" not in str(caught.value)
    assert caught.value.remediation is not None
    assert caught.value.remediation == (
        f"Restore the `experiment` this tree was created with, or {FRESH_STUDY}."
    )


def test_a_binding_for_another_ledger_names_the_storage(tmp_path: Path) -> None:
    """A changed ``storage`` is the cause named, not a ledger-or-experiment guess."""
    experiment, _ = _bound_tree(tmp_path, storage_key="0" * 64)

    with pytest.raises(ArtifactRootConflictError) as caught:
        _check_artifact_root_binding(experiment)

    assert "was created with a different storage ledger than this config names." in str(
        caught.value
    )
    assert "experiment" not in str(caught.value)
    assert caught.value.remediation is not None
    assert caught.value.remediation == (
        f"Restore the `storage` this tree was created with, or {FRESH_STUDY}."
    )


def test_a_binding_for_another_experiment_and_ledger_names_both(tmp_path: Path) -> None:
    """Both differing fields are named, in one sentence."""
    experiment, _ = _bound_tree(tmp_path, experiment="renamed", storage_key=None)

    with pytest.raises(ArtifactRootConflictError) as caught:
        _check_artifact_root_binding(experiment)

    assert (
        f"was created for experiment 'renamed', not {experiment.experiment!r}, and with a "
        "different storage ledger than this config names."
    ) in str(caught.value)
    assert caught.value.remediation is not None
    assert caught.value.remediation.startswith(
        "Restore the `experiment` and `storage` this tree was created with"
    )


@pytest.mark.parametrize("shape", ["not-an-object", "extra-field"])
def test_a_binding_of_another_shape_is_malformed_not_a_mismatch(tmp_path: Path, shape: str) -> None:
    """A binding that is no ownership record is restored, never blamed on the config."""
    if shape == "extra-field":
        experiment, _ = _bound_tree(tmp_path, unexpected="field")
    else:
        experiment, binding_path = _bound_tree(tmp_path)
        binding_path.write_text("[1, 2]\n")

    with pytest.raises(ArtifactRootConflictError, match="is malformed") as caught:
        _check_artifact_root_binding(experiment)

    assert caught.value.action is OperatorAction.RESTORE_TREE
    assert caught.value.remediation is None


def test_a_study_bound_to_another_root_gets_a_step_that_fits(tmp_path: Path) -> None:
    """A study that publishes elsewhere is told to restore the root's inputs or start fresh."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(study_name="t::p")
    study.set_user_attr(ARTIFACT_ROOT_ATTR, "/old/place/runs/t")

    with pytest.raises(ArtifactRootConflictError, match="publishes into artifact root") as caught:
        _check_study_artifact_root(study, experiment)

    assert caught.value.remediation == (
        "Restore the `workdir` and `experiment` that give the recorded artifact root, or "
        f"{FRESH_STUDY}."
    )


def test_a_changed_inherited_winner_sends_the_dependent_to_a_new_name_not_a_new_experiment() -> (
    None
):
    """When only an inherited value changed, the fix line gives the dependent and its dependents new names and does not retrain the parent."""
    parent = Phase(
        name="p",
        n_trials=1,
        gpu_policy="none",
        search_space={"lr": IntParam(type="int", low=0, high=3)},
        sampler=Sampler(type="random", seed=0),
    )
    child = Phase(name="q", n_trials=1, gpu_policy="none", inherits=["p"])
    grandchild = Phase(name="r", n_trials=1, gpu_policy="none", inherits=["q"])
    experiment = make_experiment(phases=[parent, child, grandchild])

    def winners(lr: int) -> dict[str, Winner]:
        return {"p": Winner(trial_number=0, params={}, effective_overrides={"lr": lr}, metric=0.0)}

    study = optuna.create_study()
    _verify_fingerprint(study, experiment, child, winners(1))
    study.optimize(lambda trial: 0.0, n_trials=1)

    with pytest.raises(StudyFingerprintMismatchError) as caught:
        _verify_fingerprint(study, experiment, child, winners(2))

    assert "inherited_effective_overrides.p.lr: 1 -> 2" in str(caught.value)
    assert caught.value.remediation == (
        f"A phase it inherits now has a different winner; {rename_step(['q', 'r'])}."
    )


def test_the_rename_step_is_worded_once_and_names_every_phase_it_is_given() -> None:
    """Every site that offers the rename builds it here, and the reference's timeout entry says it with no phase named."""
    assert rename_step(("p",)) == "give `p` a new `name`, which runs it again from scratch"
    assert rename_step(("p", "q", "r")) == (
        "give `p`, `q` and `r` new names and update the `inherits` that name them, which runs "
        "them again from scratch"
    )
    (timeout,) = lookup("phases[].timeout_seconds_per_trial")
    assert (
        "give the phase, and each phase that inherits it, a new `name`, updating every "
        "`inherits` to match, to run it under another value."
    ) in " ".join(timeout.comments)


@pytest.mark.integration
def test_a_raised_target_is_run_by_renaming_only_the_phases_that_inherit_it(
    tmp_path: Path,
) -> None:
    """Following the refusal's step as printed keeps the enlarged phase's finished trials and reruns its dependents."""
    trainer = write_constant_trainer(tmp_path)

    def chain(*, a_trials: int, suffix: str) -> Experiment:
        def phase(name: str, trials: int, parent: str | None = None) -> Phase:
            return Phase(
                name=name,
                n_trials=trials,
                inherits=[] if parent is None else [parent],
                sampler=Sampler(type="random", seed=0),
                gpu_policy="none",
                search_space={name[0]: IntParam(type="int", low=0, high=9)},
            )

        phases = [
            phase("a", a_trials),
            phase(f"b{suffix}", 2, "a"),
            phase(f"c{suffix}", 2, f"b{suffix}"),
        ]
        return make_experiment(persistent=tmp_path, trainer=trainer, phases=phases)

    run_experiment(chain(a_trials=2, suffix=""))
    with pytest.raises(StudyContextConflictError) as refused:
        run_experiment(chain(a_trials=3, suffix=""))
    assert refused.value.remediation is not None
    assert refused.value.remediation == (
        "Give `b` and `c` new names and update the `inherits` that name them, which runs them "
        "again from scratch; `a` keeps its trials and runs the rest. Or run a config without its "
        "dependents first, then put them back: a dependent whose inherited values come out "
        "unchanged keeps its trials."
    )

    renamed = chain(a_trials=3, suffix="2")
    run_experiment(renamed)

    assert [t.state.name for t in phase_trials(renamed, "a")] == ["COMPLETE"] * 3
    assert [t.number for t in phase_trials(renamed, "a")][:2] == [0, 1]
    assert len(phase_trials(renamed, "b2")) == len(phase_trials(renamed, "c2")) == 2


def test_long_values_are_windowed_around_their_first_difference() -> None:
    """Two long values that differ late still show where, and every value stays short."""
    old, new = _render_changed_values("x" * 100 + "a", "x" * 100 + "b")
    assert (old, new) == ("..." + "x" * 24 + 'a"', "..." + "x" * 24 + 'b"')

    old, new = _render_changed_values("x" * 200, "y" * 200)
    assert old == '"' + "x" * 79 + "..."
    assert new == '"' + "y" * 79 + "..."

    assert _render_changed_values({"low": 0}, None) == ('{"low":0}', "null")
