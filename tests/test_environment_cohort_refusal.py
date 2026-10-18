"""A refused environment cohort shows each variable that differs, with both values.

A resume from cron, ssh or tmux composes a different base environment than the shell that started the
study. Each trial records its tracked environment in plain text, so the refusal compares this
invocation with a recorded trial from the other cohort and prints ``TZ`` or ``PATH`` with the old and
the new value. Pass-through names are in neither mapping.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config.models import ExecutionContext, Experiment
from phasesweep.engine.artifacts import _write_trials_csv
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.paths import _phase_dir, _winner_path
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import TRAINER_ENV_ATTR, TRAINER_ENV_DIGEST_ATTR
from phasesweep.engine.study_policy import _validate_environment_cohort
from phasesweep.engine.trial import (
    EnvironmentIdentity,
    TrialExecutionError,
    _environment_identity,
)
from phasesweep.errors import OperatorAction
from phasesweep.evidence.models import WandbSummaryRequiredGate
from phasesweep.tracking.models import MlflowSettings
from tests.conftest import make_experiment, write_constant_trainer

_AMBIENT = {
    "PATH": "/opt/first/bin:/usr/bin",
    "HOME": "/home/alice",
    "LANG": "en_US.UTF-8",
    "USER": "alice",
    "LOGNAME": "alice",
    "TZ": "Europe/Berlin",
}
_NOT_RECORDED = "No trial from the other cohort recorded its environment"
#: The shell's own ``PATH``, which a case that launches a real trainer needs to find ``python``.
_REAL_PATH = os.environ.get("PATH", "")


@pytest.fixture(autouse=True)
def _base_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every base variable so each case starts from the same shell."""
    for name in ("LC_ALL", "TMPDIR", "HF_HOME", "API_TOKEN", "SHLVL", "OLDPWD"):
        monkeypatch.delenv(name, raising=False)
    for name, value in _AMBIENT.items():
        monkeypatch.setenv(name, value)


def _identity(experiment: Experiment) -> EnvironmentIdentity:
    """Compose the identity phase ``p`` would stamp from the current process environment."""
    return _environment_identity(experiment, "p")


def _stamped(identity: EnvironmentIdentity, *, mapping: bool = True) -> dict[str, object]:
    """Return the attributes ``_execute_objective`` stamps on an allocated trial."""
    attrs: dict[str, object] = {TRAINER_ENV_DIGEST_ATTR: identity.digest}
    if mapping:
        attrs[TRAINER_ENV_ATTR] = dict(identity.tracked)
    return attrs


def _record(identity: EnvironmentIdentity, *, mapping: bool = True) -> optuna.Study:
    """Return a study holding one trial stamped as ``_execute_objective`` stamps it."""
    study = optuna.create_study(study_name="t::p")
    study.add_trial(
        optuna.trial.create_trial(value=1.0, user_attrs=_stamped(identity, mapping=mapping))
    )
    return study


def _refusal(study: optuna.Study, current: EnvironmentIdentity) -> str:
    """Return the refusal text for ``current`` against ``study``."""
    with pytest.raises(StudyFingerprintMismatchError) as caught:
        _validate_environment_cohort(study, current)
    assert type(caught.value) is StudyFingerprintMismatchError
    assert caught.value.action is OperatorAction.FIX_CONFIG
    assert "No trial was allocated." in str(caught.value)
    return str(caught.value)


@pytest.mark.parametrize(
    "initial,changes,inherit_env,expected",
    [
        pytest.param(
            {},
            {"TZ": "Asia/Tokyo"},
            [],
            'Compared with trial 0: TZ "Europe/Berlin" -> "Asia/Tokyo". No trial was allocated.',
            id="changed-timezone",
        ),
        pytest.param(
            {},
            {"PATH": "/opt/second/bin:/usr/bin"},
            [],
            'PATH "/opt/first/bin:/usr/bin" -> "/opt/second/bin:/usr/bin".',
            id="changed-path",
        ),
        pytest.param(
            {},
            {"TZ": "Asia/Tokyo", "PATH": "/opt/second/bin:/usr/bin"},
            [],
            'Compared with trial 0: PATH "/opt/first/bin:/usr/bin" -> "/opt/second/bin:/usr/bin"; '
            'TZ "Europe/Berlin" -> "Asia/Tokyo".',
            id="two-changed-variables",
        ),
        pytest.param(
            {},
            {"HF_HOME": "/data/hf-cache"},
            ["HF_HOME"],
            'Compared with trial 0: added HF_HOME="/data/hf-cache".',
            id="added-inherited-name",
        ),
        pytest.param(
            {"HF_HOME": "/data/hf-cache"},
            {"HF_HOME": None},
            ["HF_HOME"],
            'Compared with trial 0: removed HF_HOME (was "/data/hf-cache").',
            id="removed-inherited-name",
        ),
        pytest.param(
            {"XDG_CACHE_HOME": "/data/cache"},
            {"XDG_CACHE_HOME": None, "HF_HOME": "/data/hf-cache", "TZ": "Asia/Tokyo"},
            ["HF_HOME", "XDG_CACHE_HOME"],
            'Compared with trial 0: TZ "Europe/Berlin" -> "Asia/Tokyo"; added HF_HOME="/data/hf-cache"; '
            'removed XDG_CACHE_HOME (was "/data/cache").',
            id="changed-added-removed",
        ),
    ],
)
def test_environment_mutation_names_each_difference(
    monkeypatch: pytest.MonkeyPatch,
    initial: dict[str, str],
    changes: dict[str, str | None],
    inherit_env: list[str],
    expected: str,
) -> None:
    experiment = (
        make_experiment(execution=ExecutionContext(inherit_env=inherit_env))
        if inherit_env
        else make_experiment()
    )
    for name, value in initial.items():
        monkeypatch.setenv(name, value)
    study = _record(_identity(experiment))
    for name, value in changes.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)

    assert expected in _refusal(study, _identity(experiment))


def test_pass_through_names_are_in_neither_the_mapping_nor_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass-through value cannot split a cohort, so it is not recorded and not offered as the cause."""
    experiment = make_experiment(execution=ExecutionContext(passthrough_env=["API_TOKEN"]))
    monkeypatch.setenv("API_TOKEN", "rotating-first")
    monkeypatch.setenv("TMPDIR", "/scratch/first")
    before = _identity(experiment)
    study = _record(before)
    assert not {"API_TOKEN", "TMPDIR"} & set(before.tracked)
    assert before.values["API_TOKEN"] == "rotating-first"
    monkeypatch.setenv("API_TOKEN", "rotating-second")
    monkeypatch.setenv("TMPDIR", "/scratch/second")
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    message = _refusal(study, _identity(experiment))

    assert 'TZ "Europe/Berlin" -> "Asia/Tokyo".' in message
    for forbidden in ("API_TOKEN", "TMPDIR", "rotating-", "/scratch/"):
        assert forbidden not in message
    recorded = study.get_trials(deepcopy=False)[0].user_attrs[TRAINER_ENV_ATTR]
    assert "rotating-first" not in str(recorded)


@pytest.mark.parametrize("source", ["env", "inherit_env"])
def test_changed_listed_or_configured_value_is_shown(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A credential held in ``env`` or an ``inherit_env`` list is recorded and shown: pass-through is the home for one."""
    if source == "env":
        first = make_experiment(env={"API_TOKEN": "hunter2-first"})
        second = make_experiment(env={"API_TOKEN": "hunter2-second"})
    else:
        first = second = make_experiment(execution=ExecutionContext(inherit_env=["API_TOKEN"]))
    monkeypatch.setenv("API_TOKEN", "hunter2-first")
    study = _record(_identity(first))
    monkeypatch.setenv("API_TOKEN", "hunter2-second")

    message = _refusal(study, _identity(second))

    assert 'API_TOKEN "hunter2-first" -> "hunter2-second".' in message


def test_the_tracked_mapping_holds_what_the_digest_covers(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TMPDIR`` and ``passthrough_env`` names stay out of it, unless ``env`` sets them."""
    monkeypatch.setenv("TMPDIR", "/scratch/sandbox")
    default = _identity(make_experiment())
    assert default.tracked == _AMBIENT
    assert "TMPDIR" not in default.tracked
    assert default.values["TMPDIR"] == "/scratch/sandbox"

    passthrough = _identity(make_experiment(execution=ExecutionContext(passthrough_env=["PATH"])))
    assert "PATH" not in passthrough.tracked
    assert {"HOME", "TZ"} <= set(passthrough.tracked)

    pinned = _identity(make_experiment(env={"TMPDIR": "/pinned", "TZ": "Asia/Tokyo"}))
    assert pinned.tracked["TMPDIR"] == "/pinned"
    assert pinned.tracked["TZ"] == "Asia/Tokyo"


def test_base_value_pinned_under_env_is_shown_like_an_ambient_one() -> None:
    first = make_experiment(env={"TZ": "Europe/Berlin"})
    second = make_experiment(env={"TZ": "Asia/Tokyo"})

    message = _refusal(_record(_identity(first)), _identity(second))

    assert 'Compared with trial 0: TZ "Europe/Berlin" -> "Asia/Tokyo".' in message


def test_a_trial_that_recorded_no_mapping_gets_the_generic_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allocation that died after its digest and before its mapping landed has nothing to compare."""
    experiment = make_experiment()
    study = _record(_identity(experiment), mapping=False)
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    message = _refusal(study, _identity(experiment))

    assert _NOT_RECORDED in message
    assert "Compared with" not in message


def test_a_malformed_mapping_gets_the_generic_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment()
    identity = _identity(experiment)
    study = optuna.create_study(study_name="t::p")
    study.add_trial(
        optuna.trial.create_trial(
            value=1.0,
            user_attrs={TRAINER_ENV_DIGEST_ATTR: identity.digest, TRAINER_ENV_ATTR: ["TZ", "UTC"]},
        )
    )
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    assert _NOT_RECORDED in _refusal(study, _identity(experiment))


def test_the_latest_trial_that_recorded_a_mapping_is_the_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment()
    first = _identity(experiment)
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    second = _identity(experiment)
    study = _record(first)
    study.add_trial(optuna.trial.create_trial(value=1.0, user_attrs=_stamped(second)))
    study.add_trial(
        optuna.trial.create_trial(value=1.0, user_attrs=_stamped(second, mapping=False))
    )
    monkeypatch.setenv("PATH", "/opt/second/bin:/usr/bin")

    message = _refusal(study, _identity(experiment))

    assert (
        'Compared with trial 1: PATH "/opt/first/bin:/usr/bin" -> "/opt/second/bin:/usr/bin".'
    ) in message


def test_a_long_difference_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [f"PS_COHORT_VAR_{index:02d}" for index in range(12)]
    experiment = make_experiment(execution=ExecutionContext(inherit_env=names))
    study = _record(_identity(experiment))
    for name in names:
        monkeypatch.setenv(name, "set")

    message = _refusal(study, _identity(experiment))

    assert 'added PS_COHORT_VAR_09="set"; and 2 more.' in message
    assert "PS_COHORT_VAR_10" not in message


def test_a_long_value_is_cut_to_where_the_two_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = ":".join(f"/opt/tool-{index:02d}/bin" for index in range(30))
    monkeypatch.setenv("PATH", f"{shared}:/opt/a/bin:/usr/bin")
    experiment = make_experiment()
    study = _record(_identity(experiment))
    monkeypatch.setenv("PATH", f"{shared}:/opt/b/bin:/usr/bin")

    message = _refusal(study, _identity(experiment))

    assert "PATH ..." in message
    assert "/opt/a/bin" in message
    assert "/opt/b/bin" in message
    assert "/opt/tool-00/bin" not in message


def test_the_message_keeps_environment_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment()
    study = _record(_identity(experiment))
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    message = _refusal(study, _identity(experiment))

    assert "Values under env are pinned" in message
    assert "execution.passthrough_env" in message
    assert "do not split cohorts" in message


def test_matching_environment_is_not_refused() -> None:
    experiment = make_experiment()
    _validate_environment_cohort(_record(_identity(experiment)), _identity(experiment))


def test_inherit_all_does_not_split_the_cohort_on_an_untracked_ambient_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shell state outside the base names is inherited untracked, so two shells share one cohort."""
    experiment = make_experiment(execution=ExecutionContext(inherit_env="all"))
    monkeypatch.setenv("SHLVL", "1")
    monkeypatch.setenv("OLDPWD", "/home/alice/earlier")
    before = _identity(experiment)
    study = _record(before)
    monkeypatch.setenv("SHLVL", "3")
    monkeypatch.setenv("OLDPWD", "/home/alice/later")
    monkeypatch.setenv("PS_COHORT_EXTRA", "only-in-the-second-shell")

    after = _identity(experiment)

    assert after.digest == before.digest
    assert after.tracked == before.tracked
    assert after.values["SHLVL"] == "3"
    assert after.values["PS_COHORT_EXTRA"] == "only-in-the-second-shell"
    assert not {"SHLVL", "OLDPWD", "PS_COHORT_EXTRA"} & set(after.tracked)
    _validate_environment_cohort(study, after)


def test_inherit_all_still_refuses_a_differing_base_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment(execution=ExecutionContext(inherit_env="all"))
    study = _record(_identity(experiment))
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    monkeypatch.setenv("SHLVL", "9")

    message = _refusal(study, _identity(experiment))

    assert 'Compared with trial 0: TZ "Europe/Berlin" -> "Asia/Tokyo".' in message
    assert "SHLVL" not in message


def test_inherit_all_has_the_cohort_of_inherit_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under ``all`` the tracked mapping and digest are the ``none`` ones, composed keys included."""
    monkeypatch.setenv("HF_HOME", "/data/hf-cache")
    monkeypatch.setenv("SHLVL", "2")
    monkeypatch.setenv("WANDB_MODE", "online")
    base = make_experiment(env={"CONFIGURED": "1"})
    remote = base.phases[0].model_copy(
        update={
            "gates": [
                WandbSummaryRequiredGate(
                    type="wandb_summary_required",
                    entity="entity",
                    project="project",
                    keys=["eval/loss"],
                )
            ]
        }
    )
    narrowed = base.model_copy(
        update={
            "phases": [remote],
            "mlflow": MlflowSettings(tracking_uri="http://mlflow.test:5000", experiment_name="e"),
        }
    )
    everything = narrowed.model_copy(update={"execution": ExecutionContext(inherit_env="all")})

    none_identity = _identity(narrowed)
    all_identity = _identity(everything)

    assert all_identity.digest == none_identity.digest
    assert all_identity.tracked == none_identity.tracked
    assert {"CONFIGURED", "WANDB_ENTITY", "WANDB_PROJECT", "MLFLOW_TRACKING_URI"} <= set(
        all_identity.tracked
    )
    assert {"HF_HOME", "SHLVL", "WANDB_MODE"} <= set(all_identity.values)
    assert not {"HF_HOME", "SHLVL", "WANDB_MODE"} & set(all_identity.tracked)


def test_trials_csv_carries_the_digest_but_not_the_mapping(tmp_path: Path) -> None:
    """The mapping is ledger provenance, so ``PATH`` is not a column of the results table."""
    identity = _identity(make_experiment())
    study = _record(identity)
    destination = tmp_path / "trials.csv"

    _write_trials_csv(study, destination)

    header, row = destination.read_text(encoding="utf-8").splitlines()
    columns = header.split(",")
    assert f"user_attr:{TRAINER_ENV_DIGEST_ATTR}" in columns
    assert f"user_attr:{TRAINER_ENV_ATTR}" not in columns
    assert identity.digest in row
    for value in identity.tracked.values():
        assert value not in row


@pytest.mark.integration
def test_winner_and_trial_attrs_record_trainer_environment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every trial records the environment that produced it in plain text; the winner keeps the digest.

    The ledger holds the tracked name-value pairs, pass-through values excluded. No other
    artifact carries the mapping, so ``trials.csv`` and ``winner.yaml`` show only the digest.
    """
    monkeypatch.setenv("PATH", _REAL_PATH)
    monkeypatch.setenv("PHASESWEEP_TEST_LISTED", "ambient-value")
    monkeypatch.setenv("PHASESWEEP_TEST_PASSED", "rotating-value")
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        env={"PHASESWEEP_TEST_CONFIGURED": "config-value"},
        execution=ExecutionContext(
            inherit_env=["PHASESWEEP_TEST_LISTED"], passthrough_env=["PHASESWEEP_TEST_PASSED"]
        ),
        n_trials=1,
    )

    run_experiment(experiment)

    identity = _identity(experiment)
    winner_text = _winner_path(experiment, "p").read_text()
    winner = yaml.safe_load(winner_text)
    assert winner["trainer_env_digest"] == identity.digest
    assert "trainer_inherit_env" not in winner

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(experiment.storage))
    attrs = study.get_trials(deepcopy=False)[0].user_attrs
    assert attrs[TRAINER_ENV_DIGEST_ATTR] == identity.digest
    recorded = attrs[TRAINER_ENV_ATTR]
    assert recorded == identity.tracked
    assert recorded["PHASESWEEP_TEST_LISTED"] == "ambient-value"
    assert recorded["PHASESWEEP_TEST_CONFIGURED"] == "config-value"
    assert "PHASESWEEP_TEST_PASSED" not in recorded
    assert "rotating-value" not in json.dumps(attrs, default=str)

    trials_csv = (_phase_dir(experiment, "p") / "trials.csv").read_text()
    columns = trials_csv.splitlines()[0].split(",")
    assert f"user_attr:{TRAINER_ENV_DIGEST_ATTR}" in columns
    assert f"user_attr:{TRAINER_ENV_ATTR}" not in columns
    assert "ambient-value" not in trials_csv
    assert "ambient-value" not in winner_text


def test_a_real_run_records_the_mapping_and_a_changed_tz_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The objective stamps the attribute the refusal reads, so a resume under another ``TZ`` shows it."""

    def failed_launch(**kwargs: object) -> None:
        raise TrialExecutionError("simulated trainer failure; no process launched")

    monkeypatch.setattr("phasesweep.engine.phase.launch_trial", failed_launch)
    experiment = make_experiment(
        persistent=tmp_path,
        n_trials=1,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        max_consecutive_failures=5,
    )
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    study = optuna.load_study(study_name="t::p", storage=_resolve_storage(experiment.storage))
    attrs = study.get_trials(deepcopy=False)[0].user_attrs
    assert attrs[TRAINER_ENV_ATTR] == _identity(experiment).tracked
    assert attrs[TRAINER_ENV_ATTR]["TZ"] == "Europe/Berlin"

    monkeypatch.setenv("TZ", "Asia/Tokyo")
    topped_up = make_experiment(
        persistent=tmp_path,
        n_trials=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        max_consecutive_failures=5,
    )
    with pytest.raises(StudyFingerprintMismatchError) as caught:
        run_experiment(topped_up)
    assert 'Compared with trial 0: TZ "Europe/Berlin" -> "Asia/Tokyo".' in str(caught.value)
