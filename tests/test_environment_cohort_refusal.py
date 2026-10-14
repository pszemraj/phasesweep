"""A refused environment cohort names the variables that differ and never prints a value.

A resume from cron, ssh or tmux composes a different base environment than the shell that started the
study. The refusal compares this invocation with a recorded trial from the other cohort through the
base-variable digests each trial carries, so it can name ``TZ`` or ``PATH`` without a value in sight.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import ExecutionContext, Experiment
from phasesweep.config.models import BASE_INHERITED_ENV
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    TRAINER_ENV_BASE_DIGESTS_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
)
from phasesweep.engine.study_policy import _validate_environment_cohort
from phasesweep.engine.trial import (
    EnvironmentIdentity,
    TrialExecutionError,
    _environment_identity,
)
from tests.conftest import make_experiment

_AMBIENT = {
    "PATH": "/opt/first/bin:/usr/bin",
    "HOME": "/home/alice",
    "LANG": "en_US.UTF-8",
    "USER": "alice",
    "LOGNAME": "alice",
    "TZ": "Europe/Berlin",
}
_GENERIC = "one of the values from env or inherit_env differs"


@pytest.fixture(autouse=True)
def _base_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every base variable so each case starts from the same shell."""
    for name in ("LC_ALL", "TMPDIR", "HF_HOME", "API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in _AMBIENT.items():
        monkeypatch.setenv(name, value)


def _identity(experiment: Experiment) -> EnvironmentIdentity:
    """Compose the identity phase ``p`` would stamp from the current process environment."""
    return _environment_identity(experiment, "p")


def _record(identity: EnvironmentIdentity, *, with_digests: bool = True) -> optuna.Study:
    """Return a study holding one trial stamped as ``_execute_objective`` stamps it."""
    attrs: dict[str, object] = {
        TRAINER_ENV_DIGEST_ATTR: identity.digest,
        TRAINER_ENV_NAMES_ATTR: list(identity.names),
    }
    if with_digests:
        attrs[TRAINER_ENV_BASE_DIGESTS_ATTR] = dict(identity.base_digests)
    study = optuna.create_study(study_name="t::p")
    study.add_trial(optuna.trial.create_trial(value=1.0, user_attrs=attrs))
    return study


def _refusal(study: optuna.Study, current: EnvironmentIdentity) -> str:
    """Return the refusal text for ``current`` against ``study``."""
    with pytest.raises(StudyFingerprintMismatchError) as caught:
        _validate_environment_cohort(study, current)
    return str(caught.value)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_changed_tz_names_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment()
    study = _record(_identity(experiment))
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    message = _refusal(study, _identity(experiment))

    assert "base variables with a different value: TZ;" in message
    assert "names added: none; names removed: none." in message


def test_changed_path_names_path(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment()
    study = _record(_identity(experiment))
    monkeypatch.setenv("PATH", "/opt/second/bin:/usr/bin")

    message = _refusal(study, _identity(experiment))

    assert "base variables with a different value: PATH;" in message


def test_several_changed_base_variables_are_listed_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment()
    study = _record(_identity(experiment))
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    monkeypatch.setenv("PATH", "/opt/second/bin:/usr/bin")

    assert "base variables with a different value: PATH, TZ;" in _refusal(
        study, _identity(experiment)
    )


def test_added_inherited_name_is_named_as_added(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment(execution=ExecutionContext(inherit_env=["HF_HOME"]))
    study = _record(_identity(experiment))
    monkeypatch.setenv("HF_HOME", "/data/hf-cache")

    message = _refusal(study, _identity(experiment))

    assert "base variables with a different value: none;" in message
    assert "names added: HF_HOME; names removed: none." in message
    assert "/data/hf-cache" not in message


def test_removed_inherited_name_is_named_as_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment(execution=ExecutionContext(inherit_env=["HF_HOME"]))
    monkeypatch.setenv("HF_HOME", "/data/hf-cache")
    study = _record(_identity(experiment))
    monkeypatch.delenv("HF_HOME")

    message = _refusal(study, _identity(experiment))

    assert "names added: none; names removed: HF_HOME." in message


def test_pass_through_name_present_in_one_shell_is_not_blamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass-through value cannot split a cohort, so its name is not offered as the cause."""
    experiment = make_experiment(execution=ExecutionContext(passthrough_env=["API_TOKEN"]))
    monkeypatch.setenv("API_TOKEN", "rotating")
    study = _record(_identity(experiment))
    monkeypatch.delenv("API_TOKEN")
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    message = _refusal(study, _identity(experiment))

    assert "names removed: none." in message
    assert "API_TOKEN" not in message


@pytest.mark.parametrize("source", ["env", "inherit_env"])
def test_changed_secret_value_gives_the_generic_clause_and_never_the_value(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    if source == "env":
        first = make_experiment(env={"API_TOKEN": "hunter2-first"})
        second = make_experiment(env={"API_TOKEN": "hunter2-second"})
    else:
        first = second = make_experiment(execution=ExecutionContext(inherit_env=["API_TOKEN"]))
    monkeypatch.setenv("API_TOKEN", "hunter2-first")
    before = _identity(first)
    study = _record(before)
    monkeypatch.setenv("API_TOKEN", "hunter2-second")
    after = _identity(second)
    assert after.digest != before.digest

    message = _refusal(study, after)

    assert _GENERIC in message
    for forbidden in ("hunter2", _sha256("hunter2-first"), _sha256("hunter2-second")):
        assert forbidden not in message


def test_no_non_base_name_is_ever_digested(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = make_experiment(
        env={"API_TOKEN": "hunter2"}, execution=ExecutionContext(inherit_env=["HF_HOME"])
    )
    monkeypatch.setenv("HF_HOME", "/data/hf-cache")

    identity = _identity(experiment)

    assert identity.base_digests
    assert set(identity.base_digests) <= set(BASE_INHERITED_ENV)
    recorded = json.dumps(identity.base_digests)
    for secret in ("hunter2", "/data/hf-cache"):
        assert _sha256(secret) not in recorded


def test_base_digests_hold_what_the_main_digest_covers(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TMPDIR`` and ``passthrough_env`` names stay out of both, unless ``env`` sets them."""
    monkeypatch.setenv("TMPDIR", "/scratch/sandbox")
    default = _identity(make_experiment())
    assert "TMPDIR" not in default.base_digests
    assert default.base_digests["TZ"] == _sha256("Europe/Berlin")

    passthrough = _identity(make_experiment(execution=ExecutionContext(passthrough_env=["PATH"])))
    assert "PATH" not in passthrough.base_digests
    assert {"HOME", "TZ"} <= set(passthrough.base_digests)

    pinned = _identity(make_experiment(env={"TMPDIR": "/pinned", "TZ": "Asia/Tokyo"}))
    assert pinned.base_digests["TMPDIR"] == _sha256("/pinned")
    assert pinned.base_digests["TZ"] == _sha256("Asia/Tokyo")


def test_base_value_pinned_under_env_is_named_like_an_ambient_one() -> None:
    first = make_experiment(env={"TZ": "Europe/Berlin"})
    second = make_experiment(env={"TZ": "Asia/Tokyo"})

    message = _refusal(_record(_identity(first)), _identity(second))

    assert "base variables with a different value: TZ;" in message
    assert "Asia/Tokyo" not in message


def test_a_trial_without_the_digests_gets_the_generic_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment()
    study = _record(_identity(experiment), with_digests=False)
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    message = _refusal(study, _identity(experiment))

    assert _GENERIC in message
    assert "Compared with" not in message


def test_the_latest_trial_that_recorded_its_digests_is_the_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment()
    first = _identity(experiment)
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    second = _identity(experiment)
    study = _record(first)
    study.add_trial(
        optuna.trial.create_trial(
            value=1.0,
            user_attrs={
                TRAINER_ENV_DIGEST_ATTR: second.digest,
                TRAINER_ENV_NAMES_ATTR: list(second.names),
                TRAINER_ENV_BASE_DIGESTS_ATTR: dict(second.base_digests),
            },
        )
    )
    study.add_trial(
        optuna.trial.create_trial(value=1.0, user_attrs={TRAINER_ENV_DIGEST_ATTR: second.digest})
    )
    monkeypatch.setenv("PATH", "/opt/second/bin:/usr/bin")

    message = _refusal(study, _identity(experiment))

    assert "Compared with trial 1, base variables with a different value: PATH;" in message


def test_a_long_difference_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [f"PS_COHORT_VAR_{index:02d}" for index in range(12)]
    experiment = make_experiment(execution=ExecutionContext(inherit_env=names))
    study = _record(_identity(experiment))
    for name in names:
        monkeypatch.setenv(name, "set")

    message = _refusal(study, _identity(experiment))

    assert "PS_COHORT_VAR_07 and 4 more;" in message
    assert "PS_COHORT_VAR_08" not in message


def test_the_message_keeps_the_remedies_and_never_prints_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = make_experiment()
    before = _identity(experiment)
    study = _record(before)
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    monkeypatch.setenv("PATH", "/opt/second/bin:/usr/bin")
    after = _identity(experiment)

    message = _refusal(study, after)

    for remedy in (
        "Restore the original semantic environment",
        "pin the value under env",
        "execution.passthrough_env",
        "new experiment name",
    ):
        assert remedy in message
    assert "a different PATH counts" not in message
    values = [*_AMBIENT.values(), "Asia/Tokyo", "/opt/second/bin:/usr/bin"]
    digests = [*before.base_digests.values(), *after.base_digests.values()]
    for forbidden in (*values, *digests, before.digest, after.digest):
        assert forbidden not in message


def test_matching_environment_is_not_refused() -> None:
    experiment = make_experiment()
    _validate_environment_cohort(_record(_identity(experiment)), _identity(experiment))


def test_a_real_run_records_the_digests_and_a_changed_tz_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The objective stamps the attribute the refusal reads, so a resume under another ``TZ`` names it."""

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
    assert attrs[TRAINER_ENV_BASE_DIGESTS_ATTR] == _identity(experiment).base_digests

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
    assert "Compared with trial 0, base variables with a different value: TZ;" in str(caught.value)
    assert "Asia/Tokyo" not in str(caught.value)
