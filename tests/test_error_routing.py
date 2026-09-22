"""Contract tests for the operator action every PhaseSweepError carries.

``action`` is a *routing* attribute: it names the one remediation an operator
should attempt, so a caller can steer a failure without parsing prose. The
message text stays authoritative about what went wrong, and these tests pin
that separation down in both directions -- every operator-facing error resolves
to a real ``OperatorAction``, and attaching one never edits the message.

The class walk below is deliberately exhaustive rather than a hand-maintained
list: it imports every module in the package and then recurses through
``PhaseSweepError.__subclasses__()``, so a subclass added in some far corner of
the tree is covered the moment it exists.
"""

from __future__ import annotations

import importlib
import pickle
import pkgutil

import optuna
import pytest

import phasesweep
from phasesweep.config import IntParam, Phase, WandbSummaryRequiredGate
from phasesweep.engine import fingerprints, study_policy, trial
from phasesweep.errors import OperatorAction, PhaseSweepError
from tests.conftest import make_experiment

# Importing a ``__main__`` module runs the program it guards, so the sweep skips
# those and only those. Every other module is a plain import.
_MAIN_MODULE = "__main__"

# Subclasses whose constructor needs more than a message. Empty today: no
# PhaseSweepError subclass defines its own ``__init__``. Populate it when one
# does, rather than narrowing the walk.
_CONSTRUCTOR_ARGS: dict[str, tuple[object, ...]] = {}

# ``RunRecoveryError`` should declare ``RUN_RECOVER_RUN``, but it lives in
# phasesweep/mcp/recovery.py, which a concurrent storage refactor owns while
# this contract lands. It is the one class permitted to reach the base fallback
# by inheritance; declaring it is the first task of the follow-up commit.
_DECLARATION_PENDING = frozenset({"RunRecoveryError"})

# Classes proving the module sweep reached past the error modules themselves.
# If an import ever stops happening, these vanish from the walk and say so.
_SWEEP_WITNESSES = frozenset(
    {"UnsafeLockPathError", "NoFeasibleTrialError", "RunRecoveryError", "_PolicyStateWriteError"}
)


def _import_every_module() -> None:
    """Import the whole package so no subclass is missing from the walk."""
    for module in pkgutil.walk_packages(phasesweep.__path__, f"{phasesweep.__name__}."):
        if module.name.rsplit(".", 1)[-1] == _MAIN_MODULE:
            continue
        importlib.import_module(module.name)


def _all_subclasses(root: type[BaseException]) -> list[type[PhaseSweepError]]:
    """Return every direct and indirect subclass of ``root``, deepest included."""
    found: list[type[PhaseSweepError]] = []
    for subclass in root.__subclasses__():
        found.append(subclass)
        found.extend(_all_subclasses(subclass))
    return found


def _operator_error_classes() -> list[type[PhaseSweepError]]:
    """Return every PhaseSweepError subclass defined anywhere in the package."""
    _import_every_module()
    return _all_subclasses(PhaseSweepError)


def _populated_study() -> optuna.Study:
    """Return an in-memory study holding one finished trial and no PhaseSweep attrs."""
    study = optuna.create_study(study_name="routing")
    study.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    return study


def test_every_operator_error_declares_its_action():
    classes = _operator_error_classes()
    names = {cls.__name__ for cls in classes}
    assert names >= _SWEEP_WITNESSES, f"module sweep missed: {sorted(_SWEEP_WITNESSES - names)}"

    inherits_fallback = []
    for cls in classes:
        extra = _CONSTRUCTOR_ARGS.get(cls.__name__, ())
        plain = cls("boom", *extra)
        routed = cls("boom", *extra, action=OperatorAction.RETRY)

        assert isinstance(plain.action, OperatorAction), f"{cls.__name__} has no routed action"
        assert plain.action == cls.default_action
        assert routed.action is OperatorAction.RETRY

        # The action is a routing attribute, never part of what the operator reads.
        assert str(routed) == str(plain)
        assert plain.args == routed.args

        # An instance attribute, so BaseException.__reduce__ carries it in __dict__.
        assert pickle.loads(pickle.dumps(plain)).action == plain.action

        if (
            cls.default_action is OperatorAction.INSPECT_LOGS
            and "default_action" not in cls.__dict__
        ):
            inherits_fallback.append(cls.__name__)

    assert set(inherits_fallback) <= _DECLARATION_PENDING, (
        "these subclasses silently inherit the base INSPECT_LOGS fallback instead of "
        f"declaring an action of their own: {sorted(set(inherits_fallback) - _DECLARATION_PENDING)}"
    )


def test_rewrap_preserves_inbound_action_and_explicit_action_replaces():
    from phasesweep.engine.errors import StudyStorageUnavailableError
    from phasesweep.mcp.recovery import RunRecoveryError

    inbound = StudyStorageUnavailableError("x")
    assert inbound.action is OperatorAction.RESTORE_LEDGER

    inherited = RunRecoveryError.rewrap(inbound, "y")
    assert isinstance(inherited, RunRecoveryError)
    assert inherited.action is OperatorAction.RESTORE_LEDGER
    assert str(inherited) == "y"

    overridden = RunRecoveryError.rewrap(inbound, "y", action=OperatorAction.FRESH_NAMESPACE)
    assert overridden.action is OperatorAction.FRESH_NAMESPACE
    assert str(overridden) == "y"

    # A cause with no action of its own leaves the class default in place.
    foreign = RunRecoveryError.rewrap(OSError("disk"), "y")
    assert foreign.action == RunRecoveryError.default_action

    # rewrap returns; the caller still writes the `from` clause that links the cause.
    cause = StudyStorageUnavailableError("x")
    with pytest.raises(RunRecoveryError) as excinfo:
        raise RunRecoveryError.rewrap(cause, "y") from cause
    assert excinfo.value.__cause__ is cause
    assert excinfo.value.action is OperatorAction.RESTORE_LEDGER


@pytest.mark.parametrize(
    ("label", "call"),
    [
        ("study schema", lambda: study_policy._validate_study_schema(_populated_study())),
        ("trial target", lambda: study_policy._accepted_trial_target(_populated_study())),
        (
            "environment cohort",
            lambda: study_policy._validate_environment_cohort(_populated_study(), "d" * 64),
        ),
        (
            "phase fingerprint",
            lambda: fingerprints._verify_fingerprint(
                _populated_study(),
                make_experiment(),
                Phase(
                    name="p", n_trials=2, search_space={"x": IntParam(type="int", low=0, high=10)}
                ),
                {},
            ),
        ),
    ],
)
def test_pre_cutover_refusals_route_to_the_prior_release(label, call):
    with pytest.raises(PhaseSweepError) as excinfo:
        call()
    assert excinfo.value.action is OperatorAction.USE_PRIOR_RELEASE, label
    # The routed action is additional to, not a replacement for, the remedy prose.
    assert "0.3.1" in str(excinfo.value), label


def test_trainer_environment_config_refusal_routes_to_fix_config(monkeypatch):
    experiment = make_experiment(
        gates=[
            WandbSummaryRequiredGate(
                type="wandb_summary_required", entity="e", project="p", keys=["eval/loss"]
            )
        ]
    )
    monkeypatch.setenv("WANDB_MODE", "offline")

    with pytest.raises(PhaseSweepError) as excinfo:
        trial._trainer_environment(experiment, "p")
    assert excinfo.value.action is OperatorAction.FIX_CONFIG
    assert str(excinfo.value) == (
        "W&B evidence requires online logging; remove offline/disabled W&B settings."
    )
