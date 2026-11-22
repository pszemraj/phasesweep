"""Contract tests for the operator action every PhaseSweepError carries.

``action`` routes a failure to the remediation an operator should carry out, without parsing
prose; the message stays authoritative about what went wrong, and routing never edits it.
"""

from __future__ import annotations

import errno
import importlib
import json
import os
import pkgutil
import pwd
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import optuna
import pytest

import phasesweep
import phasesweep.engine.ledger as engine_ledger
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import ExecutionContext, Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine import fingerprints, guards, study_policy, trial
from phasesweep.engine.artifact_roots import _write_artifact_root_binding
from phasesweep.engine.artifacts import _load_winner
from phasesweep.engine.attempts import (
    ATTEMPT_REGISTRY_SCHEMA_VERSION,
    _preflight_active_attempts,
    _PreflightCleanupReport,
)
from phasesweep.engine.cleanup import _reap_stale_trials, _recover_cleanup_uncertain_trials
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    LaunchEnvironmentError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrackingTargetUnavailableError,
    TrialTargetRefusedError,
    WinnerIntegrityError,
)
from phasesweep.engine.generation import InvocationGeneration
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _attempts_dir,
)
from phasesweep.engine.phase import _abort_record, _failure_policy_abort_record, _finish_phase
from phasesweep.engine.resume import _published_for_resume
from phasesweep.engine.run import run_experiment
from phasesweep.engine.selection import (
    NoFeasibleTrialError,
    PhaseAbortRefusedError,
    PhaseTrialsFailedError,
    PhaseTrialsInfeasibleError,
    ReplayWithoutFeasibleTrialError,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.trial import EnvironmentIdentity
from phasesweep.errors import (
    ACCEPT_INCOMPLETE_STEP,
    OTHER_FORMAT_STEP,
    ForeignPidNamespaceError,
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
)
from phasesweep.evidence.evaluation import EvidenceUnreadableError
from phasesweep.evidence.models import LogRegexExtractor, WandbSummaryRequiredGate
from phasesweep.evidence.wandb import require_wandb_sdk
from phasesweep.mcp import runner as mcp_runner
from phasesweep.runtime import files as runtime_files
from phasesweep.runtime.files import (
    UnsafeLockPathError,
    UnsafePrivatePathError,
    lock_dir,
    phasesweep_home,
)
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    cleanup_stale_trial_process,
)
from phasesweep.tracking.mlflow import require_mlflow_sdk
from tests.conftest import (
    make_experiment,
    patch_published_trial_read_failure,
    phase_study,
    record_accepted_partial_timeout,
    requires_nonroot,
    trial_outcome,
    write_trainer,
)
from tests.error_routing_helpers import (
    RoutingCase,
    Trigger,
    assert_payload_follows,
    assert_routes_declared_action,
    assert_runner_payload_follows_routing,
    garbage,
    materialize_damaged,
    raiser,
    read_only_once_locked,
    registered_entry,
    truncated,
)
from tests.ledger_fixtures import (
    materialize,
    pin_fixture_environment,
    republish_as_incomplete,
)

pytestmark = pytest.mark.contract

_SWEEP_WITNESSES = frozenset({"UnsafeLockPathError", "NoFeasibleTrialError", "RunRecoveryError"})


def _import_every_module() -> None:
    """Import the whole package so no subclass is missing from the walk."""
    for module in pkgutil.walk_packages(phasesweep.__path__, f"{phasesweep.__name__}."):
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


def _populated_study(name: str = "routing") -> optuna.Study:
    """Return an in-memory study holding one finished trial and no PhaseSweep attrs."""
    study = optuna.create_study(study_name=name)
    study.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    return study


def _digest_only_identity(digest: str) -> EnvironmentIdentity:
    """Return an environment identity that carries a digest and nothing else."""
    return EnvironmentIdentity(digest=digest, tracked={}, values={})


def test_every_operator_error_declares_its_action():
    classes = _operator_error_classes()
    names = {cls.__name__ for cls in classes}
    assert names >= _SWEEP_WITNESSES, f"module sweep missed: {sorted(_SWEEP_WITNESSES - names)}"

    inherits_fallback = []
    for cls in classes:
        plain = cls("boom")
        routed = cls("boom", action=OperatorAction.RETRY)

        assert isinstance(cls.default_action, OperatorAction), f"{cls.__name__} has no action"
        assert plain.action is cls.default_action
        assert routed.action is OperatorAction.RETRY

        # Boundaries render actions; the exception itself retains its diagnostic.
        assert str(routed) == str(plain)
        assert plain.args == routed.args

        if (
            cls.default_action is OperatorAction.INSPECT_LOGS
            and "default_action" not in cls.__dict__
        ):
            inherits_fallback.append(cls.__name__)

    # Each subclass declares its route rather than inheriting the base fallback.
    assert inherits_fallback == []


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
    assert foreign.action is RunRecoveryError.default_action


def test_no_feasible_subclasses_rewrap_with_the_donated_action():
    inbound = PhaseTrialsFailedError("x", action=OperatorAction.FIX_CONFIG)

    rewrapped = ReplayWithoutFeasibleTrialError.rewrap(inbound, "y")

    assert type(rewrapped) is ReplayWithoutFeasibleTrialError
    assert isinstance(rewrapped, NoFeasibleTrialError)
    assert rewrapped.action is OperatorAction.FIX_CONFIG
    assert str(rewrapped) == "y"


_UNMARKED_STUDY_DIAGNOSTIC_MARKERS = {
    "study schema": "has no format version; this PhaseSweep reads format",
    "trial target": "has trial state but no",
    "environment cohort": "without a semantic trainer-environment identity",
    "phase fingerprint": "has no semantic fingerprint",
}


@pytest.mark.parametrize(
    ("label", "call"),
    [
        ("study schema", lambda: study_policy._validate_study_schema(_populated_study())),
        ("trial target", lambda: study_policy._accepted_trial_target(_populated_study())),
        (
            "environment cohort",
            lambda: study_policy._validate_environment_cohort(
                _populated_study(), _digest_only_identity("d" * 64)
            ),
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
def test_unmarked_study_refusals_route_to_fresh_namespace(label, call):
    """A populated study with no PhaseSweep attrs is unsupported state, not just a config mismatch."""
    with pytest.raises(PhaseSweepError) as excinfo:
        call()
    assert excinfo.value.action is OperatorAction.FRESH_NAMESPACE, label
    assert _UNMARKED_STUDY_DIAGNOSTIC_MARKERS[label] in str(excinfo.value), label


def test_live_study_schema_refusal_gives_the_ledger_step():
    """A study in another format sits in a ledger, so its fix line is the one the ledger scan gives."""
    with pytest.raises(StudySchemaMismatchError) as excinfo:
        study_policy._validate_study_schema(_populated_study())

    assert excinfo.value.remediation == OTHER_FORMAT_STEP


def _collected(*errors: PhaseSweepError) -> PhaseSweepError:
    """Return the aggregate ``_raise_collected_errors`` raises for refusals with differing messages."""
    with pytest.raises(PhaseSweepError) as excinfo:
        guards._raise_collected_errors(list(errors))
    return excinfo.value


def test_collected_refusals_carry_the_remediation_they_share():
    """Studies refused for different messages but one step print that step once."""
    step = "Set phases[p].n_trials to at least 2 and rerun."
    aggregate = _collected(
        TrialTargetRefusedError("phase a", remediation=step),
        TrialTargetRefusedError("phase b", remediation=step),
    )

    assert aggregate.remediation == step
    assert aggregate.action is OperatorAction.FIX_CONFIG
    assert _operational_error_text(aggregate).endswith(f"\nfix: {step}")


def test_collected_refusals_list_each_distinct_remediation_once_in_order():
    """Differing steps are all kept, in the order their refusals were collected."""
    aggregate = _collected(
        TrialTargetRefusedError("phase a", remediation="Raise a."),
        LaunchEnvironmentError("phase b", remediation="Create b."),
        TrialTargetRefusedError("phase c", remediation="Raise a."),
    )

    assert aggregate.remediation == "Raise a. Create b."
    assert type(aggregate) is PhaseSweepError


def test_collected_refusal_without_a_remediation_contributes_its_action_step():
    """A refusal that names no step of its own is not dropped from a fix line that names others."""
    aggregate = _collected(
        TrialTargetRefusedError("phase a", remediation="Raise a."),
        TrialTargetRefusedError("phase b"),
    )

    assert aggregate.remediation == (
        "Raise a. Correct the experiment configuration or the environment it runs in."
    )


def test_collected_refusals_without_any_remediation_keep_the_action_step():
    """With no step of their own, the aggregate's action alone speaks for them."""
    aggregate = _collected(TrialTargetRefusedError("phase a"), TrialTargetRefusedError("phase b"))

    assert aggregate.remediation is None
    assert _operational_error_text(aggregate).endswith(
        "\nfix: correct the experiment configuration or the environment it runs in."
    )


def test_trainer_environment_config_refusal_routes_to_fix_config(monkeypatch):
    """A trainer environment that cannot be composed is a launch-environment refusal."""
    experiment = make_experiment(
        execution=ExecutionContext(inherit_env=["WANDB_MODE"]),
        gates=[
            WandbSummaryRequiredGate(
                type="wandb_summary_required", entity="e", project="p", keys=["eval/loss"]
            )
        ],
    )
    monkeypatch.setenv("WANDB_MODE", "offline")

    with pytest.raises(LaunchEnvironmentError) as excinfo:
        trial._environment_identity(experiment, "p")
    assert excinfo.value.action is OperatorAction.FIX_CONFIG
    assert str(excinfo.value) == (
        "W&B evidence requires online logging; remove offline/disabled W&B settings."
    )
    failure = mcp_runner._terminal_failure_payload(
        excinfo.value, run_id="r0", stage="preflight", cleanup_confirmed=True
    )
    assert (failure["code"], failure["stage"], failure["actor"]) == (
        "launch_environment_invalid",
        "preflight",
        "operator",
    )


def _storage_gone() -> Exception:
    """Return the backend failure a vanished ledger surfaces as."""
    return RuntimeError("storage went away")


def _ledger_busy() -> Exception:
    """Return a storage refusal whose own raise site routed it to a retry."""
    return StudyStorageUnavailableError(
        "Journal storage is locked by another writer.", action=OperatorAction.RETRY
    )


def _claim_while(owner: object, name: str, inbound: Callable[[], Exception]) -> Trigger:
    """Claim the current golden ledger while ``owner.name`` raises."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize("current-journal", tmp_path, mode="tree")
        monkeypatch.setattr(owner, name, raiser(inbound()))
        return engine_ledger.claim_ledger(engine_ledger.validate_ledger(materialized.experiment))

    return trigger


def _claim_while_published_trials_fail(inbound: Callable[[], Exception]) -> Trigger:
    """Claim the current golden ledger while the published check's trial reads raise.

    Opening a study reads its trials too, so the failure is confined to the
    published check rather than raised from every ``get_trials`` call.
    """

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize("current-journal", tmp_path, mode="tree")
        patch_published_trial_read_failure(monkeypatch, inbound())
        return engine_ledger.claim_ledger(engine_ledger.validate_ledger(materialized.experiment))

    return trigger


def _validate_damaged(fixture: str, damage: Callable[[Path], None]) -> Trigger:
    """Validate an unbound current golden ledger after damaging its file."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize_damaged(tmp_path, fixture, "ledger-only", damage)
        return engine_ledger.validate_ledger(materialized.experiment)

    return trigger


def _run_while_discovery_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Start a run over the current golden ledger whose phase study cannot be read."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    monkeypatch.setattr(engine_ledger, "open_existing_study", raiser(_storage_gone()))
    return run_experiment(materialized.experiment)


def _run_over_unbackuppable_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Start a run whose journal repair can lock its torn record but not back it up."""
    materialized = materialize_damaged(tmp_path, "current-journal", "tree", truncated)
    with read_only_once_locked(materialized.ledger_path, monkeypatch):
        return run_experiment(materialized.experiment)


def _claim_ledger_under(parent: Callable[[Path], Path]) -> Trigger:
    """Return a trigger that claims a fresh journal ledger to be created below ``parent``."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        ledger = parent(tmp_path) / "ledgers" / "study.journal"
        experiment = make_experiment(workdir=tmp_path / "runs", storage=f"journal:///{ledger}")
        return engine_ledger.claim_ledger(engine_ledger.validate_ledger(experiment))

    return trigger


def _regular_file(tmp_path: Path) -> Path:
    """Return a regular file where the ledger's path needs a directory."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n")
    return blocker


def _run_fails_then_cleanup_unconfirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Fail a run that launched work, then lose the root before reconciliation can confirm cleanup."""
    experiment = make_experiment(workdir=tmp_path / "runs")

    def fail_run(*_args: object, generation: InvocationGeneration, **_kwargs: object) -> None:
        generation.claim()
        monkeypatch.setattr(guards, "validate_ledger", raiser(OSError("root is gone")))
        raise PhaseTrialsFailedError("trainer failed")

    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)
    return run_experiment(experiment)


def _run_fails_then_registry_turns_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Fail a run that launched work, whose reconciliation then finds the attempt registry shared."""
    experiment = make_experiment(workdir=tmp_path / "runs")

    def fail_run(*_args: object, generation: InvocationGeneration, **_kwargs: object) -> None:
        generation.claim()
        registry = _attempts_dir(experiment)
        registry.mkdir(parents=True, exist_ok=True)
        registry.chmod(0o755)
        raise PhaseTrialsFailedError("trainer failed")

    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)
    return run_experiment(experiment)


def _reap_unreadable_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Reap stale trials from a study whose trial list cannot be read."""
    study = optuna.create_study(study_name="t::p")
    monkeypatch.setattr(optuna.Study, "get_trials", raiser(_storage_gone()))
    return _reap_stale_trials(study, make_experiment(workdir=tmp_path / "runs"), "p", confirm=True)


def _scan_registry(
    damage: Callable[[Path], None] = lambda _entry: None,
    patch: Callable[[pytest.MonkeyPatch], None] = lambda _monkeypatch: None,
) -> Trigger:
    """Preflight a registry of one entry for a real trial directory, after ``damage`` and ``patch``.

    ``damage`` receives the entry file; ``patch`` faults the scan itself, after registration,
    so the entry is written by the real code either way.
    """

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        experiment = make_experiment(workdir=tmp_path / "runs")
        trial_dir = tmp_path / "attempt"
        trial_dir.mkdir()
        damage(registered_entry(experiment, trial_dir))
        patch(monkeypatch)
        return _preflight_active_attempts(experiment, _PreflightCleanupReport(), confirm=True)

    return trigger


def _refuse_mixed_types_same_remedy(study: optuna.Study) -> None:
    """Refuse each phase with a different type that names the same remedy."""
    error_type = (
        StudySchemaMismatchError if study.study_name == "t::a" else StudyFingerprintMismatchError
    )
    raise error_type(f"{study.study_name} is unsupported.", action=OperatorAction.FRESH_NAMESPACE)


def _preflight(
    studies: Callable[[], Mapping[str, optuna.Study]],
    *,
    schema_check: Callable[[optuna.Study], None] | None = None,
    after_claim: Callable[[Experiment], None] | None = None,
) -> Trigger:
    """Run preflight over two phases whose claimed studies are ``studies()``."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        experiment = make_experiment(
            workdir=tmp_path / "runs",
            phases=[
                Phase(name="a", n_trials=1, search_space={}),
                Phase(name="b", n_trials=1, search_space={}),
            ],
        )
        claimed = engine_ledger.claim_ledger(engine_ledger.validate_ledger(experiment))
        if after_claim is not None:
            after_claim(experiment)
        if schema_check is not None:
            monkeypatch.setattr(guards, "_initialize_study_schema", schema_check)
        return guards._preflight_existing_studies(
            replace(claimed, studies=MappingProxyType(dict(studies())))
        )

    return trigger


def _two_unmarked_studies() -> dict[str, optuna.Study]:
    """Return two populated studies with no PhaseSweep schema stamp."""
    return {"a": _populated_study("t::a"), "b": _populated_study("t::b")}


def _refuse_schema_differently(study: optuna.Study) -> None:
    """Refuse each phase with the same type but a different remedy."""
    action = (
        OperatorAction.RESTORE_LEDGER
        if study.study_name == "t::a"
        else OperatorAction.FRESH_NAMESPACE
    )
    raise StudySchemaMismatchError(f"{study.study_name} is unsupported.", action=action)


def _share_registry(experiment: Experiment) -> None:
    """Create the attempt registry with a mode other users can enter."""
    registry = _attempts_dir(experiment)
    registry.mkdir(parents=True)
    registry.chmod(0o755)


def _preflight_shared_registry_and_lost_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Preflight a shared registry and a phase whose stale-trial reap loses its ledger."""
    monkeypatch.setattr(
        guards,
        "_reap_stale_trials",
        raiser(StudyStorageUnavailableError("t::a could not be marked FAIL.")),
    )
    trigger = _preflight(lambda: {"a": _populated_study("t::a")}, after_claim=_share_registry)
    return trigger(tmp_path, monkeypatch)


_PREFLIGHT_AGGREGATE = "Experiment recovery preflight found multiple unsafe studies: "

WRAP_CASES = (
    RoutingCase(
        id="claim_ledger_discovery_preserves_override",
        trigger=_claim_while(engine_ledger, "open_existing_study", _ledger_busy),
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RETRY,
        message="Could not inspect persistent study storage for phase 'p'.",
    ),
    RoutingCase(
        id="published_check_preserves_override",
        trigger=_claim_while_published_trials_fail(_ledger_busy),
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RETRY,
        message="Could not inspect persistent study storage for published phase 'p'.",
    ),
    RoutingCase(
        id="run_ownership_compose",
        trigger=_run_while_discovery_fails,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Could not check artifact ownership: Could not inspect persistent study",
        cause=StudyStorageUnavailableError,
    ),
    RoutingCase(
        # The repair's own refusal names its remedy; composing it keeps that.
        id="run_ownership_keeps_journal_repair",
        trigger=_run_over_unbackuppable_repair,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="could not be repaired",
        cause=IncompleteJournalRecordError,
        marks=(requires_nonroot,),
    ),
    RoutingCase(
        # Composed: the cleanup refusal's own repair survives the replacement.
        id="run_failure_cleanup_keeps_repair",
        trigger=_run_fails_then_registry_turns_shared,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="is not a real owner-only directory (mode 0700)",
    ),
    RoutingCase(
        # Negative: unconfirmed cleanup deliberately replaces the run's own remedy.
        id="run_failure_cleanup_replaces",
        trigger=_run_fails_then_cleanup_unconfirmed,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RUN_RECOVER_RUN,
        message="The run failed and subsequent process cleanup could not be confirmed.",
    ),
    RoutingCase(
        id="journal_garbage",
        trigger=_validate_damaged("current-journal", garbage),
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        message="could not be completely read while checking for the PhaseSweep format boundary.",
    ),
    RoutingCase(
        # recover-run reaps through the same read, so the ledger comes back first.
        id="cleanup_reap_inspect",
        trigger=_reap_unreadable_study,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Could not inspect study 't::p' for stale RUNNING trials.",
    ),
    RoutingCase(
        id="preflight_same_type_aggregate",
        trigger=_preflight(_two_unmarked_studies),
        raised=StudySchemaMismatchError,
        action=OperatorAction.FRESH_NAMESPACE,
        message=_PREFLIGHT_AGGREGATE,
    ),
    RoutingCase(
        # One type does not make one remedy: disagreeing refusals route to reading them.
        id="preflight_same_type_aggregate_disagreeing",
        trigger=_preflight(_two_unmarked_studies, schema_check=_refuse_schema_differently),
        raised=StudySchemaMismatchError,
        action=OperatorAction.INSPECT_LOGS,
        message=_PREFLIGHT_AGGREGATE,
    ),
    RoutingCase(
        # The cleanup aggregate routes by the same shared-or-INSPECT_LOGS rule.
        id="preflight_cleanup_aggregate_disagreeing",
        trigger=_preflight_shared_registry_and_lost_ledger,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.INSPECT_LOGS,
        message=_PREFLIGHT_AGGREGATE,
    ),
    RoutingCase(
        id="preflight_mixed_aggregate_agreeing",
        trigger=_preflight(_two_unmarked_studies, schema_check=_refuse_mixed_types_same_remedy),
        raised=PhaseSweepError,
        action=OperatorAction.FRESH_NAMESPACE,
        message=_PREFLIGHT_AGGREGATE,
    ),
)


def _claim_after_unlocked_bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Claim a ledger whose tree another process bound after it was validated."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    validated = engine_ledger.validate_ledger(experiment)
    _artifact_root_binding_path(experiment).parent.mkdir(parents=True)
    _write_artifact_root_binding(experiment)
    return engine_ledger.claim_ledger(validated)


def _preflight_registered_trial_without_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Scan a registry entry whose terminal trial lost its durable attempt id."""
    experiment = make_experiment(
        workdir=tmp_path / "runs", storage=f"journal:///{tmp_path / 'study.journal'}"
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    write_attempt_lifecycle(attempt_dir, attempt_id="attempt", state="allocated")
    registered_entry(experiment, attempt_dir)
    storage = engine_ledger._resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=storage)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    running = study.ask()
    running.set_user_attr(GENERATION_ID_ATTR, "generation")
    running.set_user_attr(TRIAL_DIR_ATTR, str(attempt_dir))
    running.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(running, state=optuna.trial.TrialState.FAIL)
    return _preflight_active_attempts(experiment, _PreflightCleanupReport(), confirm=True)


def _unreadable_tracker_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Experiment:
    """Return a persistent sweep whose trainer exits cleanly while its tracker gives no answer."""
    monkeypatch.setattr(
        trial, "run_extractor", raiser(EvidenceUnreadableError("W&B gave no answer"))
    )
    return make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'study.journal'}",
        n_trials=1,
        trial_command="echo x=0.5 {overrides}",
    )


def _run_while_tracker_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Run a trial whose finished training the tracker gives no answer about."""
    return run_experiment(_unreadable_tracker_sweep(tmp_path, monkeypatch))


def _resume_while_tracker_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Run again while the tracker still gives no answer about the pending trial."""
    experiment = _unreadable_tracker_sweep(tmp_path, monkeypatch)
    with pytest.raises(TrackingTargetUnavailableError):
        run_experiment(experiment)
    return run_experiment(experiment)


def _inspect_uncertain(*kept: str) -> Trigger:
    """Inspect a cleanup-uncertain FAIL trial whose ledger kept only the ``kept`` identity attrs.

    The trial directory exists and holds no identity files, so a trial that
    names both its attempt and its directory reaches the identity-file check.
    """

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        identity = {ATTEMPT_ID_ATTR: "attempt", TRIAL_DIR_ATTR: str(trial_dir)}
        study = optuna.create_study(study_name="t::p")
        uncertain = study.ask()
        for key in kept:
            uncertain.set_user_attr(key, identity[key])
        uncertain.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
        study.tell(uncertain, state=optuna.trial.TrialState.FAIL)
        experiment = make_experiment(workdir=tmp_path / "runs")
        return _recover_cleanup_uncertain_trials(study, experiment, "p", confirm=False)

    return trigger


def _lock_dir_without_account_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the lock directory for a uid the account database does not know."""
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", raiser(KeyError("uid")))
    return lock_dir()


def _lock_dir_override_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve an explicit lock directory the OS refuses to open."""
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(
        runtime_files, "open_directory_fd", raiser(PermissionError(13, "Permission denied"))
    )
    return lock_dir()


def _lock_dir_default_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the default lock directory when the OS refuses to create it."""
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path)))
    monkeypatch.setattr(
        runtime_files, "ensure_private_dir", raiser(PermissionError(13, "Permission denied"))
    )
    return lock_dir()


def _lock_dir_override_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve an explicit lock directory that does not exist."""
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(tmp_path / "typo"))
    return lock_dir()


def _lock_file_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Take a lock file the OS refuses to open."""
    monkeypatch.setattr(
        runtime_files, "_open_lock_file", raiser(PermissionError(13, "Permission denied"))
    )
    return runtime_files.try_lock_file(tmp_path / "blocked.lock")


def _home_override_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the private root through a relative override."""
    monkeypatch.setenv("PHASESWEEP_HOME", "relative-home")
    return phasesweep_home()


def _wandb_sdk_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Check for the W&B SDK in an environment that cannot import it."""
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    return require_wandb_sdk()


def _mlflow_sdk_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Check for the MLflow SDK in an environment that cannot import it."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    return require_mlflow_sdk()


def _field_dropped(entry: Path) -> None:
    """Drop a field every current registry entry carries, keeping its owner-only file."""
    payload = json.loads(entry.read_text())
    del payload["generation_id"]
    entry.write_text(json.dumps(payload))


def _previous_format(entry: Path) -> None:
    """Stamp a registry entry with the previous registry format."""
    payload = json.loads(entry.read_text())
    payload["schema_version"] = ATTEMPT_REGISTRY_SCHEMA_VERSION - 1
    entry.write_text(json.dumps(payload))


def _entry_garbled(entry: Path) -> None:
    """Overwrite a registry entry with text that is not JSON."""
    entry.write_text("{not json\n")


def _locator_cleared(entry: Path) -> None:
    """Leave a journal identity without the locator recovery requires."""
    payload = json.loads(entry.read_text())
    payload["storage_identity"] = "journal:////missing.journal"
    payload["storage_locator"] = None
    entry.write_text(json.dumps(payload))


def _trial_dir_removed(entry: Path) -> None:
    """Remove the trial directory a registry entry names."""
    Path(json.loads(entry.read_text())["trial_dir"]).rmdir()


def _tracking_kind_malformed(entry: Path) -> None:
    """Mark a trial entry as a tracking operation, whose fields it then does not match."""
    payload = json.loads(entry.read_text())
    payload["kind"] = "tracking"
    entry.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "damage",
    [_entry_garbled, _previous_format, _field_dropped, _locator_cleared, _tracking_kind_malformed],
    ids=[
        "unreadable",
        "other-format",
        "partial-schema",
        "identity-without-locator",
        "malformed-tracking",
    ],
)
def test_unusable_attempt_entry_gets_the_process_check_and_the_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: Callable[[Path], None]
) -> None:
    """Restoring files cannot clear an entry recovery cannot use; the refusal names it, and checking the process and deleting it does."""
    entries: list[Path] = []

    def recorded(entry: Path) -> None:
        entries.append(entry)
        damage(entry)

    with pytest.raises(ProcessCleanupUncertainError) as excinfo:
        _scan_registry(damage=recorded)(tmp_path, monkeypatch)

    error = excinfo.value
    assert f"registry entry {entries[0]} " in str(error)
    assert error.remediation is not None
    assert error.remediation.startswith("Read pgid from the process_identity.json")
    assert (
        "run `pgrep -a -g <pgid>` on the host or in the container the sweep ran in; once it "
        "lists none, delete the entry file."
    ) in error.remediation
    assert error.remediation.endswith("delete the entry file.")
    assert error.remediation not in str(error)
    assert _operational_error_text(error).endswith(f"\nfix: {error.remediation}")


def _fd_listing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make listing a directory by descriptor fail, as the private registry scan lists it."""
    real_listdir = os.listdir

    def listdir(path: object = ".") -> list[str]:
        if isinstance(path, int):
            raise OSError(errno.EIO, "Input/output error")
        return real_listdir(path)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "listdir", listdir)


def _reap_running_trial(tmp_path: Path, **attrs: object) -> object:
    """Reap one RUNNING in-memory trial that carries ``attrs``."""
    study = optuna.create_study(study_name="t::p")
    running = study.ask()
    for key, value in attrs.items():
        running.set_user_attr(key, value)
    return _reap_stale_trials(study, make_experiment(workdir=tmp_path / "runs"), "p", confirm=True)


def _reap_trial_with_invalid_trial_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Reap a RUNNING trial whose persisted trial directory is not a path."""
    return _reap_running_trial(tmp_path, **{TRIAL_DIR_ATTR: 7})


def _reap_trial_without_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Reap a RUNNING trial whose ledger kept its directory but lost its attempt id.

    A launch writes the attempt id before the directory, so only damage leaves this.
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    return _reap_running_trial(tmp_path, **{TRIAL_DIR_ATTR: str(trial_dir)})


def _reap_trial_with_foreign_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Reap a RUNNING trial whose lifecycle record names another attempt."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    write_attempt_lifecycle(trial_dir, attempt_id="another", state="launching")
    return _reap_running_trial(
        tmp_path, **{TRIAL_DIR_ATTR: str(trial_dir), ATTEMPT_ID_ATTR: "attempt"}
    )


def _append_failure_outcomes(
    study: optuna.Study, count: int, *, abort_for: Phase | None = None
) -> None:
    """Append failed trials after the study's last outcome, the last carrying an abort.

    :param optuna.Study study: The golden fixture's phase study.
    :param int count: Failed trials to append.
    :param Phase | None abort_for: Phase whose consecutive-failure abort the last
        appended outcome records, or ``None`` to record no abort.
    """
    trials = study.get_trials(deepcopy=False)
    digest = trials[0].user_attrs[TRAINER_ENV_DIGEST_ATTR]
    last = max(trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in trials)
    for sequence in range(last + 1, last + count + 1):
        outcome = trial_outcome(sequence, "failure")
        if abort_for is not None and sequence == last + count:
            outcome[TRIAL_OUTCOME_ABORT_KEY] = _failure_policy_abort_record(
                abort_for,
                consecutive_failures=count,
                completion_sequence=sequence,
                trial_target=abort_for.n_trials,
            )
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                user_attrs={TRAINER_ENV_DIGEST_ATTR: digest, TRIAL_OUTCOME_ATTR: outcome},
            )
        )


def _rerun_aborted_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Re-run a phase whose outcome ledger recorded an abort at its current trial target."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    study = phase_study(experiment)
    phase = experiment.phases[0].model_copy(update={"n_trials": 4})
    _append_failure_outcomes(study, 2, abort_for=phase)
    study.set_user_attr(TRIAL_TARGET_ATTR, 4)
    return run_experiment(experiment.model_copy(update={"phases": [phase]}))


def _record_fatal_abort(
    tmp_path: Path, policy: str, cause: str, attrs: Mapping[str, object]
) -> tuple[Experiment, Phase]:
    """Record a fatal abort on the golden fixture's phase, at an accepted target of 4.

    :param Path tmp_path: Where the fixture is materialized.
    :param str policy: The fatal abort policy to record.
    :param str cause: The recorded cause.
    :param Mapping[str, object] attrs: Further user attributes of the fatal trial.
    :return tuple[Experiment, Phase]: The fixture's experiment, and its phase at that target.
    """
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    study = phase_study(experiment)
    phase = experiment.phases[0].model_copy(update={"n_trials": 4})
    trials = study.get_trials(deepcopy=False)
    sequence = max(trial.user_attrs[TRIAL_OUTCOME_ATTR]["sequence"] for trial in trials) + 1
    outcome = trial_outcome(sequence, "fatal", policy=policy)
    outcome[TRIAL_OUTCOME_ABORT_KEY] = _abort_record(
        policy=policy,
        completion_sequence=sequence,
        trial_target=phase.n_trials,
        cause=cause,
    )
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={
                TRAINER_ENV_DIGEST_ATTR: trials[0].user_attrs[TRAINER_ENV_DIGEST_ATTR],
                TRIAL_OUTCOME_ATTR: outcome,
                **attrs,
            },
        )
    )
    study.set_user_attr(TRIAL_TARGET_ATTR, phase.n_trials)
    return experiment, phase


_UNCONFIRMED_CLEANUP = ("unsafe_process_cleanup", "trial cleanup could not be confirmed")


def _preview_unconfirmed_cleanup_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Preview a phase whose unsafe-cleanup abort no cleanup recovery has retired.

    The raised target needs new trials, so the preview applies the cohort check a launch
    applies first, which passes only from the fixture's pinned environment.
    """
    pin_fixture_environment(monkeypatch)
    experiment, phase = _record_fatal_abort(
        tmp_path, *_UNCONFIRMED_CLEANUP, {CLEANUP_CONFIRMED_ATTR: False}
    )
    return run_experiment(experiment.model_copy(update={"phases": [phase]}), dry_run=True)


def _finish_aborted_phase(experiment: Experiment, phase: Phase) -> object:
    """Finish a phase the way a live run does once an outcome has recorded an abort.

    A live run raises a fatal error itself before the phase is finished, so a direct call is
    the one way to reach the abort's headline with a fatal abort on record.
    """
    run = SimpleNamespace(
        experiment=experiment.model_copy(update={"phases": [phase]}),
        phase=phase,
        study=phase_study(experiment),
        optimize_deadline=None,
        deadline_exhausted=False,
        aborted=True,
    )
    return _finish_phase(run, phase_fingerprint="unused")


def _finish_after_fatal_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Finish a phase whose recorded abort is an unexpected objective error."""
    experiment, phase = _record_fatal_abort(
        tmp_path, "unexpected_objective_exception", "RuntimeError: objective defect", {}
    )
    return _finish_aborted_phase(experiment, phase)


def _finish_after_unconfirmed_cleanup_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Finish a phase whose recorded abort is a cleanup that was never confirmed."""
    experiment, phase = _record_fatal_abort(
        tmp_path, *_UNCONFIRMED_CLEANUP, {CLEANUP_CONFIRMED_ATTR: False}
    )
    return _finish_aborted_phase(experiment, phase)


def _live_abort(
    tmp_path: Path,
    *,
    body: str,
    cells: int,
    target: int,
    bounded: bool = False,
    persistent: bool = True,
) -> object:
    """Run a grid whose first two trials trip a failure limit of 2, and raise its abort.

    :param str body: The trainer's source.
    :param int cells: Cells in the grid.
    :param int target: The phase's ``n_trials``.
    :param bool bounded: Whether the metric rejects the value 5.0 the trainers print.
    :param bool persistent: Whether the study is kept on a journal.
    """
    phase = Phase(
        name="p",
        n_trials=target,
        max_consecutive_failures=2,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=cells)},
    )
    extractor = LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)")
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"journal:///{tmp_path / 'studies.journal'}" if persistent else None,
        trainer=write_trainer(tmp_path / "trainer.py", body),
        phases=[phase],
        metric=Metric(extractor=extractor, max=1.0) if bounded else None,
    )
    return run_experiment(experiment)


_FAILS = "raise SystemExit(1)"
_OUT_OF_BOUNDS = 'print("x=5.0")'


def _abort_with_failed_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Abort a persistent two-cell grid whose cells both failed."""
    return _live_abort(tmp_path, body=_FAILS, cells=2, target=2)


def _abort_with_unrun_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Abort a persistent four-cell grid on an infeasible streak, leaving two cells unrun."""
    return _live_abort(tmp_path, body=_OUT_OF_BOUNDS, cells=4, target=4, bounded=True)


def _abort_with_every_cell_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Abort a persistent two-cell grid whose cells both measured infeasible."""
    return _live_abort(tmp_path, body=_OUT_OF_BOUNDS, cells=2, target=2, bounded=True)


def _abort_without_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Abort a grid held in memory, which no later run can continue."""
    return _live_abort(tmp_path, body=_FAILS, cells=2, target=2, persistent=False)


def _resume_with_streak_at_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resume, at its accepted target, a phase whose failure streak meets a lowered limit.

    The fixture's trials record the pinned environment they were generated under, so the
    cohort check that allocation runs (``_validate_environment_cohort``) passes only from
    that environment, whatever shell runs the test.
    """
    pin_fixture_environment(monkeypatch)
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    study = phase_study(experiment)
    _append_failure_outcomes(study, 2)
    study.set_user_attr(TRIAL_TARGET_ATTR, 5)
    phase = experiment.phases[0].model_copy(update={"n_trials": 5, "max_consecutive_failures": 2})
    return run_experiment(experiment.model_copy(update={"phases": [phase]}))


def _replay_partial_decision_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Re-run a phase whose accepted partial-timeout decision this config does not accept."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    study = phase_study(experiment)
    target = len(study.get_trials(deepcopy=False)) + 1
    record_accepted_partial_timeout(study, target)
    phase = experiment.phases[0].model_copy(update={"n_trials": target})
    return run_experiment(experiment.model_copy(update={"phases": [phase]}))


def _resume_over_incomplete_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Skip a phase whose published winner is a partial result this config does not accept."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    republish_as_incomplete(experiment)
    return _load_winner(
        experiment,
        experiment.phases[0],
        {},
        publication=_published_for_resume(experiment),
    )


def _clean_trial_from_foreign_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Clean a stale trial group whose identity was recorded in another PID namespace."""
    monkeypatch.setattr("phasesweep.runtime.reaper.read_boot_id", lambda: "boot")
    monkeypatch.setattr(
        "phasesweep.runtime.reaper.read_pid_namespace", lambda: "pid:[4026539333]@100"
    )
    return cleanup_stale_trial_process(
        StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id="foreign-attempt",
            pid=4242,
            pgid=4242,
            proc_starttime=111,
            boot_id="boot",
            pid_namespace="pid:[4026539001]@100",
        )
    )


def _run_with_missing_program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Run a fresh tree whose trial command names a program missing from the trainer's PATH."""
    return run_experiment(
        make_experiment(
            workdir=tmp_path / "runs", trial_command="phasesweep-no-such-trainer {overrides}"
        )
    )


def _run_with_missing_trainer_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Run a fresh tree whose execution.cwd names no directory."""
    return run_experiment(
        make_experiment(
            workdir=tmp_path / "runs", execution=ExecutionContext(cwd=str(tmp_path / "gone"))
        )
    )


ORIGIN_CASES = (
    RoutingCase(
        id="stale_trial_in_foreign_pid_namespace",
        trigger=_clean_trial_from_foreign_namespace,
        raised=ForeignPidNamespaceError,
        action=OperatorAction.FIX_CONFIG,
        message="only the host's initial PID namespace can see another namespace's processes",
    ),
    RoutingCase(
        id="claim_ledger_tree_changed",
        trigger=_claim_after_unlocked_bind,
        raised=ArtifactRootConflictError,
        action=OperatorAction.INSPECT_LOGS,
        message="without the experiment lock; stop it and retry.",
    ),
    RoutingCase(
        id="registry_terminal_identity_missing",
        trigger=_preflight_registered_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="attempt or generation identity is missing",
    ),
    RoutingCase(
        id="uncertain_trial_attempt_missing",
        trigger=_inspect_uncertain(TRIAL_DIR_ATTR),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="'phasesweep_attempt_id' user attribute is missing or invalid",
    ),
    RoutingCase(
        # Either the ledger or the tree may be the side that changed.
        id="uncertain_trial_identity_files_mismatch",
        trigger=_inspect_uncertain(ATTEMPT_ID_ATTR, TRIAL_DIR_ATTR),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.INSPECT_LOGS,
        message="durable process identity is missing, malformed, partial",
    ),
    RoutingCase(
        id="lock_dir_no_account_home",
        trigger=_lock_dir_without_account_home,
        raised=UnsafeLockPathError,
        action=OperatorAction.FIX_CONFIG,
        message="provision an absolute lock directory and set PHASESWEEP_LOCK_DIR.",
    ),
    RoutingCase(
        id="lock_dir_override_access_refused",
        trigger=_lock_dir_override_refused,
        raised=UnsafeLockPathError,
        action=OperatorAction.PROVISION_LOCK_DIR,
        message="Cannot access lock directory",
    ),
    RoutingCase(
        id="lock_dir_default_access_refused",
        trigger=_lock_dir_default_refused,
        raised=UnsafeLockPathError,
        action=OperatorAction.PROVISION_LOCK_DIR,
        message="Cannot create or access default lock directory",
    ),
    RoutingCase(
        id="lock_dir_override_missing",
        trigger=_lock_dir_override_missing,
        raised=UnsafeLockPathError,
        action=OperatorAction.PROVISION_LOCK_DIR,
        message="does not exist.",
    ),
    RoutingCase(
        id="lock_file_access_refused",
        trigger=_lock_file_refused,
        raised=UnsafeLockPathError,
        action=OperatorAction.PROVISION_LOCK_DIR,
        message="Cannot open lock file",
    ),
    RoutingCase(
        id="phasesweep_home_override_relative",
        trigger=_home_override_relative,
        raised=UnsafePrivatePathError,
        action=OperatorAction.FIX_CONFIG,
        message="PHASESWEEP_HOME must be an absolute path",
    ),
    RoutingCase(
        id="wandb_sdk_missing",
        trigger=_wandb_sdk_not_installed,
        raised=PhaseSweepError,
        action=OperatorAction.FIX_CONFIG,
        message='pip install "phasesweep[wandb]"',
    ),
    RoutingCase(
        id="mlflow_sdk_missing",
        trigger=_mlflow_sdk_not_installed,
        raised=PhaseSweepError,
        action=OperatorAction.FIX_CONFIG,
        message='pip install "phasesweep[mlflow]"',
    ),
    RoutingCase(
        id="registry_entry_unreadable",
        trigger=_scan_registry(damage=_entry_garbled),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="is unreadable or malformed:",
    ),
    RoutingCase(
        id="registry_entry_partial_schema",
        trigger=_scan_registry(damage=_field_dropped),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="has an unsupported or partial schema.",
    ),
    RoutingCase(
        id="registry_entry_other_format",
        trigger=_scan_registry(damage=_previous_format),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message=(
            f"uses format {ATTEMPT_REGISTRY_SCHEMA_VERSION - 1}; "
            f"this PhaseSweep reads format {ATTEMPT_REGISTRY_SCHEMA_VERSION}."
        ),
    ),
    RoutingCase(
        id="registry_identity_without_locator",
        trigger=_scan_registry(damage=_locator_cleared),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="has an unsupported or partial schema.",
    ),
    RoutingCase(
        id="registry_entry_trial_dir_missing",
        trigger=_scan_registry(damage=_trial_dir_removed),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="its process state cannot be verified.",
    ),
    # recover-run runs every cleanup check below in both of its modes, so none
    # of these may route back to it.
    RoutingCase(
        id="registry_unenumerable",
        trigger=_scan_registry(patch=_fd_listing_fails),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="cannot be enumerated safely.",
    ),
    RoutingCase(
        id="running_trial_dir_invalid",
        trigger=_reap_trial_with_invalid_trial_dir,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="'phasesweep_trial_dir' user attribute is missing or invalid",
    ),
    RoutingCase(
        id="running_trial_attempt_missing",
        trigger=_reap_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="'phasesweep_attempt_id' user attribute is missing or invalid",
    ),
    RoutingCase(
        # The same split as uncertain_trial_identity_files_mismatch.
        id="running_trial_lifecycle_mismatch",
        trigger=_reap_trial_with_foreign_lifecycle,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.INSPECT_LOGS,
        message="attempt lifecycle record is malformed or belongs to another attempt",
    ),
    RoutingCase(
        id="uncertain_trial_dir_missing",
        trigger=_inspect_uncertain(ATTEMPT_ID_ATTR),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="'phasesweep_trial_dir' user attribute is missing or invalid",
    ),
    RoutingCase(
        id="prior_phase_abort",
        trigger=_rerun_aborted_phase,
        raised=PhaseAbortRefusedError,
        action=OperatorAction.FIX_CONFIG,
        message="phases[p].n_trials above 4 in the config",
    ),
    RoutingCase(
        # Raising the target does not retire an unsafe-cleanup abort; cleanup recovery does.
        id="prior_phase_abort_awaiting_cleanup_recovery",
        trigger=_preview_unconfirmed_cleanup_abort,
        raised=PhaseAbortRefusedError,
        action=OperatorAction.RUN_RECOVER_RUN,
        message="First confirm the cleanup",
    ),
    # A live abort routes like the refusal that repeats it, so `fix:` agrees with its advice.
    RoutingCase(
        id="live_abort_with_failed_cells",
        trigger=_abort_with_failed_cells,
        raised=PhaseTrialsFailedError,
        action=OperatorAction.FIX_CONFIG,
        message="phases[p].n_trials from 2 to 4 in the config",
    ),
    RoutingCase(
        id="live_abort_with_unrun_cells",
        trigger=_abort_with_unrun_cells,
        raised=PhaseTrialsInfeasibleError,
        action=OperatorAction.FIX_CONFIG,
        message="phases[p].max_consecutive_failures above 2 in the config",
    ),
    RoutingCase(
        id="live_abort_with_every_cell_measured",
        trigger=_abort_with_every_cell_measured,
        raised=PhaseTrialsInfeasibleError,
        action=OperatorAction.FIX_CONFIG,
        message="none is feasible, so change the search space or bounds",
    ),
    RoutingCase(
        id="live_abort_fatal",
        trigger=_finish_after_fatal_abort,
        raised=PhaseTrialsFailedError,
        action=OperatorAction.FIX_CONFIG,
        message="phases[p].n_trials above 4 in the config",
    ),
    RoutingCase(
        id="live_abort_awaiting_cleanup_recovery",
        trigger=_finish_after_unconfirmed_cleanup_abort,
        raised=PhaseTrialsFailedError,
        action=OperatorAction.RUN_RECOVER_RUN,
        message="First confirm the cleanup",
    ),
    RoutingCase(
        # No later run can continue an in-memory study, so only the logs are left.
        id="live_abort_without_storage",
        trigger=_abort_without_storage,
        raised=PhaseTrialsFailedError,
        action=OperatorAction.INSPECT_LOGS,
        message="for stderr logs",
    ),
    RoutingCase(
        id="failure_streak_at_limit",
        trigger=_resume_with_streak_at_limit,
        raised=PhaseAbortRefusedError,
        action=OperatorAction.FIX_CONFIG,
        message="which meets max_consecutive_failures=2.",
    ),
    RoutingCase(
        id="winner_incomplete_without_opt_in",
        trigger=_resume_over_incomplete_winner,
        raised=WinnerIntegrityError,
        action=OperatorAction.FIX_CONFIG,
        message=ACCEPT_INCOMPLETE_STEP,
    ),
    RoutingCase(
        id="partial_decision_without_opt_in",
        trigger=_replay_partial_decision_without_opt_in,
        raised=WinnerIntegrityError,
        action=OperatorAction.FIX_CONFIG,
        message=ACCEPT_INCOMPLETE_STEP,
    ),
    RoutingCase(
        id="ledger_directory_path_unusable",
        trigger=_claim_ledger_under(_regular_file),
        raised=PhaseSweepError,
        action=OperatorAction.FIX_CONFIG,
        message="Correct `storage`, or `workdir` for `storage: auto`.",
    ),
    RoutingCase(
        id="live_trial_tracker_unreadable",
        trigger=_run_while_tracker_unreadable,
        raised=TrackingTargetUnavailableError,
        action=OperatorAction.FIX_CONFIG,
        message="finished training, but the tracker could not be read",
    ),
    RoutingCase(
        id="pending_trial_tracker_unreadable",
        trigger=_resume_while_tracker_unreadable,
        raised=TrackingTargetUnavailableError,
        action=OperatorAction.FIX_CONFIG,
        message="1 completed trial(s) finished training, but the tracker could not be read",
    ),
    RoutingCase(
        id="launch_preflight_program_missing",
        trigger=_run_with_missing_program,
        raised=LaunchEnvironmentError,
        action=OperatorAction.FIX_CONFIG,
        message="program 'phasesweep-no-such-trainer' is not on the trainer's PATH",
    ),
    RoutingCase(
        id="launch_preflight_trainer_cwd_missing",
        trigger=_run_with_missing_trainer_cwd,
        raised=LaunchEnvironmentError,
        action=OperatorAction.FIX_CONFIG,
        message="which is not a directory on this host",
    ),
)


CASES = (*WRAP_CASES, *ORIGIN_CASES)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
def test_raise_sites_route_their_declared_action(
    case: RoutingCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert_routes_declared_action(case, tmp_path, monkeypatch, caplog)


def _cleanup_refusal(action: OperatorAction | None, cause: BaseException | None) -> Exception:
    """Return a cleanup refusal routing ``action`` whose chained cause is ``cause``."""
    refusal = ProcessCleanupUncertainError("cleanup could not be proven", action=action)
    refusal.__cause__ = cause
    return refusal


_LEDGER, _TREE = OperatorAction.RESTORE_LEDGER, OperatorAction.RESTORE_TREE
_LEDGER_LOST = StudyStorageUnavailableError("ledger unreadable")
_SHARED_REGISTRY = ProcessCleanupUncertainError("registry shared", action=_TREE)

# An unconfirmed cleanup, as (primary error, recorded cleanup error, routed
# repairs): the primary's repair where it is a cleanup refusal or a storage
# failure, then the one the recorded cleanup error routes, then recover-run.
# A chained cause never adds a repair.
_UNCONFIRMED_CLEANUPS = (
    (_SHARED_REGISTRY, None, [_TREE]),
    (_cleanup_refusal(_LEDGER, _LEDGER_LOST), None, [_LEDGER]),
    (_cleanup_refusal(_LEDGER, OSError("ledger unreadable")), None, [_LEDGER]),
    (_cleanup_refusal(None, _LEDGER_LOST), None, []),
    (_cleanup_refusal(None, None), None, []),
    (PhaseTrialsFailedError("trainer failed"), _SHARED_REGISTRY, [_TREE]),
    (PhaseTrialsFailedError("trainer failed"), OSError("root is gone"), []),
    (_LEDGER_LOST, None, [_LEDGER]),
    (_LEDGER_LOST, _SHARED_REGISTRY, [_LEDGER, _TREE]),
)


def test_runner_payload_follows_the_routed_steps() -> None:
    errors = [cls("boom") for cls in _operator_error_classes()]
    errors += [case.raised("boom", action=case.action) for case in CASES]
    assert_runner_payload_follows_routing(errors)
    for primary, recorded, routed in _UNCONFIRMED_CLEANUPS:
        failure = mcp_runner._terminal_failure_payload(
            primary,
            run_id="r0",
            stage="execution",
            cleanup_confirmed=False,
            cleanup_error=recorded,
        )
        assert failure["code"] == "cleanup_uncertain"
        # The durable payload schema is unchanged: the action routes, it is not stored.
        assert "action" not in failure
        assert_payload_follows(failure, routed)
    # A ledger failure with a shared-registry cleanup error composes both repairs, word for word.
    composed = mcp_runner._terminal_failure_payload(
        _LEDGER_LOST,
        run_id="r0",
        stage="execution",
        cleanup_confirmed=False,
        cleanup_error=_SHARED_REGISTRY,
    )
    assert composed["remediation"] == (
        "Ask the operator to restore or repair the storage ledger and access to it, and restore "
        "the files and permissions identified in the error, then run `phasesweep mcp "
        "recover-run --state-dir STATE_DIR --run-id r0` on the server host (STATE_DIR is the "
        "catalog's `state_dir`), then repeat it with `--confirm` before another launch. "
        "Details: `logs/r0.log` under the catalog's `state_dir`."
    )
