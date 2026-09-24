"""Contract tests for the operator action every PhaseSweepError carries.

``action`` routes a failure to the remediation an operator should carry out, without parsing
prose; the message stays authoritative about what went wrong, and routing never edits it. The class
walk imports the whole package, so a subclass is covered the moment it exists. The routing table
drives real raise and wrap sites and pins the action each routes, and the MCP runner's failure
payload, where routing reaches an agent, must say exactly those steps.
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
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import optuna
import pytest
import yaml

import phasesweep
import phasesweep.engine.ledger as engine_ledger
from phasesweep.config import Experiment, IntParam, Phase, WandbSummaryRequiredGate
from phasesweep.engine import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    WinnerIntegrityError,
    fingerprints,
    guards,
    run_experiment,
    study_policy,
    trial,
)
from phasesweep.engine.artifact_roots import _write_artifact_root_binding
from phasesweep.engine.artifacts import _load_winner
from phasesweep.engine.attempts import (
    _preflight_active_attempts,
    _PreflightCleanupReport,
    _register_active_attempt,
)
from phasesweep.engine.cleanup import _inspect_cleanup_uncertain_trials, _reap_stale_trials
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _attempts_dir,
    _generation_summary_path,
)
from phasesweep.engine.phase import _failure_policy_abort_record
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    PHASE_ABORT_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.evidence.wandb import require_wandb_sdk
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.recovery import (
    RunRecoveryError,
    _cleanup_runner,
    _finalize_stored_terminal_result_snapshot,
    _load_recovery_studies,
    _publication_recovery_action,
    recover_run,
)
from phasesweep.mcp.runs import _STATE_FORMAT_MARKER_NAME, RunHandle, RunStore
from phasesweep.mcp.snapshots import capture_pre_generation_result_snapshot
from phasesweep.runtime.files import (
    PlatformCapabilityError,
    UnsafeLockPathError,
    UnsafePrivatePathError,
    lock_dir,
    phasesweep_home,
)
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import make_experiment, requires_nonroot
from tests.ledger_fixtures import (
    Materialized,
    ledger_file,
    materialize,
    republish_as_incomplete,
)
from tests.mcp_helpers import make_run_handle, stage_dead_run, write_run_status
from tests.recovery_helpers import load_only_recovery_needs

# Classes proving the module sweep reached past the error modules themselves.
# If an import ever stops happening, these vanish from the walk and say so.
_SWEEP_WITNESSES = frozenset(
    {"UnsafeLockPathError", "NoFeasibleTrialError", "RunRecoveryError", "_PolicyStateWriteError"}
)


def _import_every_module() -> None:
    """Import the whole package so no subclass is missing from the walk."""
    for module in pkgutil.walk_packages(phasesweep.__path__, f"{phasesweep.__name__}."):
        # Importing a ``__main__`` module runs the program it guards.
        if module.name.rsplit(".", 1)[-1] == "__main__":
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


def _populated_study(name: str = "routing") -> optuna.Study:
    """Return an in-memory study holding one finished trial and no PhaseSweep attrs."""
    study = optuna.create_study(study_name=name)
    study.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    return study


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

        # The action is a routing attribute, never part of what the operator reads.
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


Trigger = Callable[[Path, pytest.MonkeyPatch], object]


@dataclass(frozen=True)
class RoutingCase:
    """One real raise site and the routing an operator must receive from it."""

    id: str
    #: Drives the real entry point until the raise under test.
    trigger: Trigger
    raised: type[PhaseSweepError]
    action: OperatorAction
    #: Stable substring of the operator text the site raises today.
    message: str
    #: Exact ``__cause__`` type, pinned only where the runner routes by it.
    cause: type[BaseException] | None = None
    marks: tuple[pytest.MarkDecorator, ...] = ()


# A wrap translates one failure into another at a layer boundary. Each row below drives a real wrap
# site through its real entry point, faulting only the call beneath it, and pins what the operator
# receives: the raised type, the action it routes to, and a stable piece of today's message. A row
# whose action differs from its raised class default is the evidence that the site preserves or
# composes the remediation rather than replacing it.


def _raiser(error: BaseException) -> Callable[..., object]:
    """Return a stand-in that raises ``error`` however it is called."""

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    return fail


def _storage_gone() -> Exception:
    """Return the backend failure a vanished ledger surfaces as."""
    return RuntimeError("storage went away")


def _ledger_busy() -> Exception:
    """Return a storage refusal whose own raise site routed it to a retry."""
    return StudyStorageUnavailableError(
        "Journal storage is locked by another writer.", action=OperatorAction.RETRY
    )


def _garbage(path: Path) -> None:
    """Overwrite a ledger file with bytes that fail to parse as journal records."""
    path.write_bytes(b"this is not a journal record\n" * 64)


def _truncated(path: Path) -> None:
    """Cut the journal's last record short, as an interrupted append leaves it."""
    path.write_bytes(path.read_bytes()[:-8])


def _trials_deleted(path: Path) -> None:
    """Drop every trial while the journal keeps its study-level records intact.

    Every trial-scoped journal record (``CREATE_TRIAL`` and later) carries an
    ``op_code`` of 4 or above; study-level records (create study, study user
    attrs) are 0-3. Keeping only the low op codes reproduces exactly what the
    old ``DELETE FROM trials`` did for the SQLite backend: the study, its
    schema stamp, and its other attributes survive, but every trial is gone.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line).get("op_code", 0) < 4]
    path.write_text("".join(kept), encoding="utf-8")


def _materialize_damaged(
    tmp_path: Path, fixture: str, mode: str, damage: Callable[[Path], None]
) -> Materialized:
    """Copy a current golden ledger and apply ``damage`` to its ledger file."""
    materialized = materialize(fixture, tmp_path, mode=mode)
    damage(ledger_file(materialized, fixture.removeprefix("current-")))
    return materialized


def _dead_uncertain_run(
    materialized: Materialized, tmp_path: Path, config: Path | None = None
) -> tuple[Path, str]:
    """Record a dead, cleanup-uncertain MCP run of ``config``, by default the fixture's own."""
    state_dir = tmp_path / "mcp-state"
    stage_dead_run(
        RunStore(state_dir),
        "wrap-recover",
        config or materialized.config_path,
        materialized.experiment.experiment,
        cleanup_uncertain=True,
    )
    return state_dir, "wrap-recover"


def _claim_while(owner: object, name: str, inbound: Callable[[], Exception]) -> Trigger:
    """Claim the current golden ledger while ``owner.name`` raises."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize("current-journal", tmp_path, mode="tree")
        monkeypatch.setattr(owner, name, _raiser(inbound()))
        return engine_ledger.claim_ledger(engine_ledger.validate_ledger(materialized.experiment))

    return trigger


def _validate_damaged(fixture: str, damage: Callable[[Path], None]) -> Trigger:
    """Validate an unbound current golden ledger after damaging its file."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = _materialize_damaged(tmp_path, fixture, "ledger-only", damage)
        return engine_ledger.validate_ledger(materialized.experiment)

    return trigger


def _recovery_studies_damaged(
    mode: str,
    damage: Callable[[Path], None],
    *,
    ownership_storage_unavailable: bool = False,
    fixture: str = "current-journal",
) -> Trigger:
    """Load recovery's studies from a current golden ledger after damage."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = _materialize_damaged(tmp_path, fixture, mode, damage)
        needs = load_only_recovery_needs(
            ownership_storage_unavailable=ownership_storage_unavailable
        )
        return _load_recovery_studies(materialized.experiment, needs)

    return trigger


def _run_while_discovery_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Start a run over the current golden ledger whose phase study cannot be read."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    monkeypatch.setattr(engine_ledger, "_load_existing_phase_study", _raiser(_storage_gone()))
    return run_experiment(materialized.experiment)


def _run_over_damaged(fixture: str, damage: Callable[[Path], None]) -> Trigger:
    """Return a trigger that starts a run over a current golden ledger ``damage`` hit."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        return run_experiment(_materialize_damaged(tmp_path, fixture, "tree", damage).experiment)

    return trigger


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


def _unwritable_directory(tmp_path: Path) -> Path:
    """Return a directory this user may read but not create entries in."""
    parent = tmp_path / "read-only"
    parent.mkdir()
    parent.chmod(0o555)
    return parent


def _run_fails_then_cleanup_unconfirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Fail a run, then lose the root before reconciliation can confirm cleanup.

    Mirrors
    ``tests/test_mcp_runner.py::test_terminal_report_marks_failed_root_discovery_uncertain``.
    """
    experiment = make_experiment(workdir=tmp_path / "runs")

    def fail_run(*_args: object, **_kwargs: object) -> None:
        monkeypatch.setattr(guards, "validate_ledger", _raiser(OSError("root is gone")))
        raise NoFeasibleTrialError("trainer failed")

    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)
    return run_experiment(experiment)


def _run_fails_then_registry_turns_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Fail a run whose reconciliation then finds the attempt registry shared."""
    experiment = make_experiment(workdir=tmp_path / "runs")

    def fail_run(*_args: object, **_kwargs: object) -> None:
        registry = _attempts_dir(experiment)
        registry.mkdir(parents=True, exist_ok=True)
        registry.chmod(0o755)
        raise NoFeasibleTrialError("trainer failed")

    monkeypatch.setattr("phasesweep.engine.run._run_experiment_inner", fail_run)
    return run_experiment(experiment)


def _reap_unreadable_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Reap stale trials from a study whose trial list cannot be read."""
    study = optuna.create_study(study_name="t::p")
    monkeypatch.setattr(optuna.Study, "get_trials", _raiser(_storage_gone()))
    return _reap_stale_trials(study, make_experiment(workdir=tmp_path / "runs"), "p")


def _recover_while_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Confirm recovery while another orchestrator holds the experiment lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    with _experiment_lock(materialized.experiment):
        return recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)


def _recover_over_pre_cutover_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover from a state directory the 0.3.1 runtime left: run handles, no format marker."""
    state_dir = tmp_path / "mcp-state"
    RunStore(state_dir).create(make_run_handle(run_id="wrap-recover"))
    (state_dir / _STATE_FORMAT_MARKER_NAME).unlink()
    return recover_run(state_dir, "wrap-recover", confirm=False, emit=lambda _message: None)


def _recover_from_mistyped_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover from an ordinary shared directory that holds no run-store layout."""
    not_state = tmp_path / "project"
    not_state.mkdir(mode=0o755)
    not_state.chmod(0o755)
    return recover_run(not_state, "wrap-recover", confirm=False, emit=lambda _message: None)


def _recover_from_shared_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover from a complete run-store layout whose runs directory lost its privacy."""
    state_dir = tmp_path / "mcp-state"
    RunStore(state_dir)
    (state_dir / "runs").chmod(0o755)
    return recover_run(state_dir, "wrap-recover", confirm=False, emit=lambda _message: None)


def _recover_pending_snapshot_while(failing: str, error: Exception) -> Trigger:
    """Confirm recovery of an orphaned pending snapshot while ``failing`` raises ``error``."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize("current-journal", tmp_path, mode="tree")
        state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
        write_run_status(
            RunStore.open_existing(state_dir),
            run_id,
            returncode=1,
            result_snapshot=capture_pre_generation_result_snapshot(materialized.experiment),
            result_snapshot_state="pending",
        )
        monkeypatch.setattr(f"phasesweep.mcp.recovery.{failing}", _raiser(error))
        return recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)

    return trigger


def _finalize_complete_snapshot_while(failing: str, error: Exception) -> Trigger:
    """Finalize a complete stored snapshot while ``failing`` raises ``error``."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        terminal_status = {
            "run_id": "wrap-recover",
            "result_snapshot": capture_pre_generation_result_snapshot(make_experiment()),
            "result_snapshot_state": "complete",
        }
        monkeypatch.setattr(f"phasesweep.mcp.recovery.{failing}", _raiser(error))
        return _finalize_stored_terminal_result_snapshot(
            RunStore(tmp_path / "mcp-state"),
            "wrap-recover",
            terminal_status,
            confirmed_attempt_ids=set(),
            confirmed_attempt_locations={},
        )

    return trigger


def _registered_entry(experiment: Experiment, trial_dir: Path) -> Path:
    """Register one attempt of phase ``p`` and return its registry entry file."""
    _register_active_attempt(
        experiment,
        attempt_id="attempt",
        phase_name="p",
        study_name="t::p",
        trial_number=0,
        trial_dir=trial_dir,
        generation_id="generation",
    )
    return _attempts_dir(experiment) / "attempt.json"


def _recover_over_malformed_registry_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Inspect recovery while an attempt registry entry no longer parses."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    trial_dir = tmp_path / "attempt"
    trial_dir.mkdir()
    _garbage(_registered_entry(materialized.experiment, trial_dir))
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


_NO_NOFOLLOW = "O_NOFOLLOW is unavailable."


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
        damage(_registered_entry(experiment, trial_dir))
        patch(monkeypatch)
        return _preflight_active_attempts(experiment, _PreflightCleanupReport())

    return trigger


def _lacks_capability(helper: str) -> Callable[[pytest.MonkeyPatch], None]:
    """Return a patch under which the registry's ``helper`` lacks a platform capability."""
    return lambda monkeypatch: monkeypatch.setattr(
        f"phasesweep.engine.attempts.{helper}", _raiser(PlatformCapabilityError(_NO_NOFOLLOW))
    )


def _moved_workdir(materialized: Materialized, tmp_path: Path) -> Experiment:
    """Return the fixture's config pointed at a workdir its studies never published into."""
    return materialized.experiment.model_copy(update={"workdir": str(tmp_path / "moved")})


def _recover_through_retargeted_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery after the symlink a run's pinned workdir names was retargeted.

    recover-run reads the run's digest-pinned config snapshot, so the workdir can stop resolving to
    the studies' root only through the filesystem, as when a volume behind a symlink moves.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    link = tmp_path / "data"
    link.symlink_to(materialized.experiment.workdir)
    config = yaml.safe_load(materialized.config_path.read_text())
    config["workdir"] = str(link)
    pinned = tmp_path / "pinned.yaml"
    pinned.write_text(yaml.safe_dump(config))
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path, pinned)
    link.unlink()
    (tmp_path / "new-volume").mkdir()
    link.symlink_to(tmp_path / "new-volume")
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


def _refuse_as_prior_release(study: optuna.Study) -> None:
    """Refuse each phase with a different type that names the same remedy."""
    error_type = (
        StudySchemaMismatchError if study.study_name == "t::a" else StudyFingerprintMismatchError
    )
    raise error_type(
        f"{study.study_name} predates this release.", action=OperatorAction.USE_PRIOR_RELEASE
    )


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
            monkeypatch.setattr(guards, "_validate_study_schema", schema_check)
        return guards._preflight_existing_studies(
            replace(claimed, studies=MappingProxyType(dict(studies())))
        )

    return trigger


def _two_precutover_studies() -> dict[str, optuna.Study]:
    """Return two populated studies that predate the schema stamp."""
    return {"a": _populated_study("t::a"), "b": _populated_study("t::b")}


def _refuse_schema_differently(study: optuna.Study) -> None:
    """Refuse each phase with the same type but a different remedy."""
    action = (
        OperatorAction.USE_PRIOR_RELEASE
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
        _raiser(StudyStorageUnavailableError("t::a could not be marked FAIL.")),
    )
    trigger = _preflight(lambda: {"a": _populated_study("t::a")}, after_claim=_share_registry)
    return trigger(tmp_path, monkeypatch)


_PREFLIGHT_AGGREGATE = "Experiment recovery preflight found multiple unsafe studies: "

WRAP_CASES = (
    RoutingCase(
        id="claim_ledger_discovery_preserves_override",
        trigger=_claim_while(engine_ledger, "_load_existing_phase_study", _ledger_busy),
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RETRY,
        message="Could not inspect persistent study storage for phase 'p'.",
    ),
    RoutingCase(
        id="published_check_preserves_override",
        trigger=_claim_while(optuna.Study, "get_trials", _ledger_busy),
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RETRY,
        message="Could not inspect persistent study storage for published phase 'p'.",
    ),
    RoutingCase(
        id="run_ownership_compose",
        trigger=_run_while_discovery_fails,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Artifact ownership could not be checked because required persistent",
        cause=StudyStorageUnavailableError,
    ),
    RoutingCase(
        # The partial record names its own repair; restoring the ledger is not it.
        id="run_ownership_keeps_journal_repair",
        trigger=_run_over_damaged("current-journal", _truncated),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Nothing was written. Cleanup state is therefore unknown. For an MCP run",
        cause=IncompleteJournalRecordError,
    ),
    RoutingCase(
        # Composed: the cleanup refusal's own repair survives the replacement.
        id="run_failure_cleanup_keeps_repair",
        trigger=_run_fails_then_registry_turns_shared,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the original registry with mode 0700 before retrying.",
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
        id="recovery_studies_storage_bound",
        trigger=_recovery_studies_damaged("tree", _garbage),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        message=(
            "Restore the original complete storage ledger and access to it, then retry "
            "phasesweep mcp recover-run."
        ),
    ),
    RoutingCase(
        id="recovery_published_missing",
        trigger=_recovery_studies_damaged(
            "tree", _trials_deleted, ownership_storage_unavailable=True
        ),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        message=(
            "Restore the original complete storage ledger and study with access to it, "
            "then retry phasesweep mcp recover-run."
        ),
    ),
    RoutingCase(
        id="recover_run_lock_busy",
        trigger=_recover_while_locked,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="Another phasesweep process appears to be using the same experiment backend",
    ),
    RoutingCase(
        # Composed: the inbound refusal is a ValueError carrying no action, so the
        # site supplies the one its message names.
        id="recover_run_pre_cutover_state",
        trigger=_recover_over_pre_cutover_state,
        raised=RunRecoveryError,
        action=OperatorAction.USE_PRIOR_RELEASE,
        message="use a fresh MCP state directory or the preserved PhaseSweep 0.3.1 runtime",
    ),
    RoutingCase(
        # Composed: a path with no run-store layout is a wrong argument, even
        # when it names some other existing, shared directory.
        id="recover_run_mistyped_state_dir",
        trigger=_recover_from_mistyped_state_dir,
        raised=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        message="Pass the state_dir from the catalog the MCP server runs with.",
    ),
    RoutingCase(
        id="recover_run_shared_state_dir",
        trigger=_recover_from_shared_state_dir,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="must be owned by uid",
    ),
    RoutingCase(
        id="recover_run_pending_snapshot_unsafe_path",
        trigger=_recover_pending_snapshot_while(
            "write_status_file", UnsafePrivatePathError("Private directory is not owner-only.")
        ),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="failed to finalize terminal result snapshot for wrap-recover: "
        "UnsafePrivatePathError",
    ),
    RoutingCase(
        id="finalize_snapshot_platform",
        trigger=_finalize_complete_snapshot_while(
            "write_status_file", PlatformCapabilityError("O_NOFOLLOW is unavailable.")
        ),
        raised=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        message="failed to finalize terminal result snapshot for wrap-recover: "
        "PlatformCapabilityError",
    ),
    RoutingCase(
        # recover-run reads the same entry, so routing it back to recover-run
        # would loop; the entry itself is what the operator repairs.
        id="recover_run_malformed_registry_entry",
        trigger=_recover_over_malformed_registry_entry,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="delete the entry file if you are certain nothing is running.",
    ),
    RoutingCase(
        # Composed: the claim path routes this refusal to fixing the config,
        # but recovery's config is pinned, so only the tree can be restored.
        id="recovery_studies_root_moved",
        trigger=_recover_through_retargeted_workdir,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the original workdir, or use a fresh artifact root",
    ),
    RoutingCase(
        id="journal_garbage",
        trigger=_validate_damaged("current-journal", _garbage),
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        message="could not be completely read while checking for the PhaseSweep format boundary.",
    ),
    RoutingCase(
        # Inspection previews the confirmed write, so it refuses a cut-short
        # last record with the repair, not recover-run's usual restore remedy.
        id="recovery_studies_journal_incomplete_record",
        trigger=_recovery_studies_damaged("tree", _truncated, fixture="current-journal"),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Stop every process that uses this journal, then retry. If the retry succeeds",
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
        # Not a damaged registry: the missing capability keeps its own routing.
        id="registry_open_platform_capability",
        trigger=_scan_registry(patch=_lacks_capability("open_directory_fd")),
        raised=PlatformCapabilityError,
        action=OperatorAction.FIX_CONFIG,
        message=_NO_NOFOLLOW,
    ),
    RoutingCase(
        id="registry_entry_platform_capability",
        trigger=_scan_registry(patch=_lacks_capability("read_private_text_at")),
        raised=PlatformCapabilityError,
        action=OperatorAction.FIX_CONFIG,
        message=_NO_NOFOLLOW,
    ),
    RoutingCase(
        id="preflight_same_type_aggregate",
        trigger=_preflight(_two_precutover_studies),
        raised=StudySchemaMismatchError,
        action=OperatorAction.USE_PRIOR_RELEASE,
        message=_PREFLIGHT_AGGREGATE,
    ),
    RoutingCase(
        # One type does not make one remedy: disagreeing refusals route to reading them.
        id="preflight_same_type_aggregate_disagreeing",
        trigger=_preflight(_two_precutover_studies, schema_check=_refuse_schema_differently),
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
        trigger=_preflight(_two_precutover_studies, schema_check=_refuse_as_prior_release),
        raised=PhaseSweepError,
        action=OperatorAction.USE_PRIOR_RELEASE,
        message=_PREFLIGHT_AGGREGATE,
    ),
)


# An origin raise states its remedy in its own message, so its action has to
# name that same remedy rather than whatever its class defaults to. Each row
# drives the real code path to one such raise and pins the pair together.


def _claim_after_unlocked_bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Claim a ledger whose tree another process bound after it was validated."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    validated = engine_ledger.validate_ledger(experiment)
    _artifact_root_binding_path(experiment).parent.mkdir(parents=True)
    _write_artifact_root_binding(experiment)
    return engine_ledger.claim_ledger(validated)


def _reconcile_prepared_over_damaged_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Reconcile a prepared publication after the last-success target lost its summary."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    published = _last_successful_generation_id(experiment)
    assert published is not None
    _generation_summary_path(experiment, published).unlink()
    needs = replace(
        load_only_recovery_needs(),
        prepared_publication_generation="prepared-generation",
    )
    return _publication_recovery_action(experiment, needs)


def _preflight_registered_trial_without_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Scan a registry entry whose terminal trial lost its durable attempt id.

    Mirrors the ``missing-attempt`` case of
    ``tests/test_stale_reaper.py::test_registry_terminal_cleanup_requires_matching_attempt_identity``.
    """
    experiment = make_experiment(
        workdir=tmp_path / "runs", storage=f"journal:///{tmp_path / 'study.journal'}"
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    write_attempt_lifecycle(attempt_dir, attempt_id="attempt", state="allocated")
    _registered_entry(experiment, attempt_dir)
    storage = engine_ledger._resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=storage)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    running = study.ask()
    running.set_user_attr(GENERATION_ID_ATTR, "generation")
    running.set_user_attr(TRIAL_DIR_ATTR, str(attempt_dir))
    running.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(running, state=optuna.trial.TrialState.FAIL)
    return _preflight_active_attempts(experiment, _PreflightCleanupReport())


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
        return _inspect_cleanup_uncertain_trials(study, "p")

    return trigger


def _lock_dir_without_account_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the lock directory for a uid the account database does not know."""
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", _raiser(KeyError("uid")))
    return lock_dir()


def _home_override_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the private root through a relative override."""
    monkeypatch.setenv("PHASESWEEP_HOME", "relative-home")
    return phasesweep_home()


def _wandb_sdk_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Check for the W&B SDK in an environment that cannot import it."""
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    return require_wandb_sdk()


def _field_dropped(entry: Path) -> None:
    """Drop a field every current registry entry carries, keeping its owner-only file."""
    payload = json.loads(entry.read_text())
    del payload["generation_id"]
    entry.write_text(json.dumps(payload))


def _trial_dir_removed(entry: Path) -> None:
    """Remove the trial directory a registry entry names."""
    Path(json.loads(entry.read_text())["trial_dir"]).rmdir()


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
    return _reap_stale_trials(study, make_experiment(workdir=tmp_path / "runs"), "p")


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


def _recover_during_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover a run id while another MCP launch holds the launch lock."""
    state_dir = tmp_path / "mcp-state"
    with RunStore(state_dir).launch_lock() as acquired:
        assert acquired
        return recover_run(state_dir, "wrap-recover", confirm=False, emit=lambda _message: None)


def _recover_unsettled_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover a launch whose preparation still holds its lease."""
    state_dir = tmp_path / "mcp-state"
    pending = make_run_handle(run_id="wrap-recover", launch_state="launching")
    preparation = RunStore(state_dir).prepare_launch(pending, b"experiment: t\n")
    try:
        return recover_run(state_dir, "wrap-recover", confirm=False, emit=lambda _message: None)
    finally:
        preparation.close()


def _recover_unknown_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery of a run id the state directory never recorded."""
    state_dir = tmp_path / "mcp-state"
    RunStore(state_dir)
    return recover_run(state_dir, "typo", confirm=False, emit=lambda _message: None)


def _recover_with_config_snapshot(change: Callable[[Path], None]) -> Trigger:
    """Inspect recovery of a dead uncertain run after ``change`` hits its config snapshot."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize("current-journal", tmp_path, mode="tree")
        state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
        change(RunStore.open_existing(state_dir).config_snapshot_path(run_id))
        return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)

    return trigger


def _recover_without_boot_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery on a host whose boot id cannot be read."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    monkeypatch.setattr("phasesweep.mcp.recovery.read_boot_id", lambda: None)
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


def _live_uncertain_run(state_dir: Path) -> tuple[RunStore, RunHandle]:
    """Record a cleanup-uncertain run whose runner is this live test process.

    A regressed liveness check would signal pytest's real group; conftest's
    ``guard_runner_signals`` fails the test instead of delivering it.
    """
    store = RunStore(state_dir)
    handle = make_run_handle(run_id="wrap-recover")
    store.create(handle)
    store.mark_cleanup_uncertain(handle)
    return store, handle


def _recover_live_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery of a run whose runner is still alive."""
    state_dir = tmp_path / "mcp-state"
    _live_uncertain_run(state_dir)
    return recover_run(state_dir, "wrap-recover", confirm=False, emit=lambda _message: None)


def _recheck_live_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Re-check, as confirmed recovery does under its lock, a runner that is alive."""
    store, handle = _live_uncertain_run(tmp_path / "mcp-state")
    return _cleanup_runner(
        store.cleanup_identity(handle),
        load_only_recovery_needs(),
        confirm=True,
        earlier_boot=False,
        cleanup_recorded=False,
    )


def _rerun_aborted_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Re-run a phase whose study durably aborted at its current trial target."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    phase = experiment.phases[0]
    study = optuna.load_study(
        study_name="t::p", storage=engine_ledger._resolve_storage(experiment.resolved_storage)
    )
    study.set_user_attr(
        PHASE_ABORT_ATTR,
        _failure_policy_abort_record(
            phase, consecutive_failures=2, completion_sequence=2, trial_target=phase.n_trials
        ),
    )
    return run_experiment(experiment)


def _resume_over_incomplete_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Skip a phase whose published winner is a partial result this config does not accept."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    republish_as_incomplete(experiment)
    return _load_winner(experiment, experiment.phases[0], {})


def _resume_after_phase_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Skip a phase whose search space changed after its winner was published."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    edited = experiment.phases[0].model_copy(
        update={"search_space": {"a": IntParam(type="int", low=0, high=20)}}
    )
    return _load_winner(experiment.model_copy(update={"phases": [edited]}), edited, {})


def _claim_from_moved_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Claim the fixture's ledger through a config whose workdir moved away from it."""
    moved = _moved_workdir(materialize("current-journal", tmp_path, mode="tree"), tmp_path)
    return engine_ledger.claim_ledger(engine_ledger.validate_ledger(moved))


def _validate_tree_bound_to_another_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Validate a config over a tree that a different storage ledger bound."""
    owner = make_experiment(
        workdir=tmp_path / "runs", storage=f"journal:///{tmp_path / 'a.journal'}"
    )
    _artifact_root_binding_path(owner).parent.mkdir(parents=True)
    _write_artifact_root_binding(owner)
    other = make_experiment(
        workdir=tmp_path / "runs", storage=f"journal:///{tmp_path / 'b.journal'}"
    )
    return engine_ledger.validate_ledger(other)


def _top_up_from_another_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Validate a study whose recorded trial ran under another trainer environment."""
    study = optuna.create_study(study_name="t::p")
    study.add_trial(
        optuna.trial.create_trial(
            value=1.0,
            params={},
            distributions={},
            user_attrs={TRAINER_ENV_DIGEST_ATTR: "e" * 64},
        )
    )
    return study_policy._validate_environment_cohort(study, "d" * 64)


ORIGIN_CASES = (
    RoutingCase(
        id="claim_ledger_tree_changed",
        trigger=_claim_after_unlocked_bind,
        raised=ArtifactRootConflictError,
        action=OperatorAction.INSPECT_LOGS,
        message="Stop that process before retrying.",
    ),
    RoutingCase(
        id="prepared_publication_unreadable",
        trigger=_reconcile_prepared_over_damaged_publication,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the publication evidence or access to it before retrying recovery.",
    ),
    RoutingCase(
        id="registry_terminal_identity_missing",
        trigger=_preflight_registered_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore the original storage ledger with its durable attempt and generation",
    ),
    RoutingCase(
        id="uncertain_trial_attempt_missing",
        trigger=_inspect_uncertain(TRIAL_DIR_ATTR),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Process identity is unknown. Restore the original storage ledger",
    ),
    RoutingCase(
        # Either the ledger or the tree may be the side that changed, so the
        # message names both and no one repair is the remedy.
        id="uncertain_trial_identity_files_mismatch",
        trigger=_inspect_uncertain(ATTEMPT_ID_ATTR, TRIAL_DIR_ATTR),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.INSPECT_LOGS,
        message="Restore the original storage ledger and this attempt's process-identity files",
    ),
    RoutingCase(
        id="lock_dir_no_account_home",
        trigger=_lock_dir_without_account_home,
        raised=UnsafeLockPathError,
        action=OperatorAction.FIX_CONFIG,
        message="provision an absolute lock directory and set PHASESWEEP_LOCK_DIR.",
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
        message='python -m pip install "phasesweep[wandb]"',
    ),
    RoutingCase(
        id="registry_entry_partial_schema",
        trigger=_scan_registry(damage=_field_dropped),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="Delete the entry file only if you are certain no process",
    ),
    RoutingCase(
        id="registry_entry_trial_dir_missing",
        trigger=_scan_registry(damage=_trial_dir_removed),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="Delete the entry file only if you are certain nothing is running.",
    ),
    # recover-run runs every cleanup check below in both of its modes, so none
    # of these may route back to it.
    RoutingCase(
        id="registry_unenumerable",
        trigger=_scan_registry(patch=_fd_listing_fails),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the original registry and access to it before retrying.",
    ),
    RoutingCase(
        id="running_trial_dir_invalid",
        trigger=_reap_trial_with_invalid_trial_dir,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore the original storage ledger before retrying recovery.",
    ),
    RoutingCase(
        id="running_trial_attempt_missing",
        trigger=_reap_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Process identity is unknown. Restore the original storage ledger",
    ),
    RoutingCase(
        # The same split as uncertain_trial_identity_files_mismatch.
        id="running_trial_lifecycle_mismatch",
        trigger=_reap_trial_with_foreign_lifecycle,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.INSPECT_LOGS,
        message="Restore the original storage ledger and this attempt's lifecycle record",
    ),
    RoutingCase(
        id="uncertain_trial_dir_missing",
        trigger=_inspect_uncertain(ATTEMPT_ID_ATTR),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore the original storage ledger before retrying recovery.",
    ),
    RoutingCase(
        id="recover_during_launch",
        trigger=_recover_during_launch,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="wait for it to finish and retry",
    ),
    RoutingCase(
        id="recover_unsettled_launch",
        trigger=_recover_unsettled_launch,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="Wait briefly and retry",
    ),
    # recover-run raises every RunRecoveryError itself, so none may route back
    # to it by default; only a confirmed run's own remedy names it.
    RoutingCase(
        id="recover_unknown_run_id",
        trigger=_recover_unknown_run,
        raised=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        message="pass a run id this MCP state directory records.",
    ),
    RoutingCase(
        id="recover_config_snapshot_missing",
        trigger=_recover_with_config_snapshot(Path.unlink),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore it before retrying recovery.",
    ),
    RoutingCase(
        id="recover_config_snapshot_unreadable",
        trigger=_recover_with_config_snapshot(lambda path: path.chmod(0o000)),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore it or access to it before retrying recovery.",
        marks=(requires_nonroot,),
    ),
    RoutingCase(
        # No mechanical remedy: the host cannot rule out PID reuse.
        id="recover_boot_id_unavailable",
        trigger=_recover_without_boot_id,
        raised=RunRecoveryError,
        action=OperatorAction.INSPECT_LOGS,
        message="runner boot id is unavailable",
    ),
    RoutingCase(
        id="recover_live_runner",
        trigger=_recover_live_runner,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="runner still appears live; use cancel_run first",
    ),
    RoutingCase(
        id="recover_recheck_live_runner",
        trigger=_recheck_live_runner,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="runner still appears live; use cancel_run first",
    ),
    RoutingCase(
        id="prior_phase_abort",
        trigger=_rerun_aborted_phase,
        raised=NoFeasibleTrialError,
        action=OperatorAction.FIX_CONFIG,
        message="Increase n_trials above 2 to explicitly schedule new recovery attempts",
    ),
    RoutingCase(
        id="winner_incomplete_without_opt_in",
        trigger=_resume_over_incomplete_winner,
        raised=WinnerIntegrityError,
        action=OperatorAction.FIX_CONFIG,
        message="unless the current config sets allow_incomplete_on_timeout: true.",
    ),
    RoutingCase(
        id="winner_phase_config_changed",
        trigger=_resume_after_phase_edit,
        raised=StudyFingerprintMismatchError,
        action=OperatorAction.FIX_CONFIG,
        message="or restore the matching config before resuming.",
    ),
    RoutingCase(
        id="study_root_workdir_moved",
        trigger=_claim_from_moved_workdir,
        raised=ArtifactRootConflictError,
        action=OperatorAction.FIX_CONFIG,
        message="Restore the original workdir, or use a fresh artifact root",
    ),
    RoutingCase(
        id="tree_bound_to_another_ledger",
        trigger=_validate_tree_bound_to_another_ledger,
        raised=ArtifactRootConflictError,
        action=OperatorAction.FIX_CONFIG,
        message="Use the config that owns this current-format tree",
    ),
    RoutingCase(
        id="ledger_directory_path_unusable",
        trigger=_claim_ledger_under(_regular_file),
        raised=PhaseSweepError,
        action=OperatorAction.FIX_CONFIG,
        message="Correct the storage path, or the workdir for auto storage",
    ),
    RoutingCase(
        id="ledger_directory_unwritable",
        trigger=_claim_ledger_under(_unwritable_directory),
        raised=PhaseSweepError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore write access to the directory that holds it, then run again.",
        marks=(requires_nonroot,),
    ),
    RoutingCase(
        id="environment_cohort_changed",
        trigger=_top_up_from_another_environment,
        raised=StudyFingerprintMismatchError,
        action=OperatorAction.FIX_CONFIG,
        message="Restore the original semantic environment",
    ),
)


CASES = (*WRAP_CASES, *ORIGIN_CASES)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
def test_raise_sites_route_their_declared_action(
    case: RoutingCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(case.raised) as excinfo:
        case.trigger(tmp_path, monkeypatch)

    error = excinfo.value
    assert type(error) is case.raised
    assert error.action is case.action
    if case.cause is not None:
        assert type(error.__cause__) is case.cause
    assert case.message in str(error)


def _defect_while_inspecting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery while one of its own steps has a bug."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    monkeypatch.setattr(
        "phasesweep.mcp.recovery._publication_recovery_action",
        _raiser(NotImplementedError("unfinished branch")),
    )
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


@pytest.mark.parametrize(
    ("trigger", "defect"),
    [
        (_defect_while_inspecting, NotImplementedError),
        (
            _recover_pending_snapshot_while(
                "write_status_file", TypeError("Object of type set is not JSON serializable")
            ),
            TypeError,
        ),
        (
            _finalize_complete_snapshot_while(
                "finalize_result_snapshot",
                RuntimeError("cleanup report identifies more attempts than the snapshot records"),
            ),
            RuntimeError,
        ),
    ],
    ids=["inspection", "snapshot_reservation", "snapshot_finalization"],
)
def test_recover_run_lets_a_defect_keep_its_traceback(
    trigger: Trigger,
    defect: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bug inside recovery is not an operator refusal with a remedy to route.

    Only a PhaseSweepError, or an OSError from recovery's own filesystem writes, becomes a
    RunRecoveryError; anything else reaches the CLI's internal-error boundary as itself.
    """
    with pytest.raises(defect) as excinfo:
        trigger(tmp_path, monkeypatch)
    assert type(excinfo.value) is defect
    assert not isinstance(excinfo.value, PhaseSweepError)


# The MCP runner is where routing reaches an agent: a run's failure payload
# says what to do next, and may say only what the raise routes. Each marker
# belongs to one step's text alone, so a payload that names a step the raise
# did not route fails as surely as one that omits a step.
_STEP_MARKERS: Mapping[OperatorAction, str] = MappingProxyType(
    {
        OperatorAction.USE_PRIOR_RELEASE: "preserved PhaseSweep release",
        OperatorAction.FRESH_NAMESPACE: "new experiment name",
        OperatorAction.RESTORE_LEDGER: "storage ledger",
        OperatorAction.RESTORE_TREE: "repair the experiment tree",
        OperatorAction.RUN_RECOVER_RUN: "recover-run",
        OperatorAction.FIX_CONFIG: "correct the experiment configuration",
        OperatorAction.INSPECT_LOGS: "inspect the PhaseSweep run log",
    }
)


def _assert_payload_follows(failure: Mapping[str, Any], routed: list[OperatorAction]) -> None:
    """Assert a runner failure payload says exactly the ``routed`` steps, in order."""
    remediation = failure["remediation"]
    steps = [step for step in routed if step is not OperatorAction.RETRY]
    if failure["code"] == "cleanup_uncertain":
        # The server launches nothing more for such a run until recover-run
        # confirms it, so recover-run is how it is retried.
        last = OperatorAction.RUN_RECOVER_RUN
        steps = [*(step for step in steps if step is not last), last]
    if steps:
        assert (failure["retryable"], failure["actor"]) == (False, "operator")
    else:
        assert (failure["retryable"], failure["actor"]) == (True, "agent")
    positions = [remediation.find(_STEP_MARKERS[step]) for step in steps]
    assert -1 not in positions, remediation
    assert positions == sorted(positions), remediation
    unrouted = [marker for action, marker in _STEP_MARKERS.items() if action not in steps]
    assert not [marker for marker in unrouted if marker in remediation], remediation
    if OperatorAction.RESTORE_TREE in steps:
        # The payload carries no message, so it keeps the safety condition.
        assert "only if certain nothing is running" in remediation
    # A step whose specifics are in the message points at the run log holding it.
    self_contained = {OperatorAction.RUN_RECOVER_RUN, OperatorAction.INSPECT_LOGS}
    points_at_log = remediation.endswith("The error in the PhaseSweep run log gives the details.")
    assert points_at_log is not self_contained.issuperset(steps), remediation


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
    (NoFeasibleTrialError("trainer failed"), _SHARED_REGISTRY, [_TREE]),
    (NoFeasibleTrialError("trainer failed"), OSError("root is gone"), []),
    (_LEDGER_LOST, None, [_LEDGER]),
    (_LEDGER_LOST, _SHARED_REGISTRY, [_LEDGER, _TREE]),
)


def test_runner_payload_follows_the_routed_steps() -> None:
    # The payload depends only on an error's type and action, and the tables
    # above prove each trigger raises exactly its row's, so each row's
    # declared routing stands in for driving its trigger again.
    errors = [cls("boom") for cls in _operator_error_classes()]
    errors += [case.raised("boom", action=case.action) for case in CASES]
    for error in errors:
        failure = mcp_runner._safe_failure_payload(error, stage="preflight")
        _assert_payload_follows(failure, [error.action])
    for primary, recorded, routed in _UNCONFIRMED_CLEANUPS:
        failure = mcp_runner._terminal_failure_payload(
            primary, stage="execution", cleanup_confirmed=False, cleanup_error=recorded
        )
        assert failure["code"] == "cleanup_uncertain"
        # The durable payload schema is unchanged: the action routes, it is not stored.
        assert "action" not in failure
        _assert_payload_follows(failure, routed)
    # The last, composed remediation, word for word.
    assert failure["remediation"] == (
        "Ask the operator to restore or repair the storage ledger and access to it, and repair "
        "the experiment tree's files and permissions, deleting a file only if certain nothing "
        "is running, then run phasesweep mcp recover-run before another launch. The error in "
        "the PhaseSweep run log gives the details."
    )
