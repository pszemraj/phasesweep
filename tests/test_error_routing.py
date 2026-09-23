"""Contract tests for the operator actions every PhaseSweepError carries.

``actions`` is a *routing* attribute: it names the remediation an operator
should carry out, so a caller can steer a failure without parsing prose. The
message text stays authoritative about what went wrong, and these tests pin
that separation down in both directions -- every operator-facing error resolves
to real ``OperatorAction`` steps, and attaching them never edits the message.

The class walk below is deliberately exhaustive rather than a hand-maintained
list: it imports every module in the package and then recurses through
``PhaseSweepError.__subclasses__()``, so a subclass added in some far corner of
the tree is covered the moment it exists.

The wrap table near the end carries the same contract across layer boundaries: a
remediation survives each translation unless the site deliberately composes or
replaces it, and every such site is driven for real rather than in isolation.
The origin table after it holds raises whose message names its own remedy to the
action that routes it. The last test keeps multi-step remediations rare: every
raise that requires more than one step must be on an explicit allowlist.
"""

from __future__ import annotations

import ast
import contextlib
import errno
import hashlib
import importlib
import json
import os
import pickle
import pkgutil
import pwd
import sqlite3
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import optuna
import pytest
import yaml

import phasesweep
import phasesweep.engine.ledger as engine_ledger
from phasesweep.config import Experiment, IntParam, Phase, WandbSummaryRequiredGate
from phasesweep.engine import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    LedgerTransactionInterruptedError,
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    PublicationIntegrityError,
    PublishedStudyMissingError,
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
    _generation_winner_path,
    _last_successful_generation_path,
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
    TRIAL_TARGET_ATTR,
)
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.evidence.wandb import require_wandb_sdk
from phasesweep.mcp.recovery import (
    RunRecoveryError,
    _cleanup_runner,
    _finalize_stored_terminal_result_snapshot,
    _load_recovery_studies,
    _publication_recovery_action,
    _RecoveryNeeds,
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
from phasesweep.runtime.process import ATTEMPT_LIFECYCLE_FILE, write_attempt_lifecycle
from tests.conftest import make_experiment
from tests.ledger_fixtures import Materialized, leave_hot_journal, ledger_file, materialize
from tests.mcp_helpers import make_run_handle, write_run_status

# Importing a ``__main__`` module runs the program it guards, so the sweep skips
# those and only those. Every other module is a plain import.
_MAIN_MODULE = "__main__"

# Subclasses whose constructor needs more than a message. Empty today: no
# PhaseSweepError subclass defines its own ``__init__``. Populate it when one
# does, rather than narrowing the walk.
_CONSTRUCTOR_ARGS: dict[str, tuple[object, ...]] = {}

# Subclasses permitted to reach the base ``INSPECT_LOGS`` fallback by
# inheritance instead of declaring an action. Empty, and meant to stay that
# way: the mechanism is kept so a future exemption has to be written down here,
# with a reason, rather than passing unnoticed.
_DECLARATION_PENDING: frozenset[str] = frozenset()

# Classes proving the module sweep reached past the error modules themselves.
# If an import ever stops happening, these vanish from the walk and say so.
_SWEEP_WITNESSES = frozenset(
    {"UnsafeLockPathError", "NoFeasibleTrialError", "RunRecoveryError", "_PolicyStateWriteError"}
)

# A remediation of two required steps, for exercising the multi-step path.
_TWO_STEPS = (OperatorAction.RESTORE_LEDGER, OperatorAction.RESTORE_TREE)

# Every raise that requires more than one remedy step, as (module under src/,
# enclosing function, steps in order). Asserted exactly, like the ledger
# contract's ratchets: a new multi-step raise fails until it is written down
# here, so a second step is a reviewed decision. Alternatives the operator would
# choose between never belong here; their raise routes the one that keeps the
# operator's existing work.
_MULTI_STEP_RAISES = frozenset(
    {
        # Either the ledger's attempt id or the tree's identity files may be the
        # side that changed, and the operator cannot tell which, so both return.
        (
            "phasesweep/engine/attempts.py",
            "_read_trial_process_identity",
            ("RESTORE_LEDGER", "RESTORE_TREE"),
        ),
        # The same split for the lifecycle record a RUNNING trial is resolved by.
        (
            "phasesweep/engine/attempts.py",
            "_attempt_lifecycle_for_reaping",
            ("RESTORE_LEDGER", "RESTORE_TREE"),
        ),
        # With no usable account home, only the override can name a lock
        # directory, and it never creates one: provision it, then point at it.
        ("phasesweep/runtime/files.py", "lock_dir", ("RESTORE_TREE", "FIX_CONFIG")),
    }
)

# The only places an ``action=`` may be computed rather than written out: each
# forwards a remediation some raise already declared, so it cannot add a step.
_ACTION_PASSTHROUGHS = frozenset(
    {
        # rewrap hands the cause's steps, or the caller's explicit ones, to the class.
        ("phasesweep/errors.py", "rewrap"),
        # A preflight aggregate forwards the steps its refusals all share.
        ("phasesweep/engine/guards.py", "_preflight_existing_studies"),
    }
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
        extra = _CONSTRUCTOR_ARGS.get(cls.__name__, ())
        plain = cls("boom", *extra)
        routed = cls("boom", *extra, action=OperatorAction.RETRY)

        # A class default is always exactly one step.
        assert isinstance(cls.default_action, OperatorAction), f"{cls.__name__} has no action"
        assert plain.actions == (cls.default_action,)
        assert routed.actions == (OperatorAction.RETRY,)

        # The action is a routing attribute, never part of what the operator reads.
        assert str(routed) == str(plain)
        assert plain.args == routed.args

        # An instance attribute, so BaseException.__reduce__ carries it in __dict__.
        assert pickle.loads(pickle.dumps(plain)).actions == plain.actions
        both = cls("boom", *extra, action=_TWO_STEPS)
        assert pickle.loads(pickle.dumps(both)).actions == _TWO_STEPS

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
    assert inbound.actions == (OperatorAction.RESTORE_LEDGER,)

    inherited = RunRecoveryError.rewrap(inbound, "y")
    assert isinstance(inherited, RunRecoveryError)
    assert inherited.actions == (OperatorAction.RESTORE_LEDGER,)
    assert str(inherited) == "y"

    overridden = RunRecoveryError.rewrap(inbound, "y", action=OperatorAction.FRESH_NAMESPACE)
    assert overridden.actions == (OperatorAction.FRESH_NAMESPACE,)
    assert str(overridden) == "y"

    # Every step a cause requires survives, in order; none is dropped for the first.
    two_steps = RunRecoveryError.rewrap(ProcessCleanupUncertainError("x", action=_TWO_STEPS), "y")
    assert two_steps.actions == _TWO_STEPS

    # A cause with no action of its own leaves the class default in place.
    foreign = RunRecoveryError.rewrap(OSError("disk"), "y")
    assert foreign.actions == (RunRecoveryError.default_action,)

    # rewrap returns; the caller still writes the `from` clause that links the cause.
    cause = StudyStorageUnavailableError("x")
    with pytest.raises(RunRecoveryError) as excinfo:
        raise RunRecoveryError.rewrap(cause, "y") from cause
    assert excinfo.value.__cause__ is cause
    assert excinfo.value.actions == (OperatorAction.RESTORE_LEDGER,)


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
    assert excinfo.value.actions == (OperatorAction.USE_PRIOR_RELEASE,), label
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
    assert excinfo.value.actions == (OperatorAction.FIX_CONFIG,)
    assert str(excinfo.value) == (
        "W&B evidence requires online logging; remove offline/disabled W&B settings."
    )


# A wrap translates one failure into another at a layer boundary. Each row below
# drives a real wrap site through its real entry point, faulting only the call
# beneath it, and pins what the operator receives: the outbound type, the action
# it routes to, the cause it keeps, and a stable piece of today's message. A row
# whose action differs from its outbound class default is the evidence that the
# site preserves or composes the remediation rather than replacing it.

Trigger = Callable[[Path, pytest.MonkeyPatch], object]


@dataclass(frozen=True)
class WrapCase:
    """One real wrap site and the routing an operator must receive from it."""

    id: str
    #: Drives the real entry point until the wrap under test raises.
    trigger: Trigger
    outbound: type[PhaseSweepError]
    action: OperatorAction
    #: Exact ``__cause__`` type, or ``None`` where the site raises ``from None``.
    cause: type[BaseException] | None
    #: Stable substring of the operator text the site raises today.
    message: str


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
        "SQLite storage is locked by another writer.", action=OperatorAction.RETRY
    )


def _garbage(path: Path) -> None:
    """Overwrite a ledger file with bytes no SQLite reader accepts."""
    path.write_bytes(b"this is not a sqlite database\n" * 64)


def _directory(path: Path) -> None:
    """Replace a ledger file with a directory SQLite cannot open."""
    path.unlink()
    path.mkdir()


def _truncated(path: Path) -> None:
    """Cut the journal's last record short, as an interrupted append leaves it."""
    path.write_bytes(path.read_bytes()[:-8])


def _middle_record_corrupted(path: Path) -> None:
    """Put an undecodable line before the journal's last record, which nothing skips."""
    *earlier, last = path.read_bytes().rstrip(b"\n").split(b"\n")
    path.write_bytes(b"\n".join([*earlier, b"not a journal record", last]) + b"\n")


def _trials_deleted(path: Path) -> None:
    """Drop every trial row while the study keeps its current format stamp."""
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DELETE FROM trials")


def _interrupted_commit_write_protected(path: Path) -> None:
    """Leave a hot journal in a ledger file the owner may no longer write."""
    leave_hot_journal(path)
    path.chmod(0o444)


def _materialize_damaged(
    tmp_path: Path, fixture: str, mode: str, damage: Callable[[Path], None]
) -> Materialized:
    """Copy a current golden ledger and apply ``damage`` to its ledger file."""
    materialized = materialize(fixture, tmp_path, mode=mode)
    damage(ledger_file(materialized, fixture.removeprefix("current-")))
    return materialized


def _recovery_needs(*, ownership_storage_unavailable: bool) -> _RecoveryNeeds:
    """Return recovery decisions that load studies and nothing else."""
    return _RecoveryNeeds(
        terminal_status=None,
        stored_snapshot=None,
        prepared_publication_generation=None,
        cleanup_needed=False,
        terminal_cleanup_uncertain=False,
        ownership_storage_unavailable=ownership_storage_unavailable,
        snapshot_recovery_required=False,
        snapshot_unavailable=False,
        snapshot_finalize_needed=False,
    )


def _dead_uncertain_run(materialized: Materialized, tmp_path: Path) -> tuple[Path, str]:
    """Record a dead, cleanup-uncertain MCP run over a materialized fixture.

    Mirrors ``tests/test_ledger_read_paths.py::_recover_inspect``: the
    handle names a long-dead PID, which is the state recovery acts on.
    """
    state_dir = tmp_path / "mcp-state"
    store = RunStore(state_dir)
    run_id = "wrap-recover"
    config_bytes = materialized.config_path.read_bytes()
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=materialized.experiment.experiment,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(config_bytes)
    store.mark_cleanup_uncertain(handle)
    return state_dir, run_id


def _claim_while(owner: object, name: str, inbound: Callable[[], Exception]) -> Trigger:
    """Claim the current SQLite golden ledger while ``owner.name`` raises."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize("current-sqlite", tmp_path, mode="tree")
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
    fixture: str = "current-sqlite",
) -> Trigger:
    """Load recovery's studies from a current golden ledger after damage."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = _materialize_damaged(tmp_path, fixture, mode, damage)
        needs = _recovery_needs(ownership_storage_unavailable=ownership_storage_unavailable)
        return _load_recovery_studies(materialized.experiment, needs)

    return trigger


def _run_while_discovery_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Start a run over the current golden ledger whose phase study cannot be read."""
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    monkeypatch.setattr(engine_ledger, "_load_existing_phase_study", _raiser(_storage_gone()))
    return run_experiment(materialized.experiment)


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


def _reap_unreadable_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Reap stale trials from a study whose trial list cannot be read."""
    study = optuna.create_study(study_name="t::p")
    monkeypatch.setattr(optuna.Study, "get_trials", _raiser(_storage_gone()))
    return _reap_stale_trials(study, make_experiment(workdir=tmp_path / "runs"), "p")


def _recover_while_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Confirm recovery while another orchestrator holds the experiment lock."""
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    with _experiment_lock(materialized.experiment):
        return recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)


def _recover_over_invalid_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery while the evidence it reads fails publication validation."""
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    monkeypatch.setattr(
        "phasesweep.mcp.recovery._recover_trial_evidence",
        _raiser(PublicationIntegrityError("Published result is invalid.")),
    )
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


def _recover_over_pre_cutover_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover from a state directory the 0.3.1 runtime left without a format marker."""
    state_dir = tmp_path / "mcp-state"
    RunStore(state_dir)
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


def _recover_pending_snapshot_unwritable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Confirm recovery of an orphaned pending snapshot whose status is unsafe to rewrite."""
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    write_run_status(
        RunStore.open_existing(state_dir),
        run_id,
        returncode=1,
        result_snapshot=capture_pre_generation_result_snapshot(materialized.experiment),
        result_snapshot_state="pending",
    )
    monkeypatch.setattr(
        "phasesweep.mcp.recovery.write_status_file",
        _raiser(UnsafePrivatePathError("Private directory is not owner-only.")),
    )
    return recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)


def _finalize_complete_snapshot_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Finalize a complete stored snapshot on a host that cannot rewrite its status safely."""
    terminal_status = {
        "run_id": "wrap-recover",
        "result_snapshot": capture_pre_generation_result_snapshot(make_experiment()),
        "result_snapshot_state": "complete",
    }
    monkeypatch.setattr(
        "phasesweep.mcp.recovery.write_status_file",
        _raiser(PlatformCapabilityError("O_NOFOLLOW is unavailable.")),
    )
    return _finalize_stored_terminal_result_snapshot(
        RunStore(tmp_path / "mcp-state"),
        "wrap-recover",
        terminal_status,
        confirmed_attempt_ids=set(),
        confirmed_attempt_locations={},
    )


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
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    trial_dir = tmp_path / "attempt"
    trial_dir.mkdir()
    _garbage(_registered_entry(materialized.experiment, trial_dir))
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


_NO_NOFOLLOW = "O_NOFOLLOW is unavailable."


def _registry_scan_without_capability(helper: str) -> Trigger:
    """Preflight a registry of one entry while ``helper`` lacks a platform capability."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        experiment = make_experiment(workdir=tmp_path / "runs")
        trial_dir = tmp_path / "attempt"
        trial_dir.mkdir()
        _registered_entry(experiment, trial_dir)
        monkeypatch.setattr(
            f"phasesweep.engine.attempts.{helper}",
            _raiser(PlatformCapabilityError(_NO_NOFOLLOW)),
        )
        return _preflight_active_attempts(experiment, _PreflightCleanupReport())

    return trigger


def _moved_workdir(materialized: Materialized, tmp_path: Path) -> Experiment:
    """Return the fixture's config pointed at a workdir its studies never published into."""
    return materialized.experiment.model_copy(update={"workdir": str(tmp_path / "moved")})


def _recovery_studies_from_moved_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Load recovery's studies through a config whose workdir moved away from them."""
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    needs = _recovery_needs(ownership_storage_unavailable=False)
    return _load_recovery_studies(_moved_workdir(materialized, tmp_path), needs)


def _accepted_target_study(name: str, target: int) -> optuna.Study:
    """Return an empty current-format study that already accepted ``target`` trials."""
    study = optuna.create_study(study_name=name)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    study.set_user_attr(TRIAL_TARGET_ATTR, target)
    return study


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
        if schema_check is not None:
            monkeypatch.setattr(guards, "_validate_study_schema", schema_check)
        return guards._preflight_existing_studies(
            replace(claimed, studies=MappingProxyType(dict(studies())))
        )

    return trigger


def _two_precutover_studies() -> dict[str, optuna.Study]:
    """Return two populated studies that predate the schema stamp."""
    return {"a": _populated_study("t::a"), "b": _populated_study("t::b")}


_RECOVERY_STORAGE = (
    "Restore the original complete storage ledger and access to it, then retry "
    "phasesweep mcp recover-run."
)
_PREFLIGHT_AGGREGATE = "Experiment recovery preflight found multiple unsafe studies: "
_SQLITE_SCAN = "could not be inspected for its PhaseSweep format without mutation."

WRAP_CASES = (
    WrapCase(
        id="claim_ledger_discovery_default",
        trigger=_claim_while(engine_ledger, "_load_existing_phase_study", _storage_gone),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=RuntimeError,
        message="Could not inspect persistent study storage for phase 'p'.",
    ),
    WrapCase(
        id="claim_ledger_discovery_preserves_override",
        trigger=_claim_while(engine_ledger, "_load_existing_phase_study", _ledger_busy),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RETRY,
        cause=StudyStorageUnavailableError,
        message="Could not inspect persistent study storage for phase 'p'.",
    ),
    WrapCase(
        id="published_check_preserves_override",
        trigger=_claim_while(optuna.Study, "get_trials", _ledger_busy),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RETRY,
        cause=StudyStorageUnavailableError,
        message="Could not inspect persistent study storage for published phase 'p'.",
    ),
    WrapCase(
        id="run_ownership_compose",
        trigger=_run_while_discovery_fails,
        outbound=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=StudyStorageUnavailableError,
        message="Artifact ownership could not be checked because required persistent",
    ),
    WrapCase(
        # Negative: unconfirmed cleanup deliberately replaces the run's own remedy.
        id="run_failure_cleanup_replaces",
        trigger=_run_fails_then_cleanup_unconfirmed,
        outbound=ProcessCleanupUncertainError,
        action=OperatorAction.RUN_RECOVER_RUN,
        cause=NoFeasibleTrialError,
        message="The run failed and subsequent process cleanup could not be confirmed.",
    ),
    WrapCase(
        id="recovery_studies_storage_unbound",
        trigger=_recovery_studies_damaged("ledger-only", _garbage),
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=StudyStorageUnavailableError,
        message=_RECOVERY_STORAGE,
    ),
    WrapCase(
        id="recovery_studies_storage_bound",
        trigger=_recovery_studies_damaged("tree", _garbage),
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=StudyStorageUnavailableError,
        message=_RECOVERY_STORAGE,
    ),
    WrapCase(
        # The ledger is intact, so the storage wrap's restore remedy must not
        # replace the confirmed recover-run that lets SQLite roll it back.
        id="recovery_studies_interrupted_transaction",
        trigger=_recovery_studies_damaged("tree", leave_hot_journal),
        outbound=RunRecoveryError,
        action=OperatorAction.RUN_RECOVER_RUN,
        cause=LedgerTransactionInterruptedError,
        message="`phasesweep mcp recover-run --confirm` holds the experiment lock",
    ),
    WrapCase(
        id="recovery_published_missing",
        trigger=_recovery_studies_damaged(
            "tree", _trials_deleted, ownership_storage_unavailable=True
        ),
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=PublishedStudyMissingError,
        message=(
            "Restore the original complete storage ledger and study with access to it, "
            "then retry phasesweep mcp recover-run."
        ),
    ),
    WrapCase(
        id="recover_run_lock_busy",
        trigger=_recover_while_locked,
        outbound=RunRecoveryError,
        action=OperatorAction.RETRY,
        cause=None,
        message="Another phasesweep process appears to be using the same experiment backend",
    ),
    WrapCase(
        id="recover_run_publication_integrity",
        trigger=_recover_over_invalid_publication,
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        cause=None,
        message="Published result is invalid.",
    ),
    WrapCase(
        # Composed: the inbound refusal is a ValueError carrying no action, so the
        # site supplies the one its message names.
        id="recover_run_pre_cutover_state",
        trigger=_recover_over_pre_cutover_state,
        outbound=RunRecoveryError,
        action=OperatorAction.USE_PRIOR_RELEASE,
        cause=None,
        message="use a fresh MCP state directory or the preserved PhaseSweep 0.3.1 runtime",
    ),
    WrapCase(
        # Composed: a path with no run-store layout is a wrong argument, even
        # when it names some other existing, shared directory.
        id="recover_run_mistyped_state_dir",
        trigger=_recover_from_mistyped_state_dir,
        outbound=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        cause=None,
        message="Pass the state_dir from the catalog the MCP server runs with.",
    ),
    WrapCase(
        id="recover_run_shared_state_dir",
        trigger=_recover_from_shared_state_dir,
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        cause=None,
        message="must be owned by uid",
    ),
    WrapCase(
        id="recover_run_pending_snapshot_unsafe_path",
        trigger=_recover_pending_snapshot_unwritable,
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        cause=None,
        message="failed to finalize terminal result snapshot for wrap-recover: "
        "UnsafePrivatePathError",
    ),
    WrapCase(
        id="finalize_snapshot_platform",
        trigger=_finalize_complete_snapshot_unwritable,
        outbound=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        cause=None,
        message="failed to finalize terminal result snapshot for wrap-recover: "
        "PlatformCapabilityError",
    ),
    WrapCase(
        # recover-run reads the same entry, so routing it back to recover-run
        # would loop; the entry itself is what the operator repairs.
        id="recover_run_malformed_registry_entry",
        trigger=_recover_over_malformed_registry_entry,
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        cause=None,
        message="delete the entry file if you are certain nothing is running.",
    ),
    WrapCase(
        id="recovery_studies_root_moved",
        trigger=_recovery_studies_from_moved_workdir,
        outbound=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        cause=ArtifactRootConflictError,
        message="Restore the original workdir, or use a fresh artifact root",
    ),
    WrapCase(
        id="sqlite_unopenable",
        trigger=_validate_damaged("current-sqlite", _directory),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=sqlite3.OperationalError,
        message=_SQLITE_SCAN,
    ),
    WrapCase(
        id="sqlite_garbage",
        trigger=_validate_damaged("current-sqlite", _garbage),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=sqlite3.DatabaseError,
        message=_SQLITE_SCAN,
    ),
    WrapCase(
        # A cut-short last record is skipped on reads, as Optuna skips it; a bad
        # line that another line follows is what a read refuses.
        id="journal_middle_record_corrupt",
        trigger=_validate_damaged("current-journal", _middle_record_corrupted),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=json.JSONDecodeError,
        message="could not be completely read while checking for the PhaseSweep format boundary.",
    ),
    WrapCase(
        # Inspection previews the confirmed write, so it refuses a cut-short
        # last record with the repair, not recover-run's usual restore remedy.
        id="recovery_studies_journal_incomplete_record",
        trigger=_recovery_studies_damaged("tree", _truncated, fixture="current-journal"),
        outbound=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=IncompleteJournalRecordError,
        message="truncate it after its last complete line",
    ),
    WrapCase(
        # recover-run reaps through the same read, so the ledger comes back first.
        id="cleanup_reap_inspect",
        trigger=_reap_unreadable_study,
        outbound=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=RuntimeError,
        message="Could not inspect study 't::p' for stale RUNNING trials.",
    ),
    WrapCase(
        # Not a damaged registry: the missing capability keeps its own routing.
        id="registry_open_platform_capability",
        trigger=_registry_scan_without_capability("open_directory_fd"),
        outbound=PlatformCapabilityError,
        action=OperatorAction.FIX_CONFIG,
        cause=None,
        message=_NO_NOFOLLOW,
    ),
    WrapCase(
        id="registry_entry_platform_capability",
        trigger=_registry_scan_without_capability("read_private_text_at"),
        outbound=PlatformCapabilityError,
        action=OperatorAction.FIX_CONFIG,
        cause=None,
        message=_NO_NOFOLLOW,
    ),
    WrapCase(
        id="preflight_same_type_aggregate",
        trigger=_preflight(_two_precutover_studies),
        outbound=StudySchemaMismatchError,
        action=OperatorAction.USE_PRIOR_RELEASE,
        cause=StudySchemaMismatchError,
        message=_PREFLIGHT_AGGREGATE,
    ),
    WrapCase(
        id="preflight_mixed_aggregate_agreeing",
        trigger=_preflight(_two_precutover_studies, schema_check=_refuse_as_prior_release),
        outbound=PhaseSweepError,
        action=OperatorAction.USE_PRIOR_RELEASE,
        cause=StudySchemaMismatchError,
        message=_PREFLIGHT_AGGREGATE,
    ),
    WrapCase(
        id="preflight_mixed_aggregate_disagreeing",
        trigger=_preflight(
            lambda: {"a": _populated_study("t::a"), "b": _accepted_target_study("t::b", 5)}
        ),
        outbound=PhaseSweepError,
        action=OperatorAction.INSPECT_LOGS,
        cause=StudySchemaMismatchError,
        message=_PREFLIGHT_AGGREGATE,
    ),
)


@pytest.mark.parametrize("case", WRAP_CASES, ids=lambda case: case.id)
def test_operator_action_survives_wrap(
    case: WrapCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(case.outbound) as excinfo:
        case.trigger(tmp_path, monkeypatch)

    error = excinfo.value
    assert type(error) is case.outbound
    assert error.actions == (case.action,)
    if case.cause is None:
        assert error.__cause__ is None
    else:
        assert type(error.__cause__) is case.cause
    assert case.message in str(error)


# An origin raise states its remedy in its own message, so its action has to
# name that same remedy rather than whatever its class defaults to. Each row
# drives the real code path to one such raise and pins the pair together.


@dataclass(frozen=True)
class OriginCase:
    """One raise site whose action must match the remedy its message gives."""

    id: str
    #: Drives the real code path until the raise under test.
    trigger: Trigger
    raised: type[PhaseSweepError]
    #: The one remedy, or every required step in the order the message gives.
    action: OperatorAction | tuple[OperatorAction, ...]
    #: Stable substring of the remedy the message gives today.
    message: str


def _auto_storage_experiment(workdir: Path, n_jobs: int) -> Experiment:
    """Return an auto-storage experiment whose parallelism selects its backend."""
    return make_experiment(
        workdir=workdir, storage="auto", n_jobs=n_jobs, allow_no_gpu_isolation=True
    )


def _validate_after_auto_backend_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Validate a sequential config over a tree its parallel predecessor bound."""
    parallel = _auto_storage_experiment(tmp_path / "runs", n_jobs=2)
    _artifact_root_binding_path(parallel).parent.mkdir(parents=True)
    _write_artifact_root_binding(parallel)
    return engine_ledger.validate_ledger(_auto_storage_experiment(tmp_path / "runs", n_jobs=1))


def _claim_after_unlocked_bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Claim a ledger whose tree another process bound after it was validated."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    validated = engine_ledger.validate_ledger(experiment)
    _artifact_root_binding_path(experiment).parent.mkdir(parents=True)
    _write_artifact_root_binding(experiment)
    return engine_ledger.claim_ledger(validated)


def _preflight_over_shared_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Scan an attempt registry other users can enter."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    registry = _attempts_dir(experiment)
    registry.mkdir(parents=True)
    registry.chmod(0o755)
    return _preflight_active_attempts(experiment, _PreflightCleanupReport())


def _reconcile_prepared_over_damaged_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Reconcile a prepared publication after the last-success target lost its summary."""
    experiment = materialize("current-sqlite", tmp_path, mode="tree").experiment
    published = _last_successful_generation_id(experiment)
    assert published is not None
    _generation_summary_path(experiment, published).unlink()
    needs = replace(
        _recovery_needs(ownership_storage_unavailable=False),
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
        workdir=tmp_path / "runs", storage=f"sqlite:///{tmp_path / 'study.db'}"
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    write_attempt_lifecycle(attempt_dir, attempt_id="attempt", state="allocated")
    _register_active_attempt(
        experiment,
        attempt_id="attempt",
        phase_name="p",
        study_name="t::p",
        trial_number=0,
        trial_dir=attempt_dir,
        generation_id="generation",
    )
    storage = engine_ledger._resolve_storage(experiment.resolved_storage)
    study = optuna.create_study(study_name="t::p", storage=storage)
    study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
    running = study.ask()
    running.set_user_attr(GENERATION_ID_ATTR, "generation")
    running.set_user_attr(TRIAL_DIR_ATTR, str(attempt_dir))
    running.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(running, state=optuna.trial.TrialState.FAIL)
    return _preflight_active_attempts(experiment, _PreflightCleanupReport())


def _inspect_uncertain_trial_without_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Inspect a cleanup-uncertain terminal trial whose ledger lost its attempt id."""
    study = optuna.create_study(study_name="t::p")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    uncertain = study.ask()
    uncertain.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    uncertain.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(uncertain, state=optuna.trial.TrialState.FAIL)
    return _inspect_cleanup_uncertain_trials(study, "p")


def _inspect_uncertain_trial_without_identity_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Inspect a cleanup-uncertain trial whose attempt id no identity files back."""
    study = optuna.create_study(study_name="t::p")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    uncertain = study.ask()
    uncertain.set_user_attr(ATTEMPT_ID_ATTR, "attempt")
    uncertain.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    uncertain.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(uncertain, state=optuna.trial.TrialState.FAIL)
    return _inspect_cleanup_uncertain_trials(study, "p")


def _lock_dir_without_account_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the lock directory for a uid the account database does not know."""
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", _raiser(KeyError("uid")))
    return lock_dir()


def _lock_dir_with_relative_account_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the lock directory for an account whose home is not absolute."""
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir="relative-home"))
    return lock_dir()


def _lock_dir_override_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the lock directory through a relative override."""
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", "relative-locks")
    return lock_dir()


def _home_override_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Resolve the private root through a relative override."""
    monkeypatch.setenv("PHASESWEEP_HOME", "relative-home")
    return phasesweep_home()


def _wandb_sdk_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Check for the W&B SDK in an environment that cannot import it."""
    monkeypatch.setitem(sys.modules, "wandb.apis.public", None)
    return require_wandb_sdk()


def _rewrite_entry(entry: Path, change: Callable[[dict[str, object]], None]) -> None:
    """Apply ``change`` to a registry entry's payload, keeping its owner-only file."""
    payload = json.loads(entry.read_text())
    change(payload)
    entry.write_text(json.dumps(payload))


def _foreign_locator(entry: Path) -> None:
    """Point a registry entry at a storage backend this release does not retain."""
    _rewrite_entry(entry, lambda payload: payload.update(storage_locator="postgresql://h/db"))


def _field_dropped(entry: Path) -> None:
    """Drop a field every current registry entry carries."""
    _rewrite_entry(entry, lambda payload: payload.pop("generation_id"))


def _registry_entry_damaged(damage: Callable[[Path], None]) -> Trigger:
    """Preflight a registry whose one entry, for a real trial directory, was damaged."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        experiment = make_experiment(workdir=tmp_path / "runs")
        trial_dir = tmp_path / "attempt"
        trial_dir.mkdir()
        damage(_registered_entry(experiment, trial_dir))
        return _preflight_active_attempts(experiment, _PreflightCleanupReport())

    return trigger


def _preflight_entry_without_trial_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Preflight a registry entry whose trial directory is gone."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    _registered_entry(experiment, tmp_path / "gone")
    return _preflight_active_attempts(experiment, _PreflightCleanupReport())


def _registered_trial_dir_damaged(damage: Callable[[Path], None]) -> Trigger:
    """Preflight a registry entry whose trial directory's evidence was damaged."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        experiment = make_experiment(workdir=tmp_path / "runs")
        trial_dir = tmp_path / "attempt"
        trial_dir.mkdir()
        _registered_entry(experiment, trial_dir)
        damage(trial_dir)
        return _preflight_active_attempts(experiment, _PreflightCleanupReport())

    return trigger


def _lifecycle_garbled(trial_dir: Path) -> None:
    """Leave an attempt lifecycle record no reader parses."""
    (trial_dir / ATTEMPT_LIFECYCLE_FILE).write_text("{not json")


def _launched_without_identity(trial_dir: Path) -> None:
    """Record a launch whose process identity files never appeared."""
    write_attempt_lifecycle(trial_dir, attempt_id="attempt", state="launching")


def _preflight_unenumerable_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Preflight a private registry whose directory listing fails."""
    experiment = make_experiment(workdir=tmp_path / "runs")
    trial_dir = tmp_path / "attempt"
    trial_dir.mkdir()
    _registered_entry(experiment, trial_dir)
    real_listdir = os.listdir

    def listdir(path: object = ".") -> list[str]:
        if isinstance(path, int):
            raise OSError(errno.EIO, "Input/output error")
        return real_listdir(path)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "listdir", listdir)
    return _preflight_active_attempts(experiment, _PreflightCleanupReport())


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


def _inspect_uncertain_trial_without_trial_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Inspect a cleanup-uncertain terminal trial whose ledger lost its directory."""
    study = optuna.create_study(study_name="t::p")
    uncertain = study.ask()
    uncertain.set_user_attr(ATTEMPT_ID_ATTR, "attempt")
    uncertain.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(uncertain, state=optuna.trial.TrialState.FAIL)
    return _inspect_cleanup_uncertain_trials(study, "p")


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


def _live_uncertain_run(state_dir: Path) -> tuple[RunStore, RunHandle]:
    """Record a cleanup-uncertain run whose runner is this live test process.

    ``make_run_handle`` gives the runner a process group no test owns, so a
    regressed liveness check still cannot signal pytest.
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
        _recovery_needs(ownership_storage_unavailable=False),
        confirm=True,
        earlier_boot=False,
        cleanup_recorded=False,
    )


def _rerun_aborted_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Re-run a phase whose study durably aborted at its current trial target."""
    experiment = materialize("current-sqlite", tmp_path, mode="tree").experiment
    phase = experiment.phases[0]
    study = optuna.load_study(study_name="t::p", storage=experiment.storage)
    study.set_user_attr(
        PHASE_ABORT_ATTR,
        _failure_policy_abort_record(
            phase, consecutive_failures=2, completion_sequence=2, trial_target=phase.n_trials
        ),
    )
    return run_experiment(experiment)


def _republish_as_incomplete(experiment: Experiment) -> None:
    """Reseal the fixture's publication as a partial result an earlier config accepted.

    The winner, the summary entry that mirrors it, the summary's hash of it,
    and the pointer's hash of the summary change together, so the publication
    still validates and the load reaches the partial-result policy.
    """
    generation = _last_successful_generation_id(experiment)
    assert generation is not None
    winner = _generation_winner_path(experiment, generation, "p")
    summary = _generation_summary_path(experiment, generation)
    complete_digest = hashlib.sha256(winner.read_bytes()).hexdigest()
    for path in (winner, summary):
        path.write_text(path.read_text().replace("incomplete: false", "incomplete: true"))
    partial_digest = hashlib.sha256(winner.read_bytes()).hexdigest()
    summary.write_text(summary.read_text().replace(complete_digest, partial_digest))
    pointer_path = _last_successful_generation_path(experiment)
    pointer = yaml.safe_load(pointer_path.read_text())
    pointer["summary_size_bytes"] = len(summary.read_bytes())
    pointer["summary_sha256"] = hashlib.sha256(summary.read_bytes()).hexdigest()
    pointer_path.write_text(yaml.safe_dump(pointer, sort_keys=False))


def _resume_over_incomplete_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Skip a phase whose published winner is a partial result this config does not accept."""
    experiment = materialize("current-sqlite", tmp_path, mode="tree").experiment
    _republish_as_incomplete(experiment)
    return _load_winner(experiment, experiment.phases[0], {})


def _resume_after_phase_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Skip a phase whose search space changed after its winner was published."""
    experiment = materialize("current-sqlite", tmp_path, mode="tree").experiment
    edited = experiment.phases[0].model_copy(
        update={"search_space": {"a": IntParam(type="int", low=0, high=20)}}
    )
    return _load_winner(experiment.model_copy(update={"phases": [edited]}), edited, {})


def _claim_from_moved_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Claim the fixture's ledger through a config whose workdir moved away from it."""
    moved = _moved_workdir(materialize("current-sqlite", tmp_path, mode="tree"), tmp_path)
    return engine_ledger.claim_ledger(engine_ledger.validate_ledger(moved))


def _open_after_interrupted_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Open a phase study through a ledger a crash left mid-commit."""
    materialized = _materialize_damaged(tmp_path, "current-sqlite", "tree", leave_hot_journal)
    experiment = materialized.experiment
    return engine_ledger.open_existing_study(
        engine_ledger.validate_ledger(experiment), experiment.phases[0]
    )


def _claim_after_interrupted_commit_write_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Claim a ledger whose interrupted commit SQLite cannot roll back."""
    materialized = _materialize_damaged(
        tmp_path, "current-sqlite", "tree", _interrupted_commit_write_protected
    )
    return engine_ledger.claim_ledger(engine_ledger.validate_ledger(materialized.experiment))


def _claim_after_torn_journal_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Claim a journal ledger whose last append a crash cut short."""
    materialized = _materialize_damaged(tmp_path, "current-journal", "tree", _truncated)
    return engine_ledger.claim_ledger(engine_ledger.validate_ledger(materialized.experiment))


def _validate_tree_bound_to_another_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Validate a config over a tree that a different storage ledger bound."""
    owner = make_experiment(workdir=tmp_path / "runs", storage=f"sqlite:///{tmp_path / 'a.db'}")
    _artifact_root_binding_path(owner).parent.mkdir(parents=True)
    _write_artifact_root_binding(owner)
    other = make_experiment(workdir=tmp_path / "runs", storage=f"sqlite:///{tmp_path / 'b.db'}")
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


_DELETE_ENTRY_IF_IDLE = "Delete the entry file only if you are certain no process"
_DELETE_ENTRY_IF_NOTHING_RUNS = "Delete the entry file only if you are certain nothing is running."


ORIGIN_CASES = (
    OriginCase(
        id="auto_storage_backend_switch",
        trigger=_validate_after_auto_backend_switch,
        raised=ArtifactRootConflictError,
        action=OperatorAction.FIX_CONFIG,
        message="Restore the previous n_jobs setting to continue this tree",
    ),
    OriginCase(
        id="claim_ledger_tree_changed",
        trigger=_claim_after_unlocked_bind,
        raised=ArtifactRootConflictError,
        action=OperatorAction.RETRY,
        message="Stop that process before retrying.",
    ),
    OriginCase(
        id="attempt_registry_not_private",
        trigger=_preflight_over_shared_registry,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the original registry with mode 0700 before retrying.",
    ),
    OriginCase(
        id="prepared_publication_unreadable",
        trigger=_reconcile_prepared_over_damaged_publication,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the publication evidence or access to it before retrying recovery.",
    ),
    OriginCase(
        id="registry_terminal_identity_missing",
        trigger=_preflight_registered_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore the original storage ledger with its durable attempt and generation",
    ),
    OriginCase(
        id="uncertain_trial_attempt_missing",
        trigger=_inspect_uncertain_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Process identity is unknown. Restore the original storage ledger",
    ),
    OriginCase(
        id="uncertain_trial_identity_files_mismatch",
        trigger=_inspect_uncertain_trial_without_identity_files,
        raised=ProcessCleanupUncertainError,
        action=(OperatorAction.RESTORE_LEDGER, OperatorAction.RESTORE_TREE),
        message="Restore the original storage ledger and this attempt's process-identity files",
    ),
    OriginCase(
        id="lock_dir_no_account_home",
        trigger=_lock_dir_without_account_home,
        raised=UnsafeLockPathError,
        action=(OperatorAction.RESTORE_TREE, OperatorAction.FIX_CONFIG),
        message="provision an absolute lock directory and set PHASESWEEP_LOCK_DIR.",
    ),
    OriginCase(
        id="lock_dir_relative_account_home",
        trigger=_lock_dir_with_relative_account_home,
        raised=UnsafeLockPathError,
        action=(OperatorAction.RESTORE_TREE, OperatorAction.FIX_CONFIG),
        message="provision an absolute lock directory and set PHASESWEEP_LOCK_DIR.",
    ),
    OriginCase(
        id="lock_dir_override_relative",
        trigger=_lock_dir_override_relative,
        raised=UnsafeLockPathError,
        action=OperatorAction.FIX_CONFIG,
        message="PHASESWEEP_LOCK_DIR must be an absolute path",
    ),
    OriginCase(
        id="phasesweep_home_override_relative",
        trigger=_home_override_relative,
        raised=UnsafePrivatePathError,
        action=OperatorAction.FIX_CONFIG,
        message="PHASESWEEP_HOME must be an absolute path",
    ),
    OriginCase(
        id="wandb_sdk_missing",
        trigger=_wandb_sdk_not_installed,
        raised=PhaseSweepError,
        action=OperatorAction.FIX_CONFIG,
        message='python -m pip install "phasesweep[wandb]"',
    ),
    OriginCase(
        id="registry_entry_unparseable",
        trigger=_registry_entry_damaged(_garbage),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="delete the entry file if you are certain nothing is running.",
    ),
    OriginCase(
        id="registry_entry_foreign_locator",
        trigger=_registry_entry_damaged(_foreign_locator),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message=_DELETE_ENTRY_IF_IDLE,
    ),
    OriginCase(
        id="registry_entry_partial_schema",
        trigger=_registry_entry_damaged(_field_dropped),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message=_DELETE_ENTRY_IF_IDLE,
    ),
    OriginCase(
        id="registry_entry_trial_dir_missing",
        trigger=_preflight_entry_without_trial_dir,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message=_DELETE_ENTRY_IF_NOTHING_RUNS,
    ),
    # recover-run runs every cleanup check below in both of its modes, so none
    # of these may route back to it: each names the repair the operator makes.
    OriginCase(
        id="registry_lifecycle_malformed",
        trigger=_registered_trial_dir_damaged(_lifecycle_garbled),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message=_DELETE_ENTRY_IF_NOTHING_RUNS,
    ),
    OriginCase(
        id="registry_identity_missing",
        trigger=_registered_trial_dir_damaged(_launched_without_identity),
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message=_DELETE_ENTRY_IF_NOTHING_RUNS,
    ),
    OriginCase(
        id="registry_unenumerable",
        trigger=_preflight_unenumerable_registry,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_TREE,
        message="Restore the original registry and access to it before retrying.",
    ),
    OriginCase(
        id="running_trial_dir_invalid",
        trigger=_reap_trial_with_invalid_trial_dir,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore the original storage ledger before retrying recovery.",
    ),
    OriginCase(
        id="running_trial_attempt_missing",
        trigger=_reap_trial_without_attempt,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Process identity is unknown. Restore the original storage ledger",
    ),
    OriginCase(
        id="running_trial_lifecycle_mismatch",
        trigger=_reap_trial_with_foreign_lifecycle,
        raised=ProcessCleanupUncertainError,
        action=(OperatorAction.RESTORE_LEDGER, OperatorAction.RESTORE_TREE),
        message="Restore the original storage ledger and this attempt's lifecycle record",
    ),
    OriginCase(
        id="uncertain_trial_dir_missing",
        trigger=_inspect_uncertain_trial_without_trial_dir,
        raised=ProcessCleanupUncertainError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore the original storage ledger before retrying recovery.",
    ),
    OriginCase(
        id="recover_during_launch",
        trigger=_recover_during_launch,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="wait for it to finish and retry",
    ),
    OriginCase(
        id="recover_unsettled_launch",
        trigger=_recover_unsettled_launch,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="Wait briefly and retry",
    ),
    OriginCase(
        id="recover_live_runner",
        trigger=_recover_live_runner,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="runner still appears live; use cancel_run first",
    ),
    OriginCase(
        id="recover_recheck_live_runner",
        trigger=_recheck_live_runner,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="runner still appears live; use cancel_run first",
    ),
    OriginCase(
        id="prior_phase_abort",
        trigger=_rerun_aborted_phase,
        raised=NoFeasibleTrialError,
        action=OperatorAction.FIX_CONFIG,
        message="Increase n_trials above 2 to explicitly schedule new recovery attempts",
    ),
    OriginCase(
        id="winner_incomplete_without_opt_in",
        trigger=_resume_over_incomplete_winner,
        raised=WinnerIntegrityError,
        action=OperatorAction.FIX_CONFIG,
        message="unless the current config sets allow_incomplete_on_timeout: true.",
    ),
    OriginCase(
        id="winner_phase_config_changed",
        trigger=_resume_after_phase_edit,
        raised=StudyFingerprintMismatchError,
        action=OperatorAction.FIX_CONFIG,
        message="or restore the matching config before resuming.",
    ),
    OriginCase(
        id="study_root_workdir_moved",
        trigger=_claim_from_moved_workdir,
        raised=ArtifactRootConflictError,
        action=OperatorAction.FIX_CONFIG,
        message="Restore the original workdir, or use a fresh artifact root",
    ),
    OriginCase(
        id="tree_bound_to_another_ledger",
        trigger=_validate_tree_bound_to_another_ledger,
        raised=ArtifactRootConflictError,
        action=OperatorAction.FIX_CONFIG,
        message="Use the config that owns this current-format tree",
    ),
    OriginCase(
        # Not RETRY: no holder is waited on, and repeating a read never rolls
        # the journal back. The confirmed recover-run is the locked command
        # the one surfacing read, recovery inspection, is a flag away from.
        id="sqlite_transaction_interrupted",
        trigger=_open_after_interrupted_commit,
        raised=LedgerTransactionInterruptedError,
        action=OperatorAction.RUN_RECOVER_RUN,
        message="`phasesweep mcp recover-run --confirm` holds the experiment lock",
    ),
    OriginCase(
        id="journal_final_record_incomplete",
        trigger=_claim_after_torn_journal_append,
        raised=IncompleteJournalRecordError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Once no process uses this journal, truncate it after its last complete line",
    ),
    OriginCase(
        id="sqlite_rollback_refused",
        trigger=_claim_after_interrupted_commit_write_protected,
        raised=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Restore write access to the ledger and its directory before retrying.",
    ),
    OriginCase(
        id="environment_cohort_changed",
        trigger=_top_up_from_another_environment,
        raised=StudyFingerprintMismatchError,
        action=OperatorAction.FIX_CONFIG,
        message="Restore the original semantic environment",
    ),
)


@pytest.mark.parametrize("case", ORIGIN_CASES, ids=lambda case: case.id)
def test_origin_raises_route_by_their_message_remedy(
    case: OriginCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(case.raised) as excinfo:
        case.trigger(tmp_path, monkeypatch)

    error = excinfo.value
    assert type(error) is case.raised
    expected = case.action if isinstance(case.action, tuple) else (case.action,)
    assert error.actions == expected
    assert case.message in str(error)


class _ActionArguments(ast.NodeVisitor):
    """Sort every ``action=`` argument into literal step lists and computed values."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.multi_step: set[tuple[str, str, tuple[str, ...]]] = set()
        self.computed: set[tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        scope = self.scope[-1] if self.scope else "<module>"
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg != "action" or isinstance(value, (ast.Attribute, ast.Constant)):
                # One named step, or an unrelated literal such as argparse's.
                continue
            if isinstance(value, (ast.Tuple, ast.List)):
                steps = tuple(
                    element.attr if isinstance(element, ast.Attribute) else ast.unparse(element)
                    for element in value.elts
                )
                self.multi_step.add((self.module, scope, steps))
            else:
                self.computed.add((self.module, scope))
        self.generic_visit(node)


def test_multi_step_remediations_are_deliberate() -> None:
    """Only allowlisted raises require several steps, each a distinct real action."""
    src = Path(__file__).resolve().parent.parent / "src"
    multi_step: set[tuple[str, str, tuple[str, ...]]] = set()
    computed: set[tuple[str, str]] = set()
    for path in sorted((src / "phasesweep").rglob("*.py")):
        visitor = _ActionArguments(path.relative_to(src).as_posix())
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        multi_step |= visitor.multi_step
        computed |= visitor.computed

    assert multi_step == _MULTI_STEP_RAISES, (
        "a raise now requires several remedy steps; if the operator must do every one, "
        "add it to _MULTI_STEP_RAISES with the reason, and if they are alternatives, "
        "route the one that keeps the operator's existing work. "
        f"Unlisted: {sorted(multi_step - _MULTI_STEP_RAISES)}; "
        f"stale: {sorted(_MULTI_STEP_RAISES - multi_step)}"
    )
    # A computed action would hide its steps from the allowlist above, so only
    # the forwarding sites may compute one; a raise writes its steps out.
    assert computed == _ACTION_PASSTHROUGHS, (
        f"write each raise's steps out literally. Computed at: {sorted(computed)}"
    )
    for _module, _function, steps in multi_step:
        assert len(steps) >= 2, f"a single step is spelled as one action, not {steps}"
        assert len(set(steps)) == len(steps), f"repeated step in {steps}"
        assert set(steps) <= set(OperatorAction.__members__), f"unknown step in {steps}"
