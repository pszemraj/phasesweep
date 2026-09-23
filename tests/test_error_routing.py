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

The wrap table near the end carries the same contract across layer boundaries: a
remediation survives each translation unless the site deliberately composes or
replaces it, and every such site is driven for real rather than in isolation.
The origin table after it holds raises whose message names its own remedy to the
action that routes it.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import pickle
import pkgutil
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import optuna
import pytest

import phasesweep
import phasesweep.engine.ledger as engine_ledger
from phasesweep.config import Experiment, IntParam, Phase, WandbSummaryRequiredGate
from phasesweep.engine import (
    ArtifactRootConflictError,
    NoFeasibleTrialError,
    ProcessCleanupUncertainError,
    PublicationIntegrityError,
    PublishedStudyMissingError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    fingerprints,
    guards,
    run_experiment,
    study_policy,
    trial,
)
from phasesweep.engine.artifact_roots import _write_artifact_root_binding
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
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.state import (
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRIAL_DIR_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.mcp.recovery import (
    RunRecoveryError,
    _finalize_stored_terminal_result_snapshot,
    _load_recovery_studies,
    _publication_recovery_action,
    _RecoveryNeeds,
    recover_run,
)
from phasesweep.mcp.runs import _STATE_FORMAT_MARKER_NAME, RunStore
from phasesweep.mcp.snapshots import capture_pre_generation_result_snapshot
from phasesweep.runtime.files import PlatformCapabilityError, UnsafePrivatePathError
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import make_experiment
from tests.ledger_fixtures import Materialized, ledger_file, materialize
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


def _trials_deleted(path: Path) -> None:
    """Drop every trial row while the study keeps its current format stamp."""
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DELETE FROM trials")


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

    Mirrors ``tests/test_ledger_read_paths.py::_recover_inspect_verdict``: the
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
    mode: str, damage: Callable[[Path], None], *, ownership_storage_unavailable: bool = False
) -> Trigger:
    """Load recovery's studies from the current SQLite golden ledger after damage."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = _materialize_damaged(tmp_path, "current-sqlite", mode, damage)
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
        id="journal_truncated",
        trigger=_validate_damaged("current-journal", _truncated),
        outbound=StudyStorageUnavailableError,
        action=OperatorAction.RESTORE_LEDGER,
        cause=ValueError,
        message="could not be completely read while checking for the PhaseSweep format boundary.",
    ),
    WrapCase(
        id="cleanup_reap_inspect",
        trigger=_reap_unreadable_study,
        outbound=ProcessCleanupUncertainError,
        action=OperatorAction.RUN_RECOVER_RUN,
        cause=RuntimeError,
        message="Could not inspect study 't::p' for stale RUNNING trials.",
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
    assert error.action is case.action
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
    action: OperatorAction
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
)


@pytest.mark.parametrize("case", ORIGIN_CASES, ids=lambda case: case.id)
def test_origin_raises_route_by_their_message_remedy(
    case: OriginCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(case.raised) as excinfo:
        case.trigger(tmp_path, monkeypatch)

    error = excinfo.value
    assert type(error) is case.raised
    assert error.action is case.action
    assert case.message in str(error)
