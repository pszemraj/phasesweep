"""Contract tests for the operator action every recover-run refusal carries.

Each case drives a real ``RunRecoveryError`` raise site until it raises, as ``test_error_routing.py``
does for the rest of the package; both modules share their case record and assertions through
``tests/error_routing_helpers.py``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from phasesweep.cli import _operational_error_text
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import _generation_summary_path
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.mcp import errors as mcp_errors
from phasesweep.mcp.recovery import (
    RunRecoveryError,
    _cleanup_runner,
    _finalize_stored_terminal_result_snapshot,
    _load_recovery_studies,
    _publication_recovery_action,
    recover_run,
)
from phasesweep.mcp.runs import _STATE_FORMAT_MARKER_NAME, RunHandle, RunnerView, RunStore
from phasesweep.mcp.snapshots import capture_pre_generation_result_snapshot
from phasesweep.runtime.files import UnsafePrivatePathError
from tests.conftest import make_experiment, reaped_pid, requires_nonroot
from tests.error_routing_helpers import (
    RoutingCase,
    Trigger,
    assert_routes_declared_action,
    assert_runner_payload_follows_routing,
    garbage,
    materialize_damaged,
    raiser,
    read_only_once_locked,
    registered_entry,
    truncated,
)
from tests.ledger_fixtures import Materialized, materialize, published_generation_id
from tests.mcp_helpers import (
    UNUSED_RUNNER_PGID,
    make_run_handle,
    stage_dead_run,
    write_run_status,
)
from tests.recovery_helpers import load_only_recovery_needs

pytestmark = pytest.mark.contract


def _trials_deleted(path: Path) -> None:
    """Drop every trial while the journal keeps its study-level records intact.

    Every trial-scoped journal record (``CREATE_TRIAL`` and later) carries an
    ``op_code`` of 4 or above; study-level records (create study, study user
    attrs) are 0-3. Keeping only the low op codes preserves the study, its
    schema stamp, and its other attributes while deleting every trial.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line).get("op_code", 0) < 4]
    path.write_text("".join(kept), encoding="utf-8")


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


def _recovery_studies_damaged(
    mode: str,
    damage: Callable[[Path], None],
    *,
    ownership_storage_unavailable: bool = False,
    fixture: str = "current-journal",
) -> Trigger:
    """Load recovery's studies from a current golden ledger after damage."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        materialized = materialize_damaged(tmp_path, fixture, mode, damage)
        needs = load_only_recovery_needs(
            ownership_storage_unavailable=ownership_storage_unavailable
        )
        return _load_recovery_studies(materialized.experiment, needs)

    return trigger


def _recovery_studies_confirmed_over_unbackuppable_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Confirm recovery over a torn journal whose repair can lock but not back up."""
    materialized = materialize_damaged(tmp_path, "current-journal", "tree", truncated)
    needs = load_only_recovery_needs()
    with (
        _experiment_lock(materialized.experiment),
        read_only_once_locked(materialized.ledger_path, monkeypatch),
    ):
        return _load_recovery_studies(materialized.experiment, needs, confirm=True)


def _recover_while_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Confirm recovery while another orchestrator holds the experiment lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    with _experiment_lock(materialized.experiment):
        return recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)


def _recover_over_unmarked_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover from a state directory holding run handles but no format marker."""
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
            result_snapshot=capture_pre_generation_result_snapshot(
                materialized.experiment
            ).model_dump(mode="json"),
            result_snapshot_state="pending",
        )
        monkeypatch.setattr(f"phasesweep.mcp.recovery.{failing}", raiser(error))
        return recover_run(state_dir, run_id, confirm=True, emit=lambda _message: None)

    return trigger


def _finalize_complete_snapshot_while(failing: str, error: Exception) -> Trigger:
    """Finalize a complete stored snapshot while ``failing`` raises ``error``."""

    def trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        terminal_status = {
            "run_id": "wrap-recover",
            "result_snapshot": capture_pre_generation_result_snapshot(make_experiment()).model_dump(
                mode="json"
            ),
            "result_snapshot_state": "complete",
        }
        monkeypatch.setattr(f"phasesweep.mcp.recovery.{failing}", raiser(error))
        return _finalize_stored_terminal_result_snapshot(
            RunStore(tmp_path / "mcp-state"),
            "wrap-recover",
            terminal_status,
            confirmed_attempt_ids=set(),
            confirmed_attempt_locations={},
        )

    return trigger


def _recover_over_malformed_registry_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Inspect recovery while an attempt registry entry no longer parses."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    trial_dir = tmp_path / "attempt"
    trial_dir.mkdir()
    garbage(registered_entry(materialized.experiment, trial_dir))
    return recover_run(state_dir, run_id, confirm=False, emit=lambda _message: None)


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


def _reconcile_prepared_over_damaged_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Reconcile a prepared publication after the last-success target lost its summary."""
    experiment = materialize("current-journal", tmp_path, mode="tree").experiment
    published = published_generation_id(experiment)
    assert published is not None
    _generation_summary_path(experiment, published).unlink()
    needs = replace(
        load_only_recovery_needs(),
        prepared_publication_generation="prepared-generation",
    )
    return _publication_recovery_action(experiment, needs)


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
        handle.run_id,
        load_only_recovery_needs(),
        confirm=True,
        view=RunnerView("observable"),
        cleanup_recorded=False,
    )


def _recheck_runner_group_that_survives_the_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    """Re-check an exited runner whose process group the kill could not confirm gone."""
    store = RunStore(tmp_path / "mcp-state")
    handle = make_run_handle(run_id="wrap-recover", pid=reaped_pid())
    store.create(handle)
    store.mark_cleanup_uncertain(handle)
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *_a, **_k: False)
    return _cleanup_runner(
        store.cleanup_identity(handle),
        handle.run_id,
        replace(load_only_recovery_needs(), cleanup_needed=True),
        confirm=True,
        view=RunnerView("observable"),
        cleanup_recorded=False,
    )


def _recover_run_from_foreign_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Recover a run whose runner was recorded in a PID namespace this process cannot observe."""
    state_dir = tmp_path / "mcp-state"
    store = RunStore(state_dir)
    handle = make_run_handle(run_id="wrap-recover", pid_namespace="pid:[4026539001]@100")
    store.create(handle)
    store.mark_cleanup_uncertain(handle)
    monkeypatch.setattr(
        "phasesweep.runtime.reaper.read_pid_namespace", lambda: "pid:[4026539333]@100"
    )
    return recover_run(state_dir, "wrap-recover", confirm=False, emit=lambda _message: None)


CASES = (
    RoutingCase(
        id="recovery_studies_storage_bound",
        trigger=_recovery_studies_damaged("tree", garbage),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Retry phasesweep mcp recover-run after repairing the storage ledger.",
    ),
    RoutingCase(
        id="recovery_published_missing",
        trigger=_recovery_studies_damaged(
            "tree", _trials_deleted, ownership_storage_unavailable=True
        ),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        message="Retry phasesweep mcp recover-run after repairing the storage ledger.",
    ),
    RoutingCase(
        id="recover_run_lock_busy",
        trigger=_recover_while_locked,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="Another phasesweep process holds the lock for experiment",
    ),
    RoutingCase(
        # Composed: the inbound refusal is a ValueError carrying no action, so the
        # site supplies the one its message names.
        id="recover_run_unmarked_state",
        trigger=_recover_over_unmarked_state,
        raised=RunRecoveryError,
        action=OperatorAction.FRESH_NAMESPACE,
        message="durable run data has no format marker",
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
        id="finalize_snapshot_unsafe_path",
        trigger=_finalize_complete_snapshot_while(
            "write_status_file", UnsafePrivatePathError("Private directory is not owner-only.")
        ),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="failed to finalize terminal result snapshot for wrap-recover: "
        "UnsafePrivatePathError",
    ),
    RoutingCase(
        # recover-run reads the same entry, so routing it back to recover-run
        # would loop; the entry itself is what the operator repairs.
        id="recover_run_malformed_registry_entry",
        trigger=_recover_over_malformed_registry_entry,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="Read pgid from the process_identity.json in the directory under",
    ),
    RoutingCase(
        # Composed: the claim path routes this refusal to fixing the config,
        # but recovery's config is pinned, so only the tree can be restored.
        id="recovery_studies_root_moved",
        trigger=_recover_through_retargeted_workdir,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="publishes into artifact root",
    ),
    RoutingCase(
        # A confirmed recovery repairs a cut-short last record itself; its own
        # refusal names the remedy, not recover-run's usual restore-ledger text.
        id="recovery_studies_journal_incomplete_record",
        trigger=_recovery_studies_confirmed_over_unbackuppable_repair,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_LEDGER,
        message="could not be repaired",
        marks=(requires_nonroot,),
    ),
    RoutingCase(
        id="prepared_publication_unreadable",
        trigger=_reconcile_prepared_over_damaged_publication,
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="last-success publication is invalid or unreadable.",
    ),
    RoutingCase(
        id="recover_during_launch",
        trigger=_recover_during_launch,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="another MCP launch is in progress",
    ),
    RoutingCase(
        id="recover_unsettled_launch",
        trigger=_recover_unsettled_launch,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="The run stays reserved",
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
        message="run config snapshot is missing:",
    ),
    RoutingCase(
        id="recover_config_snapshot_unreadable",
        trigger=_recover_with_config_snapshot(lambda path: path.chmod(0o000)),
        raised=RunRecoveryError,
        action=OperatorAction.RESTORE_TREE,
        message="cannot read run config snapshot:",
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
        message="runner still appears live",
    ),
    RoutingCase(
        id="recover_recheck_live_runner",
        trigger=_recheck_live_runner,
        raised=RunRecoveryError,
        action=OperatorAction.RETRY,
        message="runner still appears live",
    ),
    RoutingCase(
        id="recover_runner_group_survives_the_kill",
        trigger=_recheck_runner_group_that_survives_the_kill,
        raised=RunRecoveryError,
        action=OperatorAction.INSPECT_LOGS,
        message="runner process-group cleanup is still uncertain",
    ),
    RoutingCase(
        id="recover_runner_in_foreign_pid_namespace",
        trigger=_recover_run_from_foreign_namespace,
        raised=RunRecoveryError,
        action=OperatorAction.FIX_CONFIG,
        message="only the host's initial PID namespace can see another namespace's processes",
    ),
)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
def test_raise_sites_route_their_declared_action(
    case: RoutingCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert_routes_declared_action(case, tmp_path, monkeypatch, caplog)


def _defect_while_inspecting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Inspect recovery while one of its own steps has a bug."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    state_dir, run_id = _dead_uncertain_run(materialized, tmp_path)
    monkeypatch.setattr(
        "phasesweep.mcp.recovery._publication_recovery_action",
        raiser(NotImplementedError("unfinished branch")),
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


@pytest.mark.parametrize(
    ("trigger", "runner_is_live"),
    [
        (_recover_live_runner, True),
        (_recheck_live_runner, True),
        (_recheck_runner_group_that_survives_the_kill, False),
    ],
    ids=["inspection", "recheck", "group_survives_the_kill"],
)
@pytest.mark.parametrize("initial_namespace", [True, False], ids=["host", "container"])
def test_surviving_runner_refusal_names_the_runner_and_the_host_step(
    trigger: Trigger,
    runner_is_live: bool,
    initial_namespace: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusing a runner that survived names it as cancellation does, by PID and process group in the host's namespace and by run id elsewhere, with the step cancellation ends with."""
    monkeypatch.setattr(mcp_errors, "in_initial_pid_namespace", lambda: initial_namespace)
    with pytest.raises(RunRecoveryError) as excinfo:
        trigger(tmp_path, monkeypatch)

    step = excinfo.value.remediation
    assert step is not None
    if initial_namespace:
        assert re.search(r"runner PID \d+ \(process group \d+\)", step)
        if runner_is_live:
            assert f"runner PID {os.getpid()} (process group {UNUSED_RUNNER_PGID})" in step
    else:
        assert "`runner.py --run-id wrap-recover` line of `pgrep -af wrap-recover`" in step
        assert "runner PID" not in step
    assert ("cancel_run" in step) is runner_is_live
    assert "is still running on the host, clear what blocks SIGKILL" in step
    assert "uninterruptible I/O ignores it" in step
    assert step in _operational_error_text(excinfo.value)


def test_runner_payload_follows_the_routed_steps() -> None:
    assert_runner_payload_follows_routing(
        [case.raised("boom", action=case.action) for case in CASES]
    )
