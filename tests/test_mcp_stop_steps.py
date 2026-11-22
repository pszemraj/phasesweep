"""Every MCP response that requires operator recovery carries the operator's step."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from phasesweep.mcp.errors import (
    NEW_RUN_RESUMES,
    ConcurrencyLimitError,
    ExperimentBusyError,
    McpToolError,
    RunCapacityUnknownError,
    RunNamespaceMismatchError,
    check_catalog_command,
)
from phasesweep.mcp.failures import response_failure
from phasesweep.mcp.runner import FailurePayload, _cleanup_uncertain_payload
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime import reaper
from tests.mcp_helpers import (
    earlier_boot_id,
    make_dead_run_handle,
    record_confirmed_recovery,
    side_effect_app,
    stage_dead_run,
    stage_launching_run,
    stage_live_run,
    write_run_status,
    write_runner_heartbeat,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import recover_run_cli

RUN_ID = "srv-needs-recovery"
#: The operator's recover-run command for this run, as every agent-facing step spells it.
FULL_RECOVER_RUN = f"`phasesweep mcp recover-run --state-dir STATE_DIR --run-id {RUN_ID}`"
#: A mention of recover-run that leaves out the flags the command requires.
BARE_RECOVER_RUN = re.compile(r"phasesweep mcp recover-run(?! --state-dir STATE_DIR --run-id )")
#: A PID namespace the reader is not in, and a nested reader that cannot prove its processes gone.
RECORDED_NAMESPACE = "pid:[4026539001]@100"
NESTED_READER = "pid:[4026539333]@100"

Stage = Callable[[Path, PhaseSweepMCP, RunStore, pytest.MonkeyPatch], None]


def _launch_unresolved(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server saved the launch handle but never recorded whether its runner started."""
    stage_launching_run(store, RUN_ID, config)


def _launch_failure_record(
    config: Path, app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed launch whose runner group could not be confirmed gone, as the server records it."""
    pending = stage_launching_run(store, RUN_ID, config)
    store.mark_cleanup_uncertain(
        make_dead_run_handle(
            run_id=RUN_ID,
            experiment_id=pending.experiment_id,
            config_sha256=pending.config_sha256,
            config_dir=pending.config_dir,
        )
    )
    app._record_launch_failure(pending, cleanup_confirmed=False, error_class="RuntimeError")


def _runner_exited(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner exited without writing its final status."""
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False)


def _cleanup_marker_dead_runner(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup reservation outlives a runner that wrote no final status."""
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=True)


def _cleanup_marker_live_runner(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancellation reserved cleanup and never recorded its outcome; the runner still lives."""
    store.mark_cleanup_uncertain(stage_live_run(store, RUN_ID, config, "srv"))


def _runner_cleanup_unconfirmed(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner recorded that it could not confirm trial cleanup."""
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False)
    write_unsafe_cleanup_status(
        store, RUN_ID, failure=_cleanup_uncertain_payload([], run_id=RUN_ID)
    )


def _upload_cleanup_unconfirmed(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """A succeeded run whose upload worker's cleanup is unconfirmed."""
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False)
    write_run_status(
        store, RUN_ID, returncode=0, cleanup_confirmed=False, result_snapshot_state="failed"
    )


def _snapshot_finalization(
    config: Path, _app: PhaseSweepMCP, store: RunStore, _monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner died after recording its outcome but before finalizing its result snapshot."""
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False)
    write_run_status(store, RUN_ID, returncode=0, result_snapshot_state="pending")


def _stale_heartbeat(
    config: Path, _app: PhaseSweepMCP, store: RunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner in another PID namespace stopped refreshing its heartbeat."""
    monkeypatch.setattr(reaper, "read_pid_namespace", lambda: NESTED_READER)
    stage_dead_run(
        store, RUN_ID, config, "srv", cleanup_uncertain=False, pid_namespace=RECORDED_NAMESPACE
    )
    write_runner_heartbeat(store, RUN_ID, age_seconds=600.0)


CAUSES: dict[str, Stage] = {
    "launch-unresolved": _launch_unresolved,
    "launch-failure-record": _launch_failure_record,
    "runner-exited": _runner_exited,
    "cleanup-marker-dead-runner": _cleanup_marker_dead_runner,
    "cleanup-marker-live-runner": _cleanup_marker_live_runner,
    "runner-cleanup-unconfirmed": _runner_cleanup_unconfirmed,
    "upload-cleanup-unconfirmed": _upload_cleanup_unconfirmed,
    "snapshot-finalization": _snapshot_finalization,
    "stale-heartbeat": _stale_heartbeat,
}


@pytest.mark.parametrize("read_tool", ["get_run_status", "await_run", "get_latest_run"])
@pytest.mark.parametrize("cause", CAUSES)
def test_a_run_that_requires_recovery_names_the_operators_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str, read_tool: str
) -> None:
    """Every read that reports recovery_required true carries a failure naming recover-run for the operator, and so does get_run_results."""
    config, app, _registry, store = side_effect_app(tmp_path)
    CAUSES[cause](config, app, store, monkeypatch)
    monkeypatch.setattr("phasesweep.mcp.tools.AWAIT_MIN_TIMEOUT_SECONDS", 0)

    if read_tool == "get_run_status":
        run = app.status(run_id=RUN_ID)["run"]
    elif read_tool == "await_run":
        awaited = asyncio.run(app.await_run(RUN_ID, timeout_seconds=0))
        assert awaited["reason"] == "recovery_required"
        run = awaited["run"]
    else:
        run = app.latest_run("srv")["run"]
    results_failure = app.winners(run_id=RUN_ID)["failure"]

    assert run["recovery_required"] is True
    for failure in (run["failure"], results_failure):
        assert failure is not None
        assert failure["actor"] == "operator"
        assert FULL_RECOVER_RUN in failure["remediation"]
        assert BARE_RECOVER_RUN.search(failure["remediation"]) is None


@pytest.mark.parametrize(
    "refusal",
    [
        ExperimentBusyError("srv", RUN_ID, recovery_required=True),
        ExperimentBusyError("srv", RUN_ID, recovery_required=True, runner_observable=False),
        ExperimentBusyError("srv", RUN_ID, runner_observable=False),
        ConcurrencyLimitError(1, 1, [RUN_ID], recovery_required_run_ids=[RUN_ID]),
        ConcurrencyLimitError(1, 1, [RUN_ID], unobservable_run_ids=[RUN_ID]),
        RunCapacityUnknownError({f"run:{RUN_ID}"}, [RUN_ID]),
        RunNamespaceMismatchError(RUN_ID),
    ],
    ids=lambda refusal: type(refusal).__name__,
)
def test_every_refusal_that_names_recover_run_gives_the_whole_command(
    refusal: McpToolError,
) -> None:
    """A refusal that sends the operator to recover-run spells out its state-dir and run-id flags at every mention."""
    text = str(refusal)

    assert "phasesweep mcp recover-run --state-dir STATE_DIR --run-id " in text
    assert "STATE_DIR is the catalog's `state_dir`" in text
    assert BARE_RECOVER_RUN.search(text) is None


def test_a_blocker_in_another_namespace_ends_with_a_whole_recover_run_step() -> None:
    """The capacity refusal for a blocker in another PID namespace ends with the step, with no dangling "for it"."""
    text = str(ConcurrencyLimitError(1, 1, [RUN_ID], unobservable_run_ids=[RUN_ID]))

    assert text.endswith("then repeat it with `--confirm`.")
    assert " for it" not in text


def test_the_operators_catalog_check_command_quotes_a_path_for_the_shell() -> None:
    """A catalog path with a space is one word of the command the operator pastes."""
    assert (
        check_catalog_command("/etc/my catalog.yaml")
        == "phasesweep mcp check --catalog '/etc/my catalog.yaml'"
    )
    assert check_catalog_command() == "phasesweep mcp check --catalog CATALOG"


def test_a_recovered_cleanup_failure_without_a_snapshot_keeps_one_level_of_cause(
    tmp_path: Path,
) -> None:
    """A recorded cleanup failure with its own cause, settled by recovery and left without a snapshot, reads as a schema-valid failure whose cause holds no cause."""
    config, app, _registry, store = side_effect_app(tmp_path)
    handle = stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False)
    recorded = {
        **_cleanup_uncertain_payload([], run_id=RUN_ID),
        "cause": {
            "code": "storage_unavailable",
            "stage": "execution",
            "retryable": False,
            "actor": "operator",
            "remediation": "Ask the operator to restore the storage ledger.",
        },
    }
    write_unsafe_cleanup_status(store, RUN_ID, failure=recorded, result_snapshot_state="failed")
    record_confirmed_recovery(store, handle)

    failure = response_failure(store, handle, view=store.runner_view(handle))
    read_failure = app.status(run_id=RUN_ID)["run"]["failure"]

    assert failure is not None
    FailurePayload.model_validate(failure)
    assert failure == read_failure
    assert failure["code"] == "result_snapshot_unavailable"
    assert failure["cause"]["code"] == "cleanup_uncertain"
    assert "cause" not in failure["cause"]


@pytest.mark.parametrize("ended", ["recovered", "earlier-boot"])
def test_a_run_that_ended_without_status_says_a_new_run_resumes(tmp_path: Path, ended: str) -> None:
    """A runner that ended without a final status, once recovered or from an earlier boot, reads as a run that recorded no results and whose next run keeps the finished trials."""
    config, app, _registry, store = side_effect_app(tmp_path)
    boot_id = earlier_boot_id() if ended == "earlier-boot" else None
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False, boot_id=boot_id)
    if ended == "recovered":
        # The terminal status recover-run writes for a runner that exited without one.
        write_run_status(
            store,
            RUN_ID,
            returncode=1,
            error_class="RunnerExitedWithoutStatus",
            result_snapshot_state="failed",
        )

    run = app.status(run_id=RUN_ID)["run"]

    assert (run["state"], run["recovery_required"]) == ("failed", False)
    assert run["failure"]["code"] == "result_snapshot_unavailable"
    remediation = run["failure"]["remediation"]
    assert "This run recorded no results" in remediation
    assert NEW_RUN_RESUMES in remediation
    assert "substitute" not in remediation


def test_a_cancel_that_leaves_recovery_names_the_operators_step(tmp_path: Path) -> None:
    """A cancellation that confirms cleanup but leaves the result snapshot unfinished names recover-run."""
    config, app, _registry, store = side_effect_app(tmp_path)
    stage_dead_run(store, RUN_ID, config, "srv", cleanup_uncertain=False, allow_cancel=True)
    write_run_status(store, RUN_ID, returncode=143, result_snapshot_state="pending")

    result = app.cancel(RUN_ID)

    assert (result["cleanup_confirmed"], result["recovery_required"]) == (True, True)
    assert result["failure"]["code"] == "result_snapshot_unavailable"
    assert FULL_RECOVER_RUN in result["failure"]["remediation"]


def test_recover_run_tells_a_wrong_state_dir_how_a_relative_one_resolves(tmp_path: Path) -> None:
    """A path that holds no state directory gets a fix line about the catalog's relative ``state_dir``."""
    result = recover_run_cli(tmp_path / "not-state", RUN_ID)

    assert result.exit_code != 0
    assert "not an existing MCP state directory" in result.output
    assert (
        "fix: Pass the catalog's `state_dir` as an absolute path (a relative one resolves "
        "against the catalog's directory)."
    ) in " ".join(result.output.split())
