"""Trial launch environment behavior."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from phasesweep.config import ExecutionContext
from phasesweep.engine.trial import TrialExecutionError, _environment_identity, launch_trial
from phasesweep.runtime.process import ProcessResult
from tests.conftest import make_experiment


def _capture_launch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_env: dict[str, str] | None = None,
    execution: ExecutionContext | None = None,
    gpu_id: int | str | None = 2,
) -> dict[str, str]:
    captured: dict[str, str] = {}

    def fake_run_supervised(
        _cmd: str,
        *,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        timeout: float | None,
        trial_dir: Path,
        attempt_id: str,
        cwd: str | None = None,
    ) -> ProcessResult:
        captured.update(env)
        captured["run_supervised_attempt_id"] = attempt_id
        if cwd is not None:
            captured["run_supervised_cwd"] = cwd
        return ProcessResult(
            return_code=0,
            timed_out=False,
            pid=12345,
            duration_seconds=0.0,
        )

    monkeypatch.setattr("phasesweep.engine.trial.run_supervised", fake_run_supervised)
    launch_trial(
        experiment=make_experiment(env=experiment_env, execution=execution),
        phase_name="p",
        trial_id=0,
        generation_id="generation-test",
        attempt_id="attempt-test",
        trial_dir=tmp_path / "trial_0",
        overrides={},
        timeout_seconds=None,
        gpu_id=gpu_id,
    )
    return captured


@pytest.mark.parametrize(
    ("experiment_env", "gpu_id", "expected_visible", "expected_order"),
    [
        pytest.param(None, 2, "2", "PCI_BUS_ID", id="numeric-default-order"),
        pytest.param(
            {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
            2,
            "2",
            "FASTEST_FIRST",
            id="operator-device-order",
        ),
        pytest.param(
            None,
            "MIG-GPU-deadbeef/3/0",
            "MIG-GPU-deadbeef/3/0",
            "PCI_BUS_ID",
            id="opaque-mig-token",
        ),
    ],
)
def test_launch_trial_cuda_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_env: dict[str, str] | None,
    gpu_id: int | str,
    expected_visible: str,
    expected_order: str,
) -> None:
    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env=experiment_env,
        gpu_id=gpu_id,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == expected_visible
    assert env["CUDA_DEVICE_ORDER"] == expected_order
    assert env["PHASESWEEP_GENERATION_ID"] == "generation-test"
    assert env["PHASESWEEP_ATTEMPT_ID"] == "attempt-test"
    assert env["PHASESWEEP_OVERRIDES_SHA256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert env["PHASESWEEP_RUN_NAME"].endswith("-attempt-test")
    assert env["WANDB_RUN_ID"] == "attempt-test"
    assert env["run_supervised_attempt_id"] == "attempt-test"


def test_launch_trial_inherit_env_all_passes_ambient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default contract preserves historical full-environment inheritance."""
    monkeypatch.setenv("PHASESWEEP_TEST_AMBIENT_SECRET", "leak")
    env = _capture_launch_env(tmp_path, monkeypatch)
    assert env["PHASESWEEP_TEST_AMBIENT_SECRET"] == "leak"


def test_launch_trial_inherit_env_none_filters_ambient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``inherit_env: none`` drops ambient vars while keeping the base set and
    configured ``env`` entries (review v0.5.17 / blocker 4)."""
    monkeypatch.setenv("PHASESWEEP_TEST_AMBIENT_SECRET", "leak")
    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env={"KEEP_ME": "explicit"},
        execution=ExecutionContext(inherit_env="none"),
    )
    assert "PHASESWEEP_TEST_AMBIENT_SECRET" not in env
    assert env["KEEP_ME"] == "explicit"
    assert env["PATH"] == os.environ["PATH"]


@pytest.mark.parametrize("ambient_visibility", ["", "-1", "3"])
def test_launch_trial_narrow_env_drops_ambient_cuda_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_visibility: str,
) -> None:
    """Ambient GPU visibility is an unfingerprinted semantic input under a strict contract.

    Two operators running the identical config from differently-exported
    shells must not write CPU-trained and GPU-trained evaluations into one
    study under one semantic fingerprint. Note the composed path never leaves
    a disable sentinel here with ``gpu_id=None``: the GPU pool pins sentinels
    and passes them as the assigned token (next test).
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ambient_visibility)

    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        execution=ExecutionContext(inherit_env="none"),
        gpu_id=None,
    )

    assert "CUDA_VISIBLE_DEVICES" not in env


@pytest.mark.parametrize("sentinel", ["", "-1"])
def test_launch_trial_pool_pinned_sentinel_binds_visibility_under_strict_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sentinel: str,
) -> None:
    """A pool-pinned disable sentinel reaches the trainer despite ``inherit_env: none``.

    The pool leases nothing *because* the sentinel says CUDA is off; handing
    the trial an unbound environment instead would silently re-expose every
    host GPU with zero host locks held.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", sentinel)

    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        execution=ExecutionContext(inherit_env="none"),
        gpu_id=sentinel,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == sentinel


def test_launch_trial_inherit_env_all_still_passes_cuda_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default permissive contract keeps forwarding operator GPU selection."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    env = _capture_launch_env(tmp_path, monkeypatch, gpu_id=None)

    assert env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_launch_trial_inherit_env_list_can_opt_into_cuda_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the variable in the fingerprinted contract admits it explicitly."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        execution=ExecutionContext(inherit_env=["CUDA_VISIBLE_DEVICES"]),
        gpu_id=None,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_launch_trial_configured_env_binds_cuda_visibility_under_strict_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fingerprinted escape hatch beats the ambient value it replaces."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env={"CUDA_VISIBLE_DEVICES": "-1"},
        execution=ExecutionContext(inherit_env="none"),
        gpu_id=None,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_launch_trial_warns_once_when_ambient_visibility_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Dropping ambient visibility with no device leased is reported, not silent."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr("phasesweep.engine.trial._DROPPED_CUDA_VISIBILITY_WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        for _ in range(2):
            _capture_launch_env(
                tmp_path,
                monkeypatch,
                execution=ExecutionContext(inherit_env="none"),
                gpu_id=None,
            )

    warnings = [r for r in caplog.records if "ambient CUDA_VISIBLE_DEVICES" in r.message]
    assert len(warnings) == 1
    assert "env.CUDA_VISIBLE_DEVICES" in warnings[0].getMessage()


def test_launch_trial_no_visibility_warning_when_a_device_is_leased(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An assigned device already binds visibility and holds a lock."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr("phasesweep.engine.trial._DROPPED_CUDA_VISIBILITY_WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        env = _capture_launch_env(
            tmp_path,
            monkeypatch,
            execution=ExecutionContext(inherit_env="none"),
            gpu_id=2,
        )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert not [r for r in caplog.records if "ambient CUDA_VISIBLE_DEVICES" in r.message]


def test_launch_trial_inherit_env_list_adds_exactly_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHASESWEEP_TEST_TOKEN", "tok")
    monkeypatch.setenv("PHASESWEEP_TEST_AMBIENT_SECRET", "leak")
    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        execution=ExecutionContext(inherit_env=["PHASESWEEP_TEST_TOKEN"]),
    )
    assert env["PHASESWEEP_TEST_TOKEN"] == "tok"
    assert "PHASESWEEP_TEST_AMBIENT_SECRET" not in env


def test_launch_trial_execution_cwd_resolved_and_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer_cwd = tmp_path / "trainer_home"
    trainer_cwd.mkdir()
    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        execution=ExecutionContext(cwd=str(trainer_cwd)),
    )
    assert env["run_supervised_cwd"] == str(trainer_cwd.resolve())


def test_launch_trial_default_leaves_cwd_unbound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _capture_launch_env(tmp_path, monkeypatch)
    assert "run_supervised_cwd" not in env


def test_launch_trial_missing_execution_cwd_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(TrialExecutionError, match=r"execution\.cwd"):
        _capture_launch_env(
            tmp_path,
            monkeypatch,
            execution=ExecutionContext(cwd=str(tmp_path / "does_not_exist")),
        )


def test_environment_identity_digest_ignores_mapping_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest identifies the environment's content, not its insertion order."""
    monkeypatch.delenv("PHASESWEEP_TEST_AMBIENT_SECRET", raising=False)
    forward = _environment_identity(
        make_experiment(
            env={"ALPHA": "1", "BETA": "2"},
            execution=ExecutionContext(inherit_env="none"),
        )
    )
    reversed_order = _environment_identity(
        make_experiment(
            env={"BETA": "2", "ALPHA": "1"},
            execution=ExecutionContext(inherit_env="none"),
        )
    )

    assert forward.digest == reversed_order.digest
    assert len(forward.digest) == 64


def test_environment_identity_digest_tracks_values_and_name_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different values — and differently-split name/value pairs — are different environments."""
    monkeypatch.delenv("PHASESWEEP_TEST_AMBIENT_SECRET", raising=False)

    def identity_for(env: dict[str, str]) -> str:
        return _environment_identity(
            make_experiment(env=env, execution=ExecutionContext(inherit_env="none"))
        ).digest

    assert identity_for({"TOKEN": "old"}) != identity_for({"TOKEN": "new"})
    # An injective encoding must not let a name/value boundary shift produce
    # one digest: "A" -> "B=1" and "A=B" -> "1" are different environments.
    assert identity_for({"A": "B=1"}) != identity_for({"A=B": "1"})


def test_environment_identity_names_are_sorted_and_hold_no_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Names are the diagnostic surface; values stay inside the digest only."""
    monkeypatch.setenv("PHASESWEEP_TEST_TOKEN", "ambient-secret")
    identity = _environment_identity(
        make_experiment(
            env={"ZULU": "z-secret", "ALPHA": "a-secret"},
            execution=ExecutionContext(inherit_env=["PHASESWEEP_TEST_TOKEN"]),
        )
    )

    assert list(identity.names) == sorted(identity.names)
    assert {"ALPHA", "ZULU", "PHASESWEEP_TEST_TOKEN"} <= set(identity.names)
    assert not {"a-secret", "z-secret", "ambient-secret"} & set(identity.names)


def test_launch_trial_records_private_environment_json_when_opted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``record_env: true`` persists the exact environment, owner-only (0600)."""
    monkeypatch.setenv("PHASESWEEP_TEST_TOKEN", "ambient-secret")
    execution = ExecutionContext(inherit_env=["PHASESWEEP_TEST_TOKEN"], record_env=True)
    _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env={"CONFIGURED": "value"},
        execution=execution,
    )

    path = tmp_path / "trial_0" / "environment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = _environment_identity(
        make_experiment(env={"CONFIGURED": "value"}, execution=execution)
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert payload["digest"] == identity.digest
    assert payload["inherit_env"] == ["PHASESWEEP_TEST_TOKEN"]
    assert payload["env"]["PHASESWEEP_TEST_TOKEN"] == "ambient-secret"
    assert payload["env"]["CONFIGURED"] == "value"
    assert sorted(payload["env"]) == list(identity.names)


def test_launch_trial_writes_no_environment_json_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values are secrets: nothing is written unless the operator opts in."""
    _capture_launch_env(tmp_path, monkeypatch)

    assert not (tmp_path / "trial_0" / "environment.json").exists()
