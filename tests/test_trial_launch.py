"""Trial launch environment behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from phasesweep.engine.trial import launch_trial
from phasesweep.runtime.process import ProcessResult
from tests.conftest import make_experiment


def _capture_launch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_env: dict[str, str] | None = None,
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
    ) -> ProcessResult:
        captured.update(env)
        return ProcessResult(
            return_code=0,
            timed_out=False,
            pid=12345,
            duration_seconds=0.0,
        )

    monkeypatch.setattr("phasesweep.engine.trial.run_supervised", fake_run_supervised)
    launch_trial(
        experiment=make_experiment(env=experiment_env),
        phase_name="p",
        trial_id=0,
        trial_dir=tmp_path / "trial_0",
        overrides={},
        timeout_seconds=None,
        gpu_id=gpu_id,
    )
    return captured


def test_launch_trial_sets_cuda_device_order_for_leased_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _capture_launch_env(tmp_path, monkeypatch)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_launch_trial_preserves_operator_cuda_device_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_launch_trial_preserves_opaque_cuda_device_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _capture_launch_env(tmp_path, monkeypatch, gpu_id="MIG-GPU-deadbeef/3/0")

    assert env["CUDA_VISIBLE_DEVICES"] == "MIG-GPU-deadbeef/3/0"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
