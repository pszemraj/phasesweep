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
