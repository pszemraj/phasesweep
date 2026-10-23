"""GPU inventory diagnostics during read-only validation and previews."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from phasesweep import load_experiment, run_experiment
from phasesweep.cli import cli
from phasesweep.errors import GpuConfigurationError
from tests.experiments import phase_payload, write_experiment


@pytest.fixture
def single_gpu_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: (["0"], {"0": "GPU-test-0"}),
    )

    def refuse_lease(*args: object, **kwargs: object) -> None:
        pytest.fail("read-only GPU preflight attempted to acquire a host lease")

    monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", refuse_lease)


@pytest.mark.parametrize("command", ["validate", "dry-run"])
def test_gpu_preflight_refuses_nonexistent_explicit_index(
    tmp_path: Path, single_gpu_inventory: None, command: str
) -> None:
    workdir = tmp_path / "runs"
    config = write_experiment(
        tmp_path / "experiment.yaml",
        workdir=str(workdir),
        phases=[phase_payload(gpu_ids=[99])],
    )

    if command == "validate":
        result = CliRunner().invoke(cli, ["validate", str(config)])
        assert isinstance(result.exception, GpuConfigurationError), result.output
        assert "index 99 does not exist" in str(result.exception)
    else:
        with pytest.raises(GpuConfigurationError, match="index 99 does not exist"):
            run_experiment(load_experiment(config), dry_run=True)

    assert not workdir.exists()


@pytest.mark.parametrize("command", ["validate", "dry-run"])
@pytest.mark.parametrize("device", ["index", "uuid", "mig"])
def test_gpu_preflight_inventory_refusal_explains_host_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, device: str
) -> None:
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: set())

    def refuse_lease(*args: object, **kwargs: object) -> None:
        pytest.fail("read-only GPU preflight attempted to acquire a host lease")

    monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", refuse_lease)
    workdir = tmp_path / "runs"
    phase = (
        phase_payload(gpu_ids=[0])
        if device == "index"
        else phase_payload(gpu_devices=["GPU-test" if device == "uuid" else "MIG-test"])
    )
    config = write_experiment(tmp_path / "experiment.yaml", workdir=str(workdir), phases=[phase])

    if command == "validate":
        result = CliRunner().invoke(cli, ["validate", str(config)])
        assert isinstance(result.exception, GpuConfigurationError), result.output
        error = result.exception
    else:
        with pytest.raises(GpuConfigurationError) as failure:
            run_experiment(load_experiment(config), dry_run=True)
        error = failure.value

    assert "nvidia-smi" in str(error)
    assert "sandbox" in str(error)
    assert "host shell with GPU access" in str(error)
    assert not workdir.exists()


@pytest.mark.parametrize("command", ["validate", "dry-run"])
def test_gpu_preflight_warns_when_workers_exceed_devices(
    tmp_path: Path,
    single_gpu_inventory: None,
    caplog: pytest.LogCaptureFixture,
    command: str,
) -> None:
    workdir = tmp_path / "runs"
    config = write_experiment(
        tmp_path / "experiment.yaml",
        workdir=str(workdir),
        phases=[phase_payload(gpu_ids=[0], n_jobs=3, n_trials=4)],
    )

    with caplog.at_level(logging.WARNING):
        if command == "validate":
            result = CliRunner().invoke(cli, ["validate", str(config)])
            assert result.exit_code == 0, result.output
        else:
            run_experiment(load_experiment(config), dry_run=True)

    assert "n_jobs=3 but only 1 GPU device(s)" in caplog.text
    assert "Excess trials will queue for a GPU" in caplog.text
    assert not workdir.exists()
