"""Trial launch environment, pipe behavior, and the pre-launch barrier."""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import resource
import select
import shlex
import signal
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config.models import ExecutionContext, Experiment, Metric
from phasesweep.engine.artifacts import _warn_environment_drift
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import ATTEMPT_ID_ATTR, TRIAL_DIR_ATTR
from phasesweep.engine.trial import (
    EnvironmentIdentity,
    ExecutedTrial,
    TrialExecutionError,
    TrialResult,
    _environment_identity,
    _stderr_last_line,
    extract_trial_result,
    launch_trial,
    prepare_trainer_input,
)
from phasesweep.evidence.models import JsonEnvelopeExtractor, WandbSummaryRequiredGate
from phasesweep.runtime import supervisor
from phasesweep.runtime.process import ProcessResult, fd_ready, read_attempt_lifecycle
from phasesweep.tracking.session import write_record
from tests.conftest import make_experiment, make_trial_context, phase_study, write_trainer
from tests.tracking_helpers import attempt_identity, mlflow_binding

#: Every module ``runtime/supervisor.py`` may import. It runs before the trial's
#: process identity exists, so anything else it loaded could start code that
#: cleanup can never find.
SUPERVISOR_IMPORT_ALLOWLIST = frozenset(
    {
        "__future__",
        "contextlib",
        "json",
        "os",
        "pathlib",
        "signal",
        "sys",
        "tempfile",
        "time",
        "types",
    }
)


def _capture_launch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_env: dict[str, str] | None = None,
    execution: ExecutionContext | None = None,
    metric: Metric | None = None,
    gpu_id: int | str | None = 2,
    gpu_lease_fds: tuple[int, ...] = (),
    experiment_kwargs: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    environment_identity: EnvironmentIdentity | None = None,
    experiment: Experiment | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run_supervised(
        _cmd: str,
        *,
        env: dict[str, str],
        stdout: Any,
        stderr: Any,
        timeout: float | None,
        trial_dir: Path,
        attempt_id: str,
        wallclock_deadline: float | None = None,
        cwd: str | None = None,
        gpu_lease_fds: tuple[int, ...] = (),
    ) -> ProcessResult:
        captured.update(env)
        captured["process_env"] = dict(env)
        captured["command"] = _cmd
        captured["run_supervised_attempt_id"] = attempt_id
        if cwd is not None:
            captured["run_supervised_cwd"] = cwd
        captured["gpu_lease_fds"] = gpu_lease_fds
        return ProcessResult(
            return_code=0,
            timed_out=False,
            pid=12345,
            duration_seconds=0.0,
        )

    monkeypatch.setattr("phasesweep.engine.trial.run_supervised", fake_run_supervised)
    experiment = experiment or make_experiment(
        env=experiment_env,
        execution=execution,
        metric=metric,
        **(experiment_kwargs or {}),
    )
    trial_overrides = {} if overrides is None else overrides
    prepared = prepare_trainer_input(
        experiment=experiment,
        phase_name="p",
        trial_id=0,
        attempt_id="attempt-test",
        trial_dir=tmp_path / "trial_0",
        overrides=trial_overrides,
    )
    launch_trial(
        experiment=experiment,
        phase_name="p",
        trial_id=0,
        generation_id="generation-test",
        attempt_id="attempt-test",
        trial_dir=tmp_path / "trial_0",
        overrides=trial_overrides,
        timeout_seconds=None,
        gpu_id=gpu_id,
        gpu_lease_fds=gpu_lease_fds,
        prepared_input=prepared,
        environment_identity=(
            _environment_identity(experiment, "p")
            if environment_identity is None
            else environment_identity
        ),
    )
    captured["trainer_input"] = prepared.record()
    return captured


def test_launch_trial_forwards_gpu_lease_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer supervisor inherits the descriptors backing its GPU assignment."""
    env = _capture_launch_env(tmp_path, monkeypatch, gpu_lease_fds=(17, 23))

    assert env["gpu_lease_fds"] == (17, 23)


def test_launch_trial_writes_command_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-ASCII trial paths do not depend on the process locale."""
    command_encoding: list[str | None] = []
    real_write_text = Path.write_text

    def capture_encoding(path: Path, text: str, *, encoding: str | None = None) -> int:
        if path.name == "command.txt":
            command_encoding.append(encoding)
        return real_write_text(path, text, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", capture_encoding)
    _capture_launch_env(tmp_path / "tríäl", monkeypatch)

    assert command_encoding == ["utf-8"]


def test_launch_trial_injects_configured_objective_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = Metric(
        name="eval_loss",
        extractor=JsonEnvelopeExtractor(
            type="json_envelope",
            path="reports/objective.json",
            objective_name="eval_loss",
            split="validation",
            policy="final_checkpoint",
        ),
    )

    env = _capture_launch_env(tmp_path, monkeypatch, metric=metric)

    assert env["PHASESWEEP_OBJECTIVE_PATH"] == str(
        tmp_path / "trial_0" / "reports" / "objective.json"
    )


def test_launch_trial_drops_ambient_objective_path_for_non_envelope_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHASESWEEP_OBJECTIVE_PATH", "/tmp/not-this-trial.json")

    env = _capture_launch_env(tmp_path, monkeypatch)

    assert "PHASESWEEP_OBJECTIVE_PATH" not in env


@pytest.mark.parametrize(
    ("experiment_env", "gpu_id", "expected_visible", "expected_order"),
    [
        pytest.param(None, 2, "2", "PCI_BUS_ID", id="numeric-default-order"),
        pytest.param(
            {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
            2,
            "2",
            "PCI_BUS_ID",
            id="numeric-forces-lock-compatible-order",
        ),
        pytest.param(
            {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
            "GPU-deadbeef",
            "GPU-deadbeef",
            "FASTEST_FIRST",
            id="uuid-preserves-operator-device-order",
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
    assert env["run_supervised_attempt_id"] == "attempt-test"


def test_launch_trial_hashes_and_passes_complete_trainer_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default boundary gives the trainer one complete, identity-bound YAML."""
    trial_dir = tmp_path / "trial_0"
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        gpu_id=None,
        experiment_kwargs={
            "trial_command": "python train.py --config {config_path}",
            "override_format": "yaml_file",
            "trainer_config": {
                "model": {"depth": 4, "width": 128},
                "output_dir": "{trial_dir}/outputs",
            },
        },
        overrides={"model.depth": 8, "run_label": "{trial_dir}"},
    )

    config_path = trial_dir / "trainer_config.yaml"
    assert yaml.safe_load(config_path.read_text()) == {
        "model": {"depth": 8, "width": 128},
        "output_dir": f"{trial_dir}/outputs",
        # Runtime placeholders are a base-config feature. Sampled and fixed
        # string values remain literal trainer data.
        "run_label": "{trial_dir}",
    }
    assert shlex.split(captured["command"]) == [
        "python",
        "train.py",
        "--config",
        str(config_path),
    ]
    assert (
        captured["PHASESWEEP_OVERRIDES_SHA256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert captured["trainer_input"] == {
        "schema_version": 1,
        "format": "yaml_file",
        "filename": "trainer_config.yaml",
        "size_bytes": config_path.stat().st_size,
        "sha256": captured["PHASESWEEP_OVERRIDES_SHA256"],
    }
    assert json.loads((trial_dir / "overrides_resolved.json").read_text()) == {
        "model.depth": 8,
        "run_label": "{trial_dir}",
    }


@pytest.mark.parametrize("override_format", ["argparse", "hydra", "json_file"])
def test_launch_trial_delivers_trainer_config_in_every_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_format: str,
) -> None:
    """The base reaches argparse, hydra and json_file, and the trainer-input record binds the bytes."""
    trial_dir = tmp_path / "trial_0"
    placeholder = "{overrides_path}" if override_format == "json_file" else "{overrides}"
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        gpu_id=None,
        experiment_kwargs={
            "trial_command": f"python train.py {placeholder}",
            "override_format": override_format,
            "trainer_config": {"model": {"depth": 4, "width": 128}, "out": "{trial_dir}/outputs"},
        },
        overrides={"model.depth": 8},
    )

    trainer_input = trial_dir / captured["trainer_input"]["filename"]
    resolved = json.loads((trial_dir / "overrides_resolved.json").read_text())
    if override_format == "json_file":
        assert trainer_input.name == "overrides.json"
        assert json.loads(trainer_input.read_text()) == {
            "model": {"depth": 8, "width": 128},
            "out": f"{trial_dir}/outputs",
        }
        assert resolved == {"model.depth": 8}
        assert shlex.split(captured["command"]) == ["python", "train.py", str(trainer_input)]
    else:
        merged = {"model.depth": 8, "model.width": 128, "out": f"{trial_dir}/outputs"}
        assert trainer_input.name == "overrides_resolved.json"
        assert resolved == merged
        quote = '"' if override_format == "hydra" else ""
        separator = "" if override_format == "hydra" else "--"
        assert sorted(shlex.split(captured["command"])[2:]) == sorted(
            [
                f"{separator}model.depth=8",
                f"{separator}model.width=128",
                f"{separator}out={quote}{trial_dir}/outputs{quote}",
            ]
        )
    assert (
        captured["trainer_input"]["sha256"]
        == hashlib.sha256(trainer_input.read_bytes()).hexdigest()
    )
    assert captured["PHASESWEEP_OVERRIDES_SHA256"] == captured["trainer_input"]["sha256"]


def test_launch_trial_cli_input_without_trainer_config_is_the_overrides_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty base leaves the argparse trainer input as the sorted overrides JSON."""
    overrides = {"zeta.depth": 8, "alpha": 1, "zeta.width": 2}
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        gpu_id=None,
        experiment_kwargs={"trial_command": "python train.py {overrides}"},
        overrides=overrides,
    )

    expected = json.dumps(overrides, indent=2, sort_keys=True) + "\n"
    assert (tmp_path / "trial_0" / "overrides_resolved.json").read_text() == expected
    assert captured["PHASESWEEP_OVERRIDES_SHA256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert shlex.split(captured["command"])[2:] == [
        "--zeta.depth=8",
        "--alpha=1",
        "--zeta.width=2",
    ]


@pytest.mark.parametrize("override_format", ["argparse", "hydra"])
def test_cli_tuple_input_keeps_rendered_values_and_recorded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override_format: str
) -> None:
    """Supported CLI tuples serialize as JSON arrays without changing command values."""
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        gpu_id=None,
        experiment_kwargs={"override_format": override_format, "fixed_overrides": {"dims": (1, 2)}},
        overrides={"dims": (1, 2)},
    )
    expected = '{\n  "dims": [\n    1,\n    2\n  ]\n}\n'
    assert (tmp_path / "trial_0" / "overrides_resolved.json").read_text() == expected
    assert captured["PHASESWEEP_OVERRIDES_SHA256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert shlex.split(captured["command"])[-1] == (
        "--dims=[1,2]" if override_format == "argparse" else "dims=[1,2]"
    )


@pytest.mark.parametrize(
    ("execution", "expected_ambient"),
    [
        pytest.param(
            ExecutionContext(inherit_env="all"),
            {
                "PHASESWEEP_TEST_AMBIENT_SECRET": "leak",
                "PHASESWEEP_TEST_TOKEN": "tok",
                "PYTHONPATH": "/ambient/src",
            },
            id="all",
        ),
        pytest.param(None, {}, id="default-no-block"),
        pytest.param(
            ExecutionContext(inherit_env=["PHASESWEEP_TEST_TOKEN"]),
            {"PHASESWEEP_TEST_TOKEN": "tok"},
            id="allowlist",
        ),
    ],
)
def test_launch_trial_inherit_env_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution: ExecutionContext | None,
    expected_ambient: dict[str, str],
) -> None:
    """Inheritance selects ambient values while retaining configured and base values."""
    monkeypatch.setenv("PHASESWEEP_TEST_AMBIENT_SECRET", "leak")
    monkeypatch.setenv("PHASESWEEP_TEST_TOKEN", "tok")
    monkeypatch.setenv("PYTHONPATH", "/ambient/src")
    env = _capture_launch_env(
        tmp_path, monkeypatch, execution=execution, experiment_env={"KEEP_ME": "explicit"}
    )
    assert {
        key: env[key]
        for key in ("PHASESWEEP_TEST_AMBIENT_SECRET", "PHASESWEEP_TEST_TOKEN", "PYTHONPATH")
        if key in env
    } == expected_ambient
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


@pytest.mark.parametrize(
    ("ambient_visibility", "experiment_env", "execution"),
    [
        pytest.param("-1", None, ExecutionContext(inherit_env="all"), id="inherit-all"),
        pytest.param(
            "-1",
            None,
            ExecutionContext(inherit_env=["CUDA_VISIBLE_DEVICES"]),
            id="inherit-explicit-list",
        ),
        pytest.param(
            "7",
            {"CUDA_VISIBLE_DEVICES": "-1"},
            ExecutionContext(inherit_env="none"),
            id="configured-under-strict-contract",
        ),
    ],
)
def test_launch_trial_cuda_visibility_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_visibility: str,
    experiment_env: dict[str, str] | None,
    execution: ExecutionContext | None,
) -> None:
    """The environment contract forwards or replaces ambient CUDA visibility."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ambient_visibility)

    env = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env=experiment_env,
        execution=execution,
        gpu_id=None,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_launch_trial_warns_once_when_ambient_visibility_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Dropping ambient visibility with no device leased is reported, not silent."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())

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


@pytest.mark.parametrize(
    ("gpu_id", "gpu_lease_fds"),
    [pytest.param(2, (), id="cuda"), pytest.param(None, (17,), id="mps")],
)
def test_launch_trial_no_visibility_warning_when_a_device_is_leased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    gpu_id: int | None,
    gpu_lease_fds: tuple[int, ...],
) -> None:
    """CUDA visibility or an MPS host lease prevents an unleased-device warning."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        env = _capture_launch_env(
            tmp_path,
            monkeypatch,
            execution=ExecutionContext(inherit_env="none"),
            gpu_id=gpu_id,
            gpu_lease_fds=gpu_lease_fds,
        )

    if gpu_id is not None:
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
    else:
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert env["gpu_lease_fds"] == gpu_lease_fds
    assert not [r for r in caplog.records if "ambient CUDA_VISIBLE_DEVICES" in r.message]


def _exit_code_result(
    tmp_path: Path,
    experiment: Experiment,
    phase: str,
    code: int,
    *,
    stderr: bytes | None = None,
    trial_id: int = 0,
) -> TrialResult:
    """Extract the result of a trainer that exited with ``code`` after writing ``stderr``.

    :param Path tmp_path: Directory standing in for the trial directory.
    :param Experiment experiment: Experiment whose contract is under test.
    :param str phase: Phase name recorded in the trial context.
    :param int code: Trainer exit code.
    :param bytes | None stderr: Content of ``stderr.log``, or ``None`` for no such file.
    :param int trial_id: Trial number recorded in the trial context.
    :return TrialResult: The extracted (failed) result.
    """
    if stderr is None:
        (tmp_path / "stderr.log").unlink(missing_ok=True)
    else:
        (tmp_path / "stderr.log").write_bytes(stderr)
    executed = ExecutedTrial(
        ctx=make_trial_context(tmp_path, phase=phase, trial_id=trial_id),
        process=ProcessResult(return_code=code, timed_out=False, pid=1, duration_seconds=0.1),
    )
    return extract_trial_result(experiment=experiment, executed=executed)


def _environment_hints(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the narrowed-environment hints the captured log holds, in order."""
    return [r.getMessage() for r in caplog.records if "execution.inherit_env" in r.getMessage()]


OOM_LINE = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0"


@pytest.mark.parametrize(
    "execution",
    [
        pytest.param(ExecutionContext(), id="default"),
        pytest.param(ExecutionContext(inherit_env=["HF_HOME"]), id="named-list"),
    ],
)
def test_nonzero_exit_under_narrowed_environment_warns_once_per_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    execution: ExecutionContext,
) -> None:
    """The first nonzero exit in a phase names the environment contract and its three fixes."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())
    experiment = make_experiment(execution=execution)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        results = [
            _exit_code_result(tmp_path, experiment, "p", 3),
            _exit_code_result(tmp_path, experiment, "p", 3),
            _exit_code_result(tmp_path, experiment, "other", 4),
        ]

    assert [result.failure_reason for result in results] == [
        "non-zero exit code 3",
        "non-zero exit code 3",
        "non-zero exit code 4",
    ]
    hints = [r.getMessage() for r in caplog.records if "execution.inherit_env" in r.getMessage()]
    assert len(hints) == 2
    assert hints[0].startswith("[p/trial_0] the trainer exited with code 3 under ")
    assert hints[1].startswith("[other/trial_0] the trainer exited with code 4 under ")
    for fix in ("execution.inherit_env: [NAME]", "env:", "execution.inherit_env: all"):
        assert fix in hints[0]
    assert "PATH, HOME, LANG, LC_ALL, TMPDIR, USER, LOGNAME, TZ" in hints[0]
    assert ("HF_HOME" in hints[0]) == isinstance(execution.inherit_env, list)


def test_nonzero_exit_hint_is_silent_for_full_inheritance_and_for_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``inherit_env: all`` has no narrowed contract to blame, and a zero exit has no failure."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        _exit_code_result(
            tmp_path, make_experiment(execution=ExecutionContext(inherit_env="all")), "p", 2
        )
        _exit_code_result(tmp_path, make_experiment(), "p", 0)

    assert not [r for r in caplog.records if "execution.inherit_env" in r.getMessage()]


def test_nonzero_exit_reason_quotes_the_last_stderr_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A trainer's own last error names the cause, and an out-of-memory error gets no environment hint."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())
    stderr = f"Traceback (most recent call last):\n  File 'train.py', line 9\n{OOM_LINE}\n\n"

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        result = _exit_code_result(tmp_path, make_experiment(), "p", 1, stderr=stderr.encode())

    assert result.failure_reason == f"non-zero exit code 1; stderr: {OOM_LINE}"
    assert _environment_hints(caplog) == []


@pytest.mark.parametrize(
    "line",
    [
        "ModuleNotFoundError: No module named 'x'",
        "ImportError: cannot import name 'y' from 'x'",
        "/usr/bin/python: No module named x",
        "sh: 1: nvidia-smi: command not found",
        "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'",
        "KeyError: 'HF_HOME'",
        "RuntimeError: the HF_HOME environment variable must be set",
        "RuntimeError: WANDB_PROJECT is not set",
    ],
)
def test_stderr_naming_a_missing_environment_input_keeps_the_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    line: str,
) -> None:
    """A last line that suggests a missing variable, tool or module logs the hint, quoting that line."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        result = _exit_code_result(tmp_path, make_experiment(), "p", 1, stderr=line.encode())

    assert result.failure_reason == f"non-zero exit code 1; stderr: {line}"
    (hint,) = _environment_hints(caplog)
    assert f'Its last stderr line was "{line}".' in hint
    assert "execution.inherit_env: all" in hint


@pytest.mark.parametrize(
    "line",
    [
        OOM_LINE,
        "KeyError: 'loss'",
        "AssertionError: loss is nan",
        "RuntimeError: CUDA error: device-side assert triggered",
    ],
)
def test_stderr_naming_another_cause_gets_no_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    line: str,
) -> None:
    """A last line that names its own cause is reported alone."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        _exit_code_result(tmp_path, make_experiment(), "p", 1, stderr=line.encode())

    assert _environment_hints(caplog) == []


def test_a_failure_without_an_environment_cause_does_not_use_up_the_phase_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The once-per-phase hint waits for the first failure that suggests the environment."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())
    lines = [
        OOM_LINE,
        "ModuleNotFoundError: No module named 'x'",
        "ModuleNotFoundError: No module named 'y'",
    ]

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        for trial_id, line in enumerate(lines):
            _exit_code_result(
                tmp_path, make_experiment(), "p", 1, stderr=line.encode(), trial_id=trial_id
            )

    (hint,) = _environment_hints(caplog)
    assert hint.startswith("[p/trial_1] the trainer exited with code 1 under ")


@pytest.mark.parametrize(
    "stderr",
    [
        pytest.param(None, id="no-file"),
        pytest.param(b"", id="empty"),
        pytest.param(b"\n  \n\t\n", id="blank"),
        pytest.param(b"\x00\x01\xff\xfe\n\x1b[0m\n", id="binary-only"),
    ],
)
def test_nonzero_exit_without_a_stderr_line_keeps_the_reason_and_the_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stderr: bytes | None,
) -> None:
    """With no line to quote the reason is the bare exit code, and nothing rules the environment out."""
    monkeypatch.setattr("phasesweep.engine.progress._WARNED", set())

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.trial"):
        result = _exit_code_result(tmp_path, make_experiment(), "p", 2, stderr=stderr)

    assert result.failure_reason == "non-zero exit code 2"
    (hint,) = _environment_hints(caplog)
    assert "Its last stderr line" not in hint


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            b"warn\n\x1b[1;31mError:\x1b[0m\tbad\x07  value\x00\n",
            "Error: bad value",
            id="escapes-controls-and-whitespace",
        ),
        pytest.param(
            b"50%|#####     |\r100%|##########|\n", "100%|##########|", id="carriage-return"
        ),
        pytest.param(b"real error\n\xff\xfe\x00\x01\n\n", "real error", id="binary-last-line"),
        pytest.param("ünïcode ✗ error\n".encode(), "ünïcode ✗ error", id="unicode"),
        pytest.param(b"E" * 500, "E" * 197 + "...", id="long-line-is-cut"),
        pytest.param(b"E" * 200 + b"\n", "E" * 200, id="exactly-at-the-cap"),
        pytest.param(b"E" * 201 + b"\n", "E" * 197 + "...", id="one-over-the-cap"),
        pytest.param(b"first\n" + b"x\n" * 20_000 + b"last\n", "last", id="large-file"),
    ],
)
def test_stderr_quote_is_one_capped_printable_line(
    tmp_path: Path, content: bytes, expected: str
) -> None:
    """The quote has no control characters, no line break and at most 200 characters."""
    (tmp_path / "stderr.log").write_bytes(content)

    quoted = _stderr_last_line(tmp_path)

    assert quoted == expected
    assert quoted is not None and len(quoted) <= 200 and quoted.isprintable()


def test_stderr_quote_reads_only_the_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A line longer than the read window is quoted from the window, never read whole."""
    monkeypatch.setattr("phasesweep.engine.trial._STDERR_TAIL_BYTES", 16)
    (tmp_path / "stderr.log").write_bytes(b"A" * 100)

    assert _stderr_last_line(tmp_path) == "A" * 16


def test_stderr_quote_is_none_when_the_file_cannot_be_read(tmp_path: Path) -> None:
    """A missing log, and a path that is not a file, quote nothing instead of raising."""
    assert _stderr_last_line(tmp_path) is None
    (tmp_path / "stderr.log").mkdir()
    assert _stderr_last_line(tmp_path) is None


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
        ),
        "p",
    )
    reversed_order = _environment_identity(
        make_experiment(
            env={"BETA": "2", "ALPHA": "1"},
            execution=ExecutionContext(inherit_env="none"),
        ),
        "p",
    )

    assert forward.digest == reversed_order.digest
    assert len(forward.digest) == 64


def test_launch_trial_uses_prepared_environment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = ExecutionContext(inherit_env="all", record_env=True)
    monkeypatch.setenv("TZ", "Europe/Berlin")
    identity = _environment_identity(make_experiment(execution=execution), "p")
    monkeypatch.setenv("TZ", "Asia/Tokyo")

    captured = _capture_launch_env(
        tmp_path, monkeypatch, execution=execution, environment_identity=identity
    )

    assert captured["TZ"] == "Europe/Berlin"
    recorded = json.loads((tmp_path / "trial_0" / "environment.json").read_text())
    assert recorded["env"] == captured["process_env"]
    assert recorded["env"]["TZ"] == "Europe/Berlin"
    assert recorded["digest"] == identity.digest


def test_environment_identity_digest_tracks_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different values for one name are different environments."""
    monkeypatch.delenv("PHASESWEEP_TEST_AMBIENT_SECRET", raising=False)

    def identity_for(env: dict[str, str]) -> str:
        return _environment_identity(
            make_experiment(env=env, execution=ExecutionContext(inherit_env="none")), "p"
        ).digest

    assert identity_for({"TOKEN": "old"}) != identity_for({"TOKEN": "new"})


def test_environment_identity_tracks_names_and_values_in_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tracked mapping is sorted by name and holds each configured and listed value."""
    monkeypatch.setenv("PHASESWEEP_TEST_LISTED", "ambient-value")
    identity = _environment_identity(
        make_experiment(
            env={"ZULU": "z-value", "ALPHA": "a-value"},
            execution=ExecutionContext(inherit_env=["PHASESWEEP_TEST_LISTED"]),
        ),
        "p",
    )

    assert list(identity.tracked) == sorted(identity.tracked)
    assert identity.tracked["ALPHA"] == "a-value"
    assert identity.tracked["ZULU"] == "z-value"
    assert identity.tracked["PHASESWEEP_TEST_LISTED"] == "ambient-value"


def test_environment_digest_is_a_checksum_of_the_tracked_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest is the SHA-256 of the compact JSON of the tracked pairs, sorted by name."""
    monkeypatch.setenv("PHASESWEEP_TEST_LISTED", "ambient-value")
    identity = _environment_identity(
        make_experiment(
            env={"ALPHA": "a-value"},
            execution=ExecutionContext(inherit_env=["PHASESWEEP_TEST_LISTED"]),
        ),
        "p",
    )

    encoded = json.dumps(sorted(identity.tracked.items()), separators=(",", ":"))
    assert identity.digest == hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_launch_trial_records_private_environment_json_when_opted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``record_env: true`` persists the exact environment, owner-only (0600)."""
    monkeypatch.setenv("PHASESWEEP_TEST_TOKEN", "ambient-secret")
    execution = ExecutionContext(inherit_env=["PHASESWEEP_TEST_TOKEN"], record_env=True)
    metric = Metric(
        name="loss",
        extractor=JsonEnvelopeExtractor(
            type="json_envelope",
            path="reports/objective.json",
            objective_name="loss",
            split="validation",
            policy="final_checkpoint",
        ),
    )
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env={"CONFIGURED": "value"},
        execution=execution,
        metric=metric,
    )

    path = tmp_path / "trial_0" / "environment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = _environment_identity(
        make_experiment(env={"CONFIGURED": "value"}, execution=execution, metric=metric), "p"
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert payload["digest"] == identity.digest
    assert payload["inherit_env"] == ["PHASESWEEP_TEST_TOKEN"]
    assert payload["passthrough_env"] == ["TMPDIR"]
    assert payload["env"]["PHASESWEEP_TEST_TOKEN"] == "ambient-secret"
    assert payload["env"]["CONFIGURED"] == "value"
    assert payload["env"] == captured["process_env"]
    assert payload["env"]["PHASESWEEP_TRIAL_DIR"] == str(path.parent)
    assert payload["env"]["PHASESWEEP_TRIAL_ID"] == "0"
    assert payload["env"]["PHASESWEEP_PHASE"] == "p"
    assert payload["env"]["PHASESWEEP_RUN_NAME"] == "t-p-0-attempt-test"
    assert payload["env"]["PHASESWEEP_GENERATION_ID"] == "generation-test"
    assert payload["env"]["PHASESWEEP_ATTEMPT_ID"] == "attempt-test"
    assert payload["env"]["PHASESWEEP_OVERRIDES_SHA256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert payload["env"]["PHASESWEEP_OBJECTIVE_PATH"] == str(
        path.parent / "reports/objective.json"
    )
    assert payload["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert payload["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_recorded_environment_includes_wandb_attempt_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        execution=ExecutionContext(record_env=True),
        experiment_kwargs={
            "gates": [
                WandbSummaryRequiredGate(
                    type="wandb_summary_required", entity="entity", project="project", keys=["loss"]
                )
            ]
        },
    )
    recorded = json.loads((tmp_path / "trial_0" / "environment.json").read_text())

    assert recorded["env"] == captured["process_env"]
    assert recorded["env"]["WANDB_RUN_ID"] == "attempt-test"


def test_recorded_environment_resolves_mlflow_name_and_run_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = Experiment.model_validate(
        {
            **make_experiment().model_dump(),
            "execution": {"record_env": True},
            "mlflow": {"tracking_uri": "http://localhost:5000", "experiment_name": "existing"},
            "metric": {
                "name": "loss",
                "extractor": {"type": "mlflow", "metric_key": "loss", "select": "last"},
            },
        }
    )
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    write_record(trial_dir / "mlflow-expected.json", attempt_identity())
    write_record(trial_dir / "mlflow-binding.json", mlflow_binding())

    captured = _capture_launch_env(tmp_path, monkeypatch, experiment=experiment)
    recorded = json.loads((trial_dir / "environment.json").read_text())

    assert recorded["env"] == captured["process_env"]
    assert recorded["env"]["MLFLOW_EXPERIMENT_ID"] == "1"
    assert "MLFLOW_EXPERIMENT_NAME" not in recorded["env"]
    assert recorded["env"]["MLFLOW_RUN_ID"] == "native-id"


def test_passthrough_value_rotation_preserves_semantic_environment_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = ExecutionContext(inherit_env="none", passthrough_env=["WANDB_API_KEY"])
    experiment = make_experiment(execution=execution)

    monkeypatch.setenv("WANDB_API_KEY", "first-secret")
    first = _environment_identity(experiment, "p")
    monkeypatch.setenv("WANDB_API_KEY", "rotated-secret")
    second = _environment_identity(experiment, "p")

    assert first.digest == second.digest
    assert first.values["WANDB_API_KEY"] == "first-secret"
    assert second.values["WANDB_API_KEY"] == "rotated-secret"


@pytest.mark.parametrize(
    "execution",
    [
        pytest.param(ExecutionContext(), id="default"),
        pytest.param(ExecutionContext(inherit_env="all"), id="all"),
        pytest.param(ExecutionContext(inherit_env=["HF_HOME"]), id="list"),
        pytest.param(ExecutionContext(passthrough_env=["SOME_TOKEN"]), id="written-token"),
        pytest.param(ExecutionContext(passthrough_env=["TMPDIR"]), id="written-tmpdir"),
        pytest.param(
            ExecutionContext(inherit_env=["HF_HOME"], passthrough_env=["SOME_TOKEN"]),
            id="list-and-written-token",
        ),
    ],
)
def test_environment_digest_ignores_tmpdir_only(
    monkeypatch: pytest.MonkeyPatch, execution: ExecutionContext
) -> None:
    """``TMPDIR`` is built-in pass-through, so a sandbox's scratch path never splits a cohort."""
    experiment = make_experiment(execution=execution)

    monkeypatch.setenv("TMPDIR", "/sandbox/tmp")
    first = _environment_identity(experiment, "p")
    monkeypatch.setenv("TMPDIR", "/tmp")
    second = _environment_identity(experiment, "p")
    monkeypatch.setenv("PATH", "/another/bin")
    third = _environment_identity(experiment, "p")

    assert first.digest == second.digest
    assert first.values["TMPDIR"] == "/sandbox/tmp"
    assert second.values["TMPDIR"] == "/tmp"
    assert "TMPDIR" in first.values
    assert "TMPDIR" not in first.tracked
    assert third.digest != second.digest


def test_configured_tmpdir_stays_digested_beside_the_builtin_passthrough() -> None:
    """An ``env`` value for ``TMPDIR`` is semantic even though ``TMPDIR`` is passed through."""
    first = _environment_identity(make_experiment(env={"TMPDIR": "/scratch/a"}), "p")
    second = _environment_identity(make_experiment(env={"TMPDIR": "/scratch/b"}), "p")

    assert first.digest != second.digest


def test_skipped_phase_drift_uses_that_phases_remote_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated W&B phase cannot block drift checks for a skipped local phase."""
    base = make_experiment(execution=ExecutionContext(inherit_env="all"))
    remote = base.phases[0].model_copy(
        update={
            "name": "remote",
            "gates": [
                WandbSummaryRequiredGate(
                    type="wandb_summary_required",
                    entity="entity",
                    project="project",
                    keys=["eval/loss"],
                )
            ],
        }
    )
    local = base.phases[0].model_copy(update={"name": "local", "gates": []})
    experiment = base.model_copy(update={"phases": [remote, local]})
    monkeypatch.setenv("WANDB_MODE", "offline")
    stored = _environment_identity(experiment, "local").digest

    _warn_environment_drift(experiment, "local", stored)


def test_launch_trial_preserves_configured_and_passthrough_wandb_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured and pass-through W&B variables stay ordinary trainer environment entries."""
    monkeypatch.setenv("WANDB_API_KEY", "ambient-token")
    captured = _capture_launch_env(
        tmp_path,
        monkeypatch,
        experiment_env={"WANDB_MODE": "online", "WANDB_RUN_ID": "operator-run-id"},
        execution=ExecutionContext(inherit_env="none", passthrough_env=["WANDB_API_KEY"]),
    )

    assert captured["WANDB_API_KEY"] == "ambient-token"
    assert captured["WANDB_MODE"] == "online"
    assert captured["WANDB_RUN_ID"] == "operator-run-id"


def test_configured_env_value_cannot_be_exempted_as_passthrough() -> None:
    execution = ExecutionContext(inherit_env="none", passthrough_env=["WANDB_API_KEY"])
    first = _environment_identity(
        make_experiment(execution=execution, env={"WANDB_API_KEY": "configured-a"}), "p"
    )
    second = _environment_identity(
        make_experiment(execution=execution, env={"WANDB_API_KEY": "configured-b"}), "p"
    )

    assert first.digest != second.digest


def test_launch_trial_writes_no_environment_json_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full composed environment can hold credentials, so no file is written unless the operator opts in."""
    _capture_launch_env(tmp_path, monkeypatch)

    assert not (tmp_path / "trial_0" / "environment.json").exists()


def test_launch_pipe_waits_accept_descriptors_past_select_limit() -> None:
    """A launcher holding over 1024 open files can still wait on its trial pipes.

    ``select.select`` raises ``ValueError`` for any descriptor at or above
    ``FD_SETSIZE``; the payload write, the supervisor handshake, and the MCP
    runner's ready byte all wait through ``fd_ready`` instead.
    """
    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= 1502:
        pytest.skip("the open-file limit cannot reach past select's FD_SETSIZE")
    read_fd, write_fd = os.pipe()
    high_read = fcntl.fcntl(read_fd, fcntl.F_DUPFD_CLOEXEC, 1500)
    high_write = fcntl.fcntl(write_fd, fcntl.F_DUPFD_CLOEXEC, 1500)
    os.close(read_fd)
    os.close(write_fd)
    try:
        assert not fd_ready(high_read, timeout=0)
        assert fd_ready(high_write, timeout=0, write=True)
        os.write(high_write, b"x")
        # A 30-day trial timeout is longer than one poll() call accepts.
        assert fd_ready(high_read, timeout=30 * 86400)
        assert os.read(high_read, 1) == b"x"
        os.close(high_write)
        # A hang-up is ready, so the caller's read sees EOF itself.
        assert fd_ready(high_read, timeout=0)
        assert os.read(high_read, 1) == b""
    finally:
        os.close(high_read)
        with contextlib.suppress(OSError):
            os.close(high_write)
    with pytest.raises(OSError):
        fd_ready(high_read, timeout=0)


def test_launch_pipe_wait_outlasts_one_poll_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wait past poll()'s millisecond limit keeps waiting instead of timing out."""
    waits: list[int] = []

    class _Poller:
        def register(self, fd: int, events: int) -> None:
            pass

        def poll(self, timeout_ms: int) -> list[tuple[int, int]]:
            waits.append(timeout_ms)
            return [(7, select.POLLIN)] if len(waits) == 2 else []

    monkeypatch.setattr(select, "poll", _Poller)
    assert fd_ready(7, timeout=30 * 86400)
    assert len(waits) == 2
    assert max(waits) <= 2**31 - 1


def test_supervisor_imports_only_its_stdlib_allowlist() -> None:
    """The pre-identity supervisor imports nothing outside its stdlib allowlist.

    It runs under ``python -I -S`` before the trial's process identity is
    written, so a new import there, or a dynamic one, could execute code that
    forks out of the process group cleanup later signals.
    """
    tree = ast.parse(Path(str(supervisor.__file__)).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the supervisor imports nothing relative to its package"
            imported.add(str(node.module).split(".")[0])
    dynamic = [
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "__import__"
    ]

    assert "os" in imported, "the parser found none of the supervisor's imports"
    assert imported <= SUPERVISOR_IMPORT_ALLOWLIST, imported - SUPERVISOR_IMPORT_ALLOWLIST
    assert dynamic == []


def test_trial_directory_and_allocated_lifecycle_precede_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before ``launch_trial`` runs, the trial carries its directory and its lifecycle reads ``allocated``.

    Recovery reads a ``RUNNING`` trial without the directory attribute, and an
    ``allocated`` lifecycle without a process identity, as proof that nothing
    launched, so both must be durable before any process can start.
    """
    experiment = make_experiment(persistent=tmp_path, trial_command="true {overrides}", n_trials=1)
    observed: dict[str, str | None] = {}

    def inspect_before_launch(**kwargs: object) -> ExecutedTrial:
        trial_dir = Path(str(kwargs["trial_dir"]))
        study = phase_study(experiment)
        trial = next(trial for trial in study.trials if trial.number == kwargs["trial_id"])
        lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=str(kwargs["attempt_id"]))
        observed["trial_dir"] = str(trial_dir)
        observed["trial_dir_attr"] = trial.user_attrs.get(TRIAL_DIR_ATTR)
        observed["lifecycle"] = None if lifecycle is None else lifecycle.state
        raise TrialExecutionError("inspected before launch; no process started")

    monkeypatch.setattr("phasesweep.engine.phase.launch_trial", inspect_before_launch)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(experiment)

    assert observed["trial_dir_attr"] == observed["trial_dir"]
    assert observed["lifecycle"] == "allocated"


@pytest.mark.integration
def test_guardian_sigkill_fails_one_trial_and_continues_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A killed guardian leaves its live trainer for direct cleanup and later trials still run."""
    import phasesweep.runtime.process as process
    from phasesweep.runtime.reaper import _process_group_alive, read_stale_process_identity

    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse, json, os, signal, sys, time
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        args, _ = parser.parse_known_args()
        if os.environ["PHASESWEEP_TRIAL_ID"] == "0":
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            Path(args.out).write_text(json.dumps({"x": 0.25}))
            os.kill(os.getppid(), signal.SIGKILL)
            while True:
                time.sleep(1)
        Path(args.out).write_text(json.dumps({"x": 0.5}))
        print("x=0.5")
        """,
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trial_command=(
            f"exec {shlex.quote(sys.executable)} {shlex.quote(str(trainer))} "
            "--out {trial_dir}/r.json {overrides}"
        ),
        n_trials=2,
        max_consecutive_failures=2,
        timeout_seconds_per_trial=5,
        gpu_policy="none",
    )
    real_terminate = process._terminate_process_groups
    cleaned: list[int] = []

    def terminate(
        pgids: tuple[int, ...],
        *,
        grace_seconds: float,
        guardians: Mapping[int, subprocess.Popen[bytes]] | None = None,
    ) -> dict[int, bool]:
        assert guardians is None
        first = phase_study(experiment).trials[0]
        directory = first.user_attrs[TRIAL_DIR_ATTR]
        attempt = first.user_attrs[ATTEMPT_ID_ATTR]
        assert isinstance(directory, str) and isinstance(attempt, str)
        lifecycle = read_attempt_lifecycle(Path(directory), expected_attempt_id=attempt)
        assert lifecycle is not None and lifecycle.state == "terminating"
        assert lifecycle.return_code == -signal.SIGTERM
        cleaned.extend(pgids)
        return real_terminate(pgids, grace_seconds=grace_seconds, guardians=guardians)

    monkeypatch.setattr(process, "_terminate_process_groups", terminate)
    try:
        winners = run_experiment(experiment)
    finally:
        for trial in phase_study(experiment).trials:
            directory = trial.user_attrs.get(TRIAL_DIR_ATTR)
            attempt = trial.user_attrs.get(ATTEMPT_ID_ATTR)
            if isinstance(directory, str) and isinstance(attempt, str):
                identity = read_stale_process_identity(Path(directory), expected_attempt_id=attempt)
                if _process_group_alive(identity.pgid):
                    real_terminate((identity.pgid,), grace_seconds=0.05)

    first, second = phase_study(experiment).trials
    assert [first.state.name, second.state.name] == ["FAIL", "COMPLETE"]
    assert winners["p"].trial_number == second.number
    assert winners["p"].metric == 0.5
    assert len(cleaned) == 1 and not _process_group_alive(cleaned[0])
    lifecycle = read_attempt_lifecycle(
        Path(first.user_attrs[TRIAL_DIR_ATTR]),
        expected_attempt_id=first.user_attrs[ATTEMPT_ID_ATTR],
    )
    assert lifecycle is not None and lifecycle.state == "exited" and lifecycle.cleanup_confirmed
    assert lifecycle.return_code == -signal.SIGTERM
    assert "Lease guardian" in caplog.text
    assert "root process exited with code -9" not in caplog.text
