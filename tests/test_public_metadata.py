"""Public packaging metadata and engine export drift checks (no build artifacts).

Also guards the other public surface PhaseSweep ships: the shareable metadata
inside a published generation namespace. Every file there except the
owner-only config snapshot is meant to be read and passed around, so none of
them may carry a configured secret.
"""

from __future__ import annotations

import importlib
import stat
import tomllib
from pathlib import Path

import pytest

import phasesweep.engine as engine
import phasesweep.evidence as evidence
from phasesweep.engine import errors as engine_errors
from phasesweep.engine import run_experiment
from phasesweep.engine.state import _generation_dir, _last_successful_generation_id
from phasesweep.evidence import evaluation as evidence_evaluation
from tests.conftest import make_experiment, write_constant_trainer

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

CONSOLE_SCRIPTS = {
    "phasesweep": "phasesweep.cli:main",
    "phasesweep-mcp": "phasesweep.mcp.server:main",
}


def _pyproject() -> dict:
    if not PYPROJECT_PATH.is_file():
        pytest.skip(f"source pyproject.toml unavailable at {PYPROJECT_PATH}")
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_console_scripts_declare_importable_targets() -> None:
    """Both entry points stay declared and keep pointing at real callables."""
    assert _pyproject()["project"]["scripts"] == CONSOLE_SCRIPTS

    for target in CONSOLE_SCRIPTS.values():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), f"{target} is not callable"


def test_engine_exports_all_typed_preflight_errors() -> None:
    """Engine callers can catch every typed preflight failure from the public API."""
    names = (
        "ExperimentLockBusyError",
        "SamplerContinuationUnsupportedError",
        "StudyContextConflictError",
        "StudyFingerprintMismatchError",
        "StudySchemaMismatchError",
        "StudyStorageUnavailableError",
        "TrialTargetRegressionError",
    )

    assert set(names).issubset(engine.__all__)
    assert all(getattr(engine, name) is getattr(engine_errors, name) for name in names)


def test_published_generation_metadata_never_exposes_configured_secrets(
    tmp_path: Path,
) -> None:
    """A configured ``env`` value reaches disk only through the owner-only snapshot.

    ``config.snapshot.yaml`` exists so a published winner can be reproduced, so
    it necessarily holds ``env:`` values and is written ``0600``. Every other
    file in the generation namespace -- the summary, the lifecycle record, the
    winners, and ``reproducibility.json`` -- is ordinary umask-governed
    metadata an operator may share, and must carry digests rather than values
    (review v0.5.18 / finding F6).
    """
    secret = "s3cr3t-public-metadata-sentinel"
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
        env={"TRAINER_TOKEN": secret},
    )
    run_experiment(experiment)

    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None
    generation_dir = _generation_dir(experiment, generation_id)

    carriers = []
    for path in sorted(generation_dir.rglob("*")):
        if not path.is_file():
            continue
        if secret.encode() in path.read_bytes():
            carriers.append(path.relative_to(generation_dir).as_posix())

    assert carriers == ["config.snapshot.yaml"], carriers
    assert stat.S_IMODE((generation_dir / "config.snapshot.yaml").stat().st_mode) == 0o600


def test_evidence_exports_deadline_exceeded_error() -> None:
    """Custom extractor authors catch the deadline failure from the package namespace.

    ``DeadlineExceededError`` is the one extractor failure a caller may want
    to distinguish (the trial ran out of wallclock, the extractor itself is
    fine), so it must not require importing the private
    ``phasesweep.evidence.evaluation`` module path its sibling
    ``ExtractorError`` never did.
    """
    from phasesweep.evidence import DeadlineExceededError, ExtractorError

    assert DeadlineExceededError is evidence_evaluation.DeadlineExceededError
    assert issubclass(DeadlineExceededError, ExtractorError)
    assert "DeadlineExceededError" in evidence.__all__
