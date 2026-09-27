"""Protocol-layer gate evaluation tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phasesweep import load_experiment
from phasesweep.config import ArtifactSizeGate, JsonEqualsGate, Sha256Gate
from phasesweep.evidence.evaluation import evaluate_gates
from tests.conftest import make_trial_context, patch_path_method_failure, write_yaml


def test_artifact_size_gate_supports_file_directory_and_json_estimate(tmp_path: Path) -> None:
    """Artifact byte gates cover materialized artifacts and trainer-reported estimates."""
    (tmp_path / "model.bin").write_bytes(b"abcd")
    artifact_dir = tmp_path / "bundle"
    nested = artifact_dir / "nested"
    nested.mkdir(parents=True)
    (artifact_dir / "a.bin").write_bytes(b"abc")
    (nested / "b.bin").write_bytes(b"defg")
    (tmp_path / "result.json").write_text('{"artifact_estimate_bytes": 7}')
    ctx = make_trial_context(tmp_path, experiment="e")

    results = evaluate_gates(
        ctx,
        [
            ArtifactSizeGate(
                type="artifact_size",
                source="file",
                path="model.bin",
                min_bytes=4,
                max_bytes=4,
            ),
            ArtifactSizeGate(
                type="artifact_size",
                source="directory",
                path="bundle",
                min_bytes=7,
                max_bytes=7,
            ),
            ArtifactSizeGate(
                type="artifact_size",
                source="json",
                path="result.json",
                key="artifact_estimate_bytes",
                max_bytes=7,
            ),
        ],
    )

    assert [result.passed for result in results] == [True, True, True]


def test_sha256_gate_streams_file_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (b"phasesweep" * 131_072) + b"tail"
    (tmp_path / "model.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError("sha256 gate must stream instead of Path.read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    results = evaluate_gates(
        make_trial_context(tmp_path),
        [Sha256Gate(type="sha256", path="model.bin", sha256=digest)],
    )

    assert results[0].passed is True


@pytest.mark.parametrize(
    ("gate_kind", "path_method", "error_detail"),
    [
        pytest.param("sha256", "open", "could not read model.bin", id="sha256-open"),
        pytest.param("artifact_size", "stat", "could not inspect model.bin", id="size-stat"),
    ],
)
def test_file_gate_io_failures_are_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_kind: str,
    path_method: str,
    error_detail: str,
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"payload")
    patch_path_method_failure(
        monkeypatch,
        model_path,
        path_method,
        OSError("artifact metadata unavailable"),
    )

    if gate_kind == "sha256":
        gate = Sha256Gate(
            type="sha256",
            path="model.bin",
            sha256=hashlib.sha256(b"payload").hexdigest(),
        )
    else:
        gate = ArtifactSizeGate(
            type="artifact_size",
            source="file",
            path="model.bin",
            max_bytes=1024,
        )
    result = evaluate_gates(make_trial_context(tmp_path), [gate])[0]

    assert result.passed is False
    assert error_detail in result.detail


def test_json_equals_gate_requires_matching_json_type(tmp_path: Path) -> None:
    """Protocol equality is type-strict; numeric tolerance belongs in scalar bounds."""
    (tmp_path / "result.json").write_text('{"flag": true, "count": 1}')
    ctx = make_trial_context(tmp_path)

    results = evaluate_gates(
        ctx,
        [
            JsonEqualsGate(type="json_equals", path="result.json", key="flag", value=True),
            JsonEqualsGate(type="json_equals", path="result.json", key="flag", value=1),
            JsonEqualsGate(type="json_equals", path="result.json", key="count", value=1.0),
        ],
    )

    assert [result.passed for result in results] == [True, False, False]
    assert "bool" in results[1].detail
    assert "float" in results[2].detail


def _json_equals_gate_yaml(value_literal: str) -> str:
    """Return an experiment body whose only gate compares against ``value_literal``."""
    return f"""
    experiment: t
    storage: null
    provenance: {{revision: test-fixture-v1}}
    trial_command: "echo {{overrides}}"
    override_format: argparse
    metric:
      name: loss
      goal: minimize
      extractor: {{ type: json_envelope, objective_name: loss, split: test, policy: test }}
    phases:
      - name: a
        n_trials: 1
        search_space: {{ x: {{ type: float, low: 0, high: 1 }} }}
        gates:
          - type: json_equals
            path: result.json
            key: k
            value: {value_literal}
    """


@pytest.mark.parametrize(
    ("value_literal", "message"),
    [
        ("2024-01-01", "must be a JSON scalar"),
        ("{1: x}", "must be a JSON scalar"),
        ("[1, 2]", "must be a JSON scalar"),
        (".nan", "must be finite"),
        (".inf", "must be finite"),
    ],
    ids=["yaml_date", "mapping", "sequence", "nan", "inf"],
)
def test_json_equals_gate_rejects_non_json_scalars(
    tmp_path: Path, value_literal: str, message: str
) -> None:
    """Non-JSON gate values fail at load, not silently at every trial."""
    path = write_yaml(tmp_path, _json_equals_gate_yaml(value_literal))

    with pytest.raises(ValueError, match=message) as excinfo:
        load_experiment(path)

    assert "phases.0.gates.0.json_equals.value" in str(excinfo.value)


def test_json_equals_gate_accepts_quoted_date(tmp_path: Path) -> None:
    """The documented fix for a rejected YAML date is to quote it."""
    experiment = load_experiment(write_yaml(tmp_path, _json_equals_gate_yaml("'2024-01-01'")))

    assert experiment.phases[0].gates[0].value == "2024-01-01"


def test_artifact_size_gate_reports_bad_sources(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text('{"artifact_estimate_bytes": "7"}')
    ctx = make_trial_context(tmp_path, experiment="e")

    results = evaluate_gates(
        ctx,
        [
            ArtifactSizeGate(type="artifact_size", source="file", path="missing.bin", max_bytes=1),
            ArtifactSizeGate(
                type="artifact_size",
                source="json",
                path="result.json",
                key="artifact_estimate_bytes",
                max_bytes=10,
            ),
        ],
    )

    assert results[0].passed is False
    assert "not a file" in results[0].detail
    assert results[1].passed is False
    assert "not an integer" in results[1].detail
