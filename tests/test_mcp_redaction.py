"""The redaction invariant: no payload ever contains the trial command, the
storage URL, the workdir, or any env value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.engine import PhaseWinnerView
from phasesweep.engine.state import WinnerSource
from phasesweep.mcp.redaction import winners_payload
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.server import PhaseSweepMCP
from tests.mcp_helpers import assert_no_sensitive, write_mcp_catalog


def _winners_payload(
    experiment_id: str,
    views: list[PhaseWinnerView],
    *,
    visible_params: object = "none",
) -> dict[str, object]:
    return winners_payload(
        experiment_id,
        views,
        metric={"name": "loss", "goal": "minimize"},
        declared_phases=["p"],
        result_source="current_shared_study",
        visible_params=visible_params,  # type: ignore[arg-type]
    )


def _write_catalog(tmp_path: Path) -> Path:
    # A config whose dangerous fields contain unmistakable sentinels.
    config = tmp_path / "exp.yaml"
    config.write_text(
        f"""\
experiment: redact_me
storage: sqlite:///{tmp_path}/SECRET_DB.db
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path}/SECRET_WORKDIR
trial_command: "python /opt/secret/train.py --token DANGER_TOKEN --out {{trial_dir}}/r.json {{overrides}}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json_envelope, path: r.json, objective_name: loss, split: test, policy: test }}
env:
  HF_TOKEN: SECRET_ENV_VALUE
phases:
  - name: p
    n_trials: 1
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
"""
    )
    return write_mcp_catalog(tmp_path, {"redact_me": config})


def test_payloads_never_leak_sensitive_fields(tmp_path: Path) -> None:
    registry = Registry.load(_write_catalog(tmp_path))
    reg = registry.get("redact_me")
    sensitive = [
        reg.experiment.trial_command,
        reg.experiment.storage,
        *reg.experiment.env.values(),
        "SECRET_WORKDIR",
        "DANGER_TOKEN",
        "SECRET_ENV_VALUE",
    ]

    winners = _winners_payload(
        reg.id,
        [
            PhaseWinnerView(
                "p",
                0,
                0.1,
                {"lr": 3e-4},
                {"lr": 3e-4, "token": "SECRET_FIXED_OVERRIDE", "data": "/private/data"},
                None,
                False,
            )
        ],
    )
    assert_no_sensitive(winners, sensitive)
    assert str(reg.config_path) not in str(winners)  # the catalog path is never exposed
    assert "effective_overrides" not in winners["phases"][0]
    assert "SECRET_FIXED_OVERRIDE" not in str(winners)
    assert "/private/data" not in str(winners)

    app = PhaseSweepMCP(registry, RunStore(registry.state_dir))
    tool_results = [
        app.list_experiments(),
        app.validate(reg.id),
        app.latest_run(reg.id),
        app.status(experiment_id=reg.id),
        app.winners(experiment_id=reg.id),
    ]
    assert_no_sensitive(tool_results, sensitive)


@pytest.mark.parametrize(
    ("params", "policy", "expected", "redacted"),
    [
        (
            {"dataset": "SECRET_DATASET", "lr": 3e-4},
            "none",
            {"dataset": "<redacted>", "lr": "<redacted>"},
            True,
        ),
        (
            {"dataset": "SECRET_DATASET", "lr": 3e-4},
            ["lr"],
            {"dataset": "<redacted>", "lr": 3e-4},
            True,
        ),
        (
            {"dataset": "SECRET_DATASET", "lr": 3e-4},
            "all",
            {"dataset": "SECRET_DATASET", "lr": 3e-4},
            False,
        ),
        ({}, "none", {}, False),
        ({"lr": 3e-4}, ["lr", "depth"], {"lr": 3e-4}, False),
        ({"note": "<redacted>"}, "all", {"note": "<redacted>"}, False),
    ],
)
def test_winners_payload_applies_visible_params_policy(
    params: dict[str, object],
    policy: object,
    expected: dict[str, object],
    redacted: bool,
) -> None:
    view = PhaseWinnerView("p", 0, 0.1, dict(params), dict(params), None, False)
    phase = _winners_payload("redact_me", [view], visible_params=policy)["phases"][0]
    assert phase["params"] == expected
    assert phase["params_redacted"] is redacted


def test_winners_payload_computes_phase_completeness_and_provenance() -> None:
    payload = winners_payload(
        "exp",
        [
            PhaseWinnerView(
                "p1",
                2,
                0.1,
                {},
                {},
                None,
                False,
                source=WinnerSource(
                    kind="phase_trial",
                    phase="p1",
                    trial_number=2,
                    generation_id="generation-old",
                    attempt_id="attempt-old",
                ),
            )
        ],
        metric={"name": "loss", "goal": "minimize"},
        declared_phases=["p1", "p2"],
        result_source="frozen_run_snapshot",
        run_id="exp-run",
        represented_generation_id="generation-new",
    )

    assert payload["run_id"] == "exp-run"
    assert payload["result_source"] == "frozen_run_snapshot"
    assert payload["declared_phase_count"] == 2
    assert payload["winner_count"] == 1
    assert payload["missing_phases"] == ["p2"]
    assert payload["all_phases_have_winners"] is False
    assert payload["phases"][0]["winner_generation"] == "prior_generation"
