"""The redaction invariant: no payload ever contains the trial command, the
storage URL, the workdir, or any env value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep.engine import PhaseWinnerView
from phasesweep.mcp.redaction import status_payload, winners_payload
from phasesweep.mcp.registry import Registry
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
workdir: {tmp_path}/SECRET_WORKDIR
trial_command: "python /opt/secret/train.py --token DANGER_TOKEN --out {{trial_dir}}/r.json {{overrides}}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
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
    reg = Registry.load(_write_catalog(tmp_path)).get("redact_me")
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
    status = status_payload(
        reg.id,
        {"metric": {"name": "loss", "goal": "minimize"}, "phases": [], "summary_present": False},
        None,
        result_source="current_shared_study",
        elapsed_seconds=None,
    )

    assert_no_sensitive(winners, sensitive)
    assert_no_sensitive(status, sensitive)
    assert str(reg.config_path) not in str(winners)  # the catalog path is never exposed
    assert "effective_overrides" not in winners["phases"][0]
    assert "SECRET_FIXED_OVERRIDE" not in str(winners)
    assert "/private/data" not in str(winners)


def test_assert_no_sensitive_actually_catches_a_leak() -> None:
    # Guard against a vacuous scanner: it must fail when a needle IS present.
    leaky = {"experiment_id": "x", "note": "token=DANGER_TOKEN"}
    with pytest.raises(AssertionError):
        assert_no_sensitive(leaky, ["DANGER_TOKEN"])


def test_params_redacted_flag_follows_policy() -> None:
    """The boolean is computed from the policy, so agents need not string-match."""

    def flag(params: dict[str, object], policy: object) -> bool:
        payload = _winners_payload(
            "x",
            [PhaseWinnerView("p", 0, 0.1, dict(params), dict(params), None, False)],
            visible_params=policy,
        )
        return payload["phases"][0]["params_redacted"]

    assert flag({"lr": 3e-4, "depth": 6}, "all") is False
    assert flag({"lr": 3e-4, "depth": 6}, "none") is True
    assert flag({}, "none") is False  # nothing withheld when nothing was sampled
    assert flag({"lr": 3e-4, "depth": 6}, ["lr"]) is True
    assert flag({"lr": 3e-4}, ["lr", "depth"]) is False
    # A literal sentinel VALUE with an open policy must not read as redaction.
    assert flag({"note": "<redacted>"}, "all") is False


def test_winners_payload_applies_visible_params_policy() -> None:
    views = [
        PhaseWinnerView(
            "p",
            0,
            0.1,
            {"dataset": "SECRET_DATASET", "lr": 3e-4},
            {"dataset": "SECRET_DATASET", "lr": 3e-4},
            None,
            False,
        )
    ]

    assert _winners_payload("redact_me", views)["phases"][0]["params"] == {
        "dataset": "<redacted>",
        "lr": "<redacted>",
    }
    assert _winners_payload("redact_me", views, visible_params=["lr"])["phases"][0]["params"] == {
        "dataset": "<redacted>",
        "lr": 3e-4,
    }
    assert _winners_payload("redact_me", views, visible_params="all")["phases"][0]["params"] == {
        "dataset": "SECRET_DATASET",
        "lr": 3e-4,
    }


def test_winners_payload_computes_phase_completeness_and_provenance() -> None:
    payload = winners_payload(
        "exp",
        [PhaseWinnerView("p1", 2, 0.1, {}, {}, None, False)],
        metric={"name": "loss", "goal": "minimize"},
        declared_phases=["p1", "p2"],
        result_source="frozen_run_snapshot",
        run_id="exp-run",
    )

    assert payload["run_id"] == "exp-run"
    assert payload["result_source"] == "frozen_run_snapshot"
    assert payload["declared_phase_count"] == 2
    assert payload["winner_count"] == 1
    assert payload["missing_phases"] == ["p2"]
    assert payload["all_phases_have_winners"] is False
