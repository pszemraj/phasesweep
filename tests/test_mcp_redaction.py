"""The redaction invariant: no payload ever contains the trial command, the
storage URL, the workdir, or any env value.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from phasesweep.config import FloatParam, JsonEnvelopeExtractor, Metric
from phasesweep.engine import PhaseWinnerView
from phasesweep.engine.repeats import RepeatsSummary
from phasesweep.engine.state import WinnerSource
from phasesweep.mcp.redaction import intersect_visible_params, winners_payload
from phasesweep.mcp.snapshots import capture_result_snapshot
from tests.conftest import make_experiment
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    _write_experiment_config,
    assert_no_sensitive,
    make_mcp_app,
    mcp_experiment_config_text,
    patch_popen_capture,
    write_mcp_catalog,
    write_mcp_config_catalog,
    write_run_status,
)


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
        publication_integrity="ok",
        visible_params=visible_params,  # type: ignore[arg-type]
    )


def _write_catalog(
    tmp_path: Path,
    *,
    allow: dict[str, bool] | None = None,
) -> Path:
    # Dangerous fields carry unmistakable sentinels so a leak is unambiguous.
    experiment = make_experiment(
        experiment="redact_me",
        workdir=tmp_path / "SECRET_WORKDIR",
        storage=f"journal:///{tmp_path}/SECRET_DB.journal",
        trial_command=(
            "python /opt/secret/train.py --token DANGER_TOKEN --out {trial_dir}/r.json {overrides}"
        ),
        metric=Metric(
            name="loss",
            goal="minimize",
            extractor=JsonEnvelopeExtractor(
                type="json_envelope",
                path="r.json",
                objective_name="loss",
                split="test",
                policy="test",
            ),
        ),
        env={"HF_TOKEN": "SECRET_ENV_VALUE"},
        n_trials=1,
        search_space={"lr": FloatParam(type="float", low=1.0e-5, high=1.0e-2, log=True)},
    )
    config = tmp_path / "exp.yaml"
    _write_experiment_config(config, experiment)
    return write_mcp_catalog(tmp_path, {"redact_me": config}, allow=allow)


def test_payloads_never_leak_sensitive_fields(tmp_path: Path) -> None:
    app, registry, _store = make_mcp_app(_write_catalog(tmp_path))
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
                source=WinnerSource(
                    kind="phase_trial",
                    phase="p",
                    trial_number=0,
                    generation_id="generation-1",
                    attempt_id="attempt-1",
                ),
            )
        ],
    )
    assert_no_sensitive(winners, sensitive)
    assert str(reg.config_path) not in str(winners)  # the catalog path is never exposed
    assert "effective_overrides" not in winners["phases"][0]
    assert "SECRET_FIXED_OVERRIDE" not in str(winners)
    assert "/private/data" not in str(winners)

    tool_results = [
        app.list_experiments(),
        app.validate(reg.id),
        app.latest_run(reg.id),
    ]
    assert_no_sensitive(tool_results, sensitive)


def test_run_workflow_payloads_never_leak_sensitive_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, registry, store = make_mcp_app(_write_catalog(tmp_path, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("redact_me")
    patch_popen_capture(monkeypatch)
    sensitive = [
        reg.experiment.trial_command,
        reg.experiment.storage,
        *reg.experiment.env.values(),
        str(reg.config_path),
        "SECRET_WORKDIR",
        "DANGER_TOKEN",
        "SECRET_ENV_VALUE",
    ]

    launched = app.launch(reg.id)
    run_id = launched["run_id"]
    live_status = app.status(run_id=run_id)
    live_winners = app.winners(run_id=run_id)

    def fake_kill_stale_group(*args: object, **kwargs: object) -> bool:
        write_run_status(
            store,
            run_id,
            returncode=143,
            error_class="cancelled",
            cleanup_confirmed=True,
            result_snapshot_state="complete",
            result_snapshot=capture_result_snapshot(
                reg.experiment,
                generation_id=run_id,
            ),
        )
        return True

    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", fake_kill_stale_group)
    cancelled = app.cancel(run_id)
    terminal_status = app.status(run_id=run_id)
    awaited = asyncio.run(app.await_run(run_id))
    terminal_winners = app.winners(run_id=run_id)
    latest = app.latest_run(reg.id)

    assert_no_sensitive(
        [
            launched,
            live_status,
            live_winners,
            cancelled,
            terminal_status,
            awaited,
            terminal_winners,
            latest,
        ],
        sensitive,
    )


def test_assert_no_sensitive_actually_catches_a_leak() -> None:
    leaky = {"experiment_id": "x", "note": "token=DANGER_TOKEN"}

    with pytest.raises(AssertionError):
        assert_no_sensitive(leaky, ["DANGER_TOKEN"])


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
    view = PhaseWinnerView(
        "p",
        0,
        0.1,
        dict(params),
        dict(params),
        None,
        False,
        source=WinnerSource(
            kind="phase_trial",
            phase="p",
            trial_number=0,
            generation_id="generation-1",
            attempt_id="attempt-1",
        ),
    )
    phase = _winners_payload("redact_me", [view], visible_params=policy)["phases"][0]
    assert phase["params"] == expected
    assert phase["params_redacted"] is redacted


@pytest.mark.parametrize(
    ("launch_policy", "current_policy", "expected"),
    [
        ("none", "all", "none"),
        ("all", "none", "none"),
        ("all", ["lr"], ["lr"]),
        (["lr", "depth"], "all", ["lr", "depth"]),
        (["lr", "depth"], ["depth", "dropout"], ["depth"]),
        ([], "all", []),
    ],
)
def test_visible_params_intersection_never_broadens_either_policy(
    launch_policy: object,
    current_policy: object,
    expected: object,
) -> None:
    assert intersect_visible_params(launch_policy, current_policy) == expected  # type: ignore[arg-type]


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
        publication_integrity="ok",
        run_id="exp-run",
        represented_generation_id="generation-new",
        result_context="represented_generation",
        published_config_matches_current=False,
    )

    assert payload["run_id"] == "exp-run"
    assert payload["result_source"] == "frozen_run_snapshot"
    assert payload["represented_generation_id"] == "generation-new"
    # Completeness is measured against the plan the caller supplied, which for
    # a published result is that generation's own (review v0.5.16 / blocker 4).
    assert payload["declared_phase_count"] == 2
    assert payload["winner_count"] == 1
    assert payload["missing_phases"] == ["p2"]
    assert payload["all_phases_have_winners"] is False
    assert payload["phases"][0]["winner_generation"] == "prior_generation"
    assert payload["result_context"] == "represented_generation"
    assert payload["published_config_matches_current"] is False


def test_winners_payload_exposes_observed_values() -> None:
    """Report-only values reach the agent-visible phase payload and default to none."""
    source = WinnerSource(
        kind="phase_trial",
        phase="p1",
        trial_number=2,
        generation_id="generation-1",
        attempt_id="attempt-1",
    )
    payload = winners_payload(
        "exp",
        [
            PhaseWinnerView(
                "p1", 2, 0.1, {}, {}, None, False, source=source, observed={"peak": 7.5}
            ),
            PhaseWinnerView("p2", 3, 0.2, {}, {}, None, False, source=source),
        ],
        metric={"name": "loss", "goal": "minimize"},
        declared_phases=["p1", "p2"],
        result_source="frozen_run_snapshot",
        publication_integrity="ok",
        run_id="exp-run",
        represented_generation_id="generation-1",
    )

    assert [phase["observed"] for phase in payload["phases"]] == [{"peak": 7.5}, {}]
    assert [phase["repeats"] for phase in payload["phases"]] == [None, None]


def test_winner_payload_exposes_repeats_spread_but_not_repeat_key() -> None:
    """An agent sees that metric is a reduction over members, never the repeat key or values."""
    source = WinnerSource(
        kind="phase_trial",
        phase="p1",
        trial_number=5,
        generation_id="generation-1",
        attempt_id="attempt-5",
    )
    summary: RepeatsSummary = {
        "reduce": "mean",
        "n": 3,
        "mean": 0.3,
        "sd": 0.1,
        "min": 0.2,
        "max": 0.4,
        "member_trials": [3, 4, 5],
    }
    payload = winners_payload(
        "exp",
        [
            PhaseWinnerView(
                "p1", 5, 0.3, {"lr": 0.1}, {"lr": 0.1}, None, False, source=source, repeats=summary
            )
        ],
        metric={"name": "loss", "goal": "minimize"},
        declared_phases=["p1"],
        result_source="frozen_run_snapshot",
        publication_integrity="ok",
        run_id="exp-run",
        represented_generation_id="generation-1",
    )

    (phase,) = payload["phases"]
    assert phase["repeats"] == summary
    assert phase["metric"] == summary["mean"]
    assert set(phase["params"]) == {"lr"}
    assert "key" not in phase["repeats"]


def test_validate_reports_the_repeat_count_but_not_the_repeat_key(tmp_path: Path) -> None:
    phases = """\
  - name: p
    n_trials: 6
    sampler: { type: random, seed: 0 }
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
    repeats: { key: trainer.secret_repeat_key, values: [rv_alpha, rv_beta, rv_gamma] }
"""
    catalog = write_mcp_config_catalog(
        tmp_path, {"srv": mcp_experiment_config_text(tmp_path, phases=phases)}
    )
    app, _registry, _store = make_mcp_app(catalog)

    structure = app.validate("srv")

    assert [phase["repeats"] for phase in structure["phases"]] == [3]
    assert structure["phases"][0]["search_space"] == ["lr"]
    assert_no_sensitive(structure, ["secret_repeat_key", "rv_beta"])


def test_winners_payload_defaults_claim_no_historical_provenance() -> None:
    """An undisclosed provenance reads as unknown, never as "no drift".

    The defaults exist for callers that have no represented generation to
    describe (nothing published), so they must not assert that the current
    config produced the result or that it still matches.
    """
    payload = _winners_payload("exp", [])

    assert payload["result_context"] == "current_config"
    assert payload["published_config_matches_current"] is None
