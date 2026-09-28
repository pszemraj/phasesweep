"""Selection-time confound verdicts: the tri-state, each check, and the persisted block."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config import IntParam, Phase, Sampler
from phasesweep.engine import read_status
from phasesweep.engine.confound import (
    CONFOUND_CHECKS,
    CONFOUND_VERDICTS,
    _assess_population,
    _confound_summary,
    _describe_flagged,
    _flagged_checks,
    _validate_confound_block,
)
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.paths import _generation_winner_path
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    Winner,
)
from tests.conftest import make_experiment, write_constant_trainer

_COMPLETE = optuna.trial.TrialState.COMPLETE
_FAIL = optuna.trial.TrialState.FAIL
_PRUNED = optuna.trial.TrialState.PRUNED
_RUNNING = optuna.trial.TrialState.RUNNING


def _envelope(checkpoint: str = "final.pt", step: int = 1000) -> dict[str, Any]:
    """Return a current-format ``json_envelope`` objective provenance record."""
    return {
        "schema_version": 1,
        "extractor": {"kind": "json_envelope", "config_sha256": "0" * 64},
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "source": {
            "kind": "file",
            "path": "result.json",
            "size_bytes": 0,
            "sha256": "0" * 64,
            "evaluation": {
                "objective_name": "loss",
                "split": "validation",
                "policy": "final",
                "checkpoint": checkpoint,
                "step": step,
            },
        },
    }


def _log_regex() -> dict[str, Any]:
    """Return a current-format ``log_regex`` objective provenance record."""
    return {
        "schema_version": 1,
        "extractor": {"kind": "log_regex", "config_sha256": "0" * 64},
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "source": {"kind": "file", "path": "stdout.log", "size_bytes": 0, "sha256": "0" * 64},
    }


def _trials(*specs: tuple[optuna.trial.TrialState, float | None, int]) -> list[Any]:
    """Build numbered frozen trials from ``(state, value, x)`` specs."""
    study = optuna.create_study()
    for state, value, x in specs:
        study.add_trial(
            optuna.trial.create_trial(
                state=state,
                value=value,
                params={"x": x},
                distributions={"x": optuna.distributions.IntDistribution(0, 100)},
            )
        )
    return study.get_trials(deepcopy=False)


def _assess(
    trials: list[Any],
    ranked: list[Any],
    *,
    provenance: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assess ``ranked`` with the lowest value winning, as minimize selection does."""
    winner = min(ranked, key=lambda trial: (trial.value, trial.number))
    records = provenance or {trial.number: _log_regex() for trial in ranked}
    return _assess_population(trials, ranked, winner, provenance=records)


def test_verdicts_are_three_distinct_values_and_not_checked_is_never_ok() -> None:
    """``n/a`` means not checked; nothing may collapse it into ``ok``."""
    assert CONFOUND_VERDICTS == ("ok", "heterogeneous", "n/a")
    assert len(set(CONFOUND_VERDICTS)) == 3

    trials = _trials((_COMPLETE, 0.5, 1))
    block = _assess(trials, trials)

    assert block["checks"]["evaluation_point"]["verdict"] == "n/a"
    assert block["checks"]["tie"]["verdict"] == "n/a"
    assert _flagged_checks(block) == []
    assert _validate_confound_block(block)["checks"]["tie"]["verdict"] == "n/a"
    masquerading = copy.deepcopy(block)
    masquerading["checks"]["tie"]["detail"] = {"tied_trials": []}
    with pytest.raises(ValueError, match="n/a without a reason"):
        _validate_confound_block(masquerading)


def test_evaluation_point_agreement_is_ok_with_grouped_evidence() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
    block = _assess(trials, trials, provenance={0: _envelope(), 1: _envelope()})

    check = block["checks"]["evaluation_point"]
    assert check == {
        "verdict": "ok",
        "detail": {
            "checkpoint": [{"value": "final.pt", "trials": [0, 1]}],
            "step": [{"value": 1000, "trials": [0, 1]}],
        },
    }


@pytest.mark.parametrize(
    ("records", "field", "groups"),
    [
        pytest.param(
            {0: _envelope(step=1000), 1: _envelope(step=1000), 2: _envelope(step=400)},
            "step",
            [{"value": 400, "trials": [2]}, {"value": 1000, "trials": [0, 1]}],
            id="step",
        ),
        pytest.param(
            {0: _envelope(checkpoint="best.pt"), 1: _envelope(), 2: _envelope()},
            "checkpoint",
            [{"value": "best.pt", "trials": [0]}, {"value": "final.pt", "trials": [1, 2]}],
            id="checkpoint",
        ),
    ],
)
def test_evaluation_point_disagreement_is_heterogeneous(
    records: dict[int, dict[str, Any]], field: str, groups: list[dict[str, Any]]
) -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2), (_COMPLETE, 0.3, 3))
    block = _assess(trials, trials, provenance=records)

    check = block["checks"]["evaluation_point"]
    assert check["verdict"] == "heterogeneous"
    assert check["detail"][field] == groups
    assert _flagged_checks(block) == ["evaluation_point"]
    assert f"2 distinct {field}s" in _describe_flagged(block)[0]


def test_evaluation_point_is_not_checked_without_envelope_metadata() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
    block = _assess(trials, trials)

    check = block["checks"]["evaluation_point"]
    assert check["verdict"] == "n/a"
    assert "log_regex" in check["reason"]


@pytest.mark.parametrize(
    ("ranked", "infeasible", "failed", "pruned", "verdict"),
    [
        pytest.param(1, 0, 1, 0, "heterogeneous", id="last-standing"),
        pytest.param(2, 3, 0, 0, "heterogeneous", id="minority-feasible"),
        pytest.param(2, 1, 1, 0, "heterogeneous", id="half-excluded"),
        pytest.param(3, 0, 1, 1, "ok", id="majority-ranked"),
        pytest.param(1, 0, 0, 0, "ok", id="single-trial"),
    ],
)
def test_survivorship_flags_a_winner_ranked_among_at_most_half(
    ranked: int, infeasible: int, failed: int, pruned: int, verdict: str
) -> None:
    specs = (
        [(_COMPLETE, 0.5 + index, index) for index in range(ranked + infeasible)]
        + [(_FAIL, None, 50 + index) for index in range(failed)]
        + [(_PRUNED, None, 70 + index) for index in range(pruned)]
        + [(_RUNNING, None, 99)]
    )
    trials = _trials(*specs)
    block = _assess(trials, trials[:ranked])

    assert block["checks"]["survivorship"] == {
        "verdict": verdict,
        "detail": {"ranked": ranked, "infeasible": infeasible, "failed": failed, "pruned": pruned},
    }
    assert block["ranked_trials"] == list(range(ranked))


def test_tie_between_distinct_params_is_heterogeneous() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_COMPLETE, 0.9, 3))
    block = _assess(trials, trials)

    assert block["checks"]["tie"] == {"verdict": "heterogeneous", "detail": {"tied_trials": [1]}}
    assert "trial(s) 1 matched" in _describe_flagged(block)[0]


def test_tie_between_repeats_of_one_config_is_ok() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 1))
    block = _assess(trials, trials)

    assert block["checks"]["tie"] == {"verdict": "ok", "detail": {"tied_trials": []}}


def test_assessed_block_survives_yaml_and_validation_unchanged() -> None:
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_FAIL, None, 3))
    block = _assess(trials, trials[:2], provenance={0: _envelope(), 1: _envelope(step=5)})

    assert _validate_confound_block(yaml.safe_load(yaml.safe_dump(block))) == block


def _mutated(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    """Return a valid block with the value at ``path`` replaced, or deleted for ``...``."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.4, 2))
    block = _assess(trials, trials)
    target = block
    for key in path[:-1]:
        target = target[key]
    if value is ...:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return block


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("schema_version",), 2, id="schema-version"),
        pytest.param(("schema_version",), True, id="schema-version-bool"),
        pytest.param(("ranked_trials",), [1, 0], id="ranked-unsorted"),
        pytest.param(("ranked_trials",), [], id="ranked-empty"),
        pytest.param(("checks", "tie"), ..., id="missing-check"),
        pytest.param(("checks", "speed"), {"verdict": "ok", "detail": {}}, id="unknown-check"),
        pytest.param(("checks", "tie", "verdict"), "clean", id="unknown-verdict"),
        pytest.param(("checks", "evaluation_point", "reason"), ..., id="n/a-without-reason"),
        pytest.param(
            ("checks", "survivorship"),
            {"verdict": "n/a", "reason": "skipped"},
            id="survivorship-n/a",
        ),
        pytest.param(("checks", "survivorship", "detail"), ..., id="checked-without-detail"),
        pytest.param(("checks", "survivorship", "detail", "failed"), -1, id="negative-count"),
        pytest.param(("checks", "survivorship", "detail", "failed"), True, id="bool-count"),
        pytest.param(("checks", "tie", "detail", "tied_trials"), ["1"], id="string-trial"),
    ],
)
def test_validator_rejects_malformed_blocks(path: tuple[Any, ...], value: Any) -> None:
    with pytest.raises(ValueError, match="confound"):
        _validate_confound_block(_mutated(path, value))


def test_summary_keeps_only_verdicts_and_counts() -> None:
    """The agent-visible projection drops every reported string."""
    trials = _trials((_COMPLETE, 0.5, 1), (_COMPLETE, 0.5, 2), (_FAIL, None, 3))
    block = _assess(
        trials,
        trials[:2],
        provenance={0: _envelope(checkpoint="/abs/ckpt.pt"), 1: _envelope(step=5)},
    )

    summary = _confound_summary(block, winner_trial=0)

    assert summary == {
        "flagged": ["evaluation_point", "tie"],
        "evaluation_point": {
            "verdict": "heterogeneous",
            "distinct_checkpoints": 2,
            "distinct_steps": 2,
            "winner_step": 1000,
        },
        "survivorship": {
            "verdict": "ok",
            "ranked": 2,
            "infeasible": 0,
            "failed": 1,
            "pruned": 0,
        },
        "tie": {"verdict": "heterogeneous", "tied_trials": 1},
    }
    assert "/abs/ckpt.pt" not in json.dumps(summary)


def test_summary_reports_unchecked_counts_as_null() -> None:
    trials = _trials((_COMPLETE, 0.5, 1))
    summary = _confound_summary(_assess(trials, trials), winner_trial=0)

    assert summary["evaluation_point"] == {
        "verdict": "n/a",
        "distinct_checkpoints": None,
        "distinct_steps": None,
        "winner_step": None,
    }
    assert summary["tie"] == {"verdict": "n/a", "tied_trials": None}


def test_validator_rejects_an_absent_block() -> None:
    with pytest.raises(ValueError, match="confound block is missing"):
        _validate_confound_block(None)


def test_confound_block_never_moves_a_downstream_fingerprint() -> None:
    """An observation must never invalidate a study or block a top-up."""
    experiment = make_experiment(
        phases=[
            Phase(name="parent", n_trials=1),
            Phase(name="child", n_trials=1, inherits=["parent"]),
        ]
    )
    child = experiment.phases[1]
    trials = _trials((_COMPLETE, 0.5, 1), (_FAIL, None, 2))
    flagged = _assess(trials, trials[:1])
    clean = _assess(trials[:1], trials[:1])
    assert _flagged_checks(flagged) != _flagged_checks(clean)

    fingerprints = {
        _phase_fingerprint(
            experiment,
            child,
            {
                "parent": Winner(
                    trial_number=0,
                    params={"x": 1},
                    effective_overrides={"x": 1},
                    metric=0.5,
                    confound=block,
                )
            },
        )
        for block in (flagged, clean, None)
    }
    assert len(fingerprints) == 1


def _candidate_attrs(number: int, provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the user attrs a selection candidate carries."""
    return {
        FEASIBLE_ATTR: True,
        GENERATION_ID_ATTR: "generation-test",
        ATTEMPT_ID_ATTR: f"attempt-{number}",
        OBJECTIVE_PROVENANCE_ATTR: json.dumps(provenance),
        TRAINER_ENV_DIGEST_ATTR: "a" * 64,
        TRAINER_INPUT_ATTR: {"schema_version": 1},
    }


def test_selection_records_the_block_and_logs_each_flagged_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    experiment = make_experiment(phases=[Phase(name="p", n_trials=3)])
    study = optuna.create_study(direction="minimize")
    for number, (value, step) in enumerate([(0.2, 1000), (0.3, 400)]):
        study.add_trial(
            optuna.trial.create_trial(
                state=_COMPLETE,
                value=value,
                params={"x": number},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                user_attrs=_candidate_attrs(number, _envelope(step=step)),
            )
        )
    study.add_trial(optuna.trial.create_trial(state=_FAIL))
    study.add_trial(optuna.trial.create_trial(state=_FAIL))

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        selected = select_winner(study, experiment, phase_name="p")

    assert selected.trial_number == 0
    assert _flagged_checks(selected.confound) == ["evaluation_point", "survivorship"]
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.selection" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith("[p] potential confounds in the 2 ranked trials")
    assert "2 distinct steps (400, 1000)" in warnings[0]
    assert "ranked among 2 of 4 terminal trials" in warnings[0]


def _published_winner(experiment: Any) -> tuple[Path, dict[str, Any]]:
    """Return the published winner path and payload for phase ``p``."""
    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "ok"
    assert pointer.generation_id is not None
    path = _generation_winner_path(experiment, pointer.generation_id, "p")
    return path, yaml.safe_load(path.read_text())


@pytest.mark.integration
def test_published_block_is_manifest_covered(tmp_path: Path) -> None:
    """Editing a verdict after publication is tampering, not a correction."""
    experiment = make_experiment(persistent=tmp_path, trainer=write_constant_trainer(tmp_path))
    run_experiment(experiment)
    path, payload = _published_winner(experiment)
    # The constant trainer ties both trials with different params.
    assert _flagged_checks(_validate_confound_block(payload["confound"])) == ["tie"]

    payload["confound"]["checks"]["tie"] = {"verdict": "ok", "detail": {"tied_trials": []}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    assert _resolve_publication_pointer(experiment).state == "failed"
    assert read_status(experiment)["publication_integrity"] == "failed"


@pytest.mark.integration
def test_top_up_reselection_publishes_a_fresh_block_for_its_larger_population(
    tmp_path: Path,
) -> None:
    """Verdicts belong to the selection, not to the trial it re-selects."""
    experiment = make_experiment(persistent=tmp_path, trainer=write_constant_trainer(tmp_path))
    run_experiment(experiment)
    _, first = _published_winner(experiment)

    phase = experiment.phases[0].model_copy(update={"n_trials": 3})
    run_experiment(experiment.model_copy(update={"phases": [phase]}))
    _, second = _published_winner(experiment)

    assert second["winner_source"] == first["winner_source"]
    assert first["confound"]["ranked_trials"] == [0, 1]
    assert second["confound"]["ranked_trials"] == [0, 1, 2]
    assert second["confound"]["checks"]["tie"]["detail"] == {"tied_trials": [1, 2]}


def _inheritance_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return inheritance warnings logged since the last clear."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.run" and "inherits the winner" in record.getMessage()
    ]


@pytest.mark.integration
def test_inheriting_phases_are_warned_and_from_phase_carries_the_block(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The flag reaches every phase that builds on it, whichever path resolved the parent."""
    caplog.set_level(logging.WARNING, logger="phasesweep")
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        phases=[
            Phase(
                name="parent",
                n_trials=2,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
            ),
            Phase(
                name="child",
                n_trials=1,
                inherits=["parent"],
                sampler=Sampler(type="random", seed=0),
                search_space={"y": IntParam(type="int", low=0, high=10)},
            ),
        ],
    )
    first = run_experiment(experiment)
    in_process = _inheritance_warnings(caplog)
    caplog.clear()
    second = run_experiment(experiment, from_phase="child")
    resumed = _inheritance_warnings(caplog)

    # The constant trainer ties both parent trials with different params.
    assert first["parent"].confound is not None
    assert _flagged_checks(first["parent"].confound) == ["tie"]
    assert second["parent"].confound == first["parent"].confound
    assert set(CONFOUND_CHECKS) == set(second["parent"].confound["checks"])
    for warnings in (in_process, resumed):
        assert len(warnings) == 1
        assert warnings[0].startswith("[child] inherits the winner of phase 'parent'")
        assert "tie: trial(s) 1 matched" in warnings[0]
