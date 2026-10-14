"""Repeat groups: keys, statistics, and the winner's ``repeats`` block validator."""

from __future__ import annotations

import copy
import math
import statistics

import pytest
import yaml

from phasesweep.engine.repeats import (
    RepeatAxis,
    _repeats_summary,
    _validate_repeats_block,
    group_key,
    observed_means,
    ranked_group,
    reduce_values,
    spread,
    within_noise,
)
from phasesweep.runtime.json import JsonObject


def _member(number: int, seed: int, metric: float, observed: dict[str, float]) -> JsonObject:
    """Return one ``repeats.members`` record."""
    return {
        "trial_number": number,
        "repeat": seed,
        "metric": metric,
        "generation_id": "generation-a" if number < 4 else "generation-b",
        "attempt_id": f"attempt-{number}",
        "objective_provenance": {"schema_version": 1, "trial": number},
        "trainer_input": {"schema_version": 1, "sha256": f"{number:064d}"},
        "constraints": {"memory": 1.0 + number},
        "observed": dict(observed),
    }


def _payload(reduce: str = "mean") -> JsonObject:
    """Return a winner payload whose scalar fields describe its group's last member."""
    members = [
        _member(1, 0, 0.5, {"wer": 0.1}),
        _member(3, 1, 0.7, {}),
        _member(5, 2, 0.9, {"wer": 0.3}),
    ]
    values = [0.5, 0.7, 0.9]
    last = members[-1]
    return {
        "trial_number": last["trial_number"],
        "generation_id": last["generation_id"],
        "attempt_id": last["attempt_id"],
        "objective_provenance": last["objective_provenance"],
        "trainer_input": last["trainer_input"],
        "constraints": last["constraints"],
        "params": {"lr": 0.1},
        "effective_overrides": {"lr": 0.1, "model.depth": 4},
        "observed": {"wer": statistics.fmean([0.1, 0.3])},
        "repeats": {
            "key": "seed",
            "reduce": reduce,
            "spread": dict(spread(values)),
            "members": members,
        },
    }


# The confound block's ranked groups for the payload above: its own and one other.
_GROUPS = [[0, 2, 4], [1, 3, 5]]


def _metric(payload: JsonObject) -> float:
    """Return the aggregate a payload's members reduce to."""
    repeats = payload["repeats"]
    assert isinstance(repeats, dict)
    reduce = repeats["reduce"]
    assert reduce in ("mean", "median")
    return reduce_values([0.5, 0.7, 0.9], reduce)


def test_reducers_and_spread_use_sample_statistics() -> None:
    values = [1.0, 2.0, 6.0]
    assert reduce_values(values, "mean") == 3.0
    assert reduce_values(values, "median") == 2.0
    assert spread(values) == {
        "n": 3,
        "mean": 3.0,
        "sd": statistics.stdev(values),
        "min": 1.0,
        "max": 6.0,
    }


def test_group_key_keeps_seeded_random_slots_apart_and_ignores_the_repeat_key() -> None:
    random_axis = RepeatAxis(key="seed", values=(0, 1), reduce="mean", slot_size=2)
    grid_axis = RepeatAxis(key="seed", values=(0, 1), reduce="mean", slot_size=None)
    params = {"lr": 0.1, "seed": 0}

    assert group_key(0, params, random_axis) == group_key(1, {"lr": 0.1, "seed": 1}, random_axis)
    assert group_key(1, params, random_axis) != group_key(2, params, random_axis)
    assert group_key(1, params, grid_axis) == group_key(6, {"seed": 1, "lr": 0.1}, grid_axis)
    assert random_axis.index_of(1) == 1
    assert random_axis.index_of(7) is None


def test_within_noise_compares_aggregates_against_two_standard_errors() -> None:
    winner = ranked_group([(1, 0.50), (0, 0.60)], {"lr": 0.1}, "mean")
    close = ranked_group([(2, 0.52), (3, 0.62)], {"lr": 0.2}, "mean")
    far = ranked_group([(4, 0.90), (5, 0.91)], {"lr": 0.3}, "mean")

    assert winner.numbers == (0, 1)
    assert winner.representative == 1
    assert math.isclose(winner.aggregate, 0.55)
    assert within_noise(winner, close)
    assert not within_noise(winner, far)
    assert within_noise(winner, winner)


def test_observed_means_average_only_the_members_that_recorded_a_name() -> None:
    assert observed_means([{"wer": 0.1}, {}, {"wer": 0.3, "cer": 0.2}]) == {
        "cer": 0.2,
        "wer": statistics.fmean([0.1, 0.3]),
    }


@pytest.mark.parametrize("reduce", ["mean", "median"])
def test_repeats_block_survives_yaml_and_validation(reduce: str) -> None:
    payload = yaml.safe_load(yaml.safe_dump(_payload(reduce)))
    record = _validate_repeats_block(payload, metric_value=_metric(payload), ranked_groups=_GROUPS)

    assert record is not None
    assert record == payload["repeats"]
    assert _repeats_summary(record) == {
        "reduce": reduce,
        "n": 3,
        "mean": statistics.fmean([0.5, 0.7, 0.9]),
        "sd": statistics.stdev([0.5, 0.7, 0.9]),
        "min": 0.5,
        "max": 0.9,
        "member_trials": [1, 3, 5],
    }


def test_repeats_block_is_null_without_repeats_and_required_as_a_key() -> None:
    assert _validate_repeats_block({"repeats": None}, metric_value=0.5, ranked_groups=None) is None
    with pytest.raises(ValueError, match="repeats block is missing"):
        _validate_repeats_block({}, metric_value=0.5, ranked_groups=None)


def test_repeats_block_and_confound_groups_agree_on_the_winning_group() -> None:
    """The members must be one of the groups the winner's confound block ranked."""
    with pytest.raises(ValueError, match="null though the confound block ranked repeat groups"):
        _validate_repeats_block({"repeats": None}, metric_value=0.5, ranked_groups=_GROUPS)
    for groups in (None, [[0, 2, 4], [6, 7, 8]], [[1, 3], [5, 6]]):
        with pytest.raises(ValueError, match="not one of the confound block's ranked groups"):
            _validate_repeats_block(
                _payload(), metric_value=_metric(_payload()), ranked_groups=groups
            )


def _tampered(edit: str) -> JsonObject:
    """Return a valid payload with one named field tampered."""
    payload = copy.deepcopy(_payload())
    repeats = payload["repeats"]
    assert isinstance(repeats, dict)
    members = repeats["members"]
    assert isinstance(members, list)
    first, last = members[0], members[-1]
    assert isinstance(first, dict) and isinstance(last, dict)
    if edit == "member-metric":
        first["metric"] = 0.4
    elif edit == "spread":
        stats = repeats["spread"]
        assert isinstance(stats, dict)
        stats["sd"] = 0.0
    elif edit == "representative-attempt":
        payload["attempt_id"] = "attempt-1"
    elif edit == "representative-trial":
        payload["trial_number"] = 1
    elif edit == "order":
        members.reverse()
    elif edit == "repeat-value":
        last["repeat"] = 0
    elif edit == "observed":
        payload["observed"] = {"wer": 0.1}
    elif edit == "key-in-params":
        payload["effective_overrides"] = {"lr": 0.1, "seed": 2}
    elif edit == "reduce":
        repeats["reduce"] = "trimmed_mean"
    elif edit == "single-member":
        repeats["members"] = [last]
    elif edit == "member-keys":
        del first["observed"]
    return payload


@pytest.mark.parametrize(
    ("edit", "error"),
    [
        pytest.param("member-metric", "mean of the member metrics differs", id="aggregate"),
        pytest.param("spread", "spread differs", id="spread"),
        pytest.param("representative-attempt", "disagrees with the winner's attempt_id", id="rep"),
        pytest.param("representative-trial", "winner's trial_number", id="rep-trial"),
        pytest.param("order", "ascending and distinct", id="order"),
        pytest.param("repeat-value", "repeat a repeat value", id="repeat-value"),
        pytest.param("observed", "observed means differ", id="observed"),
        pytest.param("key-in-params", "appears in the winner's effective_overrides", id="key"),
        pytest.param("reduce", "reduce is not one of mean, median", id="reduce"),
        pytest.param("single-member", "at least two", id="single-member"),
        pytest.param("member-keys", "member must have exactly", id="member-keys"),
    ],
)
def test_repeats_block_refuses_every_tamper(edit: str, error: str) -> None:
    payload = _tampered(edit)
    with pytest.raises(ValueError, match=error):
        _validate_repeats_block(payload, metric_value=_metric(_payload()), ranked_groups=_GROUPS)
