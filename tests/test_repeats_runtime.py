"""Repeats end to end: grouped draws, group winners, publication, evidence, and read views."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import optuna
import pytest
import yaml

from phasesweep import run_experiment
from phasesweep.config import (
    CategoricalParam,
    Experiment,
    IntParam,
    Phase,
    Repeats,
    Sampler,
    sampler_search_space,
)
from phasesweep.engine import read_status, read_winner
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.optuna import _build_sampler, _suggest
from phasesweep.engine.paths import _generation_winner_path, _phase_dir
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.read import _winner_view
from phasesweep.runtime.json import JsonObject
from tests.conftest import make_experiment, write_trainer
from tests.ledger_fixtures import published_generation_id
from tests.test_publication_transaction import _reseal_winner


def _seed_trainer(tmp_path: Path) -> Path:
    """Drop a trainer whose objective is ``x + 0.1 * seed + 0.01 * y``."""
    return write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--out")
        parser.add_argument("--x", type=int, default=0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--y", type=int, default=0)
        args, _ = parser.parse_known_args()
        print(f"x={args.x + 0.1 * args.seed + 0.01 * args.y}")
        """,
    )


def _repeats_phase(**overrides: object) -> Phase:
    """Return a seeded random phase that runs two configurations three times each."""
    base: dict[str, object] = {
        "name": "p",
        "n_trials": 6,
        "sampler": Sampler(type="random", seed=0),
        "search_space": {"x": IntParam(type="int", low=0, high=10)},
        "repeats": Repeats(key="seed", values=[0, 1, 2]),
    }
    base.update(overrides)
    return Phase.model_validate(base)


@pytest.mark.integration
def test_seeded_random_repeats_share_draws_per_slot_across_batches_and_workers(
    tmp_path: Path,
) -> None:
    """Every trial of a slot draws one configuration, however the trials were scheduled."""
    phase = _repeats_phase(
        search_space={"x": CategoricalParam(type="categorical", choices=list(range(1000)))}
    )
    assert phase.repeats is not None
    space = sampler_search_space(phase)

    def sampled(storage: Path, batches: list[int], n_jobs: int = 1) -> list[tuple[int, int]]:
        resolved = _resolve_storage(f"journal:///{storage}")
        for n_trials in batches:
            study = optuna.create_study(
                study_name="slots::p",
                storage=resolved,
                sampler=_build_sampler(phase.sampler, space, repeats=phase.repeats),
                load_if_exists=True,
            )
            study.optimize(
                lambda trial: float(sum(_suggest(trial, k, p) for k, p in space.items())),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
        loaded = optuna.load_study(study_name="slots::p", storage=resolved)
        return [(int(t.params["x"]), int(t.params["seed"])) for t in loaded.trials]

    expected = sampled(tmp_path / "single.journal", [6])
    assert [seed for _, seed in expected] == [0, 1, 2, 0, 1, 2]
    assert expected[0][0] == expected[1][0] == expected[2][0]
    assert expected[3][0] == expected[4][0] == expected[5][0]
    assert expected[0][0] != expected[3][0]
    assert expected == sampled(tmp_path / "ones.journal", [1, 1, 4])
    assert expected == sampled(tmp_path / "pairs.journal", [2, 2, 2])
    assert sorted(expected) == sorted(sampled(tmp_path / "parallel.journal", [6], n_jobs=3))


def _published_winner(experiment_root: Path, phase: str) -> dict[str, object]:
    """Return the published winner.yaml payload of one phase."""
    loaded = yaml.safe_load((experiment_root / phase / "winner.yaml").read_text())
    assert isinstance(loaded, dict)
    return loaded


def test_repeats_phase_runs_children_without_the_repeat_key(tmp_path: Path) -> None:
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_seed_trainer(tmp_path),
        phases=[
            _repeats_phase(),
            Phase(
                name="c",
                n_trials=1,
                inherits=["p"],
                sampler=Sampler(type="random", seed=0),
                search_space={"y": IntParam(type="int", low=0, high=3)},
            ),
        ],
    )

    winners = run_experiment(experiment)

    parent = winners["p"]
    assert parent.repeats is not None
    members = parent.repeats["members"]
    numbers = [member["trial_number"] for member in members]
    assert len(numbers) == 3 and numbers == sorted(numbers)
    # Seeded random runs one configuration per slot of three consecutive trials.
    assert numbers[-1] - numbers[0] == 2 and numbers[0] % 3 == 0
    assert [member["repeat"] for member in members] == [0, 1, 2]
    assert parent.trial_number == numbers[-1]
    assert parent.metric == statistics.fmean(member["metric"] for member in members)
    assert set(parent.params) == {"x"}
    assert "seed" not in parent.effective_overrides
    assert "seed" not in winners["c"].effective_overrides
    assert winners["c"].effective_overrides["x"] == parent.params["x"]
    assert _resolve_publication_pointer(experiment).state == "ok"

    root = Path(experiment.workdir) / experiment.experiment
    published = _published_winner(root, "p")
    assert published["trial_number"] == numbers[-1]
    assert published["confound"]["ranked_groups"] == [[0, 1, 2], [3, 4, 5]]
    with (_phase_dir(experiment, "p") / "trials.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {"repeat_group", "param:seed", "param:x"} <= set(rows[0])
    assert [row["repeat_group"] for row in rows] == ["2", "2", "2", "5", "5", "5"]
    with (_phase_dir(experiment, "c") / "trials.csv").open() as handle:
        assert "repeat_group" not in next(csv.reader(handle))

    resumed = run_experiment(experiment, from_phase="c")
    assert resumed["p"].repeats == parent.repeats
    assert read_status(experiment)["publication_integrity"] == "ok"


def _repeats_publication(tmp_path: Path) -> tuple[Experiment, str]:
    """Publish one seeded random repeats phase and return it with its generation id."""
    experiment = make_experiment(
        persistent=tmp_path, trainer=_seed_trainer(tmp_path), phases=[_repeats_phase()]
    )
    run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    return experiment, generation_id


def _winner_payload(experiment: Experiment, generation_id: str) -> JsonObject:
    """Return phase ``p``'s winner as one generation published it."""
    loaded = yaml.safe_load(_generation_winner_path(experiment, generation_id, "p").read_text())
    assert isinstance(loaded, dict)
    return loaded


def _members(winner: JsonObject) -> list[JsonObject]:
    """Return the ``repeats.members`` records of a winner payload."""
    repeats = winner["repeats"]
    assert isinstance(repeats, dict)
    members = repeats["members"]
    assert isinstance(members, list) and all(isinstance(m, dict) for m in members)
    return [m for m in members if isinstance(m, dict)]


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [
        pytest.param("member-metric", "mean of the member metrics differs", id="value"),
        pytest.param("member-order", "ascending and distinct", id="order"),
        pytest.param("representative", "disagrees with the winner's attempt_id", id="identity"),
        pytest.param("spread", "spread differs", id="spread"),
        pytest.param("absent", "repeats block is missing", id="absent"),
    ],
)
def test_group_winner_repeats_block_is_manifest_validated(
    tmp_path: Path, tamper: str, reason: str
) -> None:
    experiment, generation_id = _repeats_publication(tmp_path)
    winner = _winner_payload(experiment, generation_id)
    members = _members(winner)
    if tamper == "member-metric":
        members[0]["metric"] = -1.0
    elif tamper == "member-order":
        repeats = winner["repeats"]
        assert isinstance(repeats, dict)
        repeats["members"] = members[::-1]
    elif tamper == "representative":
        members[-1]["attempt_id"] = members[0]["attempt_id"]
    elif tamper == "spread":
        repeats = winner["repeats"]
        assert isinstance(repeats, dict) and isinstance(repeats["spread"], dict)
        repeats["spread"]["sd"] = 0.0
    else:
        del winner["repeats"]
    _reseal_winner(experiment, generation_id, winner)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "winner for phase 'p' has no valid repeats block" in pointer.error
    assert reason in pointer.error


def test_winner_view_projects_the_repeats_summary(tmp_path: Path) -> None:
    """Reads expose the reducer and spread, and a malformed block hides the winner."""
    experiment, generation_id = _repeats_publication(tmp_path)
    winner = _winner_payload(experiment, generation_id)
    metrics = [member["metric"] for member in _members(winner)]
    assert all(isinstance(value, float) for value in metrics)

    view = read_winner(experiment, "p")
    assert view is not None
    assert view.repeats is not None
    assert view.repeats["reduce"] == "mean"
    assert view.repeats["member_trials"] == [m["trial_number"] for m in _members(winner)]
    assert view.repeats["sd"] == pytest.approx(statistics.stdev(v for v in metrics))
    assert view.metric == view.repeats["mean"]
    assert view.params.keys() == {"x"}

    _members(winner)[1]["metric"] = 99.0
    assert _winner_view(winner, "p") is None


def test_carried_group_winner_must_agree_on_repeats(tmp_path: Path) -> None:
    """A carried group winner keeps its source generation's members verbatim."""
    experiment, source_generation = _repeats_publication(tmp_path)
    run_experiment(experiment)
    carrying_generation = published_generation_id(experiment)
    assert carrying_generation not in (None, source_generation)
    assert carrying_generation is not None
    winner = _winner_payload(experiment, carrying_generation)
    assert winner["generation_id"] == source_generation
    assert winner["repeats"] == _winner_payload(experiment, source_generation)["repeats"]
    # A non-representative member edit keeps the block self-consistent.
    _members(winner)[0]["attempt_id"] = "a" * 32
    _reseal_winner(experiment, carrying_generation, winner)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "disagrees with the result recorded by its source generation" in pointer.error
    assert "on repeats" in pointer.error


@pytest.mark.parametrize("reduce", ["mean", "median"])
def test_seed_audit_runs_one_configuration_once_per_repeat_value(
    tmp_path: Path, reduce: str
) -> None:
    """An empty search space with repeats measures seed noise for one fixed configuration."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_seed_trainer(tmp_path),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                sampler=Sampler(type="grid"),
                fixed_overrides={"x": 2},
                repeats=Repeats(key="seed", values=[0, 1, 2], reduce=reduce),
            )
        ],
    )

    winner = run_experiment(experiment)["p"]

    assert winner.repeats is not None
    values = [member["metric"] for member in winner.repeats["members"]]
    assert sorted(values) == pytest.approx([2.0, 2.1, 2.2])
    expected = statistics.fmean(values) if reduce == "mean" else statistics.median(values)
    assert winner.metric == expected
    assert winner.params == {}
    assert winner.effective_overrides == {"x": 2}
    assert winner.confound is not None
    assert winner.confound["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "only one repeat group was ranked",
    }
