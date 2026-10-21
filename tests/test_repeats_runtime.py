"""Repeats end to end: grouped draws, group winners, publication, evidence, and read views."""

from __future__ import annotations

import csv
import shutil
import statistics
from dataclasses import replace
from pathlib import Path

import optuna
import pytest
import yaml

import phasesweep.engine.artifacts as artifact_io
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    CategoricalParam,
    IntParam,
    Repeats,
    Sampler,
    sampler_search_space,
)
from phasesweep.engine.confound import _validate_confound_block
from phasesweep.engine.errors import WinnerIntegrityError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.optuna import _build_sampler, _suggest
from phasesweep.engine.paths import (
    _generation_dir,
    _generation_summary_path,
    _generation_winner_path,
    _phase_dir,
)
from phasesweep.engine.publication import _resolve_publication_pointer
from phasesweep.engine.read import _winner_view, read_result, read_status, read_trial_tables
from phasesweep.engine.repeats import ReduceName, _validate_repeats_block, reduce_values, spread
from phasesweep.engine.resume import _published_for_resume
from phasesweep.runtime.json import JsonObject
from tests.conftest import make_experiment, write_trainer
from tests.ledger_fixtures import published_generation_id, reseal_winner


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
    assert expected == sampled(tmp_path / "parallel.journal", [6], n_jobs=3)


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


def test_child_repeats_a_key_its_parent_searched(tmp_path: Path) -> None:
    """A repeat value replaces the inherited value of a key an ancestor sampled, and is not exported."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_seed_trainer(tmp_path),
        phases=[
            Phase(
                name="p",
                n_trials=3,
                sampler=Sampler(type="random", seed=0),
                search_space={"y": IntParam(type="int", low=0, high=3)},
            ),
            Phase(
                name="c",
                n_trials=2,
                inherits=["p"],
                sampler=Sampler(type="grid"),
                repeats=Repeats(key="y", values=[10, 20]),
            ),
        ],
    )

    winners = run_experiment(experiment)

    assert set(winners["p"].params) == {"y"}
    child = winners["c"]
    assert child.repeats is not None
    # The trainer scores 0.01 * y, so a member that ran the parent's y would score at most 0.03.
    assert {member["repeat"]: member["metric"] for member in child.repeats["members"]} == (
        pytest.approx({10: 0.1, 20: 0.2})
    )
    assert child.params == {}
    assert "y" not in child.effective_overrides


def test_dry_run_placeholder_winner_drops_the_repeat_key(tmp_path: Path) -> None:
    """A previewed repeats phase passes on neither its repeat values nor the value it replaced."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_seed_trainer(tmp_path),
        phases=[
            Phase(name="p", n_trials=1, fixed_overrides={"seed": 7, "x": 1}),
            _repeats_phase(
                name="c",
                inherits=["p"],
                n_trials=3,
                search_space={"y": IntParam(type="int", low=0, high=3)},
            ),
            Phase(name="g", n_trials=1, inherits=["c"]),
        ],
    )

    winners = run_experiment(experiment, dry_run=True)

    assert winners["p"].effective_overrides == {"seed": 7, "x": 1}
    assert set(winners["c"].params) == {"y"}
    assert set(winners["c"].effective_overrides) == {"x", "y"}
    assert set(winners["g"].effective_overrides) == {"x", "y"}


def _repeats_publication(tmp_path: Path, **phase_overrides: object) -> tuple[Experiment, str]:
    """Publish one seeded random repeats phase and return it with its generation id."""
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=_seed_trainer(tmp_path),
        phases=[_repeats_phase(**phase_overrides)],
    )
    run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    return experiment, generation_id


@pytest.fixture(scope="module")
def readonly_repeats_publication(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Experiment, str]:
    """Publish one CPU-only repeat group tree for consumers that never mutate its files."""
    root = tmp_path_factory.mktemp("repeats-read")
    locks = root / "locks"
    locks.mkdir(mode=0o700)
    locks.chmod(0o700)
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("PHASESWEEP_HOME", raising=False)
        patch.setenv("XDG_STATE_HOME", str(root / "state-home"))
        patch.setenv("XDG_CACHE_HOME", str(root / "cache-home"))
        patch.setenv("PHASESWEEP_LOCK_DIR", str(locks))
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        patch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
        patch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: False)
        return _repeats_publication(root)


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
    reseal_winner(experiment, generation_id, winner)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "winner for phase 'p' has no valid repeats block" in pointer.error
    assert reason in pointer.error


_SKEWED_REPEATS = Repeats(key="seed", values=[0, 1, 5])
"""Three repeat values whose member metrics have different means and medians."""


def _block(payload: JsonObject, name: str) -> JsonObject:
    """Return one nested mapping of a winner payload."""
    block = payload[name]
    assert isinstance(block, dict)
    return block


def _forge_axis_mismatch(winner: JsonObject, tamper: str) -> float:
    """Rewrite a skewed three-repeat winner so it disagrees with the configured axis.

    Every derived field is recomputed, so the block stays consistent with itself and
    with the confound block, and only a comparison with the configured axis can refuse it.

    :param JsonObject winner: The published winner payload, edited in place.
    :param str tamper: ``reduce``, ``foreign-values`` or ``missing-member``.
    :return float: The forged aggregate, which the summary must also record.
    """
    repeats = _block(winner, "repeats")
    reduce: ReduceName = "mean"
    if tamper == "reduce":
        reduce = "median"
        repeats["reduce"] = reduce
    elif tamper == "foreign-values":
        for member, foreign in zip(_members(winner), ("a", "b", "c"), strict=True):
            member["repeat"] = foreign
    else:
        confound = _block(winner, "confound")
        ranked, groups = confound["ranked_trials"], confound["ranked_groups"]
        assert isinstance(ranked, list) and isinstance(groups, list)
        assert all(isinstance(group, list) for group in groups)
        dropped = {group[0] for group in groups if isinstance(group, list)}
        repeats["members"] = _members(winner)[1:]
        confound["ranked_trials"] = [number for number in ranked if number not in dropped]
        confound["ranked_groups"] = [
            [number for number in group if number not in dropped]
            for group in groups
            if isinstance(group, list)
        ]
        counts = _block(_block(_block(confound, "checks"), "survivorship"), "detail")
        counts["ranked"] = len(ranked) - len(dropped)
    metrics = [member["metric"] for member in _members(winner)]
    assert all(isinstance(value, float) for value in metrics)
    values = [value for value in metrics if isinstance(value, float)]
    aggregate = reduce_values(values, reduce)
    repeats["spread"] = dict(spread(values))
    metric = _block(winner, "metric")
    name = next(key for key in metric if key != "goal")
    metric[name] = aggregate
    return aggregate


def _publish_forgery(
    experiment: Experiment, generation_id: str, winner: JsonObject, aggregate: float
) -> None:
    """Record a forged aggregate in the summary, then re-anchor the winner, manifest and pointer."""
    summary_path = _generation_summary_path(experiment, generation_id)
    summary = yaml.safe_load(summary_path.read_text())
    next(item for item in summary["phases"] if item["name"] == "p")["metric"] = aggregate
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    reseal_winner(experiment, generation_id, winner)


_AXIS_MISMATCHES = [
    pytest.param(
        "reduce",
        "repeats reduce 'median' is not the configured repeats.reduce 'mean'",
        id="reduce",
    ),
    pytest.param(
        "foreign-values",
        "repeats member values are not the configured repeats.values",
        id="foreign-values",
    ),
    pytest.param(
        "missing-member",
        "repeats has 2 members but repeats.values configures 3",
        id="missing-member",
    ),
]


@pytest.mark.parametrize(("tamper", "reason"), _AXIS_MISMATCHES)
def test_manifest_binds_the_repeats_block_to_the_snapshot_axis(
    tmp_path: Path, tamper: str, reason: str
) -> None:
    """A resealed, self-consistent block that is not the snapshot phase's axis fails publication.

    With three repeat values the forged reducer changes the aggregate, so the winner and the
    summary are rewritten to agree with it and only the axis comparison can refuse the block.
    """
    experiment, generation_id = _repeats_publication(tmp_path, repeats=_SKEWED_REPEATS)
    winner = _winner_payload(experiment, generation_id)
    aggregate = _forge_axis_mismatch(winner, tamper)
    _publish_forgery(experiment, generation_id, winner, aggregate)

    pointer = _resolve_publication_pointer(experiment)
    assert pointer.state == "failed"
    assert pointer.error is not None
    assert "winner for phase 'p' has no valid repeats block" in pointer.error
    assert reason in pointer.error
    assert read_status(experiment)["publication_integrity"] == "failed"
    assert read_result(experiment, phase_names=["p"])[1] == []


@pytest.mark.parametrize(("tamper", "reason"), _AXIS_MISMATCHES)
def test_load_winner_binds_the_repeats_block_to_the_configured_axis(
    tmp_path: Path, tamper: str, reason: str
) -> None:
    """The winner a skipped phase loads must carry the repeat axis the current config declares."""
    experiment, generation_id = _repeats_publication(tmp_path, repeats=_SKEWED_REPEATS)
    winner = _winner_payload(experiment, generation_id)
    aggregate = _forge_axis_mismatch(winner, tamper)
    published = _published_for_resume(experiment)
    confound = _validate_confound_block(
        winner["confound"], winner_trial=published.winners["p"].source.trial_number
    )
    forged_winner = replace(
        published.winners["p"],
        payload=winner,
        metric=aggregate,
        confound=confound,
        repeats=_validate_repeats_block(
            winner, metric_value=aggregate, ranked_groups=confound["ranked_groups"]
        ),
    )
    forged = replace(published, winners={"p": forged_winner})

    with pytest.raises(WinnerIntegrityError) as raised:
        artifact_io._load_winner(experiment, experiment.phases[0], {}, publication=forged)

    assert "invalid or incomplete for skipped phase 'p'" in str(raised.value)
    assert reason in str(raised.value)


def test_winner_view_projects_the_repeats_summary(
    readonly_repeats_publication: tuple[Experiment, str],
) -> None:
    """Reads expose the reducer and spread, and a malformed block hides the winner."""
    experiment, generation_id = readonly_repeats_publication
    winner = _winner_payload(experiment, generation_id)
    metrics = [member["metric"] for member in _members(winner)]
    assert all(isinstance(value, float) for value in metrics)

    _, (view,) = read_result(experiment, phase_names=["p"])
    assert view is not None
    assert view.repeats is not None
    assert view.repeats["reduce"] == "mean"
    assert view.repeats["member_trials"] == [m["trial_number"] for m in _members(winner)]
    assert view.repeats["sd"] == pytest.approx(statistics.stdev(v for v in metrics))
    assert view.metric == view.repeats["mean"]
    assert view.params.keys() == {"x"}

    _members(winner)[1]["metric"] = 99.0
    assert _winner_view(winner, "p") is None


def test_trial_tables_label_repeat_groups_and_star_every_winning_member(
    readonly_repeats_publication: tuple[Experiment, str],
) -> None:
    """A repeats phase's table groups trials as trials.csv does and stars the whole winning group."""
    experiment, _ = readonly_repeats_publication
    _, (view,) = read_result(experiment, phase_names=["p"])
    assert view is not None and view.repeats is not None

    (table,) = read_trial_tables(experiment).phases

    assert table.columns == (
        "trial",
        "state",
        "value",
        "params",
        "duration_s",
        "repeat_group",
        "winner",
    )
    assert [row["repeat_group"] for row in table.rows] == [2, 2, 2, 5, 5, 5]
    params = [row["params"] for row in table.rows]
    assert all(isinstance(p, dict) for p in params)
    assert [p.get("seed") for p in params if isinstance(p, dict)] == [0, 1, 2, 0, 1, 2]
    starred = [row["trial"] for row in table.rows if row["winner"] == "*"]
    assert len(starred) == 3
    assert starred == view.repeats["member_trials"]
    with (_phase_dir(experiment, "p") / "trials.csv").open() as handle:
        assert [row["repeat_group"] for row in csv.DictReader(handle)] == [
            str(row["repeat_group"]) for row in table.rows
        ]


def test_carried_repeat_winner_reads_without_older_publication(tmp_path: Path) -> None:
    """A carried repeat group's current result embeds every member's identity."""
    published, source_generation = _repeats_publication(tmp_path)
    # A new phase gives the next run trials to launch, so its generation carries p's winner.
    added = Phase.model_validate(
        {
            "name": "q",
            "n_trials": 1,
            "sampler": Sampler(type="random", seed=0),
            "fixed_overrides": {"y": 1},
        }
    )
    experiment = published.model_copy(update={"phases": [*published.phases, added]})
    carried = run_experiment(experiment)["p"]
    assert carried.generation_id == source_generation
    assert published_generation_id(experiment) != source_generation
    assert carried.repeats is not None
    expected = read_result(experiment)
    shutil.rmtree(_generation_dir(experiment, source_generation))

    assert read_result(experiment) == expected
    assert run_experiment(experiment)["p"].repeats == carried.repeats


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
    # A grid phase has no slots, so its one configuration's three trials form one group.
    with (_phase_dir(experiment, "p") / "trials.csv").open() as handle:
        assert [row["repeat_group"] for row in csv.DictReader(handle)] == ["2", "2", "2"]
    assert winner.confound is not None
    assert winner.confound["checks"]["tie"] == {
        "verdict": "n/a",
        "reason": "only one repeat group was ranked",
    }
