"""Public shape contract for ``experiment_status``.

``experiment_status`` is a documented package-root API (docs/development.md)
and ``phasesweep status`` renders its payload verbatim, so its key set is a
contract for downstream consumers.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import phasesweep.engine.read as engine_read
from phasesweep import experiment_status, run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Sampler
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_dir,
    _generation_path,
    _generation_winner_path,
)
from phasesweep.engine.read import read_status, read_trial_tables
from phasesweep.errors import PhaseSweepError
from phasesweep.evidence.models import JsonExtractor, LogRegexExtractor
from phasesweep.runtime.json import JsonValue
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    new_phase_study,
    write_experiment_config,
    write_trainer,
)
from tests.ledger_fixtures import materialize, published_generation_id

EXPERIMENT_STATUS_KEYS = [
    "kind",
    "experiment",
    "workdir",
    "ledger_path",
    "current_generation_id",
    "published_generation_id",
    "represented_generation_id",
    "is_published",
    "publication_integrity",
    "phases",
]
"""Exact ordered key set of an experiment status payload."""

FAILED_PUBLICATION_STATUS_KEYS = [
    "kind",
    "experiment",
    "workdir",
    "ledger_path",
    "current_generation_id",
    "published_generation_id",
    "represented_generation_id",
    "is_published",
    "publication_integrity",
    "publication_error",
    "phases",
]
"""Ordered key set when the recorded publication no longer validates.

``publication_error`` is the one conditional key: it appears only alongside
``publication_integrity: "failed"`` or ``"permission_denied"`` so a healthy
payload carries no empty error field.
"""

PHASE_STATUS_KEYS = {
    "winner_generation_id",
    "winner_is_published",
    "winner_config_matches_current",
    "trials",
    "n_trials",
    "generation_trials",
    "published_study_unavailable",
    "trial_data_available",
    "name",
    "winner",
}
"""Exact key set of one phase payload inside an experiment status payload."""

# read_status computes these for the MCP read view; experiment_status
# deliberately does not republish them under the path-bearing CLI contract.
# ``result_phase_plan`` is the represented generation's own phase plan, which
# the CLI reads straight from the generation summary in ``show-winners``
# instead.
READ_STATUS_ONLY_KEYS = {
    "metric",
    "result_context",
    "published_config_matches_current",
    "result_phase_plan",
    "summary_present",
}

READ_STATUS_ONLY_PHASE_KEYS = {
    "phase",
    "winner_present",
    "running_attempts",
    "reread_pending",
}
"""Phase keys that exist only in the path-free read_status view.

``running_attempts`` is the identity of each RUNNING row in the same storage
snapshot the counts came from. The MCP terminal snapshot consumes it so it
never has to reread a study to reconcile RUNNING trials; the CLI phase payload
keeps exactly ``PHASE_STATUS_KEYS`` and republishes none of it. ``reread_pending``
is MCP's account of a fill from kept evidence, which the CLI prints as the phase's
refusal instead.
"""


def _published(tmp_path: Path) -> Experiment:
    """Return the config that reads the golden fixture's two-trial publication."""
    return materialize("current-journal", tmp_path, mode="tree").experiment


@contextmanager
def _claim_after_pointer_read(experiment: Experiment) -> Iterator[None]:
    """Claim the next generation just after inspection captures the current pointer once."""
    pointer = _generation_path(experiment)
    _generation_dir(_experiment_dir(experiment), "next-generation").mkdir()
    original_read = engine_read.read_yaml_or_none
    reads = 0

    def claim_after_read(path: Path) -> JsonValue:
        nonlocal reads
        payload = original_read(path)
        if path == pointer:
            reads += 1
            pointer.write_text(yaml.safe_dump({"generation_id": "next-generation"}))
        return payload

    with pytest.MonkeyPatch.context() as capture:
        capture.setattr(engine_read, "read_yaml_or_none", claim_after_read)
        yield
    assert reads == 1


@pytest.mark.parametrize("arguments", [("status",), ("status", "--trials"), ("show-winners",)])
def test_inspection_uses_one_current_pointer_capture(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    """A concurrent generation claim cannot split inspection identity and results."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    generation = published_generation_id(experiment)
    assert generation is not None
    with _claim_after_pointer_read(experiment):
        result = CliRunner().invoke(cli, [*arguments, str(materialized.config_path)])
        assert result.exit_code == 0, result.output
    if arguments == ("status",):
        payload = yaml.safe_load(result.output)
        assert (
            payload["current_generation_id"] == payload["represented_generation_id"] == generation
        )
        assert payload["phases"][0]["generation_trials"] == {"COMPLETE": 2}
    else:
        assert generation in result.output
    pinned = experiment_status(experiment, generation_id=generation)
    assert pinned["current_generation_id"] == "next-generation"
    assert pinned["represented_generation_id"] == generation


def test_status_counts_a_child_in_the_study_its_parents_latest_winner_addresses(
    tmp_path: Path,
) -> None:
    """Status reads the pointer once and counts a child whose parent's top-up changed its winner in the study that winner addresses, not the earlier one, even when a generation is claimed right after the read."""

    def plan(choices: list[int], child_trials: int) -> Experiment:
        return make_experiment(
            persistent=tmp_path,
            phases=[
                Phase(
                    name="parent",
                    n_trials=len(choices),
                    gpu_policy="none",
                    sampler=Sampler(type="grid"),
                    search_space={"x": CategoricalParam(type="categorical", choices=choices)},
                ),
                Phase(
                    name="child",
                    n_trials=child_trials,
                    gpu_policy="none",
                    inherits=["parent"],
                    sampler=Sampler(type="random", seed=0),
                    search_space={"y": IntParam(type="int", low=0, high=3)},
                ),
            ],
        )

    run_experiment(plan([3, 5], child_trials=2))
    grown = plan([3, 5, 1], child_trials=1)
    run_experiment(grown)
    config = write_experiment_config(tmp_path / "grown.yaml", grown)
    with _claim_after_pointer_read(grown):
        result = CliRunner().invoke(cli, ["status", str(config)])
        assert result.exit_code == 0, result.output
    assert "different config" not in result.output
    child = yaml.safe_load(result.output)["phases"][1]
    assert child["name"] == "child"
    assert child["trials"] == {"COMPLETE": 1}


@pytest.mark.integration
def test_status_uses_the_parent_winner_the_current_reader_replays(tmp_path: Path) -> None:
    """A restored reader reuses its child study even after another reader publishes a different parent."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        parser.add_argument("--x", type=int)
        parser.add_argument("--y", type=int, default=0)
        args = parser.parse_args()
        Path(args.out).write_text(json.dumps({"score": args.x, "alt": -args.x}))
        """,
    )
    parent = Phase(
        name="parent",
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )

    def plan(key: str, child_y: int) -> Experiment:
        return make_experiment(
            persistent=tmp_path,
            trainer=trainer,
            metric=Metric(
                name="score",
                goal="minimize",
                extractor=JsonExtractor(type="json", path="r.json", key=key),
            ),
            phases=[
                parent,
                Phase(
                    name="child",
                    n_trials=1,
                    gpu_policy="none",
                    inherits=["parent"],
                    fixed_overrides={"y": child_y},
                ),
            ],
        )

    original = plan("score", 0)
    original_winners = run_experiment(original)
    assert original_winners["parent"].params == {"x": 1}
    alternate_winners = run_experiment(plan("alt", 7))
    assert alternate_winners["parent"].params == {"x": 2}
    config = write_experiment_config(tmp_path / "original.yaml", original)

    status = CliRunner().invoke(cli, ["status", str(config)])
    table = CliRunner().invoke(cli, ["status", str(config), "--trials"])

    assert status.exit_code == table.exit_code == 0, (status.output, table.output)
    assert "was created with a different config" not in status.output
    assert "was created with a different config" not in table.output
    child = yaml.safe_load(status.output)["phases"][1]
    assert child["trials"] == {"COMPLETE": 1}
    assert child["winner_config_matches_current"] is False
    assert "child: 1 of 1 trial" in table.output, table.output

    conflict = write_experiment_config(tmp_path / "conflict.yaml", plan("score", 7))
    for options in ([], ["--trials"]):
        diagnosed = CliRunner().invoke(cli, ["status", str(conflict), *options])
        assert diagnosed.exit_code == 0, diagnosed.output
        assert "was created with a different config" in " ".join(diagnosed.output.split())


def _grid_parent(name: str, key: str, choices: list[int]) -> Phase:
    return Phase(
        name=name,
        n_trials=len(choices),
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={key: CategoricalParam(type="categorical", choices=choices)},
    )


@pytest.mark.integration
def test_status_waits_for_a_parent_top_up_before_applying_a_child_abort(tmp_path: Path) -> None:
    """Both status formats keep failed rows visible while a parent top-up can select another child study."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--out")
        parser.add_argument("--x", type=int)
        parser.add_argument("--y", type=int)
        args = parser.parse_args()
        if args.y is not None and args.x == 1:
            raise SystemExit(3)
        print(f"x={args.x}")
    """,
    )
    child = Phase(
        name="child",
        n_trials=2,
        gpu_policy="none",
        inherits=["parent"],
        max_consecutive_failures=1,
        sampler=Sampler(type="random", seed=0),
        search_space={"y": IntParam(type="int", low=0, high=3)},
    )

    def plan(choices: list[int]) -> Experiment:
        return make_experiment(
            persistent=tmp_path,
            trainer=trainer,
            phases=[_grid_parent("parent", "x", choices), child],
        )

    assert run_experiment(plan([3, 5]))["parent"].params == {"x": 3}
    with pytest.raises(PhaseSweepError, match="Phase 'child' aborted"):
        run_experiment(plan([3, 5, 1]))
    current = plan([3, 5, 1, 0])
    config = write_experiment_config(tmp_path / "current.yaml", current)
    for options in ([], ["--trials"]):
        shown = CliRunner().invoke(cli, ["status", str(config), *options])
        assert shown.exit_code == 0, shown.output
        assert "shows counts using the last recorded values from 'parent'" in " ".join(
            shown.output.split()
        )
        assert "previously aborted" not in shown.output
        assert "FAIL" in shown.output
    assert run_experiment(current)["child"].effective_overrides["x"] == 0


@pytest.mark.integration
def test_status_combines_a_replayed_parent_with_a_parent_still_training(tmp_path: Path) -> None:
    """A conditional child count uses the replayed value from each settled parent."""
    trainer = write_trainer(
        tmp_path,
        """
        import argparse, json
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        for key in ("x", "b", "y"):
            parser.add_argument("--" + key, type=int, default=0)
        args = parser.parse_args()
        Path(args.out).write_text(json.dumps({"score": args.x + args.b, "alt": -args.x + args.b}))
    """,
    )

    def plan(key: str, choices: list[int], target: int) -> Experiment:
        return make_experiment(
            persistent=tmp_path,
            trainer=trainer,
            metric=Metric(
                name="score",
                goal="minimize",
                extractor=JsonExtractor(type="json", path="r.json", key=key),
            ),
            phases=[
                _grid_parent("a", "x", [1, 2]),
                _grid_parent("b", "b", choices),
                Phase(
                    name="child",
                    n_trials=target,
                    gpu_policy="none",
                    inherits=["a", "b"],
                    sampler=Sampler(type="random", seed=0),
                    search_space={"y": IntParam(type="int", low=0, high=3)},
                ),
            ],
        )

    original = run_experiment(plan("score", [3, 5], 2))["child"]
    assert run_experiment(plan("alt", [3, 5], 1))["a"].params == {"x": 2}
    current = plan("score", [3, 5, 6], 2)
    config = write_experiment_config(tmp_path / "current.yaml", current)
    status = CliRunner().invoke(cli, ["status", str(config)])
    table = CliRunner().invoke(cli, ["status", str(config), "--trials"])
    assert status.exit_code == table.exit_code == 0, (status.output, table.output)
    assert yaml.safe_load(status.output)["phases"][-1]["trials"] == {"COMPLETE": 2}
    assert "child: 2 of 2 trials" in table.output
    assert "last recorded values from 'b'" in status.output
    assert run_experiment(current)["child"].attempt_id == original.attempt_id


def test_reordered_parents_keep_the_published_result_matching_the_config(tmp_path: Path) -> None:
    """Reordering a child's ``inherits`` changes no value it receives, so ``status``, ``status --trials`` and a rerun all treat the published generation as the current config's result, and nothing launches."""
    calls = tmp_path / "calls.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys
        from pathlib import Path
        with Path({str(calls)!r}).open("a") as log:
            log.write(" ".join(sys.argv[1:]) + "\\n")
        print("x=1")
        """,
    )

    def plan(parents: list[str]) -> Experiment:
        def phase(name: str, key: str, inherits: list[str]) -> Phase:
            return Phase(
                name=name,
                n_trials=2,
                gpu_policy="none",
                inherits=inherits,
                sampler=Sampler(type="grid"),
                search_space={key: CategoricalParam(type="categorical", choices=[1, 2])},
            )

        return make_experiment(
            persistent=tmp_path,
            trainer=trainer,
            phases=[phase("lr", "a", []), phase("depth", "b", []), phase("final", "c", parents)],
        )

    run_experiment(plan(["lr", "depth"]))
    published = published_generation_id(plan(["lr", "depth"]))
    launched = calls.read_text()
    reordered = plan(["depth", "lr"])
    config = write_experiment_config(tmp_path / "reordered.yaml", reordered)

    status = CliRunner().invoke(cli, ["status", str(config)])
    trials = CliRunner().invoke(cli, ["status", str(config), "--trials"])
    run_experiment(reordered)

    assert status.exit_code == 0, status.output
    phases = yaml.safe_load(status.output)["phases"]
    assert [phase["winner_config_matches_current"] for phase in phases] == [True, True, True]
    assert trials.exit_code == 0, trials.output
    assert f"published generation {published}" in trials.output
    assert "does not match the current config" not in trials.output
    assert published_generation_id(reordered) == published
    assert calls.read_text() == launched


def test_experiment_status_shape_is_pinned(tmp_path: Path) -> None:
    """A standalone experiment payload carries generation identity plus phases."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment

    payload = experiment_status(experiment)

    assert list(payload) == EXPERIMENT_STATUS_KEYS
    assert not READ_STATUS_ONLY_KEYS & set(payload)
    assert payload["ledger_path"] == str(materialized.ledger_path.resolve())
    assert "ledger_path" not in read_status(experiment)
    assert payload["kind"] == "experiment"
    assert payload["is_published"] is True
    assert payload["publication_integrity"] == "ok"
    assert payload["published_generation_id"] == payload["current_generation_id"]
    assert payload["represented_generation_id"] == payload["published_generation_id"]

    (phase,) = payload["phases"]
    assert set(phase) == PHASE_STATUS_KEYS
    assert not READ_STATUS_ONLY_PHASE_KEYS & set(phase)
    assert phase["name"] == "p"
    assert phase["winner"] is not None
    assert phase["trials"] == {"COMPLETE": 2}


def test_read_status_only_phase_keys_never_reach_the_cli_contract(tmp_path: Path) -> None:
    """The path-free phase view carries running attempts; the CLI view does not.

    Both directions matter. The MCP terminal snapshot depends on
    ``running_attempts`` arriving from the same tolerant read as the counts,
    and ``experiment_status`` is a pinned public contract that must not grow a
    key because of it.
    """
    experiment = _published(tmp_path)

    (read_phase,) = read_status(experiment)["phases"]
    (cli_phase,) = experiment_status(experiment)["phases"]

    assert set(read_phase) >= READ_STATUS_ONLY_PHASE_KEYS
    assert read_phase["trial_data_available"] is True
    assert read_phase["running_attempts"] == []
    assert set(cli_phase) == PHASE_STATUS_KEYS


def _mixed_outcome_experiment(tmp_path: Path) -> Experiment:
    """Build a three-trial grid whose trials finish COMPLETE, FAIL and INFEASIBLE in turn."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import sys

        x = next(int(arg.removeprefix("--x=")) for arg in sys.argv if arg.startswith("--x="))
        if x == 2:
            sys.exit(1)
        print(f"x={5.0 if x == 3 else 0.5}")
        """,
    )
    phase = Phase(
        name="p",
        n_trials=3,
        max_consecutive_failures=3,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=3)},
    )
    metric = Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase], metric=metric)


def test_status_counts_follow_the_trial_table_state_names(tmp_path: Path) -> None:
    """A trial that completed out of bounds reads INFEASIBLE in the counts, as in ``status --trials``.

    ``COMPLETE`` counts the trials that finished within every bound and constraint.
    Infeasible and failed trials are finished too and count toward ``n_trials``.
    """
    experiment = _mixed_outcome_experiment(tmp_path)
    run_experiment(experiment)

    (phase_status,) = experiment_status(experiment)["phases"]
    (read_phase,) = read_status(experiment)["phases"]

    expected = {"COMPLETE": 1, "INFEASIBLE": 1, "FAIL": 1}
    assert phase_status["trials"] == expected
    assert phase_status["generation_trials"] == expected
    assert read_phase["trials"] == expected
    assert next(iter(phase_status)) == "name"
    (table,) = read_trial_tables(experiment).phases
    assert sorted(str(row["state"]) for row in table.rows) == sorted(expected)


def test_status_gives_the_refusal_a_run_gives_for_a_lowered_target(tmp_path: Path) -> None:
    """A phase whose study accepted a larger ``n_trials`` shows the run's refusal and fix in both status forms, not a run command that would be refused."""
    run_experiment(make_experiment(persistent=tmp_path, n_trials=3))
    lowered = make_experiment(persistent=tmp_path, n_trials=2)
    config = write_experiment_config(tmp_path / "lowered.yaml", lowered)
    message = "Phase 'p' has already accepted a target of 3 finished trials; the config requests 2."
    fix = "Set phases[p].n_trials to at least 3."

    with pytest.raises(PhaseSweepError, match=re.escape(message)):
        run_experiment(lowered)
    status = CliRunner().invoke(cli, ["status", str(config)])
    trials = CliRunner().invoke(cli, ["status", str(config), "--trials"])

    assert status.exit_code == 0, status.output
    assert status.output.startswith(f"# {message}\n# fix: {fix}\n")
    assert trials.exit_code == 0, trials.output
    assert " ".join(trials.output.split()).count(f"{message} fix: {fix}") == 1


def test_published_completion_counts_every_measured_trial(tmp_path: Path) -> None:
    """``measured_trials`` counts the INFEASIBLE trial with the COMPLETE one, and a rerun replays that record."""
    experiment = _mixed_outcome_experiment(tmp_path)
    run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    winner_path = _generation_winner_path(experiment, generation_id, "p")
    published = winner_path.read_bytes()
    completion = yaml.safe_load(published)["completion"]
    (phase_status,) = read_status(experiment)["phases"]

    assert completion["finished_trials"] == 3
    assert phase_status["trials"]["COMPLETE"] == phase_status["trials"]["INFEASIBLE"] == 1
    assert completion["measured_trials"] == 2

    replayed = run_experiment(experiment)["p"]

    assert replayed.completion == completion
    assert published_generation_id(experiment) == generation_id
    assert winner_path.read_bytes() == published


def test_status_shape_reports_a_corrupt_publication_without_fabricating_results(
    tmp_path: Path,
) -> None:
    """A publication that no longer validates adds one key, ``publication_error``, and invents nothing.

    The payload for a corrupt publication must differ from the one for a tree
    that never published; if the two were identical, the operator's natural
    next move would be to re-run over the evidence.
    """
    experiment = _published(tmp_path)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    payload = experiment_status(experiment)

    assert list(payload) == FAILED_PUBLICATION_STATUS_KEYS
    assert not READ_STATUS_ONLY_KEYS & set(payload)
    assert payload["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in payload["publication_error"]
    assert payload["published_generation_id"] is None
    assert payload["represented_generation_id"] is None
    assert payload["is_published"] is False
    assert payload["phases"][0]["winner"] is None


@pytest.mark.parametrize(("flags", "trial_text"), [((), "RUNNING: 1"), (("--trials",), "RUNNING")])
def test_status_keeps_running_trials_visible_when_lock_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
    trial_text: str,
) -> None:
    """Print the readable ledger without idle-run advice when the lock cannot be probed."""
    experiment = make_experiment(persistent=tmp_path)
    new_phase_study(experiment, "p").ask()
    config = write_experiment_config(tmp_path / "config.yaml", experiment)
    missing_locks = tmp_path / "missing-locks"
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(missing_locks))

    assert invoke_cli_boundary(["status", str(config), *flags], monkeypatch) == 0

    captured = capsys.readouterr()
    assert trial_text in captured.out
    assert "No run is active" not in captured.out
    assert not captured.err
    assert not missing_locks.exists()


def test_in_memory_status_names_no_ledger(tmp_path: Path) -> None:
    """``storage: null`` has no ledger file, so ``ledger_path`` is null."""
    experiment = make_experiment(workdir=tmp_path / "runs", trial_command="echo x=0.5 {overrides}")
    run_experiment(experiment)
    payload = experiment_status(experiment)

    assert list(payload) == EXPERIMENT_STATUS_KEYS
    assert payload["ledger_path"] is None


def test_in_memory_trial_table_points_at_the_last_runs_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``storage: null`` the trial table names the ``trials.csv`` the last run wrote."""
    experiment = make_experiment(workdir=tmp_path / "runs", trial_command="echo x=0.5 {overrides}")
    run_experiment(experiment)
    config = write_experiment_config(tmp_path / "config.yaml", experiment)
    monkeypatch.chdir(tmp_path)

    assert invoke_cli_boundary(["status", str(config), "--trials"], monkeypatch) == 0

    assert (tmp_path / "runs" / experiment.experiment / "p" / "trials.csv").is_file()
    lines = capsys.readouterr().out.splitlines()
    heading = lines.index("p: no trial ledger (storage: null), objective (lower is better)")
    assert lines[heading + 1] == (
        f"  the trials of this phase's last run are in runs/{experiment.experiment}/p/trials.csv"
    )


def test_trial_table_prints_float_params_of_10000_or_more_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Float params from 10000 up print in plain notation, and smaller ones keep four significant digits."""
    values = [16384.0, 100000.0, 12345.678, 9999.9, 0.0003162, 1.5e22]
    trainer = write_trainer(tmp_path / "trainer.py", 'print("x=0.5")\n')
    phase = Phase(
        name="p",
        n_trials=len(values),
        sampler=Sampler(type="grid", seed=0),
        search_space={"w": CategoricalParam(type="categorical", choices=values)},
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    run_experiment(experiment)
    config = write_experiment_config(tmp_path / "config.yaml", experiment)

    assert invoke_cli_boundary(["status", str(config), "--trials"], monkeypatch) == 0

    # Each row ends with the w column, then the state and the time.
    shown = {
        line.split()[-3] for line in capsys.readouterr().out.splitlines() if "COMPLETE" in line
    }
    assert shown == {
        "16384",
        "100000",
        "12345.678",
        "9999.9",
        "0.0003162",
        "15000000000000000000000",
    }
