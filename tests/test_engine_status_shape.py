"""Public shape contract for ``experiment_status``.

``experiment_status`` is a documented package-root API (docs/development.md)
and ``phasesweep status`` renders its payload verbatim, so its key set is a
contract for downstream consumers.
"""

from __future__ import annotations

from pathlib import Path

from phasesweep import experiment_status, run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.paths import _generation_winner_path
from phasesweep.engine.read import read_status, read_trial_tables
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import make_experiment, write_trainer
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
    "trials",
    "running",
    "n_trials",
    "completed",
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
}
"""Phase keys that exist only in the path-free read_status view.

``running_attempts`` is the identity of each RUNNING row in the same storage
snapshot the counts came from. The MCP terminal snapshot consumes it so it
never has to reread a study to reconcile RUNNING trials; the CLI phase payload
keeps exactly ``PHASE_STATUS_KEYS`` and republishes none of it.
"""


def _published(tmp_path: Path) -> Experiment:
    """Return the config that reads the golden fixture's two-trial publication."""
    return materialize("current-journal", tmp_path, mode="tree").experiment


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
    assert phase["completed"] == 2


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


def test_status_counts_follow_the_trial_table_state_names(tmp_path: Path) -> None:
    """A trial that completed out of bounds reads INFEASIBLE in the counts, as in ``status --trials``.

    ``completed`` is the COMPLETE count: trials that finished within every bound and
    constraint. Infeasible and failed trials are terminal and count toward ``n_trials``.
    """
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
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, phases=[phase], metric=metric
    )
    run_experiment(experiment)

    (phase_status,) = experiment_status(experiment)["phases"]
    (read_phase,) = read_status(experiment)["phases"]

    expected = {"COMPLETE": 1, "INFEASIBLE": 1, "FAIL": 1}
    assert phase_status["trials"] == expected
    assert phase_status["generation_trials"] == expected
    assert phase_status["completed"] == 1
    assert read_phase["trials"] == expected
    assert next(iter(phase_status)) == "name"
    (table,) = read_trial_tables(experiment).phases
    assert sorted(str(row["state"]) for row in table.rows) == sorted(expected)


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


def test_in_memory_status_names_no_ledger(tmp_path: Path) -> None:
    """``storage: null`` has no ledger file, so ``ledger_path`` is null."""
    payload = experiment_status(make_experiment(workdir=tmp_path / "runs"))

    assert list(payload) == EXPERIMENT_STATUS_KEYS
    assert payload["ledger_path"] is None
