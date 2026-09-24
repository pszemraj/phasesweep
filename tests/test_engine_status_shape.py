"""Public shape contract for ``config_status`` / ``experiment_status``.

``config_status`` is a documented package-root API (docs/development.md) and
``phasesweep status`` renders its payload verbatim, so its key set is a
contract for downstream consumers. These tests pin the experiment payload so
that changes to the public CLI shape do not drift unnoticed.
"""

from __future__ import annotations

from pathlib import Path

from phasesweep import config_status
from phasesweep.config import Experiment
from phasesweep.engine import read_status
from phasesweep.engine.paths import _generation_winner_path
from phasesweep.engine.publication import _last_successful_generation_id
from phasesweep.engine.run import experiment_status
from tests.ledger_fixtures import materialize

EXPERIMENT_STATUS_KEYS = [
    "kind",
    "experiment",
    "workdir",
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
payload carries no empty error field (review v0.5.18 / finding F4).
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
# instead (review v0.5.16 / blocker 4).
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
never has to reread a study to reconcile RUNNING trials (PR #5 review /
reviewer 2, blocker 6); the CLI phase payload keeps exactly
``PHASE_STATUS_KEYS`` and republishes none of it.
"""


def _published(tmp_path: Path) -> Experiment:
    """Return the config that reads the golden fixture's two-trial publication."""
    return materialize("current-journal", tmp_path, mode="tree").experiment


def test_experiment_config_status_shape_is_pinned(tmp_path: Path) -> None:
    """A standalone experiment payload carries generation identity plus phases."""
    experiment = _published(tmp_path)

    payload = config_status(experiment)

    assert list(payload) == EXPERIMENT_STATUS_KEYS
    assert not READ_STATUS_ONLY_KEYS & set(payload)
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
    ``running_attempts`` arriving from the same tolerant read as the counts
    (PR #5 review / reviewer 2, blocker 6), and ``experiment_status`` is a
    pinned public contract that must not grow a key because of it.
    """
    experiment = _published(tmp_path)

    (read_phase,) = read_status(experiment)["phases"]
    (cli_phase,) = experiment_status(experiment)["phases"]

    assert set(read_phase) >= READ_STATUS_ONLY_PHASE_KEYS
    assert read_phase["trial_data_available"] is True
    assert read_phase["running_attempts"] == []
    assert set(cli_phase) == PHASE_STATUS_KEYS


def test_status_shape_reports_a_corrupt_publication_without_fabricating_results(
    tmp_path: Path,
) -> None:
    """A publication that no longer validates adds two keys and invents nothing.

    Review v0.5.18 / finding F4: the payload for a corrupt publication used to
    be byte-identical to a tree that had never published, so the operator's
    natural next move was to re-run over the evidence.
    """
    experiment = _published(tmp_path)
    generation_id = _last_successful_generation_id(experiment)
    assert generation_id is not None

    winner_path = _generation_winner_path(experiment, generation_id, "p")
    winner_path.write_text(winner_path.read_text() + "\n# edited after publication\n")

    payload = config_status(experiment)

    assert list(payload) == FAILED_PUBLICATION_STATUS_KEYS
    assert not READ_STATUS_ONLY_KEYS & set(payload)
    assert payload["publication_integrity"] == "failed"
    assert "does not match its recorded hash" in payload["publication_error"]
    assert payload["published_generation_id"] is None
    assert payload["represented_generation_id"] is None
    assert payload["is_published"] is False
    assert payload["phases"][0]["winner"] is None
