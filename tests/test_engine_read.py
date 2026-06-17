"""engine.read: permissive status/winner reads that never raise on a partial file."""

from __future__ import annotations

from pathlib import Path

import yaml

from phasesweep.config import Experiment, load_config
from phasesweep.engine import read_winner, read_winners
from phasesweep.engine.state import _winner_path


def _experiment(tmp_path: Path) -> Experiment:
    config = tmp_path / "exp.yaml"
    config.write_text(
        """\
experiment: read_t
workdir: {wd}
trial_command: "python x.py {{overrides}}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json, path: r.json, key: loss }}
phases:
  - name: p
    n_trials: 1
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
""".format(wd=tmp_path / "wd")
    )
    parsed = load_config(config)
    assert isinstance(parsed, Experiment)
    return parsed


def test_read_winner_parses_a_valid_file(tmp_path: Path) -> None:
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "phase": "p",
                "trial_number": 3,
                "metric": {"loss": 0.123, "goal": "minimize"},
                "params": {"lr": 0.001},
                "effective_overrides": {"lr": 0.001},
                "gates": [{"name": "g", "passed": True}],
                "completion": {"incomplete": False},
            }
        )
    )
    view = read_winner(exp, "p")
    assert view is not None
    assert view.trial_number == 3
    assert view.metric == 0.123
    assert view.gates_passed is True
    assert view.incomplete is False


def test_read_winner_tolerates_torn_or_malformed_file(tmp_path: Path) -> None:
    # winner.yaml is written non-atomically at phase completion; a status query
    # may observe it torn. Both a truncated file (invalid YAML) and a valid-YAML
    # file missing required keys must read as "no winner yet", never raise.
    exp = _experiment(tmp_path)
    path = _winner_path(exp, "p")
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text('{"trial_number": 0, "metric": {"loss":')  # truncated -> YAMLError
    assert read_winner(exp, "p") is None
    assert read_winners(exp) == []

    path.write_text("phase: p\n")  # valid YAML, missing trial_number/metric -> KeyError
    assert read_winner(exp, "p") is None
    assert read_winners(exp) == []
