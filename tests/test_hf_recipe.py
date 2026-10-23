"""The documented Hugging Face recipe composes inputs and reads trial-local results."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from phasesweep.config.io import load_experiment_bytes
from phasesweep.engine.trial import _observe
from phasesweep.evidence.evaluation import run_extractor
from phasesweep.runtime.commands import render_command
from tests.conftest import REPO, make_trial_context


def test_hf_argparse_recipe_reads_results_below_the_rendered_output_dir(tmp_path: Path) -> None:
    guide = (REPO / "docs/config.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)\n```", guide, re.DOTALL)
    recipe = next((block for block in blocks if "run_glue.py" in block), None)
    assert recipe is not None, "The guide needs a complete Hugging Face argparse recipe"
    experiment = load_experiment_bytes(
        recipe.encode(), source="documented HF recipe", config_dir=tmp_path
    )
    phase = experiment.phases[0]
    assert phase.repeats is not None
    assert phase.repeats.key == "seed"
    assert phase.n_trials == 4
    assert len(phase.repeats.values) == 2

    trial_dir = tmp_path / "trial"
    command = render_command(
        experiment.trial_command,
        {"learning_rate": 0.00002, "seed": phase.repeats.values[0]},
        experiment.override_format,
        trial_dir=trial_dir,
        trial_id=0,
        phase=phase.name,
        run_name="hf-recipe-test",
        trainer_config=experiment.trainer_config,
    )
    arguments = shlex.split(command)
    assert "--do_train" in arguments
    assert "--do_eval" in arguments
    output_dir = trial_dir / "hf"
    assert f"--output_dir={output_dir}" in arguments

    output_dir.mkdir(parents=True)
    (output_dir / "eval_results.json").write_text(
        json.dumps({"eval_loss": 0.25, "eval_accuracy": 0.875})
    )
    (output_dir / "train_results.json").write_text(json.dumps({"train_runtime": 123.0}))
    context = make_trial_context(trial_dir)

    assert run_extractor(context, experiment.metric.extractor) == 0.25
    assert _observe(context, experiment.observe, deadline=None) == {
        "accuracy": 0.875,
        "train_runtime": 123.0,
    }
