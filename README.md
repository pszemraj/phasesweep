# PhaseSweep

PhaseSweep runs phase-chained hyperparameter sweeps from one ordinary YAML file. That file contains your trainer configuration, the metric, and the search plan. PhaseSweep materializes the configured trainer input for every trial, decides what to try next, persists each phase winner, and carries selected values forward as fixed inputs to later phases.

This is useful when a full joint sweep is too expensive or hard to interpret. For example, choose architecture depth, then tune learning rate, then regularization. The [configuration guide](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#phase-composition) explains the inheritance model and its tradeoffs.

![PhaseSweep phase DAG](https://raw.githubusercontent.com/pszemraj/phasesweep/main/docs/images/diagramA_dag.png)

## How it works

The `starter.yaml` that `phasesweep init` writes searches depth before learning rate (abridged; the full file also sets storage, the working directory, and the metric extractor):

```yaml
experiment: phasesweep_starter

# The trainer receives one complete generated YAML per trial.
trial_command: "python -m phasesweep.examples.fake_train --config {config_path}"

trainer_config:
  model:
    n_layers: 8
    dropout: 0.1
  optimizer:
    lr: 0.0003
    weight_decay: 0.05

metric:
  name: eval_loss
  goal: minimize

phases:
  - name: depth
    n_trials: 2
    sampler: { type: grid }
    search_space:
      model.n_layers: { type: categorical, choices: [6, 8] }

  - name: learning_rate
    inherits: [depth] # the winning model.n_layers is fixed here
    n_trials: 2
    sampler: { type: grid }
    search_space:
      optimizer.lr: { type: categorical, choices: [0.0001, 0.0003] }
```

The packaged fake trainer reports through the optional `report_objective(...)` helper, which the `json_envelope` extractor reads. Your own trainer can report through that helper, W&B, MLflow, ordinary JSON, or logs.

## Install and try it

Requirements: Python 3.11+ and a Linux host for real runs. GPUs are optional.

```bash
pip install phasesweep
phasesweep --version

mkdir phasesweep-demo && cd phasesweep-demo
phasesweep init                     # writes the starter template, starter.yaml
phasesweep validate starter.yaml    # checks it without launching anything
phasesweep run starter.yaml         # four tiny trials, a few seconds
```

`phasesweep init` never overwrites an existing file. `phasesweep init --list` names the other templates, which cover a trainer that takes `--key=value` flags, one phase per architecture, a chained search and a [noisy objective](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#noisy-objectives-and-small-budgets); `--template NAME` writes one, and `-o PATH` names the file or a directory that receives `NAME.yaml`. Every template runs as written on the packaged toy trainer, so replace its trainer command, keys and extractor with your own.

The run log, `runs/phasesweep_starter/run.log` beside the config, shows the phase chaining directly and gains one `TRIAL` line per finished trial; `tail -f` follows it while a sweep runs (abridged):

```text
[depth/trial_0] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=depth TRIAL trial=0 state=COMPLETE outcome=success value=0.3 duration=0.8s (1/2 finished)
[depth/trial_1] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=depth TRIAL trial=1 state=COMPLETE outcome=success value=0.5 duration=0.7s (2/2 finished)
phase=depth WINNER trial=0 metric=0.3 params={'model.n_layers': 8}
[learning_rate/trial_0] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=learning_rate TRIAL trial=0 state=COMPLETE outcome=success value=0.3 duration=0.8s (1/2 finished)
[learning_rate/trial_1] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=learning_rate TRIAL trial=1 state=COMPLETE outcome=success value=0.413822 duration=0.8s (2/2 finished)
phase=learning_rate WINNER trial=0 metric=0.3 params={'optimizer.lr': 0.0003}
```

The `depth` winner's `model.n_layers: 8` is injected into every `learning_rate` trial's complete YAML. Winners persist in the working directory, so you can inspect them any time (abridged):

```bash
phasesweep show-winners starter.yaml
```

```yaml
phase: learning_rate
metric:
  eval_loss: 0.3
  goal: minimize
trial_number: 0
params:
  optimizer.lr: 0.0003
effective_overrides:
  model.n_layers: 8
  optimizer.lr: 0.0003
# ... completion state, fingerprints, objective provenance, and confound
# verdicts on the ranked population follow
```

Every trial of the same run, with its state, value, sampled params, and a `*` on each published winner, is one read-only command away:

```bash
phasesweep status starter.yaml --trials
```

```text
experiment phasesweep_starter: winner (*) marks published generation be4f520d0ec04175ac623ac6184d342e; it does not follow a rerun until that rerun publishes

=== depth === (2 trials)
trial  state     value  params            duration_s  winner
    0  COMPLETE    0.3  model.n_layers=8         0.6  *
    1  COMPLETE    0.5  model.n_layers=6         0.5

=== learning_rate === (2 trials)
trial  state        value  params               duration_s  winner
    0  COMPLETE       0.3  optimizer.lr=0.0003         0.6  *
    1  COMPLETE  0.413822  optimizer.lr=0.0001         0.6
```

## Use your own trainer

Start from the [template](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#templates) closest to your trainer, then choose the input interface it already accepts. `phasesweep explain KEY` prints the reference for one key, and a configuration error from `validate` or `run` ends with `see: phasesweep explain KEY` lines naming the keys involved.

| Input | Command placeholder |
| --- | --- |
| Complete base-plus-overrides YAML (`yaml_file`, default) | `--config {config_path}` |
| Command-line flags (`argparse`) | `{overrides}` produces `--key=value` for every `trainer_config` leaf and override |
| Native Hydra overrides (`hydra`) | `{overrides}` produces `key=value` for every `trainer_config` leaf and override |
| Nested JSON, base plus overrides (`json_file`) | `--overrides {overrides_path}` |

`argparse` renders a boolean as `--key=true` or `--key=false`, which a plain `store_true` flag rejects; `validate` and `--dry-run` warn when a configured value would render that way. Write such a flag in `trial_command` instead, or have the trainer accept a value.

> [!TIP]
> `phasesweep run starter.yaml --dry-run` prints one rendered command per phase without launching work. Use it to check that your placeholders render what the trainer expects. A phase that inherits another that still has trials to run previews with stand-in values for the parent's winner, and its line says so. Over an experiment that already has recorded trials it also reports how many trials each phase would still launch, refuses most of what a run would refuse over them, and says that a phase that has reached its `n_trials` would replay its existing result and launch nothing; the phases that inherit it preview with that result's winner. [Runtime behavior](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspection-commands) lists the refusals it does and does not make.

The trainer runs in `execution.cwd`, or in the directory you started PhaseSweep from when that is unset, with a minimal environment: `PATH`, `HOME`, `TMPDIR` and a few more, not the shell's `PYTHONPATH`, conda settings, caches or credentials. Name what it needs under `env` or `execution.inherit_env`; `validate` and `--dry-run` print the `environment:` and `trainer cwd:` lines, and a relative path in the config resolves against the config file's directory. The [trainer contract](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#trainer-contract) has the details.

For ordinary JSON output, pass an output path under `{trial_dir}` to your trainer and select it with `extractor: {type: json, path: result.json, key: eval.loss}`. For W&B, use `extractor: {type: wandb, entity: YOUR_ENTITY, project: YOUR_PROJECT, metric_key: eval/loss}`, have the trainer honor the supplied W&B identity, and install the extra:

```bash
pip install "phasesweep[wandb]"
```

For native MLflow metrics, install `phasesweep[mlflow]`, configure an existing experiment in the `mlflow` block, and use `extractor: {type: mlflow, metric_key: eval/loss, select: last}`. See the [MLflow guide](https://github.com/pszemraj/phasesweep/blob/main/docs/mlflow.md) for both run ownership modes, a complete CPU example, Databricks setup, and retrying result uploads.

Report through `phasesweep.report_objective(...)` and the `json_envelope` extractor when the checkpoint behind a value matters, as with Transformers' `load_best_model_at_end`; the same call can write report-only values with `extra=`. A top-level `observe:` list records such values per trial without ranking on them, for example a held-out test score, and `metric.min` / `metric.max` keep a collapsed run from winning. When the metric moves with the seed, spend trials on more distinct values and read the trend across neighbors; `repeats` ranks by a mean over seeds when you compare a few discrete finalists. [Noisy objectives and small budgets](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#noisy-objectives-and-small-budgets) covers these together, and [objective evidence](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#objective-evidence-constraints-and-gates) defines the extractors, bounds and `observe`.

## Connect an agent

The optional MCP server connects an AI agent to experiments you have approved without exposing config or storage paths, trainer commands, environment values, or raw logs. Sampled winner values follow the catalog's `visible_params` policy. Follow the [MCP setup](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp_setup.md) to install the extra, review the catalog authority boundary, configure a client-owned connection, and verify the result; `phasesweep mcp call` runs one tool from a shell without a client.

## Reference

- [Configuration guide](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md): Experiment YAML, search spaces, inheritance, noisy objectives and small budgets, repeats, gates, and objective extractors.
- [Configuration reference](https://github.com/pszemraj/phasesweep/blob/main/src/phasesweep/config_reference.yaml): per-key types, defaults, valid values, interactions, and lifecycle warnings. Too long to read whole; `phasesweep explain` lists the top-level keys and `phasesweep explain 'phases[].sampler'` prints one key and everything below it.
- [Minimal starting configuration](https://github.com/pszemraj/phasesweep/blob/main/examples/example.yaml): one local toy-trainer attempt; validate or run it from the checkout root.
- [Runtime behavior](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md): filesystem layout, locks, GPU leases, process supervision, fingerprints, and resume.
- [MCP setup](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp_setup.md): installed-package agent onboarding and client-owned setup.
- [MCP operator reference](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp.md): catalog fields, tools, authorization, run state, and recovery.
- [Toy experiment and MCP catalog](https://github.com/pszemraj/phasesweep/blob/main/examples/experiment.yaml): a checkout-local CLI example backed by the packaged fake trainer, plus an [MCP catalog](https://github.com/pszemraj/phasesweep/blob/main/examples/catalog.yaml) whose detached-run state and experiment outputs use absolute scratch paths under `/tmp`.
- [Tiny Decoder Enwik8 example](https://github.com/pszemraj/phasesweep/blob/main/examples/tiny_decoder_enwik8/README.md): real-trainer integration.
- [Durability invariants](https://github.com/pszemraj/phasesweep/blob/main/docs/invariants.md): the ordering rules the ledger, publication, and recovery paths must hold, and the test that owns each one.
- [Development](https://github.com/pszemraj/phasesweep/blob/main/docs/development.md): source checkout, contributor setup, and quality gates.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pszemraj/phasesweep)

## License

MIT. See [LICENSE](https://github.com/pszemraj/phasesweep/blob/main/LICENSE).
