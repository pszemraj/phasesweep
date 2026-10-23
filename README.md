# PhaseSweep

PhaseSweep runs phase-chained hyperparameter sweeps from one ordinary YAML file. That file contains your trainer configuration, the metric, and the search plan. PhaseSweep materializes the configured trainer input for every trial, decides what to try next, persists each phase winner, and carries selected values forward as fixed inputs to later phases.

This is useful when a full joint sweep is too expensive or hard to interpret. For example, choose architecture depth, then tune learning rate, then regularization. The [configuration guide](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#phase-composition) explains the inheritance model and its tradeoffs.

![PhaseSweep phase DAG](https://raw.githubusercontent.com/pszemraj/phasesweep/main/docs/images/diagramA_dag.png)

## How it works

The `starter.yaml` that `phasesweep init` writes searches depth before learning rate (abridged; the generated file adds storage, provenance, execution settings, the metric extractor, and comments):

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

The packaged fake trainer uses [`report_objective(...)`](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#reporting-an-objective). Your own trainer can report through that helper, `phasesweep report-objective` from a shell, W&B, MLflow, ordinary JSON, or logs.

## Install and try it

Requirements: Python 3.11+ and a Linux host for real runs. GPUs are optional.

PhaseSweep runs on one host. It drives Optuna's grid, random, TPE and CMA-ES samplers, keeps an [Optuna journal](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspect-the-ledger-with-optuna) as its ledger, and can run trials in parallel on that host's GPUs. It does not spread trials across machines or prune them early.

```bash
pip install phasesweep
phasesweep --version

mkdir phasesweep-demo && cd phasesweep-demo
phasesweep init                     # writes the starter template, starter.yaml
phasesweep validate starter.yaml    # checks it without launching anything
phasesweep run starter.yaml         # runs four tiny toy-trainer trials
```

`phasesweep init` never overwrites an existing file; `phasesweep init --list` shows the other [templates](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#templates).

The run log, `runs/phasesweep_starter/run.log` beside the config, shows the phase chaining (abridged to the trial commands, results and winners, without timestamps):

```text
[depth/trial_0] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=depth TRIAL trial=0 state=COMPLETE value=0.3 duration=0.8s (1/2 finished)
[depth/trial_1] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=depth TRIAL trial=1 state=COMPLETE value=0.5 duration=0.7s (2/2 finished)
phase=depth WINNER trial=0 metric=0.3 params={'model.n_layers': 8}
[learning_rate/trial_0] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=learning_rate TRIAL trial=0 state=COMPLETE value=0.3 duration=0.8s (1/2 finished)
[learning_rate/trial_1] python -m phasesweep.examples.fake_train --config .../trainer_config.yaml
phase=learning_rate TRIAL trial=1 state=COMPLETE value=0.413822 duration=0.8s (2/2 finished)
phase=learning_rate WINNER trial=0 metric=0.3 params={'optimizer.lr': 0.0003}
```

The `depth` winner's `model.n_layers: 8` is injected into every `learning_rate` trial's complete YAML. Winners persist in the working directory, so you can inspect them any time (abridged):

```bash
phasesweep show-winners starter.yaml
```

```text
# code revision: unknown (not a git work tree or git unavailable)
=== depth ===
# Compare two model depths before tuning the learning rate.
phase: depth
...
=== learning_rate ===
# Keep the winning depth fixed and compare two learning rates.
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
...
```

Inspect recorded trials without launching work:

```bash
phasesweep status starter.yaml --trials
```

With persistent storage, rerun the same config after an interruption; interrupted trials rerun automatically. See [continuation and reruns](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#timeouts-cleanup-and-continuation) for recovering an aborted phase, `--from-phase NAME` and trial budgets, and [inspection commands](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspection-commands) for status and result views.

## Use your own trainer

Start from the [template](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#templates) closest to your trainer. Choose its existing [input interface](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#trainer-inputs): complete YAML, command-line flags, Hydra overrides, or nested JSON. `phasesweep explain KEY` prints the reference for one key; configuration errors name the relevant keys.

`phasesweep run starter.yaml --dry-run` previews commands without launching work. Check it before a real run; [inspection commands](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspection-commands) explains inherited placeholders and preview limitations.

Set `execution.cwd`, `env`, and `execution.inherit_env` for the [trainer's working directory and environment](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#trainer-contract). `validate` and `--dry-run` print the launch contract so you can check it before training.

Choose an [objective extractor](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#objective-evidence-constraints-and-gates) for the trainer's output. For W&B, install the extra and follow the [W&B setup](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#wb-evidence):

```bash
pip install "phasesweep[wandb]"
```

For MLflow, install `phasesweep[mlflow]` and follow the [MLflow guide](https://github.com/pszemraj/phasesweep/blob/main/docs/mlflow.md). Use [objective reporting](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#reporting-an-objective) when the checkpoint behind a value matters, and [small-budget advice and repeats](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#noisy-objectives-and-small-budgets) for noisy measurements.

## Connect an agent

The optional MCP server connects an AI agent to experiments you have approved without exposing config or storage paths, trainer commands, environment values, or raw logs. Sampled winner values follow the catalog's `visible_params` policy. Follow the [MCP setup](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp_setup.md) to install the extra, review the catalog authority boundary, configure a client-owned connection, and verify the result; `phasesweep mcp call` runs one tool from a shell without a client.

## Reference

- [Configuration guide](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md): Experiment YAML, search spaces, inheritance, noisy objectives and small budgets, repeats, gates, and objective extractors.
- [Configuration reference](https://github.com/pszemraj/phasesweep/blob/main/src/phasesweep/config_reference.yaml): per-key types, defaults, valid values, interactions, and warnings. `phasesweep explain` lists the top-level keys, and `phasesweep explain 'phases[].sampler'` prints one key, everything below it, and the section rules that govern it.
- [Minimal starting configuration](https://github.com/pszemraj/phasesweep/blob/main/examples/example.yaml): one local toy-trainer attempt; validate or run it from the checkout root.
- [Runtime behavior](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md): filesystem layout, locks, GPU leases, process supervision, fingerprints, and resume.
- [MCP setup](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp_setup.md): installed-package agent onboarding and client-owned setup.
- [MCP operator reference](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp.md): catalog fields, tools, authorization, run state, and recovery.
- [Toy experiment and MCP catalog](https://github.com/pszemraj/phasesweep/blob/main/examples/experiment.yaml): a checkout-local CLI example on the packaged fake trainer, plus an [MCP catalog](https://github.com/pszemraj/phasesweep/blob/main/examples/catalog.yaml) that keeps its state and outputs under `/tmp`.
- [Tiny Decoder Enwik8 example](https://github.com/pszemraj/phasesweep/blob/main/examples/tiny_decoder_enwik8/README.md): real-trainer integration.
- [Durability invariants](https://github.com/pszemraj/phasesweep/blob/main/docs/invariants.md): the ordering rules the ledger, publication, and recovery paths must hold, and the test that owns each one.
- [Development](https://github.com/pszemraj/phasesweep/blob/main/docs/development.md): source checkout, contributor setup, quality gates, and declined requests.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pszemraj/phasesweep)

## License

MIT. See [LICENSE](https://github.com/pszemraj/phasesweep/blob/main/LICENSE).
