# PhaseSweep

PhaseSweep runs hyperparameter sweeps in ordered phases: choose model depth, then tune learning rate with the winning depth fixed, then regularization. One YAML file holds your trainer configuration, objective, and search plan. Each trial receives a complete trainer input with inherited, fixed, and sampled values applied.

Requires Python 3.11+ and Linux or macOS. GPUs are optional; the starter uses a packaged toy trainer. The project is unstable, so review the launch contract before using your own trainer.

## Install and try it

Install from [PyPI](https://pypi.org/project/phasesweep/):

```bash
pip install phasesweep
phasesweep --version

mkdir phasesweep-demo && cd phasesweep-demo
phasesweep init                     # writes starter.yaml without overwriting files
phasesweep validate starter.yaml    # checks the config without launching work
phasesweep run starter.yaml         # runs four toy-trainer trials
phasesweep show-winners starter.yaml
phasesweep status starter.yaml --trials
```

The starter compares two depths, then two learning rates. Its `run.log` includes these winner lines (abridged; trial numbers depend on grid order):

```text
phase=depth WINNER trial=0 metric=0.3 params={'model.n_layers': 8}
phase=learning_rate WINNER trial=0 metric=0.3 params={'optimizer.lr': 0.0003}
```

The depth winner's `model.n_layers: 8` becomes a fixed input to every learning-rate trial. Inheritance transfers parameter values; the trainer owns initialization and checkpoint loading. The complete [starter YAML](https://github.com/pszemraj/phasesweep/blob/main/src/phasesweep/templates/starter.yaml) and [phase composition guide](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#phase-composition) explain the configuration and its tradeoffs.

![PhaseSweep phase DAG](https://raw.githubusercontent.com/pszemraj/phasesweep/main/docs/images/diagramA_dag.png)

Results live under `runs/phasesweep_starter/` beside the config, including `run.log`. `show-winners` prints the validated generation directory and winners; `status --trials` inspects recorded attempts. See [inspection commands](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspection-commands) for the output.

## Use your own trainer

`phasesweep init --list` lists the [templates](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#templates). Replace the command, configuration keys, and objective extractor in the closest template:

| Configure | Choices and details |
| --- | --- |
| Trainer input | Complete YAML, command-line flags, Hydra overrides, or nested JSON: [input interfaces](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#trainer-inputs). |
| Objective | Python `report_objective(...)`, the `phasesweep report-objective` command, JSON, logs, W&B, or MLflow: [reporting](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#reporting-an-objective) and [extractors](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#objective-evidence-constraints-and-gates). |
| Environment | `execution.cwd`, `env`, and `execution.inherit_env`: [trainer contract](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#trainer-contract). |

Preview commands and the launch contract before training. Configuration errors name the relevant keys; `explain` gives their types, defaults, interactions, and warnings:

```bash
phasesweep run starter.yaml --dry-run
phasesweep explain
phasesweep explain 'phases[].sampler'
```

The [inspection guide](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspection-commands) covers inherited placeholders and preview limits.

W&B and MLflow require their optional extras, `phasesweep[wandb]` and `phasesweep[mlflow]`; follow the [W&B setup](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#wb-evidence) or [MLflow guide](https://github.com/pszemraj/phasesweep/blob/main/docs/mlflow.md). For noisy measurements, see [small-budget advice and repeats](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md#noisy-objectives-and-small-budgets).

## Scope and continuation

PhaseSweep drives Optuna's grid, random, TPE, and CMA-ES samplers on one host, with an [Optuna journal](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#inspect-the-ledger-with-optuna) for persistent trials. GPU trials can run in parallel on that host; distributed sweeps and early pruning are outside its scope.

Apple silicon Macs discover one MPS device for same-host leasing, and the trainer selects its compute device. See [GPU isolation](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#gpu-isolation).

With persistent storage, rerun the same config to resume interrupted work. [Continuation and reruns](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md#timeouts-cleanup-and-continuation) covers aborted phases, trial budgets, and `--from-phase NAME`.

## Connect an agent

The optional MCP server exposes experiments approved in an operator-owned catalog. Tool responses omit config and storage paths, trainer commands, environment values, and raw logs; sampled winner values follow `visible_params`. [MCP setup](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp_setup.md) covers installation, catalog permissions, client configuration, and verification. `phasesweep mcp call` invokes one tool from a shell without a client.

## Reference

- [Configuration guide](https://github.com/pszemraj/phasesweep/blob/main/docs/config.md) and [per-key reference](https://github.com/pszemraj/phasesweep/blob/main/src/phasesweep/config_reference.yaml).
- [Minimal configuration](https://github.com/pszemraj/phasesweep/blob/main/examples/example.yaml): one toy-trainer attempt from the checkout root.
- [Runtime](https://github.com/pszemraj/phasesweep/blob/main/docs/runtime.md): outputs, locks, leases, supervision, fingerprints, and resume.
- [MCP operator reference](https://github.com/pszemraj/phasesweep/blob/main/docs/mcp.md): catalog fields, tools, authorization, run state, and recovery.
- [Toy experiment](https://github.com/pszemraj/phasesweep/blob/main/examples/experiment.yaml) and [MCP catalog](https://github.com/pszemraj/phasesweep/blob/main/examples/catalog.yaml): checkout-local examples with MCP state and outputs under `/tmp`.
- [Tiny Decoder Enwik8](https://github.com/pszemraj/phasesweep/blob/main/examples/tiny_decoder_enwik8/README.md): real-trainer integration.
- [Durability invariants](https://github.com/pszemraj/phasesweep/blob/main/docs/invariants.md): ledger, publication, and recovery rules, with their tests.
- [Development](https://github.com/pszemraj/phasesweep/blob/main/docs/development.md): contributor setup, quality gates, and declined requests.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pszemraj/phasesweep)

## License

MIT. See [LICENSE](https://github.com/pszemraj/phasesweep/blob/main/LICENSE).
