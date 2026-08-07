# PhaseSweep

PhaseSweep runs YAML-defined, phase-chained hyperparameter sweeps over your own training script. Your trainer owns the experiment; PhaseSweep decides what to try next, persists each phase winner, and can pass selected winners forward as fixed inputs to later phases.

This is useful when a full joint sweep is too expensive or hard to interpret. For example, choose architecture depth, then tune learning rate, then regularization. The [configuration guide](docs/config.md#phase-keys) explains the inheritance model and its tradeoffs.

![PhaseSweep phase DAG](docs/images/diagramA_dag.png)

## How it works

One YAML file defines an experiment: a trainer command, a metric to optimize, and an ordered list of phases. Each phase sweeps its own small search space, and a phase that `inherits` an earlier one receives that phase's winning parameters as fixed inputs. Abridged from the starter that `phasesweep init` generates (the full file also pins storage, the working directory, and the metric extractor):

```yaml
experiment: phasesweep_starter

# The trainer is any command. This fake one ships inside the installed
# package so the starter runs anywhere; replace it with your own later.
trial_command: "python -m phasesweep.examples.fake_train --out {trial_dir}/result.json {overrides}"
override_format: argparse

metric:
  name: eval_loss
  goal: minimize

phases:
  - name: depth
    n_trials: 2
    sampler: { type: grid }
    search_space:
      n_layers: { type: categorical, choices: [6, 8] }

  - name: learning_rate
    inherits: [depth] # the winning n_layers becomes a fixed input here
    n_trials: 2
    sampler: { type: grid }
    search_space:
      lr: { type: categorical, choices: [0.0001, 0.0003] }
```

In this starter, PhaseSweep renders `{overrides}` from the sampled and inherited parameters, launches the packaged fake trainer, and reads its objective from `result.json`. Other trainers can supply objective evidence through JSON envelopes, log extraction, or W&B; see the [trainer contract](docs/config.md#trainer-contract).

## Install and try it

Requirements: Python 3.11+ and a POSIX host for real runs. GPUs are optional.

```bash
pip install git+https://github.com/pszemraj/phasesweep.git

mkdir phasesweep-demo && cd phasesweep-demo
phasesweep init                       # writes the starter experiment.yaml
phasesweep validate experiment.yaml   # checks it without launching anything
phasesweep run experiment.yaml        # four tiny trials, a few seconds
```

The run log shows the phase chaining directly (abridged):

```text
[depth/trial_0] python -m phasesweep.examples.fake_train ... --n_layers 8
[depth/trial_1] python -m phasesweep.examples.fake_train ... --n_layers 6
phase=depth WINNER trial=0 metric=0.3 params={'n_layers': 8}
[learning_rate/trial_0] python -m phasesweep.examples.fake_train ... --n_layers 8 --lr 0.0003
[learning_rate/trial_1] python -m phasesweep.examples.fake_train ... --n_layers 8 --lr 0.0001
phase=learning_rate WINNER trial=0 metric=0.3 params={'lr': 0.0003}
```

The `depth` winner's `--n_layers 8` is injected into every `learning_rate` trial as a fixed flag. Winners persist in the working directory, so you can inspect them any time (abridged):

```bash
phasesweep show-winners experiment.yaml
```

```yaml
phase: learning_rate
metric:
  eval_loss: 0.3
  goal: minimize
trial_number: 0
params:
  lr: 0.0003
effective_overrides:
  n_layers: 8
  lr: 0.0003
# ... completion state, fingerprints, and objective provenance follow
```

## Use your own trainer

Point `trial_command` at your training script and adjust the search spaces. `{trial_dir}` is the per-trial output directory. Argparse and Hydra templates use `{overrides}` for the rendered parameters; `json_file` templates use `{overrides_path}`. The trainer must accept that boundary, exit correctly, and provide finite evidence through the configured extractor, as defined by the [trainer contract](docs/config.md#trainer-contract).

Review before launching real workloads: `phasesweep run experiment.yaml --dry-run` prints every rendered command without starting training, and `validate`, `status`, and `show-winners` never launch trials either. `phasesweep init` never overwrites an existing file; pass `-o PATH` to choose another destination.

```bash
phasesweep status experiment.yaml                           # durable progress, nothing launched
phasesweep run experiment.yaml --from-phase learning_rate   # resume after prerequisites have valid winners
phasesweep rebind-workdir experiment.yaml                   # after you moved the artifact tree yourself
```

See [runtime behavior](docs/runtime.md) for locks, process cleanup, GPU isolation, fingerprints, resume, and output layout. The [Tiny Decoder Enwik8 example](examples/tiny_decoder_enwik8/README.md) is a complete real-trainer integration.

## Connect an agent

The optional MCP server connects an AI agent to experiments you have approved without exposing config or storage paths, trainer commands, environment values, or raw logs. Sampled winner values follow the catalog's `visible_params` policy. Follow the [MCP setup](docs/mcp_setup.md) to install the extra, review the catalog authority boundary, connect a supported client, and verify the result.

## Reference

- [Configuration guide](docs/config.md): trainer contract, experiment and suite YAML, search spaces, inheritance, gates, promotion, and extractors.
- [Configuration reference](docs/config_reference.yaml): per-key types, defaults, valid values, interactions, and lifecycle warnings.
- [Runtime behavior](docs/runtime.md): filesystem layout, locks, GPU leases, process supervision, fingerprints, and resume.
- [MCP setup](docs/mcp_setup.md): installed-package agent onboarding.
- [MCP operator reference](docs/mcp.md): catalog fields, tools, authorization, run state, recovery, and installer preservation semantics.
- [Toy experiment and MCP catalog](examples/experiment.yaml): checkout-local examples backed by the packaged fake trainer; the catalog is in [examples/catalog.yaml](examples/catalog.yaml).
- [Tiny Decoder Enwik8 example](examples/tiny_decoder_enwik8/README.md): real-trainer integration.
- [Development](docs/development.md): source checkout, contributor setup, and quality gates.

## License

MIT. See [LICENSE](LICENSE).
