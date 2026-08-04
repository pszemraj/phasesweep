# PhaseSweep

PhaseSweep runs YAML-defined, phase-chained hyperparameter sweeps over your own training script. Your trainer owns the experiment; PhaseSweep decides what to try next, persists each phase winner, and can pass selected winners forward as fixed inputs to later phases.

This is useful when a full joint sweep is too expensive or hard to interpret—for example, choose architecture depth, then tune learning rate, then regularization. The [configuration guide](docs/config.md#phase-keys) explains the inheritance model and its tradeoffs.

![PhaseSweep phase DAG](docs/images/diagramA_dag.png)

## Install and try it

Requirements: Python 3.11+, a POSIX host for real runs, and a trainer that follows the [trainer contract](docs/config.md#trainer-contract). GPUs are optional.

```bash
pip install git+https://github.com/pszemraj/phasesweep.git
```

Create and inspect a runnable two-phase starter without cloning this repository:

```bash
mkdir phasesweep-demo
cd phasesweep-demo
phasesweep init
phasesweep validate experiment.yaml
phasesweep run experiment.yaml --dry-run
```

`phasesweep init` never overwrites an existing file. Its fake trainer ships inside the installed package, so validation and the dry run work from any directory. When you are ready, replace the trainer command and search spaces in `experiment.yaml`; run it only after reviewing the rendered commands.

## Connect an agent

The optional MCP server lets an AI agent work with experiments you have already approved. An agent can discover and inspect catalog entries, launch when permitted and explicitly authorized, wait on a durable run ID, read terminal phase winners, and cancel only when permitted.

Follow the [five-minute MCP setup](docs/mcp_setup.md) to install the optional extra, scaffold and review a catalog, and connect a supported client. The catalog review remains a separate step on purpose.

Three safety properties do not change:

- The human controls the catalog and experiment YAML.
- The agent operates only by approved experiment ID.
- Trainer commands, paths, storage, environment, and raw logs are not agent inputs.

After setup, restart the selected client and ask:

```text
List the available PhaseSweep experiments and their permitted actions.
Do not launch anything.
```

## CLI examples

```bash
# Create a starter at another path
phasesweep init -o configs/experiment.yaml

# Inspect before running
phasesweep validate configs/experiment.yaml
phasesweep run configs/experiment.yaml --dry-run

# Run and inspect durable results
phasesweep run configs/experiment.yaml
phasesweep status configs/experiment.yaml
phasesweep show-winners configs/experiment.yaml

# Resume at a later phase after its prerequisites have valid winners
phasesweep run configs/experiment.yaml --from-phase learning_rate
```

`validate`, `run --dry-run`, `status`, and `show-winners` do not launch training trials. See [runtime behavior](docs/runtime.md) for locks, process cleanup, GPU isolation, fingerprints, resume, and output layout.

## Reference

- [Configuration guide](docs/config.md): trainer contract, experiment and suite YAML, search spaces, inheritance, gates, promotion, and extractors.
- [Configuration reference](docs/config_reference.yaml): per-key types, defaults, valid values, interactions, and lifecycle warnings.
- [Runtime behavior](docs/runtime.md): filesystem layout, locks, GPU leases, process supervision, fingerprints, and resume.
- [MCP setup](docs/mcp_setup.md): installed-package agent onboarding.
- [MCP operator reference](docs/mcp.md): catalog fields, tools, authorization, run state, recovery, and installer preservation semantics.
- [Tiny Decoder Enwik8 example](examples/tiny_decoder_enwik8/README.md): real-trainer integration.
- [Development](docs/development.md): source checkout, contributor setup, and quality gates.

## License

MIT. See [LICENSE](LICENSE).
