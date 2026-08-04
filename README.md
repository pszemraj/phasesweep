# PhaseSweep

PhaseSweep runs YAML-defined, phase-chained hyperparameter sweeps over your own training script. Your trainer owns the experiment; PhaseSweep decides what to try next, persists each phase winner, and can pass selected winners forward as fixed inputs to later phases.

This is useful when a full joint sweep is too expensive or hard to interpret. For example, choose architecture depth, then tune learning rate, then regularization. The [configuration guide](docs/config.md#phase-keys) explains the inheritance model and its tradeoffs.

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

`phasesweep init` never overwrites an existing file; pass `-o PATH` to choose another destination. Its fake trainer ships inside the installed package, so validation and the dry run work from any directory. When you are ready, replace the trainer command and search spaces in `experiment.yaml`; run it only after reviewing the rendered commands.

## Connect an agent

The optional MCP server connects an AI agent to experiments you have approved without exposing trainer commands, paths, storage, environment, or raw logs. Follow the [MCP setup](docs/mcp_setup.md) to install the extra, review the catalog authority boundary, connect a supported client, and verify the result. See the [MCP operator reference](docs/mcp.md) for the tool surface and security model.

## CLI examples

```bash
# Run and inspect durable results
phasesweep run experiment.yaml
phasesweep status experiment.yaml
phasesweep show-winners experiment.yaml

# Resume at a later phase after its prerequisites have valid winners
phasesweep run experiment.yaml --from-phase learning_rate
```

`validate`, `run --dry-run`, `status`, and `show-winners` do not launch training trials. See [runtime behavior](docs/runtime.md) for locks, process cleanup, GPU isolation, fingerprints, resume, and output layout.

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
