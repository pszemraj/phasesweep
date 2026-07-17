# phasesweep

> Orchestration layer for YAML-driven, phase-chained hyperparameter sweeps over your own training scripts

Your trainer runs the experiments. `phasesweep` decides what to try next. Define phased Optuna sweeps in YAML; each phase's winner locks in as a fixed override for every phase downstream.

Use `phasesweep` when a full joint sweep is too expensive and the search can be broken into inspectable stages, such as architecture depth, then learning rate, then regularization. Use one phase with the full search space when dimensions strongly interact.

![dag diagram](docs/images/diagramA_dag.png)

## Requirements

- Python 3.10+, OS: Linux or macOS[^1]
- A trainer command that **writes a metric artifact** and **accepts at least one [supported override format](docs/config.md#override-formats)**[^2]
- GPU optional: CUDA devices are auto-detected for same-host lease management

[^1]: Windows is unsupported; process cleanup and host locks rely on POSIX process groups and `flock`. Use WSL for Windows-hosted development.
[^2]: phasesweep orchestrates sweeps but never trains anything itself; your trainer must handle both of these.

## Install

phasesweep is currently installed from Git:

```bash
python -m pip install "phasesweep @ git+https://github.com/pszemraj/phasesweep.git"
# weights-and-biases integration is optional:
python -m pip install "phasesweep[wandb] @ git+https://github.com/pszemraj/phasesweep.git"
# MCP server, to drive sweeps from an AI agent:
python -m pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
# all dev dependencies:
python -m pip install "phasesweep[dev,wandb] @ git+https://github.com/pszemraj/phasesweep.git"
```

For local development from a checkout:

```bash
git clone https://github.com/pszemraj/phasesweep.git
cd phasesweep
# activate venv of your choice, then:
python -m pip install -e ".[dev,wandb]"
```

## Quickstart

To run the bundled toy example from a checkout:

```bash
phasesweep validate examples/experiment.yaml
phasesweep run examples/experiment.yaml --dry-run
phasesweep run examples/experiment.yaml
phasesweep show-winners examples/experiment.yaml
phasesweep status examples/experiment.yaml
```

The bundled example launches a deterministic fake trainer, runs 32 short trials, and writes outputs under `runs/`. For a real-trainer integration, see [examples/tiny_decoder_enwik8](examples/tiny_decoder_enwik8/README.md).

## Config

Start from [examples/experiment.yaml](examples/experiment.yaml) or the [config guide](docs/config.md). Your trainer must parse the selected [override format](docs/config.md#override-formats), write the configured metric artifact, and exit nonzero on failed trials.

Sequential phases are greedy. They do not replace joint optimization when parameters interact strongly. Runtime locks and storage behavior are covered in [runtime behavior](docs/runtime.md).

## MCP server (agent integration)

`phasesweep-mcp` lets an AI agent operate experiments you have already reviewed. You keep control of the config, trainer command, search space, and permissions in a catalog; the agent sees stable experiment ids and can validate, launch, monitor, cancel when allowed, and summarize winners without receiving paths, commands, raw logs, storage URLs, or workdirs.

Follow [MCP agent setup](docs/mcp_setup.md) to install the server, create and preflight a catalog, connect a supported client, and give the agent its operating instructions. See [MCP server](docs/mcp.md) for catalog fields, tool behavior, run state, and security boundaries.

## Docs

- [Config guide](docs/config.md): trainer contract, override formats, experiment YAML, suites, search spaces, gates, promotion, extractors.
- [Typed config reference](docs/config_reference.yaml): schema-complete YAML reference with every field, type, default, enum, and major validation constraint.
- [Runtime behavior](docs/runtime.md): filesystem layout, locks, GPU leases, process cleanup, fingerprints, resume.
- [MCP server](docs/mcp.md): expose an experiment to an AI agent - catalog format, the seven tools, security model, single-host operation.
- [MCP agent setup](docs/mcp_setup.md): five steps from install to a working agent - install the `[mcp]` extra, write and preflight a catalog, connect your clients with `phasesweep install`, verify, instruct the agent.
- [Tiny Decoder Enwik8 example](examples/tiny_decoder_enwik8/README.md): real-trainer `json_file` integration with a pinned submodule.
- [Development](docs/development.md): test commands and test-suite map.

## License

MIT. See [LICENSE](LICENSE).
