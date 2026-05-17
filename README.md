# phasesweep

[![CI](https://github.com/pszemraj/phasesweep/actions/workflows/ci.yml/badge.svg)](https://github.com/pszemraj/phasesweep/actions/workflows/ci.yml)

YAML-driven, phase-chained hyperparameter sweeps. Each phase is an Optuna study; the winner becomes a fixed override for downstream phases. Trials run as external subprocesses. Runtime orchestration is single-host and single-orchestrator-per-experiment.

Use phasesweep when a full joint sweep is too expensive and the search can be broken into inspectable stages, such as architecture depth, then learning rate, then regularization. Use one phase with the full search space when dimensions strongly interact.

## Requirements

- Python 3.10 or newer.
- Linux or macOS. Windows is not currently supported because process cleanup and host locks use POSIX process groups and `flock`.
- A trainer command that writes a metric artifact and parses one [supported override format](docs/config.md#override-formats). phasesweep does not train models itself.
- GPU is optional. When CUDA devices are visible, phasesweep can lease numeric GPU IDs to avoid same-host double-booking.

## Install

phasesweep is currently installed from Git:

```bash
python -m pip install "phasesweep @ git+https://github.com/pszemraj/phasesweep.git"
python -m pip install "phasesweep[wandb] @ git+https://github.com/pszemraj/phasesweep.git"
python -m pip install "phasesweep[dev] @ git+https://github.com/pszemraj/phasesweep.git"
```

For local development from a checkout:

```bash
python -m pip install -e "."
python -m pip install -e ".[dev]"
python -m pip install -e ".[wandb]"
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

The example launches a deterministic fake trainer, runs 32 short trials, and writes outputs under `runs/`.

## Config Sketch

Adapt this shape to your own trainer.

> [!IMPORTANT]
> The script in `trial_command` must parse the override format you choose. The default is `argparse`, where `{overrides}` renders flags such as `--n_layers 8 --lr 0.0003`; see [supported override formats](docs/config.md#override-formats).

```yaml
experiment: tiny_lm_16mb
storage: sqlite:///./runs/phases.db
workdir: ./runs
trial_command: "python train.py --out {trial_dir}/result.json {overrides}"

metric:
  name: eval_loss
  goal: minimize
  extractor: { type: json, path: result.json, key: eval_loss }

phases:
  - name: depth
    n_trials: 4
    sampler: { type: grid }
    search_space:
      n_layers: { type: categorical, choices: [4, 8, 12, 16] }

  - name: lr
    inherits: [depth]
    n_trials: 12
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
```

## Runtime Boundaries

- Bring your own trainer; see the [trainer contract](docs/config.md#trainer-contract).
- Sequential phases are greedy. They do not replace joint optimization when parameters interact strongly.
- Use one orchestrator per experiment on one host. Same-host conflicts are rejected with advisory locks.
- SQLite is for sequential `n_jobs == 1` studies. See [runtime behavior](docs/runtime.md#concurrency-model) for parallel storage and locking details.

## Docs

- [Config reference](docs/config.md): trainer contract, override formats, experiment YAML, suites, search spaces, gates, promotion, extractors.
- [Runtime behavior](docs/runtime.md): filesystem layout, locks, GPU leases, process cleanup, fingerprints, resume.
- [Development](docs/development.md): test commands and test-suite map.
- [Roadmap](docs/roadmap.md): future work outside the current single-host design.

## License

MIT. See [LICENSE](LICENSE).
