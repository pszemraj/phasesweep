# phasesweep

YAML-driven, phase-chained hyperparameter sweeps. Each phase is an Optuna study; the winner becomes a fixed override for downstream phases. Trials run as external subprocesses. Runtime orchestration is single-host and single-orchestrator-per-experiment.

Use phasesweep when a full joint sweep is too expensive and the search can be broken into inspectable stages, such as architecture depth, then learning rate, then regularization. Use one phase with the full search space when dimensions strongly interact.

## Install

```bash
pip install -e .        # core runtime, including Optuna CMA-ES support
pip install -e .[dev]   # pytest, ruff, mypy, types-PyYAML
pip install -e .[wandb] # optional W&B extractor
```

## Minimal Config

```yaml
experiment: tiny_lm_16mb
storage: sqlite:///./runs/phases.db
workdir: ./runs
trial_command: "python train.py --out {trial_dir}/result.json {overrides}"

override_format: hydra
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

Run it:

```bash
phasesweep validate examples/experiment.yaml
phasesweep run examples/experiment.yaml
phasesweep show-winners examples/experiment.yaml
phasesweep status examples/experiment.yaml
```

## Docs

- [Config reference](docs/config.md): experiment YAML, suites, search spaces, gates, promotion, extractors.
- [Runtime behavior](docs/runtime.md): filesystem layout, locks, GPU leases, process cleanup, fingerprints, resume.
- [Development](docs/development.md): test commands and test-suite map.
- [TODO](TODO.md): future work that is intentionally outside the current single-host design.

## Runtime Boundaries

- Bring your own trainer. phasesweep only renders overrides, launches subprocesses, and extracts results.
- No LLM runs inside the sweep loop.
- Sequential phases are greedy. They do not replace joint optimization when parameters interact strongly.
- Multi-host writers against one shared study are unsupported. Same-host conflicts are rejected with advisory locks.
- SQLite is for sequential `n_jobs == 1` studies. Parallel single-host studies should use `journal:///...` or an RDB storage URL.
