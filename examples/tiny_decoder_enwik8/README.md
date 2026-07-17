# Tiny Decoder Enwik8 Example

This example runs a tiny Enwik8 decoder training sweep with PhaseSweep. The trainer implementation comes from [`decoder-pytorch-template`](https://github.com/pszemraj/decoder-pytorch-template), checked out as the `upstream/` git submodule. The trainer accepts YAML config files but not per-key CLI overrides, so `run_trial.py` adapts PhaseSweep's `json_file` override format into one composed YAML file per trial. The model shape stays fixed in `base.yaml`; the three GPU-backed phases tune optimizer scale, regularization, and training stability.

## Setup

CLI setup from the PhaseSweep repo root (skip the first `pip install` if PhaseSweep is already installed in the environment):

```bash
git submodule update --init examples/tiny_decoder_enwik8/upstream
conda run -n tr --live-stream python -m pip install -e .
conda run -n tr --live-stream python -m pip install -e examples/tiny_decoder_enwik8/upstream
```

The submodule checkout also brings the dataset: `upstream/data/enwik8.gz` (~36 MB, from the Hutter Prize distribution) ships inside the trainer repo, so no separate download step is needed. `run_trial.py` runs the trainer with the upstream checkout as its working directory, which is how `base.yaml`'s relative `data_path: data/enwik8.gz` resolves.

For MCP runs, install PhaseSweep with the MCP extra instead:

```bash
conda run -n tr --live-stream python -m pip install -e ".[mcp]"
```

The submodule pins the external trainer revision used by this example without copying its source into PhaseSweep. Treat `upstream/` as external code: update the submodule pointer when you intentionally want a newer trainer, but keep adapter changes in this PhaseSweep example.

This is an orchestration smoke test, not a PyTorch training-template recommendation. The pinned trainer's known portability and numerical limitations are listed under [development work](../../docs/development.md#tracked-todos); fix them upstream, then update the submodule pointer here.

## CLI Smoke Sweep

```bash
conda run -n tr --live-stream phasesweep validate examples/tiny_decoder_enwik8/experiment.yaml
conda run -n tr --live-stream phasesweep run examples/tiny_decoder_enwik8/experiment.yaml --dry-run
conda run -n tr --live-stream phasesweep run examples/tiny_decoder_enwik8/experiment.yaml
conda run -n tr --live-stream phasesweep show-winners examples/tiny_decoder_enwik8/experiment.yaml
```

The real run launches 9 trials (3 phases x 3 trials, 1000 batches each) and finishes in a few minutes on a modern CUDA GPU - roughly 10-15 s per trial. Outputs land under `examples/tiny_decoder_enwik8/runs/`: the Optuna study at `runs/phases.db` and per-trial workdirs with `stdout.log`/`stderr.log` under `runs/trials/`, as configured in `experiment.yaml`.

The phase order is deliberate: pick `learning_rate` first because it is the highest-leverage optimizer scale decision, tune `weight_decay` after the update scale is fixed, then tune `grad_clip_norm` last as a stability/control knob. These are not perfectly independent, but they are closer to PhaseSweep's intended "mostly orthogonal consecutive sweeps" than mixing architecture shape, optimizer scale, and regularization in one chain.

The config uses 1000 training batches per trial. With `validate_every: 100` and steps numbered 0-999, the last validation runs at step 900, so each trial's reported `val_loss` comes from that final checkpoint rather than the very last batch - keep `validate_every` fixed when comparing runs. The upstream template does not currently expose warmup ratio or grouped-query attention controls, so this example sticks to trainer hyperparameters it supports. One more upstream quirk to know before sweeping shape keys: the SwiGLU feedforward rounds its hidden width up to a multiple of 256, so at `dim: 128` every `ffn_dim_multiplier` value up to 2.0 builds the same 256-wide FFN and sweeping the knob changes nothing until the requested width crosses a rounding step.

## MCP Smoke Sweep

The catalog pins the detached runner `cwd` to the PhaseSweep repo root, so the relative `trial_command` in `mcp_experiment.yaml` resolves consistently even if the MCP server is started from another shell cwd:

```bash
conda run -n tr --live-stream phasesweep mcp --catalog examples/tiny_decoder_enwik8/catalog.yaml
```

The MCP variant uses absolute scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-tiny-decoder-enwik8`, as required for restart-stable MCP runs.
