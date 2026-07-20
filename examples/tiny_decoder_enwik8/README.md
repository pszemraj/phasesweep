# Tiny Decoder Enwik8 example

This example runs a tiny Enwik8 decoder training sweep with phasesweep. The trainer implementation comes from [`decoder-pytorch-template`](https://github.com/pszemraj/decoder-pytorch-template), checked out as the `upstream/` git submodule. The trainer accepts YAML config files but not per-key CLI overrides, so `run_trial.py` adapts phasesweep's `json_file` override format into one composed YAML file per trial. The model shape stays fixed in `base.yaml`; the three GPU-backed phases tune optimizer scale, regularization, and training stability.

## Setup

From the phasesweep repo root, [install phasesweep](../../README.md#install), then prepare the pinned trainer:

```bash
git submodule update --init examples/tiny_decoder_enwik8/upstream
python -m pip install -e examples/tiny_decoder_enwik8/upstream
```

The submodule checkout also brings the dataset: `upstream/data/enwik8.gz` (~36 MB, from the Hutter Prize distribution) ships inside the trainer repo, so no separate download step is needed. `run_trial.py` runs the trainer with the upstream checkout as its working directory, which is how `base.yaml`'s relative `data_path: data/enwik8.gz` resolves.

For MCP runs, install the [MCP extra](../../docs/mcp_setup.md#1-install) as well.

The submodule pins the external trainer revision used by this example without copying its source into phasesweep. Treat `upstream/` as external code: update the submodule pointer when you intentionally want a newer trainer, but keep adapter changes in this phasesweep example.

This is an orchestration smoke test, not a PyTorch training-template recommendation. The pinned trainer's known portability and numerical limitations are listed under [development work](../../docs/development.md#tracked-todos); fix them upstream, then update the submodule pointer here.

## CLI smoke sweep

```bash
phasesweep validate examples/tiny_decoder_enwik8/experiment.yaml
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml --dry-run
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/experiment.yaml
```

The real run launches 9 trials (3 phases x 3 trials, 1000 batches each). Runtime depends on the local CUDA hardware and software stack. Outputs land under `examples/tiny_decoder_enwik8/runs/`: the Optuna study at `runs/phases.db` and per-trial workdirs with `stdout.log`/`stderr.log` under `runs/trials/`, as configured in `experiment.yaml`.

The phase order is deliberate: pick `learning_rate` first because it is the highest-leverage optimizer scale decision, tune `weight_decay` after the update scale is fixed, then tune `grad_clip_norm` last as a stability/control knob. These are not perfectly independent, but they are closer to PhaseSweep's intended "mostly orthogonal consecutive sweeps" than mixing architecture shape, optimizer scale, and regularization in one chain.

The config uses 1000 training batches per trial. The upstream trainer validates periodically at steps 0-900, then saves `final.pt` at step 1000. After training exits, `run_trial.py` reloads that checkpoint and evaluates it once with the same validation settings. Only this step-1000 `final_checkpoint` evaluation is published as the PhaseSweep objective. The configured [`json_envelope` extractor](../../docs/config.md#extractors) verifies its attempt identity, overrides digest, and evaluation policy.

The example sweeps only supported trainer controls. The upstream template does not expose warmup ratio or grouped-query attention, and its SwiGLU feedforward rounds hidden width to a multiple of 256. At `dim: 128`, `ffn_dim_multiplier` values up to 2.0 therefore build the same 256-wide feedforward layer.

## MCP smoke sweep

The catalog pins the detached runner `cwd` to the PhaseSweep repo root, so the relative `trial_command` in `mcp_experiment.yaml` resolves consistently even if the MCP server is started from another shell cwd:

```bash
phasesweep mcp check --catalog examples/tiny_decoder_enwik8/catalog.yaml
phasesweep mcp install --catalog examples/tiny_decoder_enwik8/catalog.yaml --dry-run
phasesweep mcp install --catalog examples/tiny_decoder_enwik8/catalog.yaml
```

Restart the selected client after installation, then ask it to list the available phasesweep experiments. To exercise the stdio server directly instead of installing a client entry:

```bash
phasesweep mcp serve --catalog examples/tiny_decoder_enwik8/catalog.yaml
```

The MCP variant uses absolute scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-tiny-decoder-enwik8`, as required for restart-stable MCP runs.

Both configs declare trainer and data provenance because they reuse persistent studies. Update those tokens whenever the wrapper, pinned template revision, base config, data preparation, or dependencies change; the changed provenance will make PhaseSweep refuse an incompatible top-up.
