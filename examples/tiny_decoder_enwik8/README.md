# Tiny Decoder Enwik8 example

This example runs a tiny Enwik8 decoder training sweep with PhaseSweep. The trainer implementation comes from the pinned [`decoder-pytorch-template`](upstream/) git submodule ([upstream project](https://github.com/pszemraj/decoder-pytorch-template)). Its complete base configuration lives under `trainer_config` in the same PhaseSweep YAML as the search plan. PhaseSweep applies each trial's inherited, fixed, and sampled values and materializes `trainer_config.yaml`; `run_trial.py` passes that file to the upstream trainer and publishes the final-checkpoint objective. The model shape stays fixed while the three phases tune optimizer scale, regularization, and training stability.

## Setup

From the PhaseSweep repo root, [install PhaseSweep](../../README.md#install-and-try-it), then prepare the pinned trainer:

```bash
git submodule update --init examples/tiny_decoder_enwik8/upstream
pip install -e examples/tiny_decoder_enwik8/upstream
```

The submodule checkout also brings the dataset: `upstream/data/enwik8.gz` (~36 MB, from the Hutter Prize distribution) ships inside the trainer repo, so no separate download step is needed. `run_trial.py` runs the trainer with the upstream checkout as its working directory, which is how the embedded relative `data_path: data/enwik8.gz` resolves.

For MCP runs, install the [MCP extra](../../docs/mcp_setup.md#1-install-the-mcp-extra) as well.

This is an orchestration example, not a PyTorch training-template recommendation. The pinned trainer's known portability and numerical limitations are listed under [development work](../../docs/development.md#tracked-todos); fix them upstream, then update the submodule pointer here.

## CLI smoke sweep

For a genuinely short GPU integration check, run the dedicated two-trial [`gpu_smoke.yaml`](gpu_smoke.yaml) first. It is also the canonical minimal schema for this example's `trainer_config`: the wrapper passes those upstream trainer keys through unchanged and requires `run_dir` for checkpoint output. The config explicitly leases CUDA device `0`, runs 10 training batches per trial, disables W&B, has bounded trial/run timeouts, and writes its experiment outputs under `/tmp/phasesweep-tiny-decoder-enwik8-gpu-smoke`:

```bash
phasesweep validate examples/tiny_decoder_enwik8/gpu_smoke.yaml
phasesweep run examples/tiny_decoder_enwik8/gpu_smoke.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/gpu_smoke.yaml
```

The smoke config deliberately uses in-memory Optuna storage so every invocation runs both trials without accumulating a reusable study. Its winner must also pass a `runtime.device_type == 'cuda'` evidence gate, which verifies final-checkpoint evaluation actually used CUDA rather than merely showing that PhaseSweep leased a GPU. `show-winners` reads the persisted last-success artifacts; a separate later `status` process cannot reconstruct the completed in-memory trial counts. The full config below uses SQLite when persistent status and reuse matter; the [configuration guide](../../docs/config.md#sampler-capability-on-persistent-storage) covers persistent sampler and top-up behavior.

Use the full three-phase example only when you intentionally want the longer experiment:

```bash
phasesweep validate examples/tiny_decoder_enwik8/experiment.yaml
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml --dry-run
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/experiment.yaml
```

A fresh study targets nine terminal trial attempts (3 phases x 3 attempts, 1000 batches each). A resumed study launches only the attempts still needed to reach that target. Runtime depends on the local hardware and software stack. Unlike the smoke config, the full configs leave device discovery to the runtime and do not enforce or gate CUDA use. Outputs land under `examples/tiny_decoder_enwik8/runs/`: the Optuna study at `runs/phases.db` and per-trial workdirs with `stdout.log`/`stderr.log` under `runs/trials/`, as configured in `experiment.yaml`.

The phase order is deliberate: `optimizer_scale` selects `learning_rate` first because it is the highest-leverage scale decision; `weight_decay` tunes that parameter after the update scale is fixed; then `clip_norm` selects `grad_clip_norm` as a stability/control knob. These are not perfectly independent, but they are closer to PhaseSweep's intended "mostly orthogonal consecutive sweeps" than mixing architecture shape, optimizer scale, and regularization in one chain.

The config uses 1000 training batches per trial. The upstream trainer validates periodically at steps 0-900, then saves `final.pt` at step 1000. After training exits, `run_trial.py` reloads that checkpoint and evaluates it once with the same validation settings. Only this step-1000 `final_checkpoint` evaluation is published through `report_objective(...)`; the periodic log minimum is not used. The configured [`json_envelope` extractor](../../docs/config.md#extractors) verifies its attempt identity, overrides digest, and evaluation policy.

The example sweeps only supported trainer controls. The upstream template does not expose warmup ratio or grouped-query attention, and its SwiGLU feedforward rounds hidden width to a multiple of 256. At `dim: 128`, `ffn_dim_multiplier` values up to 2.0 therefore build the same 256-wide feedforward layer.

## MCP full sweep

The MCP catalog exposes the same nine-attempt target with 1000 batches per attempt as the full CLI config above; it is not the two-attempt quick smoke. It pins the detached runner `cwd` to the PhaseSweep repo root, so the relative `trial_command` in `mcp_experiment.yaml` resolves consistently even if the MCP server is started from another shell cwd:

```bash
phasesweep mcp check --catalog examples/tiny_decoder_enwik8/catalog.yaml
phasesweep mcp install --catalog examples/tiny_decoder_enwik8/catalog.yaml --dry-run
phasesweep mcp install --catalog examples/tiny_decoder_enwik8/catalog.yaml
```

Restart the selected client after installation, then ask it to list the available PhaseSweep experiments. `phasesweep mcp serve` is a stdio JSON-RPC endpoint for MCP clients, not an interactive terminal interface; use `mcp check` for a direct startup preflight.

The MCP variant uses absolute scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-tiny-decoder-enwik8`, as required for restart-stable MCP runs.

Both configs declare trainer and data provenance because they reuse persistent studies. The embedded trainer config is fingerprinted automatically; update the external provenance tokens whenever the wrapper, pinned template revision, data preparation, or dependencies change so PhaseSweep refuses an incompatible top-up.
