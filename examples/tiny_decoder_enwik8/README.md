# Tiny Decoder Enwik8 example

This example runs a tiny Enwik8 decoder training sweep with PhaseSweep using the pinned [`decoder-pytorch-template`](upstream/) git submodule ([upstream project](https://github.com/pszemraj/decoder-pytorch-template)). The model shape stays fixed while three phases tune optimizer scale, regularization, and training stability; [`run_trial.py`](run_trial.py) passes each materialized trainer config to the upstream trainer and publishes the final-checkpoint objective.

## Setup

From the PhaseSweep repo root, [install PhaseSweep](../../README.md#install-and-try-it), then prepare the pinned trainer:

```bash
git submodule update --init examples/tiny_decoder_enwik8/upstream
pip install -e examples/tiny_decoder_enwik8/upstream
```

The submodule checkout also brings the [dataset](upstream/data/README.md): `upstream/data/enwik8.gz` (~36 MB, from the Hutter Prize distribution) ships inside the trainer repo, so no separate download step is needed. The wrapper runs the trainer with the upstream checkout as its working directory, which is how the embedded relative `data_path: data/enwik8.gz` resolves.

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

A fresh study targets nine terminal trial attempts (3 phases x 3 attempts, 1000 batches each). A resumed study launches only the attempts still needed to reach that target. Runtime depends on the local hardware and software stack. Unlike the smoke config, the full configs leave device discovery to the runtime and do not enforce or gate CUDA use. Outputs land under `examples/tiny_decoder_enwik8/runs/`: the Optuna study at `runs/phases.db` and per-trial workdirs with `stdout.log`/`stderr.log` under `runs/trials/`. The phase comments in [`experiment.yaml`](experiment.yaml) explain the learning-rate, weight-decay, and gradient-clipping order.

The upstream trainer validates periodically at steps 0-900, then saves `final.pt` at step 1000. After training exits, `run_trial.py` reloads that checkpoint and evaluates it once with the same validation settings. Only this step-1000 `final_checkpoint` evaluation is published through `report_objective(...)`; the periodic log minimum is not used. The configured [`json_envelope` extractor](../../docs/config.md#extractors) verifies its attempt identity, overrides digest, and evaluation policy.

The example sweeps only supported trainer controls. The upstream template does not expose warmup ratio or grouped-query attention, and its SwiGLU feedforward rounds hidden width to a multiple of 256. At `dim: 128`, `ffn_dim_multiplier` values up to 2.0 therefore build the same 256-wide feedforward layer.

## MCP smoke

For the same two-trial, 10-batch check through MCP, start from [`gpu_smoke.yaml`](gpu_smoke.yaml)
instead of the full MCP experiment. From the repo root, copy it into a fresh
scratch directory:

```bash
mkdir -p /tmp/phasesweep-tiny-decoder-mcp-smoke
cp examples/tiny_decoder_enwik8/gpu_smoke.yaml /tmp/phasesweep-tiny-decoder-mcp-smoke/experiment.yaml
```

In that copy, replace or add these **root keys**, using your actual absolute
repo path for `execution.cwd`. Keep its trainer configuration, metric, and
two-trial phase unchanged:

```yaml
experiment: tiny_decoder_enwik8_mcp_smoke
storage: auto
workdir: /tmp/phasesweep-tiny-decoder-mcp-smoke/runs
provenance:
  trainer: "run-trial-v2+decoder-template@9c90a551"
  data: "enwik8-template-download"
execution:
  cwd: /absolute/path/to/phasesweep
```

These root-key changes preserve the smoke search while satisfying the
[persistent storage and absolute path requirements for MCP](../../docs/mcp.md#paths-and-the-working-directory).

```bash
phasesweep validate /tmp/phasesweep-tiny-decoder-mcp-smoke/experiment.yaml
phasesweep mcp init-catalog --from /tmp/phasesweep-tiny-decoder-mcp-smoke/experiment.yaml -o /tmp/phasesweep-tiny-decoder-mcp-smoke/catalog.yaml
```

Review the generated catalog, enable `allow.launch`, and set `visible_params:
all` if the agent should report the winning learning rate. Follow the
[MCP client setup](../../docs/mcp_setup.md), authorize the two-trial smoke,
then use the [status response workflow](../../docs/mcp.md#reading-status-responses).
A completed persistent smoke is
reused on a later launch; choose a new experiment name and scratch directory
when you want two fresh trials. GPU leasing does not impose a VRAM quota:
verify the workload and monitor GPU memory separately when sharing a device.

## MCP full sweep

Run the full sweep through [`catalog.yaml`](catalog.yaml), which registers [`mcp_experiment.yaml`](mcp_experiment.yaml) with the PhaseSweep repo root as its runner `cwd`:

```bash
phasesweep mcp check --catalog examples/tiny_decoder_enwik8/catalog.yaml
phasesweep mcp install --catalog examples/tiny_decoder_enwik8/catalog.yaml --dry-run
phasesweep mcp install --catalog examples/tiny_decoder_enwik8/catalog.yaml
```

After installation, follow the [client restart and verification step](../../docs/mcp_setup.md#4-restart-and-verify).

The MCP variant uses scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-tiny-decoder-enwik8`.
