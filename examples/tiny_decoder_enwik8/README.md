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

The pinned trainer has [known portability and numerical limitations](../../docs/development.md#tracked-todos). Fix them upstream, then update the submodule pointer here.

## CLI smoke sweep

For a short GPU integration check, run the two-trial [`gpu_smoke.yaml`](gpu_smoke.yaml) first. The wrapper passes its `trainer_config` keys to the upstream trainer and requires `run_dir` for checkpoint output. The config leases CUDA device `0`, runs 10 training batches per trial, sets trial/run timeouts, and writes its experiment outputs under `/tmp/phasesweep-tiny-decoder-enwik8-gpu-smoke`:

```bash
phasesweep validate examples/tiny_decoder_enwik8/gpu_smoke.yaml
phasesweep run examples/tiny_decoder_enwik8/gpu_smoke.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/gpu_smoke.yaml
```

The smoke config uses in-memory Optuna storage so every `run` invocation executes both trials without accumulating a reusable study. Its winner must also pass a `runtime.device_type == 'cuda'` evidence gate, which verifies that final-checkpoint evaluation used CUDA. Use [inspection commands](../../docs/runtime.md#inspection-commands) to read the saved winners. The full config below uses SQLite for persistent status and reuse.

Use the full three-phase example only when you intentionally want the longer experiment:

```bash
phasesweep validate examples/tiny_decoder_enwik8/experiment.yaml
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml --dry-run
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/experiment.yaml
```

A fresh study targets nine terminal trial attempts (3 phases x 3 attempts, 1000 batches each). A resumed study launches only the attempts still needed to reach that target. Runtime depends on the local hardware and software stack. Unlike the smoke config, the full configs leave device discovery to the runtime and do not enforce or gate CUDA use. Outputs land under `examples/tiny_decoder_enwik8/runs/`: the Optuna study at `runs/phases.db` and per-trial workdirs with `stdout.log`/`stderr.log` under `runs/trials/`. The phase comments in [`experiment.yaml`](experiment.yaml) explain the learning-rate, weight-decay, and gradient-clipping order.

The upstream trainer records periodic validation with labels 0, 100, ..., 900, after that iteration's optimizer update, then saves `final.pt` at step 1000. After training exits, `run_trial.py` reloads that checkpoint and evaluates it once with the same validation settings. Only this step-1000 `final_checkpoint` evaluation is published through `report_objective(...)`; the periodic log minimum is not used. The configured [`json_envelope` extractor](../../docs/config.md#objective-evidence-constraints-and-gates) validates that report.

Checkpoint loading and final evaluation belong to this example's trainer
wrapper. PhaseSweep reads the reported scalar and checks its envelope metadata;
it does not evaluate models. Other trainers can use ordinary JSON, log matching,
or finished W&B summaries without the envelope helper. Phase inheritance carries
selected parameter values, not checkpoint weights.

The example sweeps only supported trainer controls. The upstream template does not expose warmup ratio or grouped-query attention, and its SwiGLU feedforward rounds hidden width to a multiple of 256. At `dim: 128`, `ffn_dim_multiplier` values up to 2.0 therefore build the same 256-wide feedforward layer.

## MCP smoke

For the same two-trial, 10-batch check through MCP, start from [`gpu_smoke.yaml`](gpu_smoke.yaml)
instead of the full MCP experiment. Following the [runtime
cutover](../../docs/runtime.md#fresh-state-cutover) and [fresh MCP state
requirement](../../docs/mcp.md#fresh-mcp-state), copy it into a new scratch
directory from the repo root:

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
[persistent storage and absolute path requirements for MCP](../../docs/mcp.md#the-catalog).

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
```

Configure your client with the absolute `phasesweep-mcp` executable and this catalog's absolute path, then verify the read-only catalog operation as described in [MCP setup](../../docs/mcp_setup.md#3-configure-your-client).

The MCP variant uses scratch `workdir`, storage, and state paths under
`/tmp/phasesweep-mcp-tiny-decoder-enwik8`; choose a fresh root before the first
run with this release as described in the [runtime
cutover](../../docs/runtime.md#fresh-state-cutover) and [fresh MCP
state](../../docs/mcp.md#fresh-mcp-state) sections.
