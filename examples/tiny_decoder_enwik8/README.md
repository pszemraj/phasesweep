# Tiny Decoder Enwik8 example

This example runs a tiny Enwik8 decoder training sweep with PhaseSweep using the pinned [`decoder-pytorch-template`](upstream/) git submodule ([upstream project](https://github.com/pszemraj/decoder-pytorch-template)). The model shape stays fixed while three phases tune optimizer scale, regularization, and training stability; [`run_trial.py`](run_trial.py) passes each materialized trainer config to the upstream trainer and publishes the final-checkpoint objective.

## Setup

From the PhaseSweep repo root, [install PhaseSweep](../../README.md#install-and-try-it), then prepare the pinned trainer:

```bash
git submodule update --init examples/tiny_decoder_enwik8/upstream
pip install -e examples/tiny_decoder_enwik8/upstream
```

The submodule checkout also brings the [dataset](https://github.com/pszemraj/decoder-pytorch-template/blob/9c90a551ae79061f0ed797e035462539c8a08403/data/README.md): `upstream/data/enwik8.gz` (~36 MB, from the Hutter Prize distribution) ships inside the trainer repo, so no separate download step is needed. The wrapper runs the trainer with the upstream checkout as its working directory, which is how the embedded relative `data_path: data/enwik8.gz` resolves.

For MCP runs, install the [MCP extra](../../docs/mcp_setup.md#1-install-the-mcp-extra) as well.

Review the pinned trainer's [device and precision support](https://github.com/pszemraj/decoder-pytorch-template/blob/9c90a551ae79061f0ed797e035462539c8a08403/README.md#device-selection--precision) and [requirements](https://github.com/pszemraj/decoder-pytorch-template/blob/9c90a551ae79061f0ed797e035462539c8a08403/README.md#requirements) before extending this example. Make trainer changes upstream, then update the submodule pointer here.

Run GPU validation, dry runs, and training from a shell with [host GPU access](../../docs/runtime.md#gpu-isolation). Run the commands below from the PhaseSweep repo root: `gpu_smoke.yaml`, `experiment.yaml` and `mlflow_gpu_smoke.yaml` leave `execution.cwd` unset, so their repo-relative `trial_command` resolves against the invocation directory.

## CLI smoke sweep

For a short GPU integration check, run the two-trial [`gpu_smoke.yaml`](gpu_smoke.yaml) first. The wrapper passes its `trainer_config` keys to the upstream trainer and requires `run_dir` for checkpoint output. The config leases one discovered GPU, runs 10 training batches per trial, sets trial/run timeouts, and writes its experiment outputs under `/tmp/phasesweep-tiny-decoder-enwik8-gpu-smoke`. The same commands use CUDA on Linux or MPS on Apple silicon macOS; the trainer chooses the device, and PhaseSweep holds its host lease:

```bash
phasesweep validate examples/tiny_decoder_enwik8/gpu_smoke.yaml
phasesweep run examples/tiny_decoder_enwik8/gpu_smoke.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/gpu_smoke.yaml
```

The smoke config uses in-memory Optuna storage so every `run` invocation executes both trials without accumulating a reusable study. Its winner must also pass a `runtime.accelerated == true` evidence gate, which verifies that final-checkpoint evaluation used CUDA or MPS; a CPU fallback cannot win. `result.json` retains the exact `runtime.device_type`. macOS discovers its MPS pool and refuses explicit CUDA device lists. Use [inspection commands](../../docs/runtime.md#inspection-commands) to read the saved winners. For a bounded comparison with seed noise and persistent resume, use [Learning rates over seeds](#learning-rates-over-seeds). The full config below uses an Optuna journal file for persistent status and reuse.

Use the full three-phase example only when you want the longer experiment:

```bash
phasesweep validate examples/tiny_decoder_enwik8/experiment.yaml
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml --dry-run
phasesweep run examples/tiny_decoder_enwik8/experiment.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/experiment.yaml
```

A fresh study targets nine terminal trial attempts (3 phases x 3 attempts, 1000 batches each). A resumed study launches only the attempts still needed to reach that target. Unlike the smoke config, the full configs leave device discovery to the runtime and do not enforce or gate CUDA use. Outputs land under `examples/tiny_decoder_enwik8/runs/`: the Optuna study at `runs/phases.journal` and per-trial workdirs with `stdout.log`/`stderr.log` under `runs/trials/`. The phase comments in [`experiment.yaml`](experiment.yaml) explain the learning-rate, weight-decay, and gradient-clipping order.

The example sweeps only supported trainer controls. The upstream template does not expose warmup ratio or grouped-query attention, and its SwiGLU feedforward rounds hidden width to a multiple of 256. At `dim: 128`, `ffn_dim_multiplier` values up to 2.0 therefore build the same 256-wide feedforward layer.

The upstream trainer records periodic validation with labels 0, 100, ..., 900, after that iteration's optimizer update, then saves `final.pt` at step 1000. After training exits, `run_trial.py` reloads that checkpoint and evaluates it once with the same validation settings. Only this step-1000 `final_checkpoint` evaluation is published through `report_objective(...)`; the periodic log minimum is not used. The configured [`json_envelope` extractor](../../docs/config.md#objective-evidence-constraints-and-gates) validates that report.

Checkpoint loading and final evaluation belong to this example's trainer wrapper. See [objective evidence](../../docs/config.md#objective-evidence-constraints-and-gates) for other reporting formats and [phase composition](../../docs/config.md#phase-composition) for inheritance.

## Learning rates over seeds

The runnable [`repeats.yaml`](repeats.yaml) demonstrates the repeats workflow: three learning rates, each run over seeds 7, 17, and 27. For a continuous key such as the learning rate, [more distinct values with one trial each](../../docs/config.md#noisy-objectives-and-small-budgets) usually tell more; repeats suit a few discrete finalists. It includes `storage: auto` and the required nonempty `provenance`, so it supports persisted status and same-config resume without editing the ephemeral smoke config. It resolves the trainer's working directory relative to the YAML file. When copying this YAML outside the checkout, set `execution.cwd` to the absolute PhaseSweep repository path. After the setup above, run from the PhaseSweep repository root:

```bash
phasesweep validate examples/tiny_decoder_enwik8/repeats.yaml
phasesweep run examples/tiny_decoder_enwik8/repeats.yaml --dry-run
phasesweep run examples/tiny_decoder_enwik8/repeats.yaml
phasesweep status examples/tiny_decoder_enwik8/repeats.yaml --trials
phasesweep show-winners examples/tiny_decoder_enwik8/repeats.yaml
phasesweep run examples/tiny_decoder_enwik8/repeats.yaml
```

The nine trials run sequentially on CUDA device `0`, with 80 optimizer updates, batch size 4, and sequence length 128 each. The objective is the saved step-80 `final.pt` loss over eight held-out validation batches. Each learning rate is ranked by its mean over the three seeds; `status --trials` includes every group's mean and sample SD. A within-noise advisory means the current seeds do not separate the candidate rates; this short example demonstrates the workflow rather than convergence.

The last command replays a completed study without training again. After an interruption, the same command keeps completed trials and reruns interrupted attempts before new cells. Artifacts and the journal stay under `/tmp/phasesweep-tiny-decoder-enwik8-repeats`. See [continuation](../../docs/runtime.md#timeouts-cleanup-and-continuation) for failure aborts and retry budgets.

## Continue one checkpoint

To continue training a particular saved model, use the pinned upstream trainer's existing `--resume` option. For a repeat winner, choose a seed explicitly; the aggregate score summarizes several models. `show-winners` prints the winning member trial numbers and a `# full record:` path. Open that record's `repeats.members` for the seeds and checkpoint evidence, then match the chosen `trial_number` to the phase's `trials.csv` column `user_attr:phasesweep_trial_dir`. This example saves its checkpoint at `trainer/final.pt` below that directory.

Run this from the PhaseSweep repository root, replacing the checkpoint path with the chosen member's file. The recipe reads the checkpoint's own trainer config and requests five additional optimizer updates. A step-80 checkpoint therefore finishes at step 85. The pinned trainer interprets `num_batches` as additional updates when resuming; setting it to 85 would instead finish at 165.

```bash
python - /absolute/path/to/selected-member/trainer/final.pt /tmp/phasesweep-decoder-continued <<'PY'
import subprocess
import sys
from pathlib import Path

import torch
import yaml

checkpoint = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=True)
config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"].copy()
config.update(num_batches=5, run_dir=str(output / "trainer"))
config_path = output / "trainer.yaml"
config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
upstream = Path("examples/tiny_decoder_enwik8/upstream").resolve()
subprocess.run(
    [sys.executable, str(upstream / "train.py"), "--config", str(config_path), "--resume", str(checkpoint)],
    cwd=upstream,
    check=True,
)
PY
```

The subprocess runs from the upstream checkout so the saved relative `data_path` still resolves. The continued checkpoint is `/tmp/phasesweep-decoder-continued/trainer/final.pt`. This standalone continuation restores model and optimizer state; the pinned trainer does not restore RNG or data-loader state, so it does not reproduce an uninterrupted stochastic trajectory. It leaves the sweep's saved validation scores intact.

## Native MLflow GPU smoke

The [MLflow smoke configuration](mlflow_gpu_smoke.yaml) uses the same two learning rates, model, and ten-batch budget as the local GPU fixture. It runs one trial at a time on CUDA device `0`. Complete the trainer setup above, then install MLflow and start the local tracking server as in [Run the CPU example locally](../../docs/mlflow.md#run-the-cpu-example-locally). In another terminal, run from the PhaseSweep repository root. Create the experiment once; if it already exists, reuse it:

```bash
python -c 'from mlflow import MlflowClient; c = MlflowClient(tracking_uri="http://127.0.0.1:5000"); c.get_experiment_by_name("phasesweep-tiny-decoder") or c.create_experiment("phasesweep-tiny-decoder")'
phasesweep validate examples/tiny_decoder_enwik8/mlflow_gpu_smoke.yaml
phasesweep run examples/tiny_decoder_enwik8/mlflow_gpu_smoke.yaml
phasesweep show-winners examples/tiny_decoder_enwik8/mlflow_gpu_smoke.yaml
phasesweep status examples/tiny_decoder_enwik8/mlflow_gpu_smoke.yaml
```

The wrapper's `--mlflow` option logs the evaluated `final.pt` loss as native `val_loss` at step `10`, with trainer parameters and evaluation/device tags. It logs `runtime/cuda=1` when final evaluation uses CUDA; a native constraint requires that value before a trial can win. Periodic upstream validation losses remain local and do not enter the objective. In this mode, the objective comes from MLflow rather than a JSON envelope. The upstream trainer runs as pinned, with no modification.

The default `mlflow.run_mode: managed` resumes PhaseSweep's supplied run. To check trainer ownership, copy the configuration, change `run_mode` to `trainer`, and choose a fresh `experiment` and `workdir`. The wrapper then creates its run, calls `bind_mlflow_run`, and closes it after evaluation. Both modes publish results and a generation summary to the same configured MLflow experiment. Open `http://127.0.0.1:5000` to inspect the native metrics and `phasesweep.*` results. See [MLflow tracking](../../docs/mlflow.md) for ownership, selectors, offline replay, and pending-upload recovery.

Unlike `gpu_smoke.yaml`, this variant uses persistent storage: a completed `run` is reused without training again. For fresh trials, copy the config with a new local experiment name and workdir. When placing a copy outside the repo, set `execution.cwd` to the absolute PhaseSweep repository path so the relative trainer command still resolves. To use another server, change `mlflow.tracking_uri` and provision `mlflow.experiment_name` there first.

## MCP smoke

For the same two-trial, 10-batch check through MCP, start from [`gpu_smoke.yaml`](gpu_smoke.yaml) instead of the full MCP experiment. Following the [fresh MCP state requirement](../../docs/mcp.md#fresh-mcp-state), copy it into a new scratch directory from the repo root:

```bash
mkdir -p /tmp/phasesweep-tiny-decoder-mcp-smoke
cp examples/tiny_decoder_enwik8/gpu_smoke.yaml /tmp/phasesweep-tiny-decoder-mcp-smoke/experiment.yaml
```

In that copy, replace or add these root keys, using your actual absolute repo path for `execution.cwd`. Keep its trainer configuration, metric, and two-trial phase unchanged:

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

These root-key changes preserve the smoke search while satisfying the [persistent storage requirement for MCP](../../docs/mcp.md#the-catalog).

```bash
phasesweep validate /tmp/phasesweep-tiny-decoder-mcp-smoke/experiment.yaml
phasesweep mcp init-catalog --from /tmp/phasesweep-tiny-decoder-mcp-smoke/experiment.yaml -o /tmp/phasesweep-tiny-decoder-mcp-smoke/catalog.yaml
```

Review the generated catalog, enable `allow.launch`, and set `visible_params: all` if the agent should report the winning learning rate. Follow the [MCP client setup](../../docs/mcp_setup.md), authorize the two-trial smoke, then use the [status response workflow](../../docs/mcp.md#reading-status-responses). A completed persistent smoke is reused on a later launch; choose a new experiment name and scratch directory when you want two fresh trials.
## MCP full sweep

Run the full sweep through [`catalog.yaml`](catalog.yaml), which registers [`mcp_experiment.yaml`](mcp_experiment.yaml) with the PhaseSweep repo root as its runner `cwd`:

```bash
phasesweep mcp check --catalog examples/tiny_decoder_enwik8/catalog.yaml
```

Configure your client with the absolute `phasesweep-mcp` executable and this catalog's absolute path as described in [MCP setup](../../docs/mcp_setup.md#3-configure-your-client), then [verify the read-only catalog operation](../../docs/mcp_setup.md#4-verify-the-connection).

The MCP variant uses scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-tiny-decoder-enwik8`; choose a fresh root before the first run as described in the [fresh MCP state](../../docs/mcp.md#fresh-mcp-state) section.
