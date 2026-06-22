# Decoder PyTorch Template Example

This example drives the real [`decoder-pytorch-template`](https://github.com/pszemraj/decoder-pytorch-template) trainer with PhaseSweep without modifying the trainer repo. The trainer currently accepts a YAML config but not per-key CLI overrides, so `run_trial.py` adapts PhaseSweep's existing `json_file` override format into one composed YAML file per trial. The model shape stays fixed in `base.yaml`; the three GPU-backed phases tune optimizer scale, regularization, and training stability.

## Setup

From the PhaseSweep repo root:

```bash
git clone https://github.com/pszemraj/decoder-pytorch-template.git examples/decoder_pytorch_template/vendor/decoder-pytorch-template
conda run -n tr --live-stream python -m pip install -e ".[mcp]"
conda run -n tr --live-stream python -m pip install -e examples/decoder_pytorch_template/vendor/decoder-pytorch-template
```

The `vendor/` checkout is gitignored on purpose. It keeps the example honest while avoiding vendored model code and dataset churn in this repo.

## CLI Smoke Sweep

```bash
conda run -n tr --live-stream phasesweep validate examples/decoder_pytorch_template/experiment.yaml
conda run -n tr --live-stream phasesweep run examples/decoder_pytorch_template/experiment.yaml --dry-run
conda run -n tr --live-stream phasesweep run examples/decoder_pytorch_template/experiment.yaml
conda run -n tr --live-stream phasesweep show-winners examples/decoder_pytorch_template/experiment.yaml
```

The phase order is deliberate: pick `learning_rate` first because it is the highest-leverage optimizer scale decision, tune `weight_decay` after the update scale is fixed, then tune `grad_clip_norm` last as a stability/control knob. These are not perfectly independent, but they are closer to PhaseSweep's intended "mostly orthogonal consecutive sweeps" than mixing architecture shape, optimizer scale, and regularization in one chain.

The committed config uses 1000 training batches per trial. It is still compact enough to run as an integration example on a single local GPU, but it is no longer a toy two-step smoke test. The upstream template does not currently expose warmup ratio or grouped-query attention controls, so this example sticks to trainer hyperparameters it actually supports.

## MCP Smoke Sweep

Run from the PhaseSweep repo root so the relative `trial_command` in `mcp_experiment.yaml` resolves correctly:

```bash
conda run -n tr --live-stream phasesweep mcp --catalog examples/decoder_pytorch_template/catalog.yaml
```

The MCP variant uses absolute scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-decoder-template`, as required for restart-stable MCP runs.

## One Agent Run

This is a report from one local validation run, not a prescription for the best decoder-pytorch-template settings. The point is to show the workflow an agent followed and the shape of the result PhaseSweep returned.

The run used an NVIDIA GeForce RTX 4070 Laptop GPU through the `tr` conda environment. PhaseSweep launched 9 trials total: 3 learning-rate trials, then 3 weight-decay trials with the winning learning rate inherited, then 3 gradient-clipping trials with learning rate and weight decay inherited. Each trial trained for 1000 batches from `base.yaml`. Trainer logs for both the CLI and MCP runs reported `Device: cuda` and BF16 mixed precision.

The CLI and MCP runs produced the same phase winners:

```text
optimizer_scale: learning_rate=0.001, val_loss=2.140831208229065
weight_decay: weight_decay=0.0, val_loss=2.140831208229065
clip_norm: grad_clip_norm=0.5, val_loss=2.096618318557739
```

The MCP path used catalog id `decoder-template-hparams`: list experiments, validate the phase structure, launch the sweep, poll status by `run_id`, and read winners by that same `run_id`. The observed MCP run completed all three phases with 3 complete trials each.
