# Decoder PyTorch Template Example

This example drives the real [`decoder-pytorch-template`](https://github.com/pszemraj/decoder-pytorch-template) trainer with PhaseSweep without modifying the trainer repo. The trainer currently accepts a YAML config but not per-key CLI overrides, so `run_trial.py` adapts PhaseSweep's existing `json_file` override format into one composed YAML file per trial.

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
CUDA_VISIBLE_DEVICES="" conda run -n tr --live-stream phasesweep validate examples/decoder_pytorch_template/experiment.yaml
CUDA_VISIBLE_DEVICES="" conda run -n tr --live-stream phasesweep run examples/decoder_pytorch_template/experiment.yaml --dry-run
CUDA_VISIBLE_DEVICES="" conda run -n tr --live-stream phasesweep run examples/decoder_pytorch_template/experiment.yaml
CUDA_VISIBLE_DEVICES="" conda run -n tr --live-stream phasesweep show-winners examples/decoder_pytorch_template/experiment.yaml
```

The committed config is deliberately tiny and CPU-forced so it is a functional integration smoke test, not a meaningful language-model training run. Remove `FORCE_DEVICE: cpu`, remove the empty `CUDA_VISIBLE_DEVICES`, increase the budgets in `base.yaml`, and widen the search space when using this as a real experiment.

## MCP Smoke Sweep

Run from the PhaseSweep repo root so the relative `trial_command` in `mcp_experiment.yaml` resolves correctly:

```bash
CUDA_VISIBLE_DEVICES="" conda run -n tr --live-stream phasesweep mcp --catalog examples/decoder_pytorch_template/catalog.yaml
```

The MCP variant uses absolute scratch `workdir`, storage, and state paths under `/tmp/phasesweep-mcp-decoder-template`, as required for restart-stable MCP runs.
