# Native MLflow tracking

PhaseSweep reads metrics that your trainer logs natively to MLflow, selects winners locally, and mirrors results to the same MLflow experiment. A sweep uses one tracking backend. W&B-to-MLflow publishing and publishing-only MLflow configuration are unsupported.

## Run the CPU example locally

Install the optional client and, for a local server, matching full MLflow:

```bash
python -m pip install -e ".[mlflow]" "mlflow==3.17.0"
mkdir -p /tmp/phasesweep-mlflow-demo/artifacts
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////tmp/phasesweep-mlflow-demo/mlflow.db \
  --serve-artifacts \
  --artifacts-destination /tmp/phasesweep-mlflow-demo/artifacts
```

In another terminal, provision the experiment explicitly, then run the example from the repository root:

```bash
python -c 'from mlflow import MlflowClient; MlflowClient(tracking_uri="http://127.0.0.1:5000").create_experiment("phasesweep-native")'
phasesweep validate examples/mlflow.yaml
phasesweep run examples/mlflow.yaml
phasesweep show-winners examples/mlflow.yaml
phasesweep status examples/mlflow.yaml
```

The [complete configuration](../examples/mlflow.yaml) launches four deterministic CPU trials across two phases. The depth winner is `model.depth=2` with loss 9; the rate phase inherits that depth and selects `optimizer.rate=3` with loss 0. The [trainer](../src/phasesweep/examples/mlflow_train.py) uses `mlflow.log_metric`, `log_params`, and `set_tag` directly.

For real training on a GPU, the [Tiny Decoder MLflow smoke guide](../examples/tiny_decoder_enwik8/README.md#native-mlflow-gpu-smoke) provides a complete two-trial, ten-batch configuration using native final-checkpoint metrics, with both run ownership modes supported.

## Configure the source experiment

```yaml
mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: phasesweep-native  # or experiment_id: "1", exactly one
  run_mode: managed                  # managed | trainer
  poll_seconds: 2
  timeout_seconds: 120
metric:
  name: eval_loss
  goal: minimize
  extractor: {type: mlflow, metric_key: eval/loss, select: last}
```

The experiment must already exist and be active. PhaseSweep never creates it implicitly. `tracking_uri` accepts HTTP(S), `databricks`, and `databricks://profile`. HTTP credentials belong in the SDK's authentication configuration or environment rather than the URI. The top-level `mlflow` block requires an MLflow objective. Local constraint extractors and local gates can still accompany it, but every remote consumer uses the bound MLflow run.

All controller network operations run in supervised workers outside GPU leases. The timeout bounds each operation; a phase or whole-run deadline can shorten it. Within that bound the MLflow client retries transient HTTP failures, except run creation, which reconciles by identity instead of being replayed. Trainer cleanup is confirmed before releasing its GPU lease, and launch checks cancellation and deadlines again after acquiring that lease.

## Choose run ownership

In **managed** mode, PhaseSweep writes the trainer input, creates the native MLflow run, and persists its returned ID before acquiring a GPU or launching training. The trainer receives `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and `MLFLOW_RUN_ID`. Resume that supplied run:

```python
import os
import mlflow

with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

PhaseSweep finishes a still-running managed run after confirmed process cleanup: FINISHED after successful trainer exit, FAILED after a failed trial, or KILLED after a pruned one. A run that is already terminal keeps its status, and its `phasesweep.outcome` tag records a later trial failure. A FAILED or KILLED run is never converted into accepted evidence. Every creation request first searches the exact generation and attempt identity tags and binds a run it finds, so a retry after a lost response does not create a second run; it creates one only when the search finds none.

In **trainer** mode, your existing logger creates and finishes the run. Bind that exact run once it exists:

```python
import mlflow
from phasesweep.tracking import bind_mlflow_run

with mlflow.start_run() as run:
    bind_mlflow_run(run.info.run_id)
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

The helper verifies the configured experiment, attaches PhaseSweep identity tags, and writes the local binding. It does not log or replace metrics. PhaseSweep never discovers runs by name or recency, finishes trainer-owned runs, or changes their existing parent relationships. The trainer must close its run successfully before evidence can be accepted. Existing framework loggers must expose their actual run ID and honor this ownership contract; other logger integrations require their own qualification.

### Transformers Trainer

For native Transformers logging, set `report_to: [mlflow]` in the trainer's arguments. Keep the managed run active across both training and any subsequent evaluation:

```python
import os
import mlflow

run_id = os.environ.pop("MLFLOW_RUN_ID")
with mlflow.start_run(run_id=run_id):
    trainer.train()
    trainer.evaluate()
```

Here `trainer` is your configured Transformers Trainer. Removing `MLFLOW_RUN_ID` before its MLflow callback starts lets the callback use the active run without taking ownership of closing it. Otherwise, the callback can close the run when training ends, and a later evaluation can log to a different run. When wrapping an official example script, keep its entire execution inside this context. For example, save this small launcher next to your experiment configuration:

```python
import os
import runpy
import sys

import mlflow

script = sys.argv[1]
sys.argv = sys.argv[1:]  # Preserve the example's own argument parsing.
run_id = os.environ.pop("MLFLOW_RUN_ID")
with mlflow.start_run(run_id=run_id):
    runpy.run_path(script, run_name="__main__")
```

Configure `trial_command` to invoke this launcher with the absolute example path followed by its normal arguments and `{overrides}`; set `override_format: argparse` for command-line Transformers arguments. This launcher is for managed ownership only. It preserves native logging through `report_to: [mlflow]`, and an exception exits the context as a failed run. Verify that the example imports its matching Transformers package using the [trainer environment preflight](config.md#source-checkout-preflight).

Native metrics retain their Trainer names, such as `eval_loss` and `eval_rouge1`; choose the exact final optimizer step when comparing fixed training budgets.

Set model storage precision explicitly when preparing mixed-precision training. In the tested Transformers checkout, BART's checkpoint loaded FP16 weights when the example omitted `dtype`, producing NaN metrics. BF16 autocast did not change that parameter storage. Loading with `from_pretrained(..., dtype=torch.float32)` supplies FP32 parameters for training while `bf16: true` controls autocast separately.

Check the model head's loss normalization when using gradient accumulation. In this checkout, the BART summarization and ModernBERT token-classification heads compute mean cross-entropy but accept `**kwargs` without using `num_items_in_batch`. Trainer consequently assumes the loss is already normalized across accumulated microbatches. For these heads, set `model.accepts_loss_kwargs = False` before constructing Trainer so it averages the microbatch losses. This is microbatch averaging, rather than a global mean over all nonignored tokens. The earlier BART qualification omitted this setting; its measured evaluation scores and tracking checks stand, but its accumulated training gradients were summed rather than averaged.

For token classification, select native `eval_f1` at the final optimizer step with `metric.goal: maximize`. Keep the validation rows and word-label alignment fixed across candidates. When sweeping `label_smoothing_factor`, compare F1: Trainer applies smoothing to evaluation loss too, so losses from different smoothing settings have different definitions.

The official summarization example accepts JSON-lines data but requires local input filenames ending in `.json`, rather than `.jsonl`. Materialize bounded samples from large Hub datasets with streaming before launching the sweep, and use distinct training and validation splits.

Managed native logging, same-experiment delivery, and no-work replay were tested with MLflow 3.17.0 and Transformers 5.19.0.dev0 (`536ecc0`), using BART-large and the official NER example with Ettin 68M, Liquid LFM2.5 Encoder 230M, and text-only NeoMME 260M. NER used FP32 parameters and optimizer states with BF16 autocast. Ettin used its native task head; Liquid and NeoMME required local token-classification heads, with NeoMME's document prefix removed before predicting word labels. Repeated validation selection and sentence overlap limit model-quality conclusions.

Trainer-owned Transformers logging and other framework/version combinations remain unverified.

## Select metrics and require evidence

Every MLflow extractor declares its selector explicitly. Selection is independent of `metric.goal`, which ranks the selected scalars across trials.

| Selector | Selection from native metric history |
| --- | --- |
| `last` | Greatest step, then latest timestamp at that step |
| `min` | Lowest value; ties use earliest step then earliest timestamp |
| `max` | Highest value; ties use earliest step then earliest timestamp |
| `step` | Latest timestamp at the required exact `step` |

For example, `{type: mlflow, metric_key: eval/loss, select: step, step: 100}` requires step 100. A `step` field with another selector is invalid. Conflicting values at an identical step and timestamp are rejected. Selected values must be finite; min/max also reject non-finite histories. Model- or dataset-scoped metric histories are outside this contract.

The controller shares one capture across the objective, MLflow constraints, and the phase's MLflow gates. Each distinct metric's complete history serves every selector requesting that key. A presence gate separates namespaces:

```yaml
gates:
  - type: mlflow_required
    metrics: [eval/loss]
    params: [learning_rate]
    tags: [dataset_revision]
```

At least one namespace must be nonempty. Missing evidence fails the trial; values from another run or a PhaseSweep result metric cannot substitute for it. Accepted evidence freezes the selected value, step, timestamp, verified native run identity, and required metadata. Local reads and no-work replay require neither an MLflow import nor server access. See [confound verdicts](runtime.md#confound-verdicts) for evaluation-position checks.

## Result delivery and recovery

Accepted trials receive reserved `phasesweep.*` metrics, parameters, and tags: objective, measured constraints, feasibility, trainer duration, effective fixed/inherited/sampled overrides, and phase/inheritance references. Trainers should leave this namespace to PhaseSweep. A generation-summary run in the same experiment groups managed attempts and holds `phasesweep/generation.json` plus winner references. Each attempt run holds `phasesweep/attempt-<attempt-id>.json`, and a winning run also holds `phasesweep/winner-<phase>.json`. Trainer-owned attempts retain their own parents.

An upload failure after acceptance warns and leaves a pending delivery record. It does not change the accepted score, complete trial, or committed winner. `phasesweep status` exposes path-free `tracking_delivery` counts across every generation, and MCP status counts the queried run's own uploads; scientific terminal snapshots remain frozen. Retry the saved projections:

```bash
phasesweep sync-tracking examples/mlflow.yaml
```

This requires the original target configuration and retries every generation's pending projections; `--generation GENERATION_ID` limits it to one generation, which for an MCP run is its run ID. It does not train, select metrics again, or rewrite scientific evidence. Already delivered projections are skipped. The command exits nonzero if uploads remain pending. Tracking-only worker recovery uses the active-work registry without changing Optuna trials.

After confirmed cleanup of a failed invocation, PhaseSweep queues terminal projections for its generation-summary run and for every managed attempt run whose trial outcome was not yet recorded: FAILED for ordinary failure, or KILLED for shutdown or keyboard interruption. Upload failures leave these projections pending for `sync-tracking`; those runs can remain RUNNING until delivery succeeds. This does not change accepted scores or committed winners. A remote terminal status is never overwritten.

Abrupt controller death, uncertain cleanup, or an unresolved generation binding can still leave no terminal projection. These require operator inspection and, where indicated, [MCP recovery](mcp.md#run-state-and-recovery) before further action. `sync-tracking` retries saved projections; it does not manufacture a missing terminal record.

Bindings and mutable delivery records stay under the experiment artifact root. Delivery records live in `tracking/<generation-id>/delivery/`, separately from committed generation manifests. See [runtime layout](runtime.md#output-layout) for worker scratch and cache placement.

## Databricks status

Databricks targets preserve URI spelling, including profile case, and delegate authentication to the MLflow/Databricks SDK. With narrowed environment inheritance, explicitly list rotating authentication/transport variables in `execution.passthrough_env`; do not put secrets in semantic `env` values. The SDK can also use its normal profile configuration. See [Databricks authentication](https://docs.databricks.com/en/dev-tools/auth/index.html).

Local HTTP behavior is covered by integration tests. Live Databricks access and framework integrations on Databricks remain **unverified**.
