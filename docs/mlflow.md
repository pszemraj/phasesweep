# Native MLflow tracking

PhaseSweep reads metrics that your trainer logs natively to MLflow, selects
winners locally, and mirrors results to the same MLflow experiment. A sweep
uses one tracking backend. W&B-to-MLflow publishing and publishing-only MLflow
configuration are unsupported.

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

In another terminal, provision the experiment explicitly, then run the example
from the repository root:

```bash
python -c 'from mlflow import MlflowClient; MlflowClient(tracking_uri="http://127.0.0.1:5000").create_experiment("phasesweep-native")'
phasesweep validate examples/mlflow.yaml
phasesweep run examples/mlflow.yaml
phasesweep show-winners examples/mlflow.yaml
phasesweep status examples/mlflow.yaml
```

The [complete configuration](../examples/mlflow.yaml) launches four
deterministic CPU trials across two phases. The depth winner is `model.depth=2`
with loss 9; the rate phase inherits that depth and selects `optimizer.rate=3`
with loss 0. The [trainer](../src/phasesweep/examples/mlflow_train.py) uses
`mlflow.log_metric`, `log_params`, and `set_tag` directly.

For real training on a GPU, the [Tiny Decoder MLflow smoke
guide](../examples/tiny_decoder_enwik8/README.md#native-mlflow-gpu-smoke)
provides a complete two-trial, ten-batch configuration using native
final-checkpoint metrics, with both run ownership modes supported.

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

The experiment must already exist and be active. PhaseSweep never creates it
implicitly. `tracking_uri` accepts HTTP(S), `databricks`, and
`databricks://profile`. HTTP credentials belong in the SDK's authentication
configuration or environment rather than the URI. The top-level `mlflow` block
requires an MLflow objective. Local constraint extractors and local gates can
still accompany it, but every remote consumer uses the bound MLflow run.

All controller network operations run in supervised workers outside GPU leases.
The timeout bounds each operation; a phase or whole-run deadline can shorten it.
Trainer cleanup is confirmed before releasing its GPU lease, and launch checks
cancellation and deadlines again after acquiring that lease.

## Choose run ownership

In **managed** mode, PhaseSweep writes the trainer input and creation intent,
creates the native MLflow run, and persists its returned ID before acquiring a
GPU or launching training. The trainer receives `MLFLOW_TRACKING_URI`,
`MLFLOW_EXPERIMENT_ID`, and `MLFLOW_RUN_ID`. Resume that supplied run:

```python
import os
import mlflow

with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

PhaseSweep can finish a still-running managed run after successful trainer exit
and confirmed process cleanup. A FAILED or KILLED run is never converted into
accepted evidence. If a creation response is lost, PhaseSweep searches by exact
generation and attempt identity tags; it does not repeat the creation request.

In **trainer** mode, your existing logger creates and finishes the run. Bind
that exact run once it exists:

```python
import mlflow
from phasesweep.tracking import bind_mlflow_run

with mlflow.start_run() as run:
    bind_mlflow_run(run.info.run_id)
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

The helper verifies the configured experiment, attaches PhaseSweep identity
tags, and writes the local binding. It does not log or replace metrics.
PhaseSweep never discovers runs by name or recency, finishes trainer-owned
runs, or changes their existing parent relationships. The trainer must close
its run successfully before evidence can be accepted. Existing framework
loggers must expose their actual run ID and honor this ownership contract;
framework-specific qualification remains follow-up work.

## Select metrics and require evidence

Every MLflow extractor declares its selector explicitly. Selection is independent
of `metric.goal`, which ranks the selected scalars across trials.

| Selector | Selection from native metric history |
| --- | --- |
| `last` | Greatest step, then latest timestamp at that step |
| `min` | Lowest value; ties use earliest step then earliest timestamp |
| `max` | Highest value; ties use earliest step then earliest timestamp |
| `step` | Latest timestamp at the required exact `step` |

For example, `{type: mlflow, metric_key: eval/loss, select: step, step: 100}`
requires step 100. A `step` field with another selector is invalid. Conflicting
values at an identical step and timestamp are rejected. Selected values must be
finite; min/max also reject non-finite histories. Model- or dataset-scoped
metric histories are outside this contract.

The controller shares one capture across the objective, MLflow constraints, and
the phase's MLflow gates. Each distinct metric's complete history serves every
selector requesting that key. A presence gate separates namespaces:

```yaml
gates:
  - type: mlflow_required
    metrics: [eval/loss]
    params: [learning_rate]
    tags: [dataset_revision]
```

At least one namespace must be nonempty. Missing evidence fails the trial;
values from another run or a PhaseSweep result metric cannot substitute for it.
Accepted evidence freezes the selected value, step, timestamp, verified native
run identity, and required metadata. Local reads and no-work replay require
neither an MLflow import nor server access. Confound reporting uses the selected
native step as its evaluation position.

## Result delivery and recovery

Accepted trials receive reserved `phasesweep.*` metrics, parameters, and tags:
objective, measured constraints, feasibility, trainer duration, effective
fixed/inherited/sampled overrides, and phase/inheritance references. Trainers
should leave this namespace to PhaseSweep. A generation-summary run in the same
experiment groups managed attempts and holds `phasesweep/summary.json` plus
winner references. Trainer-owned attempts retain their own parents.

An upload failure after acceptance warns and leaves a pending delivery record.
It does not change the accepted score, complete trial, or committed winner.
`phasesweep status` and current MCP status expose path-free `tracking_delivery`
counts; scientific terminal snapshots remain frozen. Retry the saved projections:

```bash
phasesweep sync-tracking examples/mlflow.yaml --generation GENERATION_ID
```

This requires the original target configuration. It does not train, select
metrics again, or rewrite scientific evidence. Already delivered projections
are skipped. The command exits nonzero if uploads remain pending. Tracking-only
worker recovery uses the active-work registry without changing Optuna trials.

Bindings, creation intents, and mutable delivery records stay under the
experiment artifact root. Delivery records live in
`tracking/<generation-id>/delivery/`, separately from committed generation
manifests. Disposable worker scratch lives in
`~/.cache/phasesweep/mlflow/workers/` and is removed after each operation.
`XDG_CACHE_HOME` can relocate that cache; the default uses `Path.home()`.
Host lock placement retains its existing OS-account-home contract.

## Databricks status

Databricks targets preserve URI spelling, including profile case, and delegate
authentication to the MLflow/Databricks SDK. With narrowed environment
inheritance, explicitly list rotating authentication/transport variables in
`execution.passthrough_env`; do not put secrets in semantic `env` values.
The SDK can also use its normal profile configuration. See
[Databricks authentication](https://docs.databricks.com/en/dev-tools/auth/index.html).

Local HTTP behavior is qualified by the integration tests. Live Databricks
access and framework-specific logger behavior remain **unverified** until
workplace testing; no workplace credentials or endpoints are required locally.
