# Native MLflow tracking

PhaseSweep reads metrics that your trainer logs natively to MLflow, selects winners locally, and mirrors results to the same MLflow experiment. A sweep uses one tracking backend. W&B-to-MLflow publishing and publishing-only MLflow configuration are unsupported.

## Run the CPU example locally

Install the optional client and, for a local server, matching full MLflow:

```bash
pip install "phasesweep[mlflow]" "mlflow==3.17.0"
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

The experiment must already exist and be active. PhaseSweep never creates it implicitly. Quote a numeric `experiment_id` so YAML keeps it a string. Before each phase allocates any trial, it resolves the experiment, and the first phase also creates the generation-summary run; if the server, credentials, or experiment are not ready, the run stops there without using any trial slots. The refusal names the configured experiment name or ID and tracking URI. The client retries a server that does not answer, so that refusal comes once `timeout_seconds` runs out. A target lost after a phase starts fails that phase's remaining trials, which count toward its `n_trials`. `tracking_uri` accepts HTTP(S), `databricks`, and `databricks://profile`. HTTP credentials belong in the SDK's authentication configuration or environment rather than the URI. The top-level `mlflow` block requires an MLflow objective. Local constraint extractors and local gates can still accompany it, but every remote consumer uses the bound MLflow run.

A refused server connection is reported as a connection refusal, including when MLflow wraps the transport failure. Check `tracking_uri` and that the server is running.

All controller network operations run in supervised workers outside GPU leases. The timeout bounds each operation. A phase or whole-run deadline also caps experiment resolution, run creation and evidence capture; result uploads happen after acceptance and, like other publication, can extend past those deadlines. Within that bound the MLflow client retries transient HTTP failures. HTTP(S) run creation uses [identity reconciliation](#choose-run-ownership) after an unconfirmed response. The Databricks SDK keeps its own retries for creation too, so a rare lost response there can leave an extra, unbound run in the experiment. Trainer cleanup is confirmed before releasing its GPU lease, and launch checks cancellation and deadlines again after acquiring that lease.

## Choose run ownership

In **managed** mode, PhaseSweep writes the trainer input, creates the native MLflow run named after the trial's `PHASESWEEP_RUN_NAME`, and persists its returned ID before acquiring a GPU or launching training. The trainer receives `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and `MLFLOW_RUN_ID`. Resume that supplied run:

```python
import os
import mlflow

with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

PhaseSweep finishes a still-running managed run after confirmed process cleanup: FINISHED after successful trainer exit, FAILED after a failed trial, or KILLED after an [interrupted](runtime.md#interrupted-trials-rerun) one. A rerun of an interrupted trial gets its own managed run. A run that is already terminal keeps its status, and its `phasesweep.outcome` tag records a later trial failure. A FAILED or KILLED run is never converted into accepted evidence. Every creation request first searches the exact generation and attempt identity tags and binds a run it finds, and it creates one only when the search finds none. A creation that may have been sent but was not confirmed is never sent again: a durable pending record keeps it unresolved, across controller restarts, until an identity search finds the run, which the next reconciliation or `sync-tracking` then binds.

In **trainer** mode, your existing logger creates and finishes the run. The trainer receives `MLFLOW_TRACKING_URI` and the resolved `MLFLOW_EXPERIMENT_ID`, so a plain `mlflow.start_run()` opens it in the configured experiment. Bind that exact run once it exists:

```python
import mlflow
from phasesweep.tracking import bind_mlflow_run

with mlflow.start_run() as run:
    bind_mlflow_run(run.info.run_id)
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

The helper verifies the configured experiment and that the run is still RUNNING, attaches PhaseSweep identity tags, and writes the local binding. It does not log or replace metrics. PhaseSweep never discovers runs by name or recency, finishes trainer-owned runs, or changes their existing parent relationships. The trainer must close its run successfully before evidence can be accepted. Existing framework loggers must expose their actual run ID and honor this ownership contract; other logger integrations require their own qualification.

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

With `load_best_model_at_end=True`, the logged step does not identify the evaluated checkpoint. Use `json_envelope` when that identity matters; see [evaluation positions](config.md#objective-evidence-constraints-and-gates).

## Select metrics and require evidence

Every MLflow extractor declares its selector explicitly. Selection is independent of `metric.goal`, which ranks the selected scalars across trials.

| Selector | Selection from native metric history |
| --- | --- |
| `last` | Greatest step, then latest timestamp at that step |
| `min` | Lowest value; ties use earliest step then earliest timestamp |
| `max` | Highest value; ties use earliest step then earliest timestamp |
| `step` | Latest timestamp at the required exact `step` |

For example, `{type: mlflow, metric_key: eval/loss, select: step, step: 100}` requires step 100. If it is absent, the error names the metric key, requested step, and available steps in its history. A `step` field with another selector is invalid. Conflicting values at an identical step and timestamp are rejected. Selected values must be finite; min/max also reject non-finite histories. Model- or dataset-scoped metric histories are outside this contract.

The controller shares one capture across the objective, MLflow constraints, and the phase's MLflow gates. Each distinct metric's complete history serves every selector requesting that key. A presence gate separates namespaces:

```yaml
gates:
  - type: mlflow_required
    metrics: [eval/loss]
    params: [learning_rate]
    tags: [dataset_revision]
```

At least one namespace must be nonempty. Missing evidence fails the trial; values from another run or a PhaseSweep result metric cannot substitute for it. Accepted evidence freezes the selected value, step, timestamp, verified native run identity, and required metadata. Local reads and no-work replay require neither an MLflow import nor server access. A no-work replay publishes a new local scientific generation without creating tracking generation directories or uploading to MLflow. See [confound verdicts](runtime.md#confound-verdicts) for evaluation-position checks.

## Result delivery and recovery

Accepted trials receive reserved `phasesweep.*` metrics, parameters, and tags: objective, measured constraints, report-only `observe` values (`phasesweep.observed.<name>`), feasibility, trainer duration, effective fixed/inherited/sampled overrides, and phase/inheritance references. Trainers should leave this namespace to PhaseSweep. MLflow truncates a parameter value longer than 6000 characters, so the attempt's summary artifact below is the full record of its overrides; it rejects a parameter key longer than 250 characters, which bounds an override key to 230 after the `phasesweep.override.` prefix. A generation-summary run in the same experiment groups managed attempts and holds `phasesweep/generation.json` plus winner references. Each attempt run holds `phasesweep/attempt-<attempt-id>.json`, and a winning run also holds `phasesweep/winner-<phase>.json`. For a phase with [`repeats`](config.md#repeats), the winning run is the last member of the winning group, and its winner reference records the group's reduced metric, while every member's run keeps its own objective. Trainer-owned attempts retain their own parents.

An upload failure after acceptance warns and leaves a pending delivery record. It does not change the accepted score, complete trial, or committed winner. Once a generation is published, no MLflow failure fails the run: winner references that could not be saved or sent wait for `sync-tracking`, and an upload worker that may still be running makes the successful run report unconfirmed cleanup. `phasesweep status` exposes path-free `tracking_delivery` counts across every generation, and MCP status counts the queried run's own uploads; scientific terminal snapshots remain frozen. Retry the saved projections:

```bash
phasesweep sync-tracking examples/mlflow.yaml
```

This requires the original target configuration and retries every generation's pending projections; `--generation GENERATION_ID` limits it to one generation, which for an MCP run is its run ID. It does not train, select metrics again, or rewrite scientific evidence. Already delivered projections are skipped. A projection that was never saved, because a local write failed or the controller died, is rebuilt: an attempt's from its settled trial in the Optuna ledger, and a published generation's winner references from its committed artifacts. The command exits nonzero if uploads remain pending. An upload MLflow rejects, such as an over-long parameter key or a run deleted from MLflow, fails the same way on every retry and stays pending. Tracking-only worker recovery uses the active-work registry without changing Optuna trials, and it records a confirmed worker cleanup as a receipt under `tracking/<generation-id>/cleanup-receipts/` before retiring the registry entry; `phasesweep mcp recover-run` accepts that receipt as cleanup evidence for that generation's run, or for a run whose runner reported the worker. A damaged receipt is ignored with a warning, so recovery then refuses for lack of evidence.

After confirmed cleanup of a failed invocation, PhaseSweep queues projections for its generation-summary run and for every attempt whose outcome was not yet recorded. The attempt's Optuna trial decides its projection: an accepted trial gets its accepted projection, and any other managed run is finished FAILED for ordinary failure, or KILLED for an interrupted trial or a shutdown or keyboard interruption. Upload failures leave these projections pending for `sync-tracking`; those runs can remain RUNNING until delivery succeeds. This does not change accepted scores or committed winners. A remote terminal status is never overwritten.

Abrupt controller death, uncertain cleanup, or an unresolved generation binding can still leave no terminal projection. These require operator inspection and, where indicated, [MCP recovery](mcp.md#run-state-and-recovery) before further action. `sync-tracking` rebuilds an attempt's projection only from an accepted trial, or from a failed or interrupted trial whose process cleanup is confirmed, and leaves a run whose trial Optuna still records as running, or whose cleanup is still unconfirmed, untouched. Once `recover-run` or the next run's preflight confirms that cleanup in the study's recovery ledger, the next `sync-tracking` delivers the projection; the trial's own record is never rewritten. `sync-tracking` does not manufacture a failed invocation's missing generation-summary record.

Bindings and mutable delivery records stay under the experiment artifact root. Delivery records live in `tracking/<generation-id>/delivery/`, separately from committed generation manifests. See [runtime layout](runtime.md#output-layout) for worker scratch and cache placement.

## Databricks status

Databricks targets preserve URI spelling, including profile case, and delegate authentication to the MLflow/Databricks SDK, which reads `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, OAuth client variables, or a `~/.databrickscfg` profile. Exported `DATABRICKS_*` variables take precedence over the profile named in `databricks://profile`, and some MLflow artifact uploads authenticate without that profile. Prefer `tracking_uri: databricks` with `env: {DATABRICKS_CONFIG_PROFILE: <profile>}`, or with host and token variables. Experiment names are workspace paths such as `/Users/you@example.com/sweep`.

In every `execution.inherit_env` mode, list rotating authentication and transport variables, such as `DATABRICKS_TOKEN`, `DATABRICKS_CLIENT_SECRET`, or `HTTPS_PROXY`, in `execution.passthrough_env`. This keeps them out of the study's recorded environment cohort; the optional `execution.record_env` file still includes them. See the [trainer environment contract](config.md#trainer-contract) and [Databricks authentication](https://docs.databricks.com/en/dev-tools/auth/index.html).

Local HTTP behavior is covered by integration tests. Live Databricks access and framework integrations on Databricks remain **unverified**.
