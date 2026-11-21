# Native MLflow tracking

PhaseSweep reads metrics that your trainer logs natively to MLflow, selects winners locally, and mirrors results to the same MLflow experiment. A sweep uses one tracking backend.

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

In another terminal, generate the packaged MLflow template, provision its experiment and run it. Its trainer ships with PhaseSweep, so no repository checkout is needed:

```bash
phasesweep init --template mlflow
python -c 'from mlflow import MlflowClient; MlflowClient(tracking_uri="http://127.0.0.1:5000").create_experiment("phasesweep-native")'
phasesweep validate mlflow.yaml
phasesweep run mlflow.yaml
phasesweep show-winners mlflow.yaml
phasesweep status mlflow.yaml
```

When adapting the template to your trainer, replace its command, `trainer_config` and objective metric key, and remove or replace the demo `cost` constraint and `mlflow_required` gates.

This configuration launches four deterministic CPU trials across two phases. The depth winner is `model.depth=2` with loss 9; the rate phase inherits that depth and selects `optimizer.rate=3` with loss 0. The [trainer](../src/phasesweep/examples/mlflow_train.py) uses `mlflow.log_metric`, `log_params`, and `set_tag` directly.

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

The experiment must already exist and be active. Quote a numeric `experiment_id` so YAML keeps it a string. `tracking_uri` accepts HTTP(S), `databricks`, and `databricks://profile`; HTTP credentials belong in SDK authentication or the environment rather than the URI. Before a phase allocates its first trial, PhaseSweep resolves the experiment; the first such phase also creates the generation-summary run. A server, credential or experiment failure spends none of that phase's trial slots; when no trial has launched, it writes no generation. The refusal names the target. The resolved ID is saved in `<workdir>/<experiment>/tracking/source.json`. Recreating the experiment under a different ID breaks that binding.

The MLflow client retries a server that refuses or does not answer. When those retries outlast `timeout_seconds`, the refusal says the target did not answer within `timeout_seconds`; when they end sooner, it says the server refused the connection, including when MLflow wraps the transport failure. Either way, check `tracking_uri` and that the server is running.

`tracking_uri`, `experiment_id`, `experiment_name` and `run_mode` are in the study fingerprint, so changing one needs a new experiment name; a retry reads the original attempt with the current `poll_seconds` and `timeout_seconds`. `phasesweep validate` and `run --dry-run` check neither authentication nor the experiment. The credentials need write access: PhaseSweep creates runs, logs metrics, params and tags, uploads artifacts and ends runs. An `env` entry for `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME` must equal the block's own normalized value (lowercase scheme and host, no trailing slash), so the selector the block leaves unset cannot appear, and `env` may not set `MLFLOW_RUN_ID`; config load fails otherwise.

Controller tracking workers run outside GPU leases. `timeout_seconds` bounds SDK retries. The remaining phase or run deadline also caps experiment resolution, run creation and evidence capture. Uploads can continue after that deadline. Once a phase starts, a preparation failure spends a trial and counts toward `max_consecutive_failures`. An expired phase or run deadline interrupts the trial. If MLflow cannot answer after training, [trial recovery](runtime.md#trial-execution-and-evidence) determines whether the finished result can be kept.

## Choose run ownership

In **managed** mode, PhaseSweep writes the trainer input, creates the native MLflow run named after the trial's `PHASESWEEP_RUN_NAME`, and persists its returned ID before acquiring a GPU or launching training. Creation and lookup use registered tracking workers. If the controller stops before training starts, the next run retries the same draw. The trainer receives `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and `MLFLOW_RUN_ID`. Its first `mlflow.start_run()` or `mlflow.log_*` call resumes that supplied run. Log every metric, including the final objective, into that one run. MLflow consumes the environment ID when it resumes and removes it on `end_run()`: if the trainer must close and reopen the run, first save `mlflow.active_run().info.run_id` and pass that saved ID to `mlflow.start_run(run_id=...)`. Do not call `mlflow.set_experiment()` with another name. A trainer that uses the native MLflow API needs no PhaseSweep import:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

Lightning's `MLFlowLogger` needs the supplied run ID passed as its `run_id`; read `MLFLOW_RUN_ID` before any MLflow call, or use `mlflow.active_run().info.run_id` while the run is active.

Transformers' MLflow callback ends its run at the end of training, so call `mlflow.start_run()` before constructing the Trainer to keep its final evaluation in the same active run.

After confirmed cleanup, PhaseSweep marks a failed trial's managed run FAILED even if the trainer left it FINISHED. A still-running managed run becomes FINISHED after success. A SIGINT leaves the managed run RUNNING until `phasesweep sync-tracking CONFIG` delivers KILLED for the [interrupted attempt](runtime.md#interrupted-trials-rerun). Interrupted reruns receive new managed runs. FAILED and KILLED runs cannot supply accepted evidence.

In **trainer** mode, your existing logger creates and finishes the run. The trainer receives `MLFLOW_TRACKING_URI` and the resolved `MLFLOW_EXPERIMENT_ID`, so a plain `mlflow.start_run()` opens it in the configured experiment. Bind that exact run once it exists:

```python
import mlflow
from phasesweep.tracking import bind_mlflow_run

with mlflow.start_run() as run:
    bind_mlflow_run(run.info.run_id)
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

The helper verifies the configured experiment and that the run is still RUNNING, attaches PhaseSweep identity tags, and writes the local binding. It does not log or replace metrics. PhaseSweep never discovers runs by name or recency, finishes trainer-owned runs, or changes their existing parent relationships. The trainer must close its run successfully before evidence can be accepted. A missing, foreign or conflicting binding fails the trial, and so does a bound run that was deleted, lies outside the configured experiment or carries another attempt's identity tags. Existing framework loggers must expose their actual run ID and honor this ownership contract. In either mode, disable autologging and nested-run behavior that starts runs of its own: PhaseSweep reads only the bound run, never a parent or child.

Native metrics retain their Trainer names, such as `eval_loss` and `eval_rouge1`; choose the exact final optimizer step when comparing fixed training budgets.

## Select metrics and require evidence

Every MLflow extractor declares its selector explicitly. Selection is independent of `metric.goal`, which ranks the selected scalars across trials.

With [repeats](config.md#repeats), each member has its own trial and bound MLflow run. Several native runs created inside one trial cannot supply a repeat group.

| Selector | Selection from native metric history |
| --- | --- |
| `last` | Greatest step, then latest timestamp at that step |
| `min` | Lowest value; ties use earliest step then earliest timestamp |
| `max` | Highest value; ties use earliest step then earliest timestamp |
| `step` | Latest timestamp at the required exact `step` |

For example, `{type: mlflow, metric_key: eval/loss, select: step, step: 100}` requires step 100. If it is absent, the error names the metric key, requested step, and available steps in its history. A `step` field with another selector is invalid. Conflicting values at an identical step and timestamp are rejected. Selected values must be finite; min/max also reject non-finite histories. A metric history logged with a model or dataset scope fails the trial.

The controller shares one capture across the objective, MLflow constraints, and the phase's MLflow gates. Each distinct metric's complete history serves every selector requesting that key. A presence gate separates namespaces:

```yaml
gates:
  - type: mlflow_required
    metrics: [eval/loss]
    params: [learning_rate]
    tags: [dataset_revision]
```

At least one namespace must be nonempty. A finished run missing a required metric, param or tag fails at once, naming the missing and available keys, for example `MLflow run finished without metric 'loss'; available metric keys: ['accuracy', 'eval_loss'].` When a managed run lacks the objective metric, the first such trial in a phase also logs one hint on logging into the supplied run; the stored failure reason carries the diagnostic only. An unfinished run is polled every `poll_seconds` until timeout, which reports its last state; a trainer-owned run left open fails this way. The capture budget includes worker startup and stops polling `min(5, timeout_seconds / 2)` seconds before expiry to retain that diagnostic. Accepted evidence freezes the selected value, step, timestamp, native run identity and required metadata. Local reads and no-work replay need no SDK or server access; a replay that [publishes a generation](runtime.md#timeouts-cleanup-and-continuation) creates no tracking directories or uploads. See [confound verdicts](runtime.md#confound-verdicts) for evaluation-position checks.

## Result delivery and recovery

Accepted trials receive reserved `phasesweep.*` metrics, parameters, and tags: objective, measured constraints, report-only `observe` values (`phasesweep.observed.<name>`), feasibility, trainer duration, effective fixed/inherited/sampled overrides, and phase/inheritance references. Trainers should leave this namespace to PhaseSweep. The `phasesweep.*` metrics are logged at step 0. MLflow truncates a parameter value longer than 6000 characters, so the attempt's summary artifact below is the full record of its overrides; with the `mlflow` block set, config load refuses constraint names, observation names and override keys whose published key contains characters outside alphanumerics, underscores, periods, dashes, slashes, colons and spaces, exceeds 250 characters, or is not a normalized relative path. The `phasesweep.override.` prefix bounds an override key to 230 characters. A generation-summary run in the same experiment groups managed attempts and holds `phasesweep/generation.json` plus winner references. Each attempt run holds `phasesweep/attempt-<attempt-id>.json`, and a winning run also holds `phasesweep/winner-<phase>.json`. For a phase with [`repeats`](config.md#repeats), the winning run is the last member of the winning group, and its winner reference records the group's reduced metric, while every member's run keeps its own objective. Trainer-owned attempts retain their own parents.

After confirmed cleanup, a failed invocation queues a FAILED generation summary. Shutdown or keyboard interruption queues KILLED. Accepted trials keep their measured projections. Interrupted managed attempts receive KILLED, and other terminal unaccepted managed attempts receive the invocation's status.

With persistent storage, a managed attempt whose trial is still `RUNNING` stays open until recovery settles it. With `storage: null`, confirmed cleanup allows the invocation to close its remaining managed runs. If the controller died outright, the next run checks the attempts' cleanup before closing them. Until then, `sync-tracking` can close only the generation-summary run.

The next normal run or a confirmed `phasesweep mcp recover-run` can recover trial evidence from MLflow. Once the target is reachable, it delivers recovered outcomes, including the generation summary. A normal run that launched no work resolves the target before delivery. `recover-run` reports delivered and pending counts. An unreachable target, an interrupted delivery or a record that could not be saved leaves work for `sync-tracking`. Until delivery succeeds, MLflow can still show a run as RUNNING.

Result uploads leave pending records on failure without changing accepted scores, trial states or winners. Once published, the local result stays successful even when a winner reference could not be saved or sent, or upload-worker cleanup is uncertain. A `PENDING` closeout names the generation, upload count and retry command. CLI `status` reports `tracking_delivery` counts across generations; MCP status reports its run's uploads separately from its frozen scientific result:

```bash
phasesweep sync-tracking mlflow.yaml
```

`sync-tracking` uses the original target configuration to reconcile pending run creation, rebuild missing delivery records and retry uploads. `--generation GENERATION_ID` limits retries to one generation, whose ID is the MCP run ID for detached runs. `sync-tracking` holds the experiment lock, refuses a tree whose binding or ledger format a run would refuse before writing anything, skips delivered projections, and exits nonzero while uploads remain pending, when no delivery records exist in scope, or without an `mlflow` block. Permanent remote refusals, such as a deleted run, remain pending.

Run creation searches exact generation/attempt identity tags before creating anything. An unconfirmed creation request is not sent again: its pending record survives restarts, and reconciliation or `sync-tracking` binds the run if it appears. HTTP(S) uses this identity reconciliation; Databricks retains its SDK creation retries, so a lost response there can leave an extra unbound run.

Missing projections are rebuilt from settled trials and committed winner artifacts. Only the target recorded in `source.json` can receive them. A config naming another target leaves the earlier records untouched and logs a warning. A trial that is still running or awaiting cleanup needs [operator recovery](mcp.md#run-state-and-recovery) or the next run's preflight.

If an invocation ended before saving its generation summary, the next run, `recover-run` for that run, or `sync-tracking` can rebuild it. A published generation supplies its committed artifacts. For an unpublished generation, the lifecycle record supplies FAILED or KILLED. An absent terminal record means KILLED. An unreadable or malformed record must be restored before the outcome can be delivered. An already saved summary is reused.

Tracking-worker recovery writes a receipt under `tracking/<generation-id>/cleanup-receipts/` before retiring the registry entry. The generation is the one that started the worker or, for `sync-tracking`, the one whose record it handled. `recover-run` accepts a readable receipt for that generation or for a runner that reported the worker.

## Databricks status

Databricks targets preserve URI spelling, including profile case, and delegate authentication to the MLflow/Databricks SDK, which reads `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, OAuth client variables, or a `~/.databrickscfg` profile. Exported `DATABRICKS_*` variables take precedence over the profile named in `databricks://profile`, and some MLflow artifact uploads authenticate without that profile. Prefer `tracking_uri: databricks` with `env: {DATABRICKS_CONFIG_PROFILE: <profile>}`, or with host and token variables. Experiment names are workspace paths such as `/Users/you@example.com/sweep`.

In every `execution.inherit_env` mode, list rotating authentication and transport variables, such as `DATABRICKS_TOKEN`, `DATABRICKS_CLIENT_SECRET`, or `HTTPS_PROXY`, in `execution.passthrough_env`; see the [trainer environment contract](config.md#trainer-contract) and [Databricks authentication](https://docs.databricks.com/en/dev-tools/auth/index.html).

PhaseSweep is not tested against a live Databricks workspace or with framework integrations running on Databricks.
