# Native MLflow tracking

PhaseSweep reads metrics that your trainer logs natively to MLflow, selects winners locally, and mirrors results to the same MLflow experiment. A sweep uses one tracking backend.

## Run the CPU example locally

Install the optional client and, for a local server, matching full MLflow. PhaseSweep supports MLflow 3.17 up to but not including 4:

```bash
pip install "phasesweep[mlflow]" "mlflow>=3.17,<4"
mkdir -p /tmp/phasesweep-mlflow-demo/artifacts
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////tmp/phasesweep-mlflow-demo/mlflow.db \
  --serve-artifacts \
  --artifacts-destination /tmp/phasesweep-mlflow-demo/artifacts
```

In another terminal, generate the packaged MLflow template, provision its experiment and run it. Its trainer ships with PhaseSweep, so no repository checkout is needed. If the server already has the experiment, reuse it.

```bash
phasesweep init --template mlflow
python -c 'from mlflow import MlflowClient; c = MlflowClient(tracking_uri="http://127.0.0.1:5000"); c.get_experiment_by_name("phasesweep-native") or c.create_experiment("phasesweep-native")'
phasesweep validate mlflow.yaml
phasesweep run mlflow.yaml
phasesweep show-winners mlflow.yaml
phasesweep status mlflow.yaml
```

When adapting the template to your trainer, replace its command, `trainer_config` and objective metric key, and remove or replace the demo `cost` constraint and `mlflow_required` gates. For a GPU trainer, remove both `gpu_policy: none` entries to enable [device leasing](runtime.md#gpu-isolation).

The demo runs four CPU trials across two phases without reserving a GPU. The depth winner is `model.depth=2` with loss 9. The rate phase inherits that depth and selects `optimizer.rate=3` with loss 0. The [trainer](../src/phasesweep/examples/mlflow_train.py) uses `mlflow.log_metric`, `log_params`, and `set_tag` directly.

For real training on a GPU, see the [Tiny Decoder MLflow smoke](../examples/tiny_decoder_enwik8/README.md#native-mlflow-gpu-smoke).

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

The `mlflow` block requires `metric.extractor.type: mlflow`. If the controller and trainer use separate environments, install `phasesweep[mlflow]` for the controller and make the native `mlflow` API available to the trainer. A trainer that calls `bind_mlflow_run` also needs PhaseSweep installed.

The experiment must already exist and be active. Quote a numeric `experiment_id` so YAML keeps it a string. `tracking_uri` accepts HTTP(S), `databricks`, and `databricks://profile`. Supply HTTP credentials through the SDK's authentication environment. Before a phase allocates its first trial, PhaseSweep resolves the experiment. The first such phase also creates the generation-summary run; a run that publishes without launching, such as a reread after a reading change, has its summary run created when it delivers its results. A failed target check names the target and spends none of that phase's trial slots. If it fails before the run's first trial, PhaseSweep writes no generation. The resolved ID is saved in `<workdir>/<experiment>/tracking/source.json`. PhaseSweep refuses to continue if the experiment resolves to a different ID.

The MLflow client retries a server that refuses or does not answer. When those retries outlast `timeout_seconds`, the refusal says the target did not answer within `timeout_seconds`; when they end sooner, it names the refused connection, including when MLflow wraps the transport failure. Either way, the `fix:` step is to start the server or correct `tracking_uri`, then run again. A refusal MLflow answered, such as for a missing experiment, gets a configuration fix instead.

`tracking_uri`, `experiment_id`, `experiment_name` and `run_mode` are in the study fingerprint. Use a fresh PhaseSweep study when changing one, for example by changing the top-level `experiment` with `storage: auto`. A retry reads the original attempt with the current `poll_seconds` and `timeout_seconds`. `phasesweep validate` and `run --dry-run` check neither authentication nor the experiment. The credentials need write access to create runs, log metrics, params and tags, upload artifacts and end runs.

An `env` entry for `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME` must match the block's configured value exactly. For HTTP(S), use the normalized URI with lowercase scheme and host and no trailing slash. Omit the selector that the `mlflow` block leaves unset, and leave `MLFLOW_RUN_ID` to PhaseSweep.

Controller tracking workers run outside GPU leases. `timeout_seconds` bounds each operation, including SDK retries. An uncertain managed run creation can use a second operation to look for the run. The remaining phase or run deadline also caps experiment resolution, run creation and evidence capture. The run deadline caps the pass that sends earlier runs' records before the first trial, and the phase's own budget starts once that pass ends. Other uploads can continue after the deadline.

After allocation, a run preparation error normally spends a trial and counts toward `max_consecutive_failures`. An expired phase or run deadline interrupts the draw. If a setup worker's cleanup is uncertain, PhaseSweep stops launching work. With persistent storage, the next run can retry the unlaunched draw once cleanup is confirmed. If MLflow cannot answer after training, [trial recovery](runtime.md#trial-execution-and-evidence) determines whether the finished result can be kept.

## Choose run ownership

In **managed** mode, PhaseSweep writes the trainer input, creates the native MLflow run named after the trial's `PHASESWEEP_RUN_NAME`, and persists its returned ID before acquiring a GPU or launching training. The trainer receives `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and `MLFLOW_RUN_ID`. Its first `mlflow.start_run()` or `mlflow.log_*` call resumes that supplied run. Log the attempt's metrics, including the final objective, into that one run. For distributed logging, see the [multi-process trainer contract](config.md#multi-process-and-multi-gpu-trainers). MLflow consumes the environment ID when it resumes and removes it on `end_run()`, so if the trainer must close and reopen the run, first save `mlflow.active_run().info.run_id` and pass that saved ID to `mlflow.start_run(run_id=...)`. Do not call `mlflow.set_experiment()` with another name. A trainer that uses the native MLflow API needs no PhaseSweep import:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

Lightning's `MLFlowLogger` needs the supplied run ID passed as its `run_id`; read `MLFLOW_RUN_ID` before any MLflow call, or use `mlflow.active_run().info.run_id` while the run is active.

Transformers' MLflow callback ends its run at the end of training, so call `mlflow.start_run()` before constructing the Trainer to keep its final evaluation in the same active run.

Each [rerun](runtime.md#interrupted-trials-rerun) gets a new managed run. It keeps the display name when it reuses the training directory. PhaseSweep finishes a managed run that a successful trainer left open. A trainer-reported FAILED or KILLED status fails capture for that attempt. If PhaseSweep closed a managed run after a reading failure, correcting the reader recovers its metrics, and publishing that reading finishes the run and removes the `phasesweep.outcome` and `phasesweep.status_before_delivery` tags. Earlier attempts can contribute history to a later attempt in either case. See [result delivery and recovery](#result-delivery-and-recovery) for how PhaseSweep closes unaccepted attempts.

In **trainer** mode, your existing logger creates and finishes the run. The trainer receives `MLFLOW_TRACKING_URI` and the resolved `MLFLOW_EXPERIMENT_ID`, so a plain `mlflow.start_run()` opens it in the configured experiment. Bind that exact run once it exists:

```python
import mlflow
from phasesweep.tracking import bind_mlflow_run

with mlflow.start_run() as run:
    bind_mlflow_run(run.info.run_id)
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

The helper verifies the configured experiment and that the run is still RUNNING, attaches PhaseSweep identity tags, and writes the local binding. It does not log or replace metrics. PhaseSweep never discovers runs by name or recency, finishes trainer-owned runs, or changes their existing parent relationships. The trainer must close its run successfully before evidence can be accepted. A missing, foreign or conflicting binding fails the trial, and so does a bound run that was deleted, lies outside the configured experiment or carries another attempt's identity tags. Existing framework loggers must expose their actual run ID and honor this ownership contract. In either mode, disable autologging and nested-run behavior that starts runs of its own: PhaseSweep reads only the bound run, never a parent or child.

## Select metrics and require evidence

Every MLflow extractor declares its selector explicitly. Selection is independent of `metric.goal`, which ranks the selected scalars across trials.

With [repeats](config.md#repeats), each member has its own trial and bound MLflow run. Several native runs created inside one trial cannot supply a repeat group.

| Selector | Selection from native metric history |
| --- | --- |
| `last` | Greatest step, then latest timestamp at that step |
| `min` | Lowest value; ties use earliest step then earliest timestamp |
| `max` | Highest value; ties use earliest step then earliest timestamp |
| `step` | Latest timestamp at the required exact `step` |

Set `metric_key` to the trainer's native name, such as `eval_loss` or `eval_rouge1`. Use the exact final optimizer step when comparing fixed training budgets. For example, `{type: mlflow, metric_key: eval/loss, select: step, step: 100}` requires step 100. If it is absent, the error names the metric key, requested step, and available steps in its history. A `step` field with another selector is invalid. Conflicting values at an identical step and timestamp are rejected. Selected values must be finite. `min` and `max` also reject non-finite histories. A metric history logged with a model or dataset scope fails the trial.

The controller shares one capture across the objective, MLflow constraints, and the phase's MLflow gates. Each distinct metric's complete history serves every selector requesting that key. A presence gate separates namespaces:

```yaml
gates:
  - type: mlflow_required
    metrics: [eval/loss]
    params: [learning_rate]
    tags: [dataset_revision]
```

At least one namespace must be nonempty. The finished-run check fails at once when required evidence is missing and names the missing and available keys, for example `MLflow run finished without metric 'loss'; available metric keys: ['accuracy', 'eval_loss'].` When a managed run lacks the objective metric, the first such trial in a phase also logs one hint on logging into the supplied run. The stored failure reason carries the diagnostic only. An unfinished run is polled every `poll_seconds` until timeout, which reports its last state. A trainer-owned run left open fails this way. The capture budget includes worker startup and stops polling `min(5, timeout_seconds / 2)` seconds before expiry to retain that diagnostic. Accepted evidence freezes the selected value, step, timestamp, native run identity and required metadata. See [local reads of retained evidence](runtime.md#trial-execution-and-evidence) and [evaluation-position checks](runtime.md#confound-verdicts).

On a rerun, each selector uses the retained prefix of earlier attempts' history and the retry's own points. For each metric, the retry replaces earlier history at and above its lowest logged step, even when it stops sooner. The retry must log the objective and gate evidence. A constraint can use a metric the retry omits, such as a parameter count logged once at step 0. With sparse evaluations, the retained prefix can include earlier points between the loaded checkpoint and the retry's lowest logged step. See [rerun evidence](invariants.md#33-a-trials-rerun-continues-in-its-directory-alone-and-is-scored-only-from-files-its-own-attempt-wrote).

## Result delivery and recovery

Accepted trials receive reserved `phasesweep.*` metrics, parameters, and tags for the objective, measured constraints, [report-only observations](config.md#report-only-observations), feasibility, trainer duration, effective fixed/inherited/sampled overrides, and phase/inheritance references. Trainers should leave this namespace to PhaseSweep. The `phasesweep.*` metrics are logged at step 0. MLflow truncates a parameter value longer than 6000 characters, so the attempt's summary artifact holds its full overrides. With the `mlflow` block set, config load refuses constraint names, observation names and override keys whose published key contains characters outside alphanumerics, underscores, periods, dashes, slashes, colons and spaces, exceeds 250 characters, or is not a normalized relative path. The `phasesweep.override.` prefix bounds an override key to 230 characters. A generation-summary run in the same experiment groups managed attempts and holds `phasesweep/generation.json` plus winner references. Each attempt run holds `phasesweep/attempt-<attempt-id>.json`, and a winning run also holds `phasesweep/winner-<phase>.json`. Every published generation sends its own reading, a reread or a sibling study's adopted attempts included, so an attempt run's `phasesweep.objective` is the objective the latest published reading gave it, its metric history keeps the earlier readings, and a winning run carries a `phasesweep.winner_generation.<generation>` tag for each generation it won, each with its own summary run. For a phase with [`repeats`](config.md#repeats), the winning run is the last member of the winning group, and its winner reference records the group's reduced metric, while every member's run keeps its own objective.

After confirmed cleanup, a failed invocation queues a FAILED generation summary. Shutdown or keyboard interruption queues KILLED. Accepted trials keep their measured projections. Interrupted managed attempts receive KILLED. Other terminal unaccepted managed attempts receive the failed invocation's status, even if the trainer already marked its run FINISHED.

With persistent storage, a managed attempt whose trial is still `RUNNING` stays open until recovery settles it. With `storage: null`, confirmed cleanup allows the invocation to close its remaining managed runs. If the controller died outright, the next run checks the attempts' cleanup before closing them. Until then, `sync-tracking` can close only the generation-summary run.

The next normal run or a confirmed [`phasesweep mcp recover-run`](mcp.md#run-state-and-recovery) can recover trial evidence from MLflow. When they reconcile trial cleanup, they also retry pending delivery from earlier invocations. A run does this before its first trial and at closeout. A replay only does it at closeout. If `recover-run` only finalizes a published result snapshot, use `sync-tracking` for its pending uploads. See the [delivery invariant](invariants.md#30-managed-run-status-follows-the-retained-attempt-outcome) for when each pass runs and what it can send.

An unanswered search or upload stops its delivery pass. Each trial uploads independently, so its own upload can wait up to one `mlflow.timeout_seconds` during an outage. An uncertain cleanup in the pass before the first trial stops the run. An interrupted or failed upload leaves delivery pending without changing accepted scores, trial states or winners. Until delivery succeeds, MLflow can still show a run as RUNNING.

The console counts unconfirmed creations it checks and records it sends. Its closing lines summarize recovered deliveries and pending work and print a retry command. CLI `status` reports `tracking_delivery` across generations, and [MCP status](mcp.md#reading-status-responses) reports it for one run. These counts cover saved uploads and unconfirmed creations. An unsaved outcome, such as the summary of a run killed outright, appears once a run, `recover-run` or `sync-tracking` saves it.

```bash
phasesweep sync-tracking mlflow.yaml
```

`sync-tracking` uses the original config and `--set` overrides to reconcile pending run creations, rebuild missing delivery records and retry uploads. It delivers only to the target recorded in `source.json`. `--generation GENERATION_ID` limits retries to one generation, whose ID is the MCP run ID for detached runs. Sync checks the [experiment lock and stored formats](runtime.md#storage-and-locks), confirms cleanup of tracking workers across generations and skips records it has already sent. It succeeds for a known generation with nothing left to send. It refuses a scope with no tracking activity or a config without an `mlflow` block. For pending work, see the [command exit statuses](runtime.md#following-a-run).

When the server does not answer, retry once it is reachable. If an unconfirmed creation is still inside its `timeout_seconds` window, retry at the time given in the error. When MLflow refuses a record, the warning names it and its cause. For example, a deleted run or experiment must be restored before retrying. Other records can still go through. Retry the command printed by `run` or `sync-tracking`, including its `--set` overrides and any `--generation` flag.

Run creation searches exact generation/attempt identity tags before creating anything. An unconfirmed creation request is never sent again. Once cleanup of its worker is confirmed, reconciliation binds the run if a lookup finds it. It retires the intent if a lookup started at least the current `mlflow.timeout_seconds` after the intent was written finds no match in the original experiment. Keep the original timeout when syncing to allow the same observation window. An earlier miss or a failed lookup leaves the intent pending. A scoped sync reads its generation's bindings and intents and reports any unreadable intent by path while keeping it pending. Databricks retains its SDK creation retries, so a lost response there can leave an extra unbound run.

A trial that is still running or awaiting cleanup needs [operator recovery](mcp.md#run-state-and-recovery) or the next run's preflight. A missing record is rebuilt from its own generation: attempt records of a published generation from the rows its winners were selected among, those of an unpublished one from the rows that launched its attempts, and winner references from the committed artifacts. A rebuilt record keeps its generation's metric timestamp, or the time of its `generation.yaml` when none of its records is left, and is not resent when a later reading supersedes it, so the run keeps showing the latest published reading. Pending records go oldest generation first, so a run with two readings queued during an outage ends with the newer one's status, tags and artifacts as well as its metric. For an unpublished generation, the lifecycle record supplies FAILED or KILLED. An absent terminal record means KILLED. Delivery uses an already saved summary when one exists.

Restore an unreadable or malformed record before retrying delivery of the outcome that depends on it. A malformed delivery record stops a run's batch pass before its first request, with a warning naming the file. Training and each trial's own upload can continue. `sync-tracking` sends independent records in scope before reporting the refusal and the count already delivered. See the [tracking output layout](runtime.md#output-layout) for the location of delivery records and cleanup receipts.

## Databricks status

Databricks targets preserve URI spelling, including profile case, and delegate authentication to the MLflow/Databricks SDK, which reads `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, OAuth client variables, or a `~/.databrickscfg` profile. Exported `DATABRICKS_*` variables take precedence over the profile named in `databricks://profile`, and some MLflow artifact uploads authenticate without that profile. Prefer `tracking_uri: databricks` with `env: {DATABRICKS_CONFIG_PROFILE: <profile>}`, or with host and token variables. Experiment names are workspace paths such as `/Users/you@example.com/sweep`.

In every `execution.inherit_env` mode, list rotating authentication and transport variables, such as `DATABRICKS_TOKEN`, `DATABRICKS_CLIENT_SECRET`, or `HTTPS_PROXY`, in `execution.passthrough_env`; see the [trainer environment contract](config.md#trainer-contract) and [Databricks authentication](https://docs.databricks.com/en/dev-tools/auth/index.html).

PhaseSweep is not tested against a live Databricks workspace or with framework integrations running on Databricks.
