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

The experiment must already exist and be active. Quote a numeric `experiment_id` so YAML keeps it a string. `tracking_uri` accepts HTTP(S), `databricks`, and `databricks://profile`; HTTP credentials belong in SDK authentication or the environment rather than the URI. Before a phase allocates its first trial, PhaseSweep resolves the experiment; the first such phase also creates the generation-summary run. A server, credential or experiment failure stops there without spending trial slots or writing a generation. The refusal names the target. The resolved ID is saved in `<workdir>/<experiment>/tracking/source.json`; an experiment deleted and recreated under a different ID is refused. Losing the target after a phase starts fails its remaining trials, which count toward `n_trials`.

The MLflow client retries a server that refuses or does not answer. When those retries outlast `timeout_seconds`, the refusal says the target did not answer within `timeout_seconds`; when they end sooner, it says the server refused the connection, including when MLflow wraps the transport failure. Either way, check `tracking_uri` and that the server is running.

`tracking_uri`, `experiment_id`, `experiment_name` and `run_mode` are in the study fingerprint, so changing one needs a new experiment name; `poll_seconds` and `timeout_seconds` are not. `phasesweep validate` and `run --dry-run` check neither authentication nor the experiment. The credentials need write access: PhaseSweep creates runs, logs metrics, params and tags, uploads artifacts and ends runs. An `env` entry for `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME` must equal the block's own normalized value (lowercase scheme and host, no trailing slash), so the selector the block leaves unset cannot appear, and `env` may not set `MLFLOW_RUN_ID`; config load fails otherwise.

Controller tracking operations run in supervised workers outside GPU leases. Each has a `timeout_seconds` budget; phase/run deadlines also cap experiment resolution, run creation and evidence capture. Result uploads happen after acceptance and can extend past those deadlines. The SDK retries transient HTTP failures within that budget. A failed attempt-run preparation spends a trial and counts toward `max_consecutive_failures`; a phase/run deadline records it as interrupted instead. See [trial execution and evidence](runtime.md#trial-execution-and-evidence) for process cleanup and lease ordering.

## Choose run ownership

In **managed** mode, PhaseSweep writes the trainer input, creates the native MLflow run named after the trial's `PHASESWEEP_RUN_NAME`, and persists its returned ID before acquiring a GPU or launching training. The trainer receives `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and `MLFLOW_RUN_ID`. Resume that supplied run:

```python
import os
import mlflow

with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

After confirmed cleanup, PhaseSweep finishes a still-running managed run as FINISHED after success, FAILED after a failed trial, or KILLED after an [interruption](runtime.md#interrupted-trials-rerun). Interrupted reruns receive new managed runs. Terminal statuses are preserved; `phasesweep.outcome` records a later trial failure. FAILED and KILLED runs cannot supply accepted evidence.

In **trainer** mode, your existing logger creates and finishes the run. The trainer receives `MLFLOW_TRACKING_URI` and the resolved `MLFLOW_EXPERIMENT_ID`, so a plain `mlflow.start_run()` opens it in the configured experiment. Bind that exact run once it exists:

```python
import mlflow
from phasesweep.tracking import bind_mlflow_run

with mlflow.start_run() as run:
    bind_mlflow_run(run.info.run_id)
    mlflow.log_metric("eval/loss", 0.42, step=100)
```

The helper verifies the configured experiment and that the run is still RUNNING, attaches PhaseSweep identity tags, and writes the local binding. It does not log or replace metrics. PhaseSweep never discovers runs by name or recency, finishes trainer-owned runs, or changes their existing parent relationships. The trainer must close its run successfully before evidence can be accepted. A missing, foreign or conflicting binding fails the trial, and so does a bound run that was deleted, lies outside the configured experiment or carries another attempt's identity tags. Existing framework loggers must expose their actual run ID and honor this ownership contract; other logger integrations require their own qualification. In either mode, disable autologging and nested-run behavior that starts runs of its own: PhaseSweep reads only the bound run, never a parent or child.

### Transformers Trainer

For native Transformers logging with managed ownership, set `report_to: [mlflow]` in the trainer's arguments and invoke the packaged launcher from `trial_command`:

```yaml
trial_command: "python -m phasesweep.tracking.mlflow_launch /absolute/path/to/run_seq2seq.py --report_to mlflow {overrides}"
override_format: argparse
```

Pass the absolute trainer script path, its normal arguments, and `{overrides}`. The launcher preserves the script's argument parsing and sibling imports, removes `MLFLOW_RUN_ID` before the Transformers callback starts, and keeps the supplied managed run active across the entire script, including final evaluation. This lets the callback log to the active run without taking ownership of closing it when training ends. A script error or nonzero exit marks the run failed; a successful `sys.exit(0)` finishes it. The launcher requires `mlflow.run_mode: managed` and the `phasesweep[mlflow]` extra. Verify that the example imports its matching Transformers package using the [trainer environment preflight](config.md#source-checkout-preflight).

Native metrics retain their Trainer names, such as `eval_loss` and `eval_rouge1`; choose the exact final optimizer step when comparing fixed training budgets.

With `load_best_model_at_end=True`, the logged step does not identify the evaluated checkpoint. Use `json_envelope` when that identity matters; [evaluation positions](config.md#objective-evidence-constraints-and-gates) also covers the `validate` warning for this setting.

## Select metrics and require evidence

Every MLflow extractor declares its selector explicitly. Selection is independent of `metric.goal`, which ranks the selected scalars across trials.

For a few finalists that need ranking over seeds, use `phases[].repeats` with `key: seed` and the desired seed values. PhaseSweep launches one trial and one tracked run per seed, reads each run's scalar objective, and ranks configurations by their group mean (or the configured `reduce`). A trainer that creates several MLflow runs inside one trial does not supply a repeat group; the reader uses only the bound attempt run. See [repeats](config.md#repeats) for the complete configuration.

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

At least one namespace must be nonempty. A finished run missing a required metric, param or tag fails at once, naming the missing and available keys, for example `MLflow run finished without metric 'loss'; available metric keys: ['accuracy', 'eval_loss'].` An unfinished run is polled every `poll_seconds` until timeout, which reports its last state; a trainer-owned run left open fails this way. The capture budget includes worker startup and stops polling `min(5, timeout_seconds / 2)` seconds before expiry to retain that diagnostic. Accepted evidence freezes the selected value, step, timestamp, native run identity and required metadata. Local reads and no-work replay need no SDK or server access; a replay that [publishes a generation](runtime.md#timeouts-cleanup-and-continuation) creates no tracking directories or uploads. See [confound verdicts](runtime.md#confound-verdicts) for evaluation-position checks.

## Result delivery and recovery

Accepted trials receive reserved `phasesweep.*` metrics, parameters, and tags: objective, measured constraints, report-only `observe` values (`phasesweep.observed.<name>`), feasibility, trainer duration, effective fixed/inherited/sampled overrides, and phase/inheritance references. Trainers should leave this namespace to PhaseSweep. The `phasesweep.*` metrics are logged at step 0. MLflow truncates a parameter value longer than 6000 characters, so the attempt's summary artifact below is the full record of its overrides; with the `mlflow` block set, config load refuses constraint names, observation names and override keys whose published key contains characters outside alphanumerics, underscores, periods, dashes, slashes, colons and spaces, exceeds 250 characters, or is not a normalized relative path. The `phasesweep.override.` prefix bounds an override key to 230 characters. A generation-summary run in the same experiment groups managed attempts and holds `phasesweep/generation.json` plus winner references. Each attempt run holds `phasesweep/attempt-<attempt-id>.json`, and a winning run also holds `phasesweep/winner-<phase>.json`. For a phase with [`repeats`](config.md#repeats), the winning run is the last member of the winning group, and its winner reference records the group's reduced metric, while every member's run keeps its own objective. Trainer-owned attempts retain their own parents.

After confirmed cleanup, PhaseSweep queues result projections for accepted trials and for failed invocations' generation-summary and outstanding attempt runs. Upload failures leave pending records without changing accepted scores, trial states or winners. Once published, the local result stays successful even when a winner reference could not be saved or sent, or upload-worker cleanup is uncertain. A `PENDING` closeout names the generation, upload count and retry command. CLI `status` reports `tracking_delivery` counts across generations; MCP status reports its run's uploads separately from its frozen scientific result:

```bash
phasesweep sync-tracking examples/mlflow.yaml
```

`sync-tracking` retries uploads only, using the original target configuration. `--generation GENERATION_ID` limits retries to one generation, whose ID is the MCP run ID for detached runs. `sync-tracking` holds the experiment lock, skips delivered projections, and exits nonzero while uploads remain pending, when no delivery records exist in scope, or without an `mlflow` block. Permanent remote refusals, such as a deleted run, remain pending.

Run creation searches exact generation/attempt identity tags before creating anything. An unconfirmed creation request is not sent again: its pending record survives restarts, and reconciliation or `sync-tracking` binds the run if it appears. HTTP(S) uses this identity reconciliation; Databricks retains its SDK creation retries, so a lost response there can leave an extra unbound run.

Missing projections are rebuilt from settled trials or committed winner artifacts. An attempt must be accepted or terminal with confirmed cleanup; RUNNING or uncertain-cleanup trials require [MCP recovery](mcp.md#run-state-and-recovery) or the next run's preflight before retrying. A failed invocation's missing generation-summary record is not manufactured. Tracking-worker recovery saves a receipt under `tracking/<generation-id>/cleanup-receipts/` before retiring the registry entry; `recover-run` accepts it for that generation or a runner that reported the worker. An unreadable receipt is ignored, leaving recovery without that cleanup evidence.

## Databricks status

Databricks targets preserve URI spelling, including profile case, and delegate authentication to the MLflow/Databricks SDK, which reads `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, OAuth client variables, or a `~/.databrickscfg` profile. Exported `DATABRICKS_*` variables take precedence over the profile named in `databricks://profile`, and some MLflow artifact uploads authenticate without that profile. Prefer `tracking_uri: databricks` with `env: {DATABRICKS_CONFIG_PROFILE: <profile>}`, or with host and token variables. Experiment names are workspace paths such as `/Users/you@example.com/sweep`.

In every `execution.inherit_env` mode, list rotating authentication and transport variables, such as `DATABRICKS_TOKEN`, `DATABRICKS_CLIENT_SECRET`, or `HTTPS_PROXY`, in `execution.passthrough_env`; see the [trainer environment contract](config.md#trainer-contract) and [Databricks authentication](https://docs.databricks.com/en/dev-tools/auth/index.html).

PhaseSweep is not tested against a live Databricks workspace or with framework integrations running on Databricks.
