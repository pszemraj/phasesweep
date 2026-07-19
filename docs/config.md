# Config guide

A phasesweep config is the contract between the orchestrator and your trainer. The orchestrator chooses parameter values, manages trial directories, extracts evidence, and decides which winner is exposed downstream. Your trainer parses overrides, runs the experiment, and provides the evidence that configured extractors read.

For every field, type, default, enum value, and validation constraint, use [config_reference.yaml](config_reference.yaml).

## Experiment keys

The top level of a single experiment describes identity, storage, the trial command contract, the objective, and the ordered phase plan. `experiment` is more than a display name: it is used in Optuna study names, output paths, and same-host lock identity, so it is restricted to ASCII `[A-Za-z0-9_-]+`.

`storage` selects the Optuna backend. `null` creates an in-memory study, `sqlite:///path.db` provides persistence for sequential `n_jobs == 1` studies, `journal:///path.journal` supports same-host parallel work, and an Optuna-supported RDB URL such as `postgresql://...` provides external storage. SQLite with parallel trials is rejected because concurrent Optuna writers are not a safe local parallel backend. Study reuse, top-ups, and `--from-phase` behavior are covered under [fingerprints and resume](runtime.md#fingerprints-and-resume).

Storage holds Optuna study state. `workdir` holds trial logs and result artifacts plus persisted winners, promotion decisions, and summaries.

`trial_command` is the command template for one trial. The supported placeholders are `{overrides}`, `{overrides_path}`, `{trial_dir}`, `{trial_id}`, `{phase}`, and `{run_name}`. phasesweep validates the template at config load, then shell-quotes rendered override values. The parser boundary is defined under [override formats](#override-formats).

`metric` defines the objective name, optimization direction, and extractor. `constraints` are additional finite scalar extractors with inclusive `min` and/or `max` bounds. A trial that violates a constraint is still recorded as a completed evaluation with its raw objective value. Current samplers receive that objective without feasibility guidance; infeasible trials cannot become the phase winner and count toward `max_consecutive_failures`. `contracts` are named bundles of fixed overrides and gates that phases can opt into when you need immutable comparison conditions across multiple phases.

The remaining top-level keys: `workdir` (default `./runs`) is the output root laid out in [runtime behavior](runtime.md#output-layout); `override_format` selects the trainer boundary covered in [override formats](#override-formats); `env` adds environment variables to every trial subprocess (included in semantic fingerprints); `timeout_seconds_per_run` is the whole-experiment wallclock guard described with the other timeouts in [runtime behavior](runtime.md#process-management).

For normal CLI runs, relative `workdir` values, relative paths inside `trial_command`, and file-backed storage paths resolve from the directory where `phasesweep` was invoked, not from the config file's directory. Invoke a relative-path config from one stable intended directory; changing cwd can silently select a different artifact tree and study. `phasesweep validate` checks the command template but does not require referenced executables or paths to exist. File storage URLs use three slashes for relative paths (`sqlite:///runs.db`, `journal:///runs.journal`) and four for absolute POSIX paths (`sqlite:////tmp/runs.db`). MCP-launched runs apply stricter [path and working-directory rules](mcp.md#paths-and-the-working-directory).

## Phase keys

Each phase is one Optuna study in an ordered chain. A phase may inherit winners from earlier phases; those inherited values become locked overrides for the current phase and for descendants. This greedy structure is useful for inspectable staged searches, but it is not a substitute for joint optimization when dimensions interact strongly.

Each phase declares a search space and trial-attempt budget, with optional fixed overrides, inherited winners, contracts, evidence gates, and promotion rules. The [config reference](config_reference.yaml) lists the exact phase fields and constraints. GPU allocation, timeouts, cleanup, and study top-ups are covered in [runtime behavior](runtime.md).

## Search parameters

`search_space` is a mapping from trainer override key to a typed float, integer, or categorical parameter object. Keys can be dotted paths such as `model.depth`; the same key namespace is used for inherited winners, contracts, fixed overrides, and sampled values. phasesweep rejects ambiguous compositions such as fixing a parent key while sampling one of its children, because no supported override format can represent that cleanly.

Float and integer bounds must be finite. Categorical choices must be Optuna-compatible scalars: `null`, booleans, integers, finite floats, or strings. Grid phases require a full grid unless `allow_partial_grid: true`; float grids require `step` and an evenly divisible interval. CMA-ES supports float and integer parameters, but not categorical parameters.

Search keys named `seed` or ending in `.seed` are rejected by default because optimizing a randomness source can select noise. Keep seeds fixed for ordinary comparisons, or set `allow_seed_search: true` for an explicit variance audit.

## Override formats

> [!IMPORTANT]
> The program launched by `trial_command` must parse the selected format. phasesweep renders values and validates placeholders; it does not adapt your trainer's CLI.

| Format      | Template placeholder | Trainer receives                     | Use when                                                         |
| ----------- | -------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `argparse`  | `{overrides}`        | `--key value` pairs                  | New scripts using `argparse`, Click, Typer, or similar parsers.  |
| `hydra`     | `{overrides}`        | `key=value` tokens                   | Compatibility for existing Hydra/OmegaConf applications.         |
| `json_file` | `{overrides_path}`   | Path to `<trial_dir>/overrides.json` | Recommended for structured config, nested values, MCP-launched sweeps, and agent-facing workflows. |

When a phase has inherited, fixed, or sampled overrides, `argparse` and `hydra` commands must include `{overrides}`. `json_file` commands must include `{overrides_path}`. Config validation rejects missing placeholders before any trial launches.

`argparse` renders lowercase booleans, bracketed comma-separated lists, and `None` for null. Hydra renders lowercase booleans and `null`, JSON-quotes strings, recursively renders bracketed lists, and rejects mappings. `json_file` preserves JSON types and expands dotted keys into nested objects, making it the most robust boundary for structured values. Every value must still be JSON-serializable; validation and dry-run do not write `overrides.json`, so serialization is exercised only by a real trial launch.

## Trainer contract

The command in `trial_command` is the training or evaluation program for one trial. phasesweep creates the trial directory, renders overrides, launches the process group, captures stdout/stderr, and then reads evidence. The trial process uses the directory `phasesweep run` was invoked from, or the catalog's pinned `cwd` for MCP-launched runs. The trainer must:

- Parse the selected [override format](#override-formats).
- Provide a finite objective through the configured extractor: write JSON or log evidence under `{trial_dir}`, or make the configured W&B run terminal with the metric in its summary.
- Exit nonzero when the trial failed and should be recorded as failed.
- When using W&B extraction or gates, let the W&B SDK use the injected `WANDB_RUN_ID`; `PHASESWEEP_RUN_NAME` remains available as the human-readable display name.

The trial environment starts from the phasesweep process environment, then top-level `env` overrides it. Every trial then receives `PHASESWEEP_TRIAL_DIR`, `PHASESWEEP_TRIAL_ID`, `PHASESWEEP_PHASE`, `PHASESWEEP_RUN_NAME`, `PHASESWEEP_GENERATION_ID`, and `PHASESWEEP_ATTEMPT_ID`, overriding same-named values. `WANDB_RUN_ID` is also set to the attempt ID so W&B evidence lookup uses an immutable identity instead of a reusable label. GPU assignment can override `CUDA_VISIBLE_DEVICES` and sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` only when the environment did not already define an order.

Metric extractor failures, non-finite metrics, nonzero exits, and missing required evidence fail the trial. Constraint bound violations are different: they produce completed but infeasible trials. phasesweep records their raw objective values and constraint readings, but feasibility is applied during winner selection rather than sampler guidance. Winner selection takes the best-metric feasible completed trial; metric values within an absolute `1e-12` of the best value resolve to the lowest trial number. When a swept key has no measurable effect, the selected value is therefore the lowest-numbered near-best trial's choice, not evidence of a preference.

## Override order

Within one trial, later layers override earlier layers:

1. Inherited winners' `effective_overrides`.
2. Contract `fixed_overrides`.
3. Phase `fixed_overrides`.
4. Sampled values from `search_space`.

A child phase may intentionally reset an inherited key with `fixed_overrides`. A sampled key cannot also be fixed or inherited.

## Extractors

Extractors turn trial evidence into finite floats. JSON and log extractors read files from the generation- and attempt-scoped `{trial_dir}`. JSON objectives must be numbers; numeric strings and booleans are rejected rather than coerced. W&B extractors query the immutable run ID assigned through `WANDB_RUN_ID`, accept only a `finished` run, and fail immediately when that run is `failed`, `crashed`, or `killed`. Human-readable display names do not participate in evidence correlation.

For agent-facing artifact boundaries, see the [MCP security model](mcp.md#security-model).

JSON extractors read a dotted key from a file under `{trial_dir}`. Log regex extractors read a captured numeric group named `value`. W&B extractors read a summary key from the finished run whose immutable ID matches the current attempt. The [config reference](config_reference.yaml) is the complete contract for each shape.

## Evidence gates

Supported gates are `required_file`, `json_equals`, `json_scalar_bound`, `artifact_size`, `sha256`, and `wandb_summary_required`. Gate failures mark the trial `FAIL` unless promotion has `requires_gates: false`, where they are advisory evidence.

`json_equals` is type-strict, so `true`, `1`, and `1.0` are distinct. Use `json_scalar_bound` for numeric comparisons where integer and float representations should both pass.

`artifact_size` requires `source: file|directory|json`. File and directory sources measure materialized bytes. JSON source reads an integer byte estimate from `path` plus `key`.

## Promotion

Promotion decides whether a phase or suite study winner is exposed downstream. The comparison uses signed improvement: for `minimize`, improvement is `baseline.metric - candidate.metric`; for `maximize`, it is `candidate.metric - baseline.metric`. The candidate promotes when improvement is at least the configured threshold and required gates passed.

For a phase promotion failure, `stop` raises an error, `skip` ends the remaining phases and permits a partial experiment summary, and `continue_baseline` exposes a clone of the baseline winner. Every evaluated phase promotion writes `<phase>/promotion.yaml`; an exposed candidate or baseline clone is written to `winner.yaml` and included in `summary.yaml`.

## Suites

Suites run studies sequentially in declaration order. `depends_on` requires a prior study to have produced an exposed result; it does not pass winner overrides into the dependent study. Each study compiles to a normal experiment named `<suite>__<study>`, using defaults from `suite.defaults` when the study omits a field. Study `env` values merge over default `env`; study `contracts` merge over default `contracts`. Worked experiment files remain under `examples/`; this guide deliberately does not duplicate them as templates.

Suite-level `run.log` and `suite_summary.yaml` use `suite.defaults.workdir`; each compiled study writes its normal experiment artifacts under that study's resolved `workdir`. Suite promotion `min_delta_vs` may name a prior study or `study.phase`; a bare study name resolves to that study's final phase. On promotion failure, `stop` aborts the suite, `skip` omits that study and continues until a later dependency requires it, and `continue_baseline` substitutes a clone of the baseline for the study's final winner. Suite decisions are recorded in `suite_summary.yaml`, not in a per-study `promotion.yaml`. The suite summary is written only after the suite loop completes; a stop or missing dependency can leave completed study artifacts without that summary. `--from-phase` is supported only for single-experiment configs, not suites.
