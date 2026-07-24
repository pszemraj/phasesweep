# Config guide

A phasesweep config is the contract between the orchestrator and your trainer. The orchestrator chooses parameter values, manages trial directories, extracts evidence, and decides which winner is exposed downstream. Your trainer parses overrides, runs the experiment, and provides the evidence that configured extractors read.

For every field, type, default, enum value, and validation constraint, use [config_reference.yaml](config_reference.yaml).

## Experiment keys

The top level of a single experiment describes identity, storage, the trial command contract, the objective, and the ordered phase plan. `experiment` is more than a display name: it is used in Optuna study names, output paths, and same-host lock identity, so it is restricted to ASCII `[A-Za-z0-9_-]+`.

`storage` selects the Optuna backend. `null` creates an in-memory study, `sqlite:///path.db` provides persistence for sequential `n_jobs == 1` studies, `journal:///path.journal` supports same-host parallel work, and an Optuna-supported RDB URL such as `postgresql://...` provides external storage. SQLite with parallel trials is rejected because concurrent Optuna writers are not a safe local parallel backend. Study reuse, top-ups, and `--from-phase` behavior are covered under [fingerprints and resume](runtime.md#fingerprints-and-resume).

An RDB `storage` URL is rejected at config validation unless `allow_external_rdb_single_host: true` also acknowledges the single-host requirement explained under [multi-host storage](runtime.md#concurrency-model).

Storage holds Optuna study state. `workdir` holds trial logs and result artifacts plus persisted winners, promotion decisions, and summaries.

`trial_command` is the command template for one trial. phasesweep validates the template at config load and shell-quotes rendered override values. The [config reference](config_reference.yaml) defines the supported placeholders; the parser boundary is explained under [override formats](#override-formats).

`provenance` is the operator-declared identity of inputs that the command string cannot describe, such as the trainer revision, base config, dataset, dependency lock, container, tokenizer, or starting checkpoint. Persistent storage requires at least one nonempty entry. PhaseSweep includes the complete mapping in every phase fingerprint, so change its values whenever any external input changes; use a new experiment name when results from the old and new provenance should remain separate. PhaseSweep does not infer imports or hash arbitrary shell-command inputs.

`metric` defines the objective name, optimization direction, and extractor. `constraints` are additional finite scalar extractors with inclusive `min` and/or `max` bounds. A trial that violates a constraint is still recorded as a completed evaluation with its raw objective value. Current samplers receive that objective without feasibility guidance; infeasible trials cannot become the phase winner and count toward `max_consecutive_failures`. `contracts` are named bundles of fixed overrides and gates that phases can opt into when you need immutable comparison conditions across multiple phases.

Agent-visible metric descriptors label the configured extractor's evidence assurance with per-field flags, not one coarse claim. Every extractor is `attempt_bound` (evidence is read from a trial directory or run ID uniquely scoped to this generation and attempt). Only `json_envelope` is `objective_name_bound`, `split_bound`, and `evaluation_policy_bound`, because those are required fields the runtime always checks against the envelope's own reported values; `log_regex` and W&B carry no such binding. `checkpoint_declared`/`expected_step_declared` report whether the config pinned a value for those *optional* envelope fields, and `checkpoint_value_bound`/`expected_step_value_bound` report whether the runtime actually validates the envelope against that declared value (`True` only when declared — an envelope with no declared checkpoint or step still must report *some* non-empty checkpoint and non-negative step, but nothing pins it to a specific one).

The remaining top-level keys: `workdir` (default `./runs`) is the output root laid out in [runtime behavior](runtime.md#output-layout); `override_format` selects the trainer boundary covered in [override formats](#override-formats); `env` adds environment variables to every trial subprocess (included in semantic fingerprints); `timeout_seconds_per_run` is the whole-experiment wallclock guard described with the other timeouts in [runtime behavior](runtime.md#process-management).

For normal CLI runs, relative `workdir` values, relative paths inside `trial_command`, and file-backed storage paths resolve from the directory where `phasesweep` was invoked, not from the config file's directory. Invoke a relative-path config from one stable intended directory; changing cwd can silently select a different artifact tree and study. `phasesweep validate` checks the command template but does not require referenced executables or paths to exist. File storage URLs use three slashes for relative paths (`sqlite:///runs.db`, `journal:///runs.journal`) and four for absolute POSIX paths (`sqlite:////tmp/runs.db`). MCP-launched runs apply stricter [path and working-directory rules](mcp.md#paths-and-the-working-directory).

## Phase keys

Each phase is one Optuna study in an ordered chain. A phase may inherit winners from earlier phases; those inherited values become locked overrides for the current phase and for descendants. This greedy structure is useful for inspectable staged searches, but it is not a substitute for joint optimization when dimensions interact strongly.

Each phase declares a search space and trial-attempt budget, with optional fixed overrides, inherited winners, contracts, evidence gates, and promotion rules. The [config reference](config_reference.yaml) lists the exact phase fields and constraints. GPU allocation, timeouts, cleanup, and study top-ups are covered in [runtime behavior](runtime.md).

## Search parameters

`search_space` is a mapping from trainer override key to a typed float, integer, or categorical parameter object. Keys can be dotted paths such as `model.depth`; the same key namespace is used for inherited winners, contracts, fixed overrides, and sampled values. phasesweep rejects ambiguous compositions such as fixing a parent key while sampling one of its children, because no supported override format can represent that cleanly.

Use categorical parameters for explicit choices and integer or float parameters for ranges. Grid sampling is useful when every finite combination should run; CMA-ES is useful for interacting numeric dimensions. The [config reference](config_reference.yaml) defines bounds, grid completeness, sampler compatibility, and the explicit waiver for searching seed values.

## Override formats

> [!IMPORTANT]
> The program launched by `trial_command` must parse the selected format. phasesweep renders values and validates placeholders; it does not adapt your trainer's CLI.

| Format | Use when |
| --- | --- |
| `argparse` | New scripts using `argparse`, Click, Typer, or similar parsers. |
| `hydra` | Existing Hydra/OmegaConf applications. |
| `json_file` | Structured config, nested values, MCP-launched sweeps, and agent-facing workflows. |

Each format has a required template placeholder and distinct value encoding. The [config reference](config_reference.yaml) defines that wire contract. `json_file` preserves JSON types and expands dotted keys into nested objects, making it the most robust boundary for structured values; serialization is exercised only by a real trial launch because validation and dry-run do not write `overrides.json`.

## Trainer contract

The command in `trial_command` is the training or evaluation program for one trial. phasesweep creates the trial directory, renders overrides, launches the process group, captures stdout/stderr, and then reads evidence. The trial process uses the directory `phasesweep run` was invoked from, or the catalog's pinned `cwd` for MCP-launched runs. The trainer must:

- Parse the selected [override format](#override-formats).
- Provide a finite objective through the configured extractor: write JSON or log evidence under `{trial_dir}`, or make the configured W&B run terminal with the metric in its summary.
- Exit nonzero when the trial failed and should be recorded as failed.
- When using W&B extraction or gates, let the W&B SDK use the injected `WANDB_RUN_ID`; `PHASESWEEP_RUN_NAME` remains available as the human-readable display name.
- For a `json_envelope` extractor, copy `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID`, and `PHASESWEEP_OVERRIDES_SHA256` into the result envelope. PhaseSweep verifies all three before accepting its objective.

The trial environment starts from the phasesweep process environment, then top-level `env` overrides it. Every trial then receives `PHASESWEEP_TRIAL_DIR`, `PHASESWEEP_TRIAL_ID`, `PHASESWEEP_PHASE`, `PHASESWEEP_RUN_NAME`, `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID`, and `PHASESWEEP_OVERRIDES_SHA256`, overriding same-named values. The digest covers the exact PhaseSweep-written overrides artifact used by the current override format. `WANDB_RUN_ID` is also set to the attempt ID so W&B evidence lookup uses an immutable identity instead of a reusable label. GPU assignment can override `CUDA_VISIBLE_DEVICES` and sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` only when the environment did not already define an order.

Metric extractor failures, non-finite metrics, nonzero exits, and missing required evidence fail the trial. Constraint bound violations are different: they produce completed but infeasible trials. phasesweep records their raw objective values and constraint readings, but feasibility is applied during winner selection rather than sampler guidance. Winner selection takes the best-metric feasible completed trial; metric values within an absolute `1e-12` of the best value resolve to the lowest trial number. When a swept key has no measurable effect, the selected value is therefore the lowest-numbered near-best trial's choice, not evidence of a preference.

### Result envelope

A `json_envelope` trainer publishes this versioned shape after successful evaluation:

```json
{
  "schema_version": 1,
  "status": "complete",
  "generation_id": "<current generation ID>",
  "attempt_id": "<current attempt ID>",
  "overrides_sha256": "<current resolved-overrides digest>",
  "objective": {
    "name": "val_loss",
    "split": "validation",
    "value": 0.123
  },
  "evaluation": {
    "policy": "final_checkpoint",
    "checkpoint": "final.pt",
    "step": 1000
  }
}
```

Copy the generation ID, attempt ID, and overrides digest from the reserved trial environment values listed in the [config reference](config_reference.yaml). The objective name, split, and evaluation policy must match the extractor config. The checkpoint must be a nonempty identity, the step must be a non-negative integer, and the objective value must be a finite JSON number rather than a string or boolean. Configured `checkpoint` and `expected_step` values are matched exactly.

## Override order

Within one trial, later layers override earlier layers:

1. Inherited winners' `effective_overrides`.
2. Contract `fixed_overrides`.
3. Phase `fixed_overrides`.
4. Sampled values from `search_space`.

A child phase may intentionally reset an inherited key with `fixed_overrides`. A sampled key cannot also be fixed or inherited.

## Extractors

Extractors turn trial evidence into finite floats. JSON and log extractors read files from the generation- and attempt-scoped `{trial_dir}`. Primary metrics from local JSON must use `json_envelope`, which binds the result to the current attempt, resolved overrides, objective, split, checkpoint, evaluation policy, and step. Plain `json` remains available for constraints; its selected value must be a number, not a numeric string or boolean. W&B extractors use the immutable run ID assigned through `WANDB_RUN_ID`; human-readable display names do not participate in evidence correlation.

For agent-facing artifact boundaries, see the [MCP security model](mcp.md#security-model).

The [config reference](config_reference.yaml) defines each extractor shape, including JSON keys, log capture groups, W&B terminal-state handling, and polling timeouts.

## Evidence gates

Evidence gates validate local artifacts or W&B summary values after extraction. Gate failures mark the trial `FAIL` unless promotion has `requires_gates: false`, where they are advisory evidence. The [config reference](config_reference.yaml) defines the available gate shapes.

`json_equals` is type-strict, so `true`, `1`, and `1.0` are distinct. Use `json_scalar_bound` for numeric comparisons where integer and float representations should both pass.

## Promotion

Promotion decides whether a phase or suite study winner is exposed downstream. The comparison uses signed improvement: for `minimize`, improvement is `baseline.metric - candidate.metric`; for `maximize`, it is `candidate.metric - baseline.metric`. The candidate promotes when improvement is at least the configured threshold and required gates passed.

For a phase promotion failure, `stop` raises an error, `skip` ends the remaining phases and permits a partial experiment summary, and `continue_baseline` exposes a clone of the baseline winner. Every evaluated phase promotion writes `<phase>/promotion.yaml`; an exposed candidate or baseline clone is written to `winner.yaml` and included in `summary.yaml`. Winner records keep the exposure phase separate from `winner_source`, which identifies the concrete source phase and trial; promotion metadata retains the rejected candidate.

## Suites

Suites run studies sequentially in declaration order. `depends_on` requires a prior study to have produced an exposed result; it does not pass winner overrides into the dependent study. Each study compiles to a normal experiment named `<suite>__<study>`, using defaults from `suite.defaults` when the study omits a field. Study `env` values merge over default `env`; study `contracts` merge over default `contracts`; study `provenance` replaces default `provenance` when supplied, and explicit `null` clears it. See the [worked configs](../examples/).

Suite-level `run.log` and the compatibility projection `suite_summary.yaml` use `suite.defaults.workdir`; each compiled study writes its normal experiment artifacts under that study's resolved `workdir`. Every invocation claims an immutable `suite_generations/<id>/` namespace. Its summary binds the compiled study graph, resolved component experiments, promotion rules, historical phase comments, component experiment generation IDs and phase fingerprints, timestamps, and PhaseSweep version under one `suite_fingerprint`; `last_successful_suite_generation.yaml` selects the authoritative result. A failed first run publishes none, and a failed rerun preserves the prior last-successful exposed result. `show-winners` reads the immutable selected summary, prints promotion decisions separately from exposed winners, and renders only the comments stored with that result. If the current compiled suite differs, it labels the result historical instead of decorating old evidence with current config text.

Suite promotion `min_delta_vs` may name a prior study or `study.phase`; a bare study name resolves to that study's final phase. On promotion failure, `stop` aborts the suite, `skip` omits that study and continues until a later dependency requires it, and `continue_baseline` substitutes a clone of the baseline for the study's final winner. Suite decisions are recorded in the suite-generation summary, not in a per-study `promotion.yaml`. `show-winners` never substitutes the compiled experiments' raw candidate winners. `--from-phase` is supported only for single-experiment configs, not suites.
