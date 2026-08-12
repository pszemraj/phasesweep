# Config guide

A PhaseSweep config is the contract between the orchestrator and your trainer. The orchestrator chooses parameter values, manages trial directories, extracts evidence, and decides which winner is exposed downstream. Your trainer parses overrides, runs the experiment, and provides the evidence that configured extractors read.

For every field, type, default, enum value, and validation constraint, use [config_reference.yaml](config_reference.yaml). If a config that used to load now fails validation, see [upgrading existing configs](#upgrading-existing-configs).

## Experiment keys

The top level of a single experiment describes identity, storage, the trial command contract, the objective, and the ordered phase plan. `experiment` is more than a display name: it is used in Optuna study names, output paths, and same-host lock identity, so it is restricted to ASCII `[A-Za-z0-9_-]+`.

`storage` selects the Optuna backend. `null` creates an in-memory study, `sqlite:///path.db` provides persistence for sequential `n_jobs == 1` studies, `journal:///path.journal` supports same-host parallel work, and an Optuna-supported RDB URL such as `postgresql://...` provides external storage. SQLite with parallel trials is rejected because concurrent Optuna writers are not a safe local parallel backend. Study reuse, top-ups, and `--from-phase` behavior are covered under [fingerprints and resume](runtime.md#fingerprints-and-resume).

An RDB `storage` URL is rejected at config validation unless `allow_external_rdb_single_host: true` also acknowledges the single-host requirement explained under [multi-host storage](runtime.md#concurrency-model).

Storage holds Optuna study state. `workdir` holds trial logs, result artifacts, persisted winners, promotion decisions, and summaries. With persistent storage, each phase study is bound to the resolved `<workdir>/<experiment>` root it publishes into; a different root is refused. To relocate an artifact tree or migrate a pre-binding study, follow the [workdir rebind procedure](runtime.md#fingerprints-and-resume). In-memory studies do not bind a workdir because they cannot outlive their process.

`trial_command` is the command template for one trial. PhaseSweep validates the template at config load and shell-quotes rendered override values. The [config reference](config_reference.yaml) defines the supported placeholders; the parser boundary is explained under [override formats](#override-formats).

`provenance` is the operator-declared identity of inputs that the command string cannot describe, such as the trainer revision, base config, dataset, dependency lock, container, tokenizer, or starting checkpoint. Persistent storage requires at least one nonempty entry. PhaseSweep includes the complete mapping in every phase fingerprint, so change its values whenever any external input changes; use a new experiment name when results from the old and new provenance should remain separate. PhaseSweep does not infer imports or hash arbitrary shell-command inputs.

`metric` defines the objective name, optimization direction, and extractor. `constraints` are additional finite scalar extractors with inclusive `min` and/or `max` bounds. A trial that violates a constraint is still recorded as a completed evaluation with its raw objective value. Current samplers receive that objective without feasibility guidance; infeasible trials cannot become the phase winner and count toward `max_consecutive_failures`. `contracts` are named bundles of fixed overrides and gates that phases can opt into when you need immutable comparison conditions across multiple phases.

The MCP API reports the configured extractor's guarantees through explicit [objective-evidence assurance fields](mcp.md#objective-evidence-assurance).

The remaining top-level keys: `workdir` (default `./runs`) is the output root laid out in [runtime behavior](runtime.md#output-layout); `override_format` selects the trainer boundary covered in [override formats](#override-formats); `env` adds environment variables to every trial subprocess (included in semantic fingerprints); `timeout_seconds_per_run` is the whole-experiment wallclock guard described with the other timeouts in [runtime behavior](runtime.md#process-management).

`execution` declares the trainer's execution context explicitly instead of inheriting it silently. `execution.cwd` sets the working directory every trainer subprocess runs in; the resolved effective path joins the semantic fingerprint, so one persistent study can never mix trainers reached through different working directories (a relative `trial_command` like `python trainer.py` means different code from different directories). Prefer an absolute path - a relative value resolves from the invocation directory, and two invocation directories then count as two incompatible execution contexts by design. When unset, trainers run in the invocation cwd and that resolved directory still joins the fingerprint because it affects the command. `execution.inherit_env` controls which ambient environment variables trainers inherit: `all` (default, historical behavior), `none` (the minimal base listed in the [config reference](config_reference.yaml)), or a list of names inherited on top of that base - use a narrowed contract to keep unrelated host variables and secrets out of trainer subprocesses. The inherit contract (mode or sorted names) joins the fingerprint; ambient *values* are never hashed, so put values that change trial meaning in `env`, which is always fingerprinted. Configured `env` entries apply on top of whatever is inherited. PhaseSweep records that composed base environment's digest and variable names without adding them to the fingerprint; `execution.record_env: true` also stores its values. See [output layout](runtime.md#output-layout) for the exact boundary, stored records, and drift warnings.

For normal CLI runs, relative `workdir` values, relative `execution.cwd` values, and file-backed storage paths resolve from the directory where `phasesweep` was invoked, not from the config file's directory. Relative paths used by `trial_command` resolve from the trainer's effective working directory: `execution.cwd` when set, otherwise the invocation directory. Invoke a relative-path config from one stable intended directory; changing cwd can silently select a different artifact tree, study, or trainer. `phasesweep validate` checks the command template but does not require referenced executables or paths to exist. File storage URLs use three slashes for relative paths (`sqlite:///runs.db`, `journal:///runs.journal`) and four for absolute POSIX paths (`sqlite:////tmp/runs.db`). MCP-launched runs apply stricter [path and working-directory rules](mcp.md#paths-and-the-working-directory).

## Phase keys

Each phase is one Optuna study in an ordered chain. A phase may inherit winners from earlier phases; those inherited values become locked overrides for the current phase and for descendants. This greedy structure is useful for inspectable staged searches, but it is not a substitute for joint optimization when dimensions interact strongly.

Each phase declares a search space and trial-attempt budget, with optional fixed overrides, inherited winners, contracts, evidence gates, and promotion rules. The [config reference](config_reference.yaml) lists the exact phase fields and constraints. GPU allocation, timeouts, cleanup, and study top-ups are covered in [runtime behavior](runtime.md).

## Search parameters

`search_space` is a mapping from trainer override key to a typed float, integer, or categorical parameter object. Keys can be dotted paths such as `model.depth`; the same key namespace is used for inherited winners, contracts, fixed overrides, and sampled values. PhaseSweep rejects ambiguous compositions such as fixing a parent key while sampling one of its children, because no supported override format can represent that cleanly.

Use categorical parameters for explicit choices and integer or float parameters for ranges. `choices` must be pairwise unequal under plain Python comparison, so `[1, 1.0, true]` is rejected as firmly as a literal repeat: Optuna records a sampled value as its `==` index into `choices`, so choices that compare equal cannot be told apart once persisted and the winner would name a value the trial never ran. For CLI override formats, choices must also render to distinct wire values; for example, argparse cannot distinguish `1` from `"1"`. `json_file` preserves their JSON types and may represent that pair distinctly. Give each choice a value no other choice equals in the selected override format. Grid sampling is useful when every finite combination should run; CMA-ES is useful for interacting numeric dimensions. The [config reference](config_reference.yaml) defines bounds, grid completeness, sampler compatibility, and the explicit waiver for searching seed values.

## Sampler capability on persistent storage

The `sampler` block is optional and defaults to `type: tpe` with no seed, which is fine for an in-memory run. A persistent `storage` changes that, because the study outlives the process that created it, so each phase must state two things up front rather than discover them mid-sweep:

| Sampler | Seed | `acknowledge_nonresumable` |
| --- | --- | --- |
| `grid` | optional (traversal order only) | rejected |
| `random` | required | rejected |
| `tpe`, `cmaes` | required | required (`true`) |

An unseeded `tpe`, `random`, or `cmaes` phase draws a different sequence on every invocation, so the durable trials it accumulates cannot be reproduced or explained afterwards. `tpe` and `cmaes` additionally hold process-local sampler state that Optuna storage does not persist: PhaseSweep refuses to resume such a phase mid-target or to raise its `n_trials` later (see [runtime behavior](runtime.md#fingerprints-and-resume)). Setting `acknowledge_nonresumable: true` is your statement that you accept that contract and will run each target in one invocation; setting it on `grid` or `random`, which resume safely, is rejected as meaningless config.

```yaml
storage: sqlite:///runs.db
phases:
  - name: depth
    n_trials: 4
    sampler: { type: grid }
  - name: lr
    n_trials: 12
    sampler: { type: tpe, seed: 0, acknowledge_nonresumable: true }
  - name: weight_decay
    n_trials: 8
    sampler: { type: random, seed: 1 }
```

`phasesweep validate` and `phasesweep run --dry-run` print one capability line per phase so the contract is visible before any trial runs:

```text
phase 'depth': sampler=grid (resumable)
phase 'lr': sampler=tpe seed=0 (non-resumable: run each target in one invocation)
phase 'weight_decay': sampler=random seed=1 (resumable, reproducible)
```

## Override formats

> [!IMPORTANT]
> The program launched by `trial_command` must parse the selected format. PhaseSweep renders values and validates placeholders; it does not adapt your trainer's CLI.

| Format | Use when |
| --- | --- |
| `argparse` | New scripts using `argparse`, Click, Typer, or similar parsers. |
| `hydra` | Existing Hydra/OmegaConf applications. |
| `json_file` | Structured config, nested values, MCP-launched sweeps, and agent-facing workflows. |

Each format has a required template placeholder and distinct value encoding. The [config reference](config_reference.yaml) defines that wire contract. The scalar/list CLI formats reject non-finite floats because their rendered `nan`/`inf` values have no distinct semantic JSON identity. `json_file` preserves JSON types and expands dotted keys into nested objects, making it the most robust boundary for structured values. Config load checks every statically known composed value with the same strict serializer used for real trials; see [JSON file override validation](#json-file-override-validation) for YAML scalar pitfalls.

## Trainer contract

The command in `trial_command` is the training or evaluation program for one trial. PhaseSweep creates the trial directory, renders overrides, launches the process group, captures stdout/stderr, and then reads evidence. The trial process uses `execution.cwd` when configured. Otherwise it uses the directory where `phasesweep run` was invoked, or the catalog's pinned `cwd` for MCP-launched runs. The trainer must:

- Parse the selected [override format](#override-formats).
- Provide a finite objective through the configured extractor: call `report_objective(...)` or write a compatible JSON envelope, write log evidence under `{trial_dir}`, or make the configured W&B run terminal with the metric in its summary.
- Exit nonzero when the trial failed and should be recorded as failed.
- When using W&B extraction or gates, let the W&B SDK use the injected `WANDB_RUN_ID`; `PHASESWEEP_RUN_NAME` remains available as the human-readable display name.
- When writing a `json_envelope` directly, copy `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID`, and `PHASESWEEP_OVERRIDES_SHA256` into it. `report_objective(...)` fills these fields automatically. PhaseSweep verifies all three before accepting the objective.

The trial environment starts with the ambient variables selected by `execution.inherit_env`, then top-level `env` overrides them. Every trial then receives `PHASESWEEP_TRIAL_DIR`, `PHASESWEEP_TRIAL_ID`, `PHASESWEEP_PHASE`, `PHASESWEEP_RUN_NAME`, `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID`, and `PHASESWEEP_OVERRIDES_SHA256`, overriding same-named values. A `json_envelope` trial additionally receives `PHASESWEEP_OBJECTIVE_PATH`, the absolute destination resolved from its configured trial-relative `path`; other extractor types remove any ambient value with that name. The digest covers the exact PhaseSweep-written overrides artifact used by the current override format. `WANDB_RUN_ID` is set to the attempt ID so W&B lookup uses an immutable identity. GPU leasing may override CUDA visibility; the [GPU runtime contract](runtime.md#concurrency-model) defines discovery, disable sentinels, locking, and narrowed-environment behavior.

Metric extractor failures, non-finite metrics, nonzero exits, and missing objective or constraint evidence fail the trial. Gate failures follow the separate [evidence gate](#evidence-gates) policy. Constraint bound violations are different: they produce completed but infeasible trials. PhaseSweep records their raw objective values and constraint readings, but feasibility is applied during winner selection rather than sampler guidance. Winner selection takes the best-metric feasible completed trial; ordering is exact, and only metric values exactly equal to the best value resolve to the lowest trial number. PhaseSweep applies no tolerance band, because it cannot know your objective's meaningful resolution - an absolute epsilon would reorder objectives whose natural scale sits below it. When a swept key has no measurable effect, trials that land on the same value therefore resolve to the lowest-numbered one's choice, which is not evidence of a preference; if your objective is noisy, treat near-equal winners as a tie yourself rather than expecting the selector to.

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

Direct envelope writers copy the generation ID, attempt ID, and overrides digest from the reserved trial environment values listed in the [config reference](config_reference.yaml). The objective name, split, and evaluation policy must match the extractor config. The checkpoint must be a nonempty identity, the step must be a non-negative integer, and the objective value must be a finite JSON number rather than a string or boolean. Configured `checkpoint` and `expected_step` values are matched exactly.

Python trainers can publish that envelope without reconstructing its managed fields or destination:

```python
from phasesweep import report_objective

report_objective(
    value=eval_loss,
    name="eval_loss",
    split="validation",
    policy="best_checkpoint",
    checkpoint=best_checkpoint,
    step=best_step,
)
```

The helper reads `PHASESWEEP_OBJECTIVE_PATH` and the three attempt-identity variables, then atomically replaces the configured file. Use `extra={"param_bytes": parameter_bytes}` to add top-level values consumed by constraint extractors or evidence gates. Non-Python trainers can write the same envelope directly.

Shell-based trainers can invoke the same writer without importing Python code:

```bash
phasesweep report-objective 0.123 \
  --name val_loss --split validation --policy final_checkpoint \
  --checkpoint final.pt --step 1000
```

PhaseSweep does not infer what "best" or "final" means inside a trainer. `best_checkpoint` should report the metric and checkpoint selected by the trainer's declared selection rule. `final_checkpoint` should report an evaluation of the final checkpoint, not merely the last periodic metric that happened to be logged. Log-regex `min`/`max` selects the best matching observation without proving that its weights were saved; `last` selects the last matching line. The W&B extractor reads one terminal summary key and does not scan history, so that key must already contain the intended best or final value.

## Override order

Within one trial, later layers override earlier layers:

1. Inherited winners' `effective_overrides`.
2. Contract `fixed_overrides`.
3. Phase `fixed_overrides`.
4. Sampled values from `search_space`.

A child phase may intentionally reset an inherited key with `fixed_overrides`. A sampled key cannot also be fixed or inherited.

## Extractors

Extractors turn trial evidence into finite floats. JSON and log extractors read files from the generation- and attempt-scoped `{trial_dir}`. Primary metrics from local JSON must use `json_envelope`, which binds the result to the current attempt, resolved overrides, objective, split, and evaluation policy. Every envelope must declare a checkpoint and step; their values are bound only when the extractor config declares `checkpoint` or `expected_step`. Plain `json` remains available for constraints; its selected value must be a number, not a numeric string or boolean. Plain JSON constraints are attempt-location-scoped by the unique trial directory, but their contents do not echo or cross-check the attempt identity, so trainers must write current-attempt evidence rather than copy an artifact from another trial. W&B extractors use the immutable run ID assigned through `WANDB_RUN_ID`; human-readable display names do not participate in evidence correlation.

For agent-facing artifact boundaries, see the [MCP security model](mcp.md#security-model).

The [config reference](config_reference.yaml) defines each extractor shape, including JSON keys, log capture groups, W&B terminal-state handling, and polling timeouts.

W&B extractors and gates require the optional dependency in the active environment:

```bash
pip install "phasesweep[wandb] @ git+https://github.com/pszemraj/phasesweep.git"
```

## Evidence gates

Evidence gates validate local artifacts or W&B summary values after extraction. Local file gates share the trial directory's attempt-location scoping but do not parse an identity envelope; a stale or copied artifact can therefore satisfy a gate if the trainer places it in the current trial directory. Gate failures mark the trial `FAIL` unless that phase has a promotion with `requires_gates: false`, where they are advisory evidence. A suite study's promotion is applied after its component experiment finishes, so suite-level `requires_gates: false` does not make component gates advisory; a failed gate has already failed the trial. The [config reference](config_reference.yaml) defines the available gate shapes.

`json_equals` is type-strict, so `true`, `1`, and `1.0` are distinct. Use `json_scalar_bound` for numeric comparisons where integer and float representations should both pass.

## Promotion

Promotion decides whether a phase or suite study winner is exposed downstream. The comparison uses signed improvement: for `minimize`, improvement is `baseline.metric - candidate.metric`; for `maximize`, it is `candidate.metric - baseline.metric`. The candidate promotes when improvement is at least the configured threshold and required gates passed.

For a phase promotion failure, `stop` raises an error, `skip` ends the remaining phases and permits a partial experiment summary, and `continue_baseline` exposes a clone of the baseline winner. Every evaluated phase promotion writes `<phase>/promotion.yaml`; an exposed candidate or baseline clone is written to `winner.yaml` and included in `summary.yaml`. Winner records keep the exposure phase separate from `winner_source`, which identifies the concrete source phase and trial; promotion metadata retains the rejected candidate. A promotion baseline is also a top-up dependency: once the promoted phase has a study, PhaseSweep refuses to top up the phase named by `min_delta_vs` (see [fingerprints and resume](runtime.md#fingerprints-and-resume)).

## Suites

Suites run studies sequentially in declaration order. `depends_on` requires a prior study to have produced an exposed result; it does not pass winner overrides into the dependent study. Each study compiles to a normal experiment named `<suite>__<study>`, using defaults from `suite.defaults` when the study omits a field. Study `env` values merge over default `env`; study `contracts` merge over default `contracts`; study `provenance` replaces default `provenance` when supplied, and explicit `null` clears it.

This shape compares two independently tuned optimizers under one metric contract and exposes the baseline when the candidate does not improve it:

```yaml
suite: optimizer_comparison
defaults:
  workdir: ./runs
  trial_command: "python train.py {overrides}"
  metric:
    name: val_loss
    goal: minimize
    extractor: {type: log_regex, pattern: 'val_loss=(?P<value>[0-9.eE+-]+)'}
studies:
  - name: adamw
    phases:
      - name: learning_rate
        n_trials: 8
        fixed_overrides: {optimizer: adamw}
        search_space:
          lr: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
  - name: sgd
    depends_on: [adamw]
    promotion: {min_delta_vs: adamw, min_delta: 0.0, on_fail: continue_baseline}
    phases:
      - name: learning_rate
        n_trials: 8
        fixed_overrides: {optimizer: sgd}
        search_space:
          lr: {type: float, low: 1.0e-4, high: 1.0, log: true}
```

Suite-level `run.log` and the compatibility projection `suite_summary.yaml` use `suite.defaults.workdir`; each compiled study writes its normal experiment artifacts under that study's resolved `workdir`. Every invocation claims an immutable `suite_generations/<id>/` namespace selected through `last_successful_suite_generation.yaml`; the [runtime output contract](runtime.md#output-layout) covers publication integrity and historical reads. `show-winners` prints stored promotion decisions and comments, and labels the result historical when the current compiled suite differs.

Suite promotion `min_delta_vs` may name a prior study or `study.phase`; a bare study name resolves to that study's final exposed phase. The candidate and baseline studies must resolve to identical metric contracts - name, goal, and extractor configuration - so promotion cannot subtract unrelated values. On promotion failure, `stop` aborts the suite, `skip` omits that study and continues until a later dependency requires it, and `continue_baseline` substitutes a clone of the baseline for the study's final winner. Suite decisions live in the suite-generation summary, not in a per-study `promotion.yaml`; `show-winners` never substitutes raw candidate winners from the compiled experiments. `timeout_seconds_per_run` applies independently to each compiled study and resets before the next one; there is no suite-wide deadline. `--from-phase` supports only single-experiment configs.

## Upgrading existing configs

Config models are strict (`extra="forbid"`), so most config changes below fail at load instead of warning: `phasesweep validate <config>` reports them before any trial runs and exits `2` with the failing file named and no traceback (see [exit taxonomy](runtime.md#validation-and-dry-run)). Artifact-only changes are labeled separately. Fix config failures in the order listed; the objective-extractor change is the only one that also requires trainer code.

### `metric.extractor` no longer accepts `type: json`

```text
metric.extractor
  Input tag 'json' found using 'type' does not match any of the expected tags: 'json_envelope', 'log_regex', 'wandb'
```

A primary objective read from a local JSON file must now use `json_envelope`. Plain `json` is unchanged for `constraints`, which are not the objective and do not carry attempt identity.

**This is not a YAML-only edit.** `json_envelope` requires the trainer to publish the [result envelope](#result-envelope), echoing `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID`, and `PHASESWEEP_OVERRIDES_SHA256` from its environment; PhaseSweep verifies all three plus the objective name, split, and evaluation policy before accepting a value. Update the trainer first, then the config:

```yaml
# before
metric:
  extractor: {type: json, path: result.json, key: val_loss}

# after
metric:
  extractor:
    type: json_envelope
    path: result.json
    objective_name: val_loss
    split: validation
    policy: final_checkpoint
```

[examples/tiny_decoder_enwik8/run_trial.py](../examples/tiny_decoder_enwik8/run_trial.py) is a worked trainer-side implementation. If you cannot change the trainer, `log_regex` remains available for an objective at the weakest [objective-evidence assurance tier](mcp.md#objective-evidence-assurance).

### `provenance` is required for persistent storage

```text
Value error, Persistent storage requires a nonempty provenance mapping that identifies
the trainer, data, and dependency revision used by this experiment.
```

Add at least one nonempty [provenance](#experiment-keys) entry identifying inputs outside the YAML. Every entry participates in the phase fingerprint.

### RDB storage requires an explicit single-host acknowledgement

```text
Value error, The configured storage resolves to backend 'postgresql', a shared relational
store. ... Set allow_external_rdb_single_host: true ...
```

Add `allow_external_rdb_single_host: true` only for the [single-host RDB contract](runtime.md#concurrency-model). Otherwise use Journal storage for same-host parallel work or SQLite for sequential work.

### Persistent storage now requires a seeded, acknowledged sampler

```text
Value error, Phase 'lr': sampler.type='tpe' with persistent storage (...) requires an
explicit sampler.seed. ...
Value error, Phase 'lr': sampler.type='tpe' with persistent storage (...) requires
sampler.acknowledge_nonresumable: true. ...
```

Apply the [persistent-storage sampler contract](#sampler-capability-on-persistent-storage): seed stochastic samplers and acknowledge TPE/CMA-ES restart limits. The acknowledgement is run-control, so adding it does not invalidate an existing study.

### Fingerprints now include the execution contract

Current fingerprints include the [trainer execution context](#experiment-keys). Populated studies created under earlier schemas fail their next resume or top-up:

```text
StudyFingerprintMismatchError: Study 'exp::phase' was created with a different phase config ...
```

Finish or archive in-flight studies under the old release, or use a new experiment name. Earlier published generations remain readable but report config drift rather than adopting the new execution identity.

### Categorical `choices` must be pairwise unequal

```text
phases.0.search_space.x.CategoricalParam.choices
  Value error, categorical choices must remain distinguishable after Optuna persistence;
  1.0 at index 1 compares equal to 1 at index 0.
```

Remove values that compare equal under the [categorical-choice rule](#search-parameters), including equal values of different types, or that render identically in the chosen CLI override format. Use `json_file` when distinct JSON types such as `1` and `"1"` are intentional. Adjust a fresh grid's `n_trials` to the new cardinality; use a new experiment name or storage for a populated study whose choices must change. Float grids that collapse after canonical rounding likewise need a coarser step or a different parameterization.

### JSON file override validation

```text
Value error, Phase 'p': override_format='json_file' but fixed_overrides key 'cutoff'
holds a value the overrides.json serializer cannot encode (type date) ...
```

The strict [JSON override validation](#override-formats) now runs while loading the config. Quote YAML-native dates, datetimes, or times that the trainer should receive as text; replace non-finite numbers with finite values.

```yaml
# before - datetime.date, not a string
fixed_overrides:
  cutoff: 2024-01-01

# after
fixed_overrides:
  cutoff: "2024-01-01"
```

### `whole_node` phases require an explicit device set

```text
gpu_policy='whole_node' requires an explicit gpu_ids or gpu_devices list
```

Add an explicit `gpu_ids` or `gpu_devices` list. Its size is the trainer's world size and joins the fingerprint; see the [`whole_node` runtime contract](runtime.md#concurrency-model).

### W&B `run_name_template` is removed, and `timeout_seconds` has a floor of 1

```text
metric.extractor.wandb.run_name_template
  Extra inputs are not permitted
metric.extractor.wandb.timeout_seconds
  Input should be greater than or equal to 1
metric.extractor.wandb.timeout_seconds
  Input should be a finite number
```

Both changes apply to the `wandb` extractor and `wandb_summary_required` gate. Delete `run_name_template`; evidence uses the injected immutable `WANDB_RUN_ID`, while trainers may still use `PHASESWEEP_RUN_NAME` for display. Set finite polling values and raise `timeout_seconds` to at least `1`.

### Suite and study names may no longer contain `__`

```text
suite
  Value error, Suite name 'nightly__sweep' must not contain '__': the compiled
  component experiment is named '<suite>__<study>', so a double underscore inside
  either part makes two different suite/study pairs share one artifact namespace,
  study identity, and fingerprint.
```

Rename `__` inside suite or study names to `-` or a single `_`. The double underscore separates the two parts of the compiled `<suite>__<study>` experiment identity.

### Suite promotion requires the candidate and baseline to resolve to the same metric contract

```text
Value error, Study 'candidate' promotion against 'baseline' requires the same
resolved metric contract, but the candidate resolves to {'name': 'loss', ...}
and the baseline resolves to {'name': 'loss', ...}. Put the shared metric in
suite.defaults or make both study metrics identical.
```

Move the shared metric to `suite.defaults.metric` or make both study metrics identical. The error prints each fully resolved metric so the differing name, goal, or extractor field is visible; the [suite promotion contract](#suites) explains why they must match.
