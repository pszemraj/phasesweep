# Config guide

A PhaseSweep config is one operator-authored YAML containing the trainer's base configuration and the sweep plan. The orchestrator chooses parameter values, materializes a complete per-trial trainer YAML, manages trial directories, extracts evidence, and decides which winner is exposed downstream. Your trainer reads that YAML, runs the experiment, and provides the evidence that configured extractors read.

Field types, defaults, accepted values, and validation constraints are listed in [config_reference.yaml](config_reference.yaml).

## Experiment keys

The top level of a single experiment describes identity, storage, the trainer boundary, the objective, and the ordered phase plan. `experiment` is used in study names, output paths, and same-host lock identity, not only for display.

`storage` holds Optuna study state. Use in-memory storage for disposable runs, SQLite for sequential persistent runs, Journal storage for same-host parallel work, or an external RDB under the explicit single-host contract. `workdir` holds trial logs, result artifacts, winners, promotion decisions, and summaries. Persistent studies are bound to their resolved artifact root; use the [workdir rebind procedure](runtime.md#fingerprints-and-resume) after moving a tree. The exact storage forms and concurrency constraints are in the [config reference](config_reference.yaml).

`trainer_config` is the trainer's ordinary base configuration, embedded directly in the PhaseSweep file. In the default `yaml_file` mode, PhaseSweep copies it for each trial, applies inherited, contract, fixed, and sampled dotted-path values, writes `<trial_dir>/trainer_config.yaml`, and exposes its shell-quoted path as `{config_path}`. Changing this mapping changes the experiment and phase fingerprints.

String values inside `trainer_config` may contain `{trial_dir}`, `{trial_id}`, `{phase}`, or `{run_name}`. PhaseSweep expands those runtime placeholders before applying overrides, which makes output directories and run labels trial-specific without a wrapper script. Other strings, including sampled or fixed string values, remain literal.

`trial_command` is the command template for one trial. The primary shape is `python train.py {config_path}`, adjusted to the flag or position where your trainer accepts a YAML path. The [config reference](config_reference.yaml) defines every placeholder; compatibility boundaries are explained under [override formats](#override-formats).

`provenance` is the operator-declared identity of inputs outside this YAML that the command string cannot describe, such as the trainer revision, dataset, dependency lock, container, tokenizer, or starting checkpoint. The embedded `trainer_config` is already fingerprinted and does not need a duplicate provenance token. Persistent storage requires at least one nonempty entry. PhaseSweep includes the complete mapping in every phase fingerprint, so change its values whenever any external input changes; use a new experiment name when results from the old and new provenance should remain separate. PhaseSweep does not infer imports or hash arbitrary shell-command inputs.

`metric` defines the objective name, optimization direction, and extractor. `constraints` are additional scalar checks: an infeasible trial remains a completed evaluation but cannot win. `contracts` are named bundles of fixed overrides and gates that phases can reuse for consistent comparisons.

The MCP API reports the configured extractor's guarantees through explicit [objective-evidence assurance fields](mcp.md#objective-evidence-assurance).

`execution` fixes the trainer's working directory and ambient-environment boundary. `inherit_env` selects semantic ambient inputs; `passthrough_env` adds credential or transport values that reach the trainer but may rotate without changing a persistent study's cohort. Top-level `env` values are always semantic and fingerprinted, even when their key is also listed as pass-through. Before a persistent top-up allocates a trial, the current semantic environment digest must match every existing trial. The default `inherit_env: all` therefore treats ordinary ambient churn as meaningful and can refuse a later top-up; for reusable studies, start a new experiment identity with a bounded name list and classify tokens explicitly, for example:

```yaml
execution:
  inherit_env: [DATASET_REV, TOKENIZER_REV]
  passthrough_env: [WANDB_API_KEY, HF_TOKEN]
```

Adding or changing that classification is itself a semantic config edit, so an already-populated study keeps its original contract. Populated legacy studies whose trials have no environment digest are not adopted; archive/delete them or use a new experiment name. The [runtime output contract](runtime.md#output-layout) explains the recorded identity.

For CLI runs, relative workdirs, execution directories, and file-backed storage paths resolve from the invocation directory, not the config file's directory. Relative command paths resolve from the trainer's effective cwd. Run a relative-path config from one stable directory; changing cwd can select a different artifact tree, study, or trainer. MCP runs apply stricter [path and working-directory rules](mcp.md#paths-and-the-working-directory).

## Phase keys

Each phase is one Optuna study in an ordered chain. A phase may inherit winners from earlier phases; those inherited values become locked overrides for the current phase and for descendants. This greedy structure is useful for inspectable staged searches, but it is not a substitute for joint optimization when dimensions interact strongly.

Each phase declares a search space and trial-attempt budget, with optional fixed overrides, inherited winners, contracts, evidence gates, and promotion rules. The [config reference](config_reference.yaml) lists the exact phase fields and constraints. GPU allocation, timeouts, cleanup, and study top-ups are covered in [runtime behavior](runtime.md).

## Search parameters

`search_space` is a mapping from trainer-config path to a typed float, integer, or categorical parameter object. Keys can be dotted paths such as `model.depth`; the same key namespace is used for inherited winners, contracts, fixed overrides, and sampled values. In the default mode these paths update nested values in the generated trainer YAML. PhaseSweep rejects ambiguous compositions such as fixing a parent key while sampling one of its children.

Use categorical parameters for explicit choices and integer or float parameters for ranges. Categorical choices must remain distinct after both Optuna persistence and the selected trainer serialization; for example, equal Python values collapse in Optuna, while some CLI boundaries render a number and the same numeric string identically. Grid sampling is useful when every finite combination should run; CMA-ES is useful for interacting numeric dimensions. The [config reference](config_reference.yaml) defines the exact equality, wire-format, bounds, completeness, sampler, and seed-search rules.

## Sampler capability on persistent storage

Use `storage: auto` to keep the database with the experiment artifacts. It selects
`<workdir>/<experiment>/study.db` for sequential phases, or `study.journal` when
any phase has `n_jobs > 1`, and handles the absolute URL spelling. Auto storage
requires nonempty `provenance` and the sampler contract below. Explicit URLs
retain their current behavior; omitted storage remains in-memory. A backend
change caused by editing `n_jobs` cannot resume the existing artifact tree.

The `sampler` block is optional and defaults to `type: tpe` with no seed, which is fine for an in-memory run. A persistent `storage` changes that, because the study outlives the process that created it, so each phase must state two things up front rather than discover them mid-sweep:

| Sampler | Seed | `acknowledge_nonresumable` |
| --- | --- | --- |
| `grid` | optional (traversal order only) | rejected |
| `random` | required | rejected |
| `tpe`, `cmaes` | required | required (`true`) |

An unseeded `tpe`, `random`, or `cmaes` phase draws a different sequence on every invocation, so the durable trials it accumulates cannot be reproduced or explained afterwards. `tpe` and `cmaes` additionally hold process-local sampler state that Optuna storage does not persist: PhaseSweep refuses to resume such a phase mid-target or to raise its `n_trials` later (see [runtime behavior](runtime.md#fingerprints-and-resume)). Setting `acknowledge_nonresumable: true` is your statement that you accept that contract and will run each target in one invocation; setting it on `grid` or `random`, which resume safely, is rejected as meaningless config.

```yaml
storage: auto
provenance: {revision: my-trainer-and-data-v1}
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
> `yaml_file` is the core workflow. Put the trainer's base config and PhaseSweep's search plan in the same operator-authored YAML, then pass `{config_path}` to the trainer.

The default `yaml_file` mode materializes one complete `<trial_dir>/trainer_config.yaml` from the embedded `trainer_config` for every trial. It preserves nested mappings, lists, strings, booleans, finite numbers, and nulls. Dotted overrides update nested paths, and `{config_path}` is required even when a phase has no overrides so the trainer always receives its base configuration.

| Format | Use when |
| --- | --- |
| `yaml_file` | Default and primary: the trainer accepts one complete YAML config. |
| `argparse` | Compatibility with a trainer that already accepts `--key value` tokens. |
| `json_file` | Compatibility with a trainer that already accepts an overrides-only JSON object. |
| `hydra` | Compatibility with an existing Hydra/OmegaConf entry point. |

The three non-default modes do not consume `trainer_config`; declaring a nonempty mapping with one of them is rejected rather than silently ignored. `argparse` and `hydra` require `{overrides}` and only support their documented scalar/list grammar. `json_file` requires `{overrides_path}` and writes only the composed overrides, not the trainer's complete base config. Hydra support does not make Hydra a dependency or composition layer.

W&B is separate from all four rows: it is an objective/gate evidence backend. A YAML-configured, argparse, JSON, or Hydra trainer may independently use the `wandb` extractor.

Each mode has a required template placeholder and distinct value encoding. The [config reference](config_reference.yaml) defines the full wire contract, including YAML scalars that cannot cross a selected boundary. `yaml_file` and `json_file` validate statically known values with the same serializer used at launch. CLI compatibility modes reject structured or non-finite values that have no faithful command-line representation.

## Trainer contract

The command in `trial_command` is the training or evaluation program for one trial. PhaseSweep creates the trial directory, materializes the selected input boundary, launches the process group, captures stdout/stderr, and then reads evidence. The trial process uses `execution.cwd` when configured. Otherwise it uses the directory where `phasesweep run` was invoked, or the catalog's pinned `cwd` for MCP-launched runs. The trainer must:

- Read the complete YAML at `{config_path}` in the default mode, or parse the explicitly selected compatibility [override format](#override-formats).
- Provide a finite objective through the configured extractor: call `report_objective(...)` or write a compatible JSON envelope, write log evidence under `{trial_dir}`, or make the configured W&B run terminal with the metric in its summary. `report_objective(...)` creates missing parent directories when the envelope uses a nested trial-relative path.
- Exit nonzero when the trial failed and should be recorded as failed.
- When using W&B extraction or gates, let the W&B SDK use the injected `WANDB_RUN_ID`; `PHASESWEEP_RUN_NAME` remains available as the human-readable display name. Configure the evidence source's explicit `base_url`, and point the trainer at the same deployment (for example through fingerprinted top-level `env.WANDB_BASE_URL`).
- When writing a `json_envelope` directly, copy `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID`, and `PHASESWEEP_OVERRIDES_SHA256` into it. `report_objective(...)` fills these fields automatically. PhaseSweep verifies all three before accepting the objective.

PhaseSweep composes the configured environment, then injects trial identity, evidence-path, trainer-input digest, W&B identity, and GPU-isolation values as applicable. The [config reference](config_reference.yaml) lists every reserved variable and its meaning; the [GPU runtime contract](runtime.md#concurrency-model) covers device visibility and locking.

Metric extractor failures, non-finite metrics, nonzero exits, and missing objective or constraint evidence fail the trial. Gate failures follow the separate [evidence gate](#evidence-gates) policy. Constraint bound violations are different: they produce completed but infeasible trials. PhaseSweep records their raw objective values and constraint readings, but feasibility is applied during winner selection rather than sampler guidance. Winner selection takes the best-metric feasible completed trial; ordering is exact, and only metric values exactly equal to the best value resolve to the lowest trial number. PhaseSweep applies no tolerance band, because it cannot know your objective's meaningful resolution - an absolute epsilon would reorder objectives whose natural scale sits below it. When a swept key has no measurable effect, trials that land on the same value therefore resolve to the lowest-numbered one's choice, which is not evidence of a preference; if your objective is noisy, treat near-equal winners as a tie yourself rather than expecting the selector to.

### Result envelope

A `json_envelope` trainer publishes this versioned shape after successful evaluation:

```json
{
  "schema_version": 1,
  "status": "complete",
  "generation_id": "<current generation ID>",
  "attempt_id": "<current attempt ID>",
  "overrides_sha256": "<current PhaseSweep-written trainer-input digest>",
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

Overrides apply from inherited winners through contract values and phase-fixed values to sampled values, with later layers taking precedence. A child may intentionally reset an inherited key, but a sampled key cannot also be fixed or inherited. The [config reference](config_reference.yaml) gives the exact composition and conflict rules.

## Extractors

Extractors turn trial evidence into finite floats. JSON and log extractors read files from the generation- and attempt-scoped `{trial_dir}`. Primary metrics from local JSON must use `json_envelope`, which binds the result to the current attempt, resolved overrides, objective, split, and evaluation policy. Every envelope must declare a checkpoint and step; their values are bound only when the extractor config declares `checkpoint` or `expected_step`. Plain `json` remains available for constraints; its selected value must be a number, not a numeric string or boolean. Plain JSON constraints are attempt-location-scoped by the unique trial directory, but their contents do not echo or cross-check the attempt identity, so trainers must write current-attempt evidence rather than copy an artifact from another trial. W&B extractors use the immutable run ID assigned through `WANDB_RUN_ID` and an explicit, fingerprinted `base_url`; human-readable display names and ambient `WANDB_BASE_URL` do not participate in evidence correlation. Authentication still comes from the W&B SDK environment/configuration, so `WANDB_API_KEY` can rotate through `execution.passthrough_env`.

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

Suites run studies sequentially in declaration order. `depends_on` requires a prior study to have produced an exposed result; it does not pass winner overrides into the dependent study. Each study compiles to a normal experiment named `<suite>__<study>` and inherits omitted values from `suite.defaults`. The [config reference](config_reference.yaml) defines merge, replacement, and explicit-null behavior.

This shape compares two independently tuned optimizers under one metric contract and exposes the baseline when the candidate does not improve it:

```yaml
suite: optimizer_comparison
defaults:
  workdir: ./runs
  trial_command: "python train.py {config_path}"
  trainer_config:
    optimizer:
      name: adamw
      lr: 0.001
  metric:
    name: val_loss
    goal: minimize
    extractor: {type: log_regex, pattern: 'val_loss=(?P<value>[0-9.eE+-]+)'}
studies:
  - name: adamw
    phases:
      - name: learning_rate
        n_trials: 8
        fixed_overrides: {optimizer.name: adamw}
        search_space:
          optimizer.lr: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
  - name: sgd
    depends_on: [adamw]
    promotion: {min_delta_vs: adamw, min_delta: 0.0, on_fail: continue_baseline}
    phases:
      - name: learning_rate
        n_trials: 8
        fixed_overrides: {optimizer.name: sgd}
        search_space:
          optimizer.lr: {type: float, low: 1.0e-4, high: 1.0, log: true}
```

Suite-level `run.log` and the compatibility projection `suite_summary.yaml` use `suite.defaults.workdir`; each compiled study writes its normal experiment artifacts under that study's resolved `workdir`. Every invocation claims an immutable `suite_generations/<id>/` namespace selected through `last_successful_suite_generation.yaml`; the [runtime output contract](runtime.md#output-layout) covers publication integrity and historical reads. `show-winners` prints stored promotion decisions and comments, and labels the result historical when the current compiled suite differs.

Suite promotion `min_delta_vs` may name a prior study or `study.phase`; a bare study name resolves to that study's final exposed phase. The candidate and baseline studies must resolve to identical metric contracts - name, goal, and extractor configuration - so promotion cannot subtract unrelated values. On promotion failure, `stop` aborts the suite, `skip` omits that study and continues until a later dependency requires it, and `continue_baseline` substitutes a clone of the baseline for the study's final winner. Suite decisions live in the suite-generation summary, not in a per-study `promotion.yaml`; `show-winners` never substitutes raw candidate winners from the compiled experiments. `timeout_seconds_per_run` applies independently to each compiled study and resets before the next one; there is no suite-wide deadline. `--from-phase` supports only single-experiment configs.

## Updating older development configs

PhaseSweep is alpha software and does not retain deprecated config aliases. Run `phasesweep validate <config>` before launching; strict validation names obsolete or malformed fields and exits `2` without starting trials.

Bring older configs to the current contracts rather than mixing them into populated studies. Common updates are moving the trainer's base settings into `trainer_config` with a `{config_path}` command, using `json_envelope` for local objective evidence, declaring provenance for persistent storage, seeding persistent samplers, acknowledging TPE or CMA-ES restart limits, and declaring devices for `whole_node` phases. Accepted fields and constraints are listed in the [config reference](config_reference.yaml).

Changes to the trainer boundary, base trainer config, execution context, search space, or evidence contract change fingerprints. Use a new experiment identity when the existing storage contains trials under an incompatible contract. Earlier published generations remain readable and report config drift.
