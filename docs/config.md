# Configuration guide

A PhaseSweep configuration is one operator-authored YAML document for one
ordered `Experiment`. It contains the trainer's base configuration, an
optimization metric, and a sequence of phases. PhaseSweep materializes trial
inputs, runs the trainer, reads its configured scalar, and carries chosen values into
later phases.

Run `phasesweep validate <config>` before launching. The loader rejects
duplicate YAML mapping keys and unsupported fields before it creates run
artifacts. A top-level `suite` is not a configuration variant: it is rejected.
Run formerly independent components as separate full experiment YAML files and
review their defaults explicitly.

Field types, defaults, and validation constraints are in
[config_reference.yaml](config_reference.yaml).

## Experiment YAML

The root mapping names one experiment and provides its storage, trainer
boundary, metric, and ordered `phases` list. `experiment` is used in study
names, artifact paths, and lock identity, so it must be a safe, stable name.

```yaml
experiment: model_search
storage: auto
workdir: ./runs
provenance:
  trainer: trainer-revision-42
  data: dataset-2026-09
trial_command: "python train.py --config {config_path}"
trainer_config:
  model: {depth: 8}
  optimizer: {lr: 0.0003}
metric:
  name: validation_loss
  goal: minimize
  extractor:
    type: json_envelope
    objective_name: validation_loss
    split: validation
    policy: final
phases:
  - name: depth
    n_trials: 3
    sampler: {type: grid}
    search_space:
      model.depth: {type: categorical, choices: [6, 8, 10]}
  - name: learning_rate
    inherits: [depth]
    n_trials: 4
    sampler: {type: random, seed: 7}
    search_space:
      optimizer.lr: {type: float, low: 0.00001, high: 0.001, log: true}
```

`storage: null` (or an omitted storage value) is an in-memory disposable run.
Use `sqlite:///...` for a persistent sequential local ledger and
`journal:///...` for same-host parallel trials. `storage: auto` selects
`study.db` in `<workdir>/<experiment>/` when every phase has `n_jobs: 1`, or
`study.journal` there otherwise. Persistent storage requires a nonempty
`provenance` mapping. External database URLs are unsupported and rejected
before connection or artifact creation.

See [runtime behavior](runtime.md) for artifact-root binding, current-format
state, and execution behavior.

## Phase composition

Phases execute in declaration order. `inherits` names earlier phases whose
winner values become fixed inputs for the child. Omitting `inherits` never
adds an implicit parent.

Inheritance transfers parameters, not weights or checkpoints. Each trainer
invocation owns initialization and any explicitly configured checkpoint loading.

Every exported override key has an origin. A diamond is valid when both parent
paths carry the same original key. If independent parents export the same key,
the child must resolve it with an explicit `fixed_overrides` value; that child
value becomes the new origin for descendants. A child may intentionally fix a
key it inherits, but may not put an inherited key in `search_space`.

`fixed_overrides` and `search_space` are local alternatives: one key cannot
appear in both. Dotted keys use the same namespace everywhere, so a key and
its prefix cannot coexist (for example, `optimizer` and `optimizer.lr`).
Phase names must be unique and can inherit only from earlier declared phases.

```yaml
phases:
  - name: architecture
    n_trials: 2
    sampler: {type: grid}
    search_space:
      model.depth: {type: categorical, choices: [6, 8]}
  - name: optimizer_a
    inherits: [architecture]
    n_trials: 2
    sampler: {type: grid}
    fixed_overrides: {optimizer.family: adamw}
    search_space:
      optimizer.lr: {type: categorical, choices: [0.0001, 0.0003]}
  - name: optimizer_b
    inherits: [architecture]
    n_trials: 2
    sampler: {type: random, seed: 8}
    fixed_overrides: {optimizer.family: sgd}
  - name: resolved_comparison
    inherits: [optimizer_a, optimizer_b]
    n_trials: 2
    sampler: {type: random, seed: 9}
    fixed_overrides: {optimizer.family: adamw}
```

Here the final phase resolves the independently-originated `optimizer.family`
value. It still receives the same-origin `model.depth` without a duplicate
declaration.

## Search spaces and samplers

`search_space` maps dotted trainer-config paths to `float`, `int`, or
`categorical` parameter objects. Float and integer steps must land exactly on
their upper bound. Numeric integer bounds are limited to Optuna's exactly
representable range. Categorical choices must be distinct under Python
equality; seed keys are rejected by default unless `allow_seed_search: true`
makes a variance audit explicit.

YAML decides scalar types before PhaseSweep reads fixed overrides or categorical
choices. In this loader, bare `1e-3` is a string, while `1.0e-3` and `0.001`
are floats. PhaseSweep preserves that distinction so string categories remain
usable. Write a decimal value (or an exponent with a decimal mantissa) for a
numeric override, and quote a value when you intend a string category.

Supported sampler types are `grid`, `random`, `tpe`, and `cmaes`. A grid has
at most 4,096 concrete combinations, counted before trials are materialized.
Its trial target must equal that cardinality unless `allow_partial_grid: true`
is set. CMA-ES accepts numeric spaces only.

For persistent storage, `random`, `tpe`, and `cmaes` need a seed. TPE and
CMA-ES also require `acknowledge_nonresumable: true`: their target must finish
in one invocation rather than being resumed or enlarged across processes.
Grid and seeded random phases support ordinary local continuation.

`n_trials` counts terminal Optuna attempts, not only successful objectives.
`max_consecutive_failures`, trial/phase/run timeouts, and
`allow_incomplete_on_timeout` control how failures and partial work are
handled. They do not create an alternate winner.

## Trainer inputs

PhaseSweep supports four input boundaries:

- `yaml_file` is the default. It copies `trainer_config`, applies inherited,
  fixed, and sampled dotted overrides, writes a complete YAML file in each
  trial directory, and requires `{config_path}` in `trial_command`.
- `argparse` is for a trainer that accepts command-line options. It requires
  `{overrides}` and renders shell-safe `--key=value` tokens. It accepts null,
  booleans, integers, finite floats, strings, and lists of those values.
- `hydra` renders native `key=value` tokens through `{overrides}`. Numbers,
  booleans, null, strings, and recursively supported lists preserve their types.
  Hydra grammar quoting is separate from shell quoting. Control characters and
  interpolation-like `${...}` strings are rejected before launch rather than
  interpreted as configuration expressions. Rendering does not import Hydra.
- `json_file` writes nested overrides-only JSON to `overrides.json` and requires
  `{overrides_path}`. Dotted keys expand into nested objects. Values must be
  strictly JSON-representable, with string object keys and finite numbers.

`trainer_config` is consumed only by `yaml_file`; a nonempty mapping with
another format is rejected. Incompatible placeholders are rejected too. A
constant command may omit its override placeholder when every phase's composed
overrides are empty. Flat `overrides_resolved.json` remains the audit record;
JSON input is the separate nested file. YAML input contains the complete base
configuration plus overrides. All phases' compositions are validated before
launch, including dotted-key collisions and choices indistinguishable on the
trainer wire.

## Objective evidence, constraints, and gates

The metric is a finite scalar extracted after the trainer exits successfully.
Use the reader that matches the trainer's existing reporting:

- `json` reads a finite JSON number at a saved trial-relative path and dotted
  key, for example `{type: json, path: result.json, key: eval.loss}`.
- `wandb` reads an exact scalar key such as `eval/loss` from the finished run
  whose ID is the immutable trial attempt ID. It requires the optional
  `wandb` extra and an authorized entity/project.
- `json_envelope` reads an attempt-bound versioned envelope from a
  trial-relative path. Python trainers can call `phasesweep.report_objective`
  to write it to the managed `PHASESWEEP_OBJECTIVE_PATH`.
- `log_regex` reads a trial-relative log file and selects a numeric named
  `(?P<value>...)` match; `select: last` remains the default.

`constraints` use the same scalar readers. A valid measured constraint violation
is a completed but infeasible trial; missing or invalid evidence fails the trial.
Phase `gates` include `required_file`, `json_equals`, `json_scalar_bound`,
`artifact_size`, `sha256`, and `wandb_summary_required`. A failed gate fails the trial. There is no
advisory gate mode or baseline substitution.

All local evidence paths are relative to the fresh attempt directory; there is
no latest-file discovery or fallback to a shared result. Objective extraction,
gate failure, a non-finite metric, or a nonzero trainer exit fails the attempt.
When no feasible eligible trial remains, the phase fails; PhaseSweep does not
invent a carried baseline.

These scoring decisions are separate:

- `metric.goal` ranks the scalar selected for each trial.
- `log_regex.select` chooses an observation within that trial (`last`,
  `first`, `min`, or `max`).
- W&B selects an exact finished-summary key; ordinary JSON selects a saved
  path and dotted key.
- Envelope `policy` and `expected_step` validate reported metadata. They do
  not cause evaluation or checkpoint loading.

The trainer owns evaluation cadence, early stopping, checkpoint selection,
and custom aggregation. PhaseSweep has one experiment-level metric and does
not prune training. `validate` and `run --dry-run` display the configured
source, key/path/pattern, applicable reduction, and ranking goal.

### W&B evidence

```yaml
metric:
  name: validation_loss
  goal: minimize
  extractor:
    type: wandb
    entity: YOUR_ENTITY
    project: YOUR_PROJECT
    metric_key: eval/loss
    poll_seconds: 2
    timeout_seconds: 120
```

The primary objective, W&B constraints, and each phase's W&B gates must agree
on normalized endpoint, entity, project, poll interval, and timeout. They share
one finished-summary capture. Only requested numeric values and gate-presence
evidence are saved. A gate-only configuration accepts the first finished
summary and fails if required keys are missing. Failed, crashed, killed, or
preempted remote runs cannot win.

PhaseSweep supplies `WANDB_BASE_URL`, `WANDB_ENTITY`, `WANDB_PROJECT`,
`WANDB_RUN_ID`, and `WANDB_RESUME=never`. Trainers must honor those values and
finish their run. Explicit conflicting identity settings and offline/disabled
modes are rejected before training; ambient identity defaults are normalized.
An ordinary `wandb.init()`, scalar log/summary assignment, and `finish()` are
sufficient; no PhaseSweep import or local objective mirror is needed.

The reader receives the trainer's composed authentication and transport
environment. Use `execution.passthrough_env` for rotating environment credentials
or proxies, including under `inherit_env: none`. Excluded parent credentials
are not reacquired. Supported SDK account authentication also works; an
API-key environment variable is not mandatory. Credentials are never serialized
in worker requests. Syntax validation and dry runs do not authenticate.

Polling runs after trainer cleanup and GPU release, under one deadline that
includes startup, SDK calls, retries, and summary visibility, capped by phase
and experiment budgets. Published evidence is frozen: selection, replay, CLI,
and MCP reads need no SDK or remote reread, even after remote edits/deletion.
W&B's source is keyed by attempt ID; it makes none of the envelope's evaluation
metadata or trainer-input-content assurances.

## Trainer contract

The trainer runs in `execution.cwd` when configured, otherwise the invocation
directory. It receives the configured environment plus PhaseSweep-managed
attempt identity, evidence-path, trainer-input, and GPU variables. Use
`execution.inherit_env` to choose ambient variables and
`execution.passthrough_env` for rotating credentials or transport settings.
Top-level `env` values remain semantic configuration.

When no W&B evidence consumer is configured, trainer-owned `WANDB_*` settings
are preserved unchanged.

For a trainer that chooses the optional envelope contract:

```python
from phasesweep import report_objective

# Train and evaluate using the YAML received at --config, then:
report_objective(
    value=validation_loss,
    name="validation_loss",
    split="validation",
    policy="final",
    checkpoint="final.pt",
    step=1000,
)
```

`report_objective` obtains the attempt identity and output path from the
managed environment. Do not construct those values yourself.

Trainers that cannot import the Python helper can publish the same envelope
from the managed trial environment:

```bash
phasesweep report-objective 0.123 --name validation_loss --split validation \
  --policy final --checkpoint final.pt --step 1000
```
