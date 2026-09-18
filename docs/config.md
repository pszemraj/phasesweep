# Configuration guide

A PhaseSweep configuration is one operator-authored YAML document for one
ordered `Experiment`. It contains the trainer's base configuration, an
optimization metric, and a sequence of phases. PhaseSweep materializes trial
inputs, runs the trainer, reads local evidence, and carries chosen values into
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

The root `workdir` owns generation records, phase directories, and trial
evidence. Reuse it only with the same supported-format local ledger and
semantic experiment. Do not try to move, rebind, adopt, or repair an older
namespace with this release; use 0.3.1 for the old namespace and choose fresh
state for this release.

## Phase composition

Phases execute in declaration order. `inherits` names earlier phases whose
winner values become fixed inputs for the child. Omitting `inherits` never
adds an implicit parent.

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

PhaseSweep retains two input boundaries:

- `yaml_file` is the default. It copies `trainer_config`, applies inherited,
  fixed, and sampled dotted overrides, writes a complete YAML file in each
  trial directory, and requires `{config_path}` in `trial_command`.
- `argparse` is for a trainer that accepts command-line options. It requires
  `{overrides}` and renders shell-safe `--key=value` tokens. It accepts null,
  booleans, integers, finite floats, strings, and lists of those values.

`trainer_config` is consumed only by `yaml_file`; a nonempty mapping with
`argparse` is rejected rather than ignored. The removed `json_file` and
`hydra` input formats are not accepted. JSON result envelopes remain an
objective-evidence format; they are unrelated to trainer input.

## Objective evidence, constraints, and gates

The metric is a finite scalar extracted after the trainer exits successfully.
Use one of these local objective extractors:

- `json_envelope` reads an attempt-bound versioned envelope from a
  trial-relative path. Python trainers can call `phasesweep.report_objective`
  to write it to the managed `PHASESWEEP_OBJECTIVE_PATH`.
- `log_regex` reads a trial-relative log file and selects a numeric named
  `(?P<value>...)` match.

`constraints` use the local `json` scalar extractor. A constraint violation is
a completed but infeasible trial, so it cannot win. Phase `gates` are local
post-trial checks: `required_file`, `json_equals`, `json_scalar_bound`,
`artifact_size`, and `sha256`. A failed gate fails the trial. There is no
advisory gate mode or baseline substitution.

All evidence paths are relative to a trial directory. Objective extraction,
gate failure, a non-finite metric, or a nonzero trainer exit fails the attempt.
When no feasible eligible trial remains, the phase fails; PhaseSweep does not
invent a carried baseline.

## Trainer contract

The trainer runs in `execution.cwd` when configured, otherwise the invocation
directory. It receives the configured environment plus PhaseSweep-managed
attempt identity, evidence-path, trainer-input, and GPU variables. Use
`execution.inherit_env` to choose ambient variables and
`execution.passthrough_env` for rotating credentials or transport settings.
Top-level `env` values remain semantic configuration.

Generic environment values such as `WANDB_API_KEY`, `WANDB_PROJECT`, and
`WANDB_MODE` may still be passed to a trainer. They do not configure a
PhaseSweep remote objective or gate: the trainer must report its objective
through one of the local extractors above.

For a typical YAML trainer, the contract is simply:

```python
from phasesweep import report_objective

# Train and evaluate using the YAML received at --config, then:
report_objective(
    value=validation_loss,
    objective_name="validation_loss",
    split="validation",
    policy="final",
    checkpoint="final.pt",
    step=1000,
)
```

`report_objective` obtains the attempt identity and output path from the
managed environment. Do not construct those values yourself.

## Review and run

```bash
phasesweep validate experiment.yaml
phasesweep run experiment.yaml --dry-run
phasesweep run experiment.yaml
phasesweep status experiment.yaml
phasesweep show-winners experiment.yaml
```

`validate`, `status`, and `show-winners` never launch trials. `--from-phase`
can resume a supported experiment only when earlier phases already have valid
winners. See [runtime behavior](runtime.md) for locks, storage, GPU leasing,
publication, and recovery rules.
