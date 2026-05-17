# Config Reference

A config is either one experiment or a suite of studies. A single experiment runs ordered phases. A suite runs multiple isolated experiments under one run plan.

## Experiment Keys

- `experiment`: required name used for Optuna study names, output paths, and lock identity. It must match `[A-Za-z0-9_-]+`.
- `storage`: Optuna storage URL. `sqlite:///path.db` is resumable for `n_jobs == 1`; SQLite plus parallel jobs is rejected. `journal:///path.journal` supports same-host parallel jobs. RDB URLs such as `postgresql://...` pass through to Optuna. `null` is in-memory and non-resumable.
- `workdir`: output root. Trial artifacts live under `<workdir>/<experiment>/<phase>/`.
- `trial_command`: shell template. Supported placeholders are `{overrides}`, `{overrides_path}`, `{trial_dir}`, `{trial_id}`, `{phase}`, and `{run_name}`.
- `override_format`: `argparse` by default. Also supports `hydra` and `json_file`. Values are shell-quoted before substitution.
- `metric`: objective name, `goal` (`minimize` or `maximize`), and extractor.
- `constraints`: optional extracted scalars with `min` and/or `max`. Violating trials are recorded but cannot win.
- `contracts`: named bundles of immutable `fixed_overrides` plus optional evidence gates.
- `timeout_seconds_per_run`: wallclock guard for the whole experiment.
- `phases`: ordered phase list.

## Phase Keys

- `name`: required phase name matching `[A-Za-z0-9_-]+`.
- `inherits`: prior phase names whose exposed winners become fixed overrides.
- `fixed_overrides`: hard-coded overrides for every trial in the phase.
- `contracts`: top-level contracts applied to the phase. Contract keys cannot be resampled or locally overridden.
- `search_space`: override-key to sampler spec. Dotted keys such as `model.depth` are allowed.
- `n_trials`: trial budget. Increasing it later is a compatible top-up.
- `n_jobs`: parallel trials inside the phase.
- `gpu_ids`: explicit CUDA device IDs. When omitted, numeric ambient `CUDA_VISIBLE_DEVICES` or `nvidia-smi` output is auto-detected, including for `n_jobs == 1`.
- `max_consecutive_failures`: abort threshold for consecutive failed or infeasible trials.
- `sampler`: `tpe`, `random`, `grid`, or `cmaes`. `cmaes` is installed with the core package and is useful for continuous numeric phases.
- `timeout_seconds_per_trial`: per-trial process-group timeout. `null` requires `allow_unbounded_trials: true`.
- `timeout_seconds_per_phase`: phase wallclock guard.
- `allow_incomplete_on_timeout`: select from completed trials after a phase or run timeout. Defaults to fail-closed.
- `allow_partial_grid`: permit `n_trials` smaller than the grid cardinality.
- `allow_seed_search`: permit `seed` or `*.seed` in `search_space`.
- `gates`: evidence checks evaluated after metric and constraint extraction.
- `promotion`: compare the phase winner against an earlier exposed winner before downstream use.
- `comment`: design note shown by CLI commands. It is excluded from fingerprints.

## Search Parameters

```yaml
search_space:
  lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  depth: { type: int, low: 4, high: 16, step: 4 }
  activation: { type: categorical, choices: [gelu, relu] }
```

Float and integer bounds must be finite. Categorical choices must be Optuna-compatible scalars. Grid phases require a full grid unless `allow_partial_grid: true`; float grids require `step` and an evenly divisible interval.

## Override Formats

The default command contract is a normal Python CLI:

```yaml
trial_command: "python train.py --out {trial_dir}/result.json {overrides}"
override_format: argparse
```

With `override_format: argparse`, sampled and fixed values render as `--key value` pairs. This is the recommended path for new trainers because it works with ordinary `argparse`, Typer, Click, and most command-line parsers.

For an existing Hydra application, opt in explicitly:

```yaml
trial_command: "python train.py --out {trial_dir}/result.json {overrides}"
override_format: hydra
```

Hydra format renders `key=value` tokens. phasesweep supports it for compatibility with Hydra-based trainers; it is not required for normal use.

For nested configuration or trainers that prefer one structured input file, use JSON:

```yaml
trial_command: "python train.py --out {trial_dir}/result.json --overrides-path {overrides_path}"
override_format: json_file
```

JSON format writes `<trial_dir>/overrides.json` and expands dotted keys into nested objects.

## Override Order

Within one trial, later layers override earlier layers:

1. Inherited winners' `effective_overrides`.
2. Contract `fixed_overrides`.
3. Phase `fixed_overrides`.
4. Sampled values from `search_space`.

A child phase may intentionally reset an inherited key with `fixed_overrides`. A sampled key cannot also be fixed or inherited.

## Extractors

JSON extractors read a dotted key from a file under `{trial_dir}`:

```yaml
extractor: { type: json, path: result.json, key: eval.loss }
```

Log regex extractors read a captured numeric group named `value`:

```yaml
extractor:
  type: log_regex
  file: stdout.log
  pattern: 'eval_loss=(?P<value>[0-9.eE+-]+)'
  select: last
```

W&B extractors poll a completed run summary. The trainer should use `PHASESWEEP_RUN_NAME` or the same `run_name_template`.

```yaml
extractor:
  type: wandb
  entity: my-team
  project: tiny-lm
  run_name_template: '{experiment}-{phase}-{trial_id}'
  metric_key: eval/loss
  timeout_seconds: 300
```

## Evidence Gates

Supported gates are `required_file`, `json_equals`, `json_scalar_bound`, `artifact_size`, `sha256`, and `wandb_summary_required`. Gate failures mark the trial `FAIL` unless promotion has `requires_gates: false`, where they are advisory evidence.

`json_equals` is type-strict, so `true`, `1`, and `1.0` are distinct. Use `json_scalar_bound` for numeric comparisons where integer and float representations should both pass.

`artifact_size` requires `source: file|directory|json`. File and directory sources measure materialized bytes. JSON source reads an integer byte estimate from `path` plus `key`.

## Promotion

Promotion gates whether a candidate winner is exposed downstream.

```yaml
promotion:
  min_delta_vs: baseline
  min_delta: 0.01
  requires_gates: true
  on_fail: stop
```

`on_fail` can be `stop`, `skip`, or `continue_baseline`. Every promotion writes `<phase>/promotion.yaml`; exposed winners include the decision in `winner.yaml` and `summary.yaml`.

## Suites

Suites group independent or dependency-ordered studies:

```yaml
suite: coreamp_ablation
defaults:
  workdir: ./runs
  trial_command: "python train.py --out {trial_dir}/result.json {overrides}"
  metric:
    name: bpb
    goal: minimize
    extractor: { type: json, path: result.json, key: eval.bpb }

studies:
  - name: baseline
    phases:
      - name: lr
        n_trials: 8
        search_space:
          lr: { type: float, low: 1.0e-5, high: 1.0e-3, log: true }

  - name: candidate
    depends_on: [baseline]
    promotion: { min_delta_vs: baseline, min_delta: 0.01, on_fail: stop }
    phases:
      - name: lr
        n_trials: 8
        search_space:
          lr: { type: float, low: 1.0e-5, high: 1.0e-3, log: true }
```

Each study compiles to an experiment named `<suite>__<study>`. Suite runs write `<workdir>/<suite>/run.log` and `suite_summary.yaml`. Suite promotion `min_delta_vs` may name a prior study or `study.phase`; a bare study name resolves to that study's final phase.

## Validation

Config-load validation rejects graph errors, duplicate YAML keys, unsafe override keys, dotted-key prefix collisions, sampler/search-space mismatches, SQLite parallel writes, seed searches by default, partial grids by default, missing override placeholders, non-finite bounds, and unknown contracts. Resume checks reject stale studies and incompatible persisted winners.
