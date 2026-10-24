# Configuration guide

A PhaseSweep configuration is one operator-authored YAML document for one ordered `Experiment`. It contains the trainer's base configuration, an optimization metric, and a sequence of phases. PhaseSweep materializes trial inputs, runs the trainer, reads its configured scalar, and carries chosen values into later phases.

Run `phasesweep validate <config>` before launching. The loader rejects duplicate YAML mapping keys and unsupported fields before it creates run artifacts.

Field types, defaults, and validation constraints are in [config_reference.yaml](../src/phasesweep/config_reference.yaml), which `phasesweep explain` prints key by key; [reference lookups](#reference-lookups) describes the available queries. It is a signature reference, not runnable YAML; start with a [template](#templates) written by `phasesweep init`, or with [example.yaml](../examples/example.yaml) for one local CPU toy-trainer attempt. Because relative paths resolve against the YAML file's directory, the same file writes the same artifact tree from any directory, through the CLI or MCP; an `Experiment` built in Python has no file and resolves them against the process working directory. `validate`, `run`, `status`, `show-winners` and `sync-tracking` accept the same `--set KEY=VALUE` overrides, with the same path resolution. Use the same overrides to run and inspect a study. MCP's separate catalog paths are documented in the [MCP operator guide](mcp.md#the-catalog).

## Reference lookups

The source reference links to the current development documentation. `phasesweep explain` and MCP `explain_config` pin those links to the installed release tag or source revision, including editable installs; reinstall an editable checkout to refresh its version metadata.

`phasesweep explain` lists the top-level keys; `phasesweep explain --notes` prints file-wide conventions. Use `phasesweep explain KEY` for a key's types, defaults and constraints, including its nested fields and the shared rules that apply to it. Configuration errors suggest relevant lookups, and an unknown key suggests the closest documented names. MCP offers the same reference through [`explain_config`](mcp.md#tool-workflow).

`phasesweep explain phases` lists the immediate phase fields compactly. Narrow it with a path such as `phasesweep explain 'phases[].sampler'`; `phases.sampler` and `phases.0.sampler` are equivalent. Dotted trainer keys also resolve, for example `phasesweep explain 'phases[].search_space.optimizer.lr'`. Quote keys containing brackets so the shell does not expand them.

## Templates

`phasesweep init --template NAME` writes a commented, runnable starting point. `phasesweep init --list` prints the names and purposes below, `-o PATH` names the output file, or a directory that receives `NAME.yaml`: an existing directory, or a path ending in `/`, which is created if needed. An existing file is never overwritten. Every template runs as written on the packaged toy trainer, so `phasesweep init --template NAME && phasesweep run NAME.yaml` finishes in seconds on a CPU-only machine. Comments in each file mark what to replace for a real trainer: the command, the `trainer_config` keys and the metric extractor. Each file opens with a pointer to `phasesweep explain`, and sets `execution.cwd: .` so a run from another directory resumes the same study.

| Template | Purpose | Command |
| --- | --- | --- |
| `starter` | Two chained phases, depth then learning rate, on the toy trainer. The default. | `phasesweep init --template starter` |
| `argparse` | A trainer that takes `--key=value` flags: `override_format: argparse`, `trainer_config` as the base, a seeded random learning-rate search. | `phasesweep init --template argparse` |
| `architectures` | One phase per architecture, none inheriting another, each fixing its model keys and running a small learning-rate grid, so the winners are compared by hand; [handing the best on](#comparing-options-and-handing-the-best-on) needs one categorical key in a single phase. | `phasesweep init --template architectures` |
| `chained` | A seeded random learning-rate search, then a phase that inherits it and searches a key the first phase only fixed. | `phasesweep init --template chained` |
| `noisy` | A noisy objective: a grid of eight log-spaced learning rates with one trial each, so neighbors show the trend and the scatter around it, a `metric.max` viability bound and an `observe` entry; [Noisy objectives and small budgets](#noisy-objectives-and-small-budgets) explains each. | `phasesweep init --template noisy` |

## Experiment YAML

The root mapping names one experiment and provides its storage, trainer boundary, metric, and ordered `phases` list. `experiment` is used in study names, artifact paths, and lock identity, so it must be a safe, stable name.

```yaml
experiment: model_search
storage: auto
workdir: ./runs
provenance:
  data: dataset-2026-09
  torch: "2.9.1"
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

Choose `storage: auto` or a `journal:///...` URL for persistent local work, with a nonempty `provenance` mapping. Change a `provenance` value whenever an input outside the YAML changes, such as the dataset revision or a dependency version; each generation already records the trainer's [git commit](runtime.md#output-layout) but never compares it, so a trainer code change does not refuse a resume. It is part of every phase's study fingerprint, so an existing study refuses the changed config: set a new `experiment` name to run it as a new study (with `storage: auto` a new `workdir` also works), and the old study stays readable under the old name. `storage: null` keeps no ledger, but its run still binds `<workdir>/<experiment>` to in-memory storage, so a later persistent run with the same `experiment` and `workdir` is refused; give a `storage: null` run its own `experiment` name or `workdir`. See [storage and locks](runtime.md#storage-and-locks) for path resolution, artifact-root binding, and continuation behavior.

## Phase composition

Phases execute in declaration order. `inherits` names earlier phases whose winner values become fixed inputs for the child. Omitting `inherits` never adds an implicit parent.

Inheritance transfers parameters, not weights or checkpoints. Each trainer invocation owns initialization and any explicitly configured checkpoint loading.

Every exported override key has an origin. A diamond is valid when both parent paths carry the same original key. If independent parents export the same key, the child must resolve it with an explicit `fixed_overrides` value, or by making it the phase's `repeats.key`; that child value becomes the new origin for descendants. A child may intentionally fix a key it inherits. It may also search a key its parents only fixed, and the sampled value replaces the inherited one; it may not search a key an inherited phase already sampled, because a winner's sampled value is not searched again.

`phasesweep explain phases` prints the load-time rules on which keys a phase may fix, search or inherit, such as that `optimizer` and `optimizer.lr` cannot coexist, and `phasesweep explain 'phases[].name'` the naming rules.

A list of fixed runs is one phase per configuration with `fixed_overrides`, an empty `search_space`, no `repeats` and `n_trials: 1`. Such a phase draws nothing from its sampler and keeps at most one COMPLETE or INFEASIBLE result. A larger target can only [retry failed attempts](runtime.md#retry-fixed-configurations-and-grids); to measure one configuration several times, use [repeats](#repeats).

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
    n_trials: 1
    fixed_overrides: {optimizer.family: sgd}
  - name: resolved_comparison
    inherits: [optimizer_a, optimizer_b]
    n_trials: 1
    fixed_overrides: {optimizer.family: adamw}
```

Each edge below lists the keys a child receives and, in parentheses, the phase where each key originated:

```mermaid
flowchart TD
    arch["architecture<br/>samples model.depth"]
    a["optimizer_a<br/>fixes optimizer.family: adamw<br/>samples optimizer.lr"]
    b["optimizer_b<br/>fixes optimizer.family: sgd"]
    r["resolved_comparison<br/>fixes optimizer.family: adamw"]
    arch -->|"model.depth (architecture)"| a
    arch -->|"model.depth (architecture)"| b
    a -->|"model.depth (architecture)<br/>optimizer.family (optimizer_a)<br/>optimizer.lr (optimizer_a)"| r
    b -->|"model.depth (architecture)<br/>optimizer.family (optimizer_b)"| r
```

`model.depth` reaches the final phase along both paths with the same origin, so the diamond is valid and needs no duplicate declaration. `optimizer.family` arrives from two independent origins, so validation refuses the child until it fixes that key; its `adamw` then becomes the origin that descendants inherit.

### Comparing options and handing the best on

Phases that do not inherit from each other are independent comparisons: each searches on its own and publishes its own winner, and you read the winners side by side with `phasesweep show-winners`. Nothing carries the best of them into a later phase, because a later phase inherits from named phases and not from a ranking across them. To hand the best of several options to later phases, search the options as one categorical key in a single phase, for example `model_name_or_path: {type: categorical, choices: [...]}`, so that phase publishes one winner and its children inherit the chosen value. Options that need different fixed keys cannot share one phase, since a phase has one set of `fixed_overrides` and the categorical varies one key. For those, run one phase per option and choose between them by hand, as the `architectures` template does.

### Stopping between stages

No flag stops a run after a phase; the stopping point is the config file. Run a config that holds only the phases of the first stage, read the winners with `phasesweep show-winners`, and decide. Then append the later phases to the same file and run it again with the same `experiment`, `workdir` and storage. The finished phases replay their published winners without launching a trial, and only the appended phases run; an appended phase that inherits a finished one runs with that winner's values. `phasesweep run --dry-run` on the extended file shows it first: which phases replay and what each appended phase would launch. Leave the finished phases as they are. Raising a finished phase's `n_trials` while a later phase that inherits it already holds trials is refused, and so is an edit to a finished phase's study identity, which names the changed field: its own fields, such as its sampler seed or search space, and the `trial_command`, `trainer_config`, `override_format`, `env`, `execution`, `provenance`, `metric`, `constraints` and `mlflow` target every phase shares. Unless `execution.cwd` is set, run from the [same directory](#trainer-contract). The dry run refuses the `n_trials` raise and compares study identities once inherited winners are real values; [runtime behavior](runtime.md#inspection-commands) lists what it checks. The storage must be persistent, because `storage: null` keeps nothing to replay. [`--from-phase NAME`](runtime.md#timeouts-cleanup-and-continuation) is the opposite tool: it starts a run at a later phase and never stops early.

### Run-time overrides

Derive a smoke run without copying or editing its YAML:

```bash
phasesweep init --template argparse -o smoke-source.yaml
phasesweep run smoke-source.yaml --dry-run --set workdir=/tmp/phasesweep-smoke \
  --set experiment=smoke --set phases.0.n_trials=1
```

`--set KEY=VALUE` is repeatable on `validate`, `run`, `status`, `show-winners` and `sync-tracking`; `mcp init-catalog --from` reads the config file as written. Values follow the config file's strict YAML rules (duplicate keys and invalid UTF-8 fail) and [YAML 1.1 types](#search-spaces-and-samplers), including booleans, lists, mappings and `null`, so `--set trainer_config.optimizer.lr=1e-3` hands the trainer the string `1e-3`. Later assignments to the same key win. A whole `phases` list can also be replaced; normal inheritance and sampler rules still apply. The final derived config is validated before execution and recorded in the `config.snapshot.yaml` of the [generation](runtime.md#timeouts-cleanup-and-continuation) the run writes, if it writes one. The source YAML stays unchanged. A raised `n_trials` becomes the phase's accepted target on persistent storage, so a later run without that `--set` is refused as a lower target, and recovery changes belong in the YAML.

A `KEY` is a path of steps:

| Step | Meaning | Example |
| --- | --- | --- |
| `name`, joined by `.` | A mapping key without dots, brackets or `=` | `env.NOTE`, `execution.cwd` |
| `.N` or `[N]` | The list element at zero-based position `N` | `phases.0.n_trials`, `phases[0].n_trials` |
| `[NAME]` | The list element whose `name` is `NAME`, such as a phase | `phases[learning_rate].n_trials` |
| `[KEY]` | The mapping key `KEY` taken verbatim, so it may contain dots, brackets or `=` | `phases[learning_rate].search_space[model.n_layers].choices` |

Inside `[...]`, write `\]` for a literal `]` and `\\` for a literal `\`. On a list, digits inside brackets are a position, so a phase whose name is all digits is addressed by position. A malformed bracket, an unknown name, an index past the end, a non-numeric dotted step on a list (`phases.lr`) and a step into a scalar or `null` value are refused when `--set` is read. These path errors quote the argument's key, name the failing step and repeat this syntax. Quote the argument in the shell, because `[` and `]` are glob characters.

A mapping key that the config does not have is created, so an absent optional block can be filled in and `trainer_config` takes any new key. If validation then fails at or below a key an assignment created, which is what a misspelled schema key produces, the error adds a note naming that `--set` argument's key and the key it created. A block written as `null` is not absent: replace it whole, as in `--set 'phases[NAME].repeats={key: seed, values: [0, 1]}'`.

Paths inside `--set` retain the YAML file's directory as their base, including `--set workdir=...`. Pass the same assignments to `validate`, `run`, `status`, `show-winners` and `sync-tracking`. A read that finds no study directory fails and names the path it searched. Changing `workdir` alone does not redirect an explicit journal in `storage`; use `--set storage=auto` or a fresh journal path for a derived run. Semantic edits cannot reuse a populated study; choose a fresh experiment/output namespace. Apply `--dry-run` to the same overrides before launching.

## Search spaces and samplers

`search_space` maps dotted trainer-config paths to `float`, `int`, or `categorical` parameter objects. A key whose last dotted component is `seed`, `data_seed` or `random_state` is refused: a searched seed lets seed noise win as if it were a configuration. Fix it in `fixed_overrides` or `trainer_config`. To compare finalists across seeds, use [repeats](#repeats).

> [!IMPORTANT]
> YAML decides scalar types before PhaseSweep reads fixed overrides or categorical choices. In this loader, bare `1e-3` and `1.0e3` are strings, while `1.0e-3`, `1.0e+3` and `0.001` are floats. PhaseSweep preserves that distinction so string categories remain usable. Write a decimal value (or an exponent with a decimal point and a sign) for a numeric override, and quote a value when you intend a string category. Bare `yes`, `no`, `on` and `off` likewise load as booleans and `~` as null, with no warning, so quote them as string choices. Schema numeric fields such as `low`, `high` and `step` accept `1e-5`; the typing matters only for values passed to the trainer unchanged: `trainer_config`, `fixed_overrides` and categorical `choices`. `validate` and `run --dry-run` warn about each categorical key that has a string choice Python reads as a finite float, such as `1e-4`, and name the phase, the key, the values and the spelling that loads as a number; the warning is advisory and stays for a quoted string that is meant as text. Text containing a colon followed by a space, such as `comment: 'compare: learning rates'`, also needs quotes; parse errors name the config file and line.

Use `grid` when every listed value must run: it visits configurations in a shuffle fixed by `sampler.seed`, rather than the listed choice order, so a partial or deadline-truncated search covers a spread of the space. Each configuration's repeat values run at consecutive draw positions in their listed order. `random`, seeded or not, draws independently with replacement: three trials over three categorical values cover all three only about 2 times in 9. TPE fits an adaptive model, and CMA-ES accepts numeric spaces only. See `phasesweep explain 'phases[].sampler'` for limits and [Noisy objectives and small budgets](#noisy-objectives-and-small-budgets) for choosing a sampler. Persistent grids can [retry failed cells by raising their target](runtime.md#retry-fixed-configurations-and-grids).

Persistent stochastic searches need `sampler.seed`; [Interrupted trials rerun](runtime.md#interrupted-trials-rerun) explains draw positions and reproducibility. Grid selection advises when a winner reaches a numeric boundary or cells remain unmeasured; [grid recovery](runtime.md#retry-fixed-configurations-and-grids) describes these diagnostics.

Parallel TPE and CMA-ES suggestions depend on trial completion order, even with `sampler.seed`; `validate` and previews warn when `n_jobs` exceeds one. Set a seed and use one worker when reproducible adaptive draws matter.

### Repeats

Most phases run one trial per configuration; [Noisy objectives and small budgets](#noisy-objectives-and-small-budgets) explains when `repeats` is worth its cost.

`repeats` runs every configuration once per value of one key, typically the trainer seed, and ranks configurations by the mean or median of their members instead of by a single trial. The sampler draws the repeat value as one more axis, so `N` configurations with `R` values cost `N x R` trials, and a fresh `n_trials` must cover every repeat: the number of values for an empty `search_space`, a multiple of it under seeded random, and configurations times values under grid. A persistent grid target above that matrix size can retry failed members, including a grid [seed audit](#noisy-objectives-and-small-budgets) with an empty `search_space`; each retry replaces its failed member in the group and leaves completed members alone. Only `grid` and seeded `random` support repeats, and `allow_partial_grid` is refused with it.

A comparison of three architecture finalists, ranked by their mean over three seeds, then a single-seed learning-rate search on the winner (nine trials for the comparison, then eight):

```yaml
experiment: architecture_finalists
storage: auto
workdir: ./runs
provenance:
  data: dataset-2026-09
trial_command: "python train.py --config {config_path}"
metric:
  name: validation_loss
  goal: minimize
  min: 0.0001
  extractor: {type: json, path: result.json, key: validation_loss}
phases:
  - name: architecture
    n_trials: 9
    sampler: {type: grid}
    search_space:
      model.name: {type: categorical, choices: [small, medium, wide]}
    repeats: {key: seed, values: [0, 1, 2], reduce: mean}
  - name: learning_rate
    inherits: [architecture]
    n_trials: 8
    sampler: {type: random, seed: 7}
    fixed_overrides: {seed: 0}
    search_space:
      optimizer.lr: {type: float, low: 0.00001, high: 0.001, log: true}
```

Choose the sampler by what you can afford. Grid runs every configuration with every value and is the honest choice for a small discrete space. Seeded random runs `n_trials / R` configurations, each drawn once. Both samplers run a configuration's values at `R` consecutive [draw positions](runtime.md#interrupted-trials-rerun), so with `n_jobs: 1` a phase that stops at a deadline leaves at most one group half-run, and with more workers at most the groups that were in flight. Grid shuffles the configuration order with `sampler.seed` and keeps repeat values in their listed order.

Set `metric.min` and `metric.max` so one collapsed seed disqualifies its configuration instead of dragging its mean. When comparing multiple configurations, size `max_consecutive_failures` above the number of repeat values, or one configuration that fails every repeat can abort the phase; `validate` and the dry run warn when it is not. A phase measuring only one configuration needs no such warning: any failed member already prevents its group from ranking. See [repeat groups](runtime.md#repeat-groups) for which groups rank, member accounting, the `status --trials` group summary, and the winner's reduced metric, spread and observations. The winner's params and effective overrides omit the repeat key, so a later phase that inherits it runs its own repeats, a fixed seed, or the trainer's default.

For a mean-repeats comparison, an optional phase `cost_key` names a numeric sampled parameter whose smaller values cost less, for example `cost_key: num_train_epochs`. The advisory identifies the cheapest complete feasible configuration within the best group's standard error, `sd_best / sqrt(n_best)`, when it is cheaper than the winner. Equal costs prefer the better objective, then the earlier trial order. Selection logs the advisory, and `show-winners` recomputes it from recorded groups, including when `cost_key` was added after publication. MCP captures it in each run's result snapshot, including a no-work replay; existing terminal snapshots stay frozen. Result reads recompute it only when the current selection semantics match the producing config. This advisory never changes the selected winner or inherited values. The separate `tie` warning uses two combined standard errors. `cost_key` is excluded from study fingerprints and can be added without invalidating recorded trials.

## Noisy objectives and small budgets

When one run's metric moves with the seed, or a trial is expensive enough that a phase affords a handful of them, start with more distinct values, one trial each, and read the trend across them. `phasesweep init --template noisy` writes a runnable config that does this with eight learning rates on the packaged toy trainer, plus a `metric.max` bound and an `observe` entry.

- **Spend the budget on more distinct values, then read the trend.** For a continuous hyperparameter such as the learning rate, weight decay or warmup, a trial on a new value tells you more than a second trial on an old one. Neighboring values act as replicates: run-to-run noise shows as scatter around a trend. Look for its best region according to `metric.goal`: a trough for `minimize` or a peak for `maximize`. Use a dense log-spaced `grid` (list the values; eight learning rates from 1e-4 to 1e-3 are about 1.39 times apart) or more seeded `random` draws over a `log: true` range, one trial per value. `phasesweep status CONFIG --trials` and `trials.csv` list every trial with its params, so sort by the key and look at the curve. PhaseSweep publishes the best single trial and does no smoothing: a winner that is an isolated spike among clearly worse neighbors is likely noise, and the trend across its neighbors is the better guide to which value to take.
- **Use `repeats` where more values cannot help.** `R` seeds multiply a phase's cost by `R`, so keep [repeats](#repeats) for a comparison of a few discrete finalists, such as architectures or another categorical choice, where no ordering lets neighbors borrow strength and the gap between options is close to the noise; for reporting a final configuration's spread; and whenever you want that measurement itself. The standard error of a mean is `sd` divided by the square root of the number of seeds, so three seeds cut the noise on a ranked value by about 1.7 and nine by 3. That is why more seeds rarely resolve a small gap: two configurations with per-run sd `s` and `R` seeds each are separated by the tie check only when their gap exceeds `2*s*sqrt(2/R)`, which is the same as `R` above `8*s^2/gap^2`. With `s = 0.012` and a gap of 0.005 that is 46.08, so 47 seeds per configuration, and a budget of three resolves gaps of about 0.020 and larger. The check uses each group's own sample sd. When the gap is inside the noise, treat the options as tied and decide on cost; the [tie check](runtime.md#confound-verdicts) flags it but never holds back a phase that inherits the winner. To decide a tie before inheriting phases run, use [Stopping between stages](#stopping-between-stages). Deciding on an `observe`d value turns that value into a selection signal, which is exactly what `observe` keeps out of ranking, so when it is a held-out set decide on cost instead.
- **A seed audit is an optional one-off.** A phase with an empty `search_space` and [`repeats`](#repeats) runs the unchanged baseline once per seed, so `n_trials` equals the number of seeds. Run one when the size of the noise is unknown and matters to a decision, such as how large a gap you can trust; the scatter of a dense sweep already shows the noise for most searches. Its `WINNER` line in `run.log` gives the reduced metric (the mean by default), the sample standard deviation `sd` and the range over the seeds. `phasesweep show-winners CONFIG` prints `n`, `mean`, `sd`, `min` and `max` in its concise `repeats` summary; the full winner record keeps those statistics in `repeats.spread`. Compare them with the gain you hope to see: a setting that moves the metric by less than the range is not separated from the baseline by one run. The audit's winner has empty params, so it hands nothing to a later phase; the numbers are the product.
- **Use seeded `random` over a `log: true` range when the budget is small.** `sampler: {type: random, seed: 11}` over a `float` or `int` parameter with `log: true` (which needs `low` above 0 and no `step`, apart from an `int`'s default of 1; `grid` rejects it) spreads draws evenly across orders of magnitude, and more draws give a clearer trend. Seeded random draws [reproduce across restarts and worker counts](runtime.md#interrupted-trials-rerun).
- **Size `n_startup_trials` to the budget when you use `tpe`.** TPE samples its first `n_startup_trials` trials (default 10) at random and fits its model only afterwards. Only COMPLETE and INFEASIBLE trials count toward that number, so failed attempts lengthen the random stretch. A `tpe` phase whose `n_trials` does not exceed it never fits a model, and `validate` and `run --dry-run` print a warning saying so; use seeded `random`, which samples the same way, or set `n_startup_trials` below `n_trials`. A `tpe` or `cmaes` phase learns from single trials (it cannot take `repeats`), so pin the trainer seed in `fixed_overrides` and reserve these samplers for spaces whose effect is large against the run-to-run spread, which the scatter of a dense sweep or a seed audit shows.
- **Let `timeout_seconds_per_phase` set the budget.** `n_trials` stays a required target, but with `timeout_seconds_per_phase` and `allow_incomplete_on_timeout: true` a phase ends at the deadline and publishes the winner its finished trials earned, marked `incomplete`; without the flag the run stops as a timeout. Set `n_trials` to a ceiling you would accept running in full and let the deadline be the real limit, since a persistent phase cannot lower its target later. [Timeouts and continuation](runtime.md#timeouts-cleanup-and-continuation) covers the trials a deadline interrupts, the 72-hour per-trial default, and replays of a partial result.
- **Keep a collapsed run out of the ranking with `metric.min` and `metric.max`.** A trial outside the [bounds](#objective-evidence-constraints-and-gates) is `INFEASIBLE`: it cannot win, but it counts toward `max_consecutive_failures`, so a bound set inside the ordinary spread can abort the phase. Derive the bound from the spread of ordinary trials, and set it before the sweep: it is part of the metric's identity, so changing it starts a new study.
- **Record held-out values with `observe:` so they cannot rank.** An [`observe`](#report-only-observations) entry records a value such as a test score without changing feasibility, as a `constraints` entry on it would; a trainer can write the value beside its objective with [`report_objective(..., extra=...)`](#reporting-an-objective).

A time-boxed dense search over a learning-rate range is seeded random with one trial per draw and the trainer seed pinned. It ends at the deadline with whatever it finished, and `phasesweep status CONFIG --trials` shows the trend across the draws:

```yaml
phases:
  - name: learning_rate
    n_trials: 40
    timeout_seconds_per_phase: 7200
    allow_incomplete_on_timeout: true
    sampler: {type: random, seed: 11}
    fixed_overrides: {seed: 1}
    search_space:
      optimizer.lr: {type: float, low: 0.00001, high: 0.001, log: true}
```

## Trainer inputs

Set the top-level `override_format` to select the boundary your trainer accepts. PhaseSweep supports four input boundaries:

- `yaml_file`, the default, passes a complete YAML configuration through `{config_path}`.
- `argparse` renders flattened settings as `--key=value` tokens through `{overrides}`. Put a valueless flag such as `--do_train` directly in the command; a plain `store_true` flag rejects `--key=true`.
- `hydra` renders native `key=value` tokens through `{overrides}`, preserving scalar and list types with Hydra grammar quoting separate from shell quoting.
- `json_file` passes a complete nested JSON configuration through `{overrides_path}`.

Every format starts from `trainer_config` and applies the phase's inherited, fixed and sampled overrides at their dotted paths. Keep trial-specific paths such as `output_dir: "{trial_dir}/out"` in `trainer_config`: placeholders in fixed overrides or categorical choices remain literal. `phasesweep explain trainer_config`, `phasesweep explain override_format` and `phasesweep explain trial_command` describe accepted values, placeholder requirements and rendering. The [output layout](runtime.md#output-layout) describes generated files and the recorded base and effective overrides.

### Hugging Face argparse recipe

This uses Hugging Face's [Transformers v4.57.3 `run_glue.py`](https://github.com/huggingface/transformers/blob/v4.57.3/examples/pytorch/text-classification/run_glue.py) without a wrapper trainer. Use that script with its matching Transformers library and [example dependencies](https://github.com/huggingface/transformers/blob/v4.57.3/examples/pytorch/text-classification/requirements.txt), replace `execution.cwd` with your checkout, and follow [source-checkout preflight](#source-checkout-preflight). The script's `--do_train` and `--do_eval` flags stay in the command; scalar settings become argparse options.

Save this as `hf_sst2.yaml`. It schedules four fine-tunes at four distinct learning rates with one fixed trainer seed, ranked by validation loss. `n_jobs: 1` runs them sequentially, using the default single-GPU policy when a GPU is available.

```yaml
experiment: hf_sst2
storage: auto
workdir: ./runs
provenance:
  trainer: transformers-v4.57.3-run_glue
execution:
  cwd: /absolute/path/to/transformers
trial_command: >-
  python examples/pytorch/text-classification/run_glue.py
  --do_train --do_eval {overrides}
override_format: argparse
trainer_config:
  model_name_or_path: distilbert/distilbert-base-uncased
  task_name: sst2
  output_dir: "{trial_dir}/hf"
  report_to: none
  max_seq_length: 128
  per_device_train_batch_size: 16
  num_train_epochs: 3
metric:
  name: eval_loss
  goal: minimize
  extractor: {type: json, path: hf/eval_results.json, key: eval_loss}
observe:
  - name: accuracy
    extractor: {type: json, path: hf/eval_results.json, key: eval_accuracy}
  - name: train_runtime
    extractor: {type: json, path: hf/train_results.json, key: train_runtime}
phases:
  - name: learning_rate
    n_trials: 4
    n_jobs: 1
    sampler: {type: grid}
    fixed_overrides: {seed: 11}
    search_space:
      learning_rate: {type: categorical, choices: [0.00001, 0.00002, 0.00005, 0.0001]}
```

The script saves `eval_results.json` and `train_results.json` under its `output_dir`, so these extractor paths read beneath `{trial_dir}/hf`. Accuracy and training runtime are report-only observations; the loss decides the winner. This JSON objective has `evaluation_point: n/a`; use the [`json_envelope` reporting contract](#reporting-an-objective) when the evaluated checkpoint and progress must be recorded.

Review `phasesweep validate hf_sst2.yaml` and `phasesweep run hf_sst2.yaml --dry-run` before launching `phasesweep run hf_sst2.yaml`. Model and dataset downloads and the four training runs belong to that launch.

## Objective evidence, constraints, and gates

The metric is a finite scalar extracted after the trainer exits successfully. Use the reader that matches the trainer's existing reporting:

- `json` reads a finite JSON number at a saved trial-relative path and dotted key, for example `{type: json, path: result.json, key: eval.loss}`.
- `wandb` reads an exact scalar key such as `eval/loss` from the finished run whose ID is the immutable trial attempt ID. It requires the optional `wandb` extra and an authorized entity/project.
- `mlflow` reads native metric history from the bound finished run. It requires the optional `mlflow` extra, an existing experiment in the top-level `mlflow` block, and an explicit `select: last | min | max | step`. See the [MLflow guide](mlflow.md) for ownership, selection, and result delivery.
- `json_envelope` reads an attempt-bound versioned envelope from a trial-relative path. Python trainers can call `phasesweep.report_objective` to write it to the managed `PHASESWEEP_OBJECTIVE_PATH` (the [call, its file and its behavior outside a trial](#reporting-an-objective)). It is the extractor that binds the evaluation policy, checkpoint and step, so use it when the checkpoint behind a value matters, as with Transformers `load_best_model_at_end` (see the `evaluation_point` note below).
- `log_regex` reads a trial-relative log file (`file`, default `stdout.log`) and selects a numeric named `(?P<value>...)` match; `select` defaults to `last`. Python `logging` writes to stderr by default, so a metric logged that way needs `file: stderr.log`.

`constraints` use the same scalar readers. A valid measured constraint violation is an `INFEASIBLE` trial: it finished with its measured values but cannot win; its outcome and `run.log` reason name each violated metric or constraint bound and the measured value. Missing or invalid evidence fails the trial. Phase `gates` include `required_file`, `json_equals`, `json_scalar_bound`, `artifact_size`, `sha256`, `wandb_summary_required`, and `mlflow_required`. `mlflow_required` separately names `metrics`, `params`, and `tags` from the same bound run. A failed gate fails the trial. There is no advisory gate mode or baseline substitution.

`metric.min` and `metric.max` are optional inclusive viability bounds on the metric itself, and either may be set alone. A trial whose metric falls outside them is infeasible exactly like a violated constraint: it finishes with its real value (the sampler still learns from it), counts toward `max_consecutive_failures` and as `infeasible` in the `survivorship` verdict, and selection re-checks the bounds from the stored value. A bound refuses a degenerate score, such as a loss of exactly 0 from a leaked evaluation set; it does not steer the search. When the phase reaches its target with no feasible trial, the `NoFeasibleTrialError` names the bound or bounds and the best value among the `COMPLETE` and `INFEASIBLE` trials with its trial number; a streak abort of `INFEASIBLE` trials says only that they were infeasible, and the `reason` column of `phasesweep status CONFIG --trials` names the bound each one violated. The bounds are part of the metric's identity, so changing one starts a new study.

All local evidence paths are relative to the fresh attempt directory; there is no latest-file discovery or fallback to a shared result. A failed objective extraction, a failed gate, a non-finite metric or a nonzero trainer exit fails the attempt. When no feasible eligible trial remains, the phase fails; PhaseSweep does not invent a carried baseline.

These scoring decisions are separate:

- `metric.goal` ranks the scalar selected for each trial.
- `metric.min` and `metric.max` decide whether that scalar is viable; they never rank.
- `log_regex.select` chooses an observation within that trial (`last`, `first`, `min`, or `max`).
- W&B selects an exact finished-summary key; ordinary JSON selects a saved path and dotted key.
- MLflow uses an explicit [native history selector](mlflow.md#select-metrics-and-require-evidence).
- Envelope `policy` and `expected_step` validate reported metadata; they do not cause evaluation or checkpoint loading. For envelope and W&B objectives, `evaluation_axis` names the training-progress axis the sweep holds fixed; selection's `evaluation_point` confound check compares ranked objectives only on it (see [runtime behavior](runtime.md#confound-verdicts)). It is part of the extractor config, so changing it starts a new study, like `checkpoint`/`expected_step`; constraint extractors may not set it. `json_envelope` defaults to `step`, the envelope's own evaluation step, and any other name must be reported in `evaluation.progress` (via `report_objective(..., progress={...})`); `wandb` defaults to `_step`, W&B's own history step.

For example, in a fixed-token batch-size sweep each trial's optimizer `step` count depends on its batch size, so trials differ by design in both `step` and any step-named checkpoint. Declaring `evaluation_axis: tokens` and reporting `progress={"tokens": total_tokens}` compares trials on tokens instead: the check stays `ok` when every trial reached the token budget, and flags one that stopped short. The check's 1% tolerance absorbs each batch size stopping within a batch of the budget. Likewise, a trainer that reports its best checkpoint reports a step that differs between trials by nature; report how far it trained on a declared axis instead, such as `progress={"trained_steps": total_steps}` with `evaluation_axis: trained_steps`.

Only `json_envelope`, `wandb` and `mlflow` objectives record a position, so `evaluation_point` is `n/a` for `json` and `log_regex`; `validate` and `run --dry-run` print an `evaluation_point check:` line that says whether the configured extractor supports it. With W&B or MLflow the position is where the metric was logged; it does not verify checkpoint identity. Under Transformers `load_best_model_at_end=True`, `train()` reloads the best checkpoint's weights, so a final evaluation scores that checkpoint, yet its logged position reads as the trainer's final global step for every trial. For remote objectives, `validate` and `run --dry-run` warn when `trainer_config`, a phase's `fixed_overrides` or `trial_command` explicitly enables `load_best_model_at_end` (including dotted paths, and in the command `--load_best_model_at_end`, `--load_best_model_at_end=true` or a Hydra `training.load_best_model_at_end=true`), naming where it was found. If the evaluated checkpoint matters, report through `json_envelope` with `policy`, `checkpoint` (for Transformers, `trainer.state.best_model_checkpoint`) and the evaluated `step`; report total training progress in `evaluation.progress` and select that axis with `evaluation_axis`. These are trainer-reported metadata, not proof that the trainer loaded those weights.

The trainer owns evaluation cadence, early stopping, checkpoint selection, and custom aggregation. PhaseSweep has one experiment-level metric and does not prune training. `validate` and `run --dry-run` display the configured source, key/path/pattern, applicable reduction, ranking goal, and viability bounds when set.

### Report-only observations

`observe` lists extra scalars to record from each accepted trial without affecting feasibility or selection. Each entry is a `name` and an `extractor`, and the extractor is `json`, `json_envelope` or `log_regex`: a `wandb` or `mlflow` extractor is refused at load, because remote readers require every requested key, so a value the trainer never wrote would fail the trial. `validate` and `run --dry-run` print each observation's source, so a mistyped path or key shows before a run. An observation is read after the trial is accepted, so a read that fails or yields a non-finite value logs a warning, records nothing for that name, and leaves the trial untouched. Recorded values appear as `user_attr:observed:<name>` columns in `trials.csv`, as `observed` in the selected winner's `winner.yaml` and in MCP results, and as `phasesweep.observed.<name>` metrics on the [MLflow](mlflow.md#result-delivery-and-recovery) attempt run. Observations never enter the study fingerprint, so editing `observe` does not invalidate a study. Changes apply to newly executed trials; replay does not backfill observations on recorded trials. `phasesweep explain observe` gives the naming and extractor rules.

```yaml
observe:
  - name: peak_memory_gb
    extractor: {type: json, path: result.json, key: peak_memory_gb}
```

### Training-length comparison

A phase that searches epochs or `max_steps` produces different final steps by design, and comparing those raw steps correctly flags `evaluation_point`. To compare whether trials reached their own planned endpoint, report a relative completion axis with the `json_envelope` extractor:

```yaml
metric:
  name: eval_loss
  goal: minimize
  extractor:
    type: json_envelope
    path: result.json
    objective_name: eval_loss
    split: validation
    policy: final
    evaluation_axis: schedule_fraction
```

After evaluating the final checkpoint, report the trainer's measured `eval_loss`, checkpoint path, and actual training progress:

```python
from phasesweep import report_objective

report_objective(
    eval_loss,
    name="eval_loss",
    split="validation",
    policy="final",
    checkpoint=checkpoint_path,
    step=actual_steps,
    progress={"schedule_fraction": actual_steps / planned_steps},
)
```

`planned_steps` is the positive optimizer-step target resolved for that trial before training; `actual_steps` is the number of training steps it completed, including when early stopping cuts training short. Trials finishing 100/100 and 200/200 both report `1.0`, so this check stays `ok`; a trial ending at 160/200 reports `0.8` and is flagged against a finished trial. This axis compares relative completion; the trials can still use different data or compute budgets. A heterogeneous verdict appears in MCP's `confound.flagged` list.

With the `wandb` reader, keep that reader and set `metric.extractor.evaluation_axis: schedule_fraction`. Log `schedule_fraction: actual_steps / planned_steps` in the same `wandb.log()` call as the objective so the reader can bind both to the same history row. With `mlflow`, the evaluation axis is the selected metric observation's integer `step`; log a common scaled relative-completion step, for example `int(10000 * actual_steps / planned_steps)`, alongside that objective. Retain lower positions for early stops. MCP results include the axis and a remedy for the reader in use.

### W&B evidence

Install the optional SDK with `pip install "phasesweep[wandb]"`.

To rank configurations by their mean over seeds, use `phases[].repeats: {key: seed, values: [0, 1, 2], reduce: mean}`. PhaseSweep launches one trial and one managed W&B run per seed, reads that run's objective, and ranks complete configuration groups by their mean. The trainer runs one seed per invocation; it does not need to average several tracked runs inside a trial.

```yaml
metric:
  name: validation_loss
  goal: minimize
  extractor:
    type: wandb
    entity: YOUR_ENTITY
    project: YOUR_PROJECT
    metric_key: eval/loss
    evaluation_axis: eval/step
    poll_seconds: 2
    timeout_seconds: 120
```

`metric_key` is a summary key as the W&B runs table names it. W&B reports the aggregate from `define_metric("eval/loss", summary="min")` as `eval/loss.min`, and that run has no `eval/loss` summary key, so a bare `eval/loss` fails the trial with an error that lists the available keys.

`evaluation_axis` names the history key that marks where `metric_key` was measured; it defaults to `_step`, W&B's own history step. Typically it names the `define_metric` step metric, e.g. `eval/step` under `define_metric("eval/*", step_metric="eval/step")`, or a token counter. Log it in the same `log()` call as the metric: W&B fills a step metric missing from a call with its previous value, so an axis logged separately lags one evaluation behind. It may not name the metric's own series. A plain key is positioned at the last history row whose value equals the summary value. An aggregate key such as `eval/loss.min` covers its whole `eval/loss` history series, so it is positioned where that series ends: how far the trial trained, not where its best value fell, which differs between comparable trials by nature. See [trial execution and evidence](runtime.md#trial-execution-and-evidence) for the advisory read's deadline and cleanup behavior.

For plain keys, exact history matches take priority. With no exact match, the reader can use the final finite metric row when it differs from the summary by one binary64 step and no other neighboring value occurs. Larger or ambiguous differences leave the position unavailable. This accommodates observed API numeric differences; it cannot establish whether a summary was assigned independently, and never changes the accepted scalar.

Within a phase, the primary objective, W&B constraints, and that phase's W&B gates must agree on normalized endpoint, entity, project, poll interval, and timeout. They share one finished-summary capture. Missing numeric keys keep polling for up to 30 seconds after the run first reads as finished, since finished summaries can still gain values, and then fail the trial with an error naming the missing and available summary keys; a timeout that arrives first names the same keys from the last finished summary, when one was read. When the objective and constraints are local, separate gate-only phases may use different W&B targets. A gate-only configuration accepts the first finished summary and fails if required keys are missing. Failed, crashed, killed, or preempted remote runs cannot win.

PhaseSweep supplies `WANDB_BASE_URL`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_RUN_ID`, and `WANDB_RESUME=never`. Trainers must honor those values and finish their run. Explicit conflicting identity settings and offline/disabled modes are rejected before training; ambient identity defaults are normalized. An ordinary `wandb.init()`, scalar log/summary assignment, and `finish()` are sufficient; no PhaseSweep import or local objective mirror is needed.

The reader receives the trainer's composed authentication and transport environment. List rotating environment credentials or proxies in `execution.passthrough_env` ([trainer contract](#trainer-contract)). Excluded parent credentials are not reacquired. Supported SDK account authentication also works; an API-key environment variable is not mandatory. Credentials are never serialized in worker requests. Syntax validation and dry runs do not authenticate.

W&B's source is keyed by attempt ID; it makes none of the envelope's evaluation-metadata or trainer-input-content assurances. It only records where in the run's history the objective's value was logged. Report through `json_envelope` instead when the checkpoint behind the value must be recorded.

## Trainer contract

The trainer runs in `execution.cwd` when configured (a relative value resolves against the YAML file's directory), otherwise in the directory PhaseSweep was started from, or the catalog entry's `cwd` under MCP. Neither `validate` nor the dry run checks that `execution.cwd` exists; a missing one fails each attempt before its trainer starts, and the failures count toward `max_consecutive_failures`. The trainer receives the configured environment plus the variables PhaseSweep sets for each attempt: `PHASESWEEP_TRIAL_DIR` (the absolute attempt directory), `PHASESWEEP_TRIAL_ID` (the phase's trial number), `PHASESWEEP_PHASE`, `PHASESWEEP_RUN_NAME`, `PHASESWEEP_GENERATION_ID` (the invocation's generation ID, which is the run ID under MCP), `PHASESWEEP_ATTEMPT_ID` (a fresh random ID), `PHASESWEEP_OVERRIDES_SHA256`, `PHASESWEEP_OBJECTIVE_PATH` for a `json_envelope` metric, and the leased GPU variables. `env` may not set these per-attempt `PHASESWEEP_*` names; they replace any ambient value. Other `PHASESWEEP_*` names, such as `PHASESWEEP_LOCK_DIR`, are ordinary variables. Top-level `env` values are semantic configuration and, like `provenance` values, must be YAML strings, so quote numbers and booleans, as in `OMP_NUM_THREADS: "4"`.

The working directory is part of every persistent study's identity. `phasesweep validate` and `phasesweep run --dry-run` print the effective one on a `trainer cwd:` line: the resolved `execution.cwd`, or a note that it follows the directory PhaseSweep runs from. With `execution.cwd` unset, running the same config from another directory is therefore refused as a changed phase config; the refusal names `execution.cwd` and says to run from the original directory or set `execution.cwd`.

By default (`inherit_env: none`) the trainer receives the base variables listed by `phasesweep explain execution.inherit_env`, plus top-level `env`. Shell settings such as `PYTHONPATH`, `HF_HOME`, proxies and credentials need an explicit contract: select ambient names with `inherit_env: [NAME, ...]`, pin values under `env`, or use `inherit_env: all`. `validate` and the dry run print that contract. A failure suggesting a missing environment input produces one hint per phase; inspect its [recorded reason and stderr](runtime.md#following-a-run) before widening inheritance.

A persistent study compares the tracked environment before allocating another trial. The ledger records the mapping and its checksum; exports carry the checksum alone. Tracked names are the base variables except `TMPDIR`, explicitly inherited names, top-level `env` entries and configured tracking bindings. Under `inherit_env: all`, other ambient values reach the trainer untracked. `TMPDIR` and `execution.passthrough_env` values are inherited without being recorded or compared; an explicit `env` entry is always tracked, even if its name also appears in `passthrough_env`. Put rotating credentials in `passthrough_env`, because tracked values are plain text in the ledger. See `phasesweep explain execution` for name constraints, secret-name warnings and opt-in `environment.json` recording.

An environment mismatch names the differing values. For example, another shell, server or sandbox can supply a different `PATH`, `LANG`, `LC_ALL` or `TZ`. Pin a meaningful value under `env`, or use `execution.passthrough_env` when the difference is harmless. When pinning `PATH`, also use an absolute interpreter path in `trial_command` so every invocation selects the same interpreter.

See [GPU isolation](runtime.md#gpu-isolation) for discovery precedence and how device visibility interacts with environment inheritance.

When no W&B evidence consumer is configured, trainer-owned `WANDB_*` settings are preserved except for an ambient/inherited `WANDB_RUN_ID`, which is removed. An ID explicitly configured in top-level `env.WANDB_RUN_ID` remains trainer-owned in this case.

### Reporting an objective

A trainer that chooses the optional `json_envelope` extractor reports its objective once, after its final evaluation, with `phasesweep.report_objective`. With `validation_loss` equal to 0.4217, this call:

```python
from phasesweep import report_objective

# Train and evaluate using the YAML received at --config, then:
report_objective(
    validation_loss,
    name="validation_loss",
    split="validation",
    policy="final",
    checkpoint="final.pt",
    step=1000,
    progress={"tokens": 2_048_000},
    extra={"peak_memory_gb": 21.4},
)
```

writes this file and returns its path:

```json
{
  "attempt_id": "0697c579bd1e40b4b03adce95562457e",
  "evaluation": {
    "checkpoint": "final.pt",
    "policy": "final",
    "progress": {
      "tokens": 2048000
    },
    "step": 1000
  },
  "generation_id": "4afce4b915ee48da8e6da730d0928096",
  "objective": {
    "name": "validation_loss",
    "split": "validation",
    "value": 0.4217
  },
  "overrides_sha256": "5d6392dac385a6306df6a16e52a6491bdc86bacab5f79c2f8667bf7d6a09be03",
  "peak_memory_gb": 21.4,
  "schema_version": 1,
  "status": "complete"
}
```

The arguments:

- `value` is required: the finite objective, a Python or NumPy number (call `.item()` on a tensor).
- `name`, `split` and `policy` are required keywords. They must equal the extractor's `objective_name`, `split` and `policy`; a mismatch fails the trial.
- `checkpoint` is a required nonempty string naming the evaluated checkpoint. It must equal the extractor's `checkpoint` when that is set, and PhaseSweep never loads it.
- `step` is a required non-negative integer, equal to the extractor's `expected_step` when that is set.
- `progress` is optional: a mapping from an axis name such as `tokens` to a finite non-negative number, never named `step`. A declared `evaluation_axis` of that name makes it required.
- `extra` is optional: more top-level fields for the file, as described below.

It reads four variables that PhaseSweep sets for each attempt, so a trainer must not construct them. `PHASESWEEP_OBJECTIVE_PATH` is the absolute path it writes: the extractor's `path` (default `result.json`) inside the attempt directory, set only when the metric extractor is `json_envelope`. `PHASESWEEP_GENERATION_ID`, `PHASESWEEP_ATTEMPT_ID` and `PHASESWEEP_OVERRIDES_SHA256` are copied into the file, which is how the reader ties it to this attempt. `PHASESWEEP_OVERRIDES_SHA256` is the SHA-256 of the trainer input file PhaseSweep generated for the attempt, not of the envelope. The write is atomic, and a later call in the same attempt replaces the file.

The keys of `extra` become top-level fields of that same file, beside the envelope's own. These are reserved and raise `ValueError` when passed: `schema_version`, `status`, `generation_id`, `attempt_id`, `overrides_sha256`, `objective` and `evaluation`. Values must be finite plain JSON: numbers, strings, booleans, null, lists and string-keyed mappings. A tensor, a NumPy integer or a `float32` raises `TypeError` and a NaN or infinity raises `ValueError`, so convert with `float(...)` or `.item()` first. A nested mapping is addressed by a dotted key. `observe`, `constraints` and gates read them like any local JSON file, for example `{type: json, path: result.json, key: peak_memory_gb}` ([report-only observations](#report-only-observations)).

Outside a trial, or in a trial whose metric extractor is not `json_envelope`, `PHASESWEEP_OBJECTIVE_PATH` is unset. With valid arguments, the call raises `RuntimeError: report_objective() requires PHASESWEEP_OBJECTIVE_PATH; call it from a PhaseSweep trial using a json_envelope metric extractor.` and writes nothing; an unset identity variable raises the same error naming that variable. A trainer that is also run by hand guards the call with `if os.environ.get("PHASESWEEP_OBJECTIVE_PATH"):`.

Trainers that cannot import the Python helper can publish the same envelope from the managed trial environment:

```bash
phasesweep report-objective 0.123 --name validation_loss --split validation \
  --policy final --checkpoint final.pt --step 1000
```

A repeatable `--progress tokens=2048000` reports what `progress=` does from Python. The command has no equivalent of `extra`, and outside a trial it exits 1 with the same message. A negative value goes after `--`, as in `phasesweep report-objective --name validation_loss --split validation --policy final --checkpoint final.pt --step 1000 -- -0.123`.

### Check the trainer contract with one trial

Before a long sweep, run the trainer once under the real contract. Copy the config, change `experiment` to `contract_check` so the copy leaves the sweep's artifact tree alone, set `storage: null`, and replace `phases` with one fixed-configuration phase. Leave `trial_command`, `trainer_config`, `metric`, `observe`, `execution` and `env` as they are, and keep on the phase whatever the sweep's phases set that the trainer or its evidence depends on: `gpu_policy` with its `gpu_ids` or `gpu_devices` list (`whole_node` requires the list on Linux; macOS discovers its pool), the `fixed_overrides` the trainer needs, and the `gates`:

```yaml
storage: null
phases:
  - name: contract_check
    n_trials: 1
    search_space: {}
```

Run it and inspect the winner:

```bash
phasesweep run contract_check.yaml
phasesweep show-winners contract_check.yaml
# For a json_envelope metric using its default result.json path:
cat <workdir>/contract_check/contract_check/trial_00000__*/result.json
```

A trial that does not report what the extractor expects fails with the cause on its `TRIAL` line as `reason=`, for example `reason=metric extractor: JSON envelope at ... does not report objective 'eval_loss_typo'.` Inspect `stdout.log` and `stderr.log` in that trial's directory, and the configured local evidence file or bound native tracking run. On success `show-winners` prints the metric and every recorded `observed` value. `phasesweep status CONFIG --trials` prints no trial rows here (`no trials recorded: storage is null`), because `storage: null` keeps no ledger.

### Source-checkout preflight

Before launching a source-checkout trainer, verify the package actually imported by the configured Python executable. Running a checkout's example script does not make Python import that checkout's library. Match the example to its installed library, or explicitly select the checkout's source directory:

```yaml
execution:
  cwd: /absolute/path/to/transformers
env:
  PYTHONPATH: /absolute/path/to/transformers/src
```

Run a CPU-only preflight with that same executable and explicit environment; for this example:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/absolute/path/to/transformers/src \
  /absolute/path/to/python -c 'import transformers; print(transformers.__version__); print(transformers.__file__)'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/absolute/path/to/transformers/src \
  /absolute/path/to/python /absolute/path/to/transformers/examples/pytorch/token-classification/run_ner.py --help
```

Carry all required settings into the experiment's composed environment: the default `inherit_env: none` drops ambient `PYTHONPATH` and other shell settings. Use the actual example's `--help` to choose supported arguments; configuration validation does not execute the trainer or establish library compatibility. For Transformers with managed MLflow, also apply the [full-script run context](mlflow.md#transformers-trainer) before launch so final evaluation remains in the bound run. The W&B integration uses its own native callback and managed attempt identity.
