# phasesweep

YAML-driven, phase-chained hyperparameter sweeps. Each phase is an Optuna study; the winner becomes a fixed override for the next phase. No LLM in the runtime loop. Subprocess isolation per trial. Single-orchestrator-per-experiment, single-host.

```yaml
phases:
  - name: depth          # sweep n_layers in isolation
  - name: lr             # n_layers locked to depth's winner; sweep lr
  - name: regularization # n_layers + lr locked; sweep wd + dropout jointly
```

A successful full run produces:

```
runs/
├── tiny_lm_16mb/                          # <experiment> namespace
│   ├── depth/
│   │   ├── trial_00000/                   # one dir per trial
│   │   │   ├── command.txt                # rendered shell command
│   │   │   ├── overrides_resolved.json    # final {sampled} ∪ {fixed} ∪ {inherited}
│   │   │   ├── result.json                # whatever your extractor reads
│   │   │   ├── stdout.log, stderr.log
│   │   │   └── pid, pgid, pid_starttime   # cleaned on success
│   │   ├── trials.csv
│   │   └── winner.yaml                    # carries phase_fingerprint (SHA-256)
│   ├── lr/
│   ├── regularization/
│   └── summary.yaml                       # written at end of full run
└── phases.db                              # Optuna SQLite (path from `storage:`)
```

Every winner carries a `phase_fingerprint`. Edit a parent phase's search space and `--from-phase` a child → loud refusal, never silent reuse. (Bump `n_trials` to top up trials → still works; run-control fields are excluded from the fingerprint.)

---

## What this is and isn't

**Is.** A sequential HPO orchestrator. You write a YAML. It runs phase 1 to completion, picks a winner, fixes that winner as an override, runs phase 2, repeats. Each trial is a subprocess invoking your trainer with overrides rendered as `key=value` (Hydra), `--key value` (argparse), or `{trial_dir}/overrides.json` (json_file). Metrics come back via a JSON file, log regex, or W&B read-only.

**Isn't.**
- Not a training framework. Bring your own trainer.
- Not a joint search. Sequential phases are greedy; if you need joint optimization, use one phase with the full space.
- Not a cluster scheduler. **Multi-host parallelism against a shared study is unsafe and rejected** (see "Concurrency model"). One orchestrator per host with distinct experiment names if you need multiple boxes.
- Not an analysis tool. `optuna-dashboard sqlite:///runs/phases.db` works directly for sequential SQLite studies.
- No LLMs in the runtime loop. Phasesweep does HPO; an upstream tool can write the YAML if it wants to.

---

## Why phase chaining

Most HPO tools (Optuna, Hydra+Optuna sweeper, W&B sweeps, Azure ML Sweep, NNI) run *one* sweep per launch. None treat "sweep depth, lock the winner, then sweep LR conditional on that depth" as a first-class config concept. phasesweep does exactly that and nothing more.

Sequential phase chaining is the right tool when:
- Search dimensions are roughly independent (architecture vs LR vs regularization).
- A single joint sweep over all of them is too expensive.
- You want each phase's outcome to be inspectable, resumable, and explainable on its own.

It's the wrong tool when dimensions strongly interact — sweep them jointly in one phase.

---

## Install

```bash
pip install -e .                  # core
pip install -e .[dev]             # + pytest, ruff, mypy, types-PyYAML
pip install -e .[cmaes]           # + optional CMA-ES sampler
pip install -e .[wandb]           # + optional W&B extractor
```

---

## Config: `examples/experiment.yaml`

Walk through this and you have the whole config language:

```yaml
experiment: tiny_lm_16mb                                                  # study namespace
storage: sqlite:///./runs/phases.db                                       # Optuna URL
workdir: ./runs                                                           # outputs root
trial_command: "python train.py --out {trial_dir}/result.json {overrides}"
override_format: hydra                                                    # hydra | argparse | json_file

metric:
  name: eval_loss
  goal: minimize
  extractor: { type: json, path: result.json, key: eval_loss }

constraints:                                                              # optional; trials violating any constraint
  - name: param_bytes                                                     # are excluded from winner selection but
    extractor: { type: json, path: result.json, key: param_bytes }        # still recorded.
    max: 16777216

phases:
  - name: depth
    n_trials: 4
    sampler: { type: grid }
    search_space:
      n_layers: { type: categorical, choices: [4, 8, 12, 16] }

  - name: lr
    inherits: [depth]                                                     # depth's winner becomes a fixed override
    n_trials: 12
    sampler: { type: tpe, seed: 0 }
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }

  - name: regularization
    inherits: [lr]                                                        # lr transitively carries depth.n_layers
    n_trials: 16
    sampler: { type: tpe, seed: 1 }
    search_space:
      weight_decay: { type: float, low: 0.0, high: 0.3 }
      dropout:      { type: float, low: 0.0, high: 0.3 }
```

### Top-level keys

- **`experiment`** *(required)*: study namespace. Doubles as the filesystem path component (`<workdir>/<experiment>/...`) and is part of the lock identity. Must match `[A-Za-z0-9_-]+`.
- **`storage`** *(optional, recommended)*: Optuna storage URL. Pick the scheme intentionally — phasesweep does **not** silently rewrite SQLite to JournalStorage, so study identity stays stable across `n_jobs` changes.
  - `sqlite:///path.db` — sequential studies (`n_jobs == 1`). `optuna-dashboard sqlite:///path.db` reads them directly. **SQLite + `n_jobs > 1` is rejected at config-load** because SQLite serializes writers and will deadlock. The check matches all SQLAlchemy SQLite dialects (`sqlite:///`, `sqlite+pysqlite:///`, `sqlite+pysqlcipher:///`, ...) — no dialect-spelling bypass.
  - `journal:///path.journal` — Optuna `JournalFileStorage`. Safe for parallel `n_jobs` on a single host. Not dashboard-readable.
  - `postgresql://...`, `mysql://...` — passed straight to Optuna. Use these when you want both parallel `n_jobs` and live `optuna-dashboard` access from one orchestrator.
  - `null` (default) — non-resumable in-memory study. Not recommended.
- **`workdir`** *(optional, default `./runs`)*: outputs root. Final layout is `<workdir>/<experiment>/<phase>/...`.
- **`trial_command`** *(required)*: shell template. Placeholders:
  - `{overrides}` — required for hydra/argparse phases that have any inherited / fixed / sampled overrides. **A phase with overrides whose `trial_command` does not reference `{overrides}` is rejected at config-load** (otherwise Optuna samples 20 different LRs and the trainer sees zero — silent no-op sweep). Detection uses `string.Formatter().parse()`, so `{{overrides}}` (literal text) does not satisfy it.
  - `{overrides_path}` — required for json_file phases with overrides. Same silent-no-op protection.
  - `{trial_dir}`, `{trial_id}`, `{phase}`, `{run_name}` — always available.
- **`override_format`** *(optional, default `hydra`)*: how overrides are spliced into the command.
  - `hydra` → `key=value key2=value2`
  - `argparse` → `--key value --key2 value2`
  - `json_file` → writes `{trial_dir}/overrides.json` and exposes `{overrides_path}`.
  - All values pass through `shlex.quote` before substitution — no shell-injection surface from sampled values.
- **`metric`** *(required)*: `name`, `goal` (`minimize` | `maximize`), and an `extractor` (json | log_regex | wandb).
- **`constraints`** *(optional)*: list of additional extracted scalars with min/max bounds. Constraint-violating trials are recorded but excluded from winner selection.
- **`phases`** *(required, ordered)*: see below.

### Phase keys

- **`name`** *(required)*: must match `[A-Za-z0-9_-]+`.
- **`inherits`** *(optional, default `[]`)*: list of prior phase names whose winners become fixed overrides for this phase. Transitive — inherit from `lr` and you get `depth.n_layers` for free.
- **`fixed_overrides`** *(optional)*: hard-coded overrides applied to every trial in this phase. May intentionally re-set an inherited key — this is the *sole* way to resolve a multi-parent locked-key collision.
- **`search_space`** *(optional)*: map of override-key → sampler spec. Keys may be dotted (`model.depth`) for hydra/json_file. Override-key syntax is validated — empty / leading-or-trailing-dot / consecutive-dot / whitespace-bearing / shell-special-char keys are rejected.
- **`n_trials`** *(required, ≥ 1)*: trial budget. **Bumping `n_trials` between runs is a compatible change** — the fingerprint excludes run-control fields.
- **`n_jobs`** *(optional, default 1)*: parallel trials within this phase. When combined with `gpu_ids`, each trial gets exclusive `CUDA_VISIBLE_DEVICES` via a blocking pool.
- **`gpu_ids`** *(optional)*: explicit list of CUDA device indices, e.g. `[0, 1, 2, 3]`. Honored at `n_jobs == 1` too (lets a single-job phase isolate to a specific GPU on shared hardware). Auto-detected via ambient `CUDA_VISIBLE_DEVICES` or `nvidia-smi` when omitted and `n_jobs > 1`.
- **`max_consecutive_failures`** *(optional, default 5)*: abort the phase after this many consecutive failed/infeasible trials. Prevents a broken trainer from burning through `n_trials`.
- **`sampler`** *(optional, default `{type: tpe}`)*: `tpe` | `random` | `grid` | `cmaes`. Sampler-vs-search-space compatibility is checked at config-load: cmaes-with-categoricals, grid-with-log-floats, grid-with-non-step-divisible-floats are all rejected. cmaes also import-checks the optional `cmaes` package at config-load.
- **`timeout_seconds_per_trial`** *(optional)*: kill the trial's entire process group if it runs longer. Semantic: changing this value invalidates the study fingerprint, since a different timeout changes which trials FAIL vs COMPLETE under the same sampled params.
- **`comment`** *(optional)*: free-text design note. Surfaced by `phasesweep validate` and `phasesweep show-winners` so the *why* of each phase lives next to the spec. Excluded from the fingerprint — editing the comment never invalidates the study.

### Override priority within a trial (low to high)

1. Inherited winners' `effective_overrides` (transitive).
2. Phase's `fixed_overrides`.
3. Sampled values from `phase.search_space`.

Inverse of YAML order. Reasoning: a child phase's `fixed_overrides` is the explicit, intentional knob — it must be able to override an inherited value. A sampled value is the most local, most specific decision and trumps both.

### Validation guarantees (all at config-load, before any trial runs)

- Phase graph is a valid DAG; `inherits` only points backward.
- A phase cannot sample a key already locked by an ancestor's winner. If two independent ancestor branches lock the same key, the child must resolve via `fixed_overrides`.
- A key cannot be both in `fixed_overrides` and `search_space` of the same phase.
- Dotted-key namespace collisions across `fixed_overrides` / `search_space` / inherited keys are rejected (`model` and `model.depth` together is contradictory under hydra/json_file).
- `trial_command` renders successfully against placeholder overrides (catches `{trail_dir}` typos, unbalanced braces, unknown placeholders).
- `trial_command` references `{overrides}` (or `{overrides_path}` for json_file) when the phase has overrides.
- Sampler-vs-search-space compatibility (see `sampler` above).
- Storage-vs-`n_jobs` compatibility (SQLite + parallel rejected).
- Override keys match `[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*`.
- YAML has no duplicate mapping keys (strict loader).

---

## Process management

phasesweep treats subprocess lifecycle as a first-class concern. Reviewers should focus here — it's the most adversarial surface.

**Process group isolation.** Every trial runs in its own process group (`start_new_session=True`). Timeouts, signals, and even *normal* root-process exits all verify the entire group is gone via `os.killpg`. So child processes spawned by your trainer (`torchrun` workers, dataloader subprocs, accelerator launchers) are cleaned up too. If the root shell exits cleanly but leaves descendants alive, phasesweep treats that as a lifecycle failure, kills the group, and preserves identity files for forensics.

**Signal forwarding.** SIGTERM/SIGINT to the orchestrator (Ctrl-C, `kill`, OOM-killer, SSH disconnect) → SIGTERM to every active child group → wait briefly → SIGKILL → exit. The PGID list is snapshotted *once* at signal time so a registration race during teardown can't leave a child unsignalled. Handlers are installed by both the CLI and the public `run_experiment()` entrypoint, so library callers using `from phasesweep import run_experiment` get the same cleanup contract as `phasesweep run`. The `Popen()` → `_register()` window is protected by a launch lock plus `pthread_sigmask`-based signal deferral so a SIGTERM that lands mid-launch waits until the child is registered before snapshotting.

**PID tracking.** Each trial writes `{trial_dir}/pid`, `{trial_dir}/pgid`, and `{trial_dir}/pid_starttime` on launch. They persist on failure (`nvidia-smi` → find PID → `cat runs/.../trial_00003/pid` → confirmed). On a clean exit they are removed. PID + start-time match is checked before any kill so PID reuse can't trick the reaper into killing an unrelated process.

**Stale trial reaper.** If the orchestrator dies mid-sweep, Optuna's storage shows those trials as `RUNNING` forever. On the next launch, phasesweep:
1. Looks up each `RUNNING` trial's persisted `phasesweep_trial_dir` user attribute (so changing `workdir` between runs doesn't break reaping).
2. Verifies the recorded PID is alive AND has the same start-time. Falls back to PGID-only kill when the root PID has exited but persisted PGID descendants are still alive. Refuses to kill on a starttime mismatch (PID-reuse safe path).
3. **Fail-closed**: if cleanup cannot prove the process group is gone (still alive after SIGKILL, permission denied, etc.) the reaper raises a `RuntimeError` with forensic details rather than silently marking the trial `FAIL`. Otherwise a leaked GPU-holding process could survive while new trials launch onto the same hardware.
4. Marks the trial `FAIL` in Optuna so the study can proceed.

Reap runs *before* the fingerprint check on study reuse, so a config-mismatch error cannot leave stale processes alive.

**Live cleanup, fail-closed.** The same fail-closed contract applies *during* a run, not just on restart. If a trial subprocess survives SIGKILL (or its termination otherwise can't be confirmed), the orchestrator records a hard-abort, prevents any queued worker from acquiring the just-freed GPU lease, drains in-flight trials, and surfaces `UnsafeProcessCleanupError` to the caller. This holds for `n_jobs == 1` and `n_jobs > 1`: Optuna's threaded `study.optimize` does not propagate uncaught objective exceptions, so the orchestrator owns the abort state directly rather than relying on Python exception flow.

---

## Concurrency model

phasesweep is **single-orchestrator-per-experiment, single-host**. Within one orchestrator, `n_jobs > 1` parallelizes trials inside a phase (multi-thread Optuna with subprocess trials).

**Same-host advisory locking.** A `phasesweep run` takes **two** `flock`s under `$TMPDIR/phasesweep-locks/` for the duration of the run:

- **Output lock** — keyed by the resolved `<workdir>/<experiment>/` path. Prevents two configs that share that path but disagree on storage from corrupting each other's `trial_*/`, `winner.yaml`, and `summary.yaml`.
- **Storage lock** — keyed by canonical Optuna storage identity + experiment name. Prevents two configs that share the study but live in different workdirs from interleaving phases against the same backend. SQLite identity is dialect-folded: `sqlite:///x.db` and `sqlite+pysqlite:///x.db` collide on the same lock.

Either lock alone misses a real collision mode. The locks are taken in deterministic path-sorted order; a held lock causes a second invocation to fail fast with a clear error rather than corrupt the first run.

A phase-level lock backs the run lock as defense in depth for direct callers of internal `_run_phase` (tests, future code paths). The public CLI never reaches it without first acquiring the run lock.

**Multi-host phasesweep against a shared Postgres/MySQL is unsafe today.** The stale-trial reaper marks every `RUNNING` trial it sees as `FAIL` on startup; with two orchestrators against the same study, one would `FAIL` the other's live trials. Use Postgres/MySQL for *durable storage and dashboards from a single host*, not concurrent multi-host runs. Safe multi-host needs per-trial leases plus heartbeat-based reaping — tracked in `TODO.md`.

---

## Resume semantics

There are two kinds of resume:

**Re-running the same YAML** reuses the existing Optuna study and tops it up. The study's user_attr stores the fingerprint of the producing config; re-launch recomputes it and either accepts (top-up) or refuses with a fingerprint-mismatch error. Run-control fields excluded from the fingerprint: `n_trials`, `n_jobs`, `gpu_ids`, `max_consecutive_failures`, `allow_no_gpu_isolation`, `comment`. Bumping `n_trials` to top up is therefore always compatible. Changes to search space, sampler, fixed overrides, trial command, override format, metric, constraints, env vars, inherited winners, or `timeout_seconds_per_trial` *do* invalidate the study. `timeout_seconds_per_trial` is intentionally semantic: a 60s vs 3600s budget changes which trials FAIL vs COMPLETE, which changes the observation distribution under one fingerprint and would silently mix censored and uncensored trials.

**`--from-phase <name>`** skips earlier phases by reading their `winner.yaml` files. Each `winner.yaml` is stamped with the producing phase's fingerprint. On resume, phasesweep recomputes the fingerprint of each *current* skipped phase against the *currently-resolved* inherited winners and refuses to load if:

- The stored fingerprint is missing → hand-edited file, re-run the phase.
- The stored fingerprint differs from the recomputed one → parent config has changed since the winner was produced. Re-run the parent, change the experiment name, or restore the matching config.

The earlier phases' Optuna studies are not touched by `--from-phase`.

`--dry-run` renders one example command per phase (with placeholder overrides for inherited values) without launching subprocesses or touching storage. Dry-run does not take the run lock — inspecting the plan during a real run is a legitimate workflow.

---

## Result extraction

Pick whichever extractor fits your existing setup. You don't have to instrument your trainer in any particular way.

**JSON file** (recommended). Trainer writes a result file at the end:

```yaml
metric:
  extractor: { type: json, path: result.json, key: eval.loss }
```

`key` is dotted and walks nested dicts. `path` is relative to `{trial_dir}`.

**Log regex.** Parse stdout/stderr (which phasesweep already captures) or any other log file:

```yaml
metric:
  extractor:
    type: log_regex
    file: stdout.log
    pattern: 'eval_loss=(?P<value>[0-9.eE+-]+)'
    select: last         # first | last | min | max
```

**Weights & Biases** (read-only — W&B does not drive anything):

```yaml
metric:
  extractor:
    type: wandb
    entity: my-team
    project: tiny-lm
    run_name_template: '{experiment}-{phase}-{trial_id}'
    metric_key: eval/loss
    poll_seconds: 2
    timeout_seconds: 300
```

Your training script must name its W&B run using the same template. phasesweep exposes `PHASESWEEP_RUN_NAME` in the trial environment for convenience.

Constraint extraction uses the same three extractor types.

---

## Tests

```bash
pytest                                # 244 passed, 1 skipped (cmaes optional dep)
ruff check src tests
ruff format --check src tests
mypy src/phasesweep --ignore-missing-imports
```

Tests are organized by topic, one file per surface. To find the tests that
guard a given behavior, look in the file named after that behavior:

- `tests/test_e2e.py` — full sweep + `--from-phase` replay against the fake trainer.
- `tests/test_storage_urls.py` — backend detection, SQLAlchemy dialect folding, absolute vs. relative path preservation, lock-identity collapse across equivalent URLs.
- `tests/test_locking.py` — phase / experiment-run / storage-identity flock collisions; same on-disk state must collide, unrelated state must not.
- `tests/test_process_supervision.py` — `Popen` lifecycle, signal handler installation, descendant cleanup, `cleanup_confirmed` propagation, launch-window deadlock guard. Several cases run as real subprocesses because POSIX signal delivery isn't mockable.
- `tests/test_stale_reaper.py` — startup reaper, PID-reuse detection via starttime check, fail-closed contract.
- `tests/test_fingerprint.py` — phase fingerprint, `--from-phase` verification, run-control field exclusion (so `n_trials` top-ups stay compatible).
- `tests/test_filesystem_layout.py` — `<workdir>/<experiment>/<phase>/` namespacing, `summary.yaml` placement, experiment-name validation.
- `tests/test_param_validation.py` — search-space and override validation: param-type bounds (NaN/inf rejection), categorical scalar-only, dotted-prefix collisions, override-key shell safety, sampler/search-space compatibility, trial-command template placeholder enforcement.
- `tests/test_runtime_behavior.py` — NaN/inf propagation through extractors, parallel-trial sampler config, `max_consecutive_failures` abort, Optuna logging suppression.
- `tests/test_config.py`, `test_extractors.py`, `test_overrides.py`, `test_selector.py`, `test_gpu_pool.py`, `test_cli.py`, `test_wandb_extractor.py` — schema, extractors, override composition, winner selection, GPU pool, CLI commands, W&B extractor.
