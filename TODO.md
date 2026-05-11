# phasesweep — TODO

## Deferred from review

### Multi-host phasesweep (v0.5.3 review #3)

The current stale-trial reaper assumes a single active orchestrator per
experiment namespace. With two phasesweep processes pointed at a shared
RDB, one's reaper would mark the other's `RUNNING` trials as `FAIL` and
(if it could resolve their identity files) potentially kill their training
processes.

For v0.5.x we scoped the product to single-orchestrator-per-experiment and
added two same-host `fcntl` locks (output namespace + persistent storage
identity) to make the failure mode loud. Lock files live under
`$TMPDIR/phasesweep-locks/`. Multi-host runs against a shared store are
documented as unsafe.

**Implementation path when needed (v0.6+):**

1. Per-trial owner metadata: `trial.set_user_attr("phasesweep_owner_id", ...)`
   with `f"{hostname}:{pid}:{uuid4()}"`, plus `phasesweep_owner_host`.
2. Heartbeat thread updates `phasesweep_heartbeat_ts` for active trials every
   ~10s.
3. Reaper only reaps trials whose heartbeat is older than a configured
   threshold (e.g. `3 * heartbeat_interval`). Without that, the reaper cannot
   distinguish "stale from a crash" from "live on another worker."
4. Trial-dir cleanup in the reaper has to skip trials owned by other hosts —
   the local reaper has no way to kill a remote process group anyway.

This is genuinely v0.6+ work; it changes the reaper contract and needs e2e
testing across two real hosts.

## Deferred from earlier reviews

### Pruner support (v0.5 review #3)

The Pruner schema was removed from Phase in v0.5 because phasesweep cannot
report intermediate values to Optuna from a subprocess. ASHA/Hyperband/Median
pruning requires `trial.report(value, step)` + `trial.should_prune()` in a
polling loop while the child is still running.

**Implementation path when needed:**

1. Add `intermediate_metric` config with a log_regex extractor + `poll_seconds`.
2. Replace `proc.wait(timeout=...)` with a polling loop that tails the log file,
   reports new values to the Optuna trial, and kills the process group on prune.
3. Re-add pruner to Phase config once the polling loop exists.

Shipping non-functional pruner config is worse than not shipping it.

### argv-based trial execution (v0.5 review #8, long-term)

Currently `trial_command` is a shell template expanded via `str.format()` and
executed with `shell=True`. All substituted values are `shlex.quote`-d, which
is correct but fragile. The better long-term design is `trial_command` as an
argv list (no shell), with override injection via `{overrides}` token expansion
into the list. This eliminates the shell-injection surface entirely.

### String GPU IDs (v0.5 review #1, long-term)

`gpu_ids` currently accepts `list[int]`. CUDA device identifiers can also be
UUIDs or MIG identifiers. The pool now actively rejects non-numeric
`CUDA_VISIBLE_DEVICES` rather than silently coercing, so we won't accidentally
ship broken behavior — but full support requires `list[str]` plumbing through
the pool and the `CUDA_VISIBLE_DEVICES` formatting.

### Effective n_jobs capping (v0.5.2 review aside)

The reviewer suggested defaulting `effective_n_jobs = min(n_jobs, len(gpu_ids))`
when GPUs are scarce. The post-acquire abort recheck (blocker 8 in v0.5.2)
already prevents queued threads from launching after an abort fires, so this
is now a quality-of-life knob rather than a correctness fix. Worth doing if a
user reports excessive thread churn.

## Resolved in v0.5.7

- **Filesystem outputs namespaced by experiment** (review v0.5.6 / blocker 1).
  `<workdir>/<experiment>/<phase>/...` and `<workdir>/<experiment>/summary.yaml`.
  Run lock now takes both an output-namespace lock and a storage-identity lock.
- **Hydra/argparse `{overrides}` placeholder required** when the phase has
  inherited / fixed / sampled overrides (review v0.5.6 / blocker 2). Uses
  `string.Formatter().parse()` so escaped `{{overrides}}` does not satisfy
  the check.
- **`winner.yaml` carries `phase_fingerprint`; `--from-phase` verifies it**
  against the recomputed current-config fingerprint (review v0.5.6 / blocker
  3). Stale or unfingerprinted winners are refused with a clear "re-run the
  phase" message. `n_trials` and other run-control fields remain
  fingerprint-compatible by design.
- **Strict YAML loader rejects duplicate mapping keys** at parse time
  (formerly v0.5.5 non-blocker). Pre-v0.5.7 PyYAML's default loader silently
  kept the last value for a duplicated key.
- **Override-key syntax validated at config-load** (formerly v0.5.5
  non-blocker). Rejects empty, leading/trailing-dot, consecutive-dot,
  whitespace-bearing, and shell-special-character keys for both
  `search_space` and `fixed_overrides`.
- **`cmaes` optional-dependency check moved to config-load** (formerly v0.5.6
  non-blocker). `phasesweep validate` now catches a `sampler.type='cmaes'`
  config when the optional `cmaes` package is not installed; previously the
  failure surfaced only at first trial launch. The runtime
  categorical-on-cmaes guard in `_build_sampler` is kept as defense in depth
  for direct callers of internal helpers.
