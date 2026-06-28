# Runtime Behavior

- [Platform Support](#platform-support)
- [Output Layout](#output-layout)
- [Process Management](#process-management)
- [Stale Trial Reaping](#stale-trial-reaping)
- [Concurrency Model](#concurrency-model)
- [Fingerprints and Resume](#fingerprints-and-resume)

## Platform Support

Non-dry-run execution requires a POSIX platform such as Linux or macOS. phasesweep
uses POSIX process groups for subprocess cleanup and `fcntl.flock` for same-host
locks. Config validation and `--dry-run` do not launch subprocesses and do not
take those locks, but real runs fail early on unsupported platforms.

## Output Layout

A completed run writes one namespace per experiment:

```text
runs/
  tiny_lm_16mb/
    depth/
      trial_00000/
        command.txt
        overrides_resolved.json
        result.json
        stdout.log
        stderr.log
      trials.csv
      winner.yaml
    lr/
    regularization/
    run.log
    summary.yaml
  phases.db
```

`pid`, `pgid`, and `pid_starttime` are written atomically while a trial is live. They are removed on clean exit and preserved on failure for inspection. If those identity writes fail after launch, phasesweep terminates the new process group before returning a failed trial result.

![output layout](images/diagramG_artifacttree.png)

## Process Management

![trial state machine](images/diagramB_statemachine.png)

Every trial runs in a new process group via `start_new_session=True`. Timeouts and shutdown signals target the whole group, so descendants such as launcher workers or dataloader processes are cleaned up with the root process.

`timeout_seconds_per_trial` is the normal per-trial subprocess cap. `timeout_seconds_per_phase` and top-level `timeout_seconds_per_run` are hard wallclock caps for the larger execution scope: phasesweep passes the remaining budget into GPU lease acquisition and active trial supervision, so a queued or running trial cannot extend past the phase/run deadline. When a phase or run deadline stops the phase before the requested number of completed evaluations exists, phasesweep refuses to select a partial winner unless the phase sets `allow_incomplete_on_timeout: true`.

If a wallclock timeout and `max_consecutive_failures` become true in the same phase, timeout handling takes precedence. A phase that has at least one completed feasible trial can therefore persist a timeout-marked partial winner when `allow_incomplete_on_timeout: true`, instead of having that winner masked by the consecutive-failure abort path. Without that opt-in, the same situation fails closed with `TimeoutError`.

SIGTERM, SIGINT, and SIGHUP trigger shutdown cleanup. The handler sends SIGTERM to active groups, waits briefly, sends SIGKILL to survivors, and exits with `128 + signum`. SIGKILL and hard OOM kills cannot be caught by Python.

Launch uses signal deferral around the `Popen()` to registry window. A shutdown signal cannot land between process creation and registration and leave the child unsignalled.

If cleanup cannot prove the process group is gone, phasesweep fails closed with `UnsafeProcessCleanupError`. Under parallel Optuna execution, the orchestrator records a hard abort so no queued worker can reuse the released GPU lease before the error surfaces.

## Stale Trial Reaping

On startup and before skipped phases in `--from-phase`, phasesweep reaps Optuna trials stuck in `RUNNING`:

1. Read the persisted `phasesweep_trial_dir` user attribute, or fall back to the
   canonical trial directory when a crash left a pre-launch `RUNNING` trial
   before that attr was written.
2. Match PID plus process start time to avoid PID-reuse kills.
3. Fall back to PGID cleanup when the root PID is gone but descendants remain.
4. Mark the trial `FAIL` only after cleanup is confirmed.

Reaping runs before fingerprint checks, so a config mismatch cannot leave old GPU-holding processes alive.

## Concurrency Model

phasesweep supports one orchestrator per experiment on one host. Inside one orchestrator, `n_jobs > 1` parallelizes trials in a phase.

A run always takes same-host `flock`s under `PHASESWEEP_LOCK_DIR` when set, otherwise `/var/tmp/phasesweep-locks/`:

- Output lock: resolved `<workdir>/<experiment>/` path.
- Storage lock: canonical Optuna storage identity plus experiment name when storage is persistent.

![guard layer](images/diagramE_guardlayer.png)

SQLite identities fold SQLAlchemy dialects, so `sqlite:///x.db` and `sqlite+pysqlite:///x.db` collide. Locks are taken in deterministic path order and a second process fails fast instead of corrupting output or storage.

The lock directory must resolve to one path shared by every cooperating phasesweep process on the host. Schedulers that set a per-job `TMPDIR`, containers with private `/tmp`, and systemd `PrivateTmp` units should set `PHASESWEEP_LOCK_DIR` to a host-shared path such as `/var/tmp/phasesweep-locks` or a site-managed node-local equivalent.

Upgrade note: older phasesweep builds used the process temp directory for these locks. Existing stale locks under `/tmp` or a scheduler-provided `TMPDIR` are not consulted after the default moves to `/var/tmp/phasesweep-locks`; set `PHASESWEEP_LOCK_DIR` explicitly during a staged upgrade if you need old and new processes to coordinate.

CUDA device tokens also take per-device host locks. Explicit `gpu_ids`, explicit `gpu_devices`, ambient `CUDA_VISIBLE_DEVICES` tokens, and auto-detected `nvidia-smi` numeric devices are leased even for `n_jobs == 1`, preventing independent local phasesweep runs from double-booking the same GPU. Numeric tokens keep numeric lock names; opaque UUID/MIG tokens use sanitized, hashed lock names. When a GPU is assigned, the child environment defaults `CUDA_DEVICE_ORDER=PCI_BUS_ID` unless the operator explicitly set another order. CPU-only parallel phases require `allow_no_gpu_isolation: true`.

> [!WARNING]
> Multi-host writers against one shared study are unsupported. The startup reaper owns all visible `RUNNING` trials, so two hosts could fail each other's live work. Safe multi-host orchestration would need per-trial leases, heartbeats, and host-aware stale-trial reaping.

## Fingerprints and Resume

Each phase study stores a semantic fingerprint. Run-control fields are excluded: `n_trials`, `n_jobs`, `gpu_ids`, `gpu_devices`, `max_consecutive_failures`, `allow_no_gpu_isolation`, `allow_unbounded_trials`, `timeout_seconds_per_phase`, `allow_incomplete_on_timeout`, `allow_partial_grid`, `allow_seed_search`, and `comment`.

Semantic fields are included: search space, sampler, fixed overrides, contracts, gates, promotion, trial command, [override format](config.md#override-formats), metric, constraints, environment, inherited winners, and `timeout_seconds_per_trial`.

Re-running the same YAML reuses the study and tops it up when the fingerprint matches. `--from-phase <name>` skips earlier phases by loading their `winner.yaml` files, after stale reaping and fingerprint verification. Promotion is applied before `winner.yaml` is written, so `continue_baseline` resumes from the exposed baseline winner. A persisted incomplete timeout winner only loads when the current skipped phase still sets `allow_incomplete_on_timeout: true`.

`--dry-run` renders one example command per phase without launching subprocesses, writing preview files, creating run directories, touching storage, or taking the run lock.
