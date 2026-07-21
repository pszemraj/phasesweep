# Runtime behavior

## Platform support

Non-dry-run execution requires a POSIX platform such as Linux or macOS. phasesweep uses POSIX process groups for subprocess cleanup and `fcntl.flock` for same-host locks. Config validation and `--dry-run` do not launch subprocesses and do not take those locks, but real runs fail early on unsupported platforms.

MCP operations that validate a catalog, start the server, install an MCP client entry, or recover a run require Linux with readable `/proc` process start times. This makes autonomous cancellation and crash recovery PID-reuse-safe. Instruction-only client installation does not load a catalog and remains available without the MCP runtime. The core CLI remains available on supported POSIX platforms.

## Output layout

The bundled example writes one namespace per experiment:

```text
runs/
  tiny_lm_16mb/
    depth/
      trial_00000__generation_<generation-id>__attempt_<attempt-id>/
        command.txt
        overrides_resolved.json
        process_identity.json
        result.json
        stdout.log
        stderr.log
      trials.csv
      winner.yaml
    lr/
    regularization/
    generations/
      <generation-id>/
        generation.yaml
        summary.yaml
        phases/<phase>/winner.yaml
        phases/<phase>/promotion.yaml
    generation.yaml
    last_successful_generation.yaml
    run.log
    summary.yaml
  phases.db
```

Every non-dry experiment invocation mints a generation ID, and every subprocess launch mints an attempt ID. Both are stored in Optuna before launch and included in the trial directory name, so a repeated in-memory study cannot read files left by an older trial with the same number. A generation namespace is claimed exactly once under the experiment lock; a reused caller-supplied ID is rejected before any lifecycle or trial state changes. Each generation keeps its lifecycle record, summary, winners, and promotion decisions under `generations/<generation-id>/`. `generation.yaml` identifies the current invocation, while `last_successful_generation.yaml` advances only after a complete generation has been published. The phase-level winners and root summary are compatibility projections of that last successful generation; a failed preflight or late-phase fingerprint check leaves them untouched. Resume loads both skipped winners and their promotion decisions from the immutable last-success generation, never from mutable compatibility projections. Winner entries retain the generation and attempt that produced their evidence, which can be older when a top-up reselects an existing trial or `--from-phase` reuses a validated parent winner.

Each trainer starts behind a supervisor ready/ack handshake. The supervisor cannot exec the trainer until PhaseSweep atomically writes and fsyncs `process_identity.json` with the attempt ID, PID, PGID, Linux process start time, host boot ID, and a launch nonce. If the orchestrator dies before acknowledgement, pipe EOF makes the supervisor exit without starting training. The identity record is removed on clean exit and preserved on failure for inspection. If identity persistence fails, PhaseSweep terminates the blocked supervisor group before returning a failed trial result.

`json_file` trials additionally receive `overrides.json`, and a phase with a promotion rule writes `promotion.yaml` when that rule is evaluated. Files such as `result.json` are trainer-owned evidence, not fixed phasesweep output.

## Process management

![trial state machine](images/diagramB_statemachine.png)

Every trial runs in a new process group via `start_new_session=True`. Timeouts and shutdown signals target the whole group, so descendants such as launcher workers or dataloader processes are cleaned up with the root process.

`timeout_seconds_per_trial` is the normal per-trial subprocess cap. `timeout_seconds_per_phase` bounds each Optuna optimize invocation, including a later top-up, while top-level `timeout_seconds_per_run` bounds the current whole-experiment invocation. phasesweep passes the remaining budget into GPU lease acquisition and active trial supervision, so the deadline stops new work and begins process-group termination; the SIGTERM/SIGKILL cleanup grace can finish after that deadline. When a phase or run deadline stops the phase before the requested number of terminal trial attempts exists, phasesweep refuses to select a partial winner unless the phase sets `allow_incomplete_on_timeout: true`.

If a wallclock timeout and `max_consecutive_failures` become true in the same phase, timeout handling takes precedence. A phase that has at least one completed feasible trial can therefore persist a timeout-marked partial winner when `allow_incomplete_on_timeout: true`, instead of having that winner masked by the consecutive-failure abort path. Without that opt-in, the same situation fails closed with `TimeoutError`.

Metric extraction and evidence gates run after the trainer process and outside the GPU lease. W&B extractors and gates use their own polling timeouts, which are not shortened by a phase or run deadline and can therefore finish after that deadline. `allow_incomplete_on_timeout` applies only when a phase or run deadline leaves the phase short of its terminal-attempt budget; it does not turn an ordinary per-trial timeout into a selectable result.

SIGTERM, SIGINT, and SIGHUP trigger shutdown cleanup. The handler sends SIGTERM to active groups, waits briefly, sends SIGKILL to survivors, and exits with `128 + signum`. SIGKILL and hard OOM kills cannot be caught by Python.

Launch uses signal deferral around the `Popen()` to registry window. A shutdown signal cannot land between process creation and registration and leave the child unsignalled; uncatchable parent death before identity commit cannot start the trainer because the supervisor still awaits acknowledgement.

If cleanup cannot prove the process group is gone, phasesweep fails closed with `UnsafeProcessCleanupError`. Under parallel Optuna execution, the orchestrator records a hard abort so no queued worker can reuse the released GPU lease before the error surfaces.

## Stale trial reaping

Before a new generation is published or any phase begins, phasesweep inspects every declared phase study that already exists and reaps Optuna trials stuck in `RUNNING`:

1. Read the persisted `phasesweep_trial_dir` user attribute, or fall back to the canonical trial directory when a crash left a pre-launch `RUNNING` trial before that attribute was written.
2. Require one complete atomic identity bound to the Optuna attempt. Missing, malformed, partial, or wrong-attempt identity leaves cleanup uncertain rather than proving absence.
3. On Linux, match boot ID plus PID and process start time to avoid PID- and reboot-reuse kills. A record from an earlier boot is safe to close without signalling because no process from that boot can remain. When robust process-birth identity is unavailable, automatic stale signalling fails closed for operator recovery.
4. Fall back to the verified PGID when the root PID is gone but descendants remain.
5. Mark the trial `FAIL` only after cleanup is confirmed.

Missing studies are not created by this recovery read. Reaping every existing phase runs before fingerprint checks, so a config mismatch or a later-phase orphan cannot leave old GPU-holding processes alive while earlier work starts.

## Concurrency model

phasesweep supports one orchestrator per experiment on one host. Inside one orchestrator, `n_jobs > 1` parallelizes trials in a phase.

A run always takes same-host `flock`s. By default they live in the owner-only per-user directory `~/.cache/phasesweep/locks`:

- Output lock: resolved `<workdir>/<experiment>/` path.
- Storage lock: canonical Optuna storage identity plus experiment name when storage is persistent.

![guard layer](images/diagramE_guardlayer.png)

SQLite identities fold SQLAlchemy dialects, so `sqlite:///x.db` and `sqlite+pysqlite:///x.db` collide. File-backed storage lock identities ignore URL query options, so `sqlite:///x.db?timeout=30` and `sqlite:///x.db` share a lock. Locks are taken in deterministic path order and a second process fails fast instead of corrupting output or storage.

The private default coordinates every phasesweep process running as the same user. Cross-user coordination is opt-in: an administrator must provision an existing root-owned directory with the scheduler group, setgid and sticky bits, and mode `03770`, then set `PHASESWEEP_LOCK_DIR` for every cooperating process. PhaseSweep validates that directory and creates group-readable/writable `0660` lock files; it never creates or changes a shared namespace itself. An explicit owner-only `0700` directory is also accepted for custom per-user placement. Lock directory components and final lock files are opened without following symlinks, and existing files with unexpected ownership, links, or permissions are rejected before diagnostics are written.

Upgrade note: older phasesweep builds used `/var/tmp/phasesweep-locks`. Those files are not consulted by the private default; use a validated explicit `PHASESWEEP_LOCK_DIR` during a staged upgrade if old and new processes must coordinate.

CUDA device tokens also take per-device host locks. The policies are:

- `single_per_trial` (default): lease explicit `gpu_ids`, explicit `gpu_devices`, ambient `CUDA_VISIBLE_DEVICES` tokens, or auto-detected `nvidia-smi` numeric devices, even for `n_jobs == 1`. This prevents independent local phasesweep runs from double-booking the same GPU.
- `whole_node`: require `n_jobs: 1`, lease every configured or detected token, and expose the comma-joined set to the trainer for local DDP/FSDP/DeepSpeed-style launches.
- `none`: never change `CUDA_VISIBLE_DEVICES` or acquire GPU locks. Parallel use requires `allow_no_gpu_isolation: true` because isolation is delegated to the operator or an external scheduler.

Numeric tokens keep numeric lock names; opaque UUID/MIG tokens use sanitized, hashed lock names. When a GPU is assigned, the child environment defaults `CUDA_DEVICE_ORDER=PCI_BUS_ID` unless the operator explicitly set another order.

> [!WARNING]
> Multi-host writers against one shared study are unsupported. The stale-trial reaper owns all visible `RUNNING` trials, so two hosts could fail each other's live work. Safe multi-host orchestration would need per-trial leases, heartbeats, and host-aware stale-trial reaping.

## Fingerprints and resume

Each phase study stores a semantic fingerprint. Phase run-control fields are excluded: `n_trials`, `n_jobs`, `gpu_ids`, `gpu_devices`, `max_consecutive_failures`, `allow_no_gpu_isolation`, `allow_unbounded_trials`, `timeout_seconds_per_phase`, `allow_incomplete_on_timeout`, `allow_partial_grid`, `allow_seed_search`, and `comment`. Top-level experiment name, storage, workdir, and `timeout_seconds_per_run` are also outside the fingerprint; experiment name and storage select the study identity instead.

The payload starts with an explicit fingerprint-schema version and includes the trial command; operator-declared [provenance](config.md#experiment-keys); [override format](config.md#override-formats); environment; metric and constraints; contracts applied by the phase; every remaining phase field, including its name and inheritance declaration; and each inherited winner's effective overrides. The installed PhaseSweep version is recorded in each generation lifecycle record for diagnostics, but it is not semantic experiment identity.

With persistent storage, re-running the same YAML reuses a study and tops it up when the fingerprint matches. `--from-phase <name>` skips earlier phases by loading their `winner.yaml` files after stale reaping and fingerprint verification, even when storage is in-memory. Promotion is applied before `winner.yaml` is written, so `continue_baseline` resumes from the exposed baseline winner. A persisted incomplete timeout winner only loads when the current skipped phase still sets `allow_incomplete_on_timeout: true`.

Every top-up attaches the sampler declared by the phase rather than the Optuna default. Seeded random sampling derives each draw from the durable study name, trial number, and parameter name, so one uninterrupted invocation and arbitrarily batched top-ups produce the same parameter sequence. Grid sampling resumes from stored grid assignments. TPE and CMA-ES are reconstructed with the declared configuration and stored trial history on each invocation; their adaptive history is preserved, but PhaseSweep does not promise byte-for-byte equivalence with one uninterrupted in-memory sampler process.

A phase can be topped up in place while it has no bound descendants. Once a dependent phase study has been run, PhaseSweep rejects further top-ups of that reached ancestor before launching any trial: a new upstream winner would change the descendant's semantic fingerprint while its stored study name still identifies the old inherited context. Use a new experiment name for a larger upstream budget. `--from-phase` does not apply this restriction to skipped ancestors because they are not mutated.

Persistent studies also carry a PhaseSweep storage-schema version. An empty study can be initialized in place, but a populated study without the current schema is rejected before its rows count toward a trial budget. The error names the study and affected trial numbers; use a new experiment name or archive/delete the incompatible study rather than silently mixing unscoped historical trials.

## Validation and dry-run

`phasesweep validate` loads the config and checks schema, graph, sampler/search-space compatibility, override-key composition, and command-template placeholders. It does not check trainer or path existence, parser compatibility, storage connectivity, environment behavior, evidence production, GPU availability, or runtime promotion outcomes.

`phasesweep run --dry-run` additionally renders one valid sampled command per phase using fresh in-memory Optuna studies. The displayed sample becomes the hypothetical winner inherited by downstream preview commands, so one preview forms a coherent chain. Dry-run does not read persistent trial progress, create the referenced trial directories or `overrides.json`, display the final environment, launch subprocesses, extract evidence, evaluate gates or promotion, exercise suite dependencies, write files, touch configured storage, or take runtime locks.
