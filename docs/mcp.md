# phasesweep MCP server

`phasesweep-mcp` and `phasesweep mcp serve` expose a phasesweep experiment to an AI agent over the [Model Context Protocol](https://modelcontextprotocol.io) using the [supported MCP runtime](runtime.md#platform-support). The agent can launch a sweep, monitor it, and read the winning hyperparameters. It never supplies, edits, or sees a `trial_command`, `env`, `storage`, or `workdir`. It picks an experiment from a human-curated catalog by id and uses the server's tools.

For install commands, client config, and pasteable agent instructions, use [MCP agent setup](mcp_setup.md).

## The catalog

The server starts from a catalog: a fixed allowlist mapping opaque ids to local config paths plus per-experiment permissions. The agent only ever sends an id; it cannot enumerate the filesystem, pass a path, or author a config. Author the catalog with the same trust and review process as the experiment YAML. The server never writes it.

Catalog keys:

- `state_dir: path` (required): operator-owned directory for run handles, runner logs, config snapshots, and `audit.jsonl`. Startup and a successful `mcp check` create missing directories with mode `0700`, validate existing directories without changing them, then probe each directory with a temporary file. An unsafe existing mode fails with its observed owner/mode and a concrete `chmod 700 ...` remediation when ownership is already correct.
- `max_concurrent_runs: int = 1` (minimum `1`): live-sweep cap across all catalog entries; see [concurrency and single-GPU hosts](#concurrency-and-single-gpu-hosts).
- `experiments: list` (required; at least one item): allowlisted experiment entries. Entry ids must be unique.
- `experiments[].id: str` (required): agent-visible id matching nonempty `[A-Za-z0-9_-]+`.
- `experiments[].config: path` (required): local single-experiment YAML subject to the [MCP path and storage rules](#paths-and-the-working-directory).
- `experiments[].cwd: path | null = null`: detached-runner working directory; resolution and default behavior are covered under [paths and the working directory](#paths-and-the-working-directory).
- `experiments[].visible_params: "none" | "all" | list[str] = "none"`: sampled winner values exposed to agents. List entries are stripped, must be nonempty, and are deduplicated. Parameter names remain visible; withheld values return `<redacted>`, and each winner carries `params_redacted`.
- `experiments[].description: str = ""` (maximum 500 characters): optional purpose shown by `phasesweep_list_experiments` and the catalog resource.
- `experiments[].allow: object = {}`: side-effect permissions. `launch: bool = false`, `cancel: bool = false`, and `from_phase: bool = false`; omission leaves the entry read-only.

At startup the server validates each experiment with the same loader the CLI uses, computes a content hash, initializes the private `state_dir/runs` and `state_dir/logs` directories, and refuses an unusable state path, invalid configs, suites, configs that violate the [storage and path requirements](#paths-and-the-working-directory), or two entries that resolve to the same experiment output namespace or Optuna study namespace - one engine experiment is governed by exactly one catalog id. The complete loaded registry, including `allow.cancel` and `visible_params`, is frozen for the server's lifetime; catalog edits take effect only after restart. On launch, the server verifies the config still matches the startup hash and hands the detached runner a per-run snapshot, so later edits to the original file cannot change what the runner executes. That drift check runs on `validate` and `launch` only: read tools (`status`, `winners`, `latest_run`) serve a live run from its frozen per-run snapshot, and everything else from the startup-parsed config. After a restart, the policy on a current same-id entry governs winner-value visibility and cancellation of its older runs. If the id was removed, winner values default to fully redacted and cancellation uses the permission recorded at launch.

Give every catalog its own `state_dir`. Run handles carry only the bare entry id, so two catalogs sharing one state directory share one id namespace: same-id entries would read each other's runs, and each catalog's `max_concurrent_runs` would be evaluated against the combined live set.

The server speaks JSON-RPC over stdio; all logging goes to stderr.

### Paths and the working directory

`state_dir`, `config:`, and `experiments[].cwd` paths in the catalog resolve against the catalog file when they are relative. MCP experiment configs must use absolute `workdir` values and non-empty absolute SQLite/Journal storage paths so server restarts, wrappers, IDE launches, and desktop clients monitor the same local-node artifacts and Optuna studies. External RDB storage is rejected for MCP because the current cleanup, stale-trial reaping, and GPU lock semantics are same-host only. The detached runner always runs with the catalog entry's frozen `cwd`; omission defaults to the registered config file's directory, while `init-catalog` writes `cwd: "."` so a catalog scaffolded at the project root preserves project-relative trainer commands. The runner *process* is spawned in `state_dir` and receives that `cwd` as an argument it enters only after its own process identity is durable, so interpreter startup never reads the experiment's directory (see [security model](#security-model)). Relative paths inside `trial_command` are trainer-owned shell behavior; phasesweep does not parse or rewrite commands. Note the boundary this leaves when the experiment's `execution.cwd` is unset: the *trainer* then inherits the runner's directory - the catalog entry's `cwd` - which is part of the catalog, not of the experiment config, so it joins no fingerprint, and a CLI invocation of the same config runs the trainer from the invoking shell's directory instead. Set `execution.cwd` (absolute, required for MCP) whenever the trainer's working directory matters to results; with it set, entry `cwd` affects only the runner process itself.

### Concurrency and single-GPU hosts

`max_concurrent_runs` (catalog top level, default `1`) caps how many sweeps run at once across **all** experiments. The default of `1` suits a single-GPU host: each sweep's trials use the GPU, so a second concurrent sweep would contend for the device and slow both down. A `phasesweep_launch_sweep` that would exceed the cap returns up to five blocking run IDs; await one directly and retry the refused launch after it becomes terminal, or ask the user before cancelling it. Raise the cap on multi-GPU hosts where independent sweeps can run side by side.

The cap counts MCP-launched runs recorded in `state_dir`; it does not count a concurrent CLI `phasesweep run` on the same host. CLI and MCP runs are still coordinated by the runtime locks described in [runtime behavior](runtime.md#concurrency-model).

## Tools

| Tool | Inputs | Effect | Returns |
| --- | --- | --- | --- |
| `phasesweep_list_experiments` | optional `limit` (1-100; default 50), `cursor` | read | catalog ids, description, phase names, metric name + goal, authorized capabilities, `total_count`, `next_cursor` |
| `phasesweep_validate_config` | `experiment_id` | read | capabilities and per-phase name, terminal-attempt target (`n_trials`; COMPLETE, FAIL, and PRUNED all count), sampler, inherited phases, and search-space *keys* (not ranges); a changed config is a tool error |
| `phasesweep_get_latest_run` | `experiment_id` | read | the newest durable run, selected by launch timestamp with a stable tie-breaker, or `found: false` |
| `phasesweep_get_status` | exactly one of `experiment_id` or `run_id` | read | dense cumulative state counts plus explicit before-run and this-run progress, computed remaining attempts, storage-read availability, winner presence, result provenance, run state, elapsed time, and a safe actionable failure category when terminal; terminal run-id reads require a frozen snapshot |
| `phasesweep_await_run` | `run_id`, optional `timeout_seconds` (5-600; default 120) | read (waits) | the `phasesweep_get_status` payload plus `changed` and `reason` (`recovery_required` / `terminal` / `phase_completed` / `timeout`) |
| `phasesweep_get_winners` | exactly one of `experiment_id` or `run_id` | read | objective metadata, result provenance, declared/winner counts, missing phases, all-phases completeness, safe terminal failure context, and per-winner trial number, metric, policy-filtered sampled params, gate status, and partial-winner status |
| `phasesweep_launch_sweep` | `experiment_id`, optional `from_phase` | spawn detached | `{run_id, experiment_id, state}` |
| `phasesweep_cancel_sweep` | `run_id` | signal | `{run_id, state, cleanup_confirmed, recovery_required}` |

MCP annotations mirror the effects in the table. Permissions, closed input schemas, and safety gates are enforced server-side even when a client ignores those hints.

### Run state and recovery

A launched sweep runs as a detached background process in its own session, so it survives the agent's tool call and a server restart and can be cancelled as a group. `phasesweep_get_status` reports `running` / `succeeded` / `failed` / `cancelled`. `phasesweep_await_run` waits without preventing cancellation or other MCP calls. The packaged [agent instructions](../src/phasesweep/mcp/agent_prompt.md#workflow) define the call sequence. `from_phase` requires every preceding winner to be complete and fingerprint-compatible with the current phase chain; `phasesweep_launch_sweep` performs that authoritative readiness check before spawning.

`cleanup_confirmed` on `phasesweep_cancel_sweep` is tri-state:

- `true`: the MCP runner process group is gone and the runner wrote terminal evidence confirming trial cleanup.
- `false`: cancellation was attempted but could not confirm cleanup.
- `null`: the run was already terminal and no cancellation was attempted.

Use `recovery_required` as the decision field: when true, stop monitoring and report the run to the user because it will not become terminal until an operator resolves an uncertain launch, cleanup, or interrupted result finalization. `phasesweep_await_run` returns immediately with `reason: recovery_required`; status and latest-run payloads expose the same flag in their run metadata.

Terminal run metadata includes a path-free `failure` object with `code`, `stage`, `retryable`, `actor`, and a canned `remediation`. Cleanup uncertainty is the actionable outer failure and may retain the safe trainer or storage category under `cause`; the two fields never disagree about who must act next. External experiment-lock contention is `experiment_busy`, a retryable agent-owned preflight outcome. Raw exception messages remain operator-only because they can contain storage details or filesystem paths. A refusal after generation claim has its own frozen generation snapshot with zero attempts owned by that generation. A lock refusal before generation claim freezes the declared phase shape with `current_generation_id: null` and `published_generation_id: null`, unavailable trial data, and an explicit private terminal reason, so run-specific status and winner reads remain usable without replacing the last successful experiment result.

`phasesweep_get_status` always reports four generation identities, never conflated: `current_generation_id` is the actual mutable pointer (the most recent invocation's own claim - may be failed or in-progress) and `published_generation_id` is the actual validated [last-success pointer](runtime.md#output-layout); neither is ever forced to equal a queried `run_id`. `represented_generation_id` is the generation whose `winner_present`/`summary_present`/this-run trial counts the payload shows: the queried `run_id` itself, or `published_generation_id` when querying by `experiment_id`. `is_published` is true only when `represented_generation_id` equals `published_generation_id` - a `run_id` whose own publication failed still returns its own (unpublished) winners and this-run counts with `is_published: false`, so a caller can distinguish "this generation's own results" from "these results are the experiment's official published output."

Cleanup confirmation is emitted by the engine shutdown handler after it terminates active trial process groups through the same confirmed cleanup path used by stale-trial recovery. If a spawned runner disappears without recording terminal status, if the runner group is gone but cancellation cannot observe status, or if status reports unconfirmed trial cleanup, the server writes a cleanup-uncertain marker and keeps the run counted as live so later launches do not reuse possibly-held resources. Normal runner shutdown asks the engine to tear down trial groups, and uncertain trainer leftovers are handled by the engine's stale reaper before later launches.

This state does not clear itself. A runner killed by SIGKILL or the OOM killer never gets to write terminal status, so its run stays `running` behind the cleanup-uncertain marker and keeps consuming a launch concurrency slot for as long as the state directory says so - there is no timeout, sweeper, or retry that will retire it. A host reboot is the one exception: run handles and cleanup markers record the Linux boot id, and an identity from an earlier boot proves the runner and all of its trial descendants are gone, so such a run derives `failed` with cleanup confirmed and no signal is ever aimed at the PID or process group that inherited those numbers. Handles written before boot ids were recorded, and hosts that expose no boot id, keep the conservative behavior above. `recovery_required` is true throughout, and the agent should stop polling and hand the run to a human rather than wait it out. The stuck run blocks new launches of its own experiment and counts against [`max_concurrent_runs`](#concurrency-and-single-gpu-hosts), so at the default cap of `1` a single OOM-killed sweep blocks launches for *every* experiment in the catalog until an operator clears it.

Cleanup and result-finalization recovery are operator-only. After inspecting the host, run `phasesweep mcp recover-run --state-dir <state_dir> --run-id <run_id>` to verify the saved config snapshot hash and report the cleanup or terminal-snapshot actions that recovery would attempt. This preflight is observational: it sends no signals, writes no state, and requires an existing recognizable state directory. Repeat with `--confirm` to acquire the same experiment locks as a normal run before performing any required process-group cleanup, reaping stale `RUNNING` trials, consuming terminal cleanup evidence so it cannot clear a later run, writing a failed terminal status when a failed pre-spawn launch or dead runner omitted one, recording cleanup recovery evidence, and finalizing a snapshot that the runner already captured. Lock contention aborts recovery before it sends a signal or changes study state. Recovery never rebuilds historical results from the current shared study. MCP deliberately has no tool for these operator actions.

When a `run_id` is supplied, live status is read through that run's saved config snapshot, so catalog edits after launch cannot redirect monitoring. Terminal finalization is ordered:

1. The engine captures a validated, path-free snapshot of phase counts and sampled winners before releasing the experiment lock, on both success and failure paths. On success the frozen winners come from the engine's own terminal report - the exact in-memory outcome it returned - not from a second read of winner files; the optional per-generation lifecycle record and per-phase storage counts only enrich the snapshot (unreadable counts freeze as `trial_data_available: false`), they are never a second success gate.
2. The runner persists the terminal cause, cleanup evidence, and raw captured snapshot with `result_snapshot_state: pending`.
3. It serializes only that already-frozen object and records the snapshot state as `complete` or `failed`. The engine's exit status stays the single authority on the run outcome: a run that exited 0 derives `succeeded` even when its snapshot could not be captured or finalized - result reads then fail closed with the explicit finalization state instead of contradicting an engine-defined, already-published success.

While finalization is `pending`, the run remains `running` and counts toward the launch concurrency limit. The shared-state read itself already happened under the engine lock; the pending state covers durable serialization of that immutable object. Later resumes cannot rewrite reads backed by a completed snapshot.

Terminal run reads fail closed if the snapshot is missing or malformed; they never substitute the experiment's mutable shared-study results. The tools still return the terminal run state and a structured `result_snapshot_unavailable` failure, with `result_source: terminal_snapshot_unavailable` and a config-only phase shape whose zero counts are explicitly marked unavailable. Operator recovery can apply confirmed cleanup evidence to an already-captured snapshot, but a missing historical snapshot is unrecoverable because the current study may include later work. If the run's original experiment id is no longer in the active catalog, winner parameter values use the strict `visible_params: none` behavior. `phasesweep_cancel_sweep` also accepts a decataloged run id only when that run handle recorded `allow.cancel: true` at launch; runs launched without cancel permission fail closed.

## Resource and prompt

Clients that support MCP resources can attach `phasesweep://catalog`. It returns the first catalog page as compact JSON using the same path-free payload as `phasesweep_list_experiments`. Agents should still call `phasesweep_list_experiments` when they need pagination or autonomous discovery.

Clients that support MCP prompts can use `phasesweep_run_and_monitor`, which serves the packaged [agent instructions](../src/phasesweep/mcp/agent_prompt.md).

See [Instruct the agent](mcp_setup.md#5-instruct-the-agent) for initialization instructions, project-file installation, and fallback setup.

## Security model

The catalog is the trust boundary. By construction the agent **cannot**:

- set or change `trial_command`, `env`, `storage`, `workdir`, search spaces, samplers, gates, or any safety waiver - no tool accepts a config or these fields;
- reference a config by path - every tool takes an `experiment_id` resolved against the frozen catalog; an unknown id is a clean error;
- read trainer output or rendered commands - **no tool returns log text**, because the rendered command and trainer output can carry secrets or PII. Operator-visible log locations are listed under [inspecting runs](#inspecting-runs);
- double-launch (rejected by a run-handle check and ultimately the engine's same-host lock), delete runs, or corrupt state.

Outbound payloads are built only from path-free typed views. Catalog listings are count-paginated with `limit` and `next_cursor`. Metric descriptors label objective-evidence assurance. `phasesweep_get_winners` reports the concrete `winner_source`, whether it belongs to the represented or a prior generation, and safe promotion context; it returns sampled `params` and omits composed `effective_overrides`, because those can include operator-authored fixed or inherited values such as private dataset ids, paths, or tokens. Sampled-value exposure follows the catalog's `visible_params` policy above. Keep secrets, access tokens, private paths, dataset ids, hostnames, or other sensitive values out of searchable parameter choices unless you deliberately expose them through that policy. A backstop converts any unexpected exception into a path-free operator-directed error rather than leaking a traceback; recoverable domain errors are surfaced as MCP tool errors for model self-correction.

### Objective-evidence assurance

Metric descriptors label the configured extractor's evidence assurance with per-field flags, not one coarse claim. Three flags describe attempt-binding strength as three distinct tiers, strongest first: `json_envelope`'s self-validating envelope, `wandb`'s immutable run-id keying, and `log_regex`'s bare attempt-scoped location. Every extractor is `attempt_location_scoped` (evidence is read from a location - a trial directory, or for `wandb` a run id - uniquely scoped to this generation and attempt), but location alone is the weakest guarantee: nothing in a `log_regex` file's *contents* identifies the attempt that produced it, so a file misplaced into the wrong trial directory would be read as gospel. Only `json_envelope` is also `attempt_identity_bound`: the envelope structurally echoes `generation_id`/`attempt_id`/`overrides_sha256` and the runtime cross-checks those reported values against its own identity before accepting the result - the strongest tier. Only `wandb` is `source_identity_keyed` instead: the run is addressed by the immutable `WANDB_RUN_ID=attempt_id` itself rather than by filesystem location, so identity comes from how the source is keyed rather than from a self-reported field inside the evidence - stronger than a bare log location, weaker than a self-validating envelope. `log_regex` gets neither of those two flags, only `attempt_location_scoped`. Only `json_envelope` is `objective_name_bound`, `split_bound`, and `evaluation_policy_bound`, because those are required fields the runtime always checks against the envelope's own reported values; `log_regex` and W&B carry no such binding. `checkpoint_declared`/`expected_step_declared` report whether the config pinned a value for those *optional* envelope fields, and `checkpoint_value_bound`/`expected_step_value_bound` report whether the runtime actually validates the envelope against that declared value (`True` only when declared - an envelope with no declared checkpoint or step still must report *some* non-empty checkpoint and non-negative step, but nothing pins it to a specific one).

`phasesweep_get_winners` intentionally exposes each completed phase winner's objective metric value. It does not expose per-trial metric histories, raw result files, trainer logs, datasets, target/dependent-variable values, validation labels, predictions, W&B dashboards, or rendered commands. Do not give the same agent separate filesystem or dashboard access when those artifacts must stay out of its context.

Between `exec` and the runner's first durable handle write, the server cannot yet name the process it created, so code that runs in that window could leave the recorded process group and make later cleanup confirmation false. The runner is therefore spawned from `state_dir` with `-P` (no cwd or script directory on `sys.path`), `-s` plus `PYTHONNOUSERSITE=1` (no user site directory), and with `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, and `PYTHONEXECUTABLE` removed from its environment. A project-local `phasesweep` shadow package, a project `sitecustomize.py`, or an injected module therefore cannot execute during interpreter startup. Every other environment variable is inherited unchanged.

This layer narrows the **agent's** authority. It does **not** sandbox the training subprocess, which remains as trusted as the human who wrote its command. Registering a malicious config runs it - your decision, identical to running `phasesweep run` by hand.

## Inspecting runs

Run handles and per-run logs live under `state_dir`:

- `state_dir/audit.jsonl` - launch/cancel side-effect audit records.
- `state_dir/runs/<run_id>.json` - the run handle.
- `state_dir/logs/<run_id>.log` - captured runner stdout/stderr (operator-only).
- `state_dir/logs/<run_id>.status.json` - the recorded terminal cause, explicit result-snapshot finalization state, and, when capture succeeds, the path-free status/winner snapshot used for stable run-id reads.
- `state_dir/logs/<run_id>.config.yaml` - the exact config snapshot executed by the runner (operator-only; may contain command, storage, env, and overrides).
- `state_dir/logs/<run_id>.cleanup_uncertain.json` - server-owned marker that keeps a cleanup-uncertain run counted as live.
- `state_dir/logs/<run_id>.cleanup_recovery.json` - operator recovery evidence written by `phasesweep mcp recover-run --confirm`.

The per-run `.log` captures the detached runner's control-process stdout and stderr, not the trainer's output. Each trial's rendered `command.txt`, `stdout.log`, and `stderr.log` live in its trial directory under the experiment workdir; the engine's durable `run.log` lives at that experiment root. See the [runtime output layout](runtime.md#output-layout).

`audit.jsonl` contains best-effort append-only records for launch and cancel side effects: timestamp, local stdio actor, server session id, tool name, bounded safe arguments, resolved ids, outcome, safe error details, and state-transition summaries. Read-only catalog, status, await, and winner calls are not logged. Audit records do not include tool result payloads, trainer logs, commands, config paths, storage URLs, environment values, sampled winner params, or effective overrides.

The packaged [agent workflow](../src/phasesweep/mcp/agent_prompt.md#workflow) defines polling cadence. SQLite-backed status uses a read-only direct count path. Journal-backed status goes through Optuna's full read path today, so avoid frequent polling on very large Journal-backed studies.

### Pruning terminal run history

Prune only terminal runs between campaigns, never active or recovery-required runs. Treat one run's history as a unit: `state_dir/runs/<run_id>.json` and every matching `state_dir/logs/<run_id>.*` file. Removing the handle makes the run undiscoverable through latest-run and run-id lookups; removing its sidecars makes run-specific status and winner reads unavailable. Archive the complete set when that history still matters.

### Long-running servers

The server is built to stay up across multi-hour sweeps. Exited detached runners are reaped during status and live-run scans, the server does not hold per-run log file descriptors open, and run state is derived from disk on each query rather than kept in memory, so a server restart re-discovers live runs from their handles. A runner that exits just after a scan can remain a zombie until the next scan. Run history accumulates one small file set per launch.

Run IDs contain a full UUID and their handle paths are claimed with exclusive creation under the launch lock before any config, log, or status sidecar is written. A collision is retried without touching the existing run. The only handle update preserves its immutable experiment, config, timestamp, and permission fields while moving from `launching` to `spawned`; an identical spawned update is idempotent. Terminal `status.json` files, config snapshots, and legal handle updates use descriptor-relative atomic replacement, so readers do not observe torn replacements and symlinked path components are rejected before any target is truncated or chmodded. Existing private state must remain owned by the current user with directory mode `0700` and file mode `0600`; unsafe reuse fails closed without changing it. An unresolved `launching` handle reserves concurrency across a server restart because the durable state cannot distinguish a pre-spawn crash from a child that has not self-persisted yet. The child writes its identity - PID, PGID, `/proc` start time, and boot id - before it enters the experiment's working directory or launches any training work; known spawn or bookkeeping failures write terminal failure status, while an outcome left ambiguous by a hard server crash requires operator recovery. If the server's final spawned-handle update fails, it terminates the spawned runner rather than leaving an untracked sweep behind.
