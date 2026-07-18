# phasesweep MCP server

`phasesweep-mcp` and `phasesweep mcp` expose a phasesweep experiment to an AI agent over the [Model Context Protocol](https://modelcontextprotocol.io) using the [supported MCP runtime](runtime.md#platform-support). The agent can launch a sweep, monitor it, and read the winning hyperparameters. It never supplies, edits, or sees a `trial_command`, `env`, `storage`, or `workdir`. It picks an experiment from a human-curated catalog by id and uses the server's tools. The server also exposes a read-only catalog resource and one workflow prompt for clients that support them.

For install commands, client config, and pasteable agent instructions, use [MCP agent setup](mcp_setup.md).

## The catalog

The server starts from a catalog: a fixed allowlist mapping opaque ids to local config paths plus per-experiment permissions. The agent only ever sends an id; it cannot enumerate the filesystem, pass a path, or author a config. Author the catalog with the same trust and review process as the experiment YAML. The server never writes it.

Catalog keys:

- `state_dir`: operator-owned directory for run handles, runner logs, config snapshots, and `audit.jsonl`.
- `max_concurrent_runs`: cap on live sweeps across all catalog entries. The default is `1`.
- `experiments[].id`: agent-visible id. It must match `[A-Za-z0-9_-]+`.
- `experiments[].config`: local experiment YAML path. Relative paths resolve against the catalog file.
- `experiments[].cwd`: optional detached-runner working directory. Relative paths resolve against the catalog file. The default is the registered config file's directory.
- `experiments[].visible_params`: sampled winner parameter values exposed to agents. Use `none` (default), `all`, or a list of allowed parameter keys. Parameter names remain visible; values outside the policy are returned as `<redacted>`, and each winner phase carries a `params_redacted` boolean so agents can report withheld values without string-matching the sentinel.
- `experiments[].description`: optional text shown by `phasesweep_list_experiments` and the catalog resource.
- `experiments[].allow`: optional side-effect permissions for `launch`, `cancel`, and `from_phase`.

At startup the server validates each experiment with the same loader the CLI uses, computes a content hash, initializes the private `state_dir/runs` and `state_dir/logs` directories, and refuses an unusable state path, invalid configs, suites, or configs that violate the [storage and path requirements](#paths-and-the-working-directory). The id-to-path mapping is then frozen for the server's lifetime. On launch, the server verifies the config still matches the startup hash and hands the detached runner a per-run snapshot, so later edits to the original file cannot change what the runner executes.

Omitting `allow` leaves an experiment read-only: agents can list, validate, inspect status, and read existing winners, but `phasesweep_launch_sweep`, `phasesweep_cancel_sweep`, and `from_phase` resume are refused until the operator explicitly sets the corresponding flag to `true`.

The server speaks JSON-RPC over stdio; all logging goes to stderr.

### Paths and the working directory

`state_dir`, `config:`, and `experiments[].cwd` paths in the catalog resolve against the catalog file when they are relative. MCP experiment configs must use absolute `workdir` values and non-empty absolute SQLite/Journal storage paths so server restarts, wrappers, IDE launches, and desktop clients monitor the same local-node artifacts and Optuna studies. External RDB storage is rejected for MCP because the current cleanup, stale-trial reaping, and GPU lock semantics are same-host only. The detached runner always starts with the catalog entry's frozen `cwd`, defaulting to the registered config file's directory. Relative paths inside `trial_command` are trainer-owned shell behavior; phasesweep does not parse or rewrite commands, but their base directory is no longer inherited from the MCP server process.

### Concurrency and single-GPU hosts

`max_concurrent_runs` (catalog top level, default `1`) caps how many sweeps run at once across **all** experiments. The default of `1` suits a single-GPU host: each sweep's trials use the GPU, so a second concurrent sweep would contend for the device and slow both down. A `phasesweep_launch_sweep` that would exceed the cap is refused until a running sweep finishes or is cancelled. Raise it on multi-GPU hosts where independent sweeps can run side by side.

The cap counts MCP-launched runs recorded in `state_dir`; it does not count a concurrent CLI `phasesweep run` on the same host. CLI and MCP runs are still coordinated by the runtime locks described in [runtime behavior](runtime.md#concurrency-model).

## Tools

| Tool | Inputs | Effect | Returns |
| --- | --- | --- | --- |
| `phasesweep_list_experiments` | optional `limit` (1-100; default 50), `cursor` | read | catalog ids, description, phase names, metric name + goal, authorized capabilities, `total_count`, `next_cursor` |
| `phasesweep_validate_config` | `experiment_id` | read | capabilities and per-phase name, `n_trials`, sampler, inherited phases, and search-space *keys* (not ranges); a changed config is a tool error |
| `phasesweep_get_latest_run` | `experiment_id` | read | the newest durable run, selected by launch timestamp with a stable tie-breaker, or `found: false` |
| `phasesweep_get_status` | exactly one of `experiment_id` or `run_id` | read | dense state counts, computed terminal/remaining attempts, storage-read availability, winner presence, result provenance, run state, elapsed time, and suggested polling delay; terminal run-id reads require a frozen snapshot |
| `phasesweep_await_run` | `run_id`, optional `timeout_seconds` (5-600; default 120) | read (waits) | the `phasesweep_get_status` payload plus `changed` and `reason` (`terminal` / `phase_completed` / `timeout`) |
| `phasesweep_get_winners` | exactly one of `experiment_id` or `run_id` | read | objective metadata, result provenance, declared/winner counts, missing phases, all-phases completeness, and per-winner trial number, metric, policy-filtered sampled params, gate status, and partial-winner status |
| `phasesweep_launch_sweep` | `experiment_id`, optional `from_phase` | spawn detached | `{run_id, experiment_id, state}` |
| `phasesweep_cancel_sweep` | `run_id` | signal | `{run_id, state, cleanup_confirmed, recovery_required}` |

MCP annotations describe the same contract enforced by the server. Tools marked `read` in the table set `readOnlyHint=true`, `destructiveHint=false`, `idempotentHint=true`, and `openWorldHint=false`. Launch sets those hints to `false`, `true`, `false`, and `true`; cancel sets them to `false`, `true`, `true`, and `false`. Permissions, closed input schemas, and safety gates are enforced server-side even when a client ignores the hints.

A launched sweep runs as a detached background process in its own session, so it survives the agent's tool call and a server restart and can be cancelled as a group. `phasesweep_get_status` reports `running` / `succeeded` / `failed` / `cancelled`. `phasesweep_await_run` waits without preventing cancellation or other MCP calls. The packaged [agent instructions](../src/phasesweep/mcp/agent_prompt.md#workflow) define the call sequence. `from_phase` resumes from a phase whose earlier winners already exist on disk; the server checks resume-readiness before launching.

`cleanup_confirmed` on `phasesweep_cancel_sweep` is tri-state. `true` means the MCP runner process group is gone and the runner wrote terminal evidence confirming trial cleanup. `false` means cancellation was attempted but could not confirm cleanup. `null` means the run was already terminal and no cancellation was attempted. Use `recovery_required` as the decision field: when true, an operator must resolve cleanup uncertainty before the experiment can launch again. Both status and latest-run payloads also expose `recovery_required` with their run metadata. Cleanup confirmation is emitted by the engine shutdown handler after it terminates active trial process groups through the same confirmed cleanup path used by stale-trial recovery. If a spawned runner disappears without recording terminal status, if the runner group is gone but cancellation cannot observe status, or if status reports unconfirmed trial cleanup, the server writes a cleanup-uncertain marker and keeps the run counted as live so later launches do not reuse possibly-held resources. Normal runner shutdown asks the engine to tear down trial groups, and uncertain trainer leftovers are handled by the engine's stale reaper before later launches.

Cleanup and result-finalization recovery are operator-only. After inspecting the host, run `phasesweep mcp-recover-run --state-dir <state_dir> --run-id <run_id>` to verify the saved config snapshot hash and report the cleanup or terminal-snapshot actions that recovery would attempt. This preflight is observational: it sends no signals, writes no state, and requires an existing recognizable state directory. Repeat with `--confirm` to perform any required process-group cleanup, reap stale `RUNNING` trials, consume terminal cleanup evidence so it cannot clear a later run, write a failed terminal status when a dead runner omitted one, record cleanup recovery evidence, and rebuild a missing result snapshot from the hash-verified per-run config. MCP deliberately has no tool for these operator actions.

When a `run_id` is supplied, live status is read through that run's saved config snapshot, so catalog edits after launch cannot redirect monitoring. On termination, the runner first persists the terminal cause and cleanup evidence with `result_snapshot_state: pending`, then makes three bounded attempts to capture a validated, path-free snapshot of phase counts and sampled winners before recording `complete` or `failed`. This ordering preserves cancellation evidence when storage reads are slow or transiently unavailable. Later resumes cannot rewrite reads backed by a completed snapshot. Terminal run reads fail closed if the snapshot is missing or malformed; they never substitute the experiment's mutable shared-study results. A short retry can cover finalization still in progress; a persistent error can be repaired by the operator with `mcp-recover-run`. If the run's original experiment id is no longer in the active catalog, winner parameter values use the strict `visible_params: none` behavior. `phasesweep_cancel_sweep` also accepts a decataloged run id only when that run handle recorded `allow.cancel: true` at launch; runs launched without cancel permission fail closed.

## Resource and prompt

Clients that support MCP resources can attach `phasesweep://catalog`. It returns the first catalog page as compact JSON using the same path-free payload as `phasesweep_list_experiments`. Agents should still call `phasesweep_list_experiments` when they need pagination or autonomous discovery.

Clients that support MCP prompts can use `phasesweep_run_and_monitor`, which serves the packaged [agent instructions](../src/phasesweep/mcp/agent_prompt.md).

## Security model

The catalog is the trust boundary. By construction the agent **cannot**:

- set or change `trial_command`, `env`, `storage`, `workdir`, search spaces, samplers, gates, or any safety waiver - no tool accepts a config or these fields;
- reference a config by path - every tool takes an `experiment_id` resolved against the frozen catalog; an unknown id is a clean error;
- read trainer output or rendered commands - **no tool returns log text**, because the engine logs the fully rendered command (template + absolute paths) and trainer output can carry secrets or PII. Logs stay under `state_dir` for the operator to inspect directly;
- double-launch (rejected by a run-handle check and ultimately the engine's same-host lock), delete runs, or corrupt state.

Outbound payloads are built only from path-free typed views and capped at 64 KiB of compact semantic JSON before transport framing. Catalog pages stop early and return `next_cursor` when their byte budget fills; an individually oversized result fails with a safe, actionable error. `phasesweep_get_winners` returns sampled `params` and omits composed `effective_overrides`, because those can include operator-authored fixed or inherited values such as private dataset ids, paths, or tokens. Sampled-value exposure follows the catalog's `visible_params` policy above. Keep secrets, access tokens, private paths, dataset ids, hostnames, or other sensitive values out of searchable parameter choices unless you deliberately expose them through that policy. A backstop converts any unexpected exception into a path-free operator-directed error rather than leaking a traceback; recoverable domain errors are surfaced as MCP tool errors for model self-correction.

`phasesweep_get_winners` intentionally exposes each completed phase winner's objective metric value. It does not expose per-trial metric histories, raw result files, trainer logs, datasets, target/dependent-variable values, validation labels, predictions, W&B dashboards, or rendered commands. Do not give the same agent separate filesystem or dashboard access when those artifacts must stay out of its context.

This layer narrows the **agent's** authority. It does **not** sandbox the training subprocess, which remains as trusted as the human who wrote its command. Registering a malicious config runs it - your decision, identical to running `phasesweep run` by hand.

## Inspecting runs

Run handles and per-run logs live under `state_dir`:

- `state_dir/audit.jsonl` - structured MCP tool-call audit records.
- `state_dir/runs/<run_id>.json` - the run handle.
- `state_dir/logs/<run_id>.log` - captured runner stdout/stderr (operator-only).
- `state_dir/logs/<run_id>.status.json` - the recorded terminal cause, explicit result-snapshot finalization state, and, when capture succeeds, the path-free status/winner snapshot used for stable run-id reads.
- `state_dir/logs/<run_id>.config.yaml` - the exact config snapshot executed by the runner (operator-only; may contain command, storage, env, and overrides).
- `state_dir/logs/<run_id>.cleanup_uncertain.json` - server-owned marker that keeps a cleanup-uncertain run counted as live.
- `state_dir/logs/<run_id>.cleanup_recovery.json` - operator recovery evidence written by `phasesweep mcp-recover-run --confirm`.

The engine's own durable `run.log` is under the experiment `workdir`.

`audit.jsonl` contains append-only JSON objects with timestamp, local stdio actor, server session id, tool name, bounded safe arguments (`experiment_id`, `run_id`, `from_phase`, `timeout_seconds`, pagination values), resolved ids, outcome, error type/message for safe tool errors, state transition summaries, and result counts. Launch and cancellation first append and fsync an `authorized` record containing the exact catalog or run-handle policy source and config hash; if that durable write fails, the operation is refused before any launch artifact, process spawn, cleanup marker, or signal. Later success/error records are best-effort and are not fsynced. An authorization-only tail means outcome logging may have failed, so inspect the run handle, terminal status, and host state before deciding what occurred. Audit records do not include tool result payloads, trainer logs, commands, config paths, storage URLs, environment values, sampled winner params, or effective overrides.

When a client polls `phasesweep_get_status` instead of waiting on `phasesweep_await_run`, each result carries a `poll_after_seconds` suggestion sized from the median completed-trial duration (30s until anything finishes, clamped to 15-600s) - follow it rather than a tight loop. SQLite-backed status uses a read-only direct count path. Journal-backed status goes through Optuna's full read path today, so keep polling sparse on very large Journal-backed studies until the tracked aggregate-count optimization lands.

### Long-running servers

The server is built to stay up across multi-hour sweeps. Exited detached runners are reaped during status and live-run scans, the server does not hold per-run log file descriptors open, and run state is derived from disk on each query rather than kept in memory, so a server restart re-discovers live runs from their handles. A runner that exits just after a scan can remain a zombie until the next scan. Run artifacts under `state_dir/logs` accumulate one small set per launch; prune old ones between campaigns if you launch many sweeps.

Run handles, terminal `status.json` files, and per-run config snapshots are written with atomic replace, so readers do not observe torn JSON or partial snapshots. Launch persists a `launching` handle before the detached runner starts; after `Popen`, both the server and the runner persist the spawned process identity. The runner writes its own handle before launching training work, so a server restart can still rediscover a surviving runner if the server died after `Popen` but before its own final save. If the server's final spawned-handle save fails, it terminates the spawned runner rather than leaving an untracked sweep behind.
