# phasesweep MCP server

`phasesweep-mcp` and `phasesweep mcp` expose a phasesweep experiment to an AI agent over the [Model Context Protocol](https://modelcontextprotocol.io). The supported mode for this version is local-node control over stdio: the MCP server, detached runner, process cleanup, filesystem locks, and GPU leases all assume one machine. The agent can launch a sweep, monitor it, and read the winning hyperparameters. It never supplies, edits, or sees a `trial_command`, `env`, `storage`, or `workdir`. It picks an experiment from a human-curated catalog by id and calls one of six tools. The server also exposes a read-only catalog resource and one workflow prompt for clients that support them.

For install commands, client config, and pasteable agent instructions, use [MCP agent setup](mcp_setup.md).

## The catalog

The server starts from a catalog: a fixed allowlist mapping opaque ids to local config paths plus per-experiment permissions. The agent only ever sends an id; it cannot enumerate the filesystem, pass a path, or author a config. Author the catalog with the same trust and review process as the experiment YAML. The server never writes it.

Catalog keys:

- `state_dir`: operator-owned directory for run handles, runner logs, config snapshots, and `audit.jsonl`.
- `max_concurrent_runs`: cap on live sweeps across all catalog entries. The default is `1`.
- `experiments[].id`: agent-visible id. It must match `[A-Za-z0-9_-]+`.
- `experiments[].config`: local experiment YAML path. Relative paths resolve against the catalog file.
- `experiments[].cwd`: optional detached-runner working directory. Relative paths resolve against the catalog file. The default is the registered config file's directory.
- `experiments[].visible_params`: sampled winner parameter values exposed to agents. Use `none` (default), `all`, or a list of allowed parameter keys. Parameter names remain visible; values outside the policy are returned as `<redacted>`.
- `experiments[].description`: optional text shown by `phasesweep_list_experiments` and the catalog resource.
- `experiments[].allow`: optional side-effect permissions for `launch`, `cancel`, and `from_phase`.

At startup the server validates each experiment with the same loader the CLI uses, computes a content hash, and refuses invalid configs, suites, in-memory or empty file-backed storage (`null`, `sqlite://`, `sqlite:///:memory:`, `:memory:`, `journal://`, or `journal:///`), external RDB storage URLs, and configs that violate [path stability](#paths-and-the-working-directory). The id-to-path mapping is then frozen for the server's lifetime. On launch, the server verifies the config still matches the startup hash and hands the detached runner a per-run snapshot, so later edits to the original file cannot change what the runner executes.

Omitting `allow` leaves an experiment read-only: agents can list, validate, inspect status, and read existing winners, but `phasesweep_launch_sweep`, `phasesweep_cancel_sweep`, and `from_phase` resume are refused until the operator explicitly sets the corresponding flag to `true`.

The server speaks JSON-RPC over stdio; all logging goes to stderr.

### Paths and the working directory

`state_dir`, `config:`, and `experiments[].cwd` paths in the catalog resolve against the catalog file when they are relative. MCP experiment configs must use absolute `workdir` values and non-empty absolute SQLite/Journal storage paths so server restarts, wrappers, IDE launches, and desktop clients monitor the same local-node artifacts and Optuna studies. External RDB storage is rejected for MCP because the current cleanup, stale-trial reaping, and GPU lock semantics are same-host only. The detached runner always starts with the catalog entry's frozen `cwd`, defaulting to the registered config file's directory. Relative paths inside `trial_command` are trainer-owned shell behavior; phasesweep does not parse or rewrite commands, but their base directory is no longer inherited from the MCP server process.

### Concurrency and single-GPU hosts

`max_concurrent_runs` (catalog top level, default `1`) caps how many sweeps run at once across **all** experiments. The default of `1` suits a single-GPU host: each sweep's trials use the GPU, so a second concurrent sweep would contend for the device and slow both down. A `phasesweep_launch_sweep` that would exceed the cap is refused until a running sweep finishes or is cancelled. Raise it on multi-GPU hosts where independent sweeps can run side by side.

The cap counts MCP-launched runs recorded in `state_dir`; it does not count a concurrent CLI `phasesweep run` on the same host. CLI and MCP runs are still coordinated by the runtime locks described in [runtime behavior](runtime.md#concurrency-model).

## The six tools

| Tool | Inputs | Effect | Returns |
| --- | --- | --- | --- |
| `phasesweep_list_experiments` | optional `limit`, `cursor` | read | catalog ids, description, phase names, metric name + goal, `total_count`, `next_cursor` |
| `phasesweep_validate_config` | `experiment_id` | read | per-phase name, `n_trials`, sampler, inherited phases, search-space *keys* (not ranges) |
| `phasesweep_get_status` | exactly one of `experiment_id` or `run_id` | read | per-phase trial counts + winner presence, and the run process state |
| `phasesweep_get_winners` | exactly one of `experiment_id` or `run_id` | read | per-phase trial number, metric, policy-filtered sampled params, gate status, and completeness |
| `phasesweep_launch_sweep` | `experiment_id`, optional `from_phase` | spawn detached | `{run_id, state}` |
| `phasesweep_cancel_sweep` | `run_id` | signal | `{run_id, state, cleanup_confirmed}` |

A launched sweep runs as a **detached background process** in its own session, so it survives the agent's tool call, survives a server restart, and can be cancelled as a group. `phasesweep_get_status` reports `running` / `succeeded` / `failed` / `cancelled`. `from_phase` resumes from a phase whose earlier winners already exist on disk; the server checks resume-readiness before launching.

`cleanup_confirmed` on `phasesweep_cancel_sweep` means the MCP runner process group is gone and the runner wrote a readable terminal status whose own `cleanup_confirmed` field is `true`. That field is emitted by the engine shutdown handler after it terminates active trial process groups through the same confirmed cleanup path used by stale-trial recovery. If the runner group is gone but no status was recorded, or the status reports unconfirmed trial cleanup, the server writes a cleanup-uncertain marker and keeps the run counted as live so later launches do not reuse possibly-held resources. A terminal runner failure whose status records `cleanup_confirmed: false` is also counted as live until operator recovery confirms cleanup. Normal runner shutdown asks the engine to tear down trial groups, and uncertain trainer leftovers are handled by the engine's stale reaper before later launches.

Cleanup-uncertain recovery is operator-only. After inspecting the host, run `phasesweep mcp-recover-run --state-dir <state_dir> --run-id <run_id>` to verify the saved config snapshot hash, re-check the runner identity, and invoke the engine stale-trial reaper against that snapshot. If it reports that cleanup appears confirmed, repeat with `--confirm` to write recovery evidence and clear the uncertainty. MCP deliberately has no tool for clearing this state.

`phasesweep_list_experiments` defaults to 50 entries and caps `limit` at 100. If `next_cursor` is non-null, call it again with that cursor to fetch the next page.

When a `run_id` is supplied, status and winners are read from that run's saved config snapshot, so catalog edits after launch cannot redirect monitoring or winner reads. If the run's original experiment id is no longer in the active catalog, winner parameter values use the strict `visible_params: none` behavior.

## Resource and prompt

Clients that support MCP resources can attach `phasesweep://catalog`. It returns the first catalog page as compact JSON using the same path-free payload as `phasesweep_list_experiments`. Agents should still call `phasesweep_list_experiments` when they need pagination or autonomous discovery.

Clients that support MCP prompts can use `phasesweep_run_and_monitor`. It gives the agent the safe workflow: list, validate, launch only by catalog id, poll by `run_id`, summarize winners with that same `run_id`, and avoid raw datasets, labels, predictions, trainer logs, raw result files, dashboards, and per-trial metric histories unless the user explicitly asks for that separate work.

## Security model

The catalog is the trust boundary. By construction the agent **cannot**:

- set or change `trial_command`, `env`, `storage`, `workdir`, search spaces,
  samplers, gates, or any safety waiver - no tool accepts a config or these
  fields;
- reference a config by path - every tool takes an `experiment_id` resolved
  against the frozen catalog; an unknown id is a clean error;
- read trainer output or rendered commands - **no tool returns log text**,
  because the engine logs the fully rendered command (template + absolute
  paths) and trainer output can carry secrets or PII. Logs stay under
  `state_dir` for the operator to inspect directly;
- double-launch (rejected by a run-handle check and ultimately the engine's
  same-host lock), delete runs, or corrupt state.

Outbound payloads are built only from path-free typed views. `phasesweep_get_winners` returns sampled `params` and omits composed `effective_overrides`, because those can include operator-authored fixed or inherited values such as private dataset ids, paths, or tokens. Sampled param names are visible so an agent can see which hyperparameters won; sampled values default to `<redacted>` unless the catalog entry sets `visible_params: all` or an allowlist such as `visible_params: [lr, weight_decay]`. Keep secrets, access tokens, private paths, dataset ids, hostnames, or other sensitive values out of searchable parameter choices unless you deliberately expose them through that policy. A backstop converts any unexpected error into a generic `"internal error"` rather than leaking a traceback; recoverable domain errors are surfaced as MCP tool errors for model self-correction.

`phasesweep_get_winners` intentionally exposes each completed phase winner's objective metric value. It does not expose per-trial metric histories, raw result files, trainer logs, datasets, target/dependent-variable values, validation labels, predictions, W&B dashboards, or rendered commands. If sampled parameter values should stay out of the agent context, keep `visible_params` at the default `none` and do not give the same agent separate filesystem or dashboard access to the run artifacts.

This layer narrows the **agent's** authority. It does **not** sandbox the training subprocess, which remains as trusted as the human who wrote its command. Registering a malicious config runs it - your decision, identical to running `phasesweep run` by hand.

## Inspecting runs

Run handles and per-run logs live under `state_dir`:

- `state_dir/audit.jsonl` - structured MCP tool-call audit records.
- `state_dir/runs/<run_id>.json` - the run handle.
- `state_dir/logs/<run_id>.log` - captured runner stdout/stderr (operator-only).
- `state_dir/logs/<run_id>.status.json` - the recorded terminal cause.
- `state_dir/logs/<run_id>.config.yaml` - the exact config snapshot executed by
  the runner (operator-only; may contain command, storage, env, and overrides).
- `state_dir/logs/<run_id>.cleanup_recovery.json` - operator recovery evidence written by `phasesweep mcp-recover-run --confirm`.

The engine's own durable `run.log` is under the experiment `workdir`.

`audit.jsonl` contains one JSON object per tool call with timestamp, local stdio actor, server session id, tool name, bounded safe arguments (`experiment_id`, `run_id`, `from_phase`, pagination values), resolved ids, outcome, error type/message for safe tool errors, state transition summaries, and result counts. It does not include tool result payloads, trainer logs, commands, config paths, storage URLs, environment values, sampled winner params, or effective overrides.

Poll `phasesweep_get_status` at a normal agent cadence rather than in a tight loop. SQLite-backed status uses a read-only direct count path; Journal-backed status uses Optuna's read path today, so large local studies should be polled every few seconds until the tracked aggregate-count optimization is implemented.

### Long-running servers

The server is built to stay up across multi-hour sweeps. Detached runners are
reaped as they exit (no zombie buildup), the server does not hold per-run log
file descriptors open, and run state is derived from disk on each query rather
than kept in memory - so a server restart re-discovers live runs from their
handles. Run artifacts under `state_dir/logs` accumulate one small set per
launch; prune old ones between campaigns if you launch many sweeps.

Run handles, terminal `status.json` files, and per-run config snapshots are written with atomic replace, so readers do not observe torn JSON or partial snapshots. Launch persists a `launching` handle before the detached runner starts; after `Popen`, both the server and the runner persist the spawned process identity. The runner writes its own handle before launching training work, so a server restart can still rediscover a surviving runner if the server died after `Popen` but before its own final save. If the server's final spawned-handle save fails, it terminates the spawned runner rather than leaving an untracked sweep behind.

## Limitations (v1)

- **Single-experiment configs only.** Suites are rejected at startup.
- **Local file-backed storage required.** In-memory studies, empty file-backed storage URLs, and external RDB storage are rejected at startup. Use non-empty absolute SQLite or JournalStorage file URLs for MCP catalogs.
- **Single local node.** The server, runner, cleanup recovery, stale-trial reaper, and locks assume one host. Remote control planes, multi-host writers, and distributed cleanup are future work, not warnings-only behavior.
- **Local stdio only.** Remote Streamable HTTP, OAuth, bearer-token auth, and hosted multi-user deployments are out of scope for v1.
- **No log-access tool.** A redacted, opt-in log view is a possible follow-up,
  but v1 exposes no catalog flag or tool for logs.
- The only agent-tunable knob is `from_phase`; `n_trials`, timeouts, and all
  safety waivers stay with the config author.
