# phasesweep MCP server

`phasesweep-mcp` and `phasesweep mcp` expose a phasesweep experiment to an AI agent over the [Model Context Protocol](https://modelcontextprotocol.io) so the agent can **launch a sweep, monitor it, and read the winning hyperparameters** - and nothing else. The agent never supplies, edits, or sees a `trial_command`, `env`, `storage`, or `workdir`. It picks an experiment from a human-curated catalog by id and calls one of six tools. The server also exposes a read-only catalog resource and one workflow prompt for clients that support them.

For copy/paste MCP client config and agent instructions, start with [MCP agent setup](mcp_setup.md).

## The catalog

The server is started with a **catalog**: a fixed allowlist mapping opaque ids
to local config paths plus per-experiment permissions. The agent only ever
sends an id; it cannot enumerate the filesystem, pass a path, or author a
config. Author the catalog with the same trust and out-of-band process as the
experiment YAML - in an editor, in git, reviewed by a human. The server never
writes it.

Catalog keys:

- `state_dir`: operator-owned directory for run handles, runner logs, config snapshots, and `audit.jsonl`.
- `max_concurrent_runs`: cap on live sweeps across all catalog entries. The default is `1`.
- `experiments[].id`: agent-visible id. It must match `[A-Za-z0-9_-]+`.
- `experiments[].config`: local experiment YAML path. Relative paths resolve against the catalog file.
- `experiments[].description`: optional text shown by `phasesweep_list_experiments` and the catalog resource.
- `experiments[].allow`: optional side-effect permissions for `launch`, `cancel`, and `from_phase`.

At startup the server resolves every `config` path to absolute (relative paths are resolved against the catalog file), validates it with the same loader the CLI uses, computes a content hash, and **refuses to start** if any config is invalid, is a suite, or uses in-memory storage (`null`, `sqlite://`, `sqlite:///:memory:`, or `:memory:`). Catalog ids must match `[A-Za-z0-9_-]+`. The id-to-path mapping is then frozen for the server's lifetime. On launch, the server verifies the config still matches the startup hash and hands the detached runner a per-run snapshot, so later edits to the original file cannot change what the runner executes.

Omitting `allow` leaves an experiment read-only: agents can list, validate, inspect status, and read existing winners, but `phasesweep_launch_sweep`, `phasesweep_cancel_sweep`, and `from_phase` resume are refused until the operator explicitly sets the corresponding flag to `true`.

The server speaks JSON-RPC over stdio; all logging goes to stderr.

### Paths and the working directory

Relative paths resolve against the directory you start the server from:
`state_dir` in the catalog and `workdir` / `storage` inside each experiment
YAML are all relative to the server's current working directory (matching the
engine's convention). The `config:` paths in the catalog are the exception -
they resolve against the catalog file. **For production, prefer absolute
paths** for `state_dir`, `workdir`, and `storage`, or always start the server
from the same project directory.

### Concurrency and single-GPU hosts

`max_concurrent_runs` (catalog top level, default `1`) caps how many sweeps run at once across **all** experiments. The default of `1` suits a single-GPU host: each sweep's trials use the GPU, so a second concurrent sweep would contend for the device and slow both down. A `phasesweep_launch_sweep` that would exceed the cap is refused until a running sweep finishes or is cancelled. Raise it on multi-GPU hosts where independent sweeps can run side by side.

The cap counts MCP-launched runs recorded in `state_dir`; it does not count a concurrent CLI `phasesweep run` on the same host. CLI and MCP runs are still coordinated by the engine's same-host output/storage locks and per-GPU device locks, so use a shared `PHASESWEEP_LOCK_DIR` when schedulers or containers would otherwise split the lock namespace.

This is separate from the per-experiment guard: the same experiment can never
double-launch (rejected by a run-handle check and ultimately the engine's
same-host lock), regardless of the cap.

## The six tools

| Tool | Inputs | Effect | Returns |
| --- | --- | --- | --- |
| `phasesweep_list_experiments` | optional `limit`, `cursor` | read | catalog ids, description, phase names, metric name + goal, `total_count`, `next_cursor` |
| `phasesweep_validate_config` | `experiment_id` | read | per-phase name, `n_trials`, sampler, inherited phases, search-space *keys* (not ranges) |
| `phasesweep_get_status` | exactly one of `experiment_id` or `run_id` | read | per-phase trial counts + winner presence, and the run process state |
| `phasesweep_get_winners` | exactly one of `experiment_id` or `run_id` | read | per-phase trial number, metric, sampled params, gate status, and completeness |
| `phasesweep_launch_sweep` | `experiment_id`, optional `from_phase` | spawn detached | `{run_id, state}` |
| `phasesweep_cancel_sweep` | `run_id` | signal | `{run_id, state, cleanup_confirmed}` |

A launched sweep runs as a **detached background process** in its own session, so it survives the agent's tool call, survives a server restart, and can be cancelled as a group. `phasesweep_get_status` reports `running` / `succeeded` / `failed` / `cancelled`. `from_phase` resumes from a phase whose earlier winners already exist on disk; the server checks resume-readiness before launching.

`phasesweep_list_experiments` defaults to 50 entries and caps `limit` at 100. If `next_cursor` is non-null, call it again with that cursor to fetch the next page.

When a `run_id` is supplied, status and winners are read from that run's saved config snapshot, so catalog edits after launch cannot redirect monitoring or winner reads.

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

Outbound payloads are built only from path-free typed views. `phasesweep_get_winners` returns sampled `params` and omits composed `effective_overrides`, because those can include operator-authored fixed or inherited values such as private dataset ids, paths, or tokens. Sampled `params` are intentionally visible so an agent can compare the winning hyperparameters; do not put secrets, access tokens, private paths, dataset ids, hostnames, or other sensitive values in searchable parameter choices. Keep sensitive values in the trainer environment or fixed config fields that MCP does not expose. A backstop converts any unexpected error into a generic `"internal error"` rather than leaking a traceback; recoverable domain errors are surfaced as MCP tool errors for model self-correction.

`phasesweep_get_winners` intentionally exposes each completed phase winner's objective metric value. It does not expose per-trial metric histories, raw result files, trainer logs, datasets, target/dependent-variable values, validation labels, predictions, W&B dashboards, or rendered commands. If those values should stay out of the agent context, keep them out of sampled parameter values and do not give the same agent separate filesystem or dashboard access to the run artifacts.

This layer narrows the **agent's** authority. It does **not** sandbox the training subprocess, which remains as trusted as the human who wrote its command. Registering a malicious config runs it - your decision, identical to running `phasesweep run` by hand.

## Inspecting runs

Run handles and per-run logs live under `state_dir`:

- `state_dir/audit.jsonl` - structured MCP tool-call audit records.
- `state_dir/runs/<run_id>.json` - the run handle.
- `state_dir/logs/<run_id>.log` - captured runner stdout/stderr (operator-only).
- `state_dir/logs/<run_id>.status.json` - the recorded terminal cause.
- `state_dir/logs/<run_id>.config.yaml` - the exact config snapshot executed by
  the runner (operator-only; may contain command, storage, env, and overrides).

The engine's own durable `run.log` is under the experiment `workdir`.

`audit.jsonl` contains one JSON object per tool call with timestamp, local stdio actor, server session id, tool name, safe arguments (`experiment_id`, `run_id`, `from_phase`), resolved ids, outcome, error type/message for safe tool errors, state transition summaries, and result counts. It does not include tool result payloads, trainer logs, commands, config paths, storage URLs, environment values, sampled winner params, or effective overrides.

### Long-running servers

The server is built to stay up across multi-hour sweeps. Detached runners are
reaped as they exit (no zombie buildup), the server does not hold per-run log
file descriptors open, and run state is derived from disk on each query rather
than kept in memory - so a server restart re-discovers live runs from their
handles. Run artifacts under `state_dir/logs` accumulate one small set per
launch; prune old ones between campaigns if you launch many sweeps.

Run handles are written with an atomic replace, so readers do not observe torn
JSON. The remaining crash window is the tiny interval after the detached runner
has spawned but before its first handle is persisted; if the server process dies
there, the runner may continue but a restarted MCP server cannot rediscover it
by `run_id`. Inspect the operator-owned logs and the engine's normal lock/stale
reaper state in that case.

## Limitations (v1)

- **Single-experiment configs only.** Suites are rejected at startup.
- **Persistent storage required.** In-memory studies cannot be monitored across
  processes, so `storage: null` and SQLite in-memory URL forms are rejected at
  startup.
- **Single host.** The server, runner, and lock assume one host.
- **Local stdio only.** Remote Streamable HTTP, OAuth, bearer-token auth, and hosted multi-user deployments are out of scope for v1.
- **No log-access tool.** A redacted, opt-in log view is a possible follow-up,
  but v1 exposes no catalog flag or tool for logs.
- The only agent-tunable knob is `from_phase`; `n_trials`, timeouts, and all
  safety waivers stay with the config author.
