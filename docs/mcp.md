# phasesweep MCP server

`phasesweep-mcp` exposes a phasesweep experiment to an AI agent over the
[Model Context Protocol](https://modelcontextprotocol.io) so the agent can
**launch a sweep, monitor it, and read the winning hyperparameters** — and
nothing else. The agent never supplies, edits, or sees a `trial_command`,
`env`, `storage`, or `workdir`. It picks an experiment from a human-curated
catalog by id and calls one of six tools.

## Install

```bash
pip install "phasesweep[mcp]"
```

## The catalog

The server is started with a **catalog**: a fixed allowlist mapping opaque ids
to local config paths plus per-experiment permissions. The agent only ever
sends an id; it cannot enumerate the filesystem, pass a path, or author a
config. Author the catalog with the same trust and out-of-band process as the
experiment YAML — in an editor, in git, reviewed by a human. The server never
writes it.

```yaml
# examples/catalog.yaml
state_dir: ./runs/.mcp          # run handles + per-run logs (operator-owned)
max_concurrent_runs: 1          # sweeps running at once across all experiments (single GPU -> 1)
experiments:
  - id: tiny-lm                 # the only token the agent ever sends
    config: ./experiment.yaml   # resolved relative to this catalog, frozen at startup
    description: "16 MB LM: pick depth, then lr, then regularization"
    allow:
      launch: true              # all default true; set false to make an experiment read-only
      cancel: true
      from_phase: true
```

At startup the server resolves every `config` path to absolute (relative paths
are resolved against the catalog file), validates it with the same loader the
CLI uses, computes a content hash, and **refuses to start** if any config is
invalid, is a suite, or uses in-memory storage. Catalog ids must match
`[A-Za-z0-9_-]+`. The id→path mapping is then frozen for the server's lifetime.

## Start the server

```bash
phasesweep-mcp --catalog examples/catalog.yaml
```

The server speaks JSON-RPC over stdio; all logging goes to stderr. Wire it into
an MCP client (for example, Claude Desktop) as a stdio server:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/examples/catalog.yaml"]
    }
  }
}
```

### Paths and the working directory

Relative paths resolve against the directory you start the server from:
`state_dir` in the catalog and `workdir` / `storage` inside each experiment
YAML are all relative to the server's current working directory (matching the
engine's convention). The `config:` paths in the catalog are the exception —
they resolve against the catalog file. **For production, prefer absolute
paths** for `state_dir`, `workdir`, and `storage`, or always start the server
from the same project directory.

### Concurrency and single-GPU hosts

`max_concurrent_runs` (catalog top level, default `1`) caps how many sweeps run
at once across **all** experiments. The default of `1` suits a single-GPU host:
each sweep's trials use the GPU, so a second concurrent sweep would contend for
the device and slow both down. A `launch_sweep` that would exceed the cap is
refused until a running sweep finishes or is cancelled. Raise it on multi-GPU
hosts where independent sweeps can run side by side.

This is separate from the per-experiment guard: the same experiment can never
double-launch (rejected by a run-handle check and ultimately the engine's
same-host lock), regardless of the cap.

## The six tools

| Tool | Inputs | Effect | Returns |
| --- | --- | --- | --- |
| `list_experiments` | — | read | catalog ids, description, phase names, metric name + goal |
| `validate_config` | `experiment_id` | read | per-phase name, `n_trials`, sampler, inherited phases, search-space *keys* (not ranges) |
| `get_status` | `experiment_id` or `run_id` | read | per-phase trial counts + winner presence, and the run process state |
| `get_winners` | `experiment_id` | read | per-phase trial number, metric, params, and full effective overrides |
| `launch_sweep` | `experiment_id`, optional `from_phase` | spawn detached | `{run_id, state}` |
| `cancel_sweep` | `run_id` | signal | `{run_id, state, cleanup_confirmed}` |

A launched sweep runs as a **detached background process** in its own session,
so it survives the agent's tool call, survives a server restart, and can be
cancelled as a group. `get_status` reports `running` / `succeeded` / `failed` /
`cancelled`. `from_phase` resumes from a phase whose earlier winners already
exist on disk; the server checks resume-readiness before launching.

## Security model

The catalog is the trust boundary. By construction the agent **cannot**:

- set or change `trial_command`, `env`, `storage`, `workdir`, search spaces,
  samplers, gates, or any safety waiver — no tool accepts a config or these
  fields;
- reference a config by path — every tool takes an `experiment_id` resolved
  against the frozen catalog; an unknown id is a clean error;
- read trainer output or rendered commands — **no tool returns log text**,
  because the engine logs the fully rendered command (template + absolute
  paths) and trainer output can carry secrets or PII. Logs stay under
  `state_dir` for the operator to inspect directly;
- double-launch (rejected by a run-handle check and ultimately the engine's
  same-host lock), delete runs, or corrupt state.

Outbound payloads are built only from path-free typed views, so there is
structurally nothing to scrub; a backstop converts any unexpected error into a
generic `"internal error"` rather than leaking a traceback.

This layer narrows the **agent's** authority. It does **not** sandbox the
training subprocess, which remains as trusted as the human who wrote its
command. Registering a malicious config runs it — your decision, identical to
running `phasesweep run` by hand. A secret placed in `fixed_overrides` will
surface in `get_winners` (it is, by design, "the best parameters"); that is
config hygiene, not an MCP leak.

## Inspecting runs

Run handles and per-run logs live under `state_dir`:

- `state_dir/runs/<run_id>.json` — the run handle.
- `state_dir/logs/<run_id>.log` — captured runner stdout/stderr (operator-only).
- `state_dir/logs/<run_id>.status.json` — the recorded terminal cause.

The engine's own durable `run.log` is under the experiment `workdir`.

### Long-running servers

The server is built to stay up across multi-hour sweeps. Detached runners are
reaped as they exit (no zombie buildup), the server does not hold per-run log
file descriptors open, and run state is derived from disk on each query rather
than kept in memory — so a server restart re-discovers live runs from their
handles. Run artifacts under `state_dir/logs` accumulate one small set per
launch; prune old ones between campaigns if you launch many sweeps.

## Limitations (v1)

- **Single-experiment configs only.** Suites are rejected at startup.
- **Persistent storage required.** In-memory studies cannot be monitored across
  processes, so `storage: null` is rejected at startup.
- **Single host.** The server, runner, and lock assume one host.
- **No log-access tool.** A redacted, opt-in `get_trial_logs` (behind the
  reserved `expose_trial_logs` catalog flag, present but unused in v1) is a
  possible follow-up.
- The only agent-tunable knob is `from_phase`; `n_trials`, timeouts, and all
  safety waivers stay with the config author.
