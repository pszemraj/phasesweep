# MCP agent setup

This page takes you from a fresh environment to an agent that can launch, monitor, and summarize phasesweep runs — in five steps: install the extra, write a catalog, connect your client, verify, and instruct the agent. The agent operates by catalog id only; it never sees or supplies a `trial_command`, `env`, `storage`, or `workdir`. This version is local-node only: the server, detached runner, cleanup recovery, and GPU/process locks assume one machine. Tool behavior, catalog rules, and the security model are in [MCP server](mcp.md).

## 1. Install

Install the MCP extra in the Python environment your MCP client will use:

```bash
python -m pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

Or from a local checkout:

```bash
python -m pip install -e ".[mcp]"
```

Client configs want the executable path absolute, because clients launch servers outside your shell environment. `phasesweep install` (step 3) resolves it for you; for manual setup, find it now:

```bash
which phasesweep-mcp
```

If you prefer not to install into a persistent environment, every client below can instead launch through `uvx` (shown once in [Any stdio client](#any-stdio-client)); note the first launch downloads and builds the package, so allow for a slow cold start.

## 2. Create a catalog

The server refuses to start without a catalog: a fixed allowlist mapping stable ids to reviewed experiment configs, plus per-experiment permissions. Scaffold one next to your project:

```bash
phasesweep init-catalog --from ./experiment.yaml   # add --from per experiment; -o to name the file
```

This writes an annotated, read-only `catalog.yaml` (validated with the exact server startup rules first — on failure it prints the per-entry report and writes nothing). Fill in each `description`, then deliberately enable `allow` flags and `visible_params` as needed. The equivalent hand-written catalog:

```yaml
# catalog.yaml
state_dir: /abs/path/to/project/runs/.mcp   # run handles, logs, audit.jsonl (operator-owned)
max_concurrent_runs: 1                      # 1 keeps a single-GPU host sane
experiments:
  - id: my-sweep                            # the only token the agent ever sends
    config: ./experiment.yaml               # resolved relative to this catalog file
    description: "16 MB LM: pick depth, then lr, then regularization"
    visible_params: all                     # default is none: winner values return <redacted>
    allow:                                  # side effects default to false; opt in deliberately
      launch: true
      cancel: true
      from_phase: true
```

Omitting `allow` leaves an entry read-only (list, validate, status, and existing winners still work). MCP experiment configs must use absolute `workdir` values and non-empty absolute SQLite/Journal storage paths; external RDB storage is rejected. Full key semantics, path rules, and validation behavior: [the catalog](mcp.md#the-catalog). A working reference is [examples/catalog.yaml](../examples/catalog.yaml), which points at [examples/mcp_experiment.yaml](../examples/mcp_experiment.yaml).

Confirm the catalog loads before touching any client config:

```bash
phasesweep mcp-check --catalog /abs/path/to/catalog.yaml
```

`mcp-check` runs the exact validation the server applies at startup and prints one `ok` / `FAIL` line per experiment — the offending rule and a suggested fix on failures, the enabled actions on successes. It exits 0 when every entry loads and 2 otherwise. Fix and re-run until green; a green report means the server will boot with this catalog.

## 3. Connect your client

One command writes both integrations for your coding agents — the MCP server entry, project-scoped wherever the client supports it, and the step-5 agent instructions as a marker-fenced block:

```bash
phasesweep install                        # interactive: confirm each detected agent, review the plan, apply
phasesweep install --agent claude --yes   # unattended; repeat --agent for more
```

`install` validates the catalog with the exact server startup rules before touching any client config (offering to scaffold one if it is missing), prints a plan of every file it will edit, then reports one `created` / `updated` / `unchanged` line per edit. Supported agents: `claude` (Claude Code), `claude-desktop`, `codex`, `cursor`, `vscode`, `gemini`, `opencode`. A config that is not strict JSON (comments, JSON5) is never modified — the report says `skipped` and prints the exact snippet to merge manually. `--type mcp|instructions` installs one integration only; `--project DIR` targets another project root; `--catalog PATH` picks a catalog not named `./catalog.yaml`. `phasesweep uninstall` removes exactly what install wrote: the server entry by name, the instructions block by its markers, and any file that becomes empty.

Two placements are user-scoped rather than project-scoped, and the plan flags them: Claude Desktop (single user-level config) and Codex (`~/.codex/config.toml` — Codex reads project configs only in trusted projects). A user-scoped entry means that client sees this project's sweeps from every directory.

Restart the client after any config change.

<details>
<summary>Manual setup (any client)</summary>

Every client gets the same server: command `phasesweep-mcp`, args `--catalog /abs/path/to/catalog.yaml`. Use the absolute executable path from step 1 and an absolute catalog path throughout. Prefer project scope where the client supports it — a catalog belongs to one project, and a user-global entry would let an agent in any repo control this project's sweeps.

<details>
<summary>Claude Code</summary>

One command, from the project root:

```bash
claude mcp add phasesweep --scope project -- /abs/path/to/venv/bin/phasesweep-mcp --catalog /abs/path/to/catalog.yaml
```

This writes `.mcp.json` in the project. Equivalent manual entry:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

</details>

<details>
<summary>Claude Desktop</summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows; note phasesweep itself requires Linux/macOS or WSL on the host running the server):

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

</details>

<details>
<summary>Codex</summary>

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.phasesweep]
command = "/abs/path/to/venv/bin/phasesweep-mcp"
args = ["--catalog", "/abs/path/to/catalog.yaml"]
```

</details>

<details>
<summary>Cursor</summary>

Add to `.cursor/mcp.json` in the project (preferred) or `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

</details>

<details>
<summary>VS Code</summary>

Add to `.vscode/mcp.json` in the project (note the key is `servers`, not `mcpServers`):

```json
{
  "servers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

</details>

<details>
<summary>Gemini CLI</summary>

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

</details>

<details>
<summary>OpenCode</summary>

Add to `~/.config/opencode/opencode.jsonc` (note `type: local` and the command as a single array):

```json
{
  "mcp": {
    "phasesweep": {
      "type": "local",
      "command": ["/abs/path/to/venv/bin/phasesweep-mcp", "--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

</details>

<details>
<summary>Any stdio client</summary>

Any MCP client that can launch a stdio server works with the same shape. Three launch variants, in order of preference:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

Through `uvx`, with no persistent install (slow first launch while the Git package builds):

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "uvx",
      "args": ["--from", "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git", "phasesweep-mcp", "--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

Or as a module, when only the interpreter path is convenient:

```json
{
  "mcpServers": {
    "phasesweep": {
      "command": "/abs/path/to/venv/bin/python",
      "args": ["-m", "phasesweep.mcp.server", "--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

`phasesweep mcp --catalog ...` is also equivalent to `phasesweep-mcp --catalog ...` anywhere above.

</details>

</details>

## 4. Verify

Restart the client, then ask the agent:

```text
List the available phasesweep experiments.
```

A working setup returns your catalog entries with ids, descriptions, phase names, and the metric — and nothing path-shaped. If the tool is missing or the call fails, see [Troubleshooting](#troubleshooting).

## 5. Instruct the agent

If step 3's `phasesweep install` ran with instructions enabled (the default), this step is already done: the exact text below now sits between `<!-- PHASESWEEP_START -->` markers in each agent's project instructions file (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, or `.github/copilot-instructions.md`).

Otherwise: if your client supports MCP prompts (Claude Code and Claude Desktop do), load the `phasesweep_run_and_monitor` prompt — it serves the exact text below, and nothing needs pasting. If your client supports MCP resources, `phasesweep://catalog` exposes the first catalog page; agents should still call `phasesweep_list_experiments` for pagination. As a last resort, paste this into the agent's project instructions or into chat before asking it to run a sweep. This is the same text the server ships as its MCP prompt; the packaged copy at `src/phasesweep/mcp/agent_prompt.md` is the source of truth.

```text
You have access to a phasesweep MCP server. It runs phase-chained hyperparameter sweeps from a human-curated catalog of experiments: each phase's winning hyperparameters lock in as fixed overrides for every phase downstream. You operate entirely by catalog experiment id. No tool accepts a config path, trainer command, or file, and the catalog is the sole authority for paths, commands, environment, storage, and working directories — never ask the user for those or try to infer them.

## Workflow

1. Call phasesweep_list_experiments to see what exists: ids, descriptions, phase names, and the metric with its goal. If next_cursor is non-null, call again with that cursor to page.
2. Call phasesweep_validate_config with the experiment_id you plan to run. It confirms the config loads and returns each phase's name, trial count, sampler, inherited phases, and search-space keys. Do this before every launch.
3. Call phasesweep_launch_sweep with that experiment_id. It returns {run_id, state}; the sweep runs as a detached background process that survives your tool call and even a server restart. Save the run_id — it is your handle for everything after this point. Pass from_phase only when the user explicitly asks to resume from a phase, or when earlier phase winners are already confirmed complete.
4. Monitor with phasesweep_await_run using the run_id — not the experiment id, so catalog edits after launch cannot redirect your monitoring. It blocks until the run reaches a terminal state (succeeded, failed, or cancelled), a phase gains a winner, or its timeout elapses; call it again until the state is terminal, reporting per-phase completed counts as they move. If your client cannot wait on long tool calls, poll phasesweep_get_status instead and wait poll_after_seconds between calls.
5. On a terminal state, call phasesweep_get_winners with the same run_id and summarize each phase: winning trial number, metric value, sampled params, gate status, and whether every phase completed.

When the user asks for a recommended next experiment, base it only on MCP outputs: catalog descriptions, phase shape, status counts, exposed winner metrics, and sampled params that are not redacted.

## Boundaries

- <redacted> sampled-param values are intentional catalog policy, not missing data. Report them as withheld.
- Do not inspect raw datasets, target or label columns, predictions, trainer logs, raw result files, W&B dashboards, or per-trial metric histories unless the user explicitly asks for that as separate filesystem or dashboard work.
- Do not change the objective metric, extractor, trainer command, search space, samplers, constraints, gates, storage, workdir, environment, or safety waivers unless the user explicitly asks for config-authoring help.
- Call phasesweep_cancel_sweep with the run_id only when the user asks, or when stopping is clearly necessary to prevent an unwanted active sweep. If the result reports cleanup_confirmed: false, tell the user; recovery is operator-only (phasesweep mcp-recover-run), and no MCP tool can clear it.
- A refusal such as "action 'launch' is not permitted" or a concurrency-limit error is deliberate catalog policy. Report it to the user; do not retry or work around it.
```

## Requests that work well

- `List the available phasesweep experiments and validate the tiny LM example.`
- `Launch the tiny-lm sweep, monitor it until completion, then summarize each phase winner.`
- `Check whether run <run_id> is still active and show phase-level trial counts.`
- `Read the current winners for <experiment_id> and suggest a next manual experiment using only MCP outputs.`

## Troubleshooting

- The client cannot find `phasesweep-mcp`: use the absolute path to the virtualenv or conda executable in the client config (`which phasesweep-mcp`).
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: set the corresponding `allow` flag to `true` on that catalog entry and restart the MCP client.
- `concurrency limit reached`: wait for an active MCP run to finish, cancel it, or raise `max_concurrent_runs` on hosts that can safely run multiple sweeps.
- Relative path rejected at startup: catalog `state_dir`, `config`, and `cwd` may be relative to the catalog file, but experiment `workdir` and file-backed SQLite/Journal storage paths must be non-empty and absolute for MCP.
- `storage must be persistent`, an empty file-backed storage error, or an external RDB storage error: use a persistent Optuna storage URL with a non-empty absolute SQLite/Journal path; MCP catalogs are intentionally local-node only in this version.
- A cancelled or failed run stays `running` with `cleanup_confirmed: false`: inspect the host for leftover runner/trial process groups, then run `phasesweep mcp-recover-run --state-dir <state_dir> --run-id <run_id>` and repeat with `--confirm` only if the dry run reports confirmed cleanup.
- Old MCP runs clutter status or logs: inspect `state_dir/runs` and `state_dir/logs`, then archive or prune terminal run handles between campaigns.
