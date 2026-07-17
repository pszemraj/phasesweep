# MCP agent setup

Install the extra, create a catalog, connect a client, verify the connection, and give the agent its operating instructions. Tool behavior, catalog rules, and the security model are covered in [MCP server](mcp.md).

## 1. Install

Install the MCP extra in the Python environment your MCP client will use:

```bash
python -m pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

Or from a local checkout:

```bash
python -m pip install -e ".[mcp]"
```

Client configs want the executable path absolute, because clients launch servers outside your shell environment. `phasesweep install` (step 3) resolves it for you, so you only need the path for manual setup - in that case, find it now (`phasesweep-mcp` and `phasesweep mcp` start the same server; client configs use the dedicated executable):

```bash
which phasesweep-mcp
```

If you prefer not to install into a persistent environment, every client below can instead launch through `uvx`, as shown in step 3's manual setup section. The first launch downloads and builds the package, so allow for a slow cold start.

## 2. Create a catalog

Scaffold a catalog next to your project:

```bash
phasesweep init-catalog --from ./experiment.yaml   # add --from per experiment; -o to name the file
```

This writes an annotated `catalog.yaml` only after every entry passes the server's startup checks. Each entry starts with side effects disabled and winner values redacted. Fill in its description, review the generated paths, and enable only the actions and parameter values the agent should receive. See [the catalog](mcp.md#the-catalog) for every field and validation rule. [examples/catalog.yaml](../examples/catalog.yaml) is a working catalog for [examples/mcp_experiment.yaml](../examples/mcp_experiment.yaml).

Confirm the catalog loads before touching any client config:

```bash
phasesweep mcp-check --catalog /abs/path/to/catalog.yaml
```

Fix every reported failure before connecting a client. The command exits 0 only when the catalog and state directory pass the same checks used at server startup.

## 3. Connect your client

One command writes both integrations for your coding agents - the MCP server entry, project-scoped wherever the client supports it, and the step-5 agent instructions as a marker-fenced block:

```bash
phasesweep install                        # interactive: confirm each detected agent, review the plan, apply
phasesweep install --agent claude --yes   # unattended; repeat --agent for more
```

`install` first confirms the MCP SDK from step 1 is available, then validates the catalog with the exact server startup rules before touching any client config (offering to scaffold one if it is missing), prints a plan of every file it will edit, and reports one `created` / `updated` / `unchanged` line per edit. The server command is the executable beside the Python interpreter running `phasesweep` (the active conda/virtual environment), with `PATH` as a fallback; installation stops before edits if neither is launchable. Supported agents: `claude` (Claude Code), `claude-desktop`, `codex`, `cursor`, `vscode`, `gemini`, `opencode`. A config that is not strict JSON (comments, JSON5) is never modified - the report says `skipped` and prints the exact snippet to merge manually. Codex TOML receives the same fail-safe treatment: invalid TOML, an existing unmanaged `mcp_servers.phasesweep` entry, or a merge that would not parse is left untouched with a manual snippet. `--type mcp|instructions` installs one integration only; instructions-only installation does not require the MCP SDK. `--project DIR` targets another project root; the catalog defaults to `./catalog.yaml` in that project - use `--catalog PATH` for any other name or location. `phasesweep uninstall` removes exactly what install wrote: the server entry by name, the instructions block by its markers, and any file that becomes empty.

Two placements are user-scoped rather than project-scoped, and the plan flags them: Claude Desktop (single user-level config) and Codex (`~/.codex/config.toml` - Codex reads project configs only in trusted projects). A user-scoped entry means that client sees this project's sweeps from every directory.

Restart the client after any config change.

<details>
<summary>Manual setup (any client)</summary>

Every client gets the same server: command `phasesweep-mcp`, args `--catalog /abs/path/to/catalog.yaml`. Use the absolute executable path from step 1 and an absolute catalog path throughout. Prefer project scope where the client supports it - a catalog belongs to one project, and a user-global entry would let an agent in any repo control this project's sweeps.

Claude Code can add the entry from the project root:

```bash
claude mcp add phasesweep --scope project -- /abs/path/to/venv/bin/phasesweep-mcp --catalog /abs/path/to/catalog.yaml
```

Claude Code, Claude Desktop, Cursor, and Gemini CLI use the same `mcpServers` entry:

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

| Client | Config path |
| --- | --- |
| Claude Code | `.mcp.json` |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS; `~/.config/Claude/claude_desktop_config.json` on Linux |
| Cursor | `.cursor/mcp.json` |
| Gemini CLI | `.gemini/settings.json` |

Codex uses TOML at `~/.codex/config.toml`:

```toml
[mcp_servers.phasesweep]
command = "/abs/path/to/venv/bin/phasesweep-mcp"
args = ["--catalog", "/abs/path/to/catalog.yaml"]
```

VS Code uses `servers` and requires `type: stdio` in `.vscode/mcp.json`:

```json
{
  "servers": {
    "phasesweep": {
      "type": "stdio",
      "command": "/abs/path/to/venv/bin/phasesweep-mcp",
      "args": ["--catalog", "/abs/path/to/catalog.yaml"]
    }
  }
}
```

OpenCode uses a command array in `opencode.json`:

```json
{
  "mcp": {
    "phasesweep": {
      "type": "local",
      "command": ["/abs/path/to/venv/bin/phasesweep-mcp", "--catalog", "/abs/path/to/catalog.yaml"],
      "enabled": true
    }
  }
}
```

Any stdio client can also launch through `uvx` without a persistent install. Replace the standard entry's command and args with:

```json
"command": "uvx",
"args": ["--from", "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git", "phasesweep-mcp", "--catalog", "/abs/path/to/catalog.yaml"]
```

Or launch the module through a known interpreter:

```json
"command": "/abs/path/to/venv/bin/python",
"args": ["-m", "phasesweep.mcp.server", "--catalog", "/abs/path/to/catalog.yaml"]
```

</details>

## 4. Verify

Restart the client, then ask the agent:

```text
List the available phasesweep experiments.
```

A working setup returns your catalog entries with ids, descriptions, phase names, and the metric - and nothing path-shaped. If the tool is missing or the call fails, see [Troubleshooting](#troubleshooting).

## 5. Instruct the agent

If step 3 ran with instructions enabled, the client already received the [agent instructions](../src/phasesweep/mcp/agent_prompt.md) in its project instructions file. Clients with MCP prompt support can load `phasesweep_run_and_monitor` instead. Otherwise, copy the linked instructions into the agent's project instructions or the chat before asking it to run a sweep.

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
