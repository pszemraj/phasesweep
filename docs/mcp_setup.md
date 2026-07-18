# MCP agent setup

Install the extra, create a catalog, connect a client, verify the connection, and give the agent its operating instructions. Tool behavior, catalog rules, and the security model are covered in [MCP server](mcp.md).

## Prerequisites

- The MCP [platform requirements](runtime.md#platform-support).
- Python 3.10 or newer.
- At least one phasesweep experiment config that already passes `phasesweep validate`.
- A coding client with local stdio MCP support. Local use requires no API key or OAuth credential.

## 1. Install

Install the MCP extra in the Python environment your MCP client will use:

```bash
python -m pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

Or from a local checkout:

```bash
python -m pip install -e ".[mcp]"
```

Client configs want the executable path absolute, because clients launch servers outside your shell environment. `phasesweep mcp install` (step 3) resolves it for you, so you only need the path for manual setup - in that case, find it now (`phasesweep-mcp` and `phasesweep mcp serve` start the same server; client configs use the dedicated executable):

```bash
which phasesweep-mcp
```

If you prefer not to install into a persistent environment, every client below can instead launch through `uvx`, an alias for `uv tool run`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then run `command -v uvx` and use the returned absolute path in the client config because desktop clients may not inherit your shell `PATH`. The first launch downloads and builds the package, so allow for a slow cold start.

## 2. Create a catalog

Scaffold a catalog next to your project:

```bash
phasesweep mcp init-catalog --from ./experiment.yaml   # add --from per experiment; -o to name the file
```

This stages an annotated `catalog.yaml`, applies the server's startup checks, and publishes it only after every entry passes. Startup validation provisions the configured `state_dir` and its `runs` and `logs` subdirectories immediately. Publication never overwrites an existing or concurrently created destination. Each entry starts with side effects disabled and winner values redacted. Fill in its description, review the generated paths, and enable only the actions and parameter values the agent should receive. See [the catalog](mcp.md#the-catalog) for its fields and operational constraints. [examples/catalog.yaml](../examples/catalog.yaml) is a working catalog for [examples/mcp_experiment.yaml](../examples/mcp_experiment.yaml).

Confirm the catalog loads before touching any client config:

```bash
phasesweep mcp check --catalog /abs/path/to/catalog.yaml
```

Fix every reported failure before connecting a client. The command exits 0 only when the catalog and state directory pass the same checks used at server startup; it does not start the MCP server or a sweep.

## 3. Connect your client

One command writes the MCP server entry and, where the client supports project instructions, the step-5 agent instructions as a marker-fenced block. Project scope is used wherever the client supports it.

```bash
phasesweep mcp install                        # interactive: select agents, review the plan, apply
phasesweep mcp install --agent claude --yes   # unattended; repeat --agent for more
phasesweep mcp install --agent claude --dry-run  # validate and show planned edits without writing
phasesweep mcp install --agent codex --yes --allow-user-scope  # explicit user-scope acknowledgement
```

`install` first confirms the MCP SDK from step 1 is available, then validates the catalog without changing its state directory and shows one selector containing every supported client. Detected clients sort first and are preselected; press Enter to accept them, enter a comma-separated set of menu numbers, or use `all` or `none`. Undetected clients remain selectable for advance provisioning. The installer then prints a plan of every client file it will edit. After confirmation it revalidates and provisions the private runtime state before touching client config, then reports an outcome for each edit. For an existing catalog, a rejected plan and `--dry-run` therefore leave both catalog state and client files unchanged. An interactive, non-dry-run install offers to scaffold a missing catalog; unattended and dry-run installs print the exact `init-catalog` command and exit without writing. The server command is the executable beside the Python interpreter running `phasesweep` (the active conda/virtual environment), with `PATH` as a fallback; installation stops before edits if neither is launchable. Supported agents: `claude` (Claude Code), `claude-desktop`, `codex`, `cursor`, `vscode`, `gemini`, `opencode`.

Automatic edits are limited to regular files at the expected target. Project-scoped paths must remain inside the project after symlink resolution, and direct symlink targets are refused. Successful writes use an atomic replacement in the target directory and preserve an existing file's permissions. Strict JSON configs may update or remove a `phasesweep` entry only when its shape is recognizable as installer-generated; a different pre-existing entry is reported as a conflict and left untouched. Commented JSON/JSON5, malformed containers, invalid TOML, and unmanaged Codex tables are also left untouched with a manual snippet where applicable.

`--type mcp|instructions` installs one integration only; instructions-only installation does not require the MCP SDK. `--project DIR` targets another project root; the catalog defaults to `./catalog.yaml` in that project, so use `--catalog PATH` for any other name or location. Both `install` and `uninstall` accept `--dry-run` and report `would-create`, `would-update`, or `would-remove` without writing or deleting client files. Install previews still validate an existing catalog; uninstall needs no catalog. `phasesweep mcp uninstall` removes only recognizable generated-shape JSON entries and marker-owned TOML or instruction blocks, then deletes a file if that removal leaves it empty. The shared `AGENTS.md` block records its Codex, Cursor, and opencode owners and remains in place until its last owner is uninstalled. Unmanaged same-name entries remain untouched and are reported for operator attention.

Two placements are user-scoped rather than project-scoped, and the plan flags them: Claude Desktop (single user-level config) and Codex (`~/.codex/config.toml` - Codex reads project configs only in trusted projects). A user-scoped entry means that client sees this project's sweeps from every directory. Interactive installs require the normal plan confirmation. Unattended `--yes` installs additionally require `--allow-user-scope`; the generic confirmation bypass alone never authorizes a user-scoped write.

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

Any stdio client can also launch through `uvx` without a persistent install. For clients with separate `command` and `args` fields, replace the standard entry with:

```json
"command": "/abs/path/to/uvx",
"args": ["--from", "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git", "phasesweep-mcp", "--catalog", "/abs/path/to/catalog.yaml"]
```

For Codex, use the same values as TOML `command` and `args`; for OpenCode, combine them into its `command` array.

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

A working setup returns catalog ids, operator-authored descriptions, phase names, and the metric without dedicated config-path, command, storage, environment, or workdir fields. If the tool is missing or the call fails, see [Troubleshooting](#troubleshooting).

## 5. Instruct the agent

If step 3 ran with instructions enabled, the client already received the [agent instructions](../src/phasesweep/mcp/agent_prompt.md) in its project instructions file. The server also sends the same workflow in its MCP initialization instructions, and clients with MCP prompt support can load `phasesweep_run_and_monitor`. If the client honors none of those channels, copy the linked instructions into the agent's project instructions or the chat before asking it to run a sweep.

## Requests that work well

- `List the available phasesweep experiments and validate the tiny LM example.`
- `Launch the tiny-lm sweep, monitor it until completion, then summarize each phase winner.`
- `Check whether run <run_id> is still active and show phase-level trial counts.`
- `Read the current winners for <experiment_id> and suggest a next manual experiment using only MCP outputs.`

## Troubleshooting

- The client cannot find `phasesweep-mcp`: use the absolute path to the virtualenv or conda executable in the client config (`which phasesweep-mcp`).
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: set the corresponding `allow` flag to `true` on that catalog entry and restart the MCP client.
- `concurrency limit reached`: the refusal names up to five blocking run IDs. Await one directly and retry the refused launch after it becomes terminal, ask the user before cancelling it, or raise `max_concurrent_runs` on hosts that can safely run multiple sweeps.
- `terminal result snapshot ... unavailable`: retry once after a short delay. If finalization remains `failed` or `pending`, inspect the host and use `phasesweep mcp recover-run --state-dir <state_dir> --run-id <run_id>` followed by `--confirm` to rebuild the immutable snapshot; run-ID reads never substitute mutable experiment results.
- Path or storage rejected at startup: follow the MCP [path and working-directory rules](mcp.md#paths-and-the-working-directory); catalogs support local-node SQLite and Journal storage, not in-memory or external RDB storage.
- A cancelled or failed run stays `running` with `cleanup_confirmed: false`: inspect the host for leftover runner/trial process groups, then run `phasesweep mcp recover-run --state-dir <state_dir> --run-id <run_id>` to validate the saved identity and list the cleanup it would attempt. Repeat with `--confirm` only after reviewing that preflight; only the confirmed invocation attempts process cleanup and clears recovery state after cleanup is verified.
- Old MCP runs clutter status or logs: inspect `state_dir/runs` and `state_dir/logs`, then archive or prune terminal run handles between campaigns.
