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

Client configs want the executable path absolute, because clients launch servers outside your shell environment. `phasesweep mcp install` (step 3) resolves it for you, so you only need the path for manual setup - in that case, find it now (`phasesweep-mcp` and `phasesweep mcp serve` start the same server; client configs use the dedicated executable). The generated entry intentionally remains bound to this Python environment. If you move, delete, or recreate it, install PhaseSweep in the replacement environment and rerun the same step-3 install command to update the managed entry.

```bash
which phasesweep-mcp
```

For manual setup without a persistent install, a client can launch through `uvx` (`uv tool run`). Use `command -v uvx` to get its absolute path; the first launch has a slower download/build cold start.

## 2. Create a catalog

Scaffold a catalog next to your project:

```bash
phasesweep mcp init-catalog --from ./experiment.yaml   # add --from per experiment; -o to name the file
```

This stages an annotated `catalog.yaml`, validates it through the server startup path, provisions its private state directories, and publishes it only after validation succeeds without overwriting an existing file. Each entry starts with side effects disabled and winner values redacted. Fill in its description, review the generated paths, enable only the actions and parameter values the agent should receive, then run the `phasesweep mcp check` command printed by the scaffold. See [the catalog](mcp.md#the-catalog) for its fields and operational constraints. [examples/catalog.yaml](../examples/catalog.yaml) is a working catalog for [examples/mcp_experiment.yaml](../examples/mcp_experiment.yaml).

Confirm the catalog loads before touching any client config:

```bash
phasesweep mcp check --catalog /abs/path/to/catalog.yaml
```

Fix every reported failure before connecting a client. The command exits 0 only when the catalog entries satisfy the same schema, config, storage, and path rules used at server startup and the configured state, runs, and logs directories have been provisioned and write-probed through the startup path. It does not start the MCP server or launch a sweep.

## 3. Connect your client

One command writes the MCP server entry and, where the client supports project instructions, the step-5 agent instructions as a marker-fenced block. Project scope is used wherever the client supports it.

```bash
phasesweep mcp install                        # interactive: select agents, review the plan, apply
phasesweep mcp install --agent claude --yes   # unattended; repeat --agent for more
phasesweep mcp install --agent claude --dry-run  # validate and show planned client-file edits
phasesweep mcp install --agent codex --yes --allow-user-scope  # explicit user-scope acknowledgement
```

Without `--agent`, detected clients sort first and are preselected in one menu; undetected clients remain selectable. The installer validates the catalog, provisions its state layout, prints every target path, confirms once, and reports each edit. `--dry-run` shows the same client-file plan and outcomes without editing client files; its catalog preflight may still create or secure the state directories. An interactive install can scaffold a missing catalog; unattended installs print the exact `init-catalog` command instead. Supported agents are Claude Code, Claude Desktop, Codex, Cursor, VS Code, Gemini CLI, and opencode.

Automatic edits are limited to regular files at the expected target. Project-scoped paths must remain inside the selected project after symlink resolution, and direct file symlinks are refused. Malformed configs and same-name entries that do not match the generated shape are left untouched with a manual snippet. Marker-fenced text edits replace only the managed block. Strict JSON edits change only the managed data, but re-serialize the document; JSON data, key order, detected indentation, final-newline state, and permissions are preserved, while compact or irregular whitespace may be reformatted. `uninstall` removes the same managed content and leaves shared instruction blocks until their last installed agent owner is removed.

Use `--type mcp|instructions` to install one integration and `--project DIR` to target another project root. The catalog defaults to that project's `catalog.yaml`; pass `--catalog PATH` for another location. `uninstall` accepts the same agent/type/project selectors and needs no catalog.

Claude Desktop and Codex MCP entries are user-scoped, so those clients see the server from every project. The plan flags this. Interactive installs require confirmation; unattended `--yes` installs additionally require `--allow-user-scope`.

Restart the client after any config change.

<details>
<summary>Manual setup (any client)</summary>

Every stdio client launches the same server. Use absolute paths for both values:

```json
"command": "/abs/path/to/venv/bin/phasesweep-mcp",
"args": ["--catalog", "/abs/path/to/catalog.yaml"]
```

To launch through `uvx` instead:

```json
"command": "/abs/path/to/uvx",
"args": ["--from", "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git", "phasesweep-mcp", "--catalog", "/abs/path/to/catalog.yaml"]
```

Client schemas and config paths differ. Run `phasesweep mcp install --dry-run` for the exact target, or use the manual snippet printed when an edit is skipped.

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

- The client cannot find `phasesweep-mcp`: a generated entry points to the Python environment that ran the installer. If that environment moved, was deleted, or was recreated, install the MCP extra in its replacement, rerun the same `phasesweep mcp install` command (`--dry-run` previews the repair), and restart the client. For a manual config, update `command` to the new absolute path from `which phasesweep-mcp`.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: set the corresponding `allow` flag to `true` on that catalog entry and restart the MCP client.
- `concurrency limit reached`: the refusal names up to five blocking run IDs. Await one directly and retry the refused launch after it becomes terminal, ask the user before cancelling it, or raise `max_concurrent_runs` on hosts that can safely run multiple sweeps.
- `terminal result snapshot ... unavailable`: retry once after a short delay. If finalization remains `failed` or `pending`, inspect the host and use `phasesweep mcp recover-run --state-dir <state_dir> --run-id <run_id>` followed by `--confirm` to rebuild the immutable snapshot; run-ID reads never substitute mutable experiment results.
- Path or storage rejected at startup: follow the MCP [path and working-directory rules](mcp.md#paths-and-the-working-directory); catalogs support local-node SQLite and Journal storage, not in-memory or external RDB storage.
- A cancelled or failed run stays `running` with `cleanup_confirmed: false`: inspect the host for leftover runner/trial process groups, then run `phasesweep mcp recover-run --state-dir <state_dir> --run-id <run_id>` to validate the saved identity and list the cleanup it would attempt. Repeat with `--confirm` only after reviewing that preflight; only the confirmed invocation attempts process cleanup and clears recovery state after cleanup is verified.
- Old MCP runs clutter status or logs: inspect `state_dir/runs` and `state_dir/logs`, then archive or prune terminal run handles between campaigns.
