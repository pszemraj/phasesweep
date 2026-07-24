# MCP agent setup

Install the extra, create a catalog, connect a client, verify the connection, and give the agent its operating instructions. Tool behavior, catalog rules, and the security model are covered in [MCP server](mcp.md).

## Prerequisites

- The MCP [platform requirements](runtime.md#platform-support).
- Python 3.11 or newer.
- At least one phasesweep experiment config that passes `phasesweep validate` and uses the restart-stable MCP paths described in step 2.
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

Client configs want the executable path absolute, because clients launch servers outside your shell environment. `phasesweep mcp install` (step 3) resolves it for you, so you only need the path for manual setup - in that case, find it now (`phasesweep-mcp` and `phasesweep mcp serve` start the same server; client configs use the dedicated executable). The generated entry remains bound to this Python environment; see [Troubleshooting](#troubleshooting) if it moves.

```bash
which phasesweep-mcp
```

## 2. Create a catalog

Scaffold a catalog next to your project:

```bash
phasesweep mcp init-catalog --from ./experiment.yaml   # add --from per experiment; -o to name the file
```

An ordinary CLI config with relative `workdir` or storage can pass `phasesweep validate` but is not restart-stable enough for MCP. Use an absolute `workdir` and an absolute local SQLite or Journal storage path; [examples/mcp_experiment.yaml](../examples/mcp_experiment.yaml) shows the MCP variant of the toy config.

This stages an annotated `catalog.yaml`, validates it through the server startup path, provisions its private state directories, and publishes it only after validation succeeds without overwriting an existing file. Each generated entry pins the detached trainer working directory to the catalog directory; change `cwd` if its relative `trial_command` normally starts elsewhere. Entries begin with side effects disabled and winner values redacted. Fill in each description, review the generated paths, enable only the actions and parameter values the agent should receive, then run the `phasesweep mcp check` command printed in the scaffold. See [the catalog](mcp.md#the-catalog) for its fields and operational constraints. [examples/catalog.yaml](../examples/catalog.yaml) is a working catalog for the toy MCP config.

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

Without `--agent`, interactive installs preselect detected clients in one menu while leaving undetected clients selectable; unattended `--yes` installs select every detected client. The installer validates the catalog, provisions its state layout, prints every target path, confirms once, and reports each edit. `--dry-run` shows the same client-file plan and outcomes without editing client files; its catalog preflight may still create or secure the state directories. A missing catalog is never created implicitly: the command prints the exact `init-catalog` step so you can review permissions before connecting a client. Supported agents are Claude Code, Claude Desktop, Codex, Cursor, VS Code, Gemini CLI, and opencode.

Use `--type mcp|instructions` to install one integration and `--project DIR` to target another project root. The catalog defaults to that project's `catalog.yaml`; pass `--catalog PATH` for another location. `uninstall` accepts the same agent/type/project selectors and needs no catalog.

An instructions-only install (`--type instructions`) needs neither a catalog nor the optional MCP dependency and remains available outside the MCP runtime's [platform requirements](runtime.md#platform-support).

Claude Desktop and Codex MCP entries are user-scoped, so those clients see the server from every project. The plan flags this. Interactive installs require confirmation; unattended `--yes` installs additionally require `--allow-user-scope`.

By default the generated entry pins the absolute `phasesweep-mcp` executable from the Python environment that ran `install`; if that environment is later moved, recreated, or deleted, the entry breaks (see [Troubleshooting](#troubleshooting)). Pass `--launcher uvx` to instead pin a `uvx --from phasesweep[mcp]==<installed version> phasesweep-mcp` invocation, resolved fresh by [uvx](https://docs.astral.sh/uv/) on every launch instead of bound to this environment's path. Prefer the default for a normal local dev environment you intend to keep; prefer `--launcher uvx` when the environment running `install` is disposable or expected to be rebuilt (a container, a throwaway venv, a CI-provisioned tool). The uvx mode requires `uvx` on the path that will actually launch the client's subprocess, and a real installed phasesweep version to pin (not an unbuilt source checkout) - `install` refuses to write the entry, unchanged, when either is missing. Both modes write the same client files and `uninstall` reverses either one; switching modes is just re-running `install` for the same agent.

Restart the client after any config change.

<details>
<summary>What automatic edits preserve</summary>

Automatic edits are limited to regular UTF-8 physical targets. User-scoped dotfile symlinks are followed, and project-scoped symlinks are followed only when their resolved target remains inside the selected project. Each edit pins that physical target, serializes with other PhaseSweep installers, and is refused if the file changes before replacement. Malformed configs and unmanaged same-name entries are left untouched with manual guidance.

Marker-fenced edits preserve all bytes outside the managed block. Strict JSON edits re-serialize the document while retaining key order, detected indentation and newline style, final-newline state, and permissions; compact whitespace and numeric spellings may be normalized. Duplicate keys, non-finite or overflowing numbers, comments, and JSON5 are refused. `uninstall` leaves empty files and containers in place because whole-file creation ownership is not persisted. Shared instruction blocks remain until their last installed agent owner is removed.

</details>

<details>
<summary>Manual setup (any client)</summary>

Every stdio client launches the same server. Use absolute paths for both values:

```json
"command": "/abs/path/to/python-env/bin/phasesweep-mcp",
"args": ["--catalog", "/abs/path/to/catalog.yaml"]
```

Client schemas and config paths differ. Run `phasesweep mcp install --dry-run` for the exact target, or use the manual snippet printed when an edit is skipped.

</details>

## 4. Verify

Restart the client, then ask the agent:

```text
List the available phasesweep experiments.
```

A working setup returns catalog ids, operator-authored descriptions, phase names, and the metric without dedicated config-path, command, storage, environment, or workdir fields. If the tool is missing or the call fails, see [Troubleshooting](#troubleshooting).

To check without asking the agent, or to audit every configured client at once, run the read-only:

```bash
phasesweep mcp check-install                    # every supported agent
phasesweep mcp check-install --agent claude     # one agent; repeat --agent for more
```

For each configured entry it reports the launcher command and whether it resolves: the absolute path exists and is executable (default mode), or `uvx` is on `PATH` (`--launcher uvx` mode). It never edits a client file. Agents with no phasesweep entry are reported `not configured`; entries this installer did not write are reported `unmanaged` and left unexamined. Exit code 0 means every configured launcher resolves; exit code 1 means at least one needs the repair guidance printed above it.

## 5. Instruct the agent

If step 3 ran with instructions enabled, the client already received the [agent instructions](../src/phasesweep/mcp/agent_prompt.md) in its project instructions file. The server also sends the same workflow in its MCP initialization instructions, and clients with MCP prompt support can load `phasesweep_run_and_monitor`. If the client honors none of those channels, copy the linked instructions into the agent's project instructions or the chat before asking it to run a sweep.

## Requests that work well

- `List the available phasesweep experiments and validate the tiny LM example.`
- `Launch the tiny-lm sweep, monitor it until completion, then summarize each phase winner.`
- `Check whether run <run_id> is still active and show phase-level trial counts.`
- `Read the current winners for <experiment_id>, separate observations from hypotheses, and tell me what additional evidence is needed before choosing the next manual experiment.`

## Troubleshooting

- The client cannot find `phasesweep-mcp`: run `phasesweep mcp check-install` to confirm which configured executable is missing without guessing. A default-mode entry points to the Python environment that ran the installer; if that environment moved, was deleted, or was recreated, install the MCP extra in its replacement, rerun the same `phasesweep mcp install` command (`--dry-run` previews the repair), and restart the client, or switch that agent to `phasesweep mcp install --launcher uvx` so future environment churn doesn't break it again. For a manual config, update `command` to the new absolute path from `which phasesweep-mcp`. A uvx-mode entry instead needs `uvx` on the launching `PATH`; install [uv](https://docs.astral.sh/uv/) and restart the client.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: set the corresponding `allow` flag to `true` on that catalog entry and restart the MCP client.
- `concurrency limit reached`: the refusal names up to five blocking run IDs. Await one directly and retry the refused launch after it becomes terminal, ask the user before cancelling it, or raise `max_concurrent_runs` on hosts that can safely run multiple sweeps.
- `terminal result snapshot ... unavailable`: follow [run state and recovery](mcp.md#run-state-and-recovery); historical results are never rebuilt from mutable experiment state.
- `launch outcome is unresolved`: wait briefly and retry `recover-run`. If it persists after a server crash, automated recovery cannot distinguish a pre-spawn crash from a not-yet-registered runner; inspect the host and keep the run reserved until a matching runner is ruled out.
- Path or storage rejected at startup: follow the MCP [path and working-directory rules](mcp.md#paths-and-the-working-directory); catalogs support local-node SQLite and Journal storage, not in-memory or external RDB storage.
- A cancelled or failed run stays `running` with `cleanup_confirmed: false`: follow the operator procedure under [run state and recovery](mcp.md#run-state-and-recovery).
- Old MCP runs clutter status or logs: inspect `state_dir/runs` and `state_dir/logs`, then archive or prune terminal run handles between campaigns.
