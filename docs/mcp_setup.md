# MCP agent setup

This path connects a local AI client to operator-approved PhaseSweep experiments without requiring a repository checkout. Catalog creation and client installation stay separate so a human reviews the authority boundary before an agent receives it.

Requirements: Python 3.11+, the [MCP runtime platform requirements](runtime.md#platform-support), a PhaseSweep experiment that passes validation, and a client with local stdio MCP support.

## Five-minute setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the Python environment whose executable the client should use:

```bash
pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

Reinstalling the same Git ref later may require adding `--force-reinstall` to the command above or selecting a changed ref. Contributor and editable-install setup is in [development](development.md).

### 2. Create and review the catalog

If you do not have an experiment yet, `phasesweep init` creates an installed-package starter named `experiment.yaml` without overwriting files.

```bash
phasesweep mcp init-catalog --from ./experiment.yaml
```

The command writes `catalog.yaml` with side effects disabled and winner values redacted. Its catalog validation also provisions the [private state layout](mcp.md#the-catalog) outside the project, under `${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>/`. The catalog pins that absolute path, and the private state's `origin` file identifies the catalog. Before continuing, review:

- every experiment description and config path;
- `visible_params`, which controls sampled winner values visible to the agent;
- `allow.launch`, `allow.cancel`, and `allow.from_phase`;
- the catalog state directory and each experiment working directory.

Add another `--from` for each experiment. Use `-o` to choose another catalog filename. For per-user scratch placement, provision an absolute owner-only `0700` directory and set `PHASESWEEP_HOME` before scaffolding: new MCP state goes under its `mcp/` directory and default locks under `locks/`. Keep that environment consistent for the CLI and MCP server; an existing catalog keeps its saved `state_dir`. The scaffold is staged and validated before publication and never replaces an existing path. See [the catalog reference](mcp.md#the-catalog) for storage and path rules and the [security model](mcp.md#security-model) for the resulting authority boundary.

### 3. Connect a client

```bash
phasesweep mcp install
```

The installer validates the catalog, pins the installed `phasesweep-mcp` executable, detects clients, and shows every planned target and permission before confirmation. It then applies the edits, verifies the launcher and catalog path, and prints the restart instruction. Preservation and ownership rules are described under [what the installer changes](#what-the-installer-changes).

The supported clients are Claude Code, Claude Desktop, Codex, Cursor, VS Code, Gemini CLI, and opencode. A missing catalog is never created implicitly; return to the review step instead.

### 4. Restart and verify

Restart the selected client, then ask exactly:

```text
List the available PhaseSweep experiments and their permitted actions.
Do not launch anything.
```

A working connection returns only catalog-approved experiment IDs, descriptions, phase shape, metrics, and permitted actions. It does not launch a run.

## What the installer changes

By default, `--type all` installs two independent integrations:

- an MCP server entry whose command is the absolute `phasesweep-mcp` executable from the environment running the installer and whose arguments are `--catalog` plus the reviewed catalog path;
- a marker-fenced copy of the packaged agent instructions where the client supports project instructions.

Project scope is used where the client reliably supports it. Claude Desktop and Codex MCP entries are user-scoped, and the plan labels them before confirmation. Shared instruction files contain one package-managed block with multiple client owners; if a later install updates that shared prompt, the plan names the existing owners affected.

`phasesweep mcp uninstall` removes only recognizable installer-managed entries and ownership blocks. Unmanaged same-name entries are reported and left untouched.

### File preservation

Automatic edits are limited to regular UTF-8 physical targets. User-scoped dotfile symlinks are followed. Project-scoped symlinks are followed only when the resolved target remains inside the selected project; plan, apply, post-apply verification, and `check-install` all refuse an escaping path rather than auditing the external target. Each operation pins that physical target, serializes against other PhaseSweep installers, and refuses replacement if the file changes during the transaction. Malformed configs and unmanaged same-name entries are left untouched with manual guidance.

JSON ownership is inferred from the exact generated shape; no receipt records which entry the installer created. A hand-authored entry with that shape is therefore managed and may be replaced or removed. Any differing key or argument makes it unmanaged. Codex TOML additionally requires the installer's marker lines. Shared project instructions use one marker-fenced block plus an owner set; removing one client retains the other owners' block, and removing the final owner removes it.

- Marker-fenced instructions and managed Codex TOML preserve unrelated bytes according to their marker contract.
- Strict JSON is reserialized as a complete document. Key order, number spelling (`1e2`, `1.50`), newline style, final-newline state, and permissions are preserved. An already indented document keeps its detected indentation; a compact one-line document has no indentation to detect and is expanded to the installer's two-space multiline form, so whitespace anywhere in it may change.
- Duplicate keys, comments, JSON5, non-finite values, and overflowing numbers are refused.
- Empty files and empty JSON containers remain after uninstall because whole-file creation ownership is not persisted.

Uninstall removes the managed member; it does not promise a byte-identical JSON round trip.

## Unattended installation

Use explicit targets for scripts and automation:

```bash
phasesweep mcp install --agent claude --dry-run
phasesweep mcp install --agent claude --yes
phasesweep mcp install --agent codex --yes --allow-user-scope
phasesweep mcp install --agent claude --agent cursor --type mcp --yes
```

`--agent` may be repeated. `--type mcp|instructions|all` selects the integration, `--project DIR` anchors project-scoped files, and `--catalog PATH` overrides `<project>/catalog.yaml`. Unattended user-scoped writes require `--allow-user-scope`; `--yes` alone is not sufficient. A dry run still performs the catalog preflight from step 2, but it does not edit client files.

An instructions-only install needs no catalog or MCP SDK:

```bash
phasesweep mcp install --agent claude --type instructions --yes
```

## Verification and maintenance

The installer verifies new MCP entries immediately. To check existing entries later without editing files or contacting the network:

```bash
phasesweep mcp check-install
phasesweep mcp check-install --agent claude
```

The report distinguishes a resolvable managed launcher (`ok`), missing or non-executable launchers, scripts whose shebang interpreter is gone, unreadable or missing catalogs, unmanaged entries, absent entries, and unreadable client configuration. It inspects files but deliberately does not execute a configured launcher, parse the catalog, import the MCP SDK from another environment, or test server startup. From the environment named by the launcher, `python -c 'import mcp, phasesweep.mcp.server'` checks the runtime imports and `phasesweep mcp check --catalog PATH` checks catalog startup. A recognized legacy launcher entry still reports `ok` but carries an explicit caveat that it is not the pinned absolute executable and that rerunning the installer will pin it. Executable failures are reported before catalog failures because the server cannot read a catalog if it cannot start.

For CI, explicit catalog review, or troubleshooting, run:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

This uses the server's startup validation and, only after every catalog entry passes, provisions and probes the private state layout. It launches no sweep.

After replacing or recreating the conda environment, rerun `phasesweep mcp install` from the intended environment and restart each selected client so its absolute executable path and instructions are refreshed.

## Troubleshooting

- `MCP support is not installed`: activate the intended conda environment, run the install command from step 1, then retry.
- The client cannot start `phasesweep-mcp`: run `phasesweep mcp check-install`. If its static launcher/catalog checks pass, activate the environment named by the absolute command, run `python -c 'import mcp, phasesweep.mcp.server'`, then run `phasesweep mcp check --catalog PATH`; repair the environment or rerun the installer if either check fails, then restart the client.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: change the corresponding catalog flag only if that is the authority you intend, then restart the MCP client.
- `concurrency limit reached`: await one of the returned blocking run IDs. Do not cancel it or launch a replacement automatically.
- `recovery_required: true`, unresolved launch, uncertain cleanup, or unavailable terminal snapshot: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery).
- Catalog path, storage, or working-directory rejection: follow [paths and the working directory](mcp.md#paths-and-the-working-directory).
- A client config is skipped: use the manual snippet printed by the installer and review [file preservation](#file-preservation). The installer does not overwrite malformed or unmanaged data.

## Manual entry

For a manual stdio entry, use the absolute values printed by `which phasesweep-mcp` and your reviewed catalog:

```json
{
  "command": "/absolute/path/to/conda/env/bin/phasesweep-mcp",
  "args": ["--catalog", "/absolute/path/to/catalog.yaml"]
}
```

Client schemas and config paths differ. Run `phasesweep mcp install --dry-run` for the exact target before editing manually.
