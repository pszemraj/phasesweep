# MCP agent setup

This path connects a local AI client to operator-approved PhaseSweep experiments without requiring a repository checkout. Catalog creation and client installation stay separate so a human reviews the authority boundary before an agent receives it.

Requirements: Python 3.11+, the [MCP runtime platform requirements](runtime.md#platform-support), a PhaseSweep experiment that passes validation, and a client with local stdio MCP support.

## Five-minute setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the conda environment whose executable the client should use:

```bash
pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

Reinstalling the same Git ref later may require adding `--force-reinstall` to the command above or selecting a changed ref. Contributor and editable-install setup is in [development](development.md).

### 2. Create and review the catalog

If you do not have an experiment yet, `phasesweep init` creates an installed-package starter named `experiment.yaml` without overwriting files.

```bash
phasesweep mcp init-catalog --from ./experiment.yaml
```

The command writes `catalog.yaml` with side effects disabled and winner values redacted. Before continuing, review:

- every experiment description and config path;
- `visible_params`, which controls sampled winner values visible to the agent;
- `allow.launch`, `allow.cancel`, and `allow.from_phase`;
- the catalog state directory and each experiment working directory.

Add another `--from` for each experiment. Use `-o` to choose another catalog filename. The scaffold is staged and validated before publication and never replaces an existing path. See [the catalog reference](mcp.md#the-catalog) for storage and path rules and the [security model](mcp.md#security-model) for the resulting authority boundary.

### 3. Connect a client

```bash
phasesweep mcp install
```

The installer validates the catalog, resolves the absolute installed `phasesweep-mcp` executable before asking any questions, detects and preselects clients, and prints a plan containing the catalog, experiment permissions, client paths, integration types, and user-scoped edits. After confirmation it applies safe edits, verifies the written launcher and catalog path, and prints the restart instruction.

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
- a marker-fenced copy of the packaged seven-rule agent instructions where the client supports project instructions.

Project scope is used where the client reliably supports it. Claude Desktop and Codex MCP entries are user-scoped, and the plan labels them before confirmation. Shared instruction files contain one package-managed block with multiple client owners; if a later install updates that shared prompt, the plan names the existing owners affected.

`phasesweep mcp uninstall` removes only recognizable installer-managed entries and ownership blocks. Unmanaged same-name entries are reported and left untouched. The [installer preservation contract](mcp.md#installer-file-preservation-contract) documents symlinks, locking, JSON reserialization, markers, ownership, and manual-merge behavior.

## Unattended installation

Use explicit targets for scripts and automation:

```bash
phasesweep mcp install --agent claude --dry-run
phasesweep mcp install --agent claude --yes
phasesweep mcp install --agent codex --yes --allow-user-scope
phasesweep mcp install --agent claude --agent cursor --type mcp --yes
```

`--agent` may be repeated. `--type mcp|instructions|all` selects the integration, `--project DIR` anchors project-scoped files, and `--catalog PATH` overrides `<project>/catalog.yaml`. Unattended user-scoped writes require `--allow-user-scope`; `--yes` alone is not sufficient. A dry run performs catalog preflight and may provision its private state layout, but it does not edit client files.

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

The report distinguishes a healthy entry (`ok`), missing or non-executable launchers, unreadable or missing catalogs, unmanaged entries, absent entries, and unreadable client configuration. Executable failures are reported before catalog failures because the server cannot read a catalog if it cannot start.

For CI, explicit catalog review, or troubleshooting, run:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

This uses the server's startup validation, provisions and probes the private state layout, and launches no sweep.

After replacing or recreating the conda environment, rerun `phasesweep mcp install` from the intended environment and restart each selected client so its absolute executable path and instructions are refreshed.

## Breaking development installs

The package is not published and this MCP surface intentionally carries no deprecated tool aliases or launcher migration layer. If a client was configured by an earlier development build, rerun `phasesweep mcp install` and restart the client; cached old tool names disappear on restart. If the current installer reports the old entry as unmanaged, remove that one `phasesweep` client entry manually, rerun the installer, and restart. This is the complete migration policy.

## Troubleshooting

- `MCP support is not installed`: activate the intended conda environment, run the install command from step 1, then retry.
- The client cannot start `phasesweep-mcp`: run `phasesweep mcp check-install`, activate or repair the environment named by the absolute command, rerun the installer, and restart the client.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: change the corresponding catalog flag only if that is the authority you intend, then restart the MCP client.
- `concurrency limit reached`: await one of the returned blocking run IDs. Do not cancel it or launch a replacement automatically.
- `recovery_required: true`, unresolved launch, uncertain cleanup, or unavailable terminal snapshot: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery).
- Catalog path, storage, or working-directory rejection: follow [paths and the working directory](mcp.md#paths-and-the-working-directory).
- A client config is skipped: use the manual snippet printed by the installer and review the [file preservation contract](mcp.md#installer-file-preservation-contract). The installer does not overwrite malformed or unmanaged data.

## Advanced and manual details

- [MCP operator reference](mcp.md): catalog schema, tool payloads, run lifecycle, recovery, authorization, auditing, security, and installer semantics.
- [Runtime behavior](runtime.md): platform support, process supervision, storage, locks, GPU isolation, and output layout.
- [Packaged agent instructions](../src/phasesweep/mcp/agent_prompt.md): the seven rules installed into supported project instruction files.

For a manual stdio entry, use the absolute values printed by `which phasesweep-mcp` and your reviewed catalog:

```json
{
  "command": "/absolute/path/to/conda/env/bin/phasesweep-mcp",
  "args": ["--catalog", "/absolute/path/to/catalog.yaml"]
}
```

Client schemas and config paths differ. Run `phasesweep mcp install --dry-run` for the exact target before editing manually.
