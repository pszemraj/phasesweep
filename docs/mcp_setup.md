# MCP agent setup

PhaseSweep owns the MCP server and its operator-reviewed catalog. Your MCP
client owns its configuration and any client-specific instructions; PhaseSweep
does not create, edit, inspect, or remove client files.

Requirements: Python 3.11+, the [MCP runtime requirements](mcp.md), a
PhaseSweep experiment that passes validation, and a client with local stdio
MCP support.

## Setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the Python environment
that provides the server executable:

```bash
python -m pip install "phasesweep[mcp] @ git+https://github.com/pszemraj/phasesweep.git"
```

Contributor and editable-install setup is in [development](development.md).
If cataloged experiments read W&B evidence, install the `mcp,wandb` extras
together in that environment. Config validation and frozen result reads do not
authenticate; new W&B work requires the SDK and the trainer's composed account
access.

### 2. Create and review the catalog

If you do not have an experiment yet, `phasesweep init` creates an
installed-package starter named `experiment.yaml` without overwriting files.

```bash
phasesweep mcp init-catalog --from ./experiment.yaml
```

The command writes `catalog.yaml`, validates it, and provisions a fresh
[private state layout](mcp.md#fresh-mcp-state). Review the generated file
against [the catalog fields and permission model](mcp.md#the-catalog) before
connecting a client. Add another `--from` for each experiment; use `-o` for
another catalog filename. The scaffold never replaces an existing catalog.

Validate the reviewed catalog with its absolute path:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

This exercises server startup validation and provisions/probes its private
state layout without launching a sweep.

### 3. Configure your client

Follow your MCP client's official documentation for its configuration file,
scope, and restart behavior. Configure a stdio server with the absolute
`phasesweep-mcp` executable and the absolute reviewed catalog path:

```text
/absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

For Codex CLI, the current [official MCP instructions](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) use:

```bash
codex mcp add phasesweep -- /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
codex mcp list
```

### 4. Verify the connection

Restart or reconnect the client as its documentation requires. Confirm the
connection with the existing read-only `list_experiments` catalog operation;
for example, ask the connected agent to list the available PhaseSweep
experiments and permitted actions without launching anything. The response
contains only catalog-approved IDs, descriptions, phase shape, metrics, and
permitted actions.

For an explicitly authorized run, follow the [tool workflow](mcp.md#tool-workflow)
and [response shapes](mcp.md#reading-status-responses).

## Maintenance and troubleshooting

Use `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` after a
catalog edit or when diagnosing server startup. If you replace the environment
that supplies `phasesweep-mcp`, update the client-owned entry to the new
absolute executable path using that client's documentation.

- `MCP support is not installed`: install the MCP extra in the environment that supplies `phasesweep-mcp`.
- The client cannot start the server: verify its client-owned stdio entry has absolute executable and catalog paths, then run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml`.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: change the catalog only if that is the authority you intend, then reconnect the client.
- `concurrency limit reached`: await one of the returned blocking run IDs. Do not cancel it or launch a replacement automatically.
- `recovery_required: true`, unresolved launch, uncertain cleanup, or unavailable terminal snapshot: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery).
- Catalog path, storage, or working-directory rejection: follow [the catalog requirements](mcp.md#the-catalog).
