# MCP agent setup

Requirements: Python 3.11+, the [MCP runtime requirements](mcp.md), a PhaseSweep experiment that passes validation, and a client with local stdio MCP support.

## Setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the Python environment that provides the server executable:

```bash
pip install "phasesweep[mcp]"
```

Contributor and editable-install setup is in [development](development.md). Install the tracking extra needed by the cataloged experiment in that same environment: `phasesweep[mcp,wandb]` for W&B or `phasesweep[mcp,mlflow]` for MLflow. Config validation and frozen scientific result reads do not authenticate. New work needs the selected SDK and access to its configured account or server. MLflow additionally requires an existing experiment; the client extra does not start a tracking server. See [native MLflow setup and run ownership](mlflow.md).

### 2. Create and review the catalog

If you need an experiment, `phasesweep init` writes `starter.yaml` without overwriting files; `phasesweep init --list` lists the other [templates](config.md#templates). Use the [configuration guide](config.md) and `phasesweep explain KEY` for authoring; connected agents can use `explain_config` when asked for that help.

Before cataloging a source-checkout trainer, follow the [trainer environment preflight](config.md#source-checkout-preflight) to verify its imported library and supported arguments under the configured environment. Then create the catalog:

```bash
phasesweep mcp init-catalog --from ./starter.yaml
```

The command writes `catalog.yaml`, validates it, and provisions a fresh [private state layout](mcp.md#fresh-mcp-state). Add another `--from` for each experiment; use `-o` for another catalog filename. The scaffold never replaces an existing catalog.

The scaffold records an absolute `state_dir` under `PHASESWEEP_HOME/mcp/<catalog-digest>`, or `${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>` when unset; later environment changes do not move it. A missing absolute `PHASESWEEP_HOME` is created mode 0700; relative paths, symlink components and existing directories with another mode are refused. To place state elsewhere, edit `state_dir`. It must be writable by each participating shell, including sandboxes. Give the server and `phasesweep mcp call` the same catalog, and `recover-run` the same `--state-dir`; see [state ownership and format rules](mcp.md#fresh-mcp-state).

> [!IMPORTANT]
> Review the generated file against [the catalog fields and permission model](mcp.md#the-catalog) before connecting a client. The catalog decides which experiments an agent can see, which sampled values it can read, and which side effects it may request.

Validate the reviewed catalog with its absolute path:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

This exercises server startup validation and provisions/probes its private state layout without launching a sweep.

### 3. Configure your client

You manage the client files; PhaseSweep does not create, edit, inspect, or remove them. Follow your client's official documentation for its configuration file, scope, and restart behavior. Configure a stdio server with the absolute `phasesweep-mcp` executable and the absolute reviewed catalog path:

```text
/absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

For Codex CLI, the current [official MCP instructions](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) use:

```bash
codex mcp add phasesweep -- /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
codex mcp list
```

### 4. Verify the connection

Restart or reconnect the client as its documentation requires. Confirm the connection with the read-only `list_experiments` operation; for example, ask the agent to list available experiments and permitted actions without launching anything. The response contains catalog-approved IDs, descriptions, phase names, metrics, and permitted actions.

To check the same operation without a client, or to script or debug a single tool call from a shell, run it directly; it prints the structured result as JSON, or the tool error on stderr:

```bash
phasesweep mcp call list_experiments --catalog /absolute/path/to/catalog.yaml
```

See [shell calls](mcp.md#calling-a-tool-from-a-shell) for arguments, exit codes and monitoring. For an explicitly authorized run, follow the [tool workflow](mcp.md#tool-workflow) and [response shapes](mcp.md#reading-status-responses).

## Maintenance and troubleshooting

Use `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` after a catalog edit or when diagnosing server startup. If you replace the environment that supplies `phasesweep-mcp`, update the client-owned entry to the new absolute executable path using that client's documentation.

- `MCP support is not installed`: install the MCP extra in the environment that supplies `phasesweep-mcp`.
- The client cannot start the server: verify its client-owned stdio entry has absolute executable and catalog paths, then run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml`.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: change the catalog only if that is the authority you intend, then reconnect the client.
- `concurrency limit reached`: await a returned blocking run ID; never cancel or replace it automatically.
- `recovery_required: true`, unresolved launch, or uncertain cleanup: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery).
- Unavailable terminal snapshot: inspect run/server diagnostics; [historical results](mcp.md#reading-status-responses) cannot be reconstructed from the current ledger.
- `runner_observable: false` or a PID-namespace refusal: follow [PID namespace recovery](mcp.md#pid-namespaces).
- Catalog path, storage, or working-directory rejection: follow [the catalog requirements](mcp.md#the-catalog).
- Trainer environment cohort mismatch: the detached runner uses the server's environment; reconcile it with shell launches using the [trainer contract](config.md#trainer-contract).
