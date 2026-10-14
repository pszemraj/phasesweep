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

If you do not have an experiment yet, `phasesweep init` creates an installed-package starter named `experiment.yaml` without overwriting files. The [configuration reference](../src/phasesweep/config_reference.yaml) describes the accepted experiment fields, including backend selection and environment composition.

Before cataloging a source-checkout trainer, follow the [trainer environment preflight](config.md#source-checkout-preflight) to verify its imported library and supported arguments under the configured environment. Then create the catalog:

```bash
phasesweep mcp init-catalog --from ./experiment.yaml
```

The command writes `catalog.yaml`, validates it, and provisions a fresh [private state layout](mcp.md#fresh-mcp-state). Add another `--from` for each experiment; use `-o` for another catalog filename. The scaffold never replaces an existing catalog.

Run state lives in the catalog's `state_dir`, which the scaffold records as an absolute path under `PHASESWEEP_HOME/mcp/<catalog-digest>`, or under `${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>` when `PHASESWEEP_HOME` is unset; later changes to those variables do not move it. A missing absolute `PHASESWEEP_HOME` is created owner-only (mode 0700), while a relative path, a symlink in the path, or an existing directory with another mode is refused. The directory must be writable from the shell an agent uses, including a sandboxed one, and every process that shares runs must use the same `state_dir`, so give the server, `phasesweep mcp call` and any sandboxed shell the same catalog, and `recover-run` the same `--state-dir`. To place the state elsewhere, set `state_dir` in the catalog.

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

Restart or reconnect the client as its documentation requires. Confirm the connection with the existing read-only `list_experiments` catalog operation; for example, ask the connected agent to list the available PhaseSweep experiments and permitted actions without launching anything. The response contains only catalog-approved IDs, descriptions, phase shape, metrics, and permitted actions.

To check the same operation without a client, or to script or debug a single tool call from a shell, run it directly; it prints the structured result as JSON, or the tool error on stderr:

```bash
phasesweep mcp call list_experiments --catalog /absolute/path/to/catalog.yaml
```

See [calling a tool from a shell](mcp.md#calling-a-tool-from-a-shell) for arguments, exit codes, and a loop that waits on a run past the 600-second `await_run` limit.

For an explicitly authorized run, follow the [tool workflow](mcp.md#tool-workflow) and [response shapes](mcp.md#reading-status-responses).

## Maintenance and troubleshooting

Use `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` after a catalog edit or when diagnosing server startup. If you replace the environment that supplies `phasesweep-mcp`, update the client-owned entry to the new absolute executable path using that client's documentation.

- `MCP support is not installed`: install the MCP extra in the environment that supplies `phasesweep-mcp`.
- The client cannot start the server: verify its client-owned stdio entry has absolute executable and catalog paths, then run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml`.
- `action 'launch' is not permitted` or `action 'cancel' is not permitted`: change the catalog only if that is the authority you intend, then reconnect the client.
- `concurrency limit reached`: await one of the returned blocking run IDs. Do not cancel it or launch a replacement automatically.
- `recovery_required: true`, unresolved launch, uncertain cleanup, or unavailable terminal snapshot: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery).
- `runner_observable: false`, or `cancel_run` or `recover-run` naming a recorded PID namespace: the run's runner was launched from another PID namespace than this server's, for example a different sandbox. While that sandbox runs, cancel or recover the run from inside it. Once it has ended, run `phasesweep mcp recover-run` from a host shell outside any sandbox; it names any host process it cannot rule out, as described in [PID namespaces](mcp.md#pid-namespaces).
- Catalog path, storage, or working-directory rejection: follow [the catalog requirements](mcp.md#the-catalog).
