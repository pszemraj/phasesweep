# MCP agent setup

Requirements: Python 3.11+, the [MCP runtime requirements](mcp.md), a PhaseSweep experiment that passes validation, and a client with local stdio MCP support.

## Setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the Python environment that provides the server executable:

```bash
pip install "phasesweep[mcp]"
```

Contributor and editable-install setup is in [development](development.md). Install the tracking extra needed by the cataloged experiment in that same environment: `phasesweep[mcp,wandb]` for W&B or `phasesweep[mcp,mlflow]` for MLflow. A run needs the selected SDK and access to its configured account or server. See [native MLflow setup and run ownership](mlflow.md) for MLflow's tracking server and experiment.

### 2. Create and review the catalog

If you need an experiment, `phasesweep init` writes `starter.yaml` without overwriting files; `phasesweep init --list` lists the other [templates](config.md#templates). Use the [configuration guide](config.md) and `phasesweep explain KEY` for authoring; connected agents can use `explain_config` when asked for that help.

Before cataloging a source-checkout trainer, follow the [trainer environment preflight](config.md#source-checkout-preflight) to verify its imported library and supported arguments under the configured environment. Then create the catalog:

```bash
phasesweep mcp init-catalog --from ./starter.yaml
```

The command writes `catalog.yaml`, validates it, provisions a fresh [private state layout](mcp.md#fresh-mcp-state), and prints one line per entry with the `cwd` its trainer will start in. Add another `--from` for each experiment; use `-o` for another catalog filename. The scaffold never replaces an existing catalog. Each entry takes its `id` from the config file name, sets `visible_params: none`, and leaves `description` and the `allow` block commented out, so it is read-only until you grant permissions (see [the catalog fields and permission model](mcp.md#the-catalog)). A config beneath the catalog's directory is written as a relative path with `cwd: "."` (the catalog's directory, and the trainer's while `execution.cwd` is unset). A config elsewhere is written as an absolute path with no `cwd`, so the entry starts in that config's own directory.

The scaffold records an absolute `state_dir` under `PHASESWEEP_HOME/mcp/<catalog-digest>`, or `${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>` when unset; later environment changes do not move it. A missing absolute `PHASESWEEP_HOME` is created mode 0700; relative paths, symlink components and existing directories with another mode are refused. To place state elsewhere, edit `state_dir`. It must be writable by each participating shell, including sandboxes. Give the server and `phasesweep mcp call` the same catalog, and `recover-run` the same `--state-dir`; see [state ownership and format rules](mcp.md#fresh-mcp-state).

Validate the reviewed catalog with its absolute path:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

This exercises server startup validation and provisions/probes its private state layout without launching a sweep. It prints one line per entry, `ok` with the actions it grants or `(read-only)` and the `cwd` its detached runner starts in, or `FAIL` with the reason and, when one applies, a `fix:` line, and exits 0 only when every entry loads.

### 3. Configure your client

PhaseSweep never reads or writes MCP client configuration files; the operator manages the client-owned server entry.

Configure your client's stdio server with the absolute `phasesweep-mcp` executable and reviewed catalog path, following its documentation for configuration scope and restart behavior:

```text
/absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

For Codex CLI, the [official MCP instructions](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) use:

```bash
codex mcp add phasesweep -- /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
codex mcp list
```

To keep a client from connecting the server at all, disable it in that client's own MCP settings.

The client starts the server with the client's environment, not your shell's, and trainers inherit it, so a `trial_command` program resolves on the `PATH` the client starts the server with. That `PATH` often lacks the Python environment your shell activates. Give `trial_command` an absolute interpreter path, such as `/absolute/path/to/python train.py --config {config_path}`, or set `PATH` in the client's server entry. A run whose program is missing from that `PATH` is refused before any trial starts: its `failure.remediation` gives this fix, and its runner log, `<run-id>.log` under `state_dir/logs/`, names the program and the `PATH` it searched.

### 4. Verify the connection

Restart or reconnect the client as its documentation requires. Confirm the connection with the read-only `list_experiments` operation; for example, ask the agent to use PhaseSweep to list the available experiments and permitted actions without launching anything. Name PhaseSweep in the request: the server's instructions tell agents to use its tools only when the user asks for it. The response contains catalog-approved IDs, descriptions, phase names, metrics, permitted actions and `catalog_status`; a stale catalog includes its reload refusal and operator remedy.

To check the same operation without a client, run it from a shell:

```bash
phasesweep mcp call list_experiments --catalog /absolute/path/to/catalog.yaml
```

See [shell calls](mcp.md#calling-a-tool-from-a-shell) for arguments, exit codes and monitoring. For an explicitly authorized run, follow the [tool workflow](mcp.md#tool-workflow) and [response shapes](mcp.md#reading-status-responses).

## Maintenance and troubleshooting

Use `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` after a catalog edit or when diagnosing server startup. If you replace the environment that supplies `phasesweep-mcp`, update the client-owned entry to the new absolute executable path using that client's documentation.

- `MCP support is not installed`: install the MCP extra in the environment that supplies `phasesweep-mcp`.
- The client cannot start the server: verify its client-owned stdio entry has absolute executable and catalog paths, then run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml`.
- `action 'launch' is not permitted`, `action 'cancel' is not permitted` or `action 'from_phase' is not permitted`: the running server applies a catalog change on the next call, but a grant never reaches [a run launched without cancel permission](mcp.md#the-catalog).
- `catalog_status.stale: true`, or a launch or inspection refused because the catalog or a config it lists `does not validate`, or an experiment whose entry `failed to load`: run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` and fix what it reports; the server's stderr log names the file, and a server that started with a failing entry printed it there too. A refusal for a different `state_dir` requires a server restart. Other reads keep using the entries that loaded and saved run snapshots. See [the catalog](mcp.md#the-catalog).
- `concurrency limit reached` or an experiment that already has a running sweep: [run state and recovery](mcp.md#run-state-and-recovery) says which blockers to await and which need the operator.
- `cannot prove available launch capacity`: `state_dir` holds unreadable or orphaned run records, which the refusal lists. Run the `phasesweep mcp recover-run` command it names for each pre-spawn orphan; for each other listed run id, check `pgrep -af <run-id>` on the server host and, where no runner is alive, move `runs/<run-id>.json` and `logs/<run-id>.*` out of `state_dir`.
- `another launch is in progress`: a launch sharing this `state_dir` holds the launch lock; retry in a moment.
- Lock-directory access failure (`storage_unavailable`): set the same `PHASESWEEP_LOCK_DIR` for every participating shell, MCP server, and sandbox, following the [lock directory rules](runtime.md#storage-and-locks).
- `recovery_required: true`, unresolved launch, or uncertain cleanup: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery).
- Unavailable terminal snapshot: follow `failure.remediation`; it distinguishes a runner that exited without final status from one whose host boot ended, for which process cleanup is unnecessary. Other capture failures require inspection of run/server diagnostics. [Historical results](mcp.md#reading-status-responses) cannot be reconstructed from the current ledger.
- `runner_observable: false`, alone or with `recovery_required: true`, or a PID-namespace refusal: follow [PID namespace recovery](mcp.md#pid-namespaces).
- Catalog path, storage, or working-directory rejection: follow [the catalog requirements](mcp.md#the-catalog).
- Trainer environment cohort mismatch: the [launching server or `mcp call` shell](mcp.md#the-catalog) supplies the trainer's base environment; reconcile it using the [trainer contract](config.md#trainer-contract).
- A run refused because its trial program is not on the trainer's `PATH`: the client's `PATH` lacks it; [configure the client](#3-configure-your-client) or name the program by absolute path in `trial_command`.
- `its detached runner exited with code`: the runner failed before it started work; its own error is in `<run-id>.log` under `state_dir/logs/`.
