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

If you need an experiment, `phasesweep init` writes `starter.yaml` without overwriting files. See the [configuration guide](config.md) for other templates and reference lookups, including `explain_config`.

Before cataloging a source-checkout trainer, follow the [trainer environment preflight](config.md#source-checkout-preflight) to verify its imported library and supported arguments under the configured environment. Then create the catalog:

```bash
phasesweep mcp init-catalog --from ./starter.yaml
```

The command writes `catalog.yaml`, validates it, provisions a fresh [private state layout](mcp.md#fresh-mcp-state), and prints one line per entry with its `trainer cwd`, the directory its trainer starts in: the experiment's `execution.cwd`, or the entry's `cwd` when that is unset. Add another `--from` for each experiment; use `-o` for another catalog filename. The scaffold never replaces an existing catalog. Each entry takes its `id` from the config file name, sets `visible_params: none`, and leaves `description` and the `allow` block commented out, so it is read-only until you grant permissions (see [the catalog fields and permission model](mcp.md#the-catalog)). A config beneath the catalog's directory is written as a relative path with `cwd: "."` (the catalog's directory, and the trainer's while `execution.cwd` is unset). A config elsewhere is written as an absolute path with no `cwd`, so the entry starts in that config's own directory.

The scaffold records an absolute `state_dir` under `PHASESWEEP_HOME/mcp/<catalog-digest>`, or `${XDG_STATE_HOME:-~/.local/state}/phasesweep/mcp/<catalog-digest>` when unset; later environment changes do not move it. A missing absolute `PHASESWEEP_HOME` is created mode 0700; relative paths, symlink components and existing directories with another mode are refused. To place state elsewhere, edit `state_dir`. It must be writable by each participating shell, including sandboxes. Give the server and `phasesweep mcp call` the same catalog, and `recover-run` the same `--state-dir`; see [state ownership and format rules](mcp.md#fresh-mcp-state).

Validate the reviewed catalog with its absolute path:

```bash
phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
```

The check prints `ok` with permissions and `trainer cwd` for each ready entry, or `FAIL` with a reason and a `fix:` line. It exits 0 when all checks pass. Unreadable or orphaned run records that block launch produce one error naming them and exit 2 instead of the table. A run that needs [recovery](mcp.md#run-state-and-recovery) can coexist with `ok` entries and appears in tool reads with `recovery_required: true`. For launch-enabled entries the check covers the trainer directory and a program written as an executable path. Use `phasesweep run CONFIG --dry-run` to check a program found through `PATH`. A missing trainer can fail readiness while the server still serves results and other experiments. At startup, the server logs `trainer cwd` for every entry it serves.

### 3. Configure your client

PhaseSweep never reads or writes MCP client configuration files; the operator manages the client-owned server entry.

Configure your client's stdio server with the absolute `phasesweep-mcp` executable and reviewed catalog path, following its documentation for configuration scope and restart behavior:

```text
/absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

For a server on another machine, the client's command is an [SSH command](#over-ssh) instead.

To run the stdio server directly instead:

```bash
phasesweep mcp serve --catalog /absolute/path/to/catalog.yaml
```

For Codex CLI, the [official MCP instructions](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) use:

```bash
codex mcp add phasesweep -- /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
codex mcp list
```

To keep a client from connecting the server at all, disable it in that client's own MCP settings.

The client supplies the server's environment. If a `trial_command` program is missing from its `PATH`, set `PATH` in the client's server entry or follow the [trainer contract](config.md#trainer-contract) to set it in the experiment. You can also give the program as an absolute path, such as `/absolute/path/to/python train.py --config {config_path}`. The run's `failure.remediation` gives the fix, and its runner log, `<run-id>.log` under `state_dir/logs/`, names the program. For a study started from a shell, compare the [tracked trainer environment](runtime.md#resolve-a-changed-study-or-trainer-environment) before continuing from the client.

#### Over SSH

To serve a catalog that lives on another machine, make the client's stdio command an SSH command that starts the server there, with both paths absolute and no terminal allocated:

```text
ssh -T HOST /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml
```

A relative `--catalog` resolves against the server's working directory, and a bare `phasesweep-mcp` runs only if the session's `PATH` finds it; otherwise the client sees only the connection close. A terminal (`ssh -t`) mixes the server's log lines and echoed input into the protocol stream and breaks it.

`ssh -T HOST printenv PATH LANG` shows these values in the SSH session's environment. To set `PATH` for the server, use `ssh -T HOST env PATH=/absolute/path/to/env/bin:/usr/bin:/bin /absolute/path/to/phasesweep-mcp --catalog /absolute/path/to/catalog.yaml`. If you will also run the study from a shell, pin values such as `PATH` and `LANG` under the experiment's `env` before the first trial and make sure the remaining [tracked environment](config.md#trainer-contract) matches across both launch contexts.

To verify, use the absolute `phasesweep` path and the same `env PATH=…` prefix if you set one for the server. Start the dry run from the catalog entry's `cwd`, which defaults to the experiment file's directory. An explicit `execution.cwd` still controls the trainer's directory.

```bash
ssh -T HOST /absolute/path/to/phasesweep mcp check --catalog /absolute/path/to/catalog.yaml
ssh -T HOST 'cd /absolute/path/to/trainer-checkout && /absolute/path/to/phasesweep run /absolute/path/to/experiment.yaml --dry-run'
```

The dry run names a missing program when the study still needs trials. Confirm that nothing is printed before the server starts by running `ssh -T HOST true | wc -c`. It must print `0` because output from a shell start-up file corrupts the protocol stream.

A dropped connection ordinarily leaves the detached runner and its trainer going. Follow the [reconnection workflow](mcp.md#tool-workflow) after reconnecting. If the host kills everything the session started, the run needs [recovery](mcp.md#run-state-and-recovery). Run recovery and the other troubleshooting commands in a shell on the remote machine as the server's user.

For an [environment mismatch](runtime.md#resolve-a-changed-study-or-trainer-environment), the runner log lists the recorded and current values. MCP remediation omits those values. A replay of a finished study reuses its result and logs `reused winner ran under a different trainer environment`. Confirm the difference cannot affect the metric, or rerun the phase under a new `experiment` name.

### 4. Verify the connection

Restart or reconnect the client as its documentation requires. Confirm the connection with the read-only `list_experiments` operation; for example, ask the agent to use PhaseSweep to list the available experiments and permitted actions without launching anything. Name PhaseSweep in the request: the server's instructions tell agents to use its tools only when the user asks for it. The response contains catalog-approved IDs, descriptions, phase names, metrics, permitted actions and `catalog_status`; a stale catalog includes its reload refusal and operator remedy.

To check the same operation without a client, run it from a shell:

```bash
phasesweep mcp call list_experiments --catalog /absolute/path/to/catalog.yaml
```

See [shell calls](mcp.md#calling-a-tool-from-a-shell) for arguments, exit codes and monitoring. For an explicitly authorized run, follow the [tool workflow](mcp.md#tool-workflow) and [response shapes](mcp.md#reading-status-responses).

## Maintenance and troubleshooting

Use `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` after a catalog edit or when diagnosing server startup. If you replace the environment that supplies `phasesweep-mcp`, update the client-owned entry (or the SSH command) to the new absolute executable path using that client's documentation.

- `MCP support is not installed`: install the MCP extra in the environment that supplies `phasesweep-mcp`.
- The client cannot start the server: verify the paths in its [stdio entry](#3-configure-your-client) or [SSH command](#over-ssh), then run [the catalog check](#2-create-and-review-the-catalog).
- `action 'launch' is not permitted`, `action 'cancel' is not permitted` or `action 'from_phase' is not permitted`: check the [catalog permissions](mcp.md#the-catalog). For a refused cancellation, follow the operator step the error gives for that run.
- `catalog_status.stale: true`, or a launch or inspection refused because a catalog entry `failed to load` or the catalog file `does not validate`: run `phasesweep mcp check --catalog /absolute/path/to/catalog.yaml` and fix what it reports; the server's stderr log names the file. See [catalog reload behavior](mcp.md#the-catalog), including when a restart is required.
- `concurrency limit reached` or an experiment that already has a running sweep: [run state and recovery](mcp.md#run-state-and-recovery) says which blockers to await and which need the operator.
- `cannot prove launch capacity`, or an unreadable or orphaned record for one run ID: run the `phasesweep mcp recover-run` command the launch refusal names for each abandoned launch that stopped before spawning a runner. For each other run ID listed by `mcp check`, run `pgrep -af <run-id>` to list its runner and `ps axeww | awk '/[P]HASESWEEP_GENERATION_ID=<run-id>/ {print $1}'` to print its trial PIDs by environment. Run these on the server host as the server's user so `ps` can see the trial environment. Wait for the listed processes to exit or stop them. Move `runs/<run-id>.json` and `logs/<run-id>.*` out of `state_dir` only once both commands print nothing.
- `another launch is in progress`: a launch sharing this `state_dir` holds the launch lock; retry in a moment.
- Lock-directory access failure (`storage_unavailable`): set the same `PHASESWEEP_LOCK_DIR` for every participating shell, MCP server, and sandbox, following the [lock directory rules](runtime.md#storage-and-locks).
- `recovery_required: true`, unresolved launch, or uncertain cleanup: stop agent activity and follow [run state and recovery](mcp.md#run-state-and-recovery), running the command it names in a shell on the machine that holds the catalog.
- Unavailable terminal snapshot: follow `failure.remediation` and inspect run/server diagnostics when it asks. [Historical results](mcp.md#reading-status-responses) cannot be reconstructed from the current ledger.
- `runner_observable: false`, alone or with `recovery_required: true`, or a PID-namespace refusal: follow [PID namespace recovery](mcp.md#pid-namespaces).
- Catalog path, storage, or working-directory rejection: follow [the catalog requirements](mcp.md#the-catalog).
- Trainer environment cohort mismatch: follow the [environment-mismatch remedy](runtime.md#resolve-a-changed-study-or-trainer-environment) and check which [environment the client starts with](#3-configure-your-client).
- A run refused because its trial program is missing from `PATH`: see [client environment setup](#3-configure-your-client) or [SSH environment setup](#over-ssh).
- `its runner exited with code`: see [startup failure and the runner log](mcp.md#run-state-and-recovery).
