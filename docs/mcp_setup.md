# MCP agent setup

PhaseSweep owns the MCP server and its operator-reviewed catalog. Your MCP
client owns its configuration and any client-specific instructions.

> [!NOTE]
> PhaseSweep does not create, edit, inspect, or remove client files. You add
> the server entry to your client yourself in step 3.

Requirements: Python 3.11+, the [MCP runtime requirements](mcp.md), a
PhaseSweep experiment that passes validation, and a client with local stdio
MCP support.

## Setup

### 1. Install the MCP extra

Install PhaseSweep and its optional MCP dependency in the Python environment
that provides the server executable:

```bash
python -m pip install "phasesweep[mcp]"
```

Contributor and editable-install setup is in [development](development.md).
Install the tracking extra needed by the cataloged experiment in that same
environment: `phasesweep[mcp,wandb]` for W&B or `phasesweep[mcp,mlflow]` for
MLflow. Config validation and frozen scientific result reads do not authenticate.
New work needs the selected SDK and access to its configured account or server.
MLflow additionally requires an existing experiment; the client extra does not
start a tracking server. See [native MLflow setup and run ownership](mlflow.md).

### 2. Create and review the catalog

If you do not have an experiment yet, `phasesweep init` creates an
installed-package starter named `experiment.yaml` without overwriting files.
The [configuration reference](config_reference.yaml) describes the accepted
experiment fields, including backend selection and environment composition.

Before cataloging a source-checkout trainer, verify the package actually
imported by the configured Python executable. Running a checkout's example
script does not make Python import that checkout's library. Match the example
to its installed library, or explicitly select the checkout's source directory:

```yaml
execution:
  cwd: /absolute/path/to/transformers
  inherit_env: none
env:
  PYTHONPATH: /absolute/path/to/transformers/src
```

Run a CPU-only preflight with that same executable and explicit environment;
for this example:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/absolute/path/to/transformers/src \
  /absolute/path/to/python -c 'import transformers; print(transformers.__version__); print(transformers.__file__)'
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/absolute/path/to/transformers/src \
  /absolute/path/to/python /absolute/path/to/transformers/examples/pytorch/token-classification/run_ner.py --help
```

Carry all required settings into the experiment's composed environment:
`inherit_env: none` drops ambient `PYTHONPATH` and other shell settings.
Use the actual example's `--help` to choose supported arguments; configuration
validation does not execute the trainer or establish library compatibility.
For Transformers with managed MLflow, also apply the
[full-script run context](mlflow.md#transformers-trainer) before launch so final
evaluation remains in the bound run. The W&B integration uses its own native
callback and managed attempt identity.

```bash
phasesweep mcp init-catalog --from ./experiment.yaml
```

The command writes `catalog.yaml`, validates it, and provisions a fresh
[private state layout](mcp.md#fresh-mcp-state). Add another `--from` for each
experiment; use `-o` for another catalog filename. The scaffold never replaces
an existing catalog.

> [!IMPORTANT]
> Review the generated file against
> [the catalog fields and permission model](mcp.md#the-catalog) before
> connecting a client. The catalog decides which experiments an agent can see,
> which sampled values it can read, and which side effects it may request.

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
