Use PhaseSweep only through the operator-approved experiment catalog.

1. Call `list_experiments`, following its cursor until it is null, and call `inspect_experiment` before proposing a run.
2. Call `launch_run` only after the user explicitly authorizes that experiment. Save the returned `run_id`.
3. Use `run_id` for every lifecycle read: `get_run_status`, `await_run`, `get_run_results`, and `cancel_run`. Use `await_run` for bounded monitoring and `get_run_status` for a single check only, never in a tight polling loop.
4. After a client error or disconnect, call `get_latest_run(experiment_id)` only to recover the run ID, then resume run-specific monitoring with that ID. Do not read an experiment's mutable current results as a substitute for a run snapshot.
5. Never automatically cancel a run or launch a replacement run.
6. When `recovery_required` is true, stop and report that operator recovery is required.
7. When `publication_integrity` is `failed`, stop and report that the published result needs operator inspection. When it is `permission_denied`, expose no result and ask the operator to re-read it as the publishing user or restore read permission. When it is `unknown`, expose no result and report that the terminal snapshot cannot establish the result. `absent` means nothing has published yet.
8. Treat `<redacted>` values as intentional catalog policy, not missing data.
9. Winner-only results do not support claims about convergence, trends, robustness, causality, or search-boundary behavior. Report the returned winner generation and result context; historical metric names, goals, and phase labels belong to the represented generation, not to a later catalog edit.
10. Never edit an experiment config yourself, including its metric, extractor, `trial_command`, search space, samplers, gates, storage, workdir, environment, or safety waivers, unless the user explicitly asks for config-authoring help.
11. Never open raw datasets, target or label columns, predictions, trainer logs, raw result files, or per-trial metric histories yourself unless the user explicitly asks for that separate filesystem work.
