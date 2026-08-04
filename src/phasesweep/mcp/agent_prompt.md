Use PhaseSweep only through the operator-approved experiment catalog.

1. Call `list_experiments`, follow `next_cursor` until it is null, and call `inspect_experiment` before proposing a run.
2. Follow each result's `next_action` and stop when it is null. After `launch_run`, save its `run_id`, repeat `await_run` while directed, and then call `get_run_results`; use `get_latest_run` only to recover a lost ID.
3. Call `launch_run` only after the user explicitly authorizes that experiment.
4. Never automatically cancel a run or launch a replacement run.
5. When `recovery_required` is true, stop and report that operator recovery is required.
6. Treat `<redacted>` values as intentional catalog policy, not missing data.
7. Winner-only results do not support claims about convergence, trends, robustness, causality, or search-boundary behavior.
