# phasesweep TODO

## v0.6+

- Add safe multi-host orchestration: per-trial leases, heartbeats, and stale-trial
  reaping that cannot fail another host's live trials. Current limits are described in
  [Concurrency model](README.md#concurrency-model).
