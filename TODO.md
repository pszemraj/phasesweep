# phasesweep TODO

## v0.6+

- Design safe multi-host orchestration with per-trial leases and heartbeat-based stale
  trial reaping. The current stale reaper is intentionally single-orchestrator,
  single-host only because a second orchestrator against the same study could mark live
  trials from the first orchestrator as failed.
