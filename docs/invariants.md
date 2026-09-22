# Durability invariants

PhaseSweep's local ledger, artifact-root binding, publication transaction, and
operator recovery are the only places where a wrong ordering silently destroys
work. Three review passes found bugs only here, because the ordering rules
lived in comments. This page is the enforceable version: every rule names the
function that holds it and the test that fails when it stops holding.

## Agent handoff

```
1. Take the experiment lock before touching any durable state.
2. Validate read-only before writing anything: binding check, then format scan, then bind the tree, then claim studies, then open live storage. The order is fixed and the typed handle proves it ran.
3. Pure read paths (read_status, read_winners, CLI status/show-winners, MCP snapshot) never construct file-backed storage and never write bytes. Recovery inspect validates binding+format before any open and refuses a pre-cutover ledger with bytes unchanged.
4. Only src/phasesweep/engine/ledger.py may construct Optuna or sqlite3 storage. tests/test_ledger_contract.py enforces this with equality ratchets that you update in the same change.
5. Error text is authoritative; wraps preserve the operator's remediation; never fix a failing test by weakening a message match or an ordering assertion.
```

## Invariants

Rows marked `(this PR)` in the Enforced-by cell name the chokepoint API that
the ledger milestone introduces; the rest cite code that exists today.

| # | Invariant | Enforced by | Owning test |
| --- | --- | --- | --- |
| 1 | The experiment lock is held before any durable state is read or written. | `phasesweep.engine.locking._experiment_lock`, entered as the outermost durable-state context in `phasesweep.engine.run._run_experiment_outcome` | `tests/test_locking.py::test_run_experiment_holds_experiment_lock_for_duration` |
| 2 | Validation is read-only and completes before anything is bound: binding check, then format scan. | `phasesweep.engine.ledger.validate_ledger` (this PR) | `tests/test_format_cutover.py::test_live_openers_reject_unvalidated_storage` |
| 3 | The chokepoint order is validate, bind tree, claim, open live, and the typed handle proves it: `open_phase_study` accepts only a `ClaimedLedger`. | `phasesweep.engine.ledger.validate_ledger` returning `ValidatedLedger`, `claim_ledger` returning `ClaimedLedger`, `open_phase_study` requiring `ClaimedLedger` (this PR) | `tests/test_format_cutover.py::test_claim_ledger_binds_tree_then_returns_bound_handle`, `tests/test_ledger_contract.py::test_storage_constructors_are_called_only_in_the_ledger_module` |
| 4 | Both ownership directions are resolved before a generation is claimed: the root binding names the ledger and every study's artifact-root attribute names the root. | `phasesweep.engine.artifact_roots._load_and_check_artifact_roots`, `_validate_artifact_root_binding`, `_artifact_root_claim_needed` | `tests/test_fingerprint.py::test_second_workdir_is_rejected_and_leaves_the_bound_root_untouched` |
| 5 | The tree binding is written before any study is root-claimed, so a crash cannot leave a study pointing at a tree that does not name the same ledger. | `phasesweep.engine.artifact_roots._load_and_check_artifact_roots` writes through `_validate_artifact_root_binding` before its `_claim_study_artifact_root` loop | `tests/test_fingerprint.py::test_refused_multi_phase_binding_claims_nothing` |
| 6 | Pure read paths (`read_status`, `read_winners`, CLI `status` and `show-winners`, MCP snapshot) never construct file-backed storage and never write bytes. | `phasesweep.engine.ledger.read_phase_trial_stats`, `open_existing_study`, `open_registry_study` (this PR) | `tests/test_ledger_read_paths.py::test_read_path_never_constructs_file_backed_storage_or_writes_bytes` |
| 7 | Recovery inspect validates binding and format before any open, and refuses a pre-cutover ledger with the bytes unchanged. | `phasesweep.mcp.recovery._load_recovery_studies` routed through `phasesweep.engine.ledger.validate_ledger` and `open_existing_study` (this PR) | `tests/test_ledger_read_paths.py::test_recovery_inspect_never_writes_to_a_golden_ledger` |
| 8 | The attempt registry is scanned first, and an attempt is durably registered before its process is launched. | `phasesweep.engine.attempts._preflight_active_attempts` (driven by `phasesweep.engine.guards._preflight_existing_studies`), then `phasesweep.engine.attempts._register_active_attempt` before the GPU lease and launch | `tests/test_stale_reaper.py::test_unpersistable_attempt_refuses_to_launch_its_trainer` |
| 9 | Preflight completes before a claim: environment preflight before the generation claim, and the full continuation chain before any attempt is registered or the generation is marked running. | `phasesweep.engine.run._preflight_missing_reached_phase_environments` before `phasesweep.engine.generation._claim_generation`; `phasesweep.engine.guards._preflight_existing_studies`, `phasesweep.engine.resume._preflight_skipped_winners`, `phasesweep.engine.resume._preflight_reached_fingerprint` before `phasesweep.engine.attempts._register_active_attempt` | `tests/test_runtime_behavior.py::test_existing_tree_preflights_missing_reached_phase_before_claim_or_topup` |
| 10 | No Optuna trial reaches a terminal state without its durable outcome record; a failed outcome write leaves the trial RUNNING. | `phasesweep.engine.phase._run_phase` writes the trial outcome attribute before any terminal transition and raises `phasesweep.engine.phase._TrialOutcomeUnrecordedAbort` instead | `tests/test_runtime_behavior.py::test_persistent_outcome_write_failure_leaves_trial_running_until_recovery` |
| 11 | An unreadable ledger means cleanup is uncertain, never that cleanup is confirmed. | `phasesweep.engine.cleanup._reap_stale_trials` raises `ProcessCleanupUncertainError`, `phasesweep.engine.attempts._registry_attempt_fail_stale_trial` retains the entry, and `phasesweep.engine.guards._preflight_existing_studies` marks the report uncertain | `tests/test_stale_reaper.py::test_registry_storage_failure_marks_cleanup_report_uncertain` |
| 12 | Publication is a transaction: the immutable generation is durable first, the pointer commits second, and shutdown signals are absorbed until the commit lands. | `phasesweep.engine.generation._publish_generation` running inside `phasesweep.runtime.process.absorb_shutdown_signals`, with `_validate_generation_publishable` before the pointer write and `_write_generation_record_once` for the immutable record | `tests/test_publication_transaction.py::test_shutdown_signal_during_publication_is_absorbed_until_committed` |
| 13 | A stale PID is never authority: identity is PID plus start time plus boot id, and a differing boot id settles a reboot without signalling anything. | `phasesweep.mcp.runs.identity_from_earlier_boot`, `phasesweep.mcp.runs.RunStore.from_earlier_boot`, `phasesweep.runtime.process.is_same_live_process`, `phasesweep.runtime.process.cleanup_stale_trial_process` | `tests/test_mcp_runs.py::test_state_cleanup_uncertain_on_pid_reuse_mismatch`, `tests/test_stale_reaper.py::test_cleanup_stale_trial_process_accepts_prior_boot_without_signalling` |
| 14 | A dead runner with no terminal status stays in the live set until `recover-run` decides; liveness alone never concludes the run. | `phasesweep.mcp.runs.RunStore.state`, `_settle_dead_runner`, `recovery_required`, resolved only by `phasesweep.mcp.recovery.recover_run` | `tests/test_mcp_runs.py::test_dead_runner_without_status_stays_live_until_recovery_evidence` |
| 15 | Every operator-facing error declares its remediation, and a wrap preserves it rather than replacing it with the wrapper's own advice. | `phasesweep.errors.PhaseSweepError.action`, `phasesweep.errors.PhaseSweepError.rewrap`, `phasesweep.errors.OperatorAction` (this PR) | `tests/test_error_routing.py::test_every_operator_error_declares_its_action`, `tests/test_error_routing.py::test_operator_action_survives_wrap`, `tests/test_mcp_runner.py::test_cleanup_uncertain_remediation_follows_the_operator_action` |
| 16 | The refusals are tested against real ledgers produced by the public API and by the preserved 0.3.1 tag, and read paths run under a monkeypatch that raises if a storage constructor is called. | `tests/fixtures/make_ledger_fixtures.py`, `tests/ledger_fixtures.py::forbid_file_backed_storage` (this PR) | `tests/test_ledger_read_paths.py::test_current_fixture_carries_this_release_format`, `tests/test_ledger_read_paths.py::test_required_ledger_fixtures_are_present` |

Row 5 has no fault-injection test that crashes between the tree write and the
study claim; `test_refused_multi_phase_binding_claims_nothing` covers the
weaker "a refused check claims nothing" guarantee on the same code path. Row 9
is stated as the code actually orders things: `_claim_generation` runs before
the continuation preflight chain, so the chain's guarantee is against the
attempt claim and the running-state transition, not against the generation
claim.

## Decision record: recovery machinery

Three pieces of machinery exist only to survive a hard exit. Each is cheap to
delete and expensive to be wrong about, so the cost is written down here rather
than rediscovered by the next reviewer.

### `recover-run`

Prevents a catalog capacity-slot leak after a hard exit. When a detached runner
is SIGKILLed or the host crashes, the run handle survives with no terminal
status. `RunStore.state` deliberately keeps that run live (invariant 14),
because a dead runner with no status is indistinguishable from one whose trials
are still being reaped, so the run holds one of the experiment's capacity slots
and no later launch can proceed. `recover-run` is the only path that reads the
durable evidence and decides.

Cost: 880 lines in `src/phasesweep/mcp/recovery.py` (whole module) plus about
55 lines of Click wiring for `phasesweep mcp recover-run` in
`src/phasesweep/cli.py`, roughly 950 lines.

This PR makes that cost deliberate: `_load_recovery_studies` stops borrowing
the storage-private `_load_existing_phase_study` and opens through
`phasesweep.engine.ledger.open_existing_study`, so recovery validates the
binding and the format before it opens anything (invariant 7), and the
recovery tests that spawn a real runner carry the `integration` marker and run
outside the fast tier.

### Runner boot identity

Prevents killing an unrelated process after a reboot. PID plus `/proc` start
time is unique only within one boot: the kernel restarts both counters, so a
saved pair can name a process the host started after rebooting, and a cancel
or a cleanup would signal it. A recorded boot id settles the question in the
safe direction. A differing boot id proves nothing from that boot survives, so
cleanup is complete without sending a signal; an unknown boot id on either side
refuses to signal at all.

Cost: about 180 lines. Counted as function spans in `src/phasesweep/mcp/runs.py`:
`identity_from_earlier_boot` (21), `ProcessIdentity` (8), `cleanup_identity`
(23), `RunStore.from_earlier_boot` (11), `_read_cleanup_identity` (71), and
`_valid_optional_boot_id` (14), which is 148; plus about 30 lines in
`src/phasesweep/mcp/runner.py` where the runner reads the boot id and refuses
to persist a launch receipt without one.

This PR makes that cost deliberate on the tier side: `mcp.runs` never
constructs storage, so the chokepoint does not touch it, but the runner tests
that spawn a real subprocess to exercise the identity carry the `integration`
marker and leave the fast tier.

### MCP restart recovery and the persisted spawned handle

Prevents a duplicate launch after the server restarts. A detached runner
outlives its parent by design. If the server restarts with no handle on disk,
it has no record of that runner and a second `launch_run` spawns a duplicate
sweep against the same ledger. The handle is created before `Popen`, replaced
by the runner's own receipt once the runner has persisted its identity, and
confirmed through a ready/ack handshake, so no child ever exists without a
durable handle naming it.

Cost: about 270 lines in `src/phasesweep/mcp/server.py`, counted as function
spans: `PhaseSweepMCP._spawn` (179), `_terminate_failed_spawn` (66), and
`_pending_handle` (23).

This PR makes that cost deliberate on the tier side as well: every test that
actually spawns the runner to prove the handshake carries the `integration`
marker.

### Rejected: removing any of the three

Removing `recover-run` leaves the leaked capacity slot with no owner, because
the state machine is deliberately unable to conclude a dead runner on its own.
Removing boot identity means signalling a PID a reboot may have reassigned,
which is killing an unrelated process on the operator's host. Removing the
persisted spawned handle means two sweeps writing into one ledger. Each
removal converts a bounded amount of code into an unbounded class of silent
corruption, so all three stay.

## Test tiers and gates

Plain `pytest` is authoritative and stays the merge and release check. The fast
review tier is:

```bash
pytest -m "not hardware and not integration"
```

A test is `@pytest.mark.integration` when it spawns real processes, waits on
wall-clock time, or drives a multi-step durable recovery workflow. That marker
is not a convention: `tests/tiers.py` recognizes the process and wall-clock
primitives statically, resolving through same-module helpers and fixtures, and
the `pytest_collection_modifyitems` hook in `tests/conftest.py` fails
collection when a test uses one without the marker. There is no escape hatch,
so the fast tier cannot quietly absorb a slow test.

`tests/test_ledger_contract.py` is the static contract behind invariant 3. It
parses every module under `src/phasesweep` and compares the set of banned
storage references against a ratchet dict for equality, not containment, so
adding a call site fails and removing one fails until the ratchet is tightened
in the same commit. Its three tests are
`test_storage_constructors_are_called_only_in_the_ledger_module`,
`test_ledger_private_names_are_not_imported_outside_the_ledger_module`, and
`test_ledger_public_api_returns_concrete_types`.

The remaining gates are `mypy src` under the explicit `[tool.mypy]` config in
`pyproject.toml`, `ruff check .`, and `ruff format --check .`. The last
milestone of this PR adds a `.pre-commit-config.yaml` that runs ruff, pathlint,
doc-check, and those three contract tests at commit time, and `mypy` at push
time, so the cheap static gates fire before a reviewer sees the change and the
slower type check fires before the branch leaves the machine.
