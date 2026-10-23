# AGENTS.md — phasesweep

> [!IMPORTANT]
> **Read `LOCAL.md` (beside this file) before running anything.** It holds this machine's setup: which Python environment to use and how to invoke it, tool paths, hardware and GPU notes. This file holds only the repository's rules.

phasesweep automates hyperparameter sweeps over a training codebase on a single node with one or a few GPUs. It is unstable and unfinished, and nobody uses it yet, not even the maintainer. Most of it was written by agents, so treat code, defaults and docs as changeable unless this file says otherwise or the intent is blindingly obvious.

---

## Operating Principles

- **Be thorough.** Execute every instruction fully, and be able to justify completing each one.
- **Be direct and critical.** State conclusions, correct errors and challenge flawed assumptions. No praise, disclaimers or preamble.
- **Be decisive and specific.** Take evidence-based positions with concrete numbers and examples. When options exist, recommend one and justify it.
- **Handle ambiguity with precision.** If a request lacks necessary detail, name exactly what is missing.

## Scope and Decisions

- **Single-node scope.** Check every reviewer finding and proposal, including your own, against the single-node sweep use case. Say plainly when a finding only matters outside it. Test ergonomics (tiers, guards, run-selection tooling) is in scope.
- **One way to do each thing.** The repo's recurring failure is a "this would be better with X" proposal that adds a second, ad-hoc mechanism beside an existing one. Review then finds bugs in X, and closing them breeds more. Find the existing mechanism (see Code Search) and extend it. Prefer refusing an unusual input over supporting it, and narrowing or removing an edge feature that keeps growing bugs over hardening it.
- **Decide, don't ask, for anything that follows from the request.** Raising a default the user's goal requires, fixing obvious drift, or anything cheap to measure (measure it): make the change and call it out at the top of the report under "Changed:". Never end a report with a hedged "needs your call" item. Genuine design questions still go to the user: whether a feature or check should exist at all, or something they might reasonably disagree with.
- **No unrequested features or defensive machinery.** Integrity checks here exist for reproducibility, not secrecy. Store non-secret provenance (PATH, HOME, USER, LANG, TZ) in plain text so a refusal can show old → new. Keep credentials out by scoping (`passthrough_env`), not by hashing everything. Call out any addition beyond the request at the top of the report.
- **Delete baggage, then report.** When a check, script, test or fallback is broken or stale, first ask whether it should exist at all. One nobody noticed breaking is unused: delete it rather than repair it, one commit per removal category. Then report what was cut and why, without asking for list approval first.
- **Interruptions are reality.** Ctrl-C, killed runners, OOM, closed terminals and reboots are normal on one machine. Work lost to one is an engine bug to fix. It is never a documented "known cost" or an optional decision handed to the user. The same goes for any other real-world condition a probe or review surfaces.
- **Feedback is a discoverability report.** If any reader missed something that exists (an agent using phasesweep on a real project, the maintainer, an outside reviewer), the entry point they would have hit is broken. Never reply "FOO already exists, so the report is wrong." Fix the surface they saw: error text, `validate` / `run --dry-run` output, an MCP tool description, or the doc section they land on. A rule that a change in another module could break belongs in `docs/invariants.md`, with the code that holds it and the test that proves it. A docstring alone is not enough.
- **Declined requests.** When a feedback or review round deliberately skips a request, add it with its reason to "Declined requests" in `docs/development.md` on the same branch. Check that list before treating an item as new work.

## Compatibility and Persisted Formats

- **No backward compatibility.** There is no older data to stay compatible with. When a format changes, change it outright. Do not keep readers, migrations or normalizers for older on-disk formats, fields that "predate the record", older frozen MCP snapshots, or deprecated aliases. Do not keep bespoke detection of specific legacy formats. Do not accept a reviewer's "keep accepting version N-1".
- **Every persisted-format change bumps two versions in the same change.** No-BC covers *reading* old data, never *refusing* it cleanly. Bump the record's own constant (`EVIDENCE_PROVENANCE_SCHEMA_VERSION`, `ATTEMPT_REGISTRY_SCHEMA_VERSION`, `MCP_STATE_FORMAT_VERSION`, …). Also bump the front-door stamp that guards where the record lives: `STUDY_SCHEMA_VERSION` (`engine/state.py`) for anything in the Optuna ledger, including trial attrs, or `ARTIFACT_ROOT_BINDING_SCHEMA_VERSION` (`engine/artifact_roots.py`) for anything under the artifact root. Only the stamps give the clean up-front refusal; without them old data fails late, misreported as corruption (this happened twice). Then follow `tests/fixtures/ledgers/README.md` to move the fixture targets and regenerate. In every review, list each persisted format the diff touches and check that both its version and its stamp moved.
- In PR replies and public docs, phrase the policy as "the repo is still unstable and we aren't supporting backward compatibility until it stabilizes". Never write "no one uses it" publicly.

## Git

- One logical change per commit, conventional-commit messages, validated at commit time. Prefix `WIP:` for incomplete work. Stage explicit paths only.
- **Never push.** The user pushes. "Finish", "fix", "verify" and "update the PR" do not authorize a push, and each push request authorizes exactly one. When a push is pending, hand the user the command, with the environment prefix from `LOCAL.md` (the pre-push hook needs it).
- **Never merge**, locally or remotely: no `git merge`, merge-producing `git pull`, `gh pr merge` or auto-merge. The user squash-merges on GitHub.
- **Never create, switch or delete branches without explicit permission.** All work, including cleanups and follow-ups, stays on the branch the user designated, even when a separate branch looks tidier. Exception: temporary `worktree-agent-*` branches and their worktrees are always deleted once their content has landed, without asking.
- Squash merge means only the final tree reaches main. Do not rebase to hide remove-then-restore churn inside a PR, and don't flag it in reviews; judge the end state.
- No destructive git commands (`reset --hard`, force-push, reverting unrelated changes) on the user's branches without explicit confirmation.

## Versioning and Releases

- Never make, suggest or mention a package version bump in any summary, plan, commit or PR, not even a one-line "this may warrant a bump". Never create tags, GitHub releases or version-numbered release notes. A GitHub release publishes to PyPI, and releasing is the user's step.
- The version comes from git via `setuptools-scm` and is read at runtime in `phasesweep/_metadata.py` with `importlib.metadata`. Never reintroduce `_version.py` or a hardcoded version. Internal modules import `__version__` from `phasesweep._metadata`, and it stays the first import in `phasesweep/__init__.py`.
- Handoff is the source tree. Never create packaging archives, wheel/sdist deliverables or acceptance tarballs as deliverables. Local-only packaging checks may build under `/tmp` and must clean up afterwards.
- Persisted-format version constants are a different thing (see above).

## Environment and Commands

- Use the Python environment `LOCAL.md` names, with phasesweep installed editable (`pip install -e ".[dev]"`). Never create a virtual environment (`/tmp`, `.venv`, `venv` or anywhere else) unless the user explicitly asks. Before trusting any test result, confirm `inspect.getfile(phasesweep)` points into this checkout's `src/`, not site-packages.
- Every git hook is `language: system` and resolves tools from `PATH`, so that environment must come first on `PATH` for every commit and push, and for test runs that spawn a trainer `python`. If a hook fails for environment reasons, say so at once and never work around it silently.
- `addopts` already has `-q`, so `pytest -q` yields `-qq` and no count line. Never pipe a full run through `tail`. Redirect it to a file and grep `^FAILED|^ERROR` plus the count line. Run pytest with no concurrent lint or type check: the process-supervision tests are timing-sensitive.
- Never write tests or smokes against default caches or user state. Use explicit temp roots: a fresh 0700 `PHASESWEEP_LOCK_DIR`, a scratch `XDG_STATE_HOME`, and a scratch repo under `$TMPDIR`. Clean up afterwards. When testing whether phasesweep pollutes a user repo, run `git config core.excludesFile /dev/null` in the scratch repo first, so a global gitignore can't hide the result.
- Keep generated artifacts out of the worktree and commits: `dist/`, `runs/`, `build/`, `*.egg-info`, caches and `__pycache__`. Leave ignored local caches alone unless cleanup is requested.

## Validation

- **If you changed runnable code, run it now**, with minimal input or a short timeout, before anything else. Lint and type checks prove little by themselves. Exercise the changed path, confirm state after mutations, and never trust exit codes or log lines alone.
- Storage, write paths, publication and recovery changes need a check that survives a restart or interruption.
- `docs/development.md` ("Quality gates", "Test tiers") owns the checks and test lanes. During iteration, run the mandatory checks plus the changed subsystem's tests. Plain `pytest` is the full gate.
- mypy runs only at pre-push, so passing commit hooks prove nothing about types. Before handing off any `src/` change, run `pre-commit run --all-files --hook-stage pre-push` with the environment on `PATH`.
- Scale validation to the change. Docs-only changes need targeted checks for the touched files.
- A test that tampers with a published artifact must re-anchor the summary pointer (`tests/ledger_fixtures.py`: `reanchor_summary_pointer`, `reseal_winner`) and assert the intended error text. An outcome-only assert usually passes on an earlier integrity check. The same shadowing hits config tests whose input a field validator rejects first.

## CI and Hooks

- CI minutes cost the user money. Outside the approved scope below, CI stays static: no matrix, no full or integration suite, no build or install checks. Never quietly reintroduce removed checks or move them into hooks or other workflows.
- **Approved scope, do not propose removing it:** the single Linux job in `.github/workflows/ci.yml`, on one Python version, runs Ruff check and format, mypy, the module-size check, whole-suite collection, and the contract tests (`tests/test_ledger_contract.py`, `tests/test_error_routing.py`, `tests/test_ledger_read_paths.py`). The test step's budget is about 20 seconds, with 40 seconds as the ceiling; going past about 20 seconds needs new approval. Every hook in `.pre-commit-config.yaml` is mandatory, including the contract-test and collection hooks.

## Sweep and Experiment Ownership

> [!IMPORTANT]
> You do **not** own experiment execution. The user does.

- Never launch sweeps on the user's projects, long training jobs or W&B/MLflow-logging runs unless the user says to run that exact command. Hand them the exact command and explain what it does.
- Allowed: short `phasesweep run` smokes on packaged example or fake trainers in scratch directories, and small GPU runs of `examples/tiny_decoder_enwik8` for testing and documentation probes. Check free VRAM first, and follow `LOCAL.md` for how GPU work runs on this machine. Never fall back to `gpu_policy: none` or a relocated lock dir to make a GPU smoke pass, because that changes what is verified.
- Do not silently change trial budgets, step counts, eval cadence or timeouts in sweep configs unless asked.
- `validate` and `run --dry-run` print the launch contract (commands, environment, budgets). Preserve that on changes.
- **Sweep advice, templates, `explain` entries, agent prompts and probe briefs never lead with seeds.** Default to one trial per configuration and spend the budget on more distinct values, such as 7–8 log-spaced learning rates. Noise then shows as scatter around a trend across neighbors. `repeats` is for a few discrete finalists with no ordering to borrow strength from, a one-off noise check, or an explicit request.
- In fixed-data comparisons, don't give faster shapes extra training budget. That conflates two variables.

## Code

- Use the file read/write interface for code changes. Never use `sed`, `awk` or shell text manipulation for them.
- **No `Any` anywhere in typing:** not in signatures, `dict[str, Any]` or `**kwargs: Any`, and not as a variable annotation over a library call that returns `Any`. Narrow with `isinstance` (raise on the impossible branch), use `TypedDict` for known shapes, `Protocol` for external APIs, `phasesweep.runtime.json.JsonValue` / `JsonObject` for open JSON, or `object` narrowed before use. No `typing.cast` and no `# type: ignore`. The typing-escapes hook rejects them on added lines, and editing a line that already carries one means fixing it.
- Signature changes update annotations and docstrings in the same change. Docstrings use one-line Sphinx fields:

  ```python
  def forward(x: Tensor, mask: Tensor | None = None) -> Tensor:
      """Apply multi-head attention with optional causal mask.

      :param Tensor x: Input tensor of shape (B, T, C).
      :param Tensor | None mask: Optional attention mask, defaults to None.
      :raises ValueError: If x.ndim != 3.
      :return Tensor: Output tensor of shape (B, T, C).
      """
  ```

- Comments and docstrings state why the code is the way it is now, and a test docstring states the behavior it pins. No review provenance (`review vN`, `PR #N`, `blocker N`; the hook rejects these) and no fix history ("Pre-fix, the supervisor ran …", "previously returned True"). Git holds provenance. Never copy that style from surrounding code.
- Edit the canonical source. Never create `foo_v2.py`, debug or test-* copies, and delete strays. The module-size hook caps each `src/` and `tests/` module at 2000 lines, so split modules rather than stacking branches onto them.
- Don't delete unused variables or imports without first checking whether they indicate missing functionality.
- GPU code uses bf16 autocast by default, never fp16.

## Writing: Docs, PRs, Messages

- Never hard-wrap prose. In Markdown and docs, each paragraph and list item is one line.
- Update docs in place for every behavior change; never just append. Remove duplicates and link to the detailed doc instead of declaring anything "canonical".
- The README keeps prerequisites, copyable install and verify commands, and a first example inline. Links supplement them and never replace them.
- **Install wording:** write `pip install …`, never `python -m pip install …`. User guides install released extras from PyPI (`pip install "phasesweep[mlflow]"`), and `pip install -e .` appears only where a checkout is genuinely needed. Never name uv or uvx, and never put conda/mamba, local env names or anything from `LOCAL.md` in tracked or user-facing docs. `python -m <module>` stays where it runs a packaged module with no script path, such as `python -m phasesweep.examples.fake_train` or `python -m tests.fixtures.make_ledger_fixtures`.
- **PR descriptions state the final diff against the base.** No changelog wording ("now also", "previously", "updated to"), no volatile figures (test counts, timings, sizes), and no claims about unpushed commits. A rewrite is never longer than the published body. After new commits, edit the description in place where claims drifted; commits must be self-descriptive, so never post per-commit PR comments. The exception is when the user's pasted rules ask for one comment summarizing significant behavior changes.
- **Never tell a reviewer how to review**, whether LLM, human or both. No reading guides or orders, "files you can skip", "decisions that deserve a second opinion", or "how to verify" commands, even if a pasted prompt asks for them. The testing section states what ran and passed, as fact. If something is genuinely uncertain, fix it or file an issue.

## Code Search

Use Semble for code search by behavior, following the `semble-code-search` skill. Load the skill whenever its triggers apply: before grepping for a concept, hunting through a large module, asserting from memory where something happens, or writing a skip-rationalization. `LOCAL.md` says where the skill lives and how to invoke Semble on this machine. In this repo, also run `find-related` on every fix or confirmed finding, to tell one bug from a class of bugs (sibling raise sites, copied rationales, duplicate tests).
