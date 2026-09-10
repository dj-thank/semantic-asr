---
name: semantic-asr-delivery
description: Implement Semantic ASR README intent, reproduce CI, and continue its bounded Japanese audio research pipeline. Use for semantic-asr Codex handoffs, issue implementation, pipeline repair, or evidence review; not for generic transcription or automatic model promotion.
---

# Semantic ASR delivery

Read `AGENTS.md`, `README.md`, `docs/development/CODEX_AUTOPILOT.md` and the live assigned
Issue plus dependency comments before choosing files. Recheck the actual base SHA/tree
and PR state; dated notes are not live status. This skill supplies a workflow, not
permission to bypass environment, publication, network or data-rights restrictions.

Use `bash scripts/codex_setup.sh` only in an authorized package-installation setup
phase. Inspect `python scripts/codex_verify.py --plan`, then run verification into a
fresh directory outside the checkout. Use `core` only without PyTorch, `cpu` for the
explicit optional suite, or `installed` for an honestly labelled development run.
Do not silently change profiles after a failure to obtain a green result.

Work on one bounded, unreserved issue at a time. First reproduce defects, then
implement and test; do not stop at another roadmap when execution is possible.
Reuse existing contracts, CLI, experiment/checkpoint utilities and research records.
Use subagents for independent reading/review only unless disjoint write ownership is
explicit. Do not duplicate #26/#28/#35 work or silently integrate PR #50.

For real research, follow the separate staged contract in the handoff. Fit on train/dev,
calibrate separately, freeze before unseen evaluation, keep all failed trials, preserve
observed speech, and enforce predeclared finite budgets. Stored-decision replay and
synthetic training do not establish real-audio improvement. No auto-merge, paid resource
provisioning, raw-data upload, default-profile promotion or perpetual execution.

Return exact source/evidence identities, commands and failures/skips, changed behavior,
remaining blockers and one executable next task. Report engineering, experiment and
promotion states separately. When blocked, complete independent authorized work and
report the blocker; do not invent successful runs or bypass a rejected action.
