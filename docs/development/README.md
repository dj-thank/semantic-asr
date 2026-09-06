# Execution index — release readiness

This is an implementation roadmap, not a completed release or a performance claim.
Start at [Issue #23](https://github.com/dj-thank/semantic-asr/issues/23).
Read [AGENTS](../../AGENTS.md), the [audit](AUDIT_2026-09-05.md),
[research methods](../research/METHODS_2026-09-05.md) and [templates](TEMPLATES.md).

## Work queue

| Scope | Issues | Required outcome |
|---|---|---|
| Correctness | #19, #24, #25 | Strict decode input, budget enforcement, one calibrated-score contract |
| Reproducibility | #26, #28, #29, #30 | Rights and data lineage, environment locks, fixed evaluation, complete research driver |
| Refactoring | #27, #31, #32 | Reference-based cleanup, evidence migration, small adapters and CLI facade |
| Phonetics | #33, #34 | Multiple readings, audio-bound phonetic runtime with real-audio integration |
| Real training | #35, #36, #37 | Resume/reload runner, new acoustic heads and separately trained LLM LoRA weights |
| Research and publication | #38, #39, #40 | Actual document context, equal-budget performance, reproducible release evidence |

Each number is in the [machine-readable plan](release-plan.json), with prerequisites,
current source entry points, new tests to create and responsible skill/role.
Full instructions and acceptance criteria live in the corresponding GitHub Issue.
This is a dated plan, not live status. As checked on 2026-09-06, #19 was closed by
#46, #24 by #42 and #25 by #47; #21 remains an unfinished research Issue, not a draft PR.
Read the [training handoff](EXECUTION_2026-09-06.md) and the newer
[paired-evaluation methods and handoff](../research/PAIRED_EVALUATION_2026-09-06.md)
before choosing files. #52 is the focused evaluation-integrity repair under #29;
#53 is a proposed audio-grounding experiment, not an executed study. These additions
are linked here rather than renumbering the historical machine-readable plan.

## Execution order

Begin remaining #26 and #28 only after checking their current reservations.
Do not duplicate an active branch or reopen completed #19/#24/#25 without a regression.
The inventory phase of #27 and documentation phase of #40 can also start immediately.
Reuse the #25 canonical contract for score/evidence migrations. Build #29 on #26; #35 then supplies
training infrastructure. #36 and #37 are independent acoustic and LLM experiments,
not one vague train-everything task. Full runtime validation needs #34. #30 closes
the reproducible collection/fitting/evaluation loop. Use #38 for real long recordings;
isolated dataset rows do not validate conversation context.

`depends_on` in the JSON means prerequisites for the task's complete acceptance,
not a ban on preparatory work. `promotion_requires` is a separate release gate.
The JSON records issue links, not GitHub-native dependency objects or live status.
When issues or paths change, update this plan with the same PR.

## Roles and completion states

The implementer owns one narrow change. A different reviewer should check contracts
and evidence. A data steward checks rights/split lineage; an evaluation reviewer
checks preregistration and leakage; training roles must deliver actual weights.
Roles are responsibilities, not claims that those agents have already run.

Engineering completion proves implementation and tests. Experiment completion
proves a bounded, reproducible run even when the result is negative. Promotion
requires independent, previously fixed quality/cost/rights gates. An experiment
must not be tuned repeatedly on an inspected test set until it looks successful.

## Research records and Discussions

Use Issues for executable work, versioned method notes for hypotheses and sources,
ADRs for decisions, and experiment records for all results. Proposed Discussion
categories are Research questions, Experiment results, Design RFC and Q&A.
Discussions were not enabled or populated by this change. A future discussion must
link its final decision back to the versioned ADR and implementation Issue.

## What the original planning change did not claim

The original planning PR alone did not repair runtime defects, remove legacy modules,
train weights, publish an integrated phonetic model, or establish world-leading accuracy.
Subsequent #42/#43/#44 have separate implementation and limited training evidence.
Those outcomes need the evidence specified in the work queue. Historical research
remains in its original dated documents; do not overwrite its conclusions.

## Codex execution entry

[Codex handoff and bounded pipeline](CODEX_AUTOPILOT.md) connects README intentions to
acceptance criteria, executable model-free verification, and the next #30 driver task.
Run `python scripts/codex_verify.py --plan` from the repository root. This does not
complete the real-audio research loop or change the reserved issue ownership above.
