# Execution handoff — 2026-09-06 (JST)

This is a dated coordination note, not live status or a release approval.
Audited baseline: `5eba248ca85c7c87923b2016cd0d2a764f4a28f4`, tree
`4837f1ffac0c0e1aad4a296f0d731dc9e348c5dd`. Re-read live Issues before editing.

## Completed, active and blocked work are different

#19 was closed by #46; #24 by #42. #43 trained small public-data acoustic heads and
LoRA but did not establish an ASR gain or finish #35–#37. #45 added an opt-in document
beam, not the natural-conversation quality evaluation required by #38. The original
506-test result is historical; this baseline passed 548 tests in the audited CPU
environment with optional dependencies, no skips or expected failures.

#25 reserves score/calibration files on `codex/canonical-score-contract-25`.
#26 reserves rights/split/lineage work on `codex/dataset-lineage-contract-26`.
Do not introduce another score contract or data-rights registry while they migrate.
Their comments are reservations, not proof of completed implementation.

#50 targets the old #17 branch, not current main. Its references to draft PRs #18–#21
are incorrect: #18/#20 are earlier merged changes, #19/#21 are Issues. A comment on
#50 requests an accurate reference map, preservation of current PCM/overlap/evidence/
budget/training/input-validation fixes, and tests on the actual integrated main tree.
Do not bulk-close unrelated Issues. This coordination check is not a full code review.

## Published increment: training supervision contracts

`training.py::SemanticASRMultiTask` now validates nonnegative finite loss weights,
integer head sizes and blank IDs, active phone supervision and integer frame lengths.
The same validated weight contract is used at construction and forward time. An
int32 class-label tensor is converted to long for frame cross entropy; float/bool
class labels are rejected. A non-finite total loss cannot be returned as success.

Three reproductions at the baseline were especially important: negative mora weight
produced a negative objective, phone labels with no phone head were ignored, and a
fractional encoder length was accepted on the auxiliary-only training path.
`tests/test_training_supervision_contract.py` has 52 cases (47 failed before the fix,
5 already passed). Existing valid CTC/backward and ignored-label behavior is retained.
No decoder, score schema, saved checkpoint, model architecture or default profile changes.

```bash
python -m pytest -q tests/test_training_supervision_contract.py \
  tests/test_training_optional.py tests/test_mora_training_regressions.py tests/test_weight_pilot.py
python scripts/replay_phonetic_decisions.py
```

Run in the documented CPU training environment. CI explicitly runs the new optional
suite; a base-environment skip is not its execution proof. The training objective is
not automatically normalized: zero weights remain permitted, negative weights do not.
A new adversarial maximization objective must be explicit, not an accidental negative
coefficient in this evidence-preserving acoustic trainer.

## Checkpoint/resume experiment: not published or integrated

A separate local prototype passed 40 resume/rejection tests and a 12-update versus
5+7 fresh-process fixture comparison for CTC/dropout. GitHub publication of its
`checkpoint_codec.py` was blocked by the connector's safety check. No alternate
route was used. That codec and dependent runtime/script/tests are NOT in this source.
Do not use the prototype's 588-test count as a published-source validation result.
The existing public pilot is still not resumable. See the method notes below for
next-agent acceptance requirements rather than commands for nonexistent modules.

## Next tasks: required artifacts, not broad promises

| Existing Issue | Next concrete work | Required acceptance evidence |
|---|---|---|
| #26 | Complete the reserved operation-specific rights and derivation/split contract | Include old 72 + pilot 40 exposure identities; report recording/PCM/text/session intersections; no claim of speaker independence from hashes alone |
| #28 | Lock separately compatible phonetic/training/reranking environments | Clean install, `pip check`, fixed transitive artifacts and actual optional tests; freeze output alone is not a complete lock |
| #29 | Freeze evaluation and label semantics | Strict/lenient CER, gold versus proxy phones/morae, critical semantic errors, false correction and latency; no reuse of inspected data as unseen test |
| #35 | Review and publish a supported checkpoint contract, then connect real data/PEFT loops | Continuous versus resumed parameter/optimizer/scheduler/RNG/sampler histories; rejection without partial mutation; separate last-resumable and best-dev |
| #36 | Compare newly trained heads with the existing trained HuBERT head | Same recording/label conditions; phone-only/mora-only/joint and fixed seeds; independent annotations; checkpoints, frozen-parameter checks and evaluation |
| #37 | Expand informative real candidate pairs and evaluation opportunities | Candidate coverage/oracle ceiling and tie counts; no-LoRA/simple-ranker/LoRA at equal budget; shuffled IDs/order controls; acoustic-retention gate |
| #38 | Evaluate actual ordered conversation | Frozen no-context/left/bidirectional/shuffled and distractor-context arms; session split; semantic false corrections and end-to-end latency |
| #40 | Archive evidence and maintain publication claims | Each statement marked implemented/executed/measured/hypothesis/limitation and linked to fixed source/results; preserve failures and negative outcomes |

Before the next run, pin data/exclusions, model/tokenizer and label revisions, all
seeds, objective, optimizer schedule, candidate pool, test access policy and finite
updates/audio/time/storage limits. Choose quantitative promotion thresholds BEFORE
opening evaluation results. Do not turn a small feasibility result into a quality claim.

A model beating random initialization but losing to its trained baseline is not an
improvement. A LoRA that changes weights without changing selections shows no ASR gain.
Absent correct candidates require candidate-generation work, not merely another ranker.
Spelling-only CER changes must be reported separately from meaning changes.

Research rationale and controls: [training method notes](../research/TRAINING_CONTRACTS_2026-09-06.md).
No new public-speech weights were trained by this increment. No perpetual service or
automatic promotion is introduced. Discussions were not enabled or populated here;
Issues plus versioned records remain the decision trail.
