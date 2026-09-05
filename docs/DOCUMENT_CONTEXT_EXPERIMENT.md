# Document-context experiment protocol

## Purpose

The document joint beam can select a globally coherent transcript path, but fluency is not evidence
that the selected words were spoken. This protocol is the promotion gate for Issue #38. It asks a
narrow question:

> On the exact same frozen document-candidate set, does ordered document context reduce Japanese
> transcription error without increasing context-induced false corrections?

The experiment is intentionally separated into two phases:

```text
phase 1: candidate planning                         phase 2: reference evaluation

first-pass evidence                                frozen candidate set
+ frozen non-reference context                     + arm scores and selection
        │                                                   │
        ▼                                                   ▼
reference-free document planner                    open rights-approved reference
        │                                                   │
        └──── exact candidate-set digest ───────────────────┘
```

`PlanningCaseView` does not contain the reference object. The scorer receives only frozen candidate
paths and explicitly registered external context. The reference is opened only after an arm has
selected a path.

## Required arms

A useful preregistration should include at least:

| Arm | Candidate view | Direction | Purpose |
|---|---|---|---|
| `acoustic-only` | frozen document paths | none | first-pass/document-beam baseline |
| `ordered-forward` | ordered document | forward | causal left-context language evidence |
| `ordered-bidirectional` | ordered document | bidirectional | offline full-document language evidence |
| `shuffled-bidirectional` | deterministically shuffled windows | bidirectional | checks whether gains come from real discourse order |

An optional external-context arm may name a `FrozenExternalContext`, but that object must have an
independent provenance digest and must be frozen before evaluation. A complete reference transcript
embedded in external context is rejected.

Every arm sees the same `FrozenDocumentCandidates`. Candidate generation is run once per case, and
the candidate-set digest is checked across all arms. Context arms are therefore not allowed to
silently request extra Whisper samples, a larger document beam, or additional phonetic proposals.
A separate candidate-generation experiment is needed for those questions.

## Rights and split controls

Every reference-bearing case requires:

- `rights_decision="allow"`;
- a non-empty license identifier;
- immutable dataset and split-manifest identifiers;
- source, speaker, and session identifiers;
- source-audio SHA-256 agreement between the first-pass result and the reference.

The manifest rejects:

- duplicate test audio;
- test speakers present in training or calibration;
- test sessions present in training or calibration;
- training/calibration speaker or session overlap;
- cases bound to a different split-manifest digest.

Public availability alone is not treated as permission to evaluate, export references, or publish
reference-bearing artifacts.

## Candidate freeze

`prepare_document_experiment()` calls a caller-supplied planner with `PlanningCaseView`. The planner
should use `plan_document_deliberation()` with global linguistic scoring disabled, then return its
`DocumentDeliberationPlan`. The helper freezes:

- planner-output digest;
- first-pass evidence digest;
- retained document path;
- bounded set of alternative document paths;
- every option digest and local base score;
- planning latency.

The retained path is always kept when the preregistered candidate limit truncates alternatives.

Example planner shape:

```python
from dataclasses import replace

from semantic_asr.document_deliberation import plan_document_deliberation


def planner(view):
    fair_config = replace(
        document_config,
        require_sequence_scorer=False,
        global_context_weight=0.0,
        proposal_context_arm="none",
    )
    return plan_document_deliberation(
        view.first_pass,
        config=fair_config,
        build_config=local_build_config,
        local_policy=local_policy,
        sequence_scorer=None,
        proposal_provider=frozen_proposal_provider,
        declared_context=view.context(None),
    )
```

The exact planner arguments should remain recorded in the plan digest. A proposal provider used in
this phase must itself be frozen and identical across experiment arms.

## Dependency-free linguistic baseline

`BidirectionalCharacterNgramScorer` supplies a reproducible baseline that does not interpret text as
instructions and cannot generate a transcript. It consists of:

- a frozen forward character n-gram model;
- a separately frozen model trained on reversed text;
- add-alpha smoothing;
- a held-out affine + `tanh` normalization profile;
- training and calibration manifest SHA-256 values;
- canonical count rows and deterministic model digests.

For an ordered forward arm, it measures candidate likelihood with optional declared left context.
For an ordered bidirectional arm, it averages:

```text
forward(candidate | left context)
backward(reverse(candidate) | reverse(right context))
```

The value is a bounded rank utility, not a probability of correctness. A shuffled arm uses the same
models, normalization, candidate paths, and character budget, but deterministically permutes window
order from the arm seed and case ID.

This baseline is deliberately modest. A ModernBERT, causal LLM, or audio-text deliberation model may
replace it only through the same immutable scorer interface and the same frozen-candidate protocol.

## Scored-character budget

Linguistic arms receive a fixed total character budget per case. The runner divides it across the
same number of candidate documents and, for bidirectional scoring, across both directions. If a
scorer reports more characters than the preregistered limit, the arm fails closed.

The report records:

- scorer calls;
- scored characters;
- arm latency;
- peak Python heap measured by `tracemalloc`;
- planning latency shared by every arm.

Python heap is not GPU memory, native model memory, or process RSS. Those must be measured separately
for a heavyweight scorer.

## Metrics

The primary metrics are:

- strict corpus CER;
- punctuation/symbol/whitespace-insensitive corpus CER;
- semantic-critical token errors;
- context-induced false-correction windows;
- corrected and introduced error characters;
- window revision rate;
- accepted coverage and CER on accepted cases;
- paired mean-case CER delta with deterministic bootstrap interval;
- latency, scored characters, scorer calls, and peak Python heap.

`CriticalReferenceToken` supports explicit numbers, dates, times, currency, percentages, negation,
modality, entities, repairs, fillers, and task-specific critical items. Explicit annotation is
preferred over heuristic extraction.

A context arm is not promoted merely because average CER improves. At minimum, its report must be
examined for:

```text
false-correction windows
critical-token regressions
introduced error characters
coverage collapse
shuffled-control parity
latency and memory cost
```

A gain shared by the ordered and shuffled arms is evidence against a discourse-order explanation.

## Reference-window text

`FrozenReference.window_texts` contains the reference corresponding to each first-pass window. These
rows may overlap because runtime windows overlap; they are not required to concatenate into the
non-overlapping document reference. The full document reference is scored against the overlap-aware
selected `DocumentPathHypothesis.text`, while window references are used only for revision and
false-correction attribution.

## Report

`DocumentContextExperimentReport` is canonical, digest-bound, and atomically writable. It includes:

- protocol and manifest digests;
- one result per case and arm;
- candidate-set digest for equality checks;
- selected and retained path digests;
- per-path base and language scores;
- all primary metrics;
- aggregate arm results;
- paired bootstrap intervals;
- failures when `fail_on_case_error=False`.

Negative results remain in the same report; the writer has no success-only path.

## Promotion boundary

The harness and n-gram baseline are software and experiment infrastructure. They do not establish a
quality improvement. Promotion of document context requires a locked, rights-approved,
speaker/session-disjoint continuous-speech test set and a frozen scorer artifact. Results must be
reported for every preregistered arm, including shuffled controls and negative findings.
