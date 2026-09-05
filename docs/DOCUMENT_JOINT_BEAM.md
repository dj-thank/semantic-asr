# Joint document deliberation beam

## Status

This is a **research-only, opt-in** extension of the existing long-form deliberation path. It does
not alter `transcribe()`, the measured v0.2 runtime profiles, or the immutable first-pass result.

The current long-form second pass can score each window against frozen neighbouring text. That is
useful, but it can still choose mutually inconsistent alternatives independently. The document
beam keeps several acoustically admissible paths for each window and selects one sequence for the
complete recording before applying any text change.

The governing invariant remains:

```text
context preference != acoustic proof
observed transcript != normalized transcript
```

## Pipeline

```text
frozen first-pass LongformResult
           │
           ├─ window 0 exact semantic lattice → local paths A/B/C
           ├─ window 1 exact semantic lattice → local paths D/E
           └─ window 2 exact semantic lattice → local paths F/G/H
                                      │
                                      ▼
                     unique-audio coverage attribution
                     overlap-boundary compatibility factors
                     changed/generated-window budgets
                                      │
                                      ▼
                           bounded document beam
                                      │
                                      ▼
                    complete-document rank-only scorer
                                      │
                                      ▼
                    document margin and safety decision
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  retain first pass       attach immutable
                                           deliberation receipts
```

## Unique-audio coverage

Long-form windows overlap. Summing a full local score for every window would count the shared audio
twice. The planner partitions the recording at all window boundaries. Each elementary time
interval is divided equally among the windows that cover it. These attributed durations are used as
window weights in the document score.

For windows `[0, 1000]` and `[800, 1800]`, the recording contains 1800 unique milliseconds. The
planner attributes 900 ms to each window rather than scoring 2000 ms of evidence.

This is a bookkeeping correction, not a claim that equal attribution is the optimal statistical
model. A learned attribution rule requires preregistration and a locked calibration split.

## Overlap compatibility

Two adjacent windows that cover the same audio should not freely select contradictory boundary
text. Each candidate path is projected onto the actual overlap interval using its local span
ranges. The planner compares the two overlap strings after the repository's lenient surface
normalization.

The important comparison is relative to the retained first-pass boundary:

```text
candidate overlap similarity - retained overlap similarity
```

A pair is rejected when it degrades boundary consistency by more than
`maximum_overlap_similarity_regression`. Otherwise the bounded delta contributes to the document
beam according to overlap duration. A weak or empty textual overlap is not fabricated into strong
evidence; it records `similarity=None` and contributes no preference.

This is still a timestamp-proportional boundary heuristic unless exact posteriors or forced
alignment are available. The receipt records overlap times, compared text digests, similarity,
retained similarity, delta, and compatibility.

## Local alternatives

For every ambiguous window, the implementation reuses:

- `build_semantic_deliberation_lattice()`;
- local finite-factor utilities;
- acoustic-regression guards;
- source-audio-bound phonetic proposals;
- exact source-path reconstruction.

The local decoder runs without a language/context scorer and returns several acoustically
admissible paths. The retained first-pass path is always preserved even when it falls outside the
local top-K. The document search therefore cannot lose the conservative fallback merely because a
local beam was truncated.

## Bounded document search

The search has explicit limits:

- `local_paths_per_window`;
- `beam_size`;
- `global_rescore_paths`;
- optional `maximum_changed_windows`;
- `maximum_generated_windows`.

The base document score combines unique-coverage-weighted local score deltas, overlap consistency,
a change penalty, and a separate generated-path penalty. Paths that exceed the configured whole-
document acoustic regression relative to the fully retained path are removed before linguistic
rescoring.

Generated proposals remain provisional even when the document scorer prefers them. They can only
be applied when the caller explicitly permits provisional changes, and the proposal must already
carry independent phone, audio-derived mora, or discrete-unit evidence bound to the same source
audio.

## Whole-document scorer

The scorer receives one overlap-deduplicated candidate document as a single rank-only path. Its
`DocumentContext` may contain caller-declared external context, but it does not contain a hidden
reference transcript. Every result is bound to:

- candidate document path digest;
- context digest;
- scorer source;
- immutable scorer profile digest;
- first-pass long-form evidence SHA-256.

The scorer output is a bounded linguistic utility, never acoustic proof or a correctness
probability. A score for an unknown path, stale context, or mixed scorer identity fails closed.

## Context control arms

`build_frozen_window_contexts()` provides explicit experimental arms:

- `none`: no recording context and no caller-declared context;
- `declared-only`: only context frozen before evaluation;
- `left-only`: earlier first-pass windows only;
- `bidirectional-offline`: earlier and later frozen first-pass windows;
- `shuffled-context`: deterministic order-control using a recorded seed digest.

The target window is excluded from every context receipt. `left-only` rejects current or future
window indices. Offline arms explicitly record whether future first-pass text was used; they must
not be described as streaming.

The text is untrusted evidence. Applications connecting an instruction-following LLM must use a
frozen prompt format that separates context data from instructions and must include adversarial
context controls. Serialization alone is not a proof of prompt-injection resistance.

## Application and receipts

`plan_document_deliberation()` performs search without changing text.
`apply_document_deliberation()` applies the selected document only when the document decision is
accepted, unless `apply_provisional=True` is explicitly configured.

Applied windows reuse the existing immutable `DeliberatedObservedTranscript` contract. Each window
receipt remains bound to its original first-pass candidate/ranking evidence and selected local arc
path. The top-level diagnostics additionally record:

- document decision and plan digests;
- selected document status and margin;
- scorer identity;
- candidate-document count;
- changed-window count;
- selected overlap-receipt digests.

The original `LongformResult` is never mutated. Non-overlapping identical speech remains a real
repetition and is not removed merely because the surface strings match.

## Required evaluation

Engineering tests do not complete Issue #38. A quality experiment requires an allowed continuous-
speech corpus with suitable verbatim annotation and fixed session/speaker splits. At equal
candidate and compute budgets, compare:

```text
no context
declared context only
left-only first-pass context
bidirectional offline first-pass context
shuffled context
correct external catalog
distractor external catalog
```

Report at least strict/lenient CER, semantic-critical errors, entity accuracy, filler/repair
retention, context-induced false corrections, revision rate, risk-coverage, p95 latency, and peak
memory. Synthetic concatenation is a boundary stress test, not natural-conversation evidence.

## Claim boundary

The implementation establishes a bounded, auditable search surface. It does not include a trained
document scorer, prove that textual overlap approximates exact acoustic alignment, establish
quality improvement, or complete the rights/split/evaluation prerequisites tracked by the release
roadmap. Nothing becomes a default profile until preregistered held-out evidence demonstrates a
useful quality/risk/cost frontier.
