# Joint document lattice deliberation

## Status

This is an explicit research path stacked on the window-local v0.3 deliberation work. It does not
change `transcribe()`, `cpu-ja-v1`, or any other measured v0.2 default.

The previous long-form second pass freezes the full first-pass recording but chooses one path for
each window independently. This module retains several policy-eligible paths per window and chooses
their combination in one bounded document beam.

```text
window 0 local paths ─┐
window 1 local paths ─┼─ overlap-aware document beam ─ complete-document scorer
window 2 local paths ─┘                         │
                                                ├─ document acoustic guard
                                                ├─ changed-window budget
                                                ├─ generated-path guard
                                                └─ provisional/accepted decision
```

The invariant remains:

```text
context preference != acoustic proof
candidate-derived mora != audio-derived mora
observed transcript != normalized transcript
```

## Why window-local decisions are insufficient

Suppose two overlapping windows contain:

```text
window 0 A: 計画はまた保留です。
window 0 B: 計画はまだ保留です。

window 1 A: 保留です。承認後に実行します。
window 1 B: 保留です。承認後に中止します。
```

Choosing each window independently can select a locally plausible but globally inconsistent pair.
It can also output `保留です。` twice. Joint decoding evaluates the pair, its overlap receipt, and
the complete emitted document at once.

## Per-window option construction

Each first-pass window is converted through the exact semantic lattice builder introduced in the
preceding PR:

- every first-pass hypothesis has one exact `SourcePath`;
- insertions, substitutions, and explicit epsilon/deletion arcs are retained;
- projected evidence uses a finite factor budget;
- candidate-derived reading evidence is `mora_shadow`;
- generated arcs require source-audio-bound `phone`, audio-derived `mora`, or `discrete_unit`
  evidence.

`decode_global_lattice(..., sequence_scorer=None)` applies local acoustic-retention guards and
returns bounded base alternatives. The document layer keeps at most `local_paths_per_window`, while
always retaining the immutable first-pass path.

## Hierarchical document beam

The Cartesian product of all window options grows exponentially. The implementation therefore uses
a deterministic bounded beam:

```text
state score = sum(local path factors)
            + overlap_weight × sum(overlap utilities)
```

After each window, only `document_beam_size` states are retained. Ties are broken by immutable
option digests. A hard changed-window budget is applied during expansion:

```text
changed windows <= min(maximum_changed_windows,
                       floor(maximum_changed_ratio × window_count))
```

This is a search bound, not a quality claim. Beam size and local K must be evaluated against oracle
coverage and target-device latency.

## Overlap receipts

Every emitted window has an `OverlapReceipt`. It records:

- left and right window indexes;
- overlap duration;
- exact or normalized suffix/prefix method;
- number of right-prefix characters removed;
- matched lengths and similarity;
- bounded overlap utility;
- input and emitted text hashes;
- immutable overlap-policy digest.

The resolver first attempts a sufficiently long exact suffix/prefix match. It then tries an NFKC,
case-folded, punctuation-and-space-insensitive match with an index map back to the original right
text. Short common endings are not removed. If two overlapping windows are similar but disagree in
the overlap, the resolver records `ambiguous-conflict` and removes nothing.

An ambiguous overlap can make the whole document provisional. This prevents aggressive text
stitching from silently deleting content.

## What the complete-document scorer reads

The scorer must read the **overlap-resolved emitted document**, not a concatenation of complete
window texts. Each document candidate is therefore represented by synthetic, evidence-bound
emission arcs whose metadata links back to:

- local window option digest;
- overlap receipt digest;
- underlying local path arc digests;
- source-audio SHA-256.

The existing `GlobalSequenceScorer` contract can be reused. The score is bound to the exact emitted
path digest, external `DocumentContext`, scorer source, and immutable scorer-profile digest. A stale,
missing, duplicated, or mixed-identity score fails closed.

## Document-level safety guards

The retained document is the combination of every window's original selected first-pass path.
Candidate documents are removed when their duration-weighted mean acoustic support falls more than
`maximum_document_audio_regression` below that retained document.

The final decision becomes provisional when any configured condition holds:

- document margin is too small;
- a generated local path is selected;
- an overlap conflict is ambiguous.

Provisional output is not applied unless `apply_provisional=True` is explicitly selected. When a
candidate is not applied, the result uses the retained document path but still preserves the
attempted decision receipt.

## Emission-level observed evidence

Overlap removal changes what one window emits even when its complete local path remains unchanged.
The output object therefore distinguishes:

```text
full_window_text
trim_prefix_characters
emitted text
```

`DocumentObservedTranscript` is bound to:

- full source-audio SHA-256;
- first-pass window evidence SHA-256;
- document decision digest;
- local option digest;
- overlap receipt digest;
- every selected local arc and its time range;
- complete first-pass candidates and ranking evidence.

Its path receipts reconstruct the full window text, and the overlap receipt reconstructs the
emitted suffix. The final document text is reconstructed from emitted segments. JSON, TXT, Markdown,
SRT, and VTT therefore do not repeat a verified overlap.

The old first-pass confidence and candidate timestamp spans are invalid after document selection.
Changed or trimmed output uses no stale confidence and falls back to the window time range until a
post-document forced aligner attaches fresh sub-window timing.

## Failure behavior

The production-facing default is fail closed. The final implementation records a
source-audio-bound failure receipt and returns an unchanged, output-compatible result rather than
silently pretending that no second pass was attempted. Research runs can disable fail-closed
behavior to stop on the first violated invariant.

## Current boundary

The document beam jointly selects window paths and overlap emissions. It is not an unbounded full
attention model over raw hour-long acoustic frames. The intended hierarchy is:

```text
raw frames only for uncertain local spans
local path alternatives for each window
complete emitted text for the document scorer
compact topic/entity state for external context
```

This keeps expensive audio attention selective while allowing whole-document language coherence.

## Required evaluation before promotion

At minimum:

- strict and lenient corpus CER;
- document-level paired bootstrap interval;
- context-induced false correction on first-pass-exact cases;
- number/date/currency/negation/entity regression;
- accepted coverage and accepted error rate;
- overlap duplicate removal and overlap deletion errors;
- local oracle@K and document oracle@beam;
- generated-proposal recovery outside N-best;
- latency, peak memory, and energy for each effort tier.

No default may change unless the locked test split passes the document promotion gate and a separate
speaker/source-shift evaluation shows no unacceptable regression.
