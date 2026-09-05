# Opt-in long-form semantic deliberation

## Status

This is a **research-only second pass**. It does not change `semantic_asr.transcribe()` or the
measured v0.2 default profiles. A caller must explicitly wrap a warm
`SemanticASRTranscriber` with `with_global_deliberation()` or call
`apply_longform_deliberation()` on an existing `LongformResult`.

The implementation preserves the core invariant:

```text
context preference != acoustic proof
observed transcript != normalized transcript
```

A language model may rank complete paths. It may not silently author the observed transcript.

## Why a second pass is needed

Whole-sentence N-best ranking has two structural limits:

1. the best globally coherent sentence may combine supported pieces from different ASR paths;
2. the correct local word may be absent from every complete first-pass hypothesis.

Long-form decoding adds another problem: one 28-second window can be locally plausible while being
inconsistent with the preceding or following discussion. The second pass therefore runs only after
all first-pass windows are frozen, so an offline scorer can inspect both left and right document
context without feeding its own edits back into acoustic decoding.

## Runtime flow

```text
measured first-pass long-form transcription
                 │
                 ├─ immutable window candidates and posteriors
                 ├─ immutable first-pass observed evidence hashes
                 └─ frozen neighboring-window context
                                  │
                                  ▼
                exact local confusion network
                 ├─ union of candidate edit boundaries
                 ├─ explicit epsilon/deletion arcs
                 ├─ exact source-path reconstruction
                 └─ finite local evidence-factor budget
                                  │
                                  ▼
                bounded local path beam
                                  │
                                  ▼
        full-path scorer with left + right context
                                  │
                                  ▼
                acoustic-retention guards
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
            keep first pass                 attach new immutable
                                             deliberation receipt
```

## Exact candidate projection

`build_semantic_deliberation_lattice()` aligns every candidate against the selected first-pass
candidate and divides the pivot at the union of all edit boundaries. Each candidate is projected
through every resulting span exactly once.

For every source hypothesis, the lattice stores a `SourcePath`. Concatenating the selected arc text
for that source path must reproduce the original candidate byte-for-byte and its SHA-256 must
match. Insertions are assigned to one deterministic forward span. Deletions become explicit
`is_epsilon=True` arcs rather than invalid empty strings.

If a projection cannot reconstruct a source candidate exactly, construction fails. The long-form
wrapper then either raises or retains the original first-pass window according to
`fail_closed_to_first_pass`.

## Finite evidence factors

A complete-hypothesis acoustic or fusion score cannot be copied onto every changed character. That
would turn one item of evidence into an arbitrarily large score merely because a candidate was
split into more spans.

Each active contradiction span therefore receives a deterministic `factor_weight`. The weights
sum to one across the local uncertainty structure. Path scoring uses:

```text
channel weight × bounded utility × factor weight
```

Adjacent transition factors likewise have one total budget across the window boundaries. This is a
factorization contract, not a claim that the current hand-designed factor allocation is optimal.
A learned allocation requires a locked calibration split and an ablation against the uniform and
criticality-weighted baselines.

## Mora evidence semantics

Two different quantities must not share one name:

- `mora`: a candidate-independent audio-to-mora posterior head;
- `mora_shadow`: a reading or mora feature derived from a candidate string.

Only `phone`, audio-derived `mora`, and `discrete_unit` count as independent acoustic evidence for
a generated local proposal. `mora_shadow`, language-model preference, lexical fit, first-pass
posterior, and cross-model agreement may help rank existing candidates, but cannot by themselves
authenticate text outside the observed candidate set.

## Phonetic and generated proposals

A proposal provider receives the already-built local lattice and can restrict expensive analysis to
active contradiction spans. It may use:

- phone CTC posterior evidence;
- audio-to-mora CTC posterior evidence;
- same-codebook discrete-unit centroid DTW;
- a later guarded local generator followed by one of the acoustic verifiers above.

Every observed-eligible `VerifiedSpanProposal` is bound to the full source-audio SHA-256. A proposal
from another recording, a candidate-derived mora feature mislabeled as audio evidence, or a
context-only suggestion fails closed.

## Complete-path context scoring

`GlobalSequenceScorer` receives one complete local path plus a frozen `DocumentContext`. The
long-form wrapper builds that context from:

- preceding first-pass windows;
- following first-pass windows;
- caller-declared topic and entity identifiers;
- the first-pass long-form evidence hash.

The scorer output is bound to the exact path digest, context digest, scorer source, and immutable
scorer-profile digest. `SequenceScorerGlobalAdapter` can adapt the existing teacher-forced causal
sequence scorer. It batches all retained paths in one invocation and requires an immutable model
revision, artifact/config digest, or an explicitly supplied scorer identity digest.

The adapter applies a held-out affine-plus-`tanh` normalization. Its output is a bounded ranking
preference, **not a correctness probability**.

## Application policy

The second pass applies a changed path only when all of the following hold:

1. the path survives per-span acoustic regression limits;
2. the complete path survives the mean acoustic regression limit;
3. generated arcs contain independent source-audio-bound acoustic evidence;
4. the global decision is `accepted`, unless the caller explicitly enables `apply_provisional`.

The first-pass object is never mutated. An applied edit creates a new
`DeliberatedObservedTranscript` bound to:

- source-audio SHA-256;
- original first-pass window evidence SHA-256;
- lattice-build digest;
- global-decision digest;
- exact selected-path digest;
- ordered arc receipts and their time ranges.

A changed window invalidates the old fusion posterior and old candidate timestamp spans. The public
facade therefore reports `confidence=None` and falls back to the verified window range until a new
forced alignment is attached.

## Minimal use

```python
from semantic_asr import (
    CallableGlobalSequenceScorer,
    DocumentContext,
    frozen_profile_digest,
    load_transcriber,
    with_global_deliberation,
)

warm = load_transcriber("cpu-ja-v1")

# Deterministic demonstration only. Replace this with a frozen, evaluated scorer.
scorer = CallableGlobalSequenceScorer(
    lambda path, context: 0.0,
    source="example-complete-path-scorer",
    profile_digest=frozen_profile_digest(
        "example-complete-path-scorer",
        "r1",
        {"implementation": "replace-before-evaluation"},
    ),
)

transcriber = with_global_deliberation(
    warm,
    sequence_scorer=scorer,
    declared_context=DocumentContext(topic_summary="Semantic ASR design review"),
)
result = transcriber.transcribe("meeting.wav")
print(result.observed_text)
print(result.diagnostics["globalDeliberation"])
```

A teacher-forced local causal model can be connected through
`SequenceScorerGlobalAdapter`, `GlobalScoreNormalization`, and
`TransformersCausalSequenceScorer`. The normalization profile must be fitted on a speaker-disjoint
held-out Japanese calibration split; example constants must not be promoted to runtime defaults.

## Output compatibility

`DeliberatedLongformResult` exposes the same primary fields used by the existing facade and output
writer:

```text
source_name
source_audio_sha256
duration_ms
observed_text
normalized_text
segments
evidence_sha256
diagnostics
as_dict()
```

JSON additionally contains the first-pass and deliberation evidence hashes. TXT, Markdown, SRT and
VTT use the final observed text. Changed windows use the full verified window range unless a future
post-deliberation aligner supplies fresh sub-window timestamps.

## Failure policy

`fail_closed_to_first_pass=True` is the default. Projection, scorer, proposal, provenance, or
resource failures retain the original window and record the failure class in the deliberation
trace. Set it to `False` only for research runs that should stop at the first invariant violation.

## Current boundary

This PR implements a bidirectional-context second pass over each window after the complete
first-pass document is frozen. It does not yet claim globally joint optimization of all window
choices, trained Japanese phone/mora heads, a trained context model, or measured CER improvement.
Those require:

- a hierarchical document-level beam over window-path alternatives;
- overlap-aware document path rendering;
- source-audio-bound phone/mora/discrete-unit adapters;
- locked train/calibration/test manifests;
- paired CER, semantic-critical error, false-correction, risk–coverage, latency, and memory results.

Nothing here becomes a default profile until those comparisons show a useful frontier over the
measured v0.2 first pass.
