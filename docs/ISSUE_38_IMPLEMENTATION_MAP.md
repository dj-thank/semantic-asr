# Issue 38 — context and phonetic implementation map

## Objective

Resolve the failure mode where a first-pass ASR candidate is locally close to the sound but is not
a coherent utterance in the full conversational context, without allowing language fluency to
overwrite what was actually spoken.

```text
context preference != acoustic proof
candidate-derived mora_shadow != audio-derived mora posterior
observed transcript != normalized transcript
```

## Layer 1 — document context experiment

A preregistered reference-separated harness compares ordered, shuffled, and acoustic-only document
context using one frozen candidate set. Promotion requires held-out improvement, shuffled-context
control, critical-error safety, false-correction safety, coverage, and latency constraints.

## Layer 2 — source-audio phone/mora runtime

A compact shared acoustic encoder emits independent phone and mora CTC posteriors. The runtime uses
bounded PCM16 WAV crops, source-audio SHA-256 binding, separate frozen inventories, strict score
semantics, pickle-free tensor artifacts, held-out utility normalization, and ambiguity-only local
proposal acquisition.

## Layer 3 — four-way evidence separation

```text
train       gradient updates
validation  checkpoint/hyperparameter selection
calibration candidate-utility normalization
test        final PER/MER and end-to-end evaluation
```

Repeated-label CTC targets are checked against the true minimum frame requirement before loss
computation. Impossible alignments are rejected rather than hidden by `zero_infinity`.

## Layer 4 — deterministic Japanese labels

Explicit hiragana/katakana readings are converted into frozen mora and phone labels. Kanji readings
are never guessed. Long vowels, moraic nasal, gemination, palatalized mora, and supported foreign
katakana have explicit profile-bound symbols. Rights-approved source rows materialize into
source-audio-bound four-way manifests without storing raw transcript text.

## Layer 5 — phonetic proposal ablation

Phone, mora, phone+mora, first-pass, and later discrete-unit arms reuse one reference-free frozen
candidate pool. Evaluation measures candidate oracle, recovery outside first-pass N-best,
conditional false corrections, introduced/corrected error characters, critical cases, abstention,
latency, and grouped paired confidence intervals. Promotion thresholds are preregistered.

## Layer 6 — long-form deliberation integration

Verified local proposals enter the existing exact multi-level lattice. Complete paths are scored
with frozen left/right document context, then subjected to per-span and whole-path acoustic
retention guards. Applied changes create new immutable receipts linked to the original first-pass
evidence. Old confidence and timestamps are invalidated until re-alignment.

## Default boundary

All components remain opt-in research paths. No measured v0.2 profile is changed. Software tests,
synthetic recovery cases, and artifact round trips are not Japanese accuracy claims. Default
promotion requires locked real-audio evaluation over strict/lenient CER, semantic-critical errors,
outside-N-best recovery, context-induced false corrections, risk–coverage, latency, and memory.
