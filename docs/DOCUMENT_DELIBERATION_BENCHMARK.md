# Document deliberation benchmark and promotion protocol

## Why CER is not enough

A context model can lower aggregate CER while damaging text that the first-pass ASR already got
right. It can also improve punctuation or remove overlap duplication while increasing errors in
numbers, negation, names, or disfluencies. Document-level promotion therefore requires paired
per-recording evaluation with separate correction and regression arms.

## Input rows

`scripts/benchmark_document_deliberation.py` reads JSONL rows with:

```json
{
  "caseId": "recording-0001",
  "reference": "...",
  "firstPass": "...",
  "final": "...",
  "finalStatus": "accepted",
  "firstPassSegments": ["...", "..."],
  "finalSegments": ["...", "..."],
  "criticalTokens": ["3000円", "ない", "森脇"],
  "changedWindowCount": 1,
  "metadata": {
    "speakerGroup": "...",
    "microphone": "..."
  }
}
```

References and critical-token annotations must be created under the declared evaluation protocol.
Context supplied to the recognizer must be exogenous and frozen before listening to the evaluation
audio. Reference text, corrected hypotheses, and post-hoc entity lists must never be supplied as
runtime context.

## Reported metrics

The report separates:

- first-pass and final corpus CER;
- paired corpus CER delta and bootstrap interval;
- mean per-case CER delta;
- improved, regressed, and changed case rates;
- accepted coverage and accepted error rate;
- false-correction rate among first-pass-exact cases;
- critical-token error before and after deliberation;
- adjacent-window overlap duplicate counts;
- mean changed windows per recording.

The first-pass-exact arm is mandatory. Without it, the benchmark cannot estimate how often context
corrupts already-correct transcripts.

## Paired bootstrap

Bootstrap resampling is performed over complete recordings, never isolated windows or characters.
This preserves within-recording dependence. The reported interval is for:

```text
final corpus CER - first-pass corpus CER
```

A negative value favors document deliberation. The default promotion gate requires the upper bound
to remain below zero. Larger evaluation programs should additionally use speaker/source-group
bootstrap or a hierarchical bootstrap when recordings are clustered.

## Critical tokens

`criticalTokens` is deliberately explicit rather than guessed by the benchmark. It should cover at
least:

- numbers and counters;
- dates and times;
- currency and percentages;
- negation and modality;
- names and domain entities;
- repairs and preserved disfluencies when verbatim fidelity is required.

The simple count-based token check in this baseline is transparent and deterministic. Production
evaluation should add span-aware annotation so repeated tokens and inflectional variants are scored
correctly.

## Overlap duplicates

The baseline reports adjacent segment boundaries where a suffix of at least four characters is also
a prefix of the next segment. This detects obvious repeated overlap output. It is not sufficient to
prove safe stitching: a system can remove duplication by deleting real content. Manual overlap
review and timestamp-aware reference spans remain required.

## Default promotion gate

The conservative default requires:

- at least 100 recordings;
- at least 10,000 reference characters;
- accepted coverage of at least 20%;
- false-correction rate on first-pass-exact cases no greater than 0.5%;
- regressed-case rate no greater than 10%;
- no critical-token error regression;
- paired CER-delta upper interval below zero;
- no overlap-duplication regression.

These are software defaults, not universal scientific thresholds. A deployment can declare stricter
limits, but it must not loosen them after viewing the locked test result.

## Required slices

A promotion decision should be repeated for:

- native and non-native Japanese;
- dialect and standard Japanese;
- quiet, music, reverberation, and competing speech;
- headset, laptop, phone, and far-field microphones;
- short utterances, meetings, lectures, and spontaneous conversation;
- critical-token categories;
- first-pass confidence bands;
- generated-proposal versus N-best-only decisions;
- exact versus normalized overlap resolution.

A global pass can hide a serious subgroup regression. Overall success does not override a failed
safety-critical slice.

## Exit behavior

The CLI writes the complete report, gate, and their digests. It exits with:

```text
0  promotion gate passed
2  promotion gate failed
```

A failed gate is a valid experimental result. CI or research orchestration should archive the
report rather than rewriting thresholds or discarding the run.
