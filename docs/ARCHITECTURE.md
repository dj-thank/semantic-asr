# Architecture

## 1. Invariant

```text
observedTranscript != normalizedTranscript
```

`observedTranscript` is selected from acoustic candidates and is protected by a canonical SHA-256 evidence digest. `normalizedTranscript` is a separate derivative that references the observed digest.

Rank-only normalization may select only a candidate that already existed in the observed evidence set. A local LLM cannot author observed text.

## 2. Evidence streams

Semantic ASR fuses five calibrated streams.

### Acoustic

Whisper/CTranslate2 sequence evidence, average log probability, beam rank, candidate margin and optional calibrated confidence.

### Mora

Japanese reading, mora CTC, special-mora agreement and boundary evidence. This is the primary defence against a grammatically natural candidate that is not supported by the sound.

### Lexical

Rights-gated support for proper nouns and domain terms. Lexical evidence is never presented as direct acoustic proof.

### Preservation

Evidence that a candidate retains fillers, repetitions, self-corrections and learner errors supported by the waveform.

### Cross-model

Agreement from independent ASR sources. Duplicate text returned by Whisper and Qwen is recorded as source support rather than duplicated as two unrelated sentences.

## 3. Calibration and risk

Raw score domains are not directly comparable. The fusion layer supports persisted calibration profiles and a robust median/MAD fallback. Probability-like scores remain probability-like instead of being distorted by per-set min-max normalization.

The decision layer records:

```text
posterior
candidate entropy
Jensen-Shannon evidence disagreement
evidence coverage
posterior margin
selective risk
accepted / provisional
```

A provisional result is still stored and auditable, but is not falsely presented as a confident final assertion.

## 4. Grammar Honeytrap

The teacher is a language prior, not an acoustic model. If teacher preference exceeds acoustic-family support beyond a deadband, the candidate receives a penalty.

```text
unsupported = max(0, teacher - acoustic_family - deadband)
penalty = strength × unsupported
```

The acoustic family consists of acoustic, mora and cross-model evidence and has a minimum total gate weight.

## 5. Semantic Lattice

Candidates are aligned at mora level when every candidate has a reading/Mora Shadow. Otherwise, the system explicitly falls back to normalized surface characters.

The lattice contains:

```text
Consensus Spine       units shared by candidates
Contradiction Islands localized disagreements
```

Islands are classified for semantic impact:

```text
negation meaning flip
number / quantity
date / time
currency / percentage
modality / intent
entity / domain term
Latin acronym / technical term
particle / functional word
disfluency / repair
special mora
phonetic / punctuation
```

Exact timing is used when a mora/character timeline exists. Otherwise, the runtime may use proportional timing and records `timing_source=proportional`.

## 6. Evidence acquisition

The scheduler computes:

```text
utility = expected_information_gain / estimated_cost
```

Available actions:

```text
whisper-relisten
qwen-second-ear
forced-align
lexicon-lookup
local-teacher
```

At most two actions are selected per island by default so that one difficult phrase cannot consume the entire recording budget.

## 7. Long-form runtime

- default window: 28 seconds
- default overlap: 1.2 seconds
- exact textual overlap removal
- no artificial spaces inside Japanese text
- independent immutable evidence per window
- global digest over ordered window evidence
- source path excluded from exported results

A one-window N-best adapter never concatenates hypotheses from independent windows into a false global N-best list.

## 8. Cache

The SQLite cache key binds:

```text
audio SHA-256
exact start/end time
namespace/action
adapter and model
language
beam and hypothesis count
prompt digest
hotword digest
context digest
calibration digest
schema version
```

The cache contains evidence JSON, not waveform data. Unknown schema versions and payload hash mismatches fail closed. Teacher abstention remains abstention after replay.

## 9. Qwen integration

Qwen3-ASR is an independent second ear. Its high-level wrapper normally returns one transcript per input; Semantic ASR does not mislabel this as decoder N-best.

Qwen3 Forced Aligner supplies localization evidence. It does not independently prove that every supplied token was spoken.

Qwen3.8-Flash-Next is an optional delayed local candidate teacher. It returns only probabilities over existing IDs and may abstain.

## 10. Training heads

A shared speech encoder can feed:

```text
mora CTC
phone CTC
boundary classification
accent classification
F0 regression
preservation classification
```

The module validates CTC blank usage, right padding, encoder lengths, frame labels and F0 masks. It can coexist with a normal Whisper text decoder rather than replacing it at the start of research.
