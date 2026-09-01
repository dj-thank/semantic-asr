# Semantic ASR v0.2 frontier architecture

## 1. Non-negotiable invariant

```text
observedTranscript != normalizedTranscript
```

`observedTranscript` is an auditable decision over acoustically grounded candidates. A language
model, reranker, normalizer, summarizer, or generative error corrector cannot directly author it.
`normalizedTranscript` is a separately hashed derivative.

A generated sentence can enter the observed-candidate pool only after calibrated acoustic,
mora, and alignment verification. The generator remains provenance, never acoustic evidence.

## 2. System shape

```text
rights-gated audio manifest
        │
        ▼
path-preserving faster-whisper / CTranslate2 N-best
        │
        ├── decoder paths and cumulative score domains
        ├── surface-equivalence aggregation with log-sum-exp path mass
        └── original ASR rank retained permanently
        │
        ▼
cheap evidence frontier
        ├── character N-gram
        ├── mora N-gram
        ├── Unicode subword N-gram
        ├── Semantic MBR
        └── deterministic linear/listwise student
        │
        ▼
adaptive candidate-set size and progressive reranking
        ├── margin/entropy early exit
        ├── budget frontier
        └── expensive reranker or teacher only for ambiguity
        │
        ▼
held-out calibration
        ├── speaker/group-disjoint calibration split
        ├── monotonic Platt mapping for ranker logits
        └── raw logits never treated as probabilities
        │
        ▼
acoustically constrained learned fusion
        ├── acoustic
        ├── mora
        ├── lexical
        ├── preservation
        └── cross-model
        │
        ▼
selective evidence acquisition
        ├── local Whisper re-listening
        ├── Qwen3-ASR second ear
        ├── forced alignment
        ├── acoustic candidate verifier
        └── candidate-locked local teacher
        │
        ▼
accepted / provisional observed transcript
        │
        ▼
separate guarded normalization
```

## 3. Score semantics

Every score is tagged by meaning. The runtime distinguishes at least:

```text
sequence log likelihood
average token log likelihood
candidate-set mass
uncalibrated logit
preference score
held-out calibrated probability
```

Candidate-set softmax mass is not the probability that a transcript is correct. A chat model's
self-reported JSON number is not model-token probability. Only a persisted held-out calibration
artifact may convert a raw reranker score into a probability-like fusion stream.

## 4. Candidate evidence preservation

The CTranslate2 adapter keeps decoder paths before surface aggregation. Multiple paths producing
the same Japanese surface form are represented as one surface-equivalence class with path
provenance and aggregate mass. Candidate IDs are deterministic and evidence metadata records:

- exact score domain;
- model/runtime revision;
- prompt and hotword digests;
- beam, patience, repetition, and length-penalty settings;
- source audio SHA-256;
- path count and source support.

A long recording is never converted into a false global N-best list. Evidence remains scoped to
the exact audio span.

## 5. Ranking tiers

### Ultra-light

- character/mora/subword N-gram;
- Semantic MBR;
- no neural reranker;
- no second ear.

### CPU quality

- N-gram and MBR frontier;
- deterministic linear or small encoder reranker;
- listwise semantic-MWER objective;
- optional offline-teacher distillation cache;
- no mandatory GPU.

### Edge GPU

- quality reranker such as a small multilingual reranker;
- query-selected acoustic verifier;
- ambiguity-only Qwen3-ASR second ear and forced alignment;
- adaptive throttling under memory, latency, queue, or thermal pressure.

### Research

- larger candidate pools;
- multi-teacher on-policy distillation;
- guarded generative proposals;
- delayed or token-level fusion experiments;
- full ablations and group bootstrap intervals.

## 6. Learning objectives

The repository supports multiple objectives because no single loss is assumed optimal:

1. **Pairwise preference** — learns which of two candidates has lower semantic loss.
2. **Listwise semantic MWER** — minimizes expected candidate-set loss directly.
3. **Constrained fusion learning** — learns evidence-stream priors while keeping the acoustic
   family above a configured floor.
4. **Multi-task acoustic heads** — mora/phone CTC, boundaries, accent, F0, and preservation.
5. **Acoustic candidate verification** — candidate mora queries select supporting audio frames.
6. **Multi-teacher distillation** — combines candidate-locked teacher judgments, preserving
   abstention and rejecting high teacher disagreement.

Training, calibration, and final test splits are different contracts. Code rejects cross-use where
it can be determined locally.

## 7. Query-selected acoustic verifier

`QuerySelectedAcousticVerifier` is a small candidate-conditioned model. A candidate mora sequence
queries projected acoustic frames and combines three bounded branches:

1. selected local acoustic evidence;
2. global acoustic context;
3. candidate-internal mora evidence.

Learned gates mix the branches and a balance regularizer discourages branch collapse. This is an
ASR design translation of sparse selection and gated residual ideas. It is not a reproduction of
Qwen Sparse Attention, Gated DeltaNet, Kimi Attention Residuals, or GLM internals.

## 8. Progressive speculative reranking

The system runs cheap scorers first. If the cumulative preference distribution has sufficient
margin and low entropy, it exits before loading an expensive model. Otherwise it escalates until:

- confidence is sufficient;
- the compute budget is reached; or
- all configured stages are exhausted.

This preference distribution remains uncalibrated unless a separate calibration artifact exists.

## 9. Adaptive compute throttling

Four pressure signals are combined:

```text
latency / memory / queue / thermal / battery saver
```

The runtime sheds work in this order:

1. offline teacher;
2. second ear and acoustic verifier;
3. neural reranker;
4. candidate-set and evidence budget reduction.

Hysteresis prevents rapid oscillation. A high-pressure result remains auditable and records the
throttle level and reasons.

## 10. Guarded generative error correction

A generative model may propose text absent from the N-best set, but the proposal is never accepted
on linguistic plausibility. It must:

- reference only existing observed candidate IDs;
- pass a calibrated acoustic verifier;
- meet alignment coverage and mora compatibility thresholds;
- pass stricter thresholds when numbers, dates, currency, entities, negation, or modality change;
- remain within configured surface and mora distance limits.

Rejected and provisional proposals do not enter the observed-eligible pool.

## 11. Functional pipeline contracts

Every stage can be implemented as a pure function over an immutable `ArtifactEnvelope`:

```text
(kind, schema version, payload, provenance, SHA-256)
```

Stage contracts declare input kind, output kind, estimated cost, determinism, and optionality.
Actual cost is checked against the pipeline budget. Optional stages can be safely skipped only when
their input/output contract permits pass-through.

## 12. Evaluation

The benchmark runner requires a locked test split by default and verifies that speakers/groups,
source recordings, and near-duplicate IDs do not cross splits. It reports:

- raw ASR CER;
- calibrated cascade CER;
- Semantic MBR CER;
- oracle CER at K;
- rank regret;
- adaptive K;
- additional-evidence invocation rate;
- critical and domain slices;
- paired group-bootstrap intervals.

Unit tests validate contracts and deterministic algorithms, not recognition accuracy.

## 13. Deployment gate

Quantized or exported artifacts are compared with the source artifact on the same locked test
manifest and hardware. Formats include PyTorch, TorchAO INT8/INT4, bitsandbytes, ONNX,
CTranslate2, and GGUF. A smaller artifact is rejected when it violates configured limits for:

- top-1 or pairwise accuracy;
- semantic loss;
- meaning-critical error;
- calibration or AURC;
- real-time factor or memory;
- deterministic replay.

By default, any increase in critical error blocks promotion.

## 14. Source-of-truth boundary

`semantic-asr` is the ASR evidence, training, calibration, and evaluation source of truth. Koemo is
a product integration layer for recording, AEC, channel management, live preview, UI, summaries,
and lifecycle management. Koemo must not maintain a divergent copy of the core fusion algorithms.
