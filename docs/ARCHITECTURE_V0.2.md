# Semantic ASR v0.2 architecture

## Status

This document is an implementation contract for the v0.2 research stack. It does not claim an accuracy improvement. Every quality claim requires the held-out protocol in `docs/BENCHMARK_PROTOCOL.md` plus the additional experiments defined here.

The v0.1 invariant remains non-negotiable:

```text
observedTranscript != normalizedTranscript
```

An observed transcript must remain attached to acoustic evidence. A language model may rank existing candidates or propose a new candidate, but a proposed candidate cannot become observed evidence until an acoustic verifier accepts it.

## 1. Design goals

The v0.2 system must be:

1. **Evidence preserving** — raw paths, scores, tokenizations, model revisions and source spans are retained.
2. **Semantically typed** — a log likelihood, logit, probability and model-authored preference are never interchangeable.
3. **Calibrated on held-out data** — candidate-local normalization is not presented as correctness probability.
4. **Adaptive** — candidate count, second-ear invocation and verifier cost depend on measured risk.
5. **CPU viable** — KenLM, MBR and a compact reranker form the canonical baseline; larger models are optional tiers.
6. **Composable** — candidate generation, scoring, ranking, verification and normalization are independent pure stages around explicit side-effecting adapters.
7. **Auditable** — every decision has an immutable input digest, stage trace, model revision and calibration digest.
8. **Falsifiable** — every architectural idea has a baseline, ablation, metric and stop condition.

## 2. End-to-end cascade

```text
Audio / Koemo channel spans
  -> acoustic preprocessing and segmentation
  -> diverse ASR candidate generation
  -> path-preserving surface candidate pool
  -> cheap lexical and mora scorers
  -> MBR candidate selection baseline
  -> compact learned reranker
  -> constrained calibrated fusion
  -> risk-controlled adaptive candidate-set selection
  -> confidence gate
      -> accepted existing candidate
      -> selective re-listen / second ear / forced alignment
      -> optional generative proposal
           -> acoustic verification
           -> accepted or rejected proposal
  -> immutable observed transcript
  -> separately attached normalization
```

### Runtime tiers

| Tier | Intended hardware | Required stages |
|---|---|---|
| `cpu-minimal` | small CPU | ASR, candidate aggregation, char/mora n-gram, MBR, calibration |
| `cpu-quality` | modern CPU | plus compact cross-encoder reranker |
| `gpu-compact` | small GPU | plus 0.6B reranker/second ear and selective verifier |
| `offline-teacher` | training workstation | larger teacher, synthetic hard negatives, distillation |

The same evidence contracts are used in every tier. A high-capacity tier must not silently change the meaning of an output field.

## 3. Candidate evidence model

### 3.1 Paths are not strings

Multiple beam paths can decode to the same surface text. Deduplicating those paths by retaining only the best score discards probability mass and diversity evidence. v0.2 aggregates paths into a `SurfaceCandidate`:

```text
surface candidate
  original text
  equivalence key
  decoder paths[]
  per-path token IDs
  per-path cumulative log likelihood
  aggregate path mass = logsumexp(path log likelihoods)
  source models[]
  source tokenizations[]
  provenance digests[]
```

Exact text is the default equivalence rule. Any broader Unicode or whitespace equivalence is a declared policy and preserves every original surface form.

### 3.2 Score semantics

Every score carries one of these semantics:

```text
cumulative_log_likelihood
average_log_likelihood
log_probability
probability
logit
uncalibrated_score
preference
loss
cost
```

A chat model writing `0.8` in JSON is a `preference`, not a probability. A probability field can only be constructed by an explicit calibrator fitted on a declared calibration split.

### 3.3 Provenance

Every score records:

```text
scorer name
model and revision
runtime and version
configuration digest
dataset/calibration digest
input evidence digest
```

This makes cache replay and model comparison safe across revisions.

## 4. Candidate generation

A fixed top-5 beam list is only one baseline. The experiment matrix must include:

- beam sizes and patience values;
- adaptive `K` in `{1, 3, 5, 8, 10, 16, 25, 50}` where supported;
- temperature or top-k sampling for diversity;
- prompt and hotword policies;
- independent ASR models;
- local re-listening on contradiction spans.

For every candidate generator, report:

```text
oracle CER@K
oracle kana-CER@K
oracle mora error@K
critical-token oracle error@K
unique-surface ratio
pairwise diversity
latency and memory
```

If oracle quality does not improve as K grows, reranking cannot repair the generator and the experiment should stop before spending compute on a larger reranker.

## 5. Baseline scoring and MBR

Before an LLM is introduced, v0.2 establishes these baselines:

1. ASR single best.
2. ASR N-best maximum score.
3. Character n-gram.
4. Mora n-gram.
5. Subword n-gram.
6. Existing-candidate Minimum Bayes Risk decoding.
7. Semantic MBR with a declared weighted loss.

The safe MBR path selects an existing candidate only. A confusion-network consensus may be emitted as a **proposal**, never directly as observed evidence.

A default semantic loss is a weighted combination of:

```text
character edit loss
mora edit loss
number/date/time/currency mismatch
negation or modality mismatch
entity mismatch
disfluency/preservation mismatch
unsupported insertion penalty
```

Weights are tuned on calibration data and frozen before test evaluation.

## 6. Learned reranking

### 6.1 Compact first

The first learned models are deliberately small:

- a compact Japanese encoder cross-reranker;
- a compact multilingual reranker;
- a linear constrained stacker as an interpretable baseline.

Larger causal LMs are not privileged. They must beat the compact baseline at a declared quality/latency point.

### 6.2 Features

The shared feature contract includes:

```text
aggregate acoustic log likelihood
best-path and path-mass gap
beam rank and margin
candidate count and diversity
char/mora/subword LM scores
mora and phone agreement
cross-model support
forced-alignment support
candidate length and insertion indicators
number/date/currency/negation/entity flags
preservation evidence
contextual lexical evidence
teacher preference (explicitly non-probabilistic)
```

### 6.3 Objectives

The experiment matrix compares:

- pointwise quality regression;
- pairwise logistic ranking;
- listwise softmax ranking;
- MWER-style expected loss;
- critical-token auxiliary loss;
- teacher distillation.

The primary target is not grammatical naturalness. It is expected held-out transcription loss under the observed-evidence policy.

### 6.4 Constraints

A learned fusion model must preserve safety constraints:

- lexical or language-only evidence cannot override a strong acoustic contradiction by itself;
- missing evidence cannot improve confidence;
- teacher abstention remains abstention;
- generated text has zero observed-evidence eligibility until verified;
- calibration and ranking training splits are isolated from final test speakers and recordings.

## 7. Proper language-model scoring

For a causal LM, candidate score means teacher-forced sequence log likelihood:

```text
sum_t log P(token_t | token_<t, context)
```

The implementation must not use the maximum probability of an unrelated next token as a sequence score. Length normalization and tokenization policy are explicit experiment parameters.

Dedicated rerankers expose logits or scores. These remain uncalibrated until fitted on held-out ASR examples.

## 8. Adaptive K and finite-sample risk control

The runtime does not always pass the same number of hypotheses to expensive stages. Candidate-set policies are selected from calibration observations using a finite-sample upper risk bound and multiple-policy correction.

Each policy declares:

```text
candidate count K
stages enabled
confidence threshold
maximum cost
```

A policy is eligible only if its calibrated upper risk bound is below the target. Among eligible policies, choose the lowest measured cost, then the smallest K. If no policy satisfies the target, use the conservative fallback and mark the observation provisional.

This mechanism is separate from heuristic entropy thresholds; both are retained for ablation.

## 9. Learned evidence acquisition

The v0.1 planner's hand-authored information gain and cost functions remain a deterministic baseline. v0.2 logs:

```text
state features
action
measured latency/memory
risk before action
risk after action
correctness and critical loss
```

A learned gain model estimates:

```text
E[loss_before - loss_after | state, action]
```

A learned cost model estimates platform-specific latency. The scheduler maximizes expected loss reduction per cost under a hard budget. It initially uses supervised regression; contextual-bandit exploration is a later experiment and is never enabled in production by default.

## 10. Guarded generative error correction

A generative model may propose text outside the N-best list only after the existing-candidate cascade remains uncertain.

Proposal protocol:

1. Generate one or more bounded proposals from contradiction islands only.
2. Mark every proposal as `generated_candidate`.
3. Score mora/phone compatibility.
4. Run forced alignment or a text-speech verifier.
5. Compare against the best existing candidate under calibrated risk.
6. Accept only when the verifier threshold and semantic-preservation constraints pass.
7. Otherwise retain the existing candidate and record rejection reasons.

The language model never writes directly to `ObservedTranscript`.

## 11. Acoustic verifier research track

A compact verifier receives acoustic embeddings plus a candidate mora/phone sequence and predicts compatibility. It is trained with:

- real N-best competitors;
- hard negatives for long vowels, sokuon, moraic nasal, particles, numbers and negation;
- generated corrections rejected or accepted by reference alignment;
- domain and speaker-disjoint splits.

The verifier is compared with full second-ear ASR on accuracy, calibration, RTF and memory. It is retained only if it improves the quality/cost frontier.

## 12. Architecture translations from general LLM research

Architectural ideas from general-purpose LLMs are treated as hypotheses, not copied claims.

### Sparse attention / selected memory

Translate token/block selection into selective acoustic evidence retrieval: only high-risk contradiction spans receive a second pass. This is an orchestration analogy, not a reproduction of an attention kernel.

### Gated residual and expert routing

Translate dynamic branches into constrained evidence gates and hardware-aware stage routing. The gate chooses among n-gram, compact reranker, second ear and verifier based on risk and cost.

### N-gram or local memory augmentation

Use rights-gated hashed lexical/mora memory as a cheap local prior. Source text is not reconstructable from committed artifacts.

### Long-context efficiency

Use consensus locking, span-level cache reuse and hierarchical long-form state instead of repeatedly sending a full meeting transcript through every model.

### Speculative and draft/verify execution

Use a cheap reranker as the draft decision and an acoustic verifier as the verifier. Acceptance is based on measured evidence, not token agreement alone.

### Mixture-of-experts style specialization

Train small specialist scorers for numbers, names, negation, learner errors and disfluency preservation, then route only relevant contradiction islands to them. Experts expose typed evidence rather than editing text.

Only official, revision-pinned primary sources may be named in the research ledger. Unresolved model names remain in a provisional registry and do not justify code or claims until an official paper/repository is pinned.

## 13. Koemo integration boundary

`semantic-asr` is the sole ASR research core. Koemo owns:

```text
Windows capture
mic/system channel handling
AEC
live preview
bounded audio buffers
model warm/unload lifecycle
product UI and exports
```

Semantic ASR owns:

```text
authoritative candidate evidence
ranking and calibration
selective verification
observed/normalized separation
research metrics and experiment manifests
```

Koemo regex corrections may create a normalized derivative or lexical hint, but must never mutate raw observed evidence.

## 14. Required v0.2 ablation

```text
A0  ASR single best
A1  path-preserving N-best
A2  + n-gram scorers
A3  + existing-candidate MBR
A4  + compact learned reranker
A5  + constrained calibrated fusion
A6  + adaptive K / risk control
A7  + selective re-listening
A8  + independent second ear
A9  + compact acoustic verifier
A10 + guarded generative proposal
A11 full system
```

For each step, publish paired confidence intervals and the change in:

```text
CER
kana-CER
mora error
critical semantic loss
unsupported insertion/fabrication
ECE/Brier/NLL/AURC
coverage at target risk
RTF, memory and energy proxy
```

## 15. Stop conditions

An experimental component is rejected when any of the following holds on locked evaluation data:

- no statistically credible quality improvement;
- quality improves only by violating observed-evidence preservation;
- critical-token or hallucination loss regresses beyond the declared tolerance;
- it is dominated on the quality/cost Pareto frontier;
- calibration deteriorates and cannot be repaired on the calibration split;
- licensing or data provenance is not executable and auditable.

Negative results remain in the research ledger.