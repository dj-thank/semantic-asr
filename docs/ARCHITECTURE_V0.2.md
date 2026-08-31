# Semantic ASR v0.2 architecture

## 1. Scope and invariant

v0.2 keeps the v0.1 invariant:

```text
observedTranscript != normalizedTranscript
```

A language model, reranker, MBR decoder, or generative correction model may:

- score existing candidates;
- request more acoustic evidence;
- propose a new candidate for later acoustic verification;
- select a readability-only normalized candidate.

It may not silently author the immutable observed transcript.

## 2. Candidate cascade

```text
audio
  │
  ├─ faster-whisper / CTranslate2 path-preserving N-best
  │      ├─ individual decoder paths
  │      ├─ cumulative and average log probability
  │      ├─ model/span/prompt/decode score domain
  │      └─ exact source and runtime provenance
  │
  ├─ surface-equivalence pooling
  │      ├─ logsumexp path mass only inside one score domain
  │      ├─ no arithmetic across unrelated models or spans
  │      └─ cross-model source support
  │
  ├─ cheap baselines
  │      ├─ acoustic score and beam rank
  │      ├─ character/mora/word N-gram
  │      └─ semantic Minimum Bayes Risk
  │
  ├─ adaptive K
  │      ├─ posterior-mass target
  │      ├─ selective risk
  │      ├─ semantic criticality
  │      └─ candidate diversity
  │
  ├─ optional learned reranker
  │      ├─ dependency-free linear pairwise ranker
  │      ├─ ModernBERT/CrossEncoder raw logits
  │      └─ Qwen3-Reranker raw yes-vs-no logit margin
  │
  ├─ held-out calibration
  │      └─ only calibrated values may be called probabilities
  │
  ├─ constrained five-stream fusion
  │
  ├─ fusion–MBR agreement check
  │      ├─ agreement: retain selected evidence
  │      └─ disagreement: acquire evidence or remain provisional
  │
  └─ selective evidence
         ├─ Whisper re-listening
         ├─ Qwen3-ASR second ear
         ├─ forced alignment
         ├─ query-selected acoustic verifier
         ├─ lexical lookup
         └─ optional local teacher
```

The default cascade is conservative. Semantic MBR does not override the fusion winner merely because it disagrees. It acts as an independent decision view and a trigger for additional evidence. An explicitly calibrated experiment may enable MBR tie-breaking for small fusion margins.

## 3. Score semantics

v0.2 separates five numerical meanings:

```text
raw score
log likelihood
logit
preference
probability
```

A chat model writing `0.9` into JSON produces a preference value, not a calibrated probability. A Qwen3 or CrossEncoder output is a raw logit. A decoder supplies log likelihood. Only a held-out calibration profile can convert an eligible score to a probability used by acceptance/risk logic.

`EvidenceScore` enforces this contract and rejects:

- a probability outside `[0, 1]`;
- a calibration digest without `calibrated=true`;
- a non-probability value marked as calibrated;
- use of uncalibrated preferences as probabilities.

## 4. Path-preserving surface pool

The v0.1 adapter retained only the strongest decoder path for duplicate text. v0.2 stores every path and aggregates probability mass:

```text
log P(surface) = logsumexp(log P(path_1), ..., log P(path_n))
```

The operation is legal only when paths share an explicit score domain:

```text
adapter
model revision
span
prompt/hotword policy
decode namespace
beam/patience/length penalty
```

Scores from Qwen, Whisper, another span, or another prompt are never added as if they belonged to one normalized decoder distribution. Their agreement is recorded separately as cross-model support.

## 5. Semantic MBR

For candidate `y`, v0.2 minimizes expected loss under the candidate posterior:

```text
R(y) = sum_h P(h | x) L(y, h)
```

The default semantic loss combines:

- normalized surface edit distance;
- mora edit distance when trustworthy readings exist;
- number/date/time/currency/negation/entity sequence loss;
- preservation disagreement.

Kanji-only candidates without readings do not receive a false zero mora distance. Mora loss falls back to surface evidence until a reading or Mora Shadow is supplied.

## 6. Adaptive candidate K

A fixed top-5 list is replaced by a bounded policy. K grows when:

- posterior mass remains diffuse;
- selective risk is high;
- semantic contradiction is critical;
- too few distinct surfaces are represented.

The policy stops when it reaches the posterior-mass target or the next candidate contributes negligible mass. A separate risk-control profile can be marked verified only when it names a held-out calibration manifest and satisfies minimum sample, empirical risk, and empirical coverage requirements.

## 7. Learned rerankers

### 7.1 Dependency-free linear ranker

The base installation can train a pairwise logistic ranker without PyTorch. Features include:

- acoustic/mora/lexical/preservation/cross-model evidence;
- average and sequence log probability;
- reciprocal and relative beam rank;
- decoder path count;
- source diversity;
- candidate length;
- critical-token count;
- context overlap.

The trainer records feature statistics, training-manifest SHA-256, epoch losses, pairwise accuracy, and a profile digest.

### 7.2 CrossEncoder and ModernBERT tier

The optional CrossEncoder runs with an identity activation. Pairwise/listwise raw logits are therefore preserved rather than saturated by a sigmoid before calibration. This tier is intended for CPU or small-GPU operation with Japanese encoder models.

### 7.3 Qwen3 reranker tier

The optional Qwen3 adapter returns the raw margin:

```text
logit(yes) - logit(no)
```

It does not parse a model-authored numeric probability. It remains a language-evidence branch and is calibrated before fusion.

## 8. Query-selected acoustic verifier

`QuerySelectedAcousticVerifier` is a small candidate-conditioned model. A candidate mora sequence queries acoustic frames and selects the frames most useful for verifying that candidate. Three bounded branches are mixed by learned gates:

1. query-selected local acoustic evidence;
2. global acoustic context;
3. candidate-internal mora evidence.

A balance loss discourages permanent branch collapse. The output is a candidate ranking logit, not a transcript generator.

This design is an ASR-domain translation of sparse selection and gated residual principles. It is explicitly not a reproduction of Qwen QSA/GDN, Kimi KDA/AttnRes, or GLM mHC kernels.

## 9. Sparse evidence router

The v0.2 router treats re-listening mechanisms as experts. Routing score combines:

```text
base utility
+ load-balance bonus
+ empirical reward bonus
+ semantic-criticality bonus
- redundant same-island penalty
```

Its state stores only selection counts and observed rewards. Underused evidence actions receive bounded exploration credit, preventing a historically dominant model from starving potentially useful alternatives. The policy remains deterministic for a fixed state and candidate set.

## 10. Offline-teacher probability cache

A large 8B/12B/frontier teacher need not run on the edge device. Its next-token probabilities can be exported offline into a keyed, hashed context cache:

- no raw context text;
- no raw token-context sequence;
- keyed SHA-256 context digest;
- longest-suffix backoff;
- explicit backoff penalty;
- teacher/revision provenance.

This is language-model evidence, never acoustic proof.

## 11. Generative correction boundary

A generative model may create a new hypothesis only through a future guarded-GER interface:

```text
LLM proposal
  -> new candidate ID
  -> mora/phone construction
  -> acoustic verifier or ASR forced score
  -> accepted candidate, rejected candidate, or provisional result
```

The proposal cannot directly replace observed evidence.

## 12. Koemo boundary

Koemo supplies product/runtime capabilities:

- microphone and system-audio capture;
- AEC;
- bounded live buffers;
- live provisional captions;
- model warmup/unload and CPU fallback;
- desktop UI and export.

Semantic ASR supplies the authoritative final transcript core. Koemo regex correction may feed the separate normalized layer but must never mutate immutable observed evidence.

## 13. Deployment tiers

```text
Tier 0  faster-whisper single-best
Tier 1  path N-best + MBR + linear ranker
Tier 2  + Japanese CrossEncoder / ModernBERT
Tier 3  + Qwen3-Reranker-0.6B
Tier 4  + Qwen3-ASR second ear / acoustic verifier
Tier 5  + offline 8B/12B teacher distillation and guarded GER
```

Every tier must report quality and cost independently. A heavier tier is adopted only if paired held-out evaluation shows a useful risk/quality/latency frontier.
