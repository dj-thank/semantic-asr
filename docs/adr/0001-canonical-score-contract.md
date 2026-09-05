# ADR 0001: One canonical score contract

- Status: Accepted for migration
- Date: 2026-09-06
- Issue: #25
- Base commit: `5eba248ca85c7c87923b2016cd0d2a764f4a28f4`
- Base tree: `4837f1ffac0c0e1aad4a296f0d731dc9e348c5dd`

## Context

Semantic ASR historically had two public classes named `EvidenceScore`:

- `score_semantics.py::EvidenceScore`, used by phone/mora and deliberation code;
- `score_types.py::EvidenceScore`, used by sequence scorers and trainable calibrators.

They represented different information. The first distinguished only a coarse
`log_likelihood`; the second distinguished cumulative and average likelihood but still accepted a
probability when the caller supplied `calibrated=True`. Neither type alone proved that the claimed
calibration artifact existed or applied to the score's model, revision, normalization, score domain,
input condition and held-out split.

A numeric range is not a semantic type. A chat model's `0.9`, a ranker sigmoid, an acoustic
posterior, a length-normalized sequence score and a correctness probability may all lie in `[0, 1]`
but are not interchangeable.

## Decision

`semantic_asr.score_contract.EvidenceScore` is the sole score object. The old modules re-export the
same class and may contain compatibility enums or calibrator implementations, but may not define a
second score representation.

Every score records:

1. **semantics** — cumulative log likelihood, average log likelihood, log probability, raw logit,
   uncalibrated score, preference, loss, cost, bounded utility or correctness probability;
2. **normalization** — sequence, mean token, mean frame, token-power length normalization,
   path-normalized, bounded, or none;
3. **provenance** — scorer, model, immutable revision, runtime, configuration, input evidence and
   input condition;
4. **score domain** — an optional explicit digest covering model/span/prompt/decode/temperature and
   normalization for operations such as log-mass aggregation;
5. **calibration receipt** — for correctness probabilities only.

Metadata is recursively copied and frozen at construction. Boolean numbers, NaN and infinities are
invalid. Digest fields are lowercase SHA-256.

## Correctness probability contract

`calibrated=True` and a 64-character string are insufficient. A probability is usable by decision
logic only when a frozen `CalibrationProfileRegistry` contains the referenced calibration artifact
and validates all applicable fields:

```text
source semantics
scorer
model
revision
normalization
score-domain digest
configuration digest
input-condition digest
calibration split digest and split name
profile receipt digest
source score digest, when supplied
```

A calibrator may still emit a receipt-bearing compatibility object for old callers, but
`require_probability()` rejects it unless an applicable registered profile is supplied.

## Likelihood normalization

The compatibility name `log_likelihood` is ambiguous. Migration therefore requires one of:

- `sequence` -> cumulative sequence log likelihood;
- `mean_token` -> average token log likelihood;
- `mean_frame` -> average acoustic-frame log likelihood;
- `token_power` -> length-penalized token score with the exponent in provenance metadata/config;
- `path_normalized` -> a normalized decoder-path log probability.

The only automatic legacy inference is a lossless repository-specific case: historical CTC scores
whose source begins with `ctc-` and whose receipt includes `frameCount` are mean-frame likelihoods.
Unknown legacy `log_likelihood` values fail migration instead of being guessed.

## Aggregation

Scores may be combined as members of one normalized distribution only when semantics,
normalization and explicit score-domain digest all match. Agreement from another model, span,
prompt, temperature or decoding namespace remains cross-model/cross-domain evidence rather than
arithmetic probability mass.

## Serialization and migration

The canonical JSON schema is `semantic-asr.score.v2`. Golden fixtures cover:

- the simple legacy score shape;
- the rich legacy score shape;
- canonical v2.

Legacy rich serialization receives extension keys for normalization and domain identity without
placing contract fields inside user metadata. A migration that cannot preserve meaning raises
`ScoreMigrationError`.

## Consequences

- Phone/mora CTC, sequence scorers, document deliberation and future learned evidence share one
  class and one validation path.
- Existing import paths continue to work while class identity is testable.
- Some historical caller-created probabilities become unusable until their frozen calibration
  profile is registered. This is an intentional correctness change, not a quality improvement.
- Existing float-only fusion calibration remains a bounded feature transform; it must not be
  described or consumed as a correctness probability merely because its result lies in `[0, 1]`.
- Migration can proceed module by module without collapsing all scores to untyped floats.

## Rejected alternatives

### Keep both types and add conversion helpers

Rejected because producer/consumer type choice would continue to depend on import path, and lossy
conversions would remain easy to introduce.

### Treat any `[0, 1]` value as probability

Rejected because numeric range does not establish calibration or applicability.

### Require only a non-empty calibration digest

Rejected because a self-reported digest does not prove that the artifact exists or was fitted for
the current score domain and held-out split.

### Replace every score with a generic arbitrary dictionary

Rejected because it weakens static meaning, validation and reproducibility while making arithmetic
mistakes easier.
