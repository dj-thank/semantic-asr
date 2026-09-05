# Score producer and consumer inventory

Inventory base: commit `5eba248ca85c7c87923b2016cd0d2a764f4a28f4`, tree
`4837f1ffac0c0e1aad4a296f0d731dc9e348c5dd`.

This table records semantic ownership. It is not a benchmark result.

## Producers

| Producer | Historical representation | Canonical semantics | Normalization / unit | Domain boundary | Calibration status |
|---|---|---|---|---|---|
| CTranslate2 / faster-whisper decoder paths | floats on `CandidateEvidence`; path pool log likelihood | cumulative log likelihood or decoder log probability, according to adapter field | sequence or path-normalized | model revision, span, prompt/hotwords, decode namespace, beam/patience/length penalty, temperature | decoder likelihood, not correctness probability |
| `TransformersCausalSequenceScorer` | `score_types.EvidenceScore` cumulative + average | cumulative and average log likelihood | sequence and mean-token; token-power when alpha differs from 1 | model/revision, prompt context, tokenizer/config | raw likelihood |
| Count/KenLM sequence scorers | `score_types.EvidenceScore` cumulative + average | cumulative and average log likelihood | sequence and mean-token | model artifact, vocabulary/tokenization, context | raw likelihood |
| Loopback `/v1/rerank` scorer | rich `EvidenceScore` | uncalibrated score | none | endpoint model/config and candidate set | uncalibrated |
| CrossEncoder / Qwen reranker adapters | raw float/logit features or rich score | logit or uncalibrated score | none | model/revision/template/tokenizer | uncalibrated until registered profile |
| Phone CTC | simple `EvidenceScore(LOG_LIKELIHOOD)` | average log likelihood | mean acoustic frame | audio, window, encoder/revision, phone label set, preprocessing | raw likelihood |
| Audio-to-mora CTC | simple `EvidenceScore(LOG_LIKELIHOOD)` | average log likelihood | mean acoustic frame | audio, window, encoder/revision, mora label set, preprocessing | raw likelihood |
| Discrete-unit centroid DTW | typed cost/ranking feature | cost or uncalibrated score | path-normalized DTW cost | audio, codebook/centroids, layer, collapse/projection/DTW config | raw feature |
| `PlattCalibrator` / `IsotonicCalibrator` | rich probability `EvidenceScore` | correctness probability | inherits source normalization | exact source score domain and held-out calibration split | valid only through registered applicable profile |
| `RankerCalibrationProfile` | float output | correctness probability after typed adapter | must match source score normalization | source ranker/model/revision/config and calibration manifest | registration required |
| Deliberation utility calibration | `BoundedUtility` | bounded ranking utility | bounded | utility profile + source score digest | deliberately not probability |

## Consumers

| Consumer | Accepted input | Forbidden interpretation |
|---|---|---|
| Decoder-path surface pooling | same semantics, normalization and score-domain digest | adding unrelated models/spans/prompts/temperatures as one posterior |
| `document_scorer.py` | cumulative sequence log likelihood; optional separately declared length normalization | mean-frame likelihood or arbitrary `log_likelihood` |
| `phonetic_evidence.py` | mean-frame phone/mora likelihood | treating candidate-derived reading as independent acoustic evidence |
| `deliberation_evidence.py` | exact scorer, semantics, normalization and direction declared by a utility profile | calling bounded utility a correctness probability |
| `phonetic_bridge.py` | phone/mora scores transformed by matching held-out utility profiles | mixing phone and mora raw likelihoods directly |
| `fusion.py` | per-candidate bounded evidence features | accepting an unregistered `[0,1]` value as calibrated correctness |
| Acceptance, abstention and risk gates | registry-validated correctness probability, or a separately defined empirical risk statistic | `calibrated=True` without an applicable profile receipt |
| Serialized experiment artifacts | canonical v2 or a losslessly migratable legacy schema | guessing normalization for an ambiguous legacy likelihood |

## Migration order

1. Install canonical type, registry, golden fixtures and compatibility imports.
2. Migrate phone/mora and deliberation profiles, because they currently use the coarse type.
3. Re-export canonical types from `score_types.py` and keep deterministic calibrator algorithms.
4. Bind sequence-scorer normalization and model/input domain provenance.
5. Add typed adapters for `RankerCalibrationProfile` and acceptance gates.
6. Retire direct creation of legacy probability objects after downstream callers use registries.
7. Remove the compatibility enum only in a separately versioned release after serialized artifact
   replay and external caller migration.

## Audit queries

The migration test asserts that both historical import paths expose the exact same Python class.
Future code review should reject new `class EvidenceScore` definitions outside `score_contract.py`.
A repository-level source audit should also flag direct probability construction outside registered
calibrators and raw arithmetic over scores without `require_same_score_domain()`.
