# Research translation ledger — 2026-08-31

This ledger distinguishes direct implementation, design translation, and unvalidated research
hypotheses. Similar terminology does not imply copied kernels, reproduced weights, or measured
accuracy gains.

## Pinned primary sources

| Source | Reference | Repository use |
|---|---|---|
| Whisper | Radford et al., arXiv:2212.04356 | multilingual encoder-decoder baseline |
| faster-whisper | SYSTRAN/faster-whisper | CTranslate2 production baseline |
| CTranslate2 | OpenNMT/CTranslate2 | path-preserving generation and quantized runtime |
| Whisper-LM | HiTZ/whisper-lm, arXiv:2503.23542 | n-gram/LLM shallow-fusion comparison |
| ProGRes | AdaDTur/ProGRes, IEEE SLT 2024 | prompted generative N-best rescoring baseline |
| Adaptive GER | ICLR 2026 implementation | risk-controlled adaptive hypothesis-set size |
| MBR for ASR | CyberAgent AI Lab, TMLR 2026 | minimum Bayes risk baseline and utility comparison |
| Qwen3-ASR | QwenLM/Qwen3-ASR | independent second ear and forced alignment |
| Qwen3.8-Flash-Next | official report/repository | sparse-selection and gated-efficiency inspiration |
| Kimi K3 | arXiv:2607.24653 | attention-residual, balancing, distillation, throttling inspiration |
| GLM family | official model reports/repositories | modular reasoning and efficient-serving comparison |
| Calibration | Guo et al., ICML 2017 | temperature/Platt calibration, ECE, Brier, NLL |
| Selective prediction | Geifman & El-Yaniv, NeurIPS 2017 | risk-coverage and abstention |
| Deep ensembles | Lakshminarayanan et al., NeurIPS 2017 | disagreement interpretation |
| CTC | Graves et al., ICML 2006 | mora/phone auxiliary objectives |
| SpecAugment | Park et al., Interspeech 2019 | training augmentation candidate |
| Conformer | Gulati et al., Interspeech 2020 | acoustic encoder comparison |

## Directly implemented in v0.2

- path-preserving CTranslate2 Whisper candidate generation;
- surface-equivalence aggregation and decoder-path provenance;
- character, mora, Unicode-subword, and optional KenLM ranking baselines;
- Semantic MBR with meaning-critical utility terms;
- adaptive candidate-set selection;
- pairwise and listwise semantic-MWER linear students;
- candidate-locked multi-teacher distillation;
- held-out monotonic ranker calibration;
- acoustically constrained learned fusion;
- progressive budgeted reranking and confidence early exit;
- query-selected acoustic candidate verifier;
- adaptive runtime throttling with hysteresis;
- guarded GER proposal verification;
- immutable functional pipeline artifacts;
- speaker/source/near-duplicate split isolation;
- group-bootstrap benchmark intervals;
- rights-gated real-audio manifest runner;
- quantized/exported model deployment gate.

## Architecture translations, not reproductions

### Qwen3.8-Flash-Next

| Report idea | Semantic ASR translation | Boundary |
|---|---|---|
| sparse query-selected computation | candidate mora queries select acoustic frames | no QSA kernel or weights copied |
| gated residual branches | bounded acoustic/context/mora verifier branches | decision model, not transformer block reproduction |
| n-gram embedding/local memory | char/mora/subword N-gram frontier and hashed probability cache | independently implemented scorer |
| efficient long context | span-local evidence, consensus locking, caching | orchestration translation |

### Kimi K3

| Report idea | Semantic ASR translation | Boundary |
|---|---|---|
| Attention Residuals | evidence lineage and residual candidate support | no attention-layer reproduction |
| Quantile Balancing | capped multi-teacher influence and branch-balance losses | balancing analogy |
| multi-teacher on-policy distillation | candidate-locked teacher consensus over current ASR hypotheses | text teacher cannot author observed evidence |
| adaptive throttling | latency/memory/queue/thermal compute shedding | runtime policy, not K3 serving stack |

### GLM family

| Family idea | Semantic ASR translation | Boundary |
|---|---|---|
| modular reasoning stages | typed functional ASR stages and immutable artifacts | no GLM architecture copied |
| efficient inference/serving | effort tiers, progressive reranking, quantization gate | runtime policy only |
| long-context planning | global consensus plus local contradiction islands | ASR evidence planning, not long-context transformer implementation |

## Important rejected shortcuts

1. A chat LLM's JSON number is not accepted as a probability.
2. A fluent generated sentence is not accepted as observed speech.
3. Candidate-set softmax mass is not called correctness probability.
4. A quantized model is not promoted only because it is smaller.
5. Unit-test success is not reported as CER improvement.
6. Synthetic training success is not reported as real-audio quality.
7. Training, calibration, and test records cannot share speakers, recordings, or near duplicates.

## Current unvalidated hypotheses

1. Decoder-path aggregation reduces rank regret compared with best-path text deduplication.
2. Listwise semantic MWER outperforms pairwise training on numbers, negation, entities, and repairs.
3. A character+mora N-gram frontier provides the best CPU quality/latency point for some domains.
4. A 130M–600M reranker improves quality at lower cost than a causal 1B–2B decoder scorer.
5. Progressive reranking avoids most expensive teacher calls without increasing fixed-risk error.
6. Candidate-conditioned acoustic verification admits useful GER proposals without increasing
   unsupported insertion rates.
7. Constrained learned fusion improves calibration while preserving acoustic dominance.
8. Adaptive throttling retains acceptable risk under device pressure better than abrupt model
   fallback.
9. Multi-teacher distillation improves the small student only when teacher disagreement is gated.
10. Qwen3-ASR second-ear calls are most useful on localized meaning-critical contradiction islands.

None is a measured claim until the real-audio workflow runs on an immutable held-out Japanese
manifest and publishes the exact revisions, hardware, metrics, intervals, and negative results.

## Required ablations

```text
Whisper single best
Whisper N-best
Whisper N-best + path aggregation
+ character N-gram
+ mora N-gram
+ Semantic MBR
+ pairwise student
+ listwise semantic-MWER student
+ held-out calibration
+ constrained learned fusion
+ progressive reranking
+ acoustic verifier
+ selective re-listening
+ Qwen3-ASR second ear
+ candidate-locked teacher
+ guarded GER
full adaptive system
```

Each model size and runtime must additionally be compared in FP32/BF16/FP16, INT8, and applicable
INT4 formats through the deployment gate.
