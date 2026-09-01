# Semantic ASR v0.2 experiment matrix

## 1. Principles

Every row is a separately named system. Do not change multiple components and attribute the result to one of them. Use the same locked test utterances, references and scoring code for paired comparisons.

The first question is not “does a larger LLM help?” It is:

```text
where is the current error introduced?
  candidate generation
  candidate scoring
  calibration
  decision threshold
  acoustic verification
  long-form stitching
```

Stop an experiment family when its prerequisite upper bound is absent. For example, if oracle CER@K does not improve beyond K=5, a more expensive K=50 reranker is not the next action.

## 2. Dataset axes

Minimum slices:

```text
read speech / spontaneous conversation / meeting audio
clean / reverberant / stationary noise / competing speech
headset mic / laptop mic / phone / system loopback
short (<5 s) / medium / long span
slow / normal / fast speaking rate
native / learner speech where rights allow
numbers / dates / currency / negation / entities / Latin terms
fillers / repetitions / repairs
special mora / long vowels / sokuon / moraic nasal
```

Use speaker- and source-recording-disjoint train/calibration/test splits.

## 3. Candidate generation study

### G0 — current single best

```text
faster-whisper
beam_size=1
condition_on_previous_text policy declared
```

### G1 — standard beam N-best

Candidate counts:

```text
K = 1, 3, 5, 8, 10, 16, 25, 50
```

Sweep only values supported by the runtime without changing other controls.

### G2 — beam patience

```text
patience = 1.0, 1.5, 2.0
```

### G3 — diverse sampled paths

Low-temperature/top-k candidate generation, combined with beam paths and deduplicated only after path evidence is retained.

### G4 — prompt/hotword variants

Compare:

```text
no context
approved domain context
approved hotwords
live-caption-derived hints (analysis only)
```

Hotword improvements and hallucinations are reported separately.

### G5 — independent second ear

Add Qwen3-ASR 0.6B/1.7B where hardware allows. Its one transcript is an independent source, not decoder N-best.

For G0–G5 report:

```text
oracle CER@K
oracle kana-CER@K
oracle MoraER@K
critical-token oracle@K
unique surface ratio
mean pairwise distance
path entropy
candidate generation RTF/memory
```

## 4. Scoring baselines

### S0 — acoustic score only

Best path and path-mass variants are separate rows.

### S1 — character n-gram

Orders 3/5/7; corpus size and rights manifest declared.

### S2 — mora n-gram

Requires deterministic reading/mora conversion and a separate unknown policy.

### S3 — subword n-gram

Tokenizer/revision and normalization are pinned.

### S4 — proper causal sequence likelihood

Evaluate compact Japanese causal LMs using full candidate sequence log likelihood. Compare cumulative and length-normalized values.

### S5 — dedicated compact reranker

At least:

```text
constrained linear baseline
100M-class Japanese cross-encoder
0.6B multilingual/Japanese reranker
```

A chat model writing numeric probabilities is not a scoring baseline.

## 5. Decision baselines

### D0 — maximum acoustic score

### D1 — maximum calibrated fused score

### D2 — existing-candidate character MBR

### D3 — semantic MBR

Ablate each loss component:

```text
character
mora
critical semantic
preservation
unsupported insertion
```

### D4 — learned pointwise reranker

### D5 — pairwise reranker

### D6 — listwise reranker

### D7 — MWER/multi-objective reranker

### D8 — sparse specialist reranker

Compare hard top-k routing, soft routing and no experts. Track expert utilization and collapse.

## 6. Calibration study

For every frozen scorer/reranker:

```text
uncalibrated softmax
Platt
isotonic
temperature scaling
```

Metrics:

```text
NLL
Brier
ECE with declared bins
adaptive calibration error if implemented
AURC
coverage at risk targets 1%, 2%, 5%, 10%
```

Calibration is fitted once on the calibration split. Do not choose the calibrator by test metrics.

## 7. Adaptive candidate and stage policies

### A0 — fixed K

### A1 — heuristic entropy/margin thresholds

### A2 — conservative finite-sample risk control

### A3 — learned gain/cost planner

### A4 — learned planner + hard target-risk gate

Report:

```text
mean and p95 K
second-ear invocation rate
verifier invocation rate
accepted coverage
risk at accepted coverage
latency and memory
```

## 8. Acoustic verification

### V0 — no verification

Unsafe research baseline for generated proposals only.

### V1 — forced alignment heuristics

### V2 — Qwen3-ASR second-ear agreement

### V3 — compact acoustic-text verifier

### V4 — verifier then second ear on residual uncertainty

Stress tests must contain minimally different but meaning-critical pairs.

Primary verifier metrics:

```text
false acceptance of incorrect proposals
false rejection of correct proposals
ECE/Brier
critical-class false acceptance
latency and memory
```

## 9. Generative error correction

### P0 — no generated proposal

### P1 — N-best-conditioned proposal, never eligible for observed text

Measures oracle proposal quality only.

### P2 — proposal + forced alignment

### P3 — proposal + compact verifier

### P4 — proposal + verifier + second ear

### P5 — proposal model distillation

Compare an offline larger teacher with compact students. Prompts, revisions and decoding settings are frozen.

A proposal is counted as successful only if it both lowers reference loss and passes the deployed verifier policy. “More natural Japanese” alone is not success.

## 10. End-to-end audio-conditioned model

Only after the second-pass frontier is established:

```text
E0 frozen Whisper encoder + projector + frozen Japanese LM
E1 projector + decoder LoRA
E2 auxiliary mora/phone losses
E3 quantized edge export
```

Compare with the second-pass cascade at matched parameter memory and RTF. The end-to-end model is rejected if it gains CER by increasing unsupported fabrication or erasing learner errors.

## 11. Long-form and Koemo study

### L0 — fixed 28 s windows and exact text overlap

### L1 — VAD/speaker-boundary windows

### L2 — consensus locking and contradiction-only reprocessing

### L3 — channel-aware mic/system fusion

### L4 — AEC raw/transformed evidence comparison

### L5 — live-context hints with final evidence separation

Report boundary error, duplication/omission rate, speaker/channel attribution, meeting-level RTF and stop-time p50/p95 latency.

## 12. Hardware matrix

### CPU minimal

```text
int8 faster-whisper or whisper.cpp comparison
KenLM / count n-gram
MBR
constrained linear reranker
```

### CPU quality

```text
100M-class cross-encoder
parallel candidate scoring
quantized inference
```

### Small GPU

```text
0.6B reranker
compact acoustic verifier
selective Qwen3-ASR 0.6B
```

### Training workstation

```text
larger teacher
0.6B/100M distillation
verifier and auxiliary-head training
```

Record exact CPU model, core/thread settings, GPU, drivers, power mode, RAM/VRAM and runtime versions.

## 13. Statistical plan

Primary comparisons use paired bootstrap confidence intervals on utterance-level metrics. For multiple systems, publish all planned comparisons and control the family where a formal claim depends on many selections.

Report:

```text
mean delta
95% paired interval
probability candidate is better
sample count
slice counts
```

Do not report only the best seed. For neural models, use at least three declared seeds when compute permits and retain every run.

## 14. Advancement gates

A component advances when all are true:

1. it improves the declared primary metric with a paired interval excluding no improvement;
2. no critical/fabrication metric exceeds its regression tolerance;
3. calibration and target-risk coverage are acceptable;
4. it is not dominated in quality/cost;
5. rights and provenance checks pass;
6. the result is reproducible from immutable manifests.

Otherwise record the negative result and stop or revise the hypothesis.

## 15. Recommended execution order

```text
1. G0-G3 candidate oracle curves
2. S0-S3 cheap scorers
3. D0-D3 MBR baselines
4. D4-D7 learned compact reranking
5. calibration study
6. adaptive K/risk control
7. Qwen second ear and forced aligner
8. compact acoustic verifier
9. guarded generative proposal
10. Koemo integration and long-form study
11. audio-conditioned Japanese LLM track
```

This order prevents expensive models from obscuring a weak candidate generator or uncalibrated decision layer.
