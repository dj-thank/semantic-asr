# Semantic ASR v0.2 training workflow

## 1. Separation of data roles

Use four physically and logically distinct data roles:

```text
training       fit rerankers, verifiers and auxiliary heads
validation     early stopping and model selection inside a training family
calibration    fit probabilities, risk thresholds and adaptive policies
test           locked final evaluation only
```

The public `DatasetManifest` currently expresses `train`, `calibration` and `test`. When a neural training run needs a validation subset, derive it from `train` using a committed group split manifest; never use calibration or test examples for early stopping.

Speakers, source recordings, exact audio hashes and near-duplicate utterances must not cross roles. Run the manifest leakage gate before generating model inputs.

## 2. Candidate generation dataset

For each utterance, persist all raw decoder paths before surface aggregation:

```json
{
  "sampleId": "sample-0001",
  "audioSha256": "...",
  "generator": {
    "adapter": "faster-whisper-ctranslate2-paths-v2",
    "model": "large-v3-turbo",
    "revision": "...",
    "runtimeVersions": {
      "faster-whisper": "...",
      "ctranslate2": "..."
    },
    "variant": {
      "beamSize": 8,
      "hypotheses": 8,
      "patience": 1.0,
      "temperature": 0.0
    }
  },
  "paths": []
}
```

Candidate generation is versioned independently from reranker training. Changing beam, prompt, hotword, VAD, token suppression, equivalence policy or model revision creates a new candidate dataset revision.

## 3. Target losses

Each candidate receives normalized training targets in `[0, 1]`:

```text
targetLoss
criticalLoss
preservationLoss
fabricationLoss
```

A recommended target is:

```text
targetLoss =
  0.45 * normalized CER
  + 0.18 * normalized mora error
  + 0.22 * critical semantic loss
  + 0.10 * acoustic fabrication loss
  + 0.05 * preservation loss
```

These initial coefficients are a declared hypothesis and must be tuned on validation data, then frozen before calibration/test. Report component metrics separately so a good aggregate cannot hide a number or negation regression.

## 4. Real competitors before synthetic negatives

Training groups must contain real ASR competitors whenever possible. Synthetic negatives supplement the tail; they do not replace the true error distribution.

Recommended mixture by candidate count:

```text
50-70% real N-best / sampled decoder paths
10-20% independent second-ear competitors
10-20% Japanese hard negatives
0-10% guarded generative proposals
```

`src/semantic_asr/hard_negatives.py` creates auditable labels for long vowels, sokuon, moraic nasal, particles, numbers, negation, fillers, repetition and rights-gated lexicon neighbors.

Do not train on a synthetic corruption unless its `error_type`, source span and generation configuration are retained.

## 5. Dependency-free CPU baseline

Prepare one JSON object per utterance using `schemas/v02-ranking-group.schema.json`:

```json
{
  "groupId": "sample-0001",
  "candidates": [
    {
      "candidateId": "surface-a",
      "features": {
        "aggregate_acoustic_log_likelihood": -12.3,
        "best_path_log_likelihood": -12.8,
        "path_mass_bonus": 0.5,
        "beam_confidence": 0.73,
        "beam_rank_fraction": 1.0,
        "mora_score": 0.82,
        "lexical_score": 0.10,
        "preservation_score": 0.88,
        "cross_model_score": 0.67,
        "source_count": 2,
        "path_count": 3,
        "candidate_length": 14,
        "number_flag": 0,
        "negation_flag": 0,
        "entity_flag": 1,
        "teacher_preference": 0.0,
        "missing_evidence_fraction": 0.0
      },
      "targetLoss": 0.0,
      "criticalLoss": 0.0,
      "weight": 1.0
    }
  ]
}
```

Train:

```bash
python scripts/train_reranker_v2.py \
  data/train-ranking-groups.jsonl \
  --output artifacts/reranker-linear-v1.json \
  --objective hybrid \
  --epochs 250 \
  --seed 0
```

This model is not a placeholder. It is the interpretable, monotonic, low-cost baseline that every neural reranker must beat.

## 6. Neural sparse-expert reranker

`SparseEvidenceReranker` accepts:

```text
candidate_features [batch, candidates, feature]
state_features     [batch, state]
candidate_mask     [batch, candidates]
```

The router selects a small set of specialist experts. A separate acoustic branch receives a configured minimum mixture weight. This prevents the router from replacing acoustic evidence with a fluent language prior.

Initial specialist semantics:

```text
numbers/units
names/technical terms
negation/modality
particles/special mora
preservation/disfluency
general language fit
```

Expert semantics are an analysis label, not hard-coded truth. Measure router specialization and collapse; do not claim interpretable experts solely from their configured names.

## 7. Multi-objective ranking loss

`MultiObjectiveRankingLoss` combines:

```text
posterior expected task loss        MWER-style direct objective
listwise target distribution        whole N-best ordering
pairwise preference                 local ordering robustness
expected critical loss              numbers/negation/entities/etc.
teacher distillation                optional, lowest default weight
```

The teacher term is optional. Teacher logits are detached and cannot override references. Always report the no-teacher ablation.

## 8. Acoustic-text verifier

`AcousticTextVerifier` consumes speech encoder frames and candidate text/mora embeddings. Training labels should be derived from reference/alignment evidence:

```text
1.0 exact or verified-compatible candidate
0.0 acoustically incompatible candidate
soft labels only when their construction is documented and calibrated
```

Hard negatives must emphasize errors with small edit distance but large semantic impact:

```text
東京 / 京都
15 / 50
行く / 行かない
できます / できません
一時 / 七時
```

Evaluate calibration and false acceptance of generated proposals, not only binary accuracy.

## 9. Larger teacher and distillation

A larger 8B/12B+ model may be used offline to:

- explain ambiguity classes for analysis;
- produce pairwise/listwise preferences;
- identify missing hard-negative categories;
- generate bounded proposal candidates;
- distill into the 0.6B/100M-class student.

It may not:

- define the reference transcript;
- write observed evidence directly;
- turn a generated number/name into ground truth without acoustic verification;
- leak test references into prompts or training data.

Persist the exact teacher model/revision, prompt digest, decoding settings and output. Treat numerical preferences as uncalibrated.

## 10. Calibration

After selecting a frozen reranker checkpoint, run it on the calibration split and fit:

```text
Platt calibration
isotonic calibration
optional temperature/vector scaling for neural logits
```

Select calibration by held-out NLL/Brier/AURC, not ECE alone. The calibration dataset digest is stored in every probability provenance object.

Recalibrate after any change to:

```text
candidate generator
feature definition
reranker weights
prompt/hotword policy
n-gram memory
second-ear model
fusion/gating logic
```

## 11. Adaptive policy fitting

For each calibration sample, execute candidate/stage policies and record:

```text
policy ID
bounded task loss
measured platform cost
sample ID
slice/group
```

Fit risk control with `fit_risk_control`. The initial implementation uses conservative Hoeffding-Bonferroni bounds. Compare it against fixed K and heuristic thresholds before adopting a more powerful conformal/LTT implementation.

Fit planner gain/cost models from actual action observations using `fit_learned_planner`. Cost models are platform-specific; do not transfer a GPU latency model to CPU or another phone.

## 12. Quantization and edge export

Quantization is a separate experiment condition. Record:

```text
weight format and group size
activation precision
KV-cache precision
backend and kernel revision
thread count / device / batch size
model file digest
```

For CPU and small GPU, compare:

```text
FP32 / BF16 / FP16
INT8 weight-only or dynamic
INT4 weight-only where supported
GGUF/llama.cpp dedicated reranker
CTranslate2 quantized runtime
TorchAO or backend-native quantization
```

A quantized model is accepted only after ranking, calibration and critical-token regression tests.

## 13. Run manifest

Every training run emits an immutable manifest containing:

```text
Git commit
source dataset manifest digest
candidate dataset digest
rights registry digest
feature schema digest
random seeds
model/revision/tokenizer
optimizer/scheduler and stopping rule
hardware and runtime versions
checkpoint digest
validation metrics
negative results and warnings
```

Do not overwrite runs. A symbolic `latest` pointer may reference an immutable run directory, but published results cite the digest.

## 14. Minimum acceptance criteria

A new reranker or verifier advances only when:

1. paired confidence interval supports improvement on the primary locked metric;
2. number/negation/entity/fabrication metrics do not regress beyond tolerance;
3. AURC or operational coverage improves or remains within tolerance;
4. it is not dominated on the quality/cost frontier;
5. the exact run can be reconstructed from committed code and manifests.

A successful unit test or decreasing training loss is not evidence of ASR improvement.
