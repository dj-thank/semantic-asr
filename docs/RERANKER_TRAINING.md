# Reranker training and evaluation

## 1. Data contract

Use one JSON object per utterance:

```json
{
  "exampleId": "speaker42-utt001",
  "context": "製品名と会議タイトルなど、発話前に利用可能な文脈だけ",
  "reference": "実際の参照文字列",
  "candidates": [
    {
      "candidateId": "fw-0001",
      "text": "候補文",
      "acoustic": -0.18,
      "avgLogprob": -0.18,
      "rank": 1,
      "hypothesisCount": 12,
      "metadata": {
        "pathCount": 3,
        "sourceSupport": ["faster-whisper"]
      }
    }
  ]
}
```

Instead of `reference`, a dataset builder may provide a complete `losses` map. The reference must never be placed into inference context or candidate metadata.

## 2. Split rules

Use three or four independent splits:

```text
training
validation / model selection
calibration
locked test
```

At minimum, prevent overlap by:

- speaker;
- original recording;
- meeting/session;
- near-duplicate transcript;
- synthetic template family;
- corpus source where domain transfer is being measured.

Calibration data is not used to fit ranker weights. Test data is not inspected while changing K, feature sets, thresholds, prompts, or model revisions.

## 3. Baseline lightweight trainer

Generate or collect JSONL and run:

```bash
semantic-asr train-ranker train.jsonl \
  --output artifacts/linear-ranker.json \
  --epochs 80 \
  --learning-rate 0.08 \
  --l2 0.002
```

The artifact records:

- feature weights;
- feature mean and scale;
- training-manifest SHA-256;
- before/after pairwise accuracy;
- before/after logistic loss;
- epoch losses;
- profile digest.

This trainer is a reproducible baseline, not the expected final accuracy winner.

## 4. Synthetic hard negatives

Create fixture/training augmentation from one reference sentence per line:

```bash
semantic-asr synthetic-data references.txt \
  --output synthetic.jsonl \
  --maximum-negatives 8
```

Current corruption families include:

- negation meaning flip;
- particle substitution;
- number substitution;
- geminate deletion;
- long-vowel deletion;
- moraic-nasal deletion;
- contracted-sound expansion;
- filler deletion;
- nearby phonetic substitution.

Synthetic examples are mixed with real decoder errors. They are never used as the sole test set and never justify a production-quality claim.

## 5. CrossEncoder tier

Use `sentence-transformers` CrossEncoder with identity activation so that pairwise/listwise training operates on raw logits. Recommended starting models:

```text
sbintuitions/modernbert-ja-130m
Japanese encoder checkpoints with compatible sequence-classification heads
small multilingual rerankers
```

Train and compare:

- binary pointwise correctness;
- pairwise margin loss;
- listwise softmax/cross-entropy;
- MWER / expected semantic loss;
- distillation from a larger teacher;
- combinations with critical-token auxiliary labels.

Do not apply a sigmoid during pairwise/listwise training. Calibrate output after training on the calibration split.

## 6. Qwen3-Reranker tier

The runtime adapter obtains:

```text
raw_logit = logit(yes) - logit(no)
```

The generic checkpoint is a starting point. ASR adaptation should train with:

- real N-best groups;
- acoustic score and path provenance serialized as structured features or prompt fields;
- equal-language-naturalness hard pairs where only acoustics distinguish candidates;
- learner errors and disfluencies that must not be normalized away;
- number/date/currency/negation/entity slices;
- explicit abstention or low-margin examples.

Recommended comparison:

```text
Qwen3-Reranker-0.6B frozen
Qwen3-Reranker-0.6B LoRA
Qwen3-0.6B causal sequence NLL
ModernBERT-Ja-130M CrossEncoder
linear ranker
KenLM
```

## 7. Acoustic verifier training

The query-selected verifier consumes:

```text
acoustic_hidden        [batch, frames, acoustic_hidden]
candidate_mora_ids     [batch, candidates, mora_length]
acoustic_mask          [batch, frames]
candidate_mask         [batch, candidates, mora_length]
targets                [batch]
```

Initial training plan:

1. freeze a speech encoder;
2. cache encoder states keyed by audio/model/span revision;
3. construct candidate readings and mora IDs;
4. train candidate cross-entropy plus branch-balance loss;
5. compare full-frame attention with contradiction-island crops;
6. calibrate verifier logits;
7. test whether it resolves fusion/reranker disagreement cheaper than a second ASR.

Hard negatives should share much of the same surface meaning while differing acoustically:

- `3000円` vs `30000円`;
- positive vs negative polarity;
- long vowel / geminate / moraic nasal;
- proper-noun homophones;
- particles and short functional words.

## 8. Teacher distillation

Use 8B/12B/frontier models offline to produce two different resources:

### Ranking preference

The teacher receives only the candidate set and allowed context. Its output is a preference or pairwise decision. It is not a probability.

### Next-token probability cache

Obtain actual model logits/probabilities for ASR-relevant context-target pairs, then build:

```bash
semantic-asr lm-cache-build teacher-probabilities.jsonl \
  --output artifacts/teacher-cache.json \
  --key-hex <at-least-16-byte-secret-as-hex> \
  --teacher local-12b \
  --teacher-revision <exact-revision>
```

The key is deployment-local and must not be committed. Cache files contain keyed digests, target IDs, probabilities, and provenance, not raw context.

## 9. Calibration

For each score path, fit and store a separate calibration profile:

```text
model revision
ranker checkpoint digest
candidate-generation policy
K policy
prompt/context policy
feature version
calibration corpus revision
```

Recalibrate when any item changes. Evaluate:

- NLL;
- Brier score;
- ECE and reliability diagram;
- AURC / risk-coverage;
- coverage at fixed risk;
- calibration by domain/noise/critical-token slice.

## 10. Model selection

Use a multi-objective frontier rather than one aggregate CER:

```text
quality:
  CER
  kana-CER
  mora error
  critical semantic error
  unsupported insertion/correction
  rank regret
  oracle gap

safety:
  ECE
  AURC
  risk at operational coverage
  learner-error/filler preservation

cost:
  RTF
  p50/p95 latency
  peak host memory
  peak VRAM
  model load time
  verifier/second-ear/teacher invocation rate
```

A model is promoted only when it is Pareto-useful for at least one deployment tier and does not violate the observed/normalized invariant.
