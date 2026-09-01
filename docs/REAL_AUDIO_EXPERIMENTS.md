# Real-audio experiment runbook

The repository can validate algorithms without model weights in ordinary CI. Recognition-quality
claims require this separate workflow on a rights-cleared Japanese audio manifest.

## 1. Prepare a fail-closed manifest

One JSON object per line:

```json
{
  "sampleId": "example-000001",
  "groupId": "speaker-stable-pseudonym",
  "sourceId": "recording-stable-id",
  "nearDuplicateId": "optional-near-duplicate-cluster",
  "split": "train",
  "audioPath": "/absolute/local/path/example.wav",
  "reference": "えっと、料金は三千円です。",
  "domain": "meeting",
  "rightsDecision": "allow",
  "licenseId": "internal-consent-v1"
}
```

`rightsDecision` is `allow`, `deny`, or `review`. Candidate generation rejects `deny` and `review`
by default. Do not commit local audio paths or raw audio.

Speakers/groups, source recordings, and near-duplicate clusters must not cross `train`,
`calibration`, and `test`.

## 2. Generate path-preserving N-best evidence

```bash
semantic-asr generate-candidates audio-manifest.jsonl \
  --output out/all-candidates.jsonl \
  --ranker-output out/all-ranker.jsonl \
  --model large-v3-turbo \
  --model-revision EXACT_MODEL_REVISION \
  --runtime-revision EXACT_CTRANSLATE2_VERSION \
  --device auto \
  --compute-type int8 \
  --beam-size 12 \
  --hypotheses 12
```

The exported manifest stores audio SHA-256, not the audio path. It also records score domains,
prompt/hotword digests, decoder settings, model/runtime revisions, and decoder-path provenance.

## 3. Partition and prove isolation

```bash
semantic-asr partition-manifest out/all-candidates.jsonl \
  --output-dir out/splits
```

This creates:

```text
out/splits/train.jsonl
out/splits/calibration.jsonl
out/splits/test.jsonl
out/splits/partition.json
```

The command fails if a group, recording, or near-duplicate cluster appears in multiple splits.

## 4. Train candidate rankers

### Pairwise baseline

```bash
semantic-asr train-ranker out/splits/train.jsonl \
  --output out/ranker-pairwise.json
```

### Listwise semantic MWER

```bash
semantic-asr train-listwise-ranker out/splits/train.jsonl \
  --output out/ranker-listwise.json
```

The listwise objective minimizes expected candidate-set semantic loss, rather than independent
binary decisions.

### N-gram baselines

```bash
semantic-asr train-ngram train-text.txt \
  --mode character --order 5 --output out/char-5gram.json

semantic-asr train-ngram train-text.txt \
  --mode mora --order 5 --output out/mora-5gram.json

semantic-asr train-ngram train-text.txt \
  --mode subword --order 4 --output out/subword-4gram.json
```

The pure-Python implementation is a reproducible baseline. The optional KenLM backend is intended
for larger corpora and lower latency.

## 5. Fit held-out ranker calibration

First score candidates in the calibration split:

```bash
semantic-asr score-ranker-calibration out/splits/calibration.jsonl \
  --ranker-profile out/ranker-listwise.json \
  --output out/calibration-scores.jsonl
```

Then fit a monotonic Platt mapping:

```bash
semantic-asr calibrate-ranker out/calibration-scores.jsonl \
  --source-ranker semantic-asr-listwise-mwer-v0.2 \
  --output out/calibration.json
```

The calibration artifact records sample/group counts and an immutable calibration-manifest digest.
Training or test rows are rejected by the calibration loader.

## 6. Apply calibrated evidence to the locked test split

```bash
semantic-asr apply-ranker out/splits/test.jsonl \
  --ranker-profile out/ranker-listwise.json \
  --calibration out/calibration.json \
  --output out/test-reranked.jsonl
```

The raw ASR rank remains unchanged. The reranker rank is stored separately. Only the calibrated
probability enters the lexical fusion stream.

## 7. Evaluate

```bash
semantic-asr benchmark out/test-reranked.jsonl \
  --output out/report.json \
  --ks 1,3,5,8,12,16,25,50 \
  --bootstrap-iterations 2000
```

The report includes raw ASR CER, calibrated cascade CER, Semantic MBR CER, oracle CER at K, rank
regret, adaptive K, evidence invocation rate, domain/critical slices, and paired group-bootstrap
intervals.

## 8. Train constrained fusion only on calibrated evidence

A fusion-training JSONL row must contain candidate streams already calibrated into `[0, 1]` plus a
target distribution or reference:

```bash
semantic-asr train-fusion fusion-train.jsonl \
  --acoustic-family-floor 0.72 \
  --output out/fusion.json
```

The optimizer projects weights back to the simplex after each update and never allows the acoustic,
mora, and cross-model family to fall below the declared floor.

## 9. Optional multi-teacher distillation

Teacher inputs contain the exact candidate set and one score for each existing candidate ID.
Teachers cannot insert new text.

```bash
semantic-asr distill-teachers teacher-judgments.jsonl \
  --output out/distilled-train.jsonl \
  --rejected-output out/rejected-teacher-examples.jsonl
```

Abstention and excessive Jensen-Shannon teacher disagreement block the example from training.
Large 8B–12B models are best used offline as teachers; the edge runtime should normally use the
small student.

## 10. Self-hosted GitHub workflow

`.github/workflows/real-audio-research.yml` runs the full sequence manually on a runner carrying:

```text
self-hosted, linux, x64, semantic-asr-research
```

It never provisions a hosted GPU and never uploads raw audio. Transcript/reference-bearing
artifacts are uploaded only when the dispatch input explicitly enables it. The default is local-only.

## 11. Claim boundary

A quality claim must name:

- repository commit;
- model and tokenizer revisions;
- runtime versions;
- decoding and prompt policy;
- train/calibration/test manifest digests;
- hardware;
- calibration profile digest;
- aggregate and sliced metrics;
- paired confidence interval;
- real-time factor and memory;
- negative results and representative failures where rights permit.

A passing deterministic test, a successful synthetic training run, or a smaller model artifact is
not evidence of recognition improvement.
