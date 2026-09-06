# Trainable complete-document ranker

## Purpose

The joint document lattice needs a scorer that can compare complete emitted transcripts with left
and right context. A general-purpose LLM can provide a useful research baseline, but its preference
may be unstable, expensive, difficult to calibrate, and overly attracted to fluent rewrites.

`document_ranker.py` supplies a dependency-free baseline trained directly on document-path
preferences. It remains rank-only and cannot generate a transcript.

## Inputs

Each `DocumentRankInput` contains:

- the exact overlap-resolved candidate document;
- frozen left and right external context;
- a topic summary and opaque entity identifiers;
- local lattice score;
- overlap score;
- duration-weighted acoustic support;
- changed, generated, and ambiguous-overlap window counts;
- total window count;
- whether the candidate retains every first-pass window path.

The document engine exposes these numerical quantities in the synthetic scoring arc passed to the
global scorer. The text in that arc is exactly the text that would be emitted after overlap
resolution.

## Features

The baseline combines two feature families.

### Signed hashed character n-grams

Namespaces are separate for:

- candidate document;
- left context;
- right context;
- topic summary;
- left-context/candidate boundary;
- candidate/right-context boundary;
- opaque entity identifiers.

N-grams are NFKC-normalized and case-folded. A signed BLAKE2 hash limits memory and reduces the
systematic bias of collisions. The feature configuration, dimension, n-gram range, text budgets,
and hash seed are immutable and digestible.

### Dense evidence features

Dense features include local score, overlap score, acoustic support, changed-window fraction,
generated-window fraction, ambiguous-overlap fraction, candidate length, context overlap, and the
retained-path indicator. Means and scales are fitted on the training split and stored in the model
artifact.

## Pairwise objective

Training pairs are created only within one recording/document group. The preferred candidate has a
lower objective:

```text
character error rate
+ critical_error_weight × critical error count
+ false_correction_weight × first-pass-exact alternative penalty
```

The final term makes corruption of an already-correct retained transcript explicitly expensive.
It is not inferred from aggregate CER.

The optimizer is deterministic pairwise logistic regression with bounded pairs per group, L2
regularization, recorded epoch losses, and a fixed random seed. This is a transparent CPU baseline,
not a claim that linear hashed features are the final architecture.

## Calibration

Raw ranker scores are converted to a bounded preference with a median/MAD profile fitted on a
separate calibration split:

```text
tanh((raw score - median) / robust scale)
```

This value is a ranking preference, not a correctness probability. Acceptance and selective-risk
thresholds still require their own held-out calibration.

## Artifact

`DocumentRankerArtifact` stores:

- complete feature configuration;
- sparse and dense weights;
- dense statistics;
- training configuration and manifest digests;
- training example digest;
- epoch losses and training pairwise accuracy;
- calibration profile and calibration manifest digest;
- test manifest digest;
- test pairwise and group top-1 accuracy;
- an internal artifact digest.

Serialization is canonical JSON. Pickle and implicit runtime state are not used. Loading verifies an
exact schema and the internal digest.

## Training CLI

```bash
python scripts/train_document_ranker.py \
  --train ../data/document-ranker/train.jsonl \
  --calibration ../data/document-ranker/calibration.jsonl \
  --test ../data/document-ranker/test.jsonl \
  --revision document-ranker-r1 \
  --output ../artifacts/document-ranker-r1.json \
  --report ../artifacts/document-ranker-r1-report.json
```

The CLI rejects any document group appearing in more than one split.

Each JSONL row contains:

```json
{
  "groupId": "recording-001",
  "candidateId": "path-004",
  "text": "...",
  "leftContext": "...",
  "rightContext": "...",
  "topicSummary": "...",
  "entityIds": ["entity-opaque-17"],
  "localScore": 0.41,
  "overlapScore": 0.08,
  "meanAudioSupport": 0.72,
  "changedWindowCount": 1,
  "generatedWindowCount": 0,
  "ambiguousOverlapCount": 0,
  "windowCount": 8,
  "retainedPath": false,
  "characterErrorRate": 0.043,
  "criticalErrorCount": 0,
  "firstPassExact": false
}
```

References and objective labels must never be present at runtime. They are training/evaluation data
only.

## Runtime use

```python
from semantic_asr.document_ranker import (
    DocumentRankerArtifact,
    DocumentRankerGlobalScorer,
)

artifact = DocumentRankerArtifact.load("document-ranker-r1.json")
scorer = DocumentRankerGlobalScorer(artifact)
```

Pass `scorer` to `with_joint_document_deliberation(...)`. Each returned score is bound to the exact
emitted document path, external context digest, and full artifact digest.

## Promotion boundary

Ranker pairwise accuracy is not sufficient for deployment. The complete ASR system must still pass
the document promotion protocol, including:

- paired CER interval;
- false correction among first-pass-exact recordings;
- number, date, currency, negation, entity, and repair regressions;
- accepted coverage and accepted error;
- overlap deletion review;
- speaker/source-shift slices;
- target-device latency and memory.

A stronger neural scorer replaces this baseline only if it improves that end-to-end frontier, not
merely offline pairwise accuracy.
