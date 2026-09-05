# Frozen bidirectional character n-gram scorer

## Role

This scorer is the lowest-complexity executable baseline for the document-context experiment. It is
not expected to be the final quality model. Its purpose is to establish whether the experiment
harness, context controls, score identity, and promotion rules behave correctly before introducing
a large neural scorer.

It has useful properties for a baseline:

- dependency-free runtime;
- no text generation;
- no instruction following;
- deterministic forward and reverse likelihoods;
- explicit train/calibration separation;
- immutable manifest and artifact digests;
- exact canonical JSON serialization;
- no pickle or implicit Python runtime state.

## Training manifests

The training CLI expects JSONL rows:

```json
{
  "text": "レビュー完了まではまだマージしません。",
  "speakerId": "speaker-001",
  "sessionId": "session-001",
  "licenseId": "dataset-license-id",
  "rightsDecision": "allow",
  "leftContext": "optional calibration-only left context",
  "rightContext": "optional calibration-only right context"
}
```

Training and calibration must use separate files, speakers, and sessions. Every row must explicitly
allow creation of the derived artifact.

N-gram count tables can retain information about their source text. The CLI therefore requires
`--allow-derived-artifact` and does not treat a count model as privacy-free merely because it is not
a neural checkpoint.

## CLI

```bash
python scripts/train_document_ngram_scorer.py \
  --train ../semantic-asr-data/document-lm-train.jsonl \
  --calibration ../semantic-asr-data/document-lm-calibration.jsonl \
  --output ../semantic-asr-artifacts/document-char-ngram.json \
  --name document-char-ngram \
  --revision corpus-r1-order5-alpha02 \
  --order 5 \
  --alpha 0.2 \
  --allow-derived-artifact
```

The output records:

- exact SHA-256 of both input manifest files;
- forward and reversed model revisions;
- order and smoothing coefficient;
- complete sorted count tables;
- held-out normalization center and scale;
- calibration sample count;
- top-level artifact digest.

Changing one count, smoothing value, manifest digest, or normalization value causes artifact loading
to fail.

## Score semantics

For an ordered forward arm:

```text
score = average log P(document | optional left context)
```

For an ordered bidirectional arm:

```text
score = 0.5 × [
  average log P(document | left context)
  + average log P(reverse(document) | reverse(right context))
]
```

The held-out normalization applies:

```text
tanh((raw_score - calibration_center) / calibration_scale)
```

The result is a bounded ranking utility in `[-1, 1]`; it is not a probability that the transcript
is correct.

## Shuffled control

The shuffled control deterministically orders window texts from:

```text
arm seed + case ID + window index
```

The same model artifact, candidate paths, and scored-character budget are used. If the generated
order accidentally equals the original order, the implementation rotates it for documents with
more than one window so the control is genuinely shuffled.

## Scaling limit

This implementation caches immutable count maps, but a Python character n-gram model is still a
baseline. It is suitable for contract validation and moderate corpora, not a claim of optimal
throughput. Large-scale experiments should compare:

- KenLM or an equivalent compiled character/mora model;
- a compact Japanese cross-encoder;
- a frozen causal scorer;
- an audio-text deliberation encoder.

Every replacement must retain the same scorer registry, candidate freeze, reference isolation,
shuffled control, and promotion contracts.
