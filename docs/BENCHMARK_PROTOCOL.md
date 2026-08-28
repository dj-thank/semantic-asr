# Benchmark and calibration protocol

## Splits

Use disjoint:

```text
training
calibration
test
```

Speakers, source recordings and near-duplicate utterances must not cross splits.

## Baselines

- faster-whisper single-best
- faster-whisper N-best without semantic/mora fusion
- Qwen3-ASR single transcript
- optional stronger Japanese ASR baseline permitted by licence

## Metrics

### Recognition

- CER
- kana-CER
- Mora Error Rate
- oracle CER of N-best
- top-k exact-match rate

### Meaning-critical

- number/quantity error rate
- date/time error rate
- currency error rate
- negation error rate
- critical-entity error rate
- punctuation F1

### Preservation

- filler preservation
- repair/self-correction preservation
- learner-error preservation
- unsupported correction rate

### Confidence

- Expected Calibration Error with declared bins
- Brier score
- negative log likelihood
- risk-coverage curve
- AURC
- coverage at fixed risk targets

### Efficiency

- real-time factor
- peak GPU VRAM
- peak host memory
- cache hit rate
- second-ear invocation rate
- teacher invocation and abstention rate
- information gain per added inference millisecond

## Calibration

Fit calibration on the calibration split only. Store:

```text
model/revision
runtime package versions
beam and hypothesis count
prompt and hotword policy
calibration dataset revision
profile digest
```

Recalibrate after changing the model, decoder, prompt policy, lexical memory, evidence priors or fusion algorithm.

Minimal CLI record:

```json
{"confidence": 0.73, "correct": true}
```

```bash
semantic-asr calibrate heldout.jsonl --output calibration/profile.json
```

## Statistical comparison

Use paired bootstrap intervals on the same utterances. Report both aggregate and sliced results by:

```text
domain
microphone/noise
speaker
utterance length
speaking rate
learner proficiency
presence of number/date/currency/negation/entity
```

## Acceptance policy

Thresholds for accepted/provisional and evidence acquisition are chosen on the calibration set. The final test set is locked before tuning.
