# Source-audio-only phone/mora CTC runtime

## Role

The dual CTC runtime supplies independent acoustic evidence to ambiguous Semantic ASR spans. It is
not the primary transcript generator and does not replace Whisper. Its responsibility is narrower:

```text
source audio crop
      │
      ▼
shared log-Mel + acoustic encoder
      ├─ phone CTC posterior
      └─ mora CTC posterior
               │
               ▼
existing candidate scoring or local P2G proposal
               │
               ▼
full-path context deliberation + acoustic retention guards
```

The runtime preserves the central boundary:

```text
candidate-derived mora_shadow != audio-derived mora posterior
context preference != acoustic proof
```

## Model

`DualPhoneMoraCTC` contains:

- deterministic log-Mel extraction;
- per-utterance feature normalization;
- bounded strided convolutional subsampling;
- a shared Transformer encoder;
- separate phone and mora linear CTC heads.

The heads have independent frozen inventories. CTC blank must occupy index zero in each inventory.
A phone label ID can never be interpreted as a mora label ID.

The architecture is intentionally compact and replaceable. A frozen SSL encoder, Conformer, or
streaming encoder may replace the current shared encoder only if it preserves the same posterior,
artifact, source-audio, and evaluation contracts.

## Audio contract

Version 1 accepts uncompressed PCM16 WAV at the exact model sample rate. It deliberately does not
perform implicit resampling. The reader:

- hashes the complete source file;
- enforces a byte limit and crop-duration limit;
- reads only the requested frame range;
- supports registered mono/stereo channel counts;
- downmixes stereo by the deterministic arithmetic mean;
- rejects compressed, truncated, wrong-rate, wrong-width, empty, or out-of-range input.

A posterior inferred from a crop retains the SHA-256 of the complete recording and absolute crop
timestamps.

## Artifact format

Artifacts are directories containing:

```text
metadata.json
weights.npz
```

`torch.save` and pickle are not used. `weights.npz` is loaded with `allow_pickle=False`.
`metadata.json` records:

- model and frontend configuration plus digest;
- phone and mora inventories plus digests;
- training-manifest digest;
- exact runtime revision;
- weight-file SHA-256;
- every tensor name, shape, dtype, and raw C-order SHA-256;
- top-level artifact digest.

Loading rejects unknown/missing metadata keys, changed files, changed tensor names, object arrays,
shape/dtype mismatches, or per-tensor digest mismatches. Saving never overwrites an existing
artifact directory.

## Training manifest

Each JSONL row contains at least:

```json
{
  "utteranceId": "utt-0001",
  "audioPath": "/absolute/path/utt-0001.wav",
  "sourceAudioSha256": "...",
  "sampleRate": 16000,
  "phoneSymbols": ["k", "o", "N"],
  "moraSymbols": ["コ", "ン"],
  "speakerId": "speaker-001",
  "sessionId": "session-001",
  "sourceId": "recording-001",
  "licenseId": "dataset-license",
  "rightsDecision": "allow",
  "split": "train"
}
```

A metadata sidecar named `<manifest>.metadata.json` supplies manifest name/revision and may pin the
exact JSONL SHA-256. Train, validation, calibration, and test partitions reject shared speakers, sessions, or
source recordings. Every reference-bearing row must explicitly permit the operation.

## Training

```bash
python scripts/train_dual_phonetic_ctc.py \
  --manifest /data/phonetic/manifest.jsonl \
  --phone-inventory /data/phonetic/phones.json \
  --mora-inventory /data/phonetic/moras.json \
  --artifact-dir /artifacts/dual-ctc-r1 \
  --report /artifacts/dual-ctc-r1-training.json \
  --artifact-name ja-phone-mora-ctc \
  --artifact-revision corpus-r1 \
  --runtime-revision semantic-asr-commit-plus-torch-revision \
  --device cpu
```

The trainer updates only the train split, evaluates validation loss after each epoch, keeps the
best validation checkpoint, writes the pickle-free artifact, and immediately reloads it to verify
the complete artifact contract.

Training output must be outside the repository checkout. The test split is not used to choose the
checkpoint.

## Utility normalization

`fit_ctc_utility_calibration()` consumes held-out candidate sets with exactly one correct
pronunciation per example. It computes candidate-specific CTC log likelihoods and fits a bounded
affine-plus-`tanh` utility profile.

This value is **not** a correctness probability. Calibration examples cannot mix model, inventory,
or score-source identities. The report records pairwise candidate accuracy and all example digests.

## Deliberation provider

`SourceAudioPhoneticProposalProvider`:

1. selects only active contradiction spans;
2. prioritizes semantic criticality, ambiguity, and factor weight;
3. crops the exact source audio with bounded padding;
4. infers candidate-independent phone and mora posteriors;
5. scores a frozen local pronunciation lexicon;
6. returns source-audio-bound `VerifiedSpanProposal` objects.

The provider rejects posterior evidence from another recording. The local lattice rebases proposal
utilities onto the span's finite factor budget. Generated proposals remain provisional under the
existing global deliberation policy.

The lexicon provider must be frozen before evaluation. It must not be constructed from the test
reference.

## Evaluation

```bash
python scripts/evaluate_dual_phonetic_ctc.py \
  --artifact-dir /artifacts/dual-ctc-r1 \
  --manifest /data/phonetic/manifest.jsonl \
  --split test \
  --output /reports/dual-ctc-r1-test.json
```

The report separates:

- phone error rate;
- mora error rate;
- per-utterance edits and reference lengths;
- runtime latency;
- Python heap measurements;
- posterior and prediction digests.

Raw phone/mora predictions are omitted by default and require `--include-predictions`.

## Promotion boundary

A lower phone or mora error rate is not by itself a Semantic ASR improvement. Promotion requires a
locked end-to-end comparison with:

- strict and lenient transcript CER;
- semantic-critical errors;
- proposal recovery outside Whisper N-best;
- context-induced false corrections;
- accepted coverage and risk–coverage;
- phone and mora error rates;
- per-span and end-to-end latency/memory;
- ablations for phone-only, mora-only, discrete-unit-only, and combinations.

No artifact becomes a default profile until the end-to-end frontier improves without violating the
observed-transcript preservation contract.


## Four-way split boundary

The model uses four separate evidence partitions:

```text
train       gradient updates
validation  checkpoint and hyperparameter selection
calibration CTC utility normalization for candidate fusion
test        final PER/MER and end-to-end ASR evaluation
```

Validation and calibration must not share speakers, sessions, or source recordings. The test split
is never used for checkpoint selection or utility normalization. This separation prevents an
apparently held-out utility profile from being fitted on the same rows already used to select the
model checkpoint.

## Repeated-label CTC feasibility

A target such as `a a` needs a blank-separated path and therefore at least three acoustic frames.
The runtime computes `target_length + adjacent_repeat_count` before calling the CTC loss. Impossible
alignments fail explicitly; `zero_infinity` is not used to silently turn them into zero-loss rows.

## Deterministic weight archives

Tensor arrays are written in sorted order to a ZIP archive with fixed timestamps, permissions, and
compression settings. Identical state dictionaries therefore produce byte-identical `weights.npz`
files and the same weight-file SHA-256.
