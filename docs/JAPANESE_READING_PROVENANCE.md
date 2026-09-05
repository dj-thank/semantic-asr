# Japanese reading provenance for phonetic training

## Why this layer exists

The phone/mora feature exporter accepts an explicit kana reading. Real Japanese corpora often
provide kanji-containing transcripts instead. Turning those transcripts into readings is a model
or dictionary decision and can introduce label errors, especially for names, numbers, compounds,
particles, repairs, foreign words, and domain terminology.

Semantic ASR therefore keeps four objects separate:

```text
source transcript
machine reading proposal
human review decision
resolved explicit kana reading
```

The feature exporter never performs hidden text-to-reading conversion. This preparation layer
creates the explicit reading manifest and a parallel audit receipt.

## Reading origins

### `human-explicit`

A person or source corpus supplies the reading directly. The normalized kana reading is bound to
the exact source-text SHA-256 and pronunciation-policy digest. It can be used in train,
calibration, or locked test unless a stricter experiment policy disables it.

### `machine-proposed`

A separate frozen G2P system proposes a reading. Every proposal records:

- source-text SHA-256;
- normalized reading and reading SHA-256;
- provider ID and immutable revision;
- provider configuration digest;
- provider executable/model artifact SHA-256;
- dictionary or other pronunciation-resource SHA-256;
- pronunciation-policy digest;
- optional utterance ID and non-authoritative metadata.

An unreviewed proposal is **not** eligible for locked test. It is disabled by default even for
training. Large-scale training may opt in explicitly with
`--allow-unreviewed-machine-train`, while keeping the origin visible in every receipt.

### `machine-reviewed`

A human review record is bound to the exact proposal digest, source-text hash, proposed-reading
hash, pronunciation policy, reviewer pseudonym hash, review protocol, and review-batch manifest.
The disposition is one of:

```text
approved   proposal retained exactly
corrected  reviewed reading differs from proposal
rejected   proposal cannot be used
```

Reviewed machine readings are eligible for calibration and locked test. Approval does not turn the
reading into acoustic truth; it records only that the declared review protocol accepted it.

## Default split policy

```text
train:
  human-explicit                  allowed
  unreviewed machine proposal     denied by default; explicit opt-in available
  reviewed machine proposal       allowed

calibration:
  human-explicit                  allowed
  unreviewed machine proposal     denied
  reviewed machine proposal       allowed

test:
  human-explicit                  allowed
  unreviewed machine proposal     denied
  reviewed machine proposal       allowed
```

Calibration and test review requirements cannot be disabled through the preparation CLI. A custom
research API policy may be constructed, but such a run is not compatible with the default locked
promotion protocol.

## Input manifest

The preparation input contains the original transcript and optional human reading:

```json
{
  "schemaVersion": "1",
  "utteranceId": "recording-001-span-0004",
  "split": "test",
  "audioPath": "audio/recording-001.wav",
  "audioSha256": "<source-file-sha256>",
  "sampleRate": 16000,
  "segmentStartMs": 4120,
  "segmentEndMs": 5860,
  "transcript": "学校へ行く",
  "explicitReading": null,
  "speakerId": "speaker-opaque-001",
  "sourceId": "recording-opaque-001",
  "rightsDecision": "allow",
  "licenseId": "dataset-license-id"
}
```

The schema is exact. Rights, license, speaker, source, audio identity, sample rate, and time range
are preserved unchanged in the prepared source manifest.

## Machine proposal manifest

Machine proposals are generated separately. The preparation CLI does not load or update a model.
Each JSONL row contains a frozen provider identity:

```json
{
  "schemaVersion": "1",
  "utteranceId": "recording-001-span-0004",
  "sourceTextSha256": "<sha256-of-学校へ行く>",
  "normalizedReading": "ガッコウヘイク",
  "readingSha256": "<sha256-of-normalized-reading>",
  "pronunciationPolicyDigest": "<policy-digest>",
  "provider": {
    "schemaVersion": "1",
    "providerId": "declared-g2p-provider",
    "providerRevision": "immutable-provider-revision",
    "providerConfigDigest": "<sha256>",
    "providerArtifactSha256": "<sha256>",
    "resourceArtifactSha256": "<dictionary-or-resource-sha256>"
  },
  "metadata": {}
}
```

A changed source transcript, reading, provider artifact, dictionary, configuration, or policy
creates a different proposal digest and invalidates prior reviews.

## Review ledger

Review rows are stored separately from proposals. A review binds the exact proposal and may approve,
correct, or reject it. Reviewer identifiers should be pseudonymous hashes rather than names or
email addresses.

The review batch has an external frozen manifest digest. Review rows bind that digest; the digest is
not the hash of a file containing itself. The experiment ledger should record when and under which
protocol the batch was frozen. Software can verify identity and consistency, but cannot prove that
a review occurred before a later evaluation unless the surrounding experiment process preserves
that chronology.

## Output

The command writes two atomic files:

```text
prepared-source.jsonl
prepared-source.jsonl.reading-receipts.jsonl
```

The first is the exact input expected by `export_phonetic_features.py`. It includes the normalized
explicit reading but omits the original transcript.

The second contains no raw source transcript. It records:

- source-text hash;
- input item and input manifest digests;
- output source-item digest;
- resolved reading and origin;
- proposal, provider, and review digests where applicable;
- pronunciation and resolution policy digests;
- receipt digest.

The original preparation input remains the source of the human-readable transcript. Keeping it
separate prevents the training feature manifest from quietly becoming a second reference corpus.

## CLI

Human-reading-only preparation:

```bash
python scripts/prepare_phonetic_readings.py \
  --input ../reading-input/test.jsonl \
  --split test \
  --output ../phonetic-source/test.jsonl \
  --allow-output
```

Reviewed machine proposals:

```bash
python scripts/prepare_phonetic_readings.py \
  --input ../reading-input/test.jsonl \
  --split test \
  --machine-proposals ../reading-proposals/test.jsonl \
  --review-ledger ../reading-reviews/test.jsonl \
  --review-ledger-revision review-ledger-r1 \
  --review-protocol-revision two-pass-reading-review-r1 \
  --review-batch-manifest-sha256 <64-hex-digest> \
  --output ../phonetic-source/test.jsonl \
  --allow-output
```

Unreviewed machine readings can be enabled only for `train`:

```bash
python scripts/prepare_phonetic_readings.py \
  --input ../reading-input/train.jsonl \
  --split train \
  --machine-proposals ../reading-proposals/train.jsonl \
  --allow-unreviewed-machine-train \
  --output ../phonetic-source/train.jsonl \
  --allow-output
```

## Evaluation hygiene

Do not:

- generate or correct test readings after inspecting Semantic ASR errors;
- call an LLM and record the result as `human-explicit`;
- approve a proposal after its text, provider, dictionary, or policy changed;
- mix reviewed and unreviewed items without reporting origin counts;
- hide rejected proposals by deleting them from an already frozen evaluation manifest;
- use the same reviewer decision to justify a different proposal digest;
- treat a reviewed orthographic reading as evidence that the recording acoustically contains it.

A promotion report should stratify phone/mora and end-to-end results by reading origin. If the model
performs well only on machine-generated labels from the same G2P used at inference, report that
circularity explicitly.

## Claim boundary

This layer validates reading provenance and split policy. It does not prove that a human reading is
correct, that a G2P provider is linguistically complete, or that the trained phone/mora model
improves Japanese ASR. Those require independent annotation and locked evaluation.
