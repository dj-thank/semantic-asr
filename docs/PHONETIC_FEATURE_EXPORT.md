# Frozen Japanese phonetic target and feature export

## Purpose

The joint phone/mora trainer consumes frozen frame features and integer target IDs. This export
pipeline produces those artifacts without asking an LLM to guess pronunciation and without hiding
which recording, span, encoder, layer, or mapping created them.

The complete preparation path is:

```text
rights-approved source recording
+ explicit kana reading fixed before evaluation
        |
        |- exact recording file SHA-256
        |- exact source/sample range
        |- deterministic mora segmentation
        |- frozen kana-to-phone mapping
        `- pinned encoder/revision/layer
                        |
                        v
            atomic, digest-verified .npy feature
            exact phone/mora target IDs
            per-item provenance receipt
            trainer-compatible JSONL manifest
```

This is derived training data. Export requires an explicit authorization flag and does not imply
that the resulting head improves ASR.

## Pronunciation boundary

`JapanesePronunciationPolicy` accepts an **explicit hiragana or katakana reading**. It normalizes
hiragana to katakana, removes only declared punctuation when configured to do so, splits Japanese
moras, and applies a fixed mapping.

Examples:

```text
がっこう -> ガ / ッ / コ / ウ -> g a q k o u
キョー   -> キョ / ー       -> ky o :
コン     -> コ / ン         -> k o N
```

The v1 symbols reserve:

```text
ン -> N
ッ -> q
ー -> :
```

The policy includes basic kana, voiced and semi-voiced kana, yoon, and explicitly enumerated
foreign-sound combinations. Unsupported units fail closed. Kanji, Latin spelling, accent,
devoicing, contextual nasal place, geminate realization, and dictionary pronunciation are never
guessed.

Therefore this input is valid:

```json
{"reading": "セマンティックエーエスアール"}
```

This input is deliberately invalid:

```json
{"reading": "Semantic ASR"}
```

A separate, versioned text-to-reading system may be used upstream, but its result must be frozen and
reviewed as the explicit reading. Its model/revision and uncertainty should remain separate
provenance rather than being hidden inside this exporter.

## Source manifest

Each source JSONL row has an exact schema:

```json
{
  "schemaVersion": "1",
  "utteranceId": "recording-001-span-0007",
  "split": "train",
  "audioPath": "audio/recording-001.wav",
  "audioSha256": "<file-bytes-sha256>",
  "sampleRate": 16000,
  "segmentStartMs": 15320,
  "segmentEndMs": 17180,
  "reading": "マダマージシマセン",
  "speakerId": "speaker-opaque-0042",
  "sourceId": "recording-opaque-001",
  "rightsDecision": "allow",
  "licenseId": "dataset-license-id"
}
```

Important distinctions:

- `audioSha256` is the SHA-256 of the complete encoded source file;
- `sourceId` groups all spans originating from the same recording/source;
- the extracted PCM clip receives a separate canonical audio SHA-256;
- the trainer manifest uses the clip hash as `sourceAudioSha256`;
- split validation also uses `sourceId`, so spans from one recording cannot cross splits.

Paths must be relative to the source manifest directory and cannot contain `..`. Audio is decoded
as explicitly mono. The exporter does not downmix or resample. The declared, decoded, and frozen
encoder sample rates must all agree.

Only rows with `rightsDecision=allow` and a non-empty license ID are accepted. Public availability
alone does not grant permission to export or train on derived features.

## Encoder configuration

The CLI uses an exact JSON configuration. The encoder section freezes:

- model ID;
- exact immutable revision or artifact digest policy;
- optional model artifact SHA-256;
- hidden-state layer index;
- input sample rate;
- expected feature dimension;
- frame stride used later for posterior timestamps;
- execution device;
- whether model files must already exist locally.

`TransformersAudioFeatureBackend` loads with `trust_remote_code=False`. The output hidden-state
matrix must match the frozen layer and dimension. The source audio and feature configuration digests
are attached to the matrix before export.

Device/framework precision can affect floating-point results. The export metadata therefore records
backend runtime provenance in addition to the model configuration. A different runtime identity
creates a different export run and feature revision.

## Output artifacts

For every source row, the exporter writes:

```text
features/<deterministic-id>.npy
features/<deterministic-id>.receipt.json
```

The `.npy` file:

- is written with `allow_pickle=False`;
- contains only finite 2D floating-point values;
- is atomically promoted from a temporary file;
- has a recorded SHA-256, shape, and dtype.

The sidecar receipt records:

- source item and source manifest digests;
- complete recording file SHA-256;
- canonical extracted-clip SHA-256;
- exact sample start/end and sample rate;
- pronunciation target and inventory digests;
- encoder/config/runtime digests;
- feature matrix and file digests;
- final relative feature path and feature revision.

The output JSONL is directly consumable by `train_joint_phonetic_head.py` and contains no raw audio
or reference spelling. It contains target IDs, split identities, rights metadata, feature digest,
and clip identity.

## Atomic resume

The exporter writes:

```text
<output>.jsonl.partial
<output>.jsonl.partial.meta.json
```

Each completed row is flushed, optionally `fsync`-ed, and bound to a feature and sidecar receipt. On
resume, the partial manifest must be an exact prefix of the source manifest. Existing feature and
receipt digests are revalidated before inference continues. A mismatched source manifest, encoder,
pronunciation policy, output config, inventory, or feature revision stops rather than overwriting
prior work.

When all rows succeed, the partial JSONL is atomically promoted and a final export envelope is
written. A completed output is a verified no-op on repeated execution with the same run identity.

## CLI

Install export dependencies:

```bash
python -m pip install -e '.[phonetic-export]'
```

Then run each predeclared split separately:

```bash
python scripts/export_phonetic_features.py \
  --config examples/phonetic_feature_export.config.json \
  --source ../phonetic-source/train.jsonl \
  --split train \
  --output ../phonetic-derived/train.jsonl \
  --allow-derived-export
```

Repeat with distinct `calibration` and `test` source manifests. The later training loader rejects
speaker, source recording, source audio, utterance, or feature overlap across splits.

## Configuration example

```json
{
  "schemaVersion": "1",
  "encoder": {
    "modelId": "organization/frozen-audio-encoder",
    "modelRevision": "<40-character-commit>",
    "modelArtifactSha256": null,
    "revisionPolicy": "exact-commit",
    "layerIndex": 9,
    "sampleRate": 16000,
    "featureDimension": 768,
    "frameStrideMs": 20.0,
    "device": "cpu",
    "localFilesOnly": true
  },
  "pronunciationPolicy": {
    "schemaVersion": "1",
    "blankSymbol": "<blk>",
    "nasalSymbol": "N",
    "sokuonSymbol": "q",
    "longVowelSymbol": ":",
    "ignorePunctuation": true,
    "mappingRevision": "ja-kana-mora-phone-v1"
  },
  "export": {
    "schemaVersion": "1",
    "featureDtype": "float32",
    "featureSubdirectory": "features",
    "maximumCachedRecordings": 2,
    "fsyncEachRow": true
  },
  "resources": {
    "maximumItems": 2000000,
    "maximumReadingCharacters": 20000,
    "maximumSegmentDurationMs": 120000,
    "maximumTotalAudioSamples": 100000000000,
    "maximumRecordingSamples": 2000000000
  }
}
```

The repository example intentionally contains a placeholder model identity. It is not a recommended
encoder or a claim about the best Japanese layer.

## Evaluation hygiene

The source split must be frozen before feature extraction. Do not:

- derive a reading from the evaluation reference after inspecting ASR errors;
- tune the kana-to-phone map on locked test errors;
- put spans from one recording or speaker in different splits;
- choose encoder layers from test PER/MER;
- replace unsupported readings with model guesses silently;
- reuse the test split to fit CTC acceptance thresholds.

Record negative results. A feature encoder/layer is promoted only after speaker/source-disjoint
phone, mora, candidate-discrimination, semantic-critical, and end-to-end ASR evaluation.

## Claim boundary

Successful export proves only that the derived data is reproducible and auditable. It does not prove
that the mapping is phonetically complete, that the encoder exposes optimal Japanese units, or that
the joint head improves CER. Those are locked experimental questions.
