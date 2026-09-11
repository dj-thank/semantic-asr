# Joint Japanese phone/mora CTC training

## Scope

This pipeline trains the shared phone/mora CTC head introduced for Semantic ASR v0.3. It consumes
**frozen acoustic frame features**. It does not silently choose or download an encoder, infer target
labels from evaluation references, or promote the resulting model into the default transcription
path.

The intended evidence flow is:

```text
original recording
  -> frozen audio encoder and exact layer
  -> digest-verified frame feature array
  -> shared bottleneck
       |- phone CTC posterior
       `- mora CTC posterior
  -> candidate pronunciation likelihood
  -> held-out utility calibration
  -> ambiguity-only document lattice evidence
```

Phone and mora posteriors are produced from audio before a candidate text is inspected. They remain
separate score domains.

## Frozen label inventories

A `PhoneticLabelInventory` records:

- `phone` or `mora` kind;
- exact ordered label list;
- CTC blank symbol and index;
- inventory revision;
- SHA-256 of the source label manifest.

Target arrays contain integer IDs into the frozen inventory and may not contain the blank label.
Changing label order creates a different inventory digest and invalidates existing feature
manifests, weights, and calibration profiles.

## Feature manifest

Each split is one JSONL file. Every row has an exact schema:

```json
{
  "schemaVersion": "1",
  "utteranceId": "recording-001-segment-0004",
  "split": "train",
  "featurePath": "features/recording-001-segment-0004.npy",
  "featureSha256": "...",
  "frameCount": 417,
  "featureDimension": 768,
  "featureDtype": "float32",
  "phoneTargets": [12, 5, 27],
  "moraTargets": [41, 9],
  "phoneInventoryDigest": "...",
  "moraInventoryDigest": "...",
  "speakerId": "speaker-opaque-0031",
  "sourceId": "corpus-opaque-07",
  "sourceAudioSha256": "...",
  "featureRevision": "encoder@commit:layer-9:frame-policy-v1",
  "rightsDecision": "allow",
  "licenseId": "dataset-license-id"
}
```

Rules:

- feature paths are relative to the manifest directory and cannot contain `..`;
- only `.npy` arrays are accepted;
- NumPy loads with `allow_pickle=False`;
- file SHA-256, shape, dtype, and finite values must match;
- all rows in all splits use one feature dimension, feature revision, and label inventories;
- only `rightsDecision=allow` rows with a non-empty license ID are accepted;
- resource policies cap items, frames, dimensions, targets, and total feature cells.

## Split isolation

`validate_phonetic_split_disjointness()` rejects overlap across train, calibration, and test for:

- utterance ID;
- source-audio SHA-256;
- speaker ID;
- source/corpus ID;
- feature SHA-256.

The three manifest files themselves must differ. This is stricter than random row splitting because
neighboring segments from one recording, one speaker, or one corpus source can otherwise leak
strong acoustic and lexical information.

## CTC feasibility

A target of length `N` does not always fit in `N` frames. Consecutive repeated labels require an
intervening CTC blank. The minimum number of frames is:

```text
len(target) + number of adjacent repeated labels
```

Manifests and runtime batches fail before optimization when the available frame count is below this
minimum. `zero_infinity=True` is retained as numerical protection, not as permission to train on
impossible alignments.

## Model

`JointPhoneMoraCTCHead` uses:

```text
frozen frame features
  -> LayerNorm
  -> Linear
  -> GELU
  -> Dropout
       |- phone projection
       `- mora projection
```

The two heads share acoustic evidence but retain independent vocabularies and CTC losses. The loss
is:

```text
phone_weight * phone_CTC
+ mora_weight * mora_CTC
+ optional all-blank-collapse hinge
```

The blank penalty is zero by default. It only penalizes pathological posterior collapse above a
high blank-mass threshold.

## Training CLI

Install the optional training dependencies and run:

```bash
python -m pip install -e '.[train]'

python scripts/train_joint_phonetic_head.py \
  --config ../phonetic/config.json \
  --train ../phonetic/train.jsonl \
  --calibration ../phonetic/calibration.jsonl \
  --test ../phonetic/test.jsonl \
  --rights-registry-sha256 <64-hex-digest> \
  --revision joint-phone-mora-r1 \
  --weights ../phonetic/joint-phone-mora-r1.safetensors \
  --artifact ../phonetic/joint-phone-mora-r1.json \
  --report ../phonetic/joint-phone-mora-r1-report.json
```

The configuration has an exact schema and binds the feature dimension, hidden dimension, encoder
identity/revision/artifact, both label inventories, loss weights, dropout, and architecture digest.

## Calibration and locked test

After training:

1. the calibration split fits one phone and one mora sequence-likelihood threshold at a declared
   target true-accept rate;
2. the test split uses those frozen thresholds without refitting;
3. the report records greedy phone error rate, mora error rate, candidate discrimination AUC, and
   hard-negative false-accept rate.

AUC uses the correct target sequence versus a deterministic same-length label substitution. This is
a reproducible baseline, not a complete pronunciation-confusion benchmark. Production promotion
requires natural Japanese hard negatives, number/negation/entity slices, microphone shift, and
speaker/source shift.

Phone and mora error rates are non-negative and can exceed `1.0` when insertions outnumber target
labels. AUC and false-accept rates remain bounded to `[0, 1]`.

## Serialization

Weights use safetensors only. The file metadata records the head configuration digest and
architecture. The JSON envelope records:

- head configuration and digest;
- train/calibration/test manifest digests;
- speaker/source-disjoint declaration;
- rights-registry digest;
- optimizer configuration and epoch losses;
- calibration thresholds and false-accept rates;
- locked-test metrics;
- weights SHA-256;
- `JointPhoneticArtifact` digest;
- complete envelope digest.

Pickle, arbitrary Python objects, optimizer state, and implicit downloaded model state are not
trusted.

## Runtime boundary

`posterior_configs_from_artifact()` creates distinct frozen phone and mora runtime configurations
from one verified joint artifact. A runtime backend must additionally bind the same frozen feature
encoder and layer. The trained head is not a raw-audio model by itself.

At document-lattice runtime, `SelectivePhoneticSpanProposalProvider` should:

- run only on selected contradiction spans;
- retain the full-recording SHA-256 and exact sample range;
- give the posterior sequence the canonical extracted-clip SHA-256;
- attach the phone/mora model and calibration digests;
- reject a proposal whose posterior came from another clip;
- keep generated proposals provisional unless the document policy explicitly accepts them.

## Promotion requirements

A successful tiny training smoke or low isolated PER is not enough. Promotion requires a locked,
speaker/source-disjoint Japanese experiment measuring:

- first-pass and final CER;
- phone and mora candidate discrimination;
- N-best miss recovery;
- number, date, currency, negation, modality, entity, repair, and filler errors;
- first-pass-exact false correction;
- accepted coverage and accepted error;
- latency, peak memory, and energy on target devices;
- ablations for phone-only, mora-only, discrete-unit-only, and combined evidence.

Until these gates pass, the model remains an opt-in research artifact.
