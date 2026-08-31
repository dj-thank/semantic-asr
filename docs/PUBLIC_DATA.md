# Public-data and rights protocol

“Publicly downloadable” is not equivalent to permission for training, derived-feature publication or raw redistribution.

Each exact asset/version declares:

```text
train              allow / deny / review
derive_features    allow / deny / review
redistribute_raw   allow / deny / review
export_speaker_id  allow / deny / review
```

`review` blocks the requested operation.

## Candidate sources

### Common Voice

Pin a named release and locale manifest. Pseudonymize client identifiers with a secret HMAC before indexing. Never export the original identifier.

### ReazonSpeech

Review the exact release and source-program conditions. Do not infer one licence for every repository component.

### SaSLaW

Review exact download, training and speaker-privacy conditions. Keep learner evaluation speaker-disjoint.

### JMdict

Preserve EDRDG attribution and exact snapshot metadata. Prefer non-reconstructable derived lexical features rather than redistributing source XML.

### 青空文庫

Rights vary by work. Every work needs an individual provenance and rights record.

### Project recordings

Consent must separately cover research, training, derived-feature publication, raw redistribution and withdrawal.

## Privacy

- no raw audio in Git
- no model weights in Git
- no absolute input paths in exported transcript JSON
- HMAC speaker pseudonyms
- evidence cache stores no waveform
- deletion by asset/speaker lineage
- SHA-256 manifests and split assignments
- duplicate/speaker leakage checks before evaluation
