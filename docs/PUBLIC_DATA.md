# Public-data and rights protocol

“Publicly downloadable” is not equivalent to permission for training, derived-feature
publication or raw redistribution.  Keep the evidence and the operation separate.

Each exact asset/version declares:

```text
train              allow / deny / review
derive_features    allow / deny / review
redistribute_raw   allow / deny / review
export_speaker_id  allow / deny / review
```

`review` and `deny` block the requested operation.  An ignored file is still local
data; `.gitignore` does not grant a licence or publication permission.

## Preparing a local manifest

The preparation script downloads a pinned Hugging Face test split, decodes and resamples
it, writes 16 kHz mono WAV files, and emits a JSONL manifest containing the reference and
absolute `audioPath`.  Those are raw/reference-bearing outputs, so the script refuses to
run unless the operator explicitly supplies `--allow-raw-export`.  The destination must
resolve outside the repository checkout, including through symlinks.

Install the script-only dependencies with:

```bash
python -m pip install -e '.[public-data]'
```

The extra declares `datasets`, `numpy`, `scipy`, and `soundfile`.  A minimal invocation is:

```bash
python scripts/prepare_public_manifest.py reazonspeech-test \
  --output-dir ../semantic-asr-public-data/reazon \
  --dataset-revision dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f \
  --allow-raw-export
```

Without `--rights-decision`, rights remain `review` unless the dataset key and revision
match an exact public asset supported by the repository.  For every other asset, use an
explicit `--rights-decision allow` only after the exact terms have been reviewed.  A
custom registry can be enforced as follows:

```bash
python scripts/prepare_public_manifest.py reazonspeech-test \
  --output-dir ../semantic-asr-public-data/reazon \
  --rights-registry path/to/rights-registry.json \
  --rights-asset-id reazonspeech-release \
  --rights-decision allow \
  --allow-raw-export
```

When a registry is supplied, both `derive_features` and `redistribute_raw` must be
`allow`; a CLI `allow` cannot override a registry `review` or `deny`.  The manifest keeps
the registry asset ID in `rightsAssetId` and the operation decision in `rightsDecision`.

The public test sets used here carry no speaker labels, so `groupId` falls back to the
sample identifier and the resulting split is not speaker-disjoint.  Record that
limitation with any quality claim.

## Probe output

`scripts/probe_second_ear.py` is metadata-only by default.  Its JSONL contains sample IDs,
model/revision, duration, elapsed time, and hypothesis count, but not references or raw
hypothesis text.  If transcript inspection is necessary for local research, pass
`--local-research-output` and write to an external directory; never publish that output.

```bash
python scripts/probe_second_ear.py \
  ../semantic-asr-public-data/reazon/manifest.jsonl \
  --output ../semantic-asr-public-data/reazon/second-ear.jsonl

python scripts/probe_second_ear.py \
  ../semantic-asr-public-data/reazon/manifest.jsonl \
  --output ../semantic-asr-public-data/reazon/second-ear-local.jsonl \
  --local-research-output
```

Probe output paths are also rejected when they resolve inside the checkout.  Keep all
WAVs, reference-bearing manifests, local probe logs, and model outputs in an external
local-research directory; representative `public-data/`, `data/public/`, and
`data/reazon/` paths are ignored as a last-resort guard, not as authorization.

## Candidate generation and the post-candidate pipeline

Candidate JSONL produced from a reference-bearing manifest includes the reference and N-best
hypotheses.  Keep it, the ranker manifest, and every post-candidate pipeline artifact in the
same external local-research directory; an ignored `runs/` path inside the checkout is not a
safe destination.  The pipeline requires an explicit `--allow-raw-export` authorization for
reference-bearing input/output and rejects an output directory that resolves into the checkout
(including through symlinks) or to a filesystem root.

```bash
semantic-asr generate-candidates \
  ../semantic-asr-public-data/reazon/manifest.jsonl \
  --output ../semantic-asr-public-data/reazon/all-candidates.jsonl \
  --ranker-output ../semantic-asr-public-data/reazon/all-ranker.jsonl \
  --model large-v3-turbo \
  --model-revision EXACT_MODEL_REVISION

python scripts/run_real_audio_pipeline.py \
  --candidates "../semantic-asr-public-data/reazon/all-candidates.jsonl" \
  --output-dir ../semantic-asr-public-data/reazon/pipeline \
  --allow-raw-export
```

Before writing any derived output, the pipeline checks each candidate row for rights evidence.
`rightsDecision` must be `allow` and a non-empty `license`, `licenseId`, or
`generation.licenseId` must be present.  Missing, `review`, and `deny` evidence fails closed.
Reference-free metadata-only rows retain the existing safe command path and do not require the
raw-export flag; an explicit rights field on such a row is still validated.

## Candidate sources

### Common Voice

Pin a named release and locale manifest.  Pseudonymize client identifiers with a secret
HMAC before indexing.  Never export the original identifier.

### ReazonSpeech

Review the exact release and source-program conditions.  Do not infer one licence for
every repository component.

### SaSLaW

Review exact download, training and speaker-privacy conditions.  Keep learner evaluation
speaker-disjoint.

### JMdict

Preserve EDRDG attribution and exact snapshot metadata.  Prefer non-reconstructable
derived lexical features rather than redistributing source XML.

### 青空文庫

Rights vary by work.  Every work needs an individual provenance and rights record.

### Project recordings

Consent must separately cover research, training, derived-feature publication, raw
redistribution and withdrawal.

## Privacy

- no raw audio in Git
- no model weights in Git
- no absolute input paths in exported transcript JSON
- HMAC speaker pseudonyms
- evidence cache stores no waveform
- deletion by asset/speaker lineage
- SHA-256 manifests and split assignments
- duplicate/speaker leakage checks before evaluation
