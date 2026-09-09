# Public release candidate: shortest verified path

Status: local candidate preparation. This document does not approve publication,
close #40, or certify production recognition quality.

Semantic ASR preserves the recognized utterance and exports a separate readable
transcript, candidates, provenance and uncertainty. Unscored recognizers remain
provisional even when an independent recognizer agrees. Agreement is not a
calibrated probability of correctness.

## Install and run

Use Python 3.12 for the initially verified release path. From an unpacked source
checkout, install one backend:

```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe; Unix: .venv/bin/python
python -m pip install '.[asr]'
semantic-asr run recording.wav --profile cpu-ja-v1 --output-dir transcripts
```

Activate the environment or use its absolute executable paths. The default
Whisper path downloads its pinned model on first use. CPU speed depends on the
recording and hardware; no real-time guarantee is made.

For a downloaded wheel, install its filename with `[asr]` or `[sherpa]` appended.
No PyPI release is assumed by these instructions.

The offline Reazon + optional Parakeet route uses `.[sherpa]`. Download model
artifacts separately from their publisher, check their model licenses, and supply
their local directory digests. The package does not distribute these weights.

```bash
semantic-asr run recording.wav --profile reazon-ja-research-v1 \
  --reazon-model-dir /models/reazon --reazon-artifact-sha256 REAZON_SHA256 \
  --second-ear-parakeet-model-dir /models/parakeet \
  --second-ear-parakeet-artifact-sha256 PARAKEET_SHA256 \
  --output-dir transcripts
```

Directory digests use `semantic_asr.revisions.sha256_artifact`, binding filenames
and bytes; they are not archive-file hashes. Reazon/Parakeet require mono PCM16
16kHz WAV, explicit Japanese, and do not support prompts/hotwords. `reazon-ja-v1`
runs primary only. Every Reazon profile needs the explicit local adapter in Python,
or the local artifact arguments in the CLI.

## Release order and acceptance

1. Fix product blockers: installation extras, wrong backend dispatch, unscored
   acceptance, evidence and output integrity. Reproduce defects before fixing.
2. Build the wheel, install in a new environment outside the checkout, run help
   offline and a real public development WAV through the installed CLI. Verify
   output provenance, provisional state, original candidates and normalized link.
3. Package source, wheel, SHA-256 manifest, bilingual release notes and known
   limitations. Exclude campaign recordings, references, private paths, logs and
   model weights from the public candidate.
4. Review the concrete release candidate and remaining #40 prerequisites before
   publishing. CI/publication/third-party reproduction have their own receipts.

## Known limits

- The bundled research capabilities are not all validated product profiles.
- Passing software tests does not prove verbatim, conversational or semantic
  accuracy. Inspected development sets are not unseen final tests.
- Filler/repair preservation, numbers, negation, silence, long recordings and
  input-format behavior need explicit quality evidence for the claimed use case.
- Named confidence calibration is scoped; it is not a guarantee for arbitrary
  speakers/domains. Reazon/Parakeet have no calibrated acceptance in this candidate.
- Previous observed receipts predating the added `force_provisional` field require
  version-aware handling; transparent old-receipt migration is not yet validated.
- Independent installation by another person and the remaining release gates in
  issue #40 are pending. Existing GitHub draft PR #50 is not included by this plan.

## 日本語での公開方針

最初の公開候補は、音声ファイルから観測文・正規化文・根拠・未確定状態を
保存するローカルCLI/APIに絞る。新規環境のwheel実行を確認してから公開を判断する。
未採点モデルは一致しても未確定とし、100%精度・会議品質・世界最高性能を主張しない。
上記の入力条件、モデルの別途取得、旧receipt互換性、第三者再現の未完了は英語説明と同じ制限である。
