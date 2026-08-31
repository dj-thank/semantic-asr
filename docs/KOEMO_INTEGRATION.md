# Koemo integration contract

`dj-thank/semantic-asr` is the authoritative recognition core. `dj-thank/koemo` remains the Windows meeting product and capture/runtime host.

## 1. Ownership boundary

### Koemo owns

- microphone and WASAPI loopback capture;
- acoustic echo cancellation;
- temporary audio spooling and bounded live ring buffers;
- Windows-native or rolling-Whisper provisional live captions;
- channel activity and channel selection policy;
- diarization orchestration;
- tray/UI/settings/library/export;
- model warmup, idle unload, GPU/CPU fallback;
- user-visible progress and cancellation.

### Semantic ASR owns

- final authoritative N-best generation;
- decoder path and model provenance;
- surface/mora candidate pool;
- reranking, MBR, calibration, risk, abstention;
- selective re-listening and second-ear policy;
- immutable observed transcript evidence;
- separately linked normalized transcript;
- recognition metrics and benchmark manifests.

## 2. Required invariant

Koemo currently applies `normalize_transcript_text` while reading final faster-whisper segments. That path must not be used for the authoritative Semantic ASR observation.

The integration must preserve:

```text
raw ASR candidate text
  -> Semantic ASR observed evidence
  -> optional Koemo/readability normalization
```

Regex or dictionary replacements may produce normalized text only. They must not mutate candidate text before the observed evidence hash is created.

## 3. Audio handoff

Koemo writes or retains separate mono streams:

```text
mic.wav
system.wav
```

For each active channel, hand Semantic ASR:

```json
{
  "audioPath": "local temporary or saved WAV path",
  "channel": "mic | system",
  "sampleRate": 16000,
  "captureStart": "local timestamp",
  "aecApplied": true,
  "sourceFileSha256": "...",
  "channelPolicy": "auto_dedupe | all_active",
  "captureDiagnostics": {}
}
```

The absolute path is runtime-private and is not exported. The audio SHA-256 and non-sensitive capture diagnostics are retained in evidence.

## 4. Authoritative final pass

Recommended default:

```text
Koemo stop
  -> finish audio writes / AEC
  -> determine active channels
  -> semantic-asr transcribe-v2 per channel
  -> preserve each channel's observed evidence
  -> time-align / diarize
  -> merge observed channel segments
  -> attach normalized/readable layer
  -> summarize from normalized text while retaining observed citations
```

Live captions are never recycled as the authoritative final transcript unless an explicit degraded-mode contract marks every reused span as provisional and records its source.

## 5. Model lifecycle

For an 8 GB GPU or CPU-first machine:

```text
recording:
  live ASR only

stop-time final:
  faster-whisper + lightweight ranker

uncertain spans:
  acoustic verifier or Qwen3-ASR second ear

summary:
  release speech models unless keep_warm=true
  load local summarizer
```

Recommended tiers:

| Koemo setting | Semantic ASR tier |
|---|---|
| Fast CPU | path pool + Semantic MBR + linear ranker |
| Balanced CPU | + ModernBERT-Ja CrossEncoder |
| Small GPU | + Qwen3-Reranker-0.6B |
| Accuracy GPU | + Qwen3-ASR second ear and forced aligner |
| Research | + acoustic verifier / cached teacher / guarded GER |

## 6. API shape

Koemo should depend on Semantic ASR as a package rather than copying modules.

Python entry point:

```python
from semantic_asr.advanced_adapters import (
    AdaptiveRerankingAdapter,
    PathPreservingFasterWhisperAdapter,
)
from semantic_asr.longform import SemanticASRTranscriber
```

CLI entry point:

```bash
semantic-asr transcribe-v2 recording.wav \
  --ranker-backend linear \
  --ranker-profile ~/.koemo/models/asr-ranker.json \
  --output-dir ~/.koemo/semantic-asr/run-id
```

Koemo must pin:

```text
semantic-asr package version
wheel SHA-256
ranker profile/checkpoint digest
calibration digest
ASR model revision
runtime versions
```

## 7. Result storage

Koemo meeting records should retain:

```text
observed transcript
evidence digest
normalized transcript
normalization mode
uncertainty spans
provisional/accepted status
model and calibration digests
per-channel provenance
```

Summary and chat answer citations should point to observed segment IDs/evidence digests, even when display text uses the normalized layer.

## 8. Migration plan

1. Add Semantic ASR as an optional local dependency in Koemo.
2. Add a feature flag: `final_transcriber=legacy|semantic_asr_v2`.
3. Feed raw final Whisper text into Semantic ASR; bypass `normalize_transcript_text` before evidence creation.
4. Store Semantic ASR JSON alongside current Koemo Markdown.
5. Compare both paths on the same local meeting fixtures.
6. Promote v2 only after latency, channel merge, packaging, and regression gates pass.
7. Remove duplicated MoraWeave core after the standalone package is stable.

## 9. Acceptance gates

- existing Koemo capture/live/AEC tests remain green;
- no network call unless explicitly configured;
- absolute paths and secrets absent from exported evidence;
- final observed transcript is unchanged by regex normalization;
- legacy and v2 outputs can coexist during migration;
- unsigned public Windows binaries are never described as production releases;
- package unload/reload does not leak GPU memory across recording and summary stages.
