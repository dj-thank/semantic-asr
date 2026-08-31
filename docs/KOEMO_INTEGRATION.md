# Koemo integration with Semantic ASR

## 1. Ownership boundary

`semantic-asr` is the canonical recognition, ranking, calibration and evidence package.

Koemo continues to own:

```text
Windows microphone and WASAPI loopback capture
AEC and channel-quality diagnostics
live preview window and tray application
recording spool/ring-buffer lifecycle
speaker diarization and meeting library
model warmup/unload and product settings
summary/chat/export UI
```

Semantic ASR owns:

```text
final decoder paths and surface candidate pool
typed evidence scores and calibration
MBR/reranking/risk control
selective re-listening and acoustic verification
accepted/provisional decision
immutable observed transcript
separately attached normalized derivative
benchmark and training contracts
```

Do not copy the v0.2 core back into a second `moraweave` implementation. Koemo depends on a pinned Semantic ASR revision.

## 2. Three transcript authorities

### Live provisional

Windows Speech or rolling Whisper may display immediate captions. They are UI feedback only:

```text
authority = live-provisional
```

They may be changed, replaced or discarded. They do not carry an observed-evidence digest and are never reused as the official transcript solely to reduce stop latency.

### Final observed

The final recording span is decoded by Semantic ASR. Its selected text is attached to candidate/path evidence:

```text
authority = final-observed
evidence_digest = Semantic ASR observed evidence digest
```

### Normalized derivative

Readability changes, punctuation, regex corrections or product-specific terminology normalization produce:

```text
authority = normalized-derivative
observed_evidence_digest = referenced final observed digest
```

The observed text remains available and immutable.

## 3. Channel contract

For every mic/system/imported span, Koemo passes:

```text
span ID
channel kind
start/end milliseconds
exact span audio SHA-256
source recording SHA-256
AEC applied flag and configuration digest
speaker label where available
capture/runtime metadata
```

The source path is not required in exported evidence. Digests permit cache validation without publishing waveform data.

Channel identity must survive candidate generation. Cross-channel text deduplication is a product display operation, not evidence deduplication.

## 4. AEC provenance and failure handling

Koemo's AEC output is a transformed acoustic observation. Semantic ASR must know:

```text
whether AEC ran
algorithm/configuration digest
reference channel identity
failure/fallback status
```

When AEC fails or returns the unchanged microphone signal, record that status explicitly. Do not present the transformed and raw channel as two independent acoustic sources unless both waveforms were retained and independently decoded.

## 5. Suggested Python integration

Conceptual stop-time flow:

```python
span = channel_span_from_samples(
    span_id=...,
    channel="microphone",
    start_ms=0,
    sample_rate=16000,
    sample_count=len(mic_audio),
    audio_sha256=hash_span(mic_audio),
    source_recording_sha256=recording_digest,
    aec_applied=aec_applied,
    aec_configuration_digest=aec_digest,
)

pool = faster_whisper_path_adapter.decode(path_request)
# add n-gram, MBR, compact reranker, risk control and selective verification
observed = semantic_pipeline_result.observed

final_event = observed_event(
    span_id=span.span_id,
    text=observed.text,
    evidence_digest=observed.evidence_sha256,
    backend="semantic-asr",
    backend_revision=semantic_asr_commit,
)

normalized = normalized_event(
    span_id=span.span_id,
    text=readability_text,
    observed_evidence_digest=observed.evidence_sha256,
    normalizer="koemo-readability",
    normalizer_revision=koemo_commit,
)
```

The actual product adapter should live in Koemo and import these contracts from the pinned package rather than duplicating dataclasses.

## 6. Stop-time latency strategy

Use a draft/verify cascade:

1. models are warmed according to Koemo settings;
2. path-preserving faster-whisper generation runs per selected channel;
3. cheap n-gram, path mass and constrained reranker stages execute first;
4. confident low-risk spans are accepted;
5. only contradiction islands invoke re-listening, Qwen3-ASR or the acoustic verifier;
6. instant summary may begin from accepted spans while provisional spans remain labelled;
7. the final summary is updated when all required evidence completes.

Never silently replace a provisional span in saved output without updating its evidence digest and decision trace.

## 7. Model memory lifecycle

Koemo currently releases model families to fit limited VRAM. Extend the lifecycle manager to reason in terms of stage requirements:

```text
Whisper path generator
compact reranker
Qwen3-ASR second ear
forced aligner
acoustic verifier
summary LLM
```

The planner estimates memory as well as latency. A selected action must fit the current budget or trigger an explicit unload/load transition whose measured cost is recorded.

Potential sequence on an 8 GB GPU:

```text
Whisper warm -> final candidates
Whisper retained only if re-listen likely
compact reranker on CPU or small GPU
optional second ear / verifier
release ASR models
load summary model
```

Compare this with keeping the verifier resident and unloading the full second ear.

## 8. Live-to-final context transfer

Live captions may contribute a **context hint**, but not evidence text. Safe transfer examples:

```text
meeting title
participant-approved proper nouns
application/window title
user-maintained vocabulary
stable prior topic
```

Unsafe transfer examples:

```text
copying live transcript into final observed text
using a live hallucinated number as a forced hotword
calling repeated live text an independent ASR vote
```

Every hint is digested and recorded in decoder provenance.

## 9. Regex/native correction migration

Koemo's current `normalize_transcript_text` performs regex replacement. Migrate its use as follows:

```text
before: faster-whisper segment -> regex -> saved official text

after:  faster-whisper raw segment -> Semantic ASR observed evidence
        observed text -> optional Koemo normalized derivative
```

Dictionary phrases may also be used as lexical features or hotword hypotheses. Their contribution is labelled lexical/contextual, not acoustic.

## 10. Dual-channel and speaker handling

Mic/system channels should be decoded independently before temporal merge. When channel dedupe decides one channel is likely echo/leakage, preserve:

```text
channel quality features
dedupe decision and threshold
AEC state
kept/rejected evidence digests
```

Diarization labels decorate span identity. They must not affect recognition correctness unless a speaker-specific language model is explicitly trained and evaluated without identity leakage.

## 11. Product-visible uncertainty

Koemo should expose at least three states:

```text
確定       accepted under calibrated policy
確認中     provisional; additional evidence is running
要確認     budget exhausted or target risk not met
```

Do not display a precise confidence percentage until it is a calibrated probability for the deployed condition. A model-authored preference or per-candidate softmax is not sufficient.

## 12. Integration tests

Required Koemo-side tests:

- live events cannot become authoritative;
- regex normalization cannot mutate observed evidence;
- normalized events reference an existing observed digest;
- channel/AEC provenance survives save/reopen/export;
- cache keys change when decoder prompts, hotwords, AEC or calibration change;
- generated candidates cannot be accepted before verification;
- unavailable GPU stages fall back without altering score semantics;
- meeting export includes observed and normalized text distinctly;
- stop-time partial results remain labelled provisional;
- pinned Semantic ASR package revision is recorded in every meeting evidence object.

## 13. Rollout

1. Integrate contracts and write evidence alongside the current transcript without changing UI selection.
2. Compare old final text with Semantic ASR single-best/path-mass outputs.
3. Enable n-gram/MBR/linear reranking behind an experimental setting.
4. Calibrate on rights-approved Koemo recordings or an external speaker-disjoint corpus.
5. Enable accepted/provisional UI.
6. Add selective second ear/verifier.
7. Remove the old direct regex-to-official-text path only after migration tests and export compatibility pass.
