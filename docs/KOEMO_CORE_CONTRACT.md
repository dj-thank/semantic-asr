# Koemo ↔ Semantic ASR core contract

## Ownership

`semantic-asr` owns:

- N-best evidence and score semantics;
- surface/path aggregation;
- mora and semantic lattices;
- ranking, calibration, fusion, and abstention;
- selective evidence acquisition;
- observed/normalized transcript separation;
- evaluation and deployment gates.

Koemo owns:

- microphone and WASAPI loopback capture;
- AEC, channel activity, and channel policy;
- low-latency live preview;
- meeting detection and calendar title hints;
- speaker/channel presentation;
- summary, chat, export, UI, and model lifecycle.

Koemo must depend on a pinned Semantic ASR revision rather than copying the core algorithms.

## Authoritative versus preview text

```text
Windows Speech / rolling Whisper -> live preview only
Semantic ASR final pass          -> authoritative observed evidence
normalizer                        -> separate readable derivative
summary/chat                      -> downstream derivative
```

Live preview is never merged into final observed evidence merely because it arrived earlier.

## Regex correction boundary

Koemo's existing `native_correction.py` is useful for live UI vocabulary repair. It must not mutate
the canonical observed transcript. The bridge should return both:

```text
observed_text
normalized_text
observed_evidence_sha256
```

Any Koemo-specific substitution is applied only to a derivative and records its ruleset digest.

## Dual-channel operation

Mic and system channels remain independent evidence sources. Recommended flow:

1. record mic/system streams;
2. apply AEC to the mic derivative while retaining original-file hashes;
3. run channel activity and leakage diagnostics;
4. transcribe each accepted channel independently;
5. merge by timestamps and channel/speaker labels;
6. keep per-channel evidence digests in the meeting manifest.

AEC output is transformed evidence, not raw evidence. Its parameters and source hashes should be
recorded when reproducibility is required.

## Model lifecycle

Koemo may retain its warmup and idle-unload policy. Semantic ASR exposes compute effort and adaptive
throttling decisions, allowing Koemo to choose:

```text
ultra-light
cpu-quality
edge-gpu
research
```

Before loading a summary model, Koemo can release Qwen3-ASR or the acoustic verifier while retaining
cheap N-gram and linear ranker state.

## Failure behavior

- missing Semantic ASR dependency: fall back to current faster-whisper final pass and mark the
  result `legacy-unfused`;
- uncalibrated reranker: reorder only, never inject into fusion;
- unavailable second ear: continue with provisional/accepted status from existing evidence;
- model OOM: throttle expensive stages before falling back to raw ASR;
- cache mismatch: fail closed and recompute;
- rights decision other than `allow`: do not run research training/export.

## Product API shape

A minimal bridge result should include:

```json
{
  "observedText": "...",
  "normalizedText": "...",
  "decision": "accepted",
  "observedEvidenceSha256": "...",
  "selectedPosterior": 0.0,
  "segments": [],
  "diagnostics": {},
  "engine": "semantic-asr",
  "engineRevision": "..."
}
```

`selectedPosterior` is omitted or explicitly uncalibrated unless a matching calibration artifact is
loaded.

## Migration sequence

1. Add an optional `SemanticASRBridge` without changing the default Koemo path.
2. Run both engines on the same saved WAVs and compare outputs offline.
3. Use the deployment gate and meeting-domain benchmark.
4. Enable Semantic ASR behind a settings flag.
5. Make it default only after the target-machine acceptance criteria pass.
6. Remove duplicated MoraWeave/Semantic ASR core code after compatibility is demonstrated.

## Privacy

The core can run entirely locally. A local OpenAI-compatible or Ollama teacher remains loopback-only.
Remote summary/chat providers are separate Koemo opt-ins and do not change the provenance of the
observed transcript.
