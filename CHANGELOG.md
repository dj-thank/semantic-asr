# Changelog

## 0.1.0 — 2026-08-29

- Initial public Semantic ASR foundation.
- Added immutable observed and separately linked normalized transcripts.
- Added Japanese mora normalization, timed character-to-mora merging, and Mora Shadow.
- Added calibrated five-stream candidate fusion, Grammar Honeytrap defense, selective risk, and abstention.
- Added semantic contradiction islands for numbers, dates, currency, negation, modality, entities, particles, and special mora.
- Added information-gain-per-cost evidence planning.
- Added faster-whisper CTranslate2 N-best, Qwen3-ASR second-ear, and Qwen3 Forced Aligner adapters.
- Added loopback-only Ollama and OpenAI-compatible rank-only teachers with abstention.
- Added versioned SQLite evidence cache keyed by audio, span, model, context, hotwords, and calibration.
- Added long-form transcription, overlap stitching, JSON/TXT/Markdown/SRT/VTT outputs.
- Added CER, kana-CER, mora, semantic-critical, preservation, correction, calibration, and selective metrics.
- Added optional mora/phone/boundary/accent/F0/preservation training heads.
- Added executable public-data rights registry and release CI.
