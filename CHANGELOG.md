# Changelog

## Unreleased — 2026-09-02

- Fixed the direct CTranslate2 generate path: log-mel features are now padded to one 30 s Whisper window before `encode()`, matching `faster_whisper.transcribe`. Unpadded short clips looped on every beam; on 20 ReazonSpeech test clips the utterance-mean CER fell from 5.70 to 0.30 (large-v3-turbo int8, CPU).
- Added `LoopGuardConfig`: duration-aware token budget, per-path compression-ratio / repeated n-gram / character-budget / log-probability degeneracy evidence, staged sampling fallback in separate score domains, and timestamp-enabled prompts by default (`--without-timestamps` restores the old prompt).
- Added optional sampled-candidate enrichment (`--extra-samples`) for sample-based Semantic MBR.
- Added `corpus_cer` and `lenient_corpus_cer` (punctuation/symbol-stripped, length-weighted) to benchmark reports for comparability with published Japanese ASR numbers; strict utterance-mean CER remains primary.
- Ranker training now skips utterances with a single surviving candidate instead of aborting.
- Added `scripts/prepare_public_manifest.py` (Hugging Face public test sets to rights-annotated manifests) and `scripts/run_real_audio_pipeline.py` (partition → train → calibrate → apply → benchmark through the CLI).
- Pinned eight 2025–2026 primary sources and three falsifiable translations in the research registry; see `docs/RESEARCH_2026-09-02.md`.

## 0.2.0 — 2026-08-31

- Preserved all same-surface decoder paths and aggregated probability mass with score-domain-safe `logsumexp` instead of strongest-path-only deduplication.
- Added explicit raw score, log-likelihood, logit, preference, and calibrated-probability semantics.
- Added Semantic Minimum Bayes Risk decoding over surface, mora, meaning-critical, and preservation losses.
- Added adaptive candidate-set size from posterior mass, selective risk, semantic criticality, and diversity.
- Added a conservative fusion–MBR cascade that requests evidence on disagreement instead of silently rewriting observed text.
- Added dependency-free pairwise ranker training with feature normalization, manifest hashing, metrics, and reproducible artifacts.
- Added Japanese hard-negative generation for negation, particles, numbers, special mora, fillers, and phonetic neighbors.
- Added optional raw-logit CrossEncoder/ModernBERT and Qwen3-Reranker adapters with held-out calibration hooks.
- Added a query-selected acoustic candidate verifier with bounded acoustic/context/mora branches and balance regularization.
- Added quantile-balanced sparse evidence routing, empirical reward state, and bounded residual branch mixing.
- Added a keyed, hashed offline-teacher next-token probability cache with longest-suffix backoff.
- Added the `cascade`, `synthetic-data`, `train-ranker`, `lm-cache-build`, `research-smoke`, and `transcribe-v2` commands while retaining v0.1 CLI compatibility.
- Added Qwen3.8-Flash-Next, Kimi K3, GLM-5.3, recent ASR fusion, cached-LM, Adaptive GER, MBR, retrieval, and guarded-GER research translations.
- Added an explicit Koemo integration contract that keeps regex normalization outside immutable observed evidence.
- Added model-free optimization tests and CPU PyTorch forward/backward tests for the new acoustic verifier.

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
