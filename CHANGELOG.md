# Changelog

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
