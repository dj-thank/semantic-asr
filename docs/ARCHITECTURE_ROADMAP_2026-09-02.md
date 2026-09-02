# Semantic ASR architecture and technology roadmap — 2026-09-02

## Decision

Do not keep adding global reranker features to the current linear fusion. On the locked
ReazonSpeech test split, learned rankers, Semantic MBR, Wikipedia and in-domain n-grams,
Qwen3-ASR-0.6B agreement, Qwen candidate insertion, and calibration-selected Qwen gating are
neutral or harmful. The next architecture should be an evidence-aware recognizer with three
separate concerns:

1. a stable primary recognizer that preserves the observed transcript;
2. selective, provenance-bound evidence modules that may abstain;
3. an evaluation module whose reference-aware diagnostics cannot enter runtime selection.

The recommended external Interface is the common-caller design, backed internally by justified
ports and adapters:

```python
result = transcribe(
    "meeting.wav",
    profile="cpu-ja-v1",
    catalog=context_catalog,  # optional, frozen and exogenous
)
```

`TranscriptResult` should contain the observed and normalized transcripts, decision/abstention,
immutable provenance, and runtime diagnostics. A private `EvidenceAwareRecognizer` implementation
hides windowing, path pooling, calibration, optional retrieval, second-ear routing, fusion, and
evidence receipts. Streaming should get a session Interface only when a second real streaming
adapter exists; it should not enlarge the first offline Interface speculatively.

## Why this seam

Deleting this Module would force every caller to recreate model revision checks, Whisper
windowing, score-domain rules, context freezing, second-ear gates, calibration, observed versus
normalized transcript binding, and evidence hashes. That gives the Module Depth, Leverage, and
Locality. The Interface stays small; backend-specific knobs move into immutable named profiles.

Dependencies behind the seam:

- in-process: candidate pooling, score semantics, calibration, fusion, MBR, abstention;
- local-substitutable: frozen JSON/SQLite context catalog, cache, in-memory test adapters;
- true external adapters: faster-whisper/CTranslate2, Qwen3-ASR, sherpa-onnx, neural scorers;
- provisioning, not runtime: Hub snapshot download, model conversion, catalog construction.

Only seams with two justified adapters should become public ports. Today that means primary versus
second-ear decoders and frozen-file versus in-memory context catalogs. Clock, budget, audit sink,
and streaming remain internal seams until a second implementation is actually needed.

## Technology selection

| Need | Selection | Status and evidence | Stop / promotion condition |
|---|---|---|---|
| CPU offline primary | faster-whisper large-v3-turbo, CTranslate2 int8 | **keep**; only stack measured end-to-end here (`0.2518` strict utterance mean, `0.1162` lenient corpus) | replace only on same immutable manifest with strict and non-boundary slices non-inferior |
| High-quality independent ear | Qwen3-ASR-1.7B | **experiment on GPU**; the 1.7B AuT+LLM model is stronger than 0.6B in the official report, supports Japanese and unified offline/streaming | same 600 clips, exact revision, first measure standalone CER; enter fusion only if at least non-inferior and complementary |
| Current Qwen 0.6B | single-hypothesis contradiction signal | **do not promote**; all-on insertion is harmful and the best calibration-selected gate changed three rows without strict gain | retain probe infrastructure only; no default weight or candidate authoring |
| Japanese CPU/edge adapter | ReazonSpeech-k2-v2 via sherpa-onnx (159M Zipformer RNN-T, ONNX) | **benchmark as a deployment adapter**, not as a claimed quality upgrade; architecturally independent and portable | test exact model on the immutable 600 clips plus latency/RAM; do not infer from vendor benchmark normalization |
| Very small edge lane | Moonshine tiny Japanese (27M) | **defer to device profile**; attractive size, but not evidence for better reranking or broadcast accuracy | require device RTF/RAM and the same strict/lenient/entity slices |
| Generic count LM | Wikipedia or train-reference n-gram | **reject**; both fail the locked test, including calibration-only model selection | reopen only with a genuinely matched external domain corpus and a preregistered cache-coverage hypothesis |
| Cached language evidence | short-context cached causal-LLM probabilities (`K<=8`) | **next diagnostic**; existing `cached_lm.py` matches the 2026 method, but the paper is flat/harmful for Whisper-large-v3 and stronger gains occur on weaker recognizers | first run direct teacher scoring as an upper-bound diagnostic; build a cache only if calibration and locked test improve without boundary/length regression |
| Entity recovery | frozen exogenous catalog + compact phonetic/semantic retrieval + no-bias gate | **primary product direction**; dynamic vocabulary, Deferred NAM, CLAR and RECOVER support phrase retrieval and constrained correction | catalog must pre-exist test references; report entity recall/CER, non-entity CER, distractor FP and abstention separately |
| Confidence | small CEM over beam score/rank, margin, entropy, degeneracy and stage features | **data-blocked**; current ranker is dominated by rank and the calibration set is too small for robust tail guarantees | collect a larger speaker/domain-disjoint calibration/audit set; require ECE, risk-coverage, deletion and OOD slices |
| Boundary handling | VAD/segmentation adapter plus diagnostic contiguous alignment | **diagnostic shipped; runtime fix later** | never select candidates with reference alignment; validate a reference-free segmenter on concatenated audio and a clean-boundary corpus |
| Free-form LLM GER | none in the observed transcript path | **reject as default**; it can author unsupported text | allow only entity-scoped proposals from a frozen catalog with deterministic verify/apply and full abstention |

## Latest architecture implications

Qwen3-ASR uses an AuT audio encoder, projector, and causal LLM with dynamic attention windows;
the official high-level wrapper returns one transcript (plus language and optional transcript-
conditioned alignment), not a calibrated N-best confidence contract. Therefore it belongs behind a
decoder Adapter and must not impersonate the CTranslate2 score Interface.

Retrieval-based contextual ASR is converging on a common pattern: retrieve a small phrase set,
inject it as context, and explicitly train or gate a `NO_BIAS` path. Many published headline gains
use oracle or reference-derived lists; those are upper bounds, not deployment evidence. This repo
should require an exogenous catalog digest and a distractor-only arm before an entity feature can
be enabled.

Cached LLM probability retrieval is the closest replacement for the failed n-gram: offline teacher
probabilities, exact short-context lookup, shorter-context backoff, and optional selective scoring.
However, its own results show the operating regime matters: it helps weaker Whisper variants much
more than Whisper-large-v3. The direct-teacher diagnostic is therefore the cheapest honest gate.

For boundaries, forced alignment estimates timestamps for a supplied transcript; it does not prove
the transcript correct. Reference-aware realignment can explain annotation mismatch but must remain
outside candidate selection. Runtime segmentation should instead use audio/VAD/timestamp evidence
and be evaluated on concatenated sessions where the expected boundaries are independently fixed.

## Data and evaluation architecture

Keep the current locked test untouched. Add two new datasets before training a larger decision
model:

1. `calibration-audit`: at least a few thousand speaker/domain-disjoint clips, including deletion,
   noise, dialect, named-entity and no-speech strata;
2. `context-bias`: an exogenous catalog frozen before audio evaluation, with relevant-context,
   distractor-only, homophone, catalog-missing and no-context arms.

Primary gates remain strict corpus CER and strict utterance-mean CER on identical audio/reference
boundaries. Report lenient corpus CER for comparability, entity/number/negation errors for product
risk, and fixed boundary/length diagnostics for explanation. Thresholds, model choice, context size,
and weights are selected only on train/calibration; the locked test is read once per preregistered
experiment.

## Execution order

1. **Now:** immutable model/dataset/n-gram provenance, loader/metadata equality checks, and
   diagnostic-only boundary metrics.
2. **Next cheap falsification:** direct LFM/Qwen causal-LM sequence scoring on existing N-best;
   promote to a `K<=8` cache only on positive held-out evidence.
3. **Next product slice:** define the frozen `ContextCatalog` Interface and run a small real
   catalog/distractor experiment; do not synthesize a catalog from test references.
4. **GPU experiment:** benchmark Qwen3-ASR-1.7B standalone, then test a preregistered selective
   policy only if it is strong enough.
5. **Runtime profiles:** add `cpu-ja-v1`, then measured `gpu-ja-second-ear-v1`; add edge/streaming
   profiles only after ReazonSpeech/Moonshine device evidence exists.
6. **Deep Module migration:** introduce the one-call facade, route one CLI vertical slice through
   it, then move pooling/context/gating/provenance behind the seam while keeping low-level research
   functions available.

## Sources used for selection

The review used Context7 to recheck the current Interfaces of
`/systran/faster-whisper`, `/opennmt/ctranslate2`, `/qwenlm/qwen3-asr`,
`/huggingface/huggingface_hub`, `/huggingface/datasets`, and
`/huggingface/transformers`. Exa searches covered model architecture, Japanese ASR,
context/entity retrieval, confidence, N-best rescoring, boundary handling, and CPU/edge
deployment. More than 300 result slots across root and independent design lanes were reduced to
about 30 primary or official source families; blogs, mirrors, product roundups, and duplicate
paper pages were excluded from decision-grade evidence.

- Qwen3-ASR technical report: <https://arxiv.org/abs/2601.21337>
- Cached LLM Probability Retrieval: <https://arxiv.org/abs/2608.16023>
- Contextualized ASR with Dynamic Vocabulary: <https://arxiv.org/abs/2405.13344>
- Deferred NAM: <https://aclanthology.org/2024.naacl-industry.26/>
- ConEC real-context benchmark: <https://aclanthology.org/2024.lrec-main.328/>
- RECOVER constrained entity correction: <https://arxiv.org/abs/2603.16411>
- Re-evaluating MBR for ASR: <https://arxiv.org/abs/2510.19471>
- ReazonSpeech models: <https://github.com/reazon-research/reazonspeech>
- Moonshine tiny specialized ASR: <https://arxiv.org/abs/2509.02523>
- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>
- CTranslate2: <https://github.com/OpenNMT/CTranslate2>

All recommendations remain `LOCAL_PASS` planning or local measurement. They do not establish device,
provider, public-release, or human-approval state.
