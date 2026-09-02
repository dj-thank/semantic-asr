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

### Design It Twice comparison

Four independent Interface designs were compared:

1. **Minimal recognizer:** one `recognize(request)` entry point maximizes Depth and keeps safety
   local, but a generic diagnostics dictionary and constructor wiring still leak knowledge.
2. **Flexible session:** `open/push/finish` covers streaming, multiple decoders and budgets, but
   makes every offline caller learn a lifecycle before a second streaming Adapter is proven.
3. **Default-caller facade:** `transcribe(audio, profile=...)` makes the measured CPU case trivial
   and moves variation into immutable profiles; it has the best present-day Leverage.
4. **Ports & adapters:** Decoder, Scorer and Context ports localize external dependencies well,
   but Clock/Budget/Provenance ports would be hypothetical with only one current implementation.

The selected hybrid is design 3 externally and the justified subset of design 4 internally. A
private minimal recognizer owns orchestration. Streaming session state, remote context services,
and additional public ports are added only when a second Adapter makes each seam real.

## Technology selection

| Need | Selection | Status and evidence | Stop / promotion condition |
|---|---|---|---|
| CPU offline primary | faster-whisper large-v3-turbo, CTranslate2 int8 | **keep**; only stack measured end-to-end here (`0.2518` strict utterance mean, `0.1162` lenient corpus) | replace only on same immutable manifest with strict and non-boundary slices non-inferior |
| High-quality independent ear | Qwen3-ASR-1.7B on CUDA/ROCm/vLLM | **experiment on a supported GPU runtime**; the 1.7B AuT+LLM model is stronger than 0.6B in the official report, supports Japanese and unified offline/streaming | same 600 clips, exact revision, first measure standalone CER; enter fusion only if at least non-inferior and complementary |
| Current Qwen 0.6B | single-hypothesis contradiction signal | **do not promote**; all-on insertion is harmful and the best calibration-selected gate changed three rows without strict gain | retain probe infrastructure only; no default weight or candidate authoring |
| Context-capable Japanese challenger | Fun-ASR-Nano-2512 (800M) | **benchmark next, do not promote yet**; its official card supports Japanese, prompt/hotwords, CPU or CUDA, but requires revision-bound custom code and currently lists timestamps as TODO | start with a 20-clip native/verbatim plus entity pilot; expand to 600 only if content and filler/entity guards pass |
| Native streaming challenger | Voxtral Mini 4B Realtime 2602 | **defer to a supported >=16 GB vLLM GPU**; official Japanese FLEURS WER is reported, but no conversational/verbatim evidence exists and RX 7600 XT is not a supported local path | measure 480 ms and 2400 ms delay on the same native/locked slices; reject if filler/content or latency guard fails |
| GLM-ASR-Nano-2512 | 1.5B Transformers ASR | **do not prioritize for Japanese**; the official model metadata and examples are English/Chinese and published benchmark emphasis is Chinese, despite a broader language-support image | reopen only after an exact Japanese benchmark/runtime is identified; never infer Japanese quality from the aggregate claim |
| Japanese CPU/edge adapter | ReazonSpeech-k2-v2 via sherpa-onnx (159M Zipformer RNN-T, ONNX) | **benchmark as a deployment adapter**, not as a claimed quality upgrade; architecturally independent and portable | test exact model on the immutable 600 clips plus latency/RAM; do not infer from vendor benchmark normalization |
| Very small edge lane | Moonshine tiny Japanese (27M) | **defer to device profile**; attractive size, but not evidence for better reranking or broadcast accuracy | require device RTF/RAM and the same strict/lenient/entity slices |
| Generic count LM | Wikipedia or train-reference n-gram | **reject**; both fail the locked test, including calibration-only model selection | reopen only with a genuinely matched external domain corpus and a preregistered cache-coverage hypothesis |
| Cached language evidence | short-context cached causal-LLM probabilities (`K<=8`) | **defer**; direct LFM2.5-350M scoring failed protected fit metrics and slightly worsened test strict mean, so no cache was built; the paper is also flat/harmful for Whisper-large-v3 | reopen only with a stronger/base teacher plus matched non-test text and a new preregistered audit split |
| Entity recovery | frozen exogenous catalog + compact phonetic/semantic retrieval + no-bias gate | **primary product direction**; dynamic vocabulary, Deferred NAM, CLAR and RECOVER support phrase retrieval and constrained correction | catalog must pre-exist test references; report entity recall/CER, non-entity CER, distractor FP and abstention separately |
| Confidence | small CEM over beam score/rank, margin, entropy, degeneracy and stage features | **data-blocked**; current ranker is dominated by rank and the calibration set is too small for robust tail guarantees | collect a larger speaker/domain-disjoint calibration/audit set; require ECE, risk-coverage, deletion and OOD slices |
| Boundary handling | VAD/segmentation adapter plus diagnostic contiguous alignment | **diagnostic shipped; runtime fix later** | never select candidates with reference alignment; validate a reference-free segmenter on concatenated audio and a clean-boundary corpus |
| Verbatim/disfluency control | explicit filler/hesitation labels plus observed/intended outputs | **primary medium-term training direction**; Japanese CSJ experiments show that treating disfluencies as recognition targets improves spontaneous ASR and exposes removable labels | require licensed speaker-disjoint native speech; report filler/repair F1 and observed CER before deriving an intended/readable transcript |
| Free-form LLM GER | none in the observed transcript path | **reject as default**; it can author unsupported text | allow only entity-scoped proposals from a frozen catalog with deterministic verify/apply and full abstention |

## Latest architecture implications

Qwen3-ASR uses an AuT audio encoder, projector, and causal LLM with dynamic attention windows;
the official high-level wrapper returns one transcript (plus language and optional transcript-
conditioned alignment), not a calibrated N-best confidence contract. Therefore it belongs behind a
decoder Adapter and must not impersonate the CTranslate2 score Interface.

The local RX 7600 XT is usable through DirectML for ordinary tensor operations, but the pinned
Qwen3-ASR-0.6B failed a one-clip probe across its variable-length audio path and text embedding
path. Bounded workarounds for `masked_scatter` and CPU audio-tower/GPU decoder placement exposed
additional DirectML backend incompatibilities. Do not download or claim the 1.7B run on this
runtime; use a supported CUDA/ROCm/vLLM host or an official converted artifact instead.

Fun-ASR-Nano-2512 is the strongest unmeasured near-term challenger for this product shape because
the official 800M checkpoint explicitly supports Japanese plus contextual prompts/hotwords and a
CPU path. That is still only a capability claim: its model code must be pinned and reviewed, its
timestamp path is incomplete, and its native Japanese filler behavior has not been measured here.
Voxtral Realtime is a separate streaming profile, not a drop-in offline replacement; the official
card requires vLLM and at least 16 GB GPU memory and shows a quality/latency tradeoff by delay.
GLM-ASR remains below both until its exact Japanese support and benchmark are inspectable rather
than inferred from a multilingual image or Chinese-heavy aggregate.

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

The evaluation corpus roles must also be separated:

| Corpus | Proper role | Why / limitation |
|---|---|---|
| ReazonSpeech locked 600 | conditional legacy broadcast regression and candidate-oracle baseline | public and already measured, but caption/segment boundaries are imperfect, speaker IDs are absent, two normalized references cross train/calibration, and only 6/116 test references contain a detected filler; regenerate strict splits before promotion |
| HTH Japanese casual conversation preview | immediate verbatim-fidelity pilot | 69 human-checked native casual utterances / 324 s, with 16 explicitly tagged filler events and 4 repairs; preview terms have no clear repository license, so keep results local and do not redistribute |
| HTH full 120 h | candidate production-grade casual-speech audit set | human-verified, filler/disfluency tags and speaker-separated dialogue; commercial access is a separate human decision |
| Corpus of Spontaneous Japanese (CSJ) | strongest established spontaneous-speech gold set | 661 h, orthographic/phonetic transcripts and explicit filled-pause/repair tags; paid/application-controlled, with commercial use reviewed separately |
| J-CHAT | large-scale pretraining/domain exposure only | native in-the-wild dialogue at large scale, but transcripts are produced by ReazonSpeech-NeMo rather than human gold |
| STUDIES | scripted dialogue/prosody checks | open research-use studio dialogue from actors reading prepared lines, not an unscripted filler benchmark |

Verbatim evaluation has three independent outputs. `spoken_reference_surface` removes only
annotation wrappers while preserving filler and repair content. Strict/content CER excludes rows
with inaudible, uncertain, or anonymized spans unless a masked-span scorer is explicitly used.
Tagged-filler precision/recall/F1 is reported separately; a readable normalized transcript may
remove fillers only as a distinct downstream product output and never feeds back into observed
transcript scoring.

Metrics alone cannot make an ASR model emit omitted speech. The next trained architecture should
encode fillers and hesitation/repair spans as explicit recognition targets, then deterministically
derive the intended/readable transcript from those labels. This preserves both user needs: a
verbatim observed channel and a clean downstream channel, without letting normalization hide
recognition errors.

Primary gates remain strict corpus CER and strict utterance-mean CER on identical audio/reference
boundaries. Report lenient corpus CER for comparability, entity/number/negation errors for product
risk, and fixed boundary/length diagnostics for explanation. Thresholds, model choice, context size,
and weights are selected only on train/calibration; the locked test is read once per preregistered
experiment.

## Execution order

1. **Now:** immutable model/dataset/n-gram provenance, loader/metadata equality checks, and
   diagnostic-only boundary metrics.
2. **Completed negative:** direct LFM causal-LM sequence scoring did not pass fit/test guardrails;
   do not build the cache from this teacher.
3. **Next product slice:** define the frozen `ContextCatalog` Interface and run a small real
   catalog/distractor experiment; do not synthesize a catalog from test references.
4. **Nearest challenger:** run a bounded, revision-pinned Fun-ASR-Nano-2512 pilot with no-context,
   frozen-hotword and distractor arms; expand only after native verbatim/content guards pass.
5. **GPU experiment:** benchmark Qwen3-ASR-1.7B standalone on a supported host, then test a
   preregistered selective policy only if it is strong enough.
6. **Runtime profiles:** add `cpu-ja-v1`, then measured `gpu-ja-second-ear-v1`; add edge/streaming
   profiles only after ReazonSpeech/Moonshine device evidence exists.
7. **Deep Module migration:** introduce the one-call facade, route one CLI vertical slice through
   it, then move pooling/context/gating/provenance behind the seam while keeping low-level research
   functions available.
8. **Verbatim audit expansion:** use the 69-row HTH preview as a local pilot now; promote only after
   the audio-to-transcript mapping, filler tags, uncertain spans and usage rights are all bound.
   Acquire CSJ or the full human-verified casual corpus only at an explicit human/cost decision.
9. **Verbatim training lane:** once licensed speaker-disjoint data exists, compare explicit
   filler/hesitation labels against removal and unlabelled baselines; keep observed and intended
   outputs separate.

The completed preview pilot exposes the immediate quality priority: large-v3-turbo produced
`0/16` exact tagged-filler variant matches (while emitting two filler-like forms) and matched only
`2/4` repair-span surfaces. Before another generic reranker, add a family-aware verbatim
profile/metric and a larger human-transcribed native-speech audit set.

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
- Corpus of Spontaneous Japanese: <https://clrd.ninjal.ac.jp/csj/en/data-index.html>
- HTH casual-conversation preview: <https://huggingface.co/datasets/HTH-inc/japanese-casual-conversational-speech-golden-dataset-preview>
- Japanese disfluency labeling: <https://www.isca-archive.org/interspeech_2022/horii22_interspeech.html>
- Japanese disfluency-labeling extension (2026): <https://www.jstage.jst.go.jp/article/transinf/advpub/0/advpub_2025EDP7157/_article/-char/en>
- Fun-ASR-Nano-2512 model card: <https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512>
- Voxtral Mini 4B Realtime 2602 model card: <https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602>
- GLM-ASR-Nano-2512 model card: <https://huggingface.co/zai-org/GLM-ASR-Nano-2512>

All recommendations remain `LOCAL_PASS` planning or local measurement. They do not establish device,
provider, public-release, or human-approval state.
