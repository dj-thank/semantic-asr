# Semantic ASR v0.3 — Multi-level acoustic–phonetic deliberation

## 1. Problem

The v0.2 candidate cascade preserves evidence correctly, but its principal decision unit is still a
completed ASR hypothesis. This leaves three gaps:

1. a hypothesis may be locally close to the audio but globally incoherent;
2. the correct sentence may not exist as one complete N-best hypothesis;
3. text-derived mora evidence is not independent acoustic evidence.

v0.3 introduces a second-pass confusion network whose arcs may come from separate first-pass paths,
independent phone/mora posteriorgrams, and acoustically verified local proposals. A global scorer
may read the completed path and bidirectional document context, but it remains rank-only.

The invariant is unchanged:

```text
context preference != acoustic proof
observed transcript != normalized transcript
```

## 2. Architecture

```text
audio
  │
  ├─ Whisper / CTranslate2 path N-best
  │
  ├─ frozen audio encoder ── phone posteriorgram ─┐
  │                                              ├─ exact CTC candidate likelihood
  ├─ frozen audio encoder ── mora posteriorgram ──┘
  │
  └─ frozen SSL/codebook ── discrete units ── centroid DTW   (#15)
                         │
                         ▼
             held-out utility normalization
                         │
                         ▼
       ordered multi-level deliberation lattice
       ┌──────────────┬──────────────┬──────────────┐
       │ consensus    │ ambiguity    │ consensus    │
       │ one arc      │ A / B / P2G  │ one arc      │
       └──────────────┴──────────────┴──────────────┘
                         │
              base beam / Viterbi search
                         │
                         ▼
        full-path bidirectional context scorer
        (rank only; cannot write a new transcript)
                         │
                         ▼
             acoustic-retention hard guards
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
          accepted                provisional
```

## 3. Independent phone and mora evidence

`PosteriorSequence` represents a frame-level distribution produced from audio before any candidate
is inspected. It freezes:

- evidence kind (`phone` or `mora`);
- complete label vocabulary and CTC blank;
- frame timestamps and complete per-frame distributions;
- acoustic encoder, immutable encoder revision and label-set revision;
- source-audio SHA-256.

A `CandidatePronunciation` is produced separately by a frozen G2P or lexicon adapter and is bound to
the SHA-256 of the exact candidate text. `ctc_pronunciation_score()` computes the CTC forward
likelihood of that pronunciation under the audio posteriorgram.

Phone and mora likelihoods are never added directly. They remain raw log-likelihood evidence in
separate score domains.

### Why both levels

Phone posteriors preserve fine distinctions such as voicing and consonant identity. Mora
posteriors preserve Japanese timing units and special mora such as `ン`, `ッ` and `ー`. Agreement
between the branches is useful evidence; disagreement is a trigger for further verification, not a
reason to silently average incompatible raw values.

## 4. Phoneme-to-grapheme proposal boundary

`FrozenPronunciationLexicon` is the dependency-free P2G baseline. It maps frozen surface entries to
phone and/or mora sequences. `propose_text_from_pronunciation()`:

1. scores each entry against independent phone/mora posteriorgrams;
2. converts each raw score through a held-out `UtilityCalibrationProfile`;
3. ranks only the resulting bounded utilities;
4. emits a `PhoneticTextProposal` that can become a local lattice arc.

A future neural P2G model should implement the same proposal contract. It may propose text absent
from Whisper N-best, but that text is not observed-eligible until phone, mora, discrete-unit or
another acoustic verifier binds it to the source audio.

The lexicon baseline is deliberately local. It is intended for contradiction islands, entities,
numbers and short uncertain spans, not exhaustive decoding over an unrestricted Japanese
vocabulary.

## 5. Multi-level deliberation lattice

A `DeliberationLattice` is an ordered confusion network. Each `DeliberationSpan` contains:

- one retained first-pass ASR arc;
- zero or more first-pass alternatives;
- optional acoustically verified phonetic or guarded-generation arcs.

A path selects exactly one arc from every span. This allows a final transcript to combine the
supported prefix of one N-best hypothesis, the middle of another, and a verified phone-to-text
proposal. It therefore removes the requirement that the correct full sentence already exist as a
single N-best row.

Every utility attached to an arc is:

- bounded to `[-1, 1]`;
- assigned an explicit channel;
- bound to the raw input score digest;
- bound to a frozen normalization-profile digest.

The bounded value is a utility for path ranking, not a calibrated probability.

## 6. Global deliberation

`GlobalSequenceScorer` receives:

- every arc of one completed candidate path;
- left document context;
- right document context;
- a topic summary;
- frozen entity identifiers and other declared metadata.

This is the seam for a bidirectional Transformer, cross-encoder or local LLM. The scorer must
return a bounded path preference bound to the exact path digest and exact context digest. A stale
score or a score computed for a different text path fails closed.

The initial dependency-free implementation first keeps the strongest base paths in a bounded beam,
then applies the complete-path scorer. This makes the global model operational without granting it
text-generation authority. Later experiments may replace the beam with lattice-aware Transformer
rescoring, provided the same provenance and acoustic guards remain.

## 7. Acoustic-retention guards

Context is useful but dangerous. A fluent model will prefer grammatical text even when a speaker
actually produced a repair, learner error or unusual phrase.

`DeliberationPolicy` therefore applies two hard guards before context ranking:

1. **per-span audio regression** — an alternative is removed when its bounded audio support falls
   too far below the retained first-pass arc;
2. **mean path audio regression** — a complete path is removed when its mean audio support falls too
   far below the retained complete path.

Generated/context proposals additionally require an independent audio channel (`phone`, `mora` or
`discrete_unit`) before they can enter the observed-eligible path set. Semantic or contextual
preference alone is insufficient.

A small retention bonus is available as a conservative tie-break. It must not be used to conceal a
poorly calibrated evidence branch.

## 8. Homophones and orthographic uncertainty

Audio cannot distinguish true homophones such as `仕様` and `使用` when their spoken reading is the
same. An arc may therefore carry a `pronunciation_key`. When context selects a different surface
with the same key, the decision records:

```text
context-resolved-orthography
```

This means:

- the spoken form is acoustically supported;
- the written form is resolved by context;
- the orthography was not independently proved by audio.

Other resolution modes distinguish retained first-pass text, acoustic/context consensus and an
acoustically verified generated proposal.

## 9. Relationship to discrete-unit evidence in PR #15

PR #15 supplies an additional candidate-specific acoustic branch:

```text
Audio2DUnit + Text2DUnit + same-codebook centroid DTW
```

Its raw DTW score should pass a held-out `UtilityCalibrationProfile(channel="discrete_unit")` before
joining a deliberation arc. Audio-only surprisal remains a routing signal because it is shared by
all candidates for the utterance. It must not become a candidate-ranking utility.

Phone CTC, mora CTC and centroid DTW are complementary:

- phone CTC: fine phonetic identity;
- mora CTC: Japanese rhythmic/phonological structure;
- discrete-unit DTW: representation-space acoustic alignment.

They must be evaluated independently before any learned fusion is promoted.

## 10. Runtime integration sequence

### Slice A — implemented in this PR

- typed audio-only phone/mora posteriorgrams;
- exact candidate-bound CTC scoring;
- frozen held-out utility normalization;
- frozen lexicon P2G baseline;
- ordered multi-level lattice and complete-path decoding;
- complete-path context-scorer interface;
- per-span and whole-path acoustic-retention guards;
- homophone resolution provenance;
- dependency-free executable example and tests.

### Slice B — next

- adapter from `SemanticLattice.contradiction_islands` to local deliberation spans;
- span-local score distribution so a full-hypothesis score is not counted once per character;
- optional global second pass in `SemanticASRTranscriber` after all first-pass windows complete;
- trace/export additions under the immutable observed evidence hash.

### Slice C — model work

- frozen Japanese phone CTC head;
- frozen Japanese mora CTC head;
- speaker-disjoint train/calibration/test manifests;
- small bidirectional complete-path scorer;
- neural P2G proposal adapter for N-best misses;
- target-device latency and memory profiles.

### Slice D — long-form and streaming

Offline transcription can use left and right context. Streaming must use only committed left
context and mark the result provisional until the right-context second pass completes. A streaming
preview must never overwrite the immutable final observed evidence object.

## 11. Evaluation

Promotion requires separate results for:

- strict and lenient CER;
- phone error rate and mora error rate;
- semantic-critical errors: numbers, dates, currency, negation, modality and entities;
- N-best oracle coverage and proposal recovery outside N-best;
- context correction rate;
- context-induced false correction rate;
- homophone orthographic accuracy;
- retention accuracy on disfluencies, repairs and learner errors;
- calibration/normalization drift;
- risk–coverage;
- latency, peak memory and energy per effort tier.

Required ablations:

```text
first-pass only
+ global context only
+ phone only
+ mora only
+ discrete unit only
+ phone + mora
+ all acoustic branches without context
+ all branches with context and retention guards
```

Evaluation context must be exogenous. Reference transcripts, corrected hypotheses and information
created after listening to evaluation audio may not be injected as context.

## 12. Claim boundary

This PR establishes an executable decoding and evidence contract. It does not include trained
Japanese phone/mora heads, a neural global-attention model or a measured CER improvement. The
lexicon bridge and callable full-path scorer are deterministic research baselines and integration
seams. No component becomes a runtime default until locked Japanese audio demonstrates a useful
quality/risk/latency frontier over the measured v0.2 baseline.
