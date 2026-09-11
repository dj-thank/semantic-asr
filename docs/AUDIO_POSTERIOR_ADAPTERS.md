# Frozen phone and mora posterior adapters

## Purpose

Semantic ASR must distinguish two fundamentally different kinds of evidence:

```text
candidate-derived reading / mora shadow
candidate-independent audio-to-phone or audio-to-mora posterior
```

The first is useful for comparing spellings and pronunciations but cannot prove a generated
candidate was spoken. The second is produced from audio before a text candidate is inspected and
can independently verify a candidate pronunciation.

`audio_posterior_adapters.py` supplies the strict model boundary for this second category.

## Frozen model configuration

`FrozenPosteriorModelConfig` binds:

- evidence kind: `phone` or `mora`;
- model identifier;
- immutable model revision;
- exact ordered label vocabulary;
- CTC blank symbol;
- required sample rate;
- frame stride;
- optional artifact SHA-256;
- logits temperature and numerical floor.

The default revision policy requires an exact 40-character commit. An artifact-digest policy is
available for a locally packaged model, but it requires a full SHA-256. Mutable aliases such as
`main`, `latest`, or an unpinned model name are rejected.

## No implicit audio transformation

The generic extractor requires mono samples at the exact configured sample rate. It does not:

- guess channel orientation;
- downmix stereo;
- resample;
- normalize loudness;
- clip values;
- trim silence.

Those operations can materially change phonetic evidence and must be explicit, versioned stages.
`canonical_audio_sha256()` hashes the declared sample rate and canonical little-endian float32
sample stream. A caller may instead provide the original full-file SHA-256; the backend must return
the same source binding.

## Logit conversion

`PosteriorLogits` is bound to both source audio and the frozen model configuration. Conversion to a
`PosteriorSequence` checks:

- rectangular frame-by-label shape;
- exact vocabulary width;
- finite logits;
- maximum frame and vocabulary limits;
- permitted frame stride;
- source and model digests.

Softmax uses the frozen temperature and probability floor, then renormalizes each frame. The output
preserves complete distributions rather than collapsing to one phone or mora ID.

## Resource controls

`PosteriorResourcePolicy` caps:

- audio duration;
- frame count;
- vocabulary size;
- minimum and maximum frame duration.

Long recordings should not be sent through this adapter wholesale. The intended use is selective
analysis of contradiction islands, numbers, names, negation, and other high-risk spans identified by
the document lattice.

## Dual extraction

`DualPosteriorExtractor` runs phone and mora backends against the same canonical audio and requires
both outputs to share one `source_audio_sha256`. This prevents a phone sequence from one clip and a
mora sequence from another clip from being fused into a valid proposal.

## Optional Transformers backend

`TransformersCTCBackend` is an optional adapter for an explicitly supplied Hugging Face CTC model.
It uses:

```text
trust_remote_code=False
exact model revision
exact id2label order
model.eval()
torch.inference_mode()
```

The backend does not select a model or claim that a generic grapheme CTC model is a phone/mora
model. The caller must provide a model whose frozen labels actually represent the declared evidence
kind. A Japanese phone or mora head becomes eligible for evaluation only after its training data,
label inventory, revision, rights, and speaker-disjoint calibration/test manifests are recorded.

## Integration with phonetic proposals

The output flow is:

```text
uncertain audio span
   ├─ phone CTC backend ─ phone posteriorgram ─┐
   └─ mora CTC backend  ─ mora posteriorgram  ─┼─ candidate CTC likelihood
                                               │
                                   held-out utility calibration
                                               │
                              P2G / frozen pronunciation lexicon
                                               │
                              source-audio-bound span proposal
```

A proposal must still pass the local and document acoustic-retention guards. The presence of a
phone or mora score is not itself a correctness guarantee.

## Required model evaluation

Before enabling a model in any profile:

- phone error rate or mora error rate on a locked Japanese split;
- calibration of candidate sequence likelihoods;
- performance on voiced/unvoiced consonants, long vowels, `ン`, `ッ`, palatalized mora, and vowel
  devoicing;
- microphone, speaker, gender, age, dialect, speaking-rate, and noise slices;
- generated-candidate false acceptance;
- incremental CER and semantic-critical error when added to the document lattice;
- target-device latency, memory, and energy.

Negative or neutral results must remain recorded and must not be promoted into a default effort
profile.
