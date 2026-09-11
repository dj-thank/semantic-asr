# Semantic ASR v0.3 integrated research status

## Goal

Semantic ASR v0.3 changes the decision unit from one completed N-best sentence to an auditable,
multi-level document path:

```text
frozen first-pass audio evidence
    -> exact local candidate lattice
    -> optional source-audio-bound phone/mora/discrete-unit proposals
    -> overlap-aware document path beam
    -> complete-document ranker with left/right context
    -> acoustic retention and false-correction guards
    -> immutable observed-path receipt or first-pass retention
```

The measured v0.2 transcription profiles remain the production baseline. Every v0.3 component is
opt-in research code until a locked Japanese evaluation demonstrates a useful quality, risk,
latency, and memory frontier.

## Integrated components

### Exact local candidate lattice

- reconstructs every source hypothesis exactly;
- represents insertion and deletion explicitly;
- allocates finite evidence factors across local spans;
- separates candidate-derived `mora_shadow` from audio-derived `mora`;
- requires independent audio evidence for generated observed candidates.

### Document-wide decoding

- carries bounded local alternatives through a hierarchical document beam;
- resolves overlapping long-form windows with explicit receipts;
- scores the exact text that would be emitted;
- applies duration-weighted acoustic regression and change-budget guards;
- keeps scorer identity separate from path-selection identity;
- preserves the immutable first-pass result when the second pass fails or abstains.

### Trainable document ranker

- signed hashed Japanese character n-grams over candidate and bidirectional context;
- dense local, overlap, acoustic, change, generation, ambiguity, and retention features;
- pairwise training on CER, semantic-critical errors, and first-pass-exact false corrections;
- group-disjoint train/calibration/test splits;
- canonical JSON artifact with internal digest and held-out score normalization;
- rank-only runtime surface with no text-generation authority.

### Independent phone and mora evidence

- frozen frame-level phone and mora posterior contracts;
- shared-bottleneck dual CTC head;
- speaker/source/audio-disjoint feature manifests;
- safe `.npy` input, safetensors output, and immutable encoder/inventory binding;
- greedy PER/MER, candidate-discrimination AUC, and hard-negative false-accept evaluation;
- calibration thresholds fitted only on calibration data and applied unchanged to locked test.

### Japanese pronunciation and feature preparation

- explicit kana reading only; no hidden kanji or Latin pronunciation guess;
- fixed mora and phone mapping for basic kana, yoon, special mora, and enumerated foreign sounds;
- recording-file SHA-256 and extracted-clip canonical SHA-256 kept separate;
- exact sample-range and encoder/runtime sidecar receipts;
- resumable, prefix-verified, atomic derived-feature export;
- rights decision and license required for every source row.

## Core invariants

```text
context preference != acoustic proof
candidate-derived mora != audio-derived mora
recording file hash != canonical extracted-clip hash
ranker preference != correctness probability
observed transcript != normalized transcript
software validation != quality promotion
```

## Integrated software gate

The integrated branch is reviewable only after one source tree passes:

- Ruff format and lint;
- Python compilation;
- complete dependency-free tests;
- optional CPU PyTorch joint-head and acoustic-verifier tests;
- NumPy/SoundFile feature-export tests;
- tiny document-ranker and joint-CTC training CLIs;
- example and CLI discovery;
- wheel build and clean-wheel imports;
- source-tree immutability after validation;
- digest-bound integrated validation ledger.

Intermediate stacked drafts are retained until this gate succeeds. They are then superseded by the
integrated PR rather than merged independently.

## Required real-data promotion matrix

A real Japanese promotion decision requires locked, speaker/source-disjoint data and paired
comparison against the measured v0.2 first pass.

### Recognition quality

- strict and lenient corpus CER;
- utterance-mean CER;
- phone error rate and mora error rate;
- N-best oracle coverage;
- proposal recovery outside N-best;
- document path oracle coverage.

### Meaning preservation

- numbers, dates, times, currency, percentages;
- negation and modality;
- people, products, places, and domain entities;
- particles, repairs, fillers, and learner errors;
- homophone orthographic resolution separated from spoken-form confidence.

### Safety and selectivity

- false correction among first-pass-exact recordings;
- accepted coverage and accepted error;
- provisional proposal quality;
- acoustic-regression guard precision/recall;
- context-only distractor and wrong-right-context arms;
- speaker, microphone, source, and domain shift;
- overlap deletion, duplication, and boundary corruption.

### Cost

- wall-clock real-time factor;
- stage and end-to-end latency;
- peak RAM and VRAM;
- model and artifact size;
- energy where measurable;
- CPU, small-GPU, and target-device Pareto frontiers.

## Current claim boundary

The integrated code establishes algorithms, contracts, training paths, serialization, receipts,
and executable test fixtures. It does not yet contain a promoted Japanese phone/mora checkpoint, a
promoted document ranker, a validated best encoder layer, or a measured end-to-end CER improvement.
Negative real-data results must be recorded and are valid outcomes.
