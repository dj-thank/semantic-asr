# Runtime reliability audit — 2026-09-05

## Scope and reproduced defects

This audit is stacked on the opt-in v0.3 work in PRs #15–#17, without promoting
unmeasured models or deliberation policies to default inference.
The independently exported starting source passed 372 tests on Python 3.13.5,
but the following end-to-end regressions were reproduced outside those fixtures:

* Converting every signed int16 value through the public array materializer
  changed 65,533 of 65,536 values. Integer PCM was cast to float and clipped
  before normalization. The corrected path round-trips all 65,536 values exactly.
* Two disjoint windows containing `はい。` were rendered as one `はい。`.
  Textual overlap is now considered only when the audio windows overlap in time.
* Passing actual first-pass output to the second pass failed with
  `ValueError: first-pass long-form evidence hash mismatch`. Different producers
  and consumers had independently defined incompatible root hash payloads.

These are deterministic waveform, orchestration and rendering tests, not a
Japanese speech-recognition benchmark or a claim of new CER improvement.

## Implemented behavior

### Sample-preserving input and bounded window reads

`audio.py` normalizes signed integer PCM and unsigned 8-bit PCM before downmixing.
Boolean, complex, object and unsupported unsigned formats fail explicitly.
The public array path retains its 16 kHz contract; it does not silently resample
or infer another sample rate. Float input is saturated without integer wraparound.

The default path-preserving adapter and the v2 path adapter use seek-based reads
for mono, 16 kHz, signed 16-bit PCM WAV. Other formats retain the existing decoder
fallback. Truncated native payloads are rejected. A sample-count regression checks
that twelve one-second windows read twelve one-second ranges, not the entire
recording twelve times. No whole-file process cache or new model is introduced.
The legacy `FasterWhisperAdapter` remains unchanged; see the outstanding item below.

### Shared, versioned evidence contract

`LongformResult.create()` and `verify()` share the
`semantic-asr-longform-evidence-v2` payload. It binds the source audio hash,
duration, ordered windows, observed and normalized text, original segment
evidence hashes and complete normalization receipts. Export verifies this
contract before writing. The second pass uses the same verifier, checks supplied
audio identity, and retains the exact original candidate and ranking evidence.

**Compatibility:** historical long-form roots used an incompatible unversioned
hash. They are not silently accepted as v2. Recreate results from verified source
recordings and retained evidence; do not merely relabel an old hash. Direct fixture
construction should use `LongformResult.create()`. Confidence remains a separate,
domain-dependent estimate rather than an authenticated proof of correctness.

### Transcript and subtitle fidelity

Disjoint-window repetitions are preserved. Subtitle rows must reconstruct the
selected observed text, ignoring whitespace only. Incomplete, malformed or
internally overlapping timing rows fall back to the complete observed window.
A changed deliberation path does not reuse old confidence or timestamp spans.
Facade export checks the source name, text channels, segment index/status/times
and regenerated utterances against verified long-form data.
Temporal-overlap deduplication is still a text-boundary heuristic, not forced
alignment and not a claim of perfect document rendering.

### Calibration scope

The old CPU calibration is withheld for GPU, quality and custom profiles,
unknown adapters, prompted decoding, changed loop guards, decoder penalties,
timestamp mode, explicit CPU thread counts, fusion settings, evidence budgets,
extra enrichment, second-ear ASR, teachers and aligners. The immutable calibration
reference cannot be replaced by editing the public profile registry.
Matching configuration does not prove a new speaker, hardware, dependency version
or recording domain matches the original calibration distribution.

### Output integrity

Unique temporary files, flush/fsync and exclusive hard-link publication prevent
same-process writer collisions and silent replacement with `overwrite=False`.
The complete output set is rendered and checked for existing destinations before
publication, including facade JSON and utterance SRT. Empty format sets write
nothing; non-finite JSON values are rejected instead of exported.

Publication is atomic **per file**, not a transaction over the entire directory.
A crash or concurrent destination creation can still leave a partial set. A
filesystem without same-directory hard links fails safely for exclusive writes.

## Independent validation

The final publishable source was exercised with Python 3.13.5, NumPy 2.3.5,
CPU PyTorch 2.10.0 and Ruff 0.16.6:

* Full suite: **444 passed, 3 expected failures**. No unexpected failures.
* Ruff formatting and lint, and compileall: passed.
* Editable installation, model-free demo and deterministic research-smoke: passed.
* Wheel build and isolated wheel installation: tested separately from source imports.
* GitHub CI adds explicit NumPy-equipped Ubuntu and Windows jobs so sample-level
  regressions cannot silently skip in the dependency-free job.
* The read-only evidence workflow exports the tested source archive, commit/tree
  identity, archive SHA-256, dependency versions and JUnit results.

The three strict expected failures document the legacy `DecodeRequest` accepting
`True`, `1.2` and NaN as beam sizes. A local repair was tested, but the connector
rejected writing `src/semantic_asr/adapters.py`; that file is not claimed fixed.
The expanded local prototype had 470 passing tests, but that is **not** the test
count for the published source. Runtime profiles and the long-form constructor
validate their integer limits before calling the default decoder.

No new real-speech model evaluation, training on public speech, GPU run, Japanese
CER/WER measurement, or target-device latency claim was performed in this audit.

## Reference

The native WAV reader uses the standard-library PCM metadata, `setpos` and
`readframes` interfaces: https://docs.python.org/3/library/wave.html
