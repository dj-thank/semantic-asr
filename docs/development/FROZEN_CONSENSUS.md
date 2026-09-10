# Opt-in selection from captured Japanese hypotheses

`semantic_asr.frozen_consensus.select_frozen_consensus` implements the fixed
full-window rule measured in the Japanese campaign. Qwen's original candidate is
returned only when its nonempty NFKC text, excluding whitespace, punctuation and
symbols, equals both Reazon and Japanese Parakeet. Otherwise the original Whisper
candidate is returned. No words are spliced, normalized output is not substituted
for observed text, and no score or correctness probability is manufactured.

This is a standalone, opt-in selection operation. It neither changes the default
transcriber nor implements a four-model inference service. The returned candidate
is not automatically an accepted `ObservedTranscript` or a calibrated ranking.

## Input and execution

The Python function accepts exactly the keys `whisper`, `qwen`, `reazon`, and
`parakeet`, each containing an existing `CandidateEvidence`. Each candidate needs
a distinct ID, `rank=1`, and these metadata fields:

- `audioWindowSha256`: SHA-256 of the actual decoded little-endian float32 PCM
  window, not its compressed file or unquantized dataset ancestor.
- `startMs`, `sampleCount`, `sampleRate=16000`, and `language="ja"`. All four
  candidates must refer to the identical window, at most 30 seconds.
- `model` and either `modelArtifactSha256` (64 hex characters) or `modelRevision`
  (immutable 40/64 hex revision). Reused identities are rejected even if renamed.

The caller must capture provenance from actual inference. These fields and their
digests do not independently prove model execution or that a named engine was used.
The selector never compares raw likelihoods across engines. Existing candidate
scores are preserved without interpreting them.

For a local JSON object with the same four keys and `CandidateEvidence.as_dict()`
values:

```bash
python scripts/select_frozen_consensus.py captured-window.json \
  --output-dir /external/new-selection --allow-local-research
```

The input is bounded to 1 MiB and unknown candidate fields are rejected. A new
output directory is required. `selected.txt` contains the selected original text;
`selection.json` retains all four candidates, a canonical input digest, policy ID,
and evidence digest. The receipt explicitly says `provisional`, `newInference=false`
and `promotion=false`. This may contain private transcripts and must not be
published without their own permission and rights checks.

## Evidence ceiling

The rule was first explored on exposed development data, then frozen and tested
on another 64 FLEURS train-derived development recordings (860.16 seconds), with
known prior IDs, reference hashes and source PCM hashes excluded. Strict CER fell
from 6.331% to 5.821%; punctuation/symbol-stripped CER fell from 3.449% to 3.313%.
The latter paired bootstrap interval included zero improvement. No previously
error-free utterance became erroneous in this cohort under either metric; that
does not establish a population false-correction rate of zero. Speaker independence
is unknown. The final test was not opened and there is no automatic promotion.

The frozen rule can miss better candidates and can agree on wrong text. Agreement
is evidence for a bounded heuristic, not proof of acoustic or semantic correctness.
