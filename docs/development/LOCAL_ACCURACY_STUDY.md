# Local accuracy study and native Whisper direction

Use public licensed development audio to compare actual public-CLI output with
saved recognizer outputs. Do not infer semantic accuracy from character differences.

## Execute the bounded study

From the repository root, use Python 3.12 with the `sherpa` extra plus `soundfile`,
`pyarrow`, and `fsspec[http]`. The optional inference models are local artifacts.
The initial prepared campaign inputs live under `work/campaign`; they are not
included in the public package or silently downloaded by model inference.

```bash
python scripts/local_accuracy_study.py prepare --output /external/new-study
python scripts/local_accuracy_study.py develop --output /external/new-study
python scripts/local_accuracy_study.py fresh --output /external/new-study
python scripts/local_accuracy_study.py report --output /external/new-study
```

`prepare` validates the existing campaign's 64 WAVs, IDs and reference hashes,
checks historical aggregate metrics, separates inferential input from references,
and declares three normalization trials and a six-hour wall-clock limit.
`develop` invokes the same `run_transcription` entry used by `semantic-asr run`,
reusing already loaded model objects. The child inference process receives only
IDs, WAV paths and hashes, never references. Optional expensive calls retain the
existing per-window budget. Output and failures are append-preserving.

The original three trials are numeric normalization, numeric normalization plus
a terminal full stop, and that same transformation on the dual-recognizer path.
These are normalized-layer hypotheses, not new acoustic training. The original
selection is frozen before retrieving 64 additional official-train samples from
the pinned FLEURS revision. Exclusions cover prior IDs, reference hashes and PCM
hashes. Speaker independence is unknown. This is a development check, not final
publication evaluation. Re-running a completed stage is not a new quality result.

The HTML contains audio controls, references, output text, highlighted surface
differences, raw/normalized string-match counts, per-item timing, unknown or
provisional status, and local manual-review controls. Provisional examples remain
in every aggregate denominator. Review edits are saved in the browser and can be
exported as JSON; they do not silently modify the original dataset or CER.

## Current evaluation criterion (2026-09-09)

The latest explicit plan restores strict reference matching as the primary goal.
The report uses literal codepoint CER, including punctuation, spaces, numeric
representation and kanji/kana differences, and separately reports literal utterance
exact-match rate. Provisional rows stay in the denominator; accepted-only CER and
exact-match rate are supplementary. Unknown cached states are never treated as accepted.

The already completed three-trial selection used the older NFKC/whitespace-free
CER. Its freeze and scores remain intact; the new report recomputes summaries from
saved per-row measurements without another model call or retroactive selection.
The 64 additional records have now been evaluated and are exposed regression data.
This metric clarification is not another optimization trial or unseen measurement.
Only CER zero AND literal exact-match rate 100% on the fixed set satisfy the stated
goal. This does not establish semantic correctness or general audio accuracy.

`surface_review` recognizes only conservative NFKC/case/space/punctuation/kana
equivalence. English-to-katakana aliases and kanji homophones require a reviewed
lexicon or listening; generated pronunciation is not observed pronunciation.
Unreviewed cases are explicitly unresolved, not automatically wrong or correct.
No general semantic accuracy percentage is claimed from this study.

## Architecture: preserve the strong native recognizer first

The user also requested improving Whisper/Qwen themselves and allowed architecture
changes. The next comparison therefore uses the same local Whisper large-v3-turbo
weights for native inference and the existing N-best Semantic ASR integration.
Candidate outputs and timings establish whether the added pipeline helps or hurts.

`NativeWhisperAdapter` and the opt-in `whisper-native-cpu-v1` profile preserve
Whisper's original transcription pipeline as the primary observation. Native
segment scores and utterance times are retained; no whole-window confidence is
fabricated. Subsequent candidate proposals must retain this original observation
and acquire independent evidence before applying content changes. The existing
default is not silently promoted or switched.

```python
from semantic_asr import transcribe

result = transcribe("recording.wav", profile="whisper-native-cpu-v1")
result.write("transcripts")
```

The default named model loader obtains pinned public weights if absent. For an
offline local model use a SHA-256-verified `NativeWhisperAdapter` and pass it as
`adapter=`. This native profile provides a baseline to protect, not a claim of
100% transcription or a replacement for testing Qwen-assisted corrections.

```bash
semantic-asr run recording.wav --profile whisper-native-cpu-v1 \
  --whisper-model-dir /models/faster-whisper-large-v3-turbo \
  --whisper-artifact-sha256 VERIFIED_DIRECTORY_SHA256 --output-dir transcripts
```

The native CPU profile uses two threads. This is the same thread setting as the
local baseline comparison; INT8 results can differ with other thread counts.
