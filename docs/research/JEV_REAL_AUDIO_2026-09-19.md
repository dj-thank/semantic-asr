# Jev phoneme-grounded real-audio development study — 2026-09-19

Tracking: #69. This document specifies an executed-development workflow, not a
claim of superior recognition or permission to enable Jev in production.

## Question and invariants

Ask which utterance is supported by observed phonemes, NOT which sentence is
natural. No reference transcript, reference pronunciation, candidate text, context,
ASR rank or CTC score enters the principal Jev arms. Candidate pronunciations are
G2P hypotheses. Acoustic phonemes are noisy estimates, not gold annotations.
`observedTranscript`, normalizers, realtime finals, production profiles, weights
and score contracts are unchanged. This script emits separate shadow receipts.

All audio in this session is public Japanese FLEURS, CC-BY-4.0, attributed to
Google / Conneau et al., FLEURS (2022). Phone strings can reveal speech content;
they are not anonymized simply because raw audio is not sent. No private audio
or credentials may be supplied or published through this protocol.

## Frozen inputs

- ASR base commit: `fdf2dbb6d77beb6311855229b1439d964be1c8bf`.
- Base tree: `6a6cf59112866554018003d0f1ebd6ee7bff79db`.
- Dataset: `google/fleurs`, subset `ja_jp`.
- Dataset revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
- Phone model: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4` (Apache-2.0).
- Phone revision: `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`.
- Existing profile: `cpu-ja-v1`, faster-whisper large-v3-turbo, CPU/int8.
- Whisper revision: `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`.
- Jev: fixed `jev-1.13.0`, never `jev-latest`.

`collect_phonetic_public_probe.py` selects the first distinct-text-ID rows from
sorted pinned parquet paths, 2–20 seconds, 48 validation + 48 test clips. Previously
inspected public test rows are **development/exposed-regression**, not unseen
held-out evaluation. Selection is not random population sampling. Verified
speaker/session separation, training-contamination exclusion and manually
annotated phones are absent. A file named `test` does not change these limitations.

## Finite trials

The session limit is 96 unique clean clips, 600 authenticated Jev requests,
4,000,000 input tokens, 6 GiB task storage and 3,600 seconds inference wall budget.
The existing collector has a record limit but not an OS quota/watchdog. A proposed
extra monitoring-script write was blocked and was not retried through another
route. Storage/wall limits for collection are operational limits, not a claimed
hard sandbox. Model downloads and virtual environments are setup, not inference.

1. Pilot: first 24 already collected clips, explicit partial-manifest snapshot,
   three principal arms + missing-observation and exact-repeat diagnostics on the
   first 12 clips: 96 calls maximum.
2. Full comparison: the 96 clips, the same three arms, same 12 diagnostics:
   312 calls maximum.
3. Preregistered after pilot but before further Jev calls: format-only ablation
   on the same 96 clips, original/reversed order: 192 calls maximum.

The same recording is not counted again as an independent sample in stages 2/3.
The format trial is development informed by the pilot, not untouched confirmation.
No automatic test-set tuning, model training or promotion occurs.

## Arms

- `baseline`: existing Semantic ASR selected candidate.
- `local_ctc`: existing exact `ctc_pronunciation_score`, no Jev.
- `oracle`: lowest reference CER among fixed candidates; evaluation-only ceiling.
- `greedy`: audio-only greedy phone token array + candidate phone token arrays.
- `paths`: greedy plus three valid candidate-independent CTC prefix alternatives.
- `reordered`: greedy arm with reversed candidates and rebound aliases.
- `no_observation`: empty observation, expected abstention.
- `repeat`: another live request with exactly the greedy payload.
- `joined` / `joined_reordered`: same phone tokens as greedy/reordered, encoded as
  space-separated strings rather than JSON arrays; unchanged instructions.

CTC prefix search uses width 8 and frame top-5 plus blank. Each retained sequence
has a valid CTC path, but search is pruned and lost probability mass is not zero.
The alternatives are not independent acoustic votes. Tiny-array exhaustive path
sums test the recurrence; those fixtures do not measure real-audio accuracy.

## Decision and integrity controls

An allowlist projection constructs requests. Raw input records contain evaluation
references and must NEVER be sent wholesale. Request hashes bind the exact model,
question, observation, candidate alias mapping and serialization. Responses must
match the fixed model, Choice schema, candidate set and finite numeric fields.
Confidence is recorded but never interpreted as ASR correctness probability.

Identical-pronunciation candidates, absent observation, model abstention and
malformed/failed responses preserve the baseline. Raw choices and gate reasons
are retained so the gate cannot conceal invalid model behavior. Candidate IDs
are mapped back before comparing permutations. Homophone/order statistics need
eligible denominators, not only the total including one-candidate clips.

The client uses one HTTPS endpoint, refuses redirects, never stores headers or
error response bodies, bounds calls/bytes/wall-time and conservatively reserves
64k input tokens before each request. Failures retain the reservation when usage
is unknown. `input_tokens_or_reserved` is not necessarily the final provider bill.
Per-run limits do not by themselves enforce a budget shared between invocations;
record the cumulative session total. API-disabled runs are `offline-only`.

Output directories must be new. The manifest bytes are frozen before hashing, so
an explicitly permitted partial collection cannot change the run's input identity.
Per-record JSON and NPZ SHA-256 are checked. Prefix decoding and CTC scoring do
not consume evaluation references. Results remain local unless reviewed for
publication. No waveform or model weights are automatically uploaded.

## Reproduction

Use a dedicated Python 3.12 environment. Install the repository and optional
acoustic dependencies as documented by the existing public collector. The session
used faster-whisper 1.2.1, CTranslate2 4.8.2, Transformers 4.57.6 and
pyopenjtalk-plus 0.4.1.post9; the collection manifest records the other packages.
A package list is an observed environment, not a hash-locked supply-chain proof.

```bash
python scripts/collect_phonetic_public_probe.py \
  --output ../semantic-asr-evidence/clean-v1 --per-split 48 \
  --dataset-revision 70bb2e84b976b7e960aa89f1c648e09c59f894dd \
  --phone-revision f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99

# Set TYPESAFE_API_KEY securely outside source control before opting into API calls.
python scripts/run_jev_phonetic_shadow.py \
  --probe-dir ../semantic-asr-evidence/clean-v1 \
  --output-dir ../semantic-asr-evidence/shadow-full96 \
  --allow-public-fleurs --allow-api --max-records 96 --max-calls 312 \
  --max-input-tokens 1000000 --max-wall-seconds 900

python scripts/run_jev_phonetic_shadow.py \
  --probe-dir ../semantic-asr-evidence/clean-v1 \
  --output-dir ../semantic-asr-evidence/shadow-joined96 \
  --allow-public-fleurs --allow-api --max-records 96 --max-calls 192 \
  --max-input-tokens 1000000 --max-wall-seconds 900 \
  --diagnostic-records 0 --main-arms joined joined_reordered

python -m pytest -q tests/test_jev_phonetic_shadow.py
python -m ruff check scripts/run_jev_phonetic_shadow.py tests/test_jev_phonetic_shadow.py
python -m ruff format --check scripts/run_jev_phonetic_shadow.py tests/test_jev_phonetic_shadow.py
```

## Report interpretation and remaining gates

CER uses NFKC and removes Unicode punctuation/whitespace, preserving case.
Report paired improve/tie/harm counts, candidate oracle coverage, raw/gated
abstention, order/repeat consistency, latency and actual usage. A reference G2P
phone edit metric, if added in analysis, is only an orthographic-pronunciation
proxy, NOT gold phoneme error rate. Same-phone spelling changes are not acoustic
proof. Missing gold candidates cannot be repaired by closed-set selection.

Dedicated contract-test success, full repository test success, actual inference,
accuracy improvement and production promotion are separate results. The pilot
already shows both an improvement and a harm, plus candidate-order dependence.
Full results and validation failures must be retained even when negative.

Before promotion: independently annotated and speaker/source-disjoint spontaneous
speech, acoustic failure analysis, preservation/critical-error gates, optional
model environments and the repository's existing evaluation/calibration/rights
requirements must be satisfied. Do not enable Jev or close all of #69 based on
these exposed public reading-speech trials.

Primary references: https://docs.typesafe.ai/api ; https://docs.typesafe.ai/models ;
https://huggingface.co/datasets/google/fleurs ;
https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 .
