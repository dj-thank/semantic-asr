# Grouped realtime refine scheduler (Issue #66)

Status (2026-10-01): the PR #65 first pass and PR #68 scheduler are connected by an **opt-in grouped runtime and WAV runner**. Software validation is separate from real-audio accuracy/latency evaluation. Default profiles and model promotion remain unchanged.

## Motivation

Hayamimi's realtime pipeline keeps the fast final visible and later re-decodes a larger utterance group once enough silence arrives. Its desktop refiner uses a 2 second group gap and a 25 second maximum group span, forces the last group at shutdown, and keeps expensive refine work off the immediate-final hot path.

Semantic ASR needs the same useful latency/quality separation, but with a stronger provenance rule: a refine result may be a child of first-pass evidence; it may not silently replace that evidence.

## Why grouped audio is not `b"".join(final.pcm)`

The realtime first pass can retain pre-roll and endpoint silence in each final buffer. Two neighboring final buffers can therefore overlap or contain duplicated context. Concatenating them would manufacture an audio sequence that never existed on the input timeline.

`BoundedPcmHistory` instead stores the raw PCM stream once with absolute sample positions. A grouped request reads exactly `[first_parent.start_sample, last_parent.end_sample)` from that continuous timeline. Silence and context between finals are therefore represented exactly once.

## Scheduler defaults

| control | default | purpose |
|---|---:|---|
| sample rate | 16,000 Hz | current realtime Reazon contract |
| idle gap | 2,000 ms | close group after a real VAD-confirmed pause |
| max group | 25,000 ms | prevent unbounded context/refine work |
| minimum group | 500 ms | skip second-pass work that is too short to justify |
| PCM history | 30,000 ms | covers 25 s group + 2 s idle detection with margin |
| pending finals | 64 | finite metadata bound |

The 2 s / 25 s values are Hayamimi-inspired starting points, not measured Semantic ASR optima. They stay behind an engineering boundary until #39/#29 paired evaluation proves a better setting or validates these defaults.

## Evidence model

Every parent final contributes:

- `utterance_id`
- immutable `final_digest`
- first-pass `audio_sha256`
- absolute start/end samples
- observed text for downstream display/debug context

Every `GroupedRefineRequest` contributes:

- ordered parent final proofs
- group/session identity and monotonic sequence
- exact continuous PCM bytes
- absolute group start/end samples
- `group_audio_sha256`
- close trigger (`idle-gap`, `max-duration`, `parent-limit`, `eof`, `reset`, `manual`)
- whether the group is eligible for decode, with explicit skip reason otherwise
- canonical request evidence digest

The request evidence digest binds the parent final digests, not a second unaudited copy of parent text. The parent final digest is the authoritative link back to the original first-pass event.

## VAD activity is part of the idle contract

Elapsed samples alone are not evidence of silence. A new utterance may remain active for longer than the 2 second refine gap, and closing the previous group merely because two seconds elapsed would make grouped refinement race the live speech path.

`GroupedRefineScheduler.feed_pcm16()` therefore accepts the caller's `speech` decision for each raw PCM chunk. Live integration must pass the **same serialized VAD decision** already used by the realtime Reazon session:

- `speech=True` resets accumulated idle silence and never causes an **idle** close; the independent maximum-span guard can still close finalized parents;
- only consecutive `speech=False` PCM counts toward the 2 second idle threshold;
- a new first-pass final resets the idle accumulator;
- malformed/non-boolean activity fails before PCM history advances.

The default non-speech value exists for deterministic/offline scheduler fixtures. Production/live callers must pass the VAD decision explicitly; wall-clock or sample distance must not be substituted for VAD-confirmed silence.

## Closing rules

1. Close with `idle-gap` only after consecutive VAD-confirmed non-speech PCM reaches 2 seconds. Any speech chunk resets that counter.
2. If adding the next final would make the group exceed 25 s, close the existing group first with `max-duration`, then start the next group with the new final.
3. If a group lands exactly on the 25 s bound, close immediately. Before appending PCM that would take the timeline more than 25 s past the oldest pending parent, close those finalized parents first. An ongoing next utterance must not evict their PCM before it finalizes. Never include unfinished speech or call this an idle pause.
4. If 64 parents are already pending, close before accepting another parent.
5. EOF/manual close never fabricates additional silence.
6. A group shorter than 500 ms is emitted as an explicit `group-too-short` skip instead of disappearing.
7. Reset requires the caller to choose `flush_pending=True` or `False`. A non-flushing reset emits a `GroupedRefineDiscard` receipt binding the discarded parent final digests.

## Failure policy

Fail closed when:

- PCM history is discontinuous;
- a supplied activity flag is not boolean;
- requested parent audio has already fallen outside retained continuous history;
- parent final order moves backward or its end does not strictly advance;
- SHA-256 fields are malformed;
- grouped PCM length/digest and sample bounds disagree;
- the configured PCM history cannot cover `max_group + idle_gap`;
- numeric bounds are invalid.

No reference transcript, gold text, candidate-derived dictionary, network lookup, or model download is involved in this module.

## Executable integration (2026-10-01)

`realtime_refine_runtime.GroupedRealtimeReazon` checks each first-pass event
against its retained `FinalUtterance` (session, text, final digest and PCM digest),
then verifies the PCM against continuous history before registering a parent.
The same serialized PCM/VAD decision feeds both paths.

A single lazy FIFO worker owns a **separate warm decoder**. It receives the
existing `RealtimeDecodeInput`: audio only, not parent text, references, hotwords
or an LLM instruction. The WAV runner constructs a second pinned
`ReazonSpeechK2Adapter` on that worker thread and reuses it across groups; it never
shares the native first-pass recognizer concurrently. Reset releases/reconstructs
the decoder on its owner thread after any active call returns.

`GroupedRefineOutcome` is a separate JSONL revision:

- `group_refine`: unscored `candidateText`, grouped request digest, ordered parent
  proofs, exact group PCM identity, pinned decoder identifier and outcome digest.
- `group_refine_warning`: `skipped`, `empty` or `error` with an explicit reason.
- `automaticallyApplied=false` and `independentEvidence=false` always. A second
  decode with the same model is NOT an independent second ear.
- Latency metadata is excluded from the digest. No correctness probability or
  inherited first-pass acoustic score is manufactured.

Fast finals are never overwritten, even if a candidate is shorter or more fluent.
This complements #73's time-aligned span-candidate generation, but does not import
its unmerged branch or imply admission through its verifier/lattice. Candidate
generation is not validated transcript selection. Identity hashes are provenance,
not proof of acoustic correctness.

### Running the opt-in WAV path

Use only an authorized local mono PCM16, 16 kHz WAV and approved existing local
Reazon/Silero artifacts with their actual SHA-256 identities:

```bash
python scripts/realtime_reazon.py "$AUDIO_WAV" \
  --model-dir "$REAZON_DIR" --model-sha256 "$REAZON_SHA256" \
  --vad-model "$SILERO_ONNX" --vad-model-sha256 "$SILERO_SHA256" \
  --allow-local-research --grouped-refine \
  --max-pending-groups 2 --refine-shutdown-timeout 5 \
  --events-jsonl runs/grouped-events.jsonl
```

The output path must not already exist. `--grouped-refine` defaults OFF. Without
it, the runner retains one decoder and its existing event/summary shape. Imports
and `--help` do not load models or require optional backends. This is WAV-based
simulated realtime, not a new microphone/Discord/WebSocket/subtitle integration.

Defaults remain 2 s VAD-confirmed idle, 25 s group, 0.5 s minimum group and 30 s
history. The idle control reuses `--refine-idle-ms`. In grouped mode,
`--max-speech` cannot exceed 24.968 s: one 32 ms boundary chunk must also fit inside
25 s. Default first pass is still 12 s with 800 ms pre-roll.

### Bounds, ordering, errors and lifecycle

Default capacity is **two total outstanding groups**, including queued, running
and completed-but-not-consumed work. Overload emits `queue-capacity`, never waits
on a blocking queue put. Short groups skip without loading a model. Retained
second-pass text is capped at 16,000 characters, not a backend's temporary memory
allocation while returning a string.

Accepted results are FIFO. Immediate overload/policy-skip receipts can precede
older still-running candidates; join by session/group/request identity rather
than arrival order. The owner polls on feed/flush/poll/close; the worker does not
call UI code. A result completed during synchronous first-pass decoding waits
for the next owner poll.

EOF publishes the last fast final **before** the bounded second-pass drain.
Error, empty output, overload or shutdown timeout keep first-pass finals and make
the runner summary `partial` with exit code 2. A policy-short group alone remains
a completed run with an explicit skip, not a measured recognition result.
Exception messages may contain private paths/speech and are not published; only
failure categories and exception class names are emitted.

If a factory or decoder unexpectedly terminates the worker (including
`SystemExit` or `KeyboardInterrupt` on that thread), every accepted unfinished
group receives an error outcome. Already completed outcomes remain intact.
Later submissions receive the same terminal error without restarting the model;
the owner must create a new runtime to resume refinement. These error outcomes
make the WAV runner report `partial`, even though the failed thread has exited.

A synchronous fast-decoder failure still raises the original exception. Groups
already closed by the scheduler are retained for `abort()` discard receipts.
The failed coordinator rejects further feed/flush/reset calls; `close()` aborts
without retrying that decoder. Immutable first-pass finals remain available.

`reset(flush_pending=True, timeout_seconds=...)` flushes/drains old work within
the deadline. `False` explicitly discards it. Both invalidate old results and
emit receipts before starting a new session ID. An old in-flight call still
counts against capacity until it returns; no replacement worker hides a stuck
call or exceeds the bound.

**A Python thread cannot forcibly interrupt native inference.** The deadline is
a bound on waiting, not proof of model termination or a hard inference timeout.
`worker_alive` / `workerStillRunning` expose this case; late results are discarded.
The daemon worker releases its decoder when the call returns and it exits or
changes session. First-pass calls remain synchronous. A separate model consumes
extra CPU/memory: scheduling separation does not establish unchanged real p95
latency. Cooperative/thread-safe backend assumptions need actual profiling.

### Reproduced failure and software tests

Previously, maximum group span was checked only when another final arrived. A
continuing next utterance could evict pending parents from the 30 s history first.
The regression uses scaled 3 s group / 4 s history limits and failed before the
fix. Closure now happens before the destructive append, without fabricated idle.

New model-free tests cover exact PCM and overlapping pre-roll, parent tampering,
history rollover, blocked refine while fast finals continue, FIFO/capacity,
lazy/warm decoder ownership, failure/empty/oversized text, reset isolation,
deadline receipts, EOF, default-off and synthetic-WAV runner integration.
Failure-receipt regressions additionally cover a fast partial/final exception
immediately after maximum-duration closure and terminal worker interruptions.
These are software tests, not real Reazon inference, training or CER evidence.
Exact commands, source identities, failures/skips and results belong in the PR.

### Remaining experiment / promotion boundary

For opt-in queue/service/factory timing and source/config identity capture, see
[grouped runtime measurement](GROUPED_RUNTIME_MEASUREMENT.md). This preparation
does not supply real-audio results or a transcript-quality evaluator.

Compare first-pass-only and grouped decode on identical, authorized,
source/speaker-disjoint recordings with pinned artifacts. Measure candidate
coverage/oracle CER separately from selected CER, false corrections, negation,
numbers/names, improve/tie/harm, fast-final/refine p50/p95, queue pressure, memory
and CPU. Inspected #70 examples remain exposed regression data. No default
promotion or established accuracy improvement follows from these tests.

## Reference and ownership

Hayamimi `oboroge0/hayamimi`, inspected commit
`09c8081420c7c88374eb24ae533ea470de794203`, `scripts/realtime_transcribe.py`
(`Refiner`, `GROUP_GAP_S`, `GROUP_MAX_S`) and `README.ja.md` motivated the
longer-context second pass, retained fast final and FIFO scheduling. Its benchmark
numbers are not Semantic ASR results. New worker/runtime code uses this
repository's contracts. Existing MIT notices and model-license boundaries remain
intact. The upstream Hayamimi repository is unchanged.
