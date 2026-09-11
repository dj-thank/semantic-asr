import hashlib

import pytest

from semantic_asr.realtime_reazon import (
    FinalUtterance,
    RealtimeDecodeInput,
    RealtimeEvent,
    RealtimeReazonConfig,
    RealtimeReazonSession,
)


def pcm(milliseconds: int, *, sample_rate: int = 16_000, value: int = 1) -> bytes:
    samples = round(sample_rate * milliseconds / 1000)
    return int(value).to_bytes(2, "little", signed=True) * samples


class RecordingDecoder:
    def __init__(self) -> None:
        self.requests: list[RealtimeDecodeInput] = []

    def __call__(self, request: RealtimeDecodeInput) -> str:
        self.requests.append(request)
        return f"{request.mode}:{len(request.pcm16le) // 2}"


def kinds(events):
    return [event.kind for event in events]


def test_partial_is_display_only_and_final_redecodes_exact_pcm():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(preroll_ms=0),
        session_id="s",
    )

    first = session.feed_pcm16(pcm(250), speech=True)
    second = session.feed_pcm16(pcm(250), speech=True)
    third = session.feed_pcm16(pcm(200), speech=False)
    fourth = session.feed_pcm16(pcm(150), speech=False)

    assert kinds(first) == ["speech_start"]
    assert kinds(second) == ["partial"]
    assert third == ()
    assert kinds(fourth) == ["final"]

    partial, final = decoder.requests
    assert partial.mode == "partial"
    assert final.mode == "final"
    assert len(partial.pcm16le) < len(final.pcm16le)
    assert fourth[0].text == final.mode + f":{len(final.pcm16le) // 2}"
    assert fourth[0].text != second[0].text
    assert fourth[0].audio_sha256 == hashlib.sha256(final.pcm16le).hexdigest()
    assert session.finals[0].text == fourth[0].text
    assert session.finals[0].final_digest == fourth[0].evidence_digest


def test_partial_cadence_is_bounded():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(partial_interval_ms=500, preroll_ms=0),
        session_id="s",
    )

    emitted = []
    for _ in range(9):
        emitted.extend(session.feed_pcm16(pcm(100), speech=True))

    assert kinds(emitted) == ["speech_start", "partial"]
    assert [request.mode for request in decoder.requests] == ["partial"]
    assert len(decoder.requests[0].pcm16le) // 2 == 8_000


def test_silence_finalizes_once_and_binds_digest():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(min_silence_ms=350, preroll_ms=0),
        session_id="s",
    )
    session.feed_pcm16(pcm(200), speech=True)
    assert session.feed_pcm16(pcm(200), speech=False) == ()
    events = session.feed_pcm16(pcm(150), speech=False)
    assert kinds(events) == ["final"]
    final = events[0]
    assert final.reason == "silence"
    assert final.audio_sha256 == session.finals[0].audio_sha256

    later = session.feed_pcm16(pcm(500), speech=False)
    assert "final" not in kinds(later)
    assert len([request for request in decoder.requests if request.mode == "final"]) == 1


def test_max_speech_forces_bounded_final():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(
            partial_interval_ms=100,
            max_speech_ms=300,
            min_silence_ms=100,
            preroll_ms=0,
        ),
        session_id="s",
    )

    session.feed_pcm16(pcm(100), speech=True)
    session.feed_pcm16(pcm(100), speech=True)
    events = session.feed_pcm16(pcm(100), speech=True)
    assert events[-1].kind == "final"
    assert events[-1].reason == "max-speech"
    assert session.active_utterance_id is None
    assert len(session.finals[0].pcm16le) // 2 == 4_800


def test_preroll_is_included_without_changing_partial_cadence():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(preroll_ms=200, partial_interval_ms=500),
        session_id="s",
    )
    session.feed_pcm16(pcm(300, value=2), speech=False)
    start = session.feed_pcm16(pcm(250, value=3), speech=True)
    assert kinds(start) == ["speech_start"]
    assert decoder.requests == []

    partial = session.feed_pcm16(pcm(250, value=3), speech=True)
    assert kinds(partial) == ["partial"]
    request = decoder.requests[0]
    assert request.start_sample == 1_600
    assert request.end_sample == 12_800
    assert len(request.pcm16le) // 2 == 11_200


def test_refine_is_separate_child_of_immutable_final():
    decoder = RecordingDecoder()
    seen: list[FinalUtterance] = []

    def refiner(final: FinalUtterance) -> str:
        seen.append(final)
        return "refined-text"

    session = RealtimeReazonSession(
        decoder,
        refiner=refiner,
        config=RealtimeReazonConfig(
            preroll_ms=0,
            min_silence_ms=100,
            refine_idle_ms=200,
        ),
        session_id="s",
    )
    session.feed_pcm16(pcm(200), speech=True)
    final_event = session.feed_pcm16(pcm(100), speech=False)[0]
    assert final_event.kind == "final"
    original = session.finals[0]

    assert session.feed_pcm16(pcm(100), speech=False) == ()
    refine_events = session.feed_pcm16(pcm(100), speech=False)
    assert kinds(refine_events) == ["refine"]
    refine = refine_events[0]
    assert refine.text == "refined-text"
    assert refine.audio_sha256 == original.audio_sha256
    assert refine.parent_final_digest == original.final_digest == final_event.evidence_digest
    assert session.finals[0] == original
    assert seen == [original]
    assert session.feed_pcm16(pcm(500), speech=False) == ()


def test_refiner_failure_emits_warning_once_without_mutating_final():
    decoder = RecordingDecoder()

    def broken(_final: FinalUtterance) -> str:
        raise RuntimeError("fixture")

    session = RealtimeReazonSession(
        decoder,
        refiner=broken,
        config=RealtimeReazonConfig(
            preroll_ms=0,
            min_silence_ms=100,
            refine_idle_ms=100,
        ),
        session_id="s",
    )
    session.feed_pcm16(pcm(100), speech=True)
    session.feed_pcm16(pcm(100), speech=False)
    original = session.finals[0]
    warning = session.feed_pcm16(pcm(100), speech=False)
    assert kinds(warning) == ["warning"]
    assert warning[0].reason == "refine-failed:RuntimeError"
    assert session.finals[0] == original
    assert session.feed_pcm16(pcm(500), speech=False) == ()


def test_reset_drops_old_audio_history_and_starts_new_identity():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(preroll_ms=0),
        session_id="old",
    )
    session.feed_pcm16(pcm(100), speech=True)
    reset_events = session.reset()
    assert kinds(reset_events) == ["final", "session_summary"]
    assert reset_events[-1].session_id == "old"
    assert session.session_id != "old"
    assert session.sample_cursor == 0
    assert session.finals == ()
    assert session.active_utterance_id is None

    next_events = session.feed_pcm16(pcm(100), speech=True)
    assert next_events[0].session_id == session.session_id
    assert next_events[0].sequence == 1
    assert next_events[0].utterance_id != reset_events[0].utterance_id


def test_history_is_bounded():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(
        decoder,
        config=RealtimeReazonConfig(
            preroll_ms=0,
            min_silence_ms=100,
            max_history_utterances=2,
        ),
        session_id="s",
    )
    for _ in range(3):
        session.feed_pcm16(pcm(100), speech=True)
        session.feed_pcm16(pcm(100), speech=False)
    assert [item.utterance_id for item in session.finals] == ["s:u2", "s:u3"]


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (RealtimeReazonConfig, None),
    ],
)
def test_default_config_is_constructible(config, error):
    if error is None:
        assert config().sample_rate == 16_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 8_000},
        {"partial_interval_ms": 0},
        {"min_silence_ms": 12_000},
        {"max_speech_ms": 30_001},
        {"preroll_ms": 12_001},
        {"max_chunk_ms": 0},
        {"max_history_utterances": 0},
    ],
)
def test_invalid_config_fails_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RealtimeReazonConfig(**kwargs)


def test_invalid_chunks_fail_closed_without_advancing_time():
    decoder = RecordingDecoder()
    session = RealtimeReazonSession(decoder, session_id="s")
    for bad in (b"", b"\0"):
        with pytest.raises(ValueError):
            session.feed_pcm16(bad, speech=False)
    with pytest.raises(TypeError):
        session.feed_pcm16(pcm(10), speech=1)
    with pytest.raises(ValueError):
        session.feed_pcm16(pcm(1_001), speech=False)
    assert session.sample_cursor == 0


def test_decode_input_rejects_digest_or_bounds_mismatch():
    data = pcm(10)
    digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match="digest"):
        RealtimeDecodeInput("s", "u", "final", data, 16_000, 0, 160, "0" * 64)
    with pytest.raises(ValueError, match="bounds"):
        RealtimeDecodeInput("s", "u", "final", data, 16_000, 0, 159, digest)


def test_refine_event_requires_parent_and_matching_shape():
    with pytest.raises(ValueError, match="parent"):
        RealtimeEvent(
            sequence=1,
            kind="refine",
            session_id="s",
            utterance_id="u",
            emitted_at_sample=1,
            text="x",
            audio_sha256="0" * 64,
        )
