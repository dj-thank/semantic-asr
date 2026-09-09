from dataclasses import replace

import pytest

from semantic_asr.audio_phone_runtime import AudioPhoneObservation, decoded_phone_runs
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence


def observation(symbols, *, offset=0):
    frames = tuple(
        PosteriorFrame.from_mapping(
            start_ms=offset + i * 20,
            end_ms=offset + (i + 1) * 20,
            probabilities={p: float(p == symbol) for p in ("PAD", "a", "i")},
        )
        for i, symbol in enumerate(symbols)
    )
    posterior = PosteriorSequence(
        "phone",
        "PAD",
        ("PAD", "a", "i"),
        frames,
        "test-encoder",
        "artifact:" + "a" * 64,
        "labels:" + "b" * 64,
        "c" * 64,
    )
    return AudioPhoneObservation(
        posterior=posterior,
        window_start_sample=offset * 16,
        window_end_sample=(offset + len(symbols) * 20) * 16,
        sample_rate=16000,
        input_pcm_sha256="d" * 64,
        preprocessing_sha256="e" * 64,
        model_artifact_sha256="a" * 64,
        runtime_revision="fixture-only",
        frame_stride_samples=320,
        receptive_field_samples=400,
    )


def test_blank_separated_repetitions_survive_ctc_decode_and_mora_grouping():
    observed = observation(["a", "a", "PAD", "a", "PAD", "i"])
    runs = decoded_phone_runs(observed.posterior)
    assert [r["phone"] for r in runs] == ["a", "a", "i"]
    assert [(r["start_ms"], r["end_ms"]) for r in runs] == [(0, 40), (60, 80), (100, 120)]
    result = observed.as_dict()
    assert [m["phones"] for m in result["moras"]] == [("a",), ("a",), ("i",)]
    assert result["mora_evidence"] == "derived-from-same-phone-posterior-not-independent"


def test_audio_window_and_pcm_identity_are_bound():
    original = observation(["a", "PAD", "i"])
    later = observation(["a", "PAD", "i"], offset=1000)
    assert later.digest != original.digest
    assert replace(original, input_pcm_sha256="f" * 64).digest != original.digest
    with pytest.raises(ValueError, match="window"):
        replace(original, window_start_sample=16000, window_end_sample=16960)


def test_overlapping_or_outside_frames_are_rejected():
    original = observation(["a", "i"])
    frame = replace(original.posterior.frames[1], start_ms=10)
    invalid = replace(original.posterior, frames=(original.posterior.frames[0], frame))
    with pytest.raises(ValueError, match="grid"):
        replace(original, posterior=invalid)
    with pytest.raises(ValueError, match="window"):
        replace(original, window_end_sample=320)


def test_unknown_symbols_and_blank_only_audio_remain_explicit():
    observed = observation(["PAD", "PAD"])
    assert observed.as_dict()["moras"] == []
    assert observed.as_dict()["status"] == "provisional"
    with pytest.raises(ValueError, match="model"):
        replace(observed, model_artifact_sha256="f" * 64)


@pytest.mark.parametrize("samples", [16007, 16008, 16015, 16016])
def test_rounded_recording_end_preserves_every_sample(samples):
    from semantic_asr.audio_phone_runtime import _window_end_sample

    assert _window_end_sample(samples, round(samples / 16)) == samples
    assert _window_end_sample(samples, None) == samples
    # An interior request remains exact; an out-of-range request is not clamped.
    assert _window_end_sample(samples, 900) == 14400
    assert _window_end_sample(samples, 1100) == 17600


def test_candidate_checks_use_audio_and_preserve_partial_failures():
    import json
    from types import SimpleNamespace

    from semantic_asr.audio_phone_runtime import check_candidate_pronunciations
    from semantic_asr.phonetic_evidence import CandidatePronunciation

    obs = observation(["a", "PAD", "i"])
    candidates = tuple(
        SimpleNamespace(candidate_id=name, text=text, metadata=meta)
        for name, text, meta in [
            ("right", "あい", {"decodeStartMs": 0, "decodeEndMs": 60}),
            ("wrong", "いあ", {"decodeStartMs": 0, "decodeEndMs": 60}),
            ("partial", "あ", {"decodeStartMs": 0, "decodeEndMs": 20}),
        ]
    )
    segment = SimpleNamespace(
        window=SimpleNamespace(start_ms=0, end_ms=60),
        observed=SimpleNamespace(
            candidates=candidates,
            selected_candidate_id="wrong",
            source_audio_sha256=obs.posterior.source_audio_sha256,
        ),
    )

    def pronounce(candidate):
        assert candidate.candidate_id != "partial"
        return CandidatePronunciation.create(
            candidate_id=candidate.candidate_id,
            text=candidate.text,
            kind="phone",
            symbols=("a", "i") if candidate.candidate_id == "right" else ("i", "a"),
            producer="fixture",
            producer_revision="1",
        )

    result = check_candidate_pronunciations(obs, segment, pronounce=pronounce)
    assert result["total_candidates"] == 3
    assert result["scored_candidates"] == 1  # wrong pronunciation has exactly zero support
    assert result["candidates"][0]["score"]["log_likelihood"] == 0.0
    assert "not finite" in result["candidates"][1]["reason"]
    assert "whole-window" in result["candidates"][2]["reason"]
    assert not result["text_selection_applied"]
    assert segment.observed.selected_candidate_id == "wrong"
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert (
        restored["candidates"][0]["score"]["evidence"]["provenance"]["metadata"]["posteriorDigest"]
        == obs.posterior.digest
    )
    with pytest.raises(ValueError, match="window"):
        check_candidate_pronunciations(
            observation(["a", "PAD", "i"], offset=1000), segment, pronounce=pronounce
        )
