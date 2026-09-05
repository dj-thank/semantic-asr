"""Exercise real orchestration rather than hand-built evidence payloads."""

from __future__ import annotations

import json
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest, MockASRAdapter
from semantic_asr.adapters_v2 import DecodeVariant, PathDecodeRequest
from semantic_asr.api import RuntimeProfile, calibrated_confidence, runtime_profile, transcribe
from semantic_asr.contracts import CandidateEvidence, NormalizedTranscript
from semantic_asr.global_scorer import CallableGlobalSequenceScorer
from semantic_asr.japanese import join_timed_fragments
from semantic_asr.longform import plan_windows
from semantic_asr.longform_deliberation import (
    LongformDeliberationConfig,
    apply_longform_deliberation,
)
from semantic_asr.outputs import atomic_write, write_outputs


def actual_result(tmp_path: Path):
    audio = tmp_path / "actual.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 32000)
    rows = [
        CandidateEvidence("spoken", "昨日学校を行きました。", acoustic=0.9, mora=0.9),
        CandidateEvidence("other", "昨日学校に行きました。", acoustic=0.88, mora=0.88),
    ]
    profile = RuntimeProfile(name="test", description="fixture", window_ms=1000, overlap_ms=0)
    return transcribe(audio, profile=profile, adapter=MockASRAdapter(rows)), audio


def test_real_nonoverlapping_windows_preserve_deliberate_repetition(tmp_path):
    result, _ = actual_result(tmp_path)
    assert len(result.segments) == 2
    assert result.observed_text == "昨日学校を行きました。昨日学校を行きました。"
    result.longform.verify()
    assert result.longform.evidence_schema == "semantic-asr-longform-evidence-v2"


@pytest.mark.parametrize("second_start", [1000, 1100])
def test_equal_text_is_not_evidence_of_temporal_overlap(second_start):
    assert (
        join_timed_fragments([(0, 1000, "はい。"), (second_start, 2000, "はい。")])
        == "はい。はい。"
    )


def test_temporally_overlapping_prefix_can_still_be_joined():
    assert (
        join_timed_fragments([(0, 1000, "今日は学校"), (800, 1800, "学校です")]) == "今日は学校です"
    )


def test_actual_first_pass_runs_through_second_pass_contract(tmp_path):
    facade, audio = actual_result(tmp_path)
    first_pass = facade.longform
    scorer = CallableGlobalSequenceScorer(
        lambda path, context: 0.0, source="fixture", profile_digest="f" * 64
    )
    result = apply_longform_deliberation(first_pass, sequence_scorer=scorer, audio_path=audio)
    result.verify()
    assert result.first_pass_evidence_sha256 == first_pass.evidence_sha256
    assert any(segment.trace.attempted for segment in result.segments)
    assert first_pass.observed_text == facade.observed_text
    paths = write_outputs(result, tmp_path / "second-pass", formats={"json", "observed"})
    assert json.loads(Path(paths["json"]).read_text())["observed_text"] == result.observed_text


def test_second_pass_rejects_another_recording_before_provider_execution(tmp_path):
    facade, audio = actual_result(tmp_path)
    audio.write_bytes(b"different source audio")
    with pytest.raises(ValueError, match="source recording"):
        apply_longform_deliberation(facade.longform, audio_path=audio)


@pytest.mark.parametrize(
    "field,value", [("observed_text", "改変"), ("normalized_text", "改変"), ("duration_ms", 3000)]
)
def test_first_pass_root_binds_both_text_channels_and_time(tmp_path, field, value):
    result, _ = actual_result(tmp_path)
    tampered = replace(result.longform, **{field: value})
    with pytest.raises(ValueError):
        tampered.verify()


def test_window_retiming_and_normalization_metadata_are_hash_bound(tmp_path):
    result, _ = actual_result(tmp_path)
    segment = result.longform.segments[0]
    changed = replace(segment, window=replace(segment.window, end_ms=999))
    with pytest.raises(ValueError, match="hash"):
        replace(result.longform, segments=(changed, *result.longform.segments[1:])).verify()
    changed = replace(
        segment,
        normalized=replace(segment.normalized, semantic_change_warnings=("tampered provenance",)),
    )
    with pytest.raises(ValueError, match="hash"):
        replace(result.longform, segments=(changed, *result.longform.segments[1:])).verify()


def test_rank_only_requires_existing_candidate_identity(tmp_path):
    result, _ = actual_result(tmp_path)
    observed = result.longform.segments[0].observed
    with pytest.raises(ValueError, match="candidate"):
        NormalizedTranscript.attach(observed, text="創作した文章", mode="rank-only")
    with pytest.raises(ValueError):
        NormalizedTranscript.attach(observed, text=observed.text, mode="made-up-mode")


def test_export_rejects_tampered_facade_before_creating_files(tmp_path):
    result, _ = actual_result(tmp_path)
    with pytest.raises(ValueError, match="facade"):
        replace(result, observed_text="創作した文章").write(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_empty_formats_means_no_outputs(tmp_path):
    result, _ = actual_result(tmp_path)
    assert result.write(tmp_path / "empty", formats=set()) == {}
    assert not (tmp_path / "empty").exists()


def test_all_destinations_are_preflighted_including_facade_json(tmp_path):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    root.mkdir()
    existing = root / "actual.transcript.json"
    existing.write_text("old content")
    with pytest.raises(FileExistsError):
        result.write(root)
    assert list(root.iterdir()) == [existing]
    assert existing.read_text() == "old content"


def test_json_failure_cannot_leave_half_an_export(tmp_path):
    result, _ = actual_result(tmp_path)
    result.provenance["nonFinite"] = float("nan")
    with pytest.raises(ValueError):
        result.write(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_concurrent_exclusive_writers_never_clobber_each_other(tmp_path):
    target = tmp_path / "exclusive.txt"

    def write(index):
        try:
            atomic_write(target, str(index) * 10000)
            return index
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(16)))
    winners = [index for index in results if index is not None]
    assert len(winners) == 1
    assert target.read_text() == str(winners[0]) * 10000
    assert list(tmp_path.iterdir()) == [target]


def test_failed_atomic_publication_cleans_its_temporary_file(tmp_path, monkeypatch):
    import semantic_asr.outputs as outputs

    def fail(*args):
        raise OSError("injected link failure")

    monkeypatch.setattr(outputs.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        atomic_write(tmp_path / "failed.txt", "new content")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [True, 1.2, float("nan")])
def test_window_and_decoder_integer_bounds_are_strict(bad):
    for build in (
        lambda: plan_windows(bad),
        lambda: RuntimeProfile(name="bad", description="", window_ms=bad),
        lambda: PathDecodeRequest(audio_path="x", start_ms=bad),
        lambda: LongformDeliberationConfig(maximum_left_windows=bad),
    ):
        with pytest.raises((TypeError, ValueError)):
            build()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_nonfinite_decode_values_and_calibration_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        DecodeVariant("bad", patience=bad)
    with pytest.raises((TypeError, ValueError)):
        RuntimeProfile(name="bad", description="", confidence_calibration=(bad, 0.0))


def test_zero_duration_is_not_treated_as_an_omitted_argument(tmp_path):
    _, audio = actual_result(tmp_path)
    with pytest.raises(ValueError, match="duration_ms"):
        transcribe(audio, duration_ms=0, adapter=MockASRAdapter([CandidateEvidence("a", "はい")]))


def test_confidence_is_disabled_for_unmeasured_profiles_and_numerically_stable():
    assert runtime_profile("gpu-ja-v1").confidence_calibration is None
    assert runtime_profile("cpu-ja-quality-v1").confidence_calibration is None
    assert RuntimeProfile(name="custom", description="").confidence_calibration is None
    extreme = RuntimeProfile(name="extreme", description="", confidence_calibration=(1000.0, 0.0))
    assert calibrated_confidence(extreme, 0.01) == 0.0
    assert calibrated_confidence(extreme, 0.99) == 1.0
    assert calibrated_confidence(extreme, True) is None


@pytest.mark.xfail(
    strict=True,
    reason="Legacy DecodeRequest still accepts invalid beam sizes; legacy adapter repair deferred.",
)
@pytest.mark.parametrize("bad", [True, 1.2, float("nan")])
def test_legacy_decode_request_integer_validation_gap(bad):
    with pytest.raises((TypeError, ValueError)):
        DecodeRequest(audio_path="x", beam_size=bad)
