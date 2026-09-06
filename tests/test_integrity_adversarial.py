"""Public boundary and provenance regressions independent of model downloads."""

from __future__ import annotations

import wave
from dataclasses import replace

import pytest

from semantic_asr.adapters import MockASRAdapter
from semantic_asr.advanced_adapters import LoopGuardConfig, PathPreservingFasterWhisperAdapter
from semantic_asr.api import (
    PROFILES,
    RuntimeProfile,
    _candidate_spans,
    _confidence_eligible,
    load_transcriber,
    runtime_profile,
    transcribe,
)
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.fusion import FusionConfig
from semantic_asr.planner import EvidenceBudget


@pytest.mark.parametrize("spans", [1, True, "bad", {"text": "not a sequence"}, None])
def test_malformed_timestamp_containers_fall_back_instead_of_crashing(spans):
    candidate = CandidateEvidence("a", "発話", metadata={"utteranceSpans": spans})
    assert _candidate_spans(candidate) == []


def measured_runtime():
    """A configuration-only object: no model is instantiated or evaluated here."""
    profile = runtime_profile("cpu-ja-v1")
    adapter = object.__new__(PathPreservingFasterWhisperAdapter)
    for name, value in {
        "model_name": profile.model,
        "model_revision": profile.model_revision,
        "device": profile.device,
        "compute_type": profile.compute_type,
        "patience": profile.patience,
        "loop_guard": LoopGuardConfig(),
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "without_timestamps": False,
        "cpu_threads": 0,
    }.items():
        setattr(adapter, name, value)
    return profile, adapter, load_transcriber(profile, adapter=adapter)


def eligible(profile, adapter, transcriber):
    return _confidence_eligible(
        profile, adapter, language="ja", prompted=False, transcriber=transcriber
    )


def test_default_configuration_is_eligible_but_not_a_new_accuracy_measurement():
    assert eligible(*measured_runtime())


@pytest.mark.parametrize(
    "name,value",
    [
        ("length_penalty", 0.7),
        ("repetition_penalty", 1.2),
        ("no_repeat_ngram_size", 3),
        ("without_timestamps", True),
        ("cpu_threads", 2),
        ("loop_guard", LoopGuardConfig(extra_samples=4)),
        ("loop_guard", LoopGuardConfig(compression_ratio_threshold=3.0)),
    ],
)
def test_modified_decoder_cannot_inherit_measured_confidence(name, value):
    profile, adapter, transcriber = measured_runtime()
    setattr(adapter, name, value)
    assert not eligible(profile, adapter, transcriber)


@pytest.mark.parametrize(
    "name,value",
    [
        ("surface_policy", "exact"),
        ("fusion_config", FusionConfig(posterior_temperature=0.5)),
        ("evidence_budget", EvidenceBudget(total_cost_ms=1, max_actions=1)),
        ("evidence_enricher", lambda row: row),
        ("second_ear", object()),
        ("teacher", object()),
        ("forced_aligner", object()),
    ],
)
def test_modified_pipeline_cannot_inherit_measured_confidence(name, value):
    profile, adapter, transcriber = measured_runtime()
    setattr(transcriber, name, value)
    assert not eligible(profile, adapter, transcriber)


def test_registry_replacement_cannot_redefine_which_profile_was_calibrated(monkeypatch):
    profile, adapter, transcriber = measured_runtime()
    replacement = replace(profile, confidence_calibration=(10.0, 10.0))
    monkeypatch.setitem(PROFILES, "cpu-ja-v1", replacement)
    assert not eligible(replacement, adapter, transcriber)


def actual_result(tmp_path):
    audio = tmp_path / "actual.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 32000)
    profile = RuntimeProfile(name="test", description="fixture", window_ms=1000, overlap_ms=0)
    return transcribe(
        audio, profile=profile, adapter=MockASRAdapter([CandidateEvidence("a", "はい。")])
    ), audio


@pytest.mark.parametrize("change", [{"status": "invented"}, {"index": 99}])
def test_facade_cannot_relabel_verified_segment_state(tmp_path, change):
    result, _ = actual_result(tmp_path)
    changed = replace(result.segments[0], **change)
    with pytest.raises(ValueError, match="facade segment"):
        replace(result, segments=(changed, *result.segments[1:])).verify()


def test_facade_source_name_must_match_longform(tmp_path):
    result, _ = actual_result(tmp_path)
    with pytest.raises(ValueError, match="source_name"):
        replace(result, source_name="another.wav").verify()
