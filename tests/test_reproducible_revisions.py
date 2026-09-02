from __future__ import annotations

import sys
from types import ModuleType

import pytest

from semantic_asr.adapters import (
    DecodeRequest,
    FasterWhisperAdapter,
    Qwen3ASRAdapter,
    Qwen3ForcedAlignerAdapter,
)
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.experiment_runner import (
    AudioManifestRecord,
    CandidateGenerationConfig,
    audio_record_from_row,
    generate_candidates,
)
from semantic_asr.revisions import QWEN_FORCED_ALIGNER_REVISIONS, sha256_artifact


def test_faster_whisper_revision_is_bound_to_the_loaded_model(monkeypatch) -> None:
    calls: dict[str, object] = {}
    module = ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            calls.update({"model": model, **kwargs})

    module.WhisperModel = _WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    adapter = FasterWhisperAdapter(
        model="publisher/model",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        device="cpu",
        compute_type="int8",
        cpu_threads=6,
    )

    assert calls["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert adapter.model_revision == calls["revision"]
    assert calls["cpu_threads"] == 6


def test_unknown_faster_whisper_hub_model_requires_an_exact_revision(monkeypatch) -> None:
    module = ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, _model: str, **_kwargs: object) -> None:
            raise AssertionError("unknown floating Hub model must fail before loading")

    module.WhisperModel = _WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    with pytest.raises(ValueError, match="exact 40-character revision"):
        FasterWhisperAdapter(model="publisher/unknown-model", device="cpu")


def test_qwen_revision_resolves_one_snapshot_for_model_and_processor(monkeypatch) -> None:
    snapshot_calls: dict[str, object] = {}
    model_calls: dict[str, object] = {}

    hub = ModuleType("huggingface_hub")

    def _snapshot_download(*, repo_id: str, revision: str) -> str:
        snapshot_calls.update({"repo_id": repo_id, "revision": revision})
        return "C:/cache/snapshots/revision"

    hub.snapshot_download = _snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    qwen = ModuleType("qwen_asr")

    class _Qwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: object) -> object:
            model_calls.update({"model": model, **kwargs})
            return object()

    qwen.Qwen3ASRModel = _Qwen3ASRModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)

    torch = ModuleType("torch")
    torch.float16 = object()  # type: ignore[attr-defined]
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    adapter = Qwen3ASRAdapter(
        model="Qwen/Qwen3-ASR-0.6B",
        model_revision="fedcba9876543210fedcba9876543210fedcba98",
        dtype="auto",
        device_map="cpu",
    )

    assert snapshot_calls["revision"] == "fedcba9876543210fedcba9876543210fedcba98"
    assert model_calls["model"] == "C:/cache/snapshots/revision"
    assert adapter.model_revision == snapshot_calls["revision"]


def test_unknown_qwen_hub_model_requires_an_exact_revision(monkeypatch) -> None:
    qwen = ModuleType("qwen_asr")

    class _Qwen3ASRModel:
        @classmethod
        def from_pretrained(cls, _model: str, **_kwargs: object) -> object:
            raise AssertionError("unknown floating Hub model must fail before loading")

    qwen.Qwen3ASRModel = _Qwen3ASRModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)

    with pytest.raises(ValueError, match="exact 40-character revision"):
        Qwen3ASRAdapter(model="publisher/unknown-qwen", dtype="auto", device_map="cpu")


def test_forced_aligner_uses_the_known_exact_revision(monkeypatch) -> None:
    calls: dict[str, object] = {}
    qwen = ModuleType("qwen_asr")

    class _Qwen3ForcedAligner:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: object) -> object:
            calls.update({"model": model, **kwargs})
            return object()

    qwen.Qwen3ForcedAligner = _Qwen3ForcedAligner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)

    torch = ModuleType("torch")
    torch.float16 = object()  # type: ignore[attr-defined]
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    adapter = Qwen3ForcedAlignerAdapter(dtype="auto", device_map="cpu")

    assert calls["model"] == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert calls["revision"] == QWEN_FORCED_ALIGNER_REVISIONS["Qwen/Qwen3-ForcedAligner-0.6B"]
    assert adapter.model_revision == calls["revision"]


def test_unknown_forced_aligner_hub_model_requires_an_exact_revision(monkeypatch) -> None:
    qwen = ModuleType("qwen_asr")

    class _Qwen3ForcedAligner:
        @classmethod
        def from_pretrained(cls, _model: str, **_kwargs: object) -> object:
            raise AssertionError("unknown floating Hub model must fail before loading")

    qwen.Qwen3ForcedAligner = _Qwen3ForcedAligner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)

    with pytest.raises(ValueError, match="exact 40-character revision"):
        Qwen3ForcedAlignerAdapter(
            model="publisher/unknown-aligner",
            dtype="auto",
            device_map="cpu",
        )


def test_local_model_directory_uses_a_separate_verified_artifact_digest(
    monkeypatch, tmp_path
) -> None:
    calls: dict[str, object] = {}
    module = ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            calls.update({"model": model, **kwargs})

    module.WhisperModel = _WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model": "fixture"}\n', encoding="utf-8")
    artifact_sha256 = sha256_artifact(model)

    adapter = FasterWhisperAdapter(
        model=str(model),
        model_artifact_sha256=artifact_sha256,
        device="cpu",
        compute_type="int8",
    )

    assert calls["model"] == str(model)
    assert calls["revision"] is None
    assert adapter.model_revision is None
    assert adapter.model_artifact_sha256 == artifact_sha256
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FasterWhisperAdapter(model=str(model), model_artifact_sha256="a" * 64)


def test_score_domain_changes_with_decode_controls(monkeypatch) -> None:
    module = ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, _model: str, **_kwargs: object) -> None:
            pass

    module.WhisperModel = _WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    request = DecodeRequest(audio_path="fixture.wav", beam_size=5, hypotheses=3)
    first = FasterWhisperAdapter(
        model="publisher/model",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        runtime_revision="runtime-r1",
        length_penalty=1.0,
        device="cpu",
        compute_type="int8",
    )
    second = FasterWhisperAdapter(
        model="publisher/model",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        runtime_revision="runtime-r1",
        length_penalty=1.1,
        device="cpu",
        compute_type="int8",
    )
    third = FasterWhisperAdapter(
        model="publisher/model",
        model_revision="fedcba9876543210fedcba9876543210fedcba98",
        runtime_revision="runtime-r1",
        length_penalty=1.0,
        device="cpu",
        compute_type="int8",
    )

    assert first._score_domain(request, language="ja") != second._score_domain(
        request, language="ja"
    )
    assert first._score_domain(request, language="ja") != third._score_domain(
        request, language="ja"
    )


def test_generation_rejects_a_revision_label_that_does_not_match_the_adapter(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    class _Adapter:
        name = "fixture"
        model_name = "fixture-model"
        model_revision = "actual-revision"

        def decode(self, _request: object) -> list[CandidateEvidence]:
            raise AssertionError("revision mismatch must fail before inference")

    record = AudioManifestRecord(
        sample_id="s",
        group_id="g",
        source_id="source",
        split="test",
        audio_path=str(audio),
        reference="参照",
        rights_decision="allow",
    )

    with pytest.raises(ValueError, match="model revision"):
        generate_candidates(
            record,
            _Adapter(),
            config=CandidateGenerationConfig(model_revision="claimed-revision"),
        )


def test_generation_rejects_revision_provenance_from_an_unbound_adapter(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    class _UnboundAdapter:
        name = "unbound"

        def decode(self, _request: object) -> list[CandidateEvidence]:
            raise AssertionError("an unbound adapter must fail before inference")

    record = AudioManifestRecord(
        sample_id="s",
        group_id="g",
        source_id="source",
        split="test",
        audio_path=str(audio),
        reference="参照",
        rights_decision="allow",
    )
    with pytest.raises(ValueError, match="model revision"):
        generate_candidates(
            record,
            _UnboundAdapter(),
            config=CandidateGenerationConfig(model_revision="claimed-revision"),
        )


def test_local_directory_cannot_claim_an_unrelated_hub_revision(monkeypatch, tmp_path) -> None:
    module = ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, _model: str, **_kwargs: object) -> None:
            raise AssertionError("invalid local provenance must fail before model loading")

    module.WhisperModel = _WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="local model directory"):
        FasterWhisperAdapter(model=str(model), model_revision="a" * 40)


def test_dataset_revision_survives_manifest_loading() -> None:
    record = audio_record_from_row(
        {
            "sampleId": "s",
            "groupId": "g",
            "sourceId": "source",
            "split": "test",
            "audioPath": "audio.wav",
            "reference": "参照",
            "datasetName": "publisher/dataset",
            "datasetRevision": "0123456789abcdef0123456789abcdef01234567",
        }
    )

    assert record.dataset_name == "publisher/dataset"
    assert record.dataset_revision == "0123456789abcdef0123456789abcdef01234567"
