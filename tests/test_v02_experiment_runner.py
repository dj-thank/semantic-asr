from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.experiment_runner import (
    AudioManifestRecord,
    CandidateGenerationConfig,
    finalize_generated_checkpoint,
    generate_candidates,
    generate_manifest_to_checkpoint,
    verify_manifest_isolation,
    write_generated_manifests,
)


class _MockAdapter:
    name = "mock-asr"
    model_name = "mock-asr-v1"
    model_revision = "fixture-model-sha"

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        assert request.language == "ja"
        assert request.hypotheses == 3
        return [
            CandidateEvidence(
                "path-a",
                "料金は3000円です",
                acoustic=-0.10,
                avg_logprob=-0.10,
                rank=1,
                hypothesis_count=3,
                source=self.name,
                metadata={
                    "scoreDomain": "fixture",
                    "cumulativeLogprob": -0.3,
                },
            ),
            CandidateEvidence(
                "path-a2",
                "料金は3000円です",
                acoustic=-0.20,
                avg_logprob=-0.20,
                rank=2,
                hypothesis_count=3,
                source=self.name,
                metadata={
                    "scoreDomain": "fixture",
                    "cumulativeLogprob": -0.6,
                },
            ),
            CandidateEvidence(
                "path-b",
                "料金は30000円です",
                acoustic=-0.30,
                avg_logprob=-0.30,
                rank=3,
                hypothesis_count=3,
                source=self.name,
                metadata={
                    "scoreDomain": "fixture",
                    "cumulativeLogprob": -0.9,
                },
            ),
        ]


def _record(path: Path, *, rights: str = "allow") -> AudioManifestRecord:
    return AudioManifestRecord(
        sample_id="sample-1",
        group_id="speaker-1",
        source_id="source-1",
        split="test",
        audio_path=str(path),
        reference="料金は3000円です",
        rights_decision=rights,
        license_id="fixture-license",
    )


def test_generate_candidates_hashes_audio_and_never_exports_source_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "audio.wav"
        audio.write_bytes(b"RIFF-fixture-audio")
        result = generate_candidates(
            _record(audio),
            _MockAdapter(),
            config=CandidateGenerationConfig(
                beam_size=3,
                hypotheses=3,
                model_revision="fixture-model-sha",
                runtime_revision="fixture-runtime-sha",
            ),
        )
        assert len(result.audio_sha256) == 64
        assert len(result.candidates) == 2
        assert result.candidates[0].metadata["audioSha256"] == result.audio_sha256
        benchmark = result.as_benchmark_row()
        rendered = json.dumps(benchmark, ensure_ascii=False)
        assert str(audio) not in rendered
        assert benchmark["generation"]["model"] == "mock-asr-v1"
        assert benchmark["generation"]["licenseId"] == "fixture-license"


def test_non_allow_rights_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "audio.wav"
        audio.write_bytes(b"audio")
        with pytest.raises(PermissionError, match="rights decision"):
            generate_candidates(
                _record(audio, rights="review"),
                _MockAdapter(),
                config=CandidateGenerationConfig(beam_size=3, hypotheses=3),
            )


def test_manifest_isolation_rejects_near_duplicate_leakage() -> None:
    records = [
        AudioManifestRecord(
            sample_id="train",
            group_id="speaker-a",
            source_id="source-a",
            split="train",
            audio_path="train.wav",
            reference="東京です",
            near_duplicate_id="duplicate-a",
            rights_decision="allow",
        ),
        AudioManifestRecord(
            sample_id="test",
            group_id="speaker-b",
            source_id="source-b",
            split="test",
            audio_path="test.wav",
            reference="東京です",
            near_duplicate_id="duplicate-a",
            rights_decision="allow",
        ),
    ]
    with pytest.raises(ValueError, match="near-duplicate leakage"):
        verify_manifest_isolation(records)


def test_resumable_generation_keeps_verified_prefix_after_failure(tmp_path) -> None:
    records = []
    for index in range(2):
        audio = tmp_path / f"audio-{index}.wav"
        audio.write_bytes(f"RIFF-{index}".encode())
        records.append(
            AudioManifestRecord(
                sample_id=f"sample-{index}",
                group_id=f"speaker-{index}",
                source_id=f"source-{index}",
                split="test",
                audio_path=str(audio),
                reference=f"参照{index}",
                rights_decision="allow",
            )
        )

    class _FailSecond(_MockAdapter):
        def __init__(self) -> None:
            self.calls = 0

        def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("fixture interruption")
            return super().decode(request)

    checkpoint = tmp_path / "candidates.jsonl.partial"
    config = CandidateGenerationConfig(beam_size=3, hypotheses=3)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        generate_manifest_to_checkpoint(
            records,
            _FailSecond(),
            config=config,
            checkpoint_path=checkpoint,
        )
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1

    resumed = _MockAdapter()
    rows = generate_manifest_to_checkpoint(
        records,
        resumed,
        config=config,
        checkpoint_path=checkpoint,
    )
    assert len(rows) == 2
    output = tmp_path / "candidates.jsonl"
    ranker = tmp_path / "ranker.jsonl"
    finalize_generated_checkpoint(
        checkpoint,
        output_path=output,
        ranker_path=ranker,
    )
    assert output.is_file()
    assert ranker.is_file()
    assert not checkpoint.exists()
    assert [json.loads(line)["sampleId"] for line in output.read_text("utf-8").splitlines()] == [
        "sample-0",
        "sample-1",
    ]


def test_resumable_generation_rejects_a_different_configuration(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    records = [_record(audio)]
    checkpoint = tmp_path / "candidates.jsonl.partial"
    generate_manifest_to_checkpoint(
        records,
        _MockAdapter(),
        config=CandidateGenerationConfig(beam_size=3, hypotheses=3),
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint configuration"):
        generate_manifest_to_checkpoint(
            records,
            _MockAdapter(),
            config=CandidateGenerationConfig(beam_size=4, hypotheses=3),
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize("trailer", [b'{"sampleId":', b""])
def test_resumable_generation_repairs_only_an_unterminated_trailing_row(
    tmp_path, trailer: bytes
) -> None:
    records = []
    for index in range(2):
        audio = tmp_path / f"torn-audio-{index}.wav"
        audio.write_bytes(f"RIFF-torn-{index}".encode())
        records.append(
            AudioManifestRecord(
                sample_id=f"torn-sample-{index}",
                group_id=f"speaker-{index}",
                source_id=f"source-{index}",
                split="test",
                audio_path=str(audio),
                reference=f"参照{index}",
                rights_decision="allow",
            )
        )

    checkpoint = tmp_path / "candidates.jsonl.partial"
    config = CandidateGenerationConfig(beam_size=3, hypotheses=3)
    generate_manifest_to_checkpoint(
        records[:1],
        _MockAdapter(),
        config=config,
        checkpoint_path=checkpoint,
    )
    payload = checkpoint.read_bytes()
    payload = payload + trailer if trailer else payload.rstrip(b"\r\n")
    checkpoint.write_bytes(payload)

    rows = generate_manifest_to_checkpoint(
        records,
        _MockAdapter(),
        config=config,
        checkpoint_path=checkpoint,
    )

    assert [row["sampleId"] for row in rows] == ["torn-sample-0", "torn-sample-1"]
    assert checkpoint.read_bytes().endswith(b"\n")
    reparsed = [json.loads(line) for line in checkpoint.read_text("utf-8").splitlines()]
    assert [row["sampleId"] for row in reparsed] == ["torn-sample-0", "torn-sample-1"]


def test_resumable_generation_rejects_a_corrupt_terminated_row(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    checkpoint = tmp_path / "candidates.jsonl.partial"
    checkpoint.write_bytes(b'{"sampleId":\n')

    with pytest.raises(json.JSONDecodeError):
        generate_manifest_to_checkpoint(
            [_record(audio)],
            _MockAdapter(),
            config=CandidateGenerationConfig(beam_size=3, hypotheses=3),
            checkpoint_path=checkpoint,
        )


def test_generated_benchmark_and_ranker_manifests_are_compatible_jsonl() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "audio.wav"
        audio.write_bytes(b"RIFF-fixture-audio")
        generated = generate_candidates(
            _record(audio),
            _MockAdapter(),
            config=CandidateGenerationConfig(beam_size=3, hypotheses=3),
        )
        benchmark = root / "benchmark.jsonl"
        ranker = root / "ranker.jsonl"
        write_generated_manifests(
            [generated],
            benchmark_path=benchmark,
            ranker_path=ranker,
        )
        benchmark_row = json.loads(benchmark.read_text(encoding="utf-8"))
        ranker_row = json.loads(ranker.read_text(encoding="utf-8"))
        assert benchmark_row["split"] == "test"
        assert benchmark_row["reference"] == "料金は3000円です"
        assert ranker_row["exampleId"] == "sample-1"
        assert ranker_row["audioSha256"] == generated.audio_sha256
