from __future__ import annotations

import json
from pathlib import Path

from _phonetic_runtime_fixture import write_wav

from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.phonetic_runtime.evaluation import evaluate_phonetic_runtime
from semantic_asr.phonetic_runtime.manifest import (
    PhoneticManifestRow,
    PhoneticSplitManifest,
)


class PerfectRuntime:
    profile_digest = "d" * 64

    def infer(self, audio_path, **kwargs):
        source = kwargs["expected_source_audio_sha256"]
        return make_posterior("phone", ("k", "a"), source), make_posterior("mora", ("カ",), source)


def make_posterior(kind: str, symbols: tuple[str, ...], source: str) -> PosteriorSequence:
    vocabulary = ("<blk>", *symbols)
    sequence = ["<blk>"]
    for symbol in symbols:
        sequence.extend((symbol, "<blk>"))
    frames = tuple(
        PosteriorFrame.from_mapping(
            start_ms=index * 20,
            end_ms=(index + 1) * 20,
            probabilities={value: 1.0 if value == winner else 0.0 for value in vocabulary},
        )
        for index, winner in enumerate(sequence)
    )
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=frames,
        encoder="perfect",
        encoder_revision="r1",
        label_set_revision=f"{kind}-r1",
        source_audio_sha256=source,
    )


def manifest(tmp_path: Path) -> PhoneticSplitManifest:
    rows = []
    for index, split in enumerate(("train", "calibration", "test"), 1):
        path = tmp_path / f"{split}.wav"
        digest = write_wav(path, frequency=200.0 + index * 100)
        rows.append(
            PhoneticManifestRow(
                utterance_id=split,
                audio_path=path.resolve(),
                source_audio_sha256=digest,
                sample_rate=16_000,
                phone_symbols=("k", "a"),
                mora_symbols=("カ",),
                speaker_id=f"{split}-speaker",
                session_id=f"{split}-session",
                source_id=f"{split}-source",
                license_id="fixture-license",
                rights_decision="allow",
                split=split,  # type: ignore[arg-type]
            )
        )
    return PhoneticSplitManifest(
        name="fixture",
        revision="r1",
        rows=tuple(rows),
        source_manifest_sha256="a" * 64,
    )


def test_phone_and_mora_errors_are_reported_separately(tmp_path: Path) -> None:
    report = evaluate_phonetic_runtime(
        PerfectRuntime(),  # type: ignore[arg-type]
        manifest(tmp_path),
        split="test",
    )

    assert report.phone_error_rate == 0.0
    assert report.mora_error_rate == 0.0
    assert len(report.utterances) == 1


def test_prediction_text_is_hash_only_by_default(tmp_path: Path) -> None:
    report = evaluate_phonetic_runtime(
        PerfectRuntime(),  # type: ignore[arg-type]
        manifest(tmp_path),
        split="test",
    )
    destination = report.write(tmp_path / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    row = payload["utterances"][0]

    assert "phonePrediction" not in row
    assert "moraPrediction" not in row
    assert "phonePredictionSha256" in row

    raw_destination = report.write(
        tmp_path / "report-local.json",
        include_predictions=True,
    )
    raw = json.loads(raw_destination.read_text(encoding="utf-8"))
    assert "phonePrediction" in raw["utterances"][0]
