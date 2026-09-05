from __future__ import annotations

import math

import pytest

from semantic_asr.phonetic_evidence import (
    CandidatePronunciation,
    PhoneticCandidateEvidence,
    PosteriorFrame,
    PosteriorSequence,
    ctc_pronunciation_score,
    rank_candidate_pronunciations,
)


def frame(index: int, winner: str, vocabulary: tuple[str, ...]) -> PosteriorFrame:
    high = 0.88
    rest = (1.0 - high) / (len(vocabulary) - 1)
    return PosteriorFrame.from_mapping(
        start_ms=index * 20,
        end_ms=(index + 1) * 20,
        probabilities={symbol: high if symbol == winner else rest for symbol in vocabulary},
    )


def phone_posterior() -> PosteriorSequence:
    vocabulary = ("<blk>", "m", "a", "d", "t")
    winners = ("<blk>", "m", "a", "<blk>", "d", "a", "<blk>")
    return PosteriorSequence(
        kind="phone",
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frame(index, winner, vocabulary) for index, winner in enumerate(winners)),
        encoder="test-encoder",
        encoder_revision="enc-r1",
        label_set_revision="phones-r1",
        source_audio_sha256="a" * 64,
    )


def mora_posterior() -> PosteriorSequence:
    vocabulary = ("<blk>", "マ", "ダ", "タ")
    winners = ("<blk>", "マ", "<blk>", "ダ", "<blk>")
    return PosteriorSequence(
        kind="mora",
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frame(index, winner, vocabulary) for index, winner in enumerate(winners)),
        encoder="test-encoder",
        encoder_revision="enc-r1",
        label_set_revision="mora-r1",
        source_audio_sha256="a" * 64,
    )


def pronunciation(candidate_id: str, text: str, kind: str, symbols: tuple[str, ...]):
    return CandidatePronunciation.create(
        candidate_id=candidate_id,
        text=text,
        kind=kind,  # type: ignore[arg-type]
        symbols=symbols,
        producer="test-g2p",
        producer_revision="r1",
    )


def test_ctc_prefers_audio_supported_phone_sequence() -> None:
    posterior = phone_posterior()
    mada = ctc_pronunciation_score(
        posterior,
        pronunciation("mada", "まだ", "phone", ("m", "a", "d", "a")),
    )
    mata = ctc_pronunciation_score(
        posterior,
        pronunciation("mata", "また", "phone", ("m", "a", "t", "a")),
    )

    assert mada.log_likelihood > mata.log_likelihood
    assert mada.mean_frame_log_likelihood > mata.mean_frame_log_likelihood
    assert mada.evidence.metadata["posteriorDigest"] == posterior.digest
    assert math.isfinite(mada.evidence.value)


def test_phone_and_mora_remain_separate_evidence_domains() -> None:
    phone = ctc_pronunciation_score(
        phone_posterior(),
        pronunciation("mada", "まだ", "phone", ("m", "a", "d", "a")),
    )
    mora = ctc_pronunciation_score(
        mora_posterior(),
        pronunciation("mada", "まだ", "mora", ("マ", "ダ")),
    )
    bundle = PhoneticCandidateEvidence(candidate_id="mada", phone=phone, mora=mora)

    assert {score.source.split(":", 1)[0] for score in bundle.scores} == {
        "ctc-phone",
        "ctc-mora",
    }


def test_rank_candidate_pronunciations_is_deterministic() -> None:
    posterior = phone_posterior()
    rows = rank_candidate_pronunciations(
        posterior,
        (
            pronunciation("mata", "また", "phone", ("m", "a", "t", "a")),
            pronunciation("mada", "まだ", "phone", ("m", "a", "d", "a")),
        ),
    )

    assert [row.candidate_id for row in rows] == ["mada", "mata"]


def test_candidate_pronunciation_is_bound_to_exact_text() -> None:
    row = CandidatePronunciation.create(
        candidate_id="x",
        text="仕様",
        kind="mora",
        symbols=("シ", "ヨ", "ー"),
        producer="lexicon",
        producer_revision="r1",
    )

    with pytest.raises(ValueError, match="exact candidate text"):
        CandidatePronunciation(
            candidate_id=row.candidate_id,
            text="使用",
            kind=row.kind,
            symbols=row.symbols,
            source_text_sha256=row.source_text_sha256,
            producer=row.producer,
            producer_revision=row.producer_revision,
        )


def test_posterior_requires_full_frozen_vocabulary_per_frame() -> None:
    with pytest.raises(ValueError, match="frozen vocabulary exactly"):
        PosteriorSequence(
            kind="phone",
            blank_symbol="<blk>",
            vocabulary=("<blk>", "a"),
            frames=(
                PosteriorFrame.from_mapping(
                    start_ms=0,
                    end_ms=20,
                    probabilities={"<blk>": 1.0},
                ),
            ),
            encoder="enc",
            encoder_revision="r1",
            label_set_revision="l1",
            source_audio_sha256="a" * 64,
        )


def test_ctc_rejects_unknown_pronunciation_symbol() -> None:
    with pytest.raises(ValueError, match="unknown symbols"):
        ctc_pronunciation_score(
            phone_posterior(),
            pronunciation("x", "未知", "phone", ("z",)),
        )
