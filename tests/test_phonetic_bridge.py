from __future__ import annotations

from semantic_asr.multilevel_lattice import UtilityCalibrationProfile
from semantic_asr.phonetic_bridge import (
    FrozenPronunciationLexicon,
    PhoneticBridgeConfig,
    PronunciationLexiconEntry,
    propose_text_from_pronunciation,
)
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.score_semantics import ScoreKind


def frame(index: int, winner: str, vocabulary: tuple[str, ...]) -> PosteriorFrame:
    high = 0.9
    rest = (1.0 - high) / (len(vocabulary) - 1)
    return PosteriorFrame.from_mapping(
        start_ms=index * 20,
        end_ms=(index + 1) * 20,
        probabilities={symbol: high if symbol == winner else rest for symbol in vocabulary},
    )


def posterior(kind: str) -> PosteriorSequence:
    if kind == "phone":
        vocabulary = ("<blk>", "m", "a", "d", "t")
        winners = ("<blk>", "m", "a", "<blk>", "d", "a", "<blk>")
        label_revision = "phones-r1"
    else:
        vocabulary = ("<blk>", "マ", "ダ", "タ")
        winners = ("<blk>", "マ", "<blk>", "ダ", "<blk>")
        label_revision = "mora-r1"
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frame(index, winner, vocabulary) for index, winner in enumerate(winners)),
        encoder="test-encoder",
        encoder_revision="enc-r1",
        label_set_revision=label_revision,
        source_audio_sha256="a" * 64,
    )


def calibration(kind: str) -> UtilityCalibrationProfile:
    return UtilityCalibrationProfile(
        channel=kind,  # type: ignore[arg-type]
        score_source=(
            f"ctc-{kind}:test-encoder@enc-r1:{kind}s-r1"
            if kind == "phone"
            else "ctc-mora:test-encoder@enc-r1:mora-r1"
        ),
        score_kind=ScoreKind.LOG_LIKELIHOOD,
        center=-1.0,
        scale=0.8,
        fitted_manifest_sha256="b" * 64,
        revision="cal-r1",
    )


def lexicon() -> FrozenPronunciationLexicon:
    return FrozenPronunciationLexicon(
        name="test-ja",
        revision="lex-r1",
        entries=(
            PronunciationLexiconEntry(
                entry_id="mada",
                text="まだ",
                phone_symbols=("m", "a", "d", "a"),
                mora_symbols=("マ", "ダ"),
            ),
            PronunciationLexiconEntry(
                entry_id="mata",
                text="また",
                phone_symbols=("m", "a", "t", "a"),
                mora_symbols=("マ", "タ"),
            ),
        ),
    )


def test_audio_phone_and_mora_propose_the_supported_text() -> None:
    proposals = propose_text_from_pronunciation(
        lexicon(),
        phone_posterior=posterior("phone"),
        mora_posterior=posterior("mora"),
        phone_calibration=calibration("phone"),
        mora_calibration=calibration("mora"),
    )

    assert [proposal.text for proposal in proposals] == ["まだ", "また"]
    assert {utility.channel for utility in proposals[0].utilities} == {"phone", "mora"}
    arc = proposals[0].as_lattice_arc(span_id="ambiguous-word")
    assert arc.origin == "phonetic-proposal"
    assert arc.observed_eligible
    assert arc.independent_audio_channels == {"phone", "mora"}


def test_bridge_applies_threshold_after_held_out_normalization() -> None:
    proposals = propose_text_from_pronunciation(
        lexicon(),
        phone_posterior=posterior("phone"),
        phone_calibration=calibration("phone"),
        config=PhoneticBridgeConfig(minimum_combined_utility=0.0),
    )

    assert proposals
    assert all(proposal.combined_utility >= 0.0 for proposal in proposals)
