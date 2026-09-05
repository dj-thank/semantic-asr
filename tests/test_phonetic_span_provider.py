from __future__ import annotations

from semantic_asr.audio_posterior_adapters import (
    DualPosteriorExtractor,
    FrozenAudioPosteriorExtractor,
    FrozenPosteriorModelConfig,
    PosteriorLogits,
)
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.deliberation_evidence import UtilityCalibrationProfile
from semantic_asr.deliberation_lattice import DocumentContext
from semantic_asr.longform import LongformSegment, Window
from semantic_asr.phonetic_bridge import (
    FrozenPronunciationLexicon,
    PronunciationLexiconEntry,
)
from semantic_asr.phonetic_span_provider import (
    LoadedMonoAudio,
    PhoneticSpanProviderConfig,
    SelectivePhoneticSpanProposalProvider,
    StaticSpanLexiconProvider,
)
from semantic_asr.score_semantics import ScoreKind
from semantic_asr.semantic_deliberation import build_semantic_deliberation_lattice

AUDIO = "a" * 64
REVISION = "1" * 40


def candidate(candidate_id: str, text: str, score: float) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        text=text,
        acoustic=score,
        mora=score - 0.02,
        lexical=score - 0.04,
        preservation=score - 0.01,
        cross_model=score - 0.03,
        source="test-asr",
    )


def model_config(kind: str) -> FrozenPosteriorModelConfig:
    if kind == "phone":
        vocabulary = ("<blk>", "m", "a", "d", "t")
    else:
        vocabulary = ("<blk>", "マ", "ダ", "タ")
    return FrozenPosteriorModelConfig(
        kind=kind,  # type: ignore[arg-type]
        model_id=f"test-{kind}",
        model_revision=REVISION,
        vocabulary=vocabulary,
        blank_symbol="<blk>",
        sample_rate=1_000,
        frame_stride_ms=20.0,
    )


class FakeBackend:
    def __init__(self, config):
        self.config = config

    def infer_logits(self, samples, *, sample_rate, source_audio_sha256):
        del samples, sample_rate
        if self.config.kind == "phone":
            winners = ("<blk>", "m", "a", "<blk>", "d", "a", "<blk>")
        else:
            winners = ("<blk>", "マ", "<blk>", "ダ", "<blk>")
        rows = []
        for winner in winners:
            rows.append(
                tuple(5.0 if symbol == winner else -3.0 for symbol in self.config.vocabulary)
            )
        return PosteriorLogits(
            values=tuple(rows),
            source_audio_sha256=source_audio_sha256,
            model_config_digest=self.config.digest,
        )


class FakeLoader:
    def __init__(self, source_sha=AUDIO):
        self.source_sha = source_sha
        self.calls = 0

    def load(self, path):
        del path
        self.calls += 1
        return LoadedMonoAudio(
            samples=tuple(0.0 for _ in range(2_000)),
            sample_rate=1_000,
            source_audio_sha256=self.source_sha,
            source_name="meeting.wav",
        )


def lexicon():
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


def build():
    rows = (
        candidate("mata", "計画はまた保留です", 0.7),
        candidate("mada", "計画はまだ保留です", 0.68),
    )
    return build_semantic_deliberation_lattice(
        rows,
        posterior={"mata": 0.55, "mada": 0.45},
        pivot_candidate_id="mata",
        document_id="doc-window-0",
        source_audio_sha256=AUDIO,
        segment_start_ms=0,
        segment_end_ms=1_000,
    )


def calibration(config, channel):
    return UtilityCalibrationProfile(
        channel=channel,
        score_source=f"ctc-{channel}:{config.model_id}@{config.model_revision}:{config.digest}",
        score_kind=ScoreKind.LOG_LIKELIHOOD,
        center=-1.0,
        scale=1.0,
        fitted_manifest_sha256="b" * 64,
        revision="cal-r1",
    )


def provider(loader=None):
    phone_config = model_config("phone")
    mora_config = model_config("mora")
    extractor = DualPosteriorExtractor(
        phone=FrozenAudioPosteriorExtractor(FakeBackend(phone_config)),
        mora=FrozenAudioPosteriorExtractor(FakeBackend(mora_config)),
    )
    return SelectivePhoneticSpanProposalProvider(
        extractor=extractor,
        lexicon_provider=StaticSpanLexiconProvider(lexicon()),
        phone_calibration=calibration(phone_config, "phone"),
        mora_calibration=calibration(mora_config, "mora"),
        audio_loader=loader or FakeLoader(),
        config=PhoneticSpanProviderConfig(
            padding_ms=20,
            minimum_combined_utility=-1.0,
            proposals_per_span=2,
        ),
    )


def test_provider_extracts_only_contradiction_spans_and_binds_receipts() -> None:
    build_result = build()
    loader = FakeLoader()
    subject = provider(loader)
    segment = LongformSegment(
        window=Window(index=0, start_ms=0, end_ms=1_000),
        observed=None,  # type: ignore[arg-type]
        normalized=None,  # type: ignore[arg-type]
        diagnostics={},
    )

    proposals = subject(
        audio_path="meeting.wav",
        segment_index=0,
        segment=segment,
        build=build_result,
        context=DocumentContext(),
        source_audio_sha256=AUDIO,
    )

    assert loader.calls == 1
    assert proposals
    target_span = next(iter(proposals))
    texts = [row.text for row in proposals[target_span]]
    assert texts[0] == "まだ"
    assert target_span in subject.receipts
    receipt = subject.receipts[target_span]
    assert receipt.source_audio_sha256 == AUDIO
    assert receipt.canonical_clip_sha256 != AUDIO
    assert all(row.source_audio_sha256 == AUDIO for row in proposals[target_span])
    assert all("spanAudioReceiptDigest" in row.metadata for row in proposals[target_span])


def test_audio_is_loaded_once_across_repeated_window_calls() -> None:
    loader = FakeLoader()
    subject = provider(loader)
    segment = LongformSegment(
        window=Window(index=0, start_ms=0, end_ms=1_000),
        observed=None,  # type: ignore[arg-type]
        normalized=None,  # type: ignore[arg-type]
        diagnostics={},
    )
    for _ in range(2):
        subject(
            audio_path="meeting.wav",
            segment_index=0,
            segment=segment,
            build=build(),
            context=DocumentContext(),
            source_audio_sha256=AUDIO,
        )

    assert loader.calls == 1


def test_wrong_source_audio_fails_before_posterior_fusion() -> None:
    subject = provider(FakeLoader(source_sha="c" * 64))
    segment = LongformSegment(
        window=Window(index=0, start_ms=0, end_ms=1_000),
        observed=None,  # type: ignore[arg-type]
        normalized=None,  # type: ignore[arg-type]
        diagnostics={},
    )

    try:
        subject(
            audio_path="meeting.wav",
            segment_index=0,
            segment=segment,
            build=build(),
            context=DocumentContext(),
            source_audio_sha256=AUDIO,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("wrong source audio was accepted")
