from __future__ import annotations

import pytest

from semantic_asr.deliberation_evidence import UtilityCalibrationProfile
from semantic_asr.phonetic_bridge import (
    FrozenPronunciationLexicon,
    PronunciationLexiconEntry,
)
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.phonetic_runtime.provider import (
    PhoneticProposalProviderConfig,
    SourceAudioPhoneticProposalProvider,
)
from semantic_asr.score_semantics import ScoreKind
from semantic_asr.semantic_deliberation import build_semantic_deliberation_lattice

from _document_experiment_fixture import AUDIO, first_pass


def posterior(kind: str, *, audio_sha256: str = AUDIO) -> PosteriorSequence:
    if kind == "phone":
        vocabulary = ("<blk>", "m", "a", "d", "t")
        winners = ("<blk>", "m", "a", "<blk>", "d", "a", "<blk>")
    else:
        vocabulary = ("<blk>", "マ", "ダ", "タ")
        winners = ("<blk>", "マ", "<blk>", "ダ", "<blk>")
    frames = []
    for index, winner in enumerate(winners):
        high = 0.9
        rest = (1.0 - high) / (len(vocabulary) - 1)
        frames.append(
            PosteriorFrame.from_mapping(
                start_ms=index * 20,
                end_ms=(index + 1) * 20,
                probabilities={
                    symbol: high if symbol == winner else rest for symbol in vocabulary
                },
            )
        )
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frames),
        encoder="fake-dual-ctc",
        encoder_revision="runtime-r1",
        label_set_revision=f"{kind}-labels-r1",
        source_audio_sha256=audio_sha256,
    )


class FakeRuntime:
    profile_digest = "d" * 64
    source = "fake-dual-ctc"

    def __init__(self, *, audio_sha256: str = AUDIO) -> None:
        self.audio_sha256 = audio_sha256
        self.calls = []

    def infer(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return (
            posterior("phone", audio_sha256=self.audio_sha256),
            posterior("mora", audio_sha256=self.audio_sha256),
        )


def calibration(kind: str) -> UtilityCalibrationProfile:
    return UtilityCalibrationProfile(
        channel=kind,  # type: ignore[arg-type]
        score_source=f"ctc-{kind}:fake-dual-ctc@runtime-r1:{kind}-labels-r1",
        score_kind=ScoreKind.LOG_LIKELIHOOD,
        center=-1.0,
        scale=0.8,
        fitted_manifest_sha256="e" * 64,
        revision=f"{kind}-cal-r1",
    )


def lexicon() -> FrozenPronunciationLexicon:
    return FrozenPronunciationLexicon(
        name="fixture-ja",
        revision="r1",
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
    segment = first_pass().segments[0]
    return segment, build_semantic_deliberation_lattice(
        segment.observed.candidates,
        posterior=dict(segment.observed.ranked[0].gate.posterior),
        pivot_candidate_id=segment.observed.selected_candidate_id,
        document_id="fixture-window",
        source_audio_sha256=AUDIO,
        segment_start_ms=segment.window.start_ms,
        segment_end_ms=segment.window.end_ms,
    )


def test_provider_calls_runtime_only_for_active_contradiction_spans() -> None:
    segment, lattice_build = build()
    runtime = FakeRuntime()
    provider = SourceAudioPhoneticProposalProvider(
        runtime=runtime,
        lexicon_provider=lambda span, context: lexicon(),
        phone_calibration=calibration("phone"),
        mora_calibration=calibration("mora"),
        config=PhoneticProposalProviderConfig(maximum_spans=2),
    )

    proposals = provider(
        audio_path="meeting.wav",
        segment_index=0,
        segment=segment,
        build=lattice_build,
        context=None,  # type: ignore[arg-type]
        source_audio_sha256=AUDIO,
    )

    active = [
        span
        for span in lattice_build.lattice.spans
        if bool(span.metadata.get("isContradiction"))
    ]
    assert len(runtime.calls) == min(2, len(active))
    assert proposals
    for rows in proposals.values():
        for row in rows:
            assert row.source_audio_sha256 == AUDIO
            assert {utility.channel for utility in row.utilities} == {"phone", "mora"}


def test_provider_rejects_posteriors_from_another_audio() -> None:
    segment, lattice_build = build()
    provider = SourceAudioPhoneticProposalProvider(
        runtime=FakeRuntime(audio_sha256="b" * 64),
        lexicon_provider=lambda span, context: lexicon(),
        phone_calibration=calibration("phone"),
        mora_calibration=calibration("mora"),
    )

    with pytest.raises(ValueError, match="different source audio"):
        provider(
            audio_path="meeting.wav",
            segment_index=0,
            segment=segment,
            build=lattice_build,
            context=None,  # type: ignore[arg-type]
            source_audio_sha256=AUDIO,
        )
