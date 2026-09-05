from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_asr.deliberation_lattice import DocumentContext
from semantic_asr.phonetic_bridge import (
    FrozenPronunciationLexicon,
    PronunciationLexiconEntry,
)
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.phonetic_runtime.calibration import (
    PhoneticCalibrationCandidate,
    PhoneticCalibrationExample,
    fit_ctc_utility_calibration,
)
from semantic_asr.phonetic_runtime.calibration_artifact import DualCTCUtilityArtifact
from semantic_asr.phonetic_runtime.provider import (
    PhoneticProposalProviderConfig,
    SourceAudioPhoneticProposalProvider,
)
from semantic_asr.semantic_deliberation import build_semantic_deliberation_lattice

from _document_experiment_fixture import AUDIO, first_pass


class CountingRuntime:
    source = "dual-ctc:fixture@r1"

    def __init__(self, profile_digest: str = "c" * 64) -> None:
        self.profile_digest = profile_digest
        self.calls = []

    def infer(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return posterior("phone"), posterior("mora")


def posterior(kind: str) -> PosteriorSequence:
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
        encoder="dual-ctc:fixture@r1",
        encoder_revision="c" * 64,
        label_set_revision=f"{kind}-labels-r1",
        source_audio_sha256=AUDIO,
    )


def calibration_report(kind: str):
    correct_symbols = ("m", "a", "d", "a") if kind == "phone" else ("マ", "ダ")
    wrong_symbols = ("m", "a", "t", "a") if kind == "phone" else ("マ", "タ")
    example = PhoneticCalibrationExample(
        example_id=f"{kind}-example",
        posterior=posterior(kind),
        candidates=(
            PhoneticCalibrationCandidate(
                candidate_id="correct",
                text="まだ",
                symbols=correct_symbols,
                correct=True,
            ),
            PhoneticCalibrationCandidate(
                candidate_id="wrong",
                text="また",
                symbols=wrong_symbols,
                correct=False,
            ),
        ),
    )
    return fit_ctc_utility_calibration(
        (example,),
        held_out_manifest_sha256="e" * 64,
        revision=f"{kind}-cal-r1",
    )


def utility_artifact() -> DualCTCUtilityArtifact:
    return DualCTCUtilityArtifact.from_reports(
        calibration_report("phone"),
        calibration_report("mora"),
        name="fixture-utilities",
        revision="r1",
        runtime_profile_digest="c" * 64,
    )


def lexicon() -> FrozenPronunciationLexicon:
    return FrozenPronunciationLexicon(
        name="fixture",
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
    result = build_semantic_deliberation_lattice(
        segment.observed.candidates,
        posterior=dict(segment.observed.ranked[0].gate.posterior),
        pivot_candidate_id=segment.observed.selected_candidate_id,
        document_id="fixture-window",
        source_audio_sha256=AUDIO,
        segment_start_ms=segment.window.start_ms,
        segment_end_ms=segment.window.end_ms,
    )
    return segment, result


def test_provider_factory_requires_exact_runtime_utility_binding() -> None:
    artifact = utility_artifact()

    with pytest.raises(ValueError, match="different runtime profile"):
        SourceAudioPhoneticProposalProvider.from_utility_artifact(
            runtime=CountingRuntime(profile_digest="d" * 64),
            lexicon_provider=lambda span, context: lexicon(),
            utility_artifact=artifact,
        )

    provider = SourceAudioPhoneticProposalProvider.from_utility_artifact(
        runtime=CountingRuntime(),
        lexicon_provider=lambda span, context: lexicon(),
        utility_artifact=artifact,
    )
    assert provider.utility_artifact_digest == artifact.digest


def test_provider_stops_before_total_crop_budget_is_exceeded() -> None:
    segment, lattice_build = build()
    runtime = CountingRuntime()
    provider = SourceAudioPhoneticProposalProvider.from_utility_artifact(
        runtime=runtime,
        lexicon_provider=lambda span, context: lexicon(),
        utility_artifact=utility_artifact(),
        config=PhoneticProposalProviderConfig(
            maximum_spans=8,
            maximum_total_audio_ms=1,
            padding_ms=0,
        ),
    )

    proposals = provider(
        audio_path="meeting.wav",
        segment_index=0,
        segment=segment,
        build=lattice_build,
        context=DocumentContext(),
        source_audio_sha256=AUDIO,
    )

    assert proposals == {}
    assert runtime.calls == []


def test_provider_rejects_oversized_lexicon_before_audio_inference() -> None:
    segment, lattice_build = build()
    runtime = CountingRuntime()
    oversized = replace(
        lexicon(),
        entries=tuple(
            PronunciationLexiconEntry(
                entry_id=f"entry-{index}",
                text=f"候補{index}",
                phone_symbols=("m", "a"),
                mora_symbols=("マ",),
            )
            for index in range(3)
        ),
    )
    provider = SourceAudioPhoneticProposalProvider.from_utility_artifact(
        runtime=runtime,
        lexicon_provider=lambda span, context: oversized,
        utility_artifact=utility_artifact(),
        config=PhoneticProposalProviderConfig(maximum_lexicon_entries=2),
    )

    with pytest.raises(ValueError, match="maximum_lexicon_entries"):
        provider(
            audio_path="meeting.wav",
            segment_index=0,
            segment=segment,
            build=lattice_build,
            context=DocumentContext(),
            source_audio_sha256=AUDIO,
        )
    assert runtime.calls == []
