from __future__ import annotations

from semantic_asr.multilevel_lattice import (
    BoundedUtility,
    CallableGlobalSequenceScorer,
    DeliberationLattice,
    DeliberationPolicy,
    DeliberationSpan,
    DocumentContext,
    LatticeArc,
    UtilityCalibrationProfile,
    decode_global_lattice,
    frozen_profile_digest,
)
from semantic_asr.phonetic_bridge import (
    FrozenPronunciationLexicon,
    PronunciationLexiconEntry,
    propose_text_from_pronunciation,
)
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.score_semantics import ScoreKind

AUDIO_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64


def frame(index: int, winner: str, vocabulary: tuple[str, ...]) -> PosteriorFrame:
    strong = 0.90
    other = (1.0 - strong) / (len(vocabulary) - 1)
    return PosteriorFrame.from_mapping(
        start_ms=index * 20,
        end_ms=(index + 1) * 20,
        probabilities={symbol: strong if symbol == winner else other for symbol in vocabulary},
    )


def posterior(
    *,
    kind: str,
    vocabulary: tuple[str, ...],
    winners: tuple[str, ...],
    label_revision: str,
) -> PosteriorSequence:
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frame(index, winner, vocabulary) for index, winner in enumerate(winners)),
        encoder="example-acoustic-encoder",
        encoder_revision="r1",
        label_set_revision=label_revision,
        source_audio_sha256=AUDIO_SHA256,
    )


phone = posterior(
    kind="phone",
    vocabulary=("<blk>", "m", "a", "d", "t"),
    winners=("<blk>", "m", "a", "<blk>", "d", "a", "<blk>"),
    label_revision="phones-r1",
)
mora = posterior(
    kind="mora",
    vocabulary=("<blk>", "マ", "ダ", "タ"),
    winners=("<blk>", "マ", "<blk>", "ダ", "<blk>"),
    label_revision="mora-r1",
)
lexicon = FrozenPronunciationLexicon(
    name="example-ja",
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
phone_calibration = UtilityCalibrationProfile(
    channel="phone",
    score_source="ctc-phone:example-acoustic-encoder@r1:phones-r1",
    score_kind=ScoreKind.LOG_LIKELIHOOD,
    center=-1.0,
    scale=0.8,
    fitted_manifest_sha256=MANIFEST_SHA256,
    revision="example-cal-r1",
)
mora_calibration = UtilityCalibrationProfile(
    channel="mora",
    score_source="ctc-mora:example-acoustic-encoder@r1:mora-r1",
    score_kind=ScoreKind.LOG_LIKELIHOOD,
    center=-1.0,
    scale=0.8,
    fitted_manifest_sha256=MANIFEST_SHA256,
    revision="example-cal-r1",
)
proposals = propose_text_from_pronunciation(
    lexicon,
    phone_posterior=phone,
    mora_posterior=mora,
    phone_calibration=phone_calibration,
    mora_calibration=mora_calibration,
)

asr_profile = frozen_profile_digest("example-asr-utility", "r1", {"split": "held-out"})
retained = LatticeArc(
    arc_id="first-pass-mata",
    span_id="ambiguous-word",
    text="また",
    origin="first-pass",
    utilities=(
        BoundedUtility(
            channel="asr_acoustic",
            value=0.55,
            source="example-asr",
            profile_digest=asr_profile,
            input_digest="c" * 64,
        ),
    ),
    pronunciation_key=lexicon.entries[1].pronunciation_key,
)
phonetic_mada = next(proposal for proposal in proposals if proposal.text == "まだ")
span = DeliberationSpan(
    span_id="ambiguous-word",
    index=0,
    start_ms=0,
    end_ms=140,
    arcs=(retained, phonetic_mada.as_lattice_arc(span_id="ambiguous-word")),
    retained_arc_id=retained.arc_id,
)
lattice = DeliberationLattice(
    document_id="example",
    source_audio_sha256=AUDIO_SHA256,
    spans=(span,),
)
context = DocumentContext(
    left_context="レビューが終わるまでは保留です。",
    right_context="マージしないでください。",
)
scorer = CallableGlobalSequenceScorer(
    lambda path, document: (
        1.0 if path[0].text == "まだ" and "保留" in document.left_context else -1.0
    ),
    source="example-full-document-scorer",
    profile_digest=frozen_profile_digest(
        "example-global-context",
        "r1",
        {"scope": "complete-path+left+right"},
    ),
)
policy = DeliberationPolicy(
    channel_weights=(
        ("asr_acoustic", 1.0),
        ("phone", 0.8),
        ("mora", 0.8),
    ),
    global_context_weight=0.5,
    maximum_span_audio_regression=0.2,
    maximum_mean_audio_regression=0.2,
)
decision = decode_global_lattice(
    lattice,
    policy=policy,
    context=context,
    sequence_scorer=scorer,
)

print("phonetic proposals:", [(row.text, round(row.combined_utility, 3)) for row in proposals])
print("selected:", decision.observed_text)
print("status:", decision.status)
print("resolution:", decision.resolutions[0].mode)
