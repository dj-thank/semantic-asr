from __future__ import annotations

from pathlib import Path

from semantic_asr.context_phonetic_experiment.context_scorer import (
    CallableCandidateContextScorer,
    ContextCandidate,
)
from semantic_asr.context_phonetic_experiment.protocol import (
    ContextPhoneticArm,
    ContextPhoneticCase,
    ContextPhoneticManifest,
    ContextPhoneticProtocol,
    FrozenContextSnapshot,
)
from semantic_asr.contracts import sha256_json
from semantic_asr.deliberation_evidence import UtilityCalibrationProfile
from semantic_asr.phonetic_bridge import (
    FrozenPronunciationLexicon,
    PronunciationLexiconEntry,
)
from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence
from semantic_asr.phonetic_experiment.protocol import (
    FirstPassSpanCandidate,
    FrozenSpanReference,
    PhoneticAblationArm,
    PhoneticAblationCase,
    PhoneticAblationProtocol,
)
from semantic_asr.phonetic_runtime.calibration_artifact import DualCTCUtilityArtifact
from semantic_asr.score_semantics import ScoreKind

RUNTIME_DIGEST = "c" * 64
HELD_OUT_DIGEST = "d" * 64
SPLIT_DIGEST = "e" * 64
RIGHTS_DIGEST = "f" * 64
CONTEXT_SOURCE_DIGEST = "1" * 64

SURFACES = {
    "mada": ("まだ", ("m", "a", "d", "a"), ("マ", "ダ")),
    "mata": ("また", ("m", "a", "t", "a"), ("マ", "タ")),
    "tada": ("ただ", ("t", "a", "d", "a"), ("タ", "ダ")),
    "tama": ("たま", ("t", "a", "m", "a"), ("タ", "マ")),
}


def lexicon() -> FrozenPronunciationLexicon:
    return FrozenPronunciationLexicon(
        name="factorial-fixture",
        revision="r1",
        entries=tuple(
            PronunciationLexiconEntry(
                entry_id=entry_id,
                text=text,
                phone_symbols=phones,
                mora_symbols=moras,
                reading="".join(moras),
                tags=("fixture",),
            )
            for entry_id, (text, phones, moras) in SURFACES.items()
        ),
    )


def _score_source(kind: str) -> str:
    return (
        f"ctc-{kind}:dual-ctc:factorial@r1@{RUNTIME_DIGEST}:"
        f"{kind}-labels-r1"
    )


def utility_artifact() -> DualCTCUtilityArtifact:
    return DualCTCUtilityArtifact(
        name="factorial-utilities",
        revision="r1",
        runtime_profile_digest=RUNTIME_DIGEST,
        held_out_manifest_sha256=HELD_OUT_DIGEST,
        phone_profile=UtilityCalibrationProfile(
            channel="phone",
            score_source=_score_source("phone"),
            score_kind=ScoreKind.LOG_LIKELIHOOD,
            center=-1.0,
            scale=0.8,
            fitted_manifest_sha256=HELD_OUT_DIGEST,
            revision="phone-r1",
        ),
        mora_profile=UtilityCalibrationProfile(
            channel="mora",
            score_source=_score_source("mora"),
            score_kind=ScoreKind.LOG_LIKELIHOOD,
            center=-1.0,
            scale=0.8,
            fitted_manifest_sha256=HELD_OUT_DIGEST,
            revision="mora-r1",
        ),
        phone_pairwise_accuracy=1.0,
        mora_pairwise_accuracy=1.0,
        phone_example_digests=("2" * 64,),
        mora_example_digests=("3" * 64,),
    )


def _posterior(kind: str, target_id: str, source_audio_sha256: str) -> PosteriorSequence:
    if kind == "phone":
        vocabulary = ("<blk>", "m", "a", "d", "t")
        symbols = SURFACES[target_id][1]
    else:
        vocabulary = ("<blk>", "マ", "ダ", "タ")
        symbols = SURFACES[target_id][2]
    emitted = ["<blk>"]
    for symbol in symbols:
        emitted.extend((symbol, "<blk>"))
    frames = []
    for index, winner in enumerate(emitted):
        high = 0.94
        rest = (1.0 - high) / (len(vocabulary) - 1)
        frames.append(
            PosteriorFrame.from_mapping(
                start_ms=index * 20,
                end_ms=(index + 1) * 20,
                probabilities={
                    symbol: high if symbol == winner else rest
                    for symbol in vocabulary
                },
            )
        )
    return PosteriorSequence(
        kind=kind,  # type: ignore[arg-type]
        blank_symbol="<blk>",
        vocabulary=vocabulary,
        frames=tuple(frames),
        encoder="dual-ctc:factorial@r1",
        encoder_revision=RUNTIME_DIGEST,
        label_set_revision=f"{kind}-labels-r1",
        source_audio_sha256=source_audio_sha256,
    )


class FactorialFakeRuntime:
    profile_digest = RUNTIME_DIGEST
    source = "dual-ctc:factorial@r1"

    def __init__(self, targets: dict[str, str]) -> None:
        self.targets = targets
        self.calls = []

    def infer(self, audio_path, **kwargs):
        path = str(audio_path)
        target = self.targets[path]
        source = kwargs["expected_source_audio_sha256"]
        self.calls.append((path, kwargs))
        return _posterior("phone", target, source), _posterior("mora", target, source)


class CountingContextScorer(CallableCandidateContextScorer):
    def __init__(self) -> None:
        self.calls = []
        super().__init__(
            self._score,
            source="factorial-context-fixture",
            profile_digest="4" * 64,
        )

    def _score(self, candidate: ContextCandidate, context: FrozenContextSnapshot) -> float:
        self.calls.append((candidate.candidate_id, context.context_id))
        preference = context.topic_summary.removeprefix("prefer:")
        return 1.0 if candidate.text == preference else -0.5


def _phonetic_case(
    tmp_path: Path,
    *,
    case_id: str,
    first_pass: tuple[tuple[str, float, bool], ...],
    reference: str,
    critical: bool,
) -> PhoneticAblationCase:
    audio = (tmp_path / f"{case_id}.wav").resolve()
    audio.write_bytes(f"fixture:{case_id}".encode("utf-8"))
    source_digest = sha256_json({"caseId": case_id, "audio": "fixture"})
    return PhoneticAblationCase(
        case_id=case_id,
        audio_path=audio,
        source_audio_sha256=source_digest,
        start_ms=0,
        end_ms=500,
        first_pass_candidates=tuple(
            FirstPassSpanCandidate(
                candidate_id=f"{case_id}:{text}",
                text=text,
                posterior=posterior,
                selected=selected,
            )
            for text, posterior, selected in first_pass
        ),
        lexicon=lexicon(),
        reference=FrozenSpanReference(
            reference_id=f"reference:{case_id}",
            text=reference,
            semantic_kind="critical" if critical else "general",
            critical=critical,
        ),
        speaker_id=f"speaker:{case_id}",
        session_id=f"session:{case_id}",
        source_id=f"source:{case_id}",
        license_id="fixture-license",
        rights_decision="allow",
        dataset_revision="fixture-r1",
        split_manifest_sha256=SPLIT_DIGEST,
    )


def factorial_manifest(tmp_path: Path):
    definitions = (
        (
            "outside-recovery",
            (("また", 0.6, True), ("ただ", 0.4, False)),
            "まだ",
            "まだ",
            "mada",
            True,
        ),
        (
            "correct-retention",
            (("また", 0.75, True), ("まだ", 0.25, False)),
            "また",
            "また",
            "mata",
            False,
        ),
        (
            "context-rescue",
            (("ただ", 0.7, True), ("たま", 0.3, False)),
            "ただ",
            "ただ",
            "tama",
            True,
        ),
        (
            "second-recovery",
            (("まだ", 0.55, True), ("また", 0.45, False)),
            "たま",
            "たま",
            "tama",
            False,
        ),
    )
    cases = []
    targets = {}
    for case_id, first_pass, reference, context_preference, acoustic_target, critical in definitions:
        phonetic = _phonetic_case(
            tmp_path,
            case_id=case_id,
            first_pass=first_pass,
            reference=reference,
            critical=critical,
        )
        context = FrozenContextSnapshot(
            context_id=f"ordered:{case_id}",
            left_context=f"left context for {case_id}",
            right_context=f"right context for {case_id}",
            topic_summary=f"prefer:{context_preference}",
            entity_ids=(f"entity:{case_id}",),
            source_case_id=case_id,
        )
        cases.append(
            ContextPhoneticCase(
                phonetic_case=phonetic,
                ordered_context=context,
                context_group_id=f"preference:{context_preference}",
            )
        )
        targets[str(phonetic.audio_path)] = acoustic_target
    phonetic_digest = sha256_json(tuple(case.phonetic_case.digest for case in cases))
    context_digest = sha256_json(tuple(case.ordered_context.digest for case in cases))
    return (
        ContextPhoneticManifest(
            name="factorial-fixture",
            revision="r1",
            cases=tuple(cases),
            phonetic_manifest_digest=phonetic_digest,
            context_source_digest=context_digest,
            rights_registry_sha256=RIGHTS_DIGEST,
        ),
        FactorialFakeRuntime(targets),
        CountingContextScorer(),
    )


def phonetic_protocol() -> PhoneticAblationProtocol:
    return PhoneticAblationProtocol(
        name="factorial-phonetic",
        revision="r1",
        arms=(
            PhoneticAblationArm(
                name="first-pass",
                channel_weights=(("first_pass", 1.0),),
                allow_outside_first_pass=False,
            ),
            PhoneticAblationArm(
                name="phone",
                channel_weights=(("phone", 1.0),),
            ),
            PhoneticAblationArm(
                name="mora",
                channel_weights=(("mora", 1.0),),
            ),
            PhoneticAblationArm(
                name="phone+mora",
                channel_weights=(("phone", 1.0), ("mora", 1.0)),
            ),
        ),
        baseline_arm="first-pass",
        maximum_candidates=16,
        maximum_crop_ms=1_000,
        bootstrap_resamples=100,
        bootstrap_seed="phonetic-fixture",
    )


def factorial_protocol() -> ContextPhoneticProtocol:
    phonetic = phonetic_protocol()
    arms = tuple(
        ContextPhoneticArm(
            name=f"{phonetic_name}:{condition}",
            phonetic_arm_name=phonetic_name,
            context_condition=condition,
            context_weight=0.0 if condition == "none" else 2.5,
        )
        for phonetic_name in ("first-pass", "phone", "mora", "phone+mora")
        for condition in ("none", "ordered", "shuffled")
    )
    return ContextPhoneticProtocol(
        name="factorial-fixture",
        revision="r1",
        phonetic_protocol=phonetic,
        arms=arms,
        baseline_arm="first-pass:none",
        target_arm="phone+mora:ordered",
        shuffled_control_arm="phone+mora:shuffled",
        shuffle_seed="factorial-shuffle-r1",
        bootstrap_group="speaker",
        bootstrap_resamples=100,
    )
