from __future__ import annotations

from pathlib import Path

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
    PhoneticAblationManifest,
    PhoneticAblationProtocol,
)
from semantic_asr.phonetic_runtime.calibration_artifact import DualCTCUtilityArtifact
from semantic_asr.score_semantics import ScoreKind

RUNTIME_DIGEST = "c" * 64
HELD_OUT_DIGEST = "d" * 64
SPLIT_DIGEST = "e" * 64
RIGHTS_DIGEST = "f" * 64


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
                reading="マダ",
            ),
            PronunciationLexiconEntry(
                entry_id="mata",
                text="また",
                phone_symbols=("m", "a", "t", "a"),
                mora_symbols=("マ", "タ"),
                reading="マタ",
            ),
            PronunciationLexiconEntry(
                entry_id="tada",
                text="ただ",
                phone_symbols=("t", "a", "d", "a"),
                mora_symbols=("タ", "ダ"),
                reading="タダ",
            ),
        ),
    )


def utility_artifact() -> DualCTCUtilityArtifact:
    phone_source = f"ctc-phone:dual-ctc:fixture@r1@{RUNTIME_DIGEST}:phone-labels-r1"
    mora_source = f"ctc-mora:dual-ctc:fixture@r1@{RUNTIME_DIGEST}:mora-labels-r1"
    return DualCTCUtilityArtifact(
        name="fixture-utilities",
        revision="r1",
        runtime_profile_digest=RUNTIME_DIGEST,
        held_out_manifest_sha256=HELD_OUT_DIGEST,
        phone_profile=UtilityCalibrationProfile(
            channel="phone",
            score_source=phone_source,
            score_kind=ScoreKind.LOG_LIKELIHOOD,
            center=-1.0,
            scale=0.8,
            fitted_manifest_sha256=HELD_OUT_DIGEST,
            revision="phone-r1",
        ),
        mora_profile=UtilityCalibrationProfile(
            channel="mora",
            score_source=mora_source,
            score_kind=ScoreKind.LOG_LIKELIHOOD,
            center=-1.0,
            scale=0.8,
            fitted_manifest_sha256=HELD_OUT_DIGEST,
            revision="mora-r1",
        ),
        phone_pairwise_accuracy=1.0,
        mora_pairwise_accuracy=1.0,
        phone_example_digests=("1" * 64,),
        mora_example_digests=("2" * 64,),
    )


def posterior(kind: str, target: str, source_audio_sha256: str) -> PosteriorSequence:
    if kind == "phone":
        sequences = {
            "まだ": ("m", "a", "d", "a"),
            "また": ("m", "a", "t", "a"),
            "ただ": ("t", "a", "d", "a"),
        }
        vocabulary = ("<blk>", "m", "a", "d", "t")
        labels = "phone-labels-r1"
    else:
        sequences = {
            "まだ": ("マ", "ダ"),
            "また": ("マ", "タ"),
            "ただ": ("タ", "ダ"),
        }
        vocabulary = ("<blk>", "マ", "ダ", "タ")
        labels = "mora-labels-r1"
    emitted = ["<blk>"]
    for symbol in sequences[target]:
        emitted.extend((symbol, "<blk>"))
    frames = []
    for index, winner in enumerate(emitted):
        high = 0.92
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
        encoder_revision=RUNTIME_DIGEST,
        label_set_revision=labels,
        source_audio_sha256=source_audio_sha256,
    )


class FakeRuntime:
    profile_digest = RUNTIME_DIGEST
    source = "dual-ctc:fixture@r1"

    def __init__(self, targets: dict[str, str]) -> None:
        self.targets = targets
        self.calls = []

    def infer(self, audio_path, **kwargs):
        path = str(audio_path)
        target = self.targets[path]
        source = kwargs["expected_source_audio_sha256"]
        self.calls.append((path, kwargs))
        return posterior("phone", target, source), posterior("mora", target, source)


def case(
    tmp_path: Path,
    *,
    case_id: str,
    first_pass: tuple[tuple[str, float, bool], ...],
    reference: str,
    critical: bool,
) -> PhoneticAblationCase:
    audio_path = (tmp_path / f"{case_id}.wav").resolve()
    audio_path.write_bytes(b"fixture-audio-not-read-by-fake-runtime")
    source_digest = {
        "outside-recovery": "a" * 64,
        "correct-retention": "b" * 64,
        "false-correction": "3" * 64,
    }[case_id]
    return PhoneticAblationCase(
        case_id=case_id,
        audio_path=audio_path,
        source_audio_sha256=source_digest,
        start_ms=0,
        end_ms=500,
        first_pass_candidates=tuple(
            FirstPassSpanCandidate(
                candidate_id=f"{case_id}-{text}",
                text=text,
                posterior=probability,
                selected=selected,
            )
            for text, probability, selected in first_pass
        ),
        lexicon=lexicon(),
        reference=FrozenSpanReference(
            reference_id=f"ref-{case_id}",
            text=reference,
            semantic_kind="negation" if critical else "general",
            critical=critical,
        ),
        speaker_id=f"speaker-{case_id}",
        session_id=f"session-{case_id}",
        source_id=f"source-{case_id}",
        license_id="fixture-license",
        rights_decision="allow",
        dataset_revision="fixture-r1",
        split_manifest_sha256=SPLIT_DIGEST,
    )


def manifest(tmp_path: Path) -> tuple[PhoneticAblationManifest, FakeRuntime]:
    cases = (
        case(
            tmp_path,
            case_id="outside-recovery",
            first_pass=(("また", 0.6, True), ("ただ", 0.4, False)),
            reference="まだ",
            critical=True,
        ),
        case(
            tmp_path,
            case_id="correct-retention",
            first_pass=(("まだ", 0.7, True), ("また", 0.3, False)),
            reference="まだ",
            critical=False,
        ),
        case(
            tmp_path,
            case_id="false-correction",
            first_pass=(("また", 0.8, True), ("ただ", 0.2, False)),
            reference="また",
            critical=True,
        ),
    )
    targets = {
        str(cases[0].audio_path): "まだ",
        str(cases[1].audio_path): "まだ",
        str(cases[2].audio_path): "まだ",
    }
    runtime = FakeRuntime(targets)
    value = PhoneticAblationManifest(
        name="fixture",
        revision="r1",
        cases=cases,
        runtime_profile_digest=RUNTIME_DIGEST,
        utility_artifact_digest=utility_artifact().digest,
        rights_registry_sha256=RIGHTS_DIGEST,
        split_manifest_sha256=SPLIT_DIGEST,
    )
    return value, runtime


def protocol() -> PhoneticAblationProtocol:
    return PhoneticAblationProtocol(
        name="fixture",
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
        maximum_candidates=8,
        maximum_crop_ms=1_000,
        bootstrap_resamples=100,
        bootstrap_seed="fixture-seed",
    )
