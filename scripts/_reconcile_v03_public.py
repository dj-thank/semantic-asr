#!/usr/bin/env python3
"""One-shot reconciler for integrated v0.3 public API, fixtures, and permanent CI."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def replace_ranker_dataset_test() -> None:
    content = r'''
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from semantic_asr.deliberation_lattice import DocumentContext, LatticeArc
from semantic_asr.document_joint_engine import (
    DocumentDeliberationConfig,
    DocumentDeliberationDecision,
    DocumentPathCandidate,
    OverlapPolicy,
    OverlapReceipt,
    WindowPathOption,
)
from semantic_asr.document_ranker_dataset import (
    label_document_decision,
    write_labeled_groups,
)
from semantic_asr.global_deliberation import DeliberationPolicy, PathHypothesis
from semantic_asr.longform import Window

AUDIO = "a" * 64


@dataclass(frozen=True)
class FakeSpan:
    span_id: str


@dataclass(frozen=True)
class FakeLattice:
    source_audio_sha256: str
    spans: tuple[FakeSpan, ...]


@dataclass(frozen=True)
class FakeBuild:
    lattice: FakeLattice
    digest: str


def option(text: str, *, retained: bool) -> WindowPathOption:
    arc = LatticeArc(
        arc_id=f"arc-{text}",
        span_id="span-0",
        text=text,
        origin="first-pass",
        utilities=(),
        source_audio_sha256=AUDIO,
    )
    path = PathHypothesis(
        arcs=(arc,),
        base_score=0.5,
        mean_audio_support=0.7,
        final_score=0.5,
    )
    build = FakeBuild(
        lattice=FakeLattice(AUDIO, (FakeSpan("span-0"),)),
        digest="b" * 64,
    )
    return WindowPathOption(
        segment_index=0,
        window=Window(index=0, start_ms=0, end_ms=1_000),
        build=build,  # type: ignore[arg-type]
        path=path,
        retained_path_digest=path.digest if retained else "c" * 64,
        option_rank=0,
    )


def receipt(text: str) -> OverlapReceipt:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return OverlapReceipt(
        left_window_index=None,
        right_window_index=0,
        overlap_ms=0,
        method="first-window",
        right_trim_characters=0,
        matched_characters=0,
        normalized_matched_characters=0,
        similarity=0.0,
        utility=0.0,
        left_text_sha256=None,
        right_text_sha256=digest,
        emitted_text_sha256=digest,
        policy_digest=OverlapPolicy().digest,
    )


def candidate(text: str, *, retained: bool) -> DocumentPathCandidate:
    row = option(text, retained=retained)
    return DocumentPathCandidate(
        options=(row,),
        emitted_texts=(text,),
        overlap_receipts=(receipt(text),),
        local_score=0.5,
        overlap_score=0.0,
        mean_audio_support=0.7,
        final_score=0.5,
    )


def decision() -> tuple[DocumentDeliberationDecision, DocumentContext]:
    retained = candidate("三千円です。", retained=True)
    alternative = candidate("三万円です。", retained=False)
    context = DocumentContext(topic_summary="費用確認")
    return (
        DocumentDeliberationDecision(
            selected=retained,
            retained=retained,
            alternatives=(retained, alternative),
            status="accepted",
            applied=False,
            margin=0.1,
            reasons=("retained-first-pass-document",),
            first_pass_evidence_sha256="d" * 64,
            config_digest=DocumentDeliberationConfig().digest,
            local_policy_digest=DeliberationPolicy.conservative_default().digest,
            context_digest=context.digest,
            source_audio_sha256=AUDIO,
        ),
        context,
    )


def test_offline_labeler_uses_runtime_shape_and_reference_only_for_labels() -> None:
    value, context = decision()
    group = label_document_decision(
        value,
        group_id="recording-1",
        reference="三千円です。",
        first_pass_text="三千円です。",
        context=context,
        critical_tokens=("三千円",),
    )

    assert len(group.examples) == 2
    retained = next(row for row in group.examples if row.rank_input.retained_path)
    changed = next(row for row in group.examples if not row.rank_input.retained_path)
    assert retained.character_error_rate == 0.0
    assert retained.critical_error_count == 0
    assert changed.character_error_rate > 0.0
    assert changed.critical_error_count == 1
    assert "三千円です。" not in json.dumps(
        retained.rank_input.metadata, ensure_ascii=False
    )


def test_labeler_rejects_context_mismatch() -> None:
    value, _ = decision()
    with pytest.raises(ValueError, match="context digests differ"):
        label_document_decision(
            value,
            group_id="recording-1",
            reference="三千円です。",
            first_pass_text="三千円です。",
            context=DocumentContext(topic_summary="別の文脈"),
        )


def test_labeled_group_writes_reference_free_runtime_rows(tmp_path: Path) -> None:
    value, context = decision()
    group = label_document_decision(
        value,
        group_id="recording-1",
        reference="三千円です。",
        first_pass_text="三千円です。",
        context=context,
        critical_tokens=("三千円",),
    )
    output = write_labeled_groups((group,), tmp_path / "train.jsonl")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 2
    assert manifest["referenceBoundary"] == "references-used-offline-only"
    assert all("reference" not in row for row in rows)
'''
    Path("tests/test_document_ranker_dataset.py").write_text(
        textwrap.dedent(content).lstrip(), encoding="utf-8"
    )


def reconcile_ranker_tamper_test() -> None:
    path = Path("tests/test_document_ranker.py")
    text = path.read_text(encoding="utf-8")
    old = '''    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"bias": 0.0', '"bias": 1.0'), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        DocumentRankerArtifact.load(path)
'''
    if old in text:
        new = '''    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["bias"] = float(payload["model"]["bias"]) + 1.0
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        DocumentRankerArtifact.load(path)
'''
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def reconcile_root_api() -> None:
    path = Path("src/semantic_asr/__init__.py")
    text = path.read_text(encoding="utf-8")
    modules = (
        "audio_posterior_adapters",
        "document_deliberation_benchmark",
        "document_joint_engine",
        "document_ranker",
        "document_ranker_dataset",
        "japanese_phonetic_targets",
        "joint_phonetic_runtime_optional",
        "phonetic_dataset",
        "phonetic_feature_export",
        "phonetic_span_provider",
        "phonetic_trainer_optional",
        "phonetic_training",
    )
    for module in modules:
        text = re.sub(rf"\nfrom \.{module} import \(.*?\n\)\n", "\n", text, flags=re.S)
    imports = r'''
from .audio_posterior_adapters import (
    AudioPosteriorBackend,
    DualPosteriorExtractor,
    FrozenAudioPosteriorExtractor,
    FrozenPosteriorModelConfig,
    PosteriorBundle,
    PosteriorLogits,
    PosteriorResourcePolicy,
    TransformersCTCBackend,
    canonical_audio_sha256,
    posterior_sequence_from_logits,
    read_mono_wav,
)
from .document_deliberation_benchmark import (
    BootstrapInterval,
    DocumentBenchmarkReport,
    DocumentEvaluationCase,
    DocumentPromotionGate,
    PromotionDecision,
    apply_document_promotion_gate,
    character_error_rate,
    evaluate_document_deliberation,
    paired_bootstrap_cer_delta,
)
from .document_joint_engine import (
    DocumentArcReceipt,
    DocumentDeliberatedResult,
    DocumentDeliberatedSegment,
    DocumentDeliberationConfig,
    DocumentDeliberationDecision,
    DocumentFailureReceipt,
    DocumentNormalizedTranscript,
    DocumentObservedTranscript,
    DocumentPassThroughResult,
    DocumentPathCandidate,
    DocumentProposalProvider,
    JointDocumentSemanticASRTranscriber,
    OverlapPolicy,
    OverlapReceipt,
    WindowPathOption,
    apply_joint_document_deliberation,
    resolve_window_overlap,
    with_joint_document_deliberation,
)
from .document_ranker import (
    DocumentFeatureConfig,
    DocumentLinearRanker,
    DocumentRankExample,
    DocumentRankInput,
    DocumentRankTrainingConfig,
    DocumentRankerArtifact,
    DocumentRankerCalibration,
    DocumentRankerGlobalScorer,
    HashedDocumentFeatureExtractor,
    fit_document_ranker_calibration,
    group_top1_accuracy,
    pairwise_accuracy,
    train_document_ranker,
)
from .document_ranker_dataset import (
    DocumentRankerLabeledGroup,
    label_document_decision,
    rank_input_from_candidate,
    write_labeled_groups,
)
from .japanese_phonetic_targets import (
    JapanesePronunciationPolicy,
    JapanesePronunciationTarget,
    japanese_pronunciation_target,
)
from .joint_phonetic_runtime_optional import (
    FrozenAudioFeatureConfig,
    FrozenFeatureMatrix,
    JointPhoneticPosteriorExtractor,
    TransformersAudioFeatureBackend,
)
from .phonetic_dataset import (
    PhoneticDatasetResourcePolicy,
    PhoneticFeatureItem,
    PhoneticFeatureManifest,
    file_sha256,
    load_feature_array,
    load_phonetic_feature_manifest,
    minimum_ctc_frames,
    validate_phonetic_split_disjointness,
)
from .phonetic_feature_export import (
    LoadedSourceRecording,
    PhoneticFeatureExportConfig,
    PhoneticFeatureExportResult,
    PhoneticFeatureExporter,
    PhoneticFeatureReceipt,
    PhoneticSourceItem,
    PhoneticSourceManifest,
    PhoneticSourceResourcePolicy,
    SoundFileSourceAudioLoader,
    load_phonetic_source_manifest,
)
from .phonetic_heads_optional import JointPhoneMoraCTCHead
from .phonetic_span_provider import (
    LoadedMonoAudio,
    MonoAudioLoader,
    PhoneticSpanProviderConfig,
    SelectivePhoneticSpanProposalProvider,
    SoundFileMonoAudioLoader,
    SpanAudioReceipt,
    SpanLexiconProvider,
    SpanProposalFailure,
    StaticSpanLexiconProvider,
)
from .phonetic_trainer_optional import (
    PhoneticEvaluationResult,
    PhoneticOptimizationConfig,
    PhoneticSequenceCalibration,
    PhoneticTrainingHistory,
    build_joint_phonetic_artifact,
    evaluate_joint_phonetic_head,
    save_joint_phonetic_weights,
    train_joint_phonetic_head,
)
from .phonetic_training import (
    JointPhoneticArtifact,
    JointPhoneticHeadConfig,
    PhoneticLabelInventory,
    PhoneticTrainingManifest,
    PhoneticValidationMetrics,
    posterior_configs_from_artifact,
)
'''
    anchor = "from .fusion import FusionConfig, fuse_candidates\n"
    if anchor not in text:
        raise RuntimeError("root API import anchor missing")
    text = text.replace(anchor, textwrap.dedent(imports).lstrip() + anchor, 1)
    start = "# semantic-asr-v03-integrated-public-api-start"
    end = "# semantic-asr-v03-integrated-public-api-end"
    text = re.sub(rf"\n{re.escape(start)}.*?{re.escape(end)}\n", "\n", text, flags=re.S)
    names = (
        "AudioPosteriorBackend",
        "BootstrapInterval",
        "DocumentArcReceipt",
        "DocumentBenchmarkReport",
        "DocumentDeliberatedResult",
        "DocumentDeliberatedSegment",
        "DocumentDeliberationConfig",
        "DocumentDeliberationDecision",
        "DocumentEvaluationCase",
        "DocumentFailureReceipt",
        "DocumentFeatureConfig",
        "DocumentLinearRanker",
        "DocumentNormalizedTranscript",
        "DocumentObservedTranscript",
        "DocumentPassThroughResult",
        "DocumentPathCandidate",
        "DocumentPromotionGate",
        "DocumentProposalProvider",
        "DocumentRankExample",
        "DocumentRankInput",
        "DocumentRankTrainingConfig",
        "DocumentRankerArtifact",
        "DocumentRankerCalibration",
        "DocumentRankerGlobalScorer",
        "DocumentRankerLabeledGroup",
        "DualPosteriorExtractor",
        "FrozenAudioFeatureConfig",
        "FrozenAudioPosteriorExtractor",
        "FrozenFeatureMatrix",
        "FrozenPosteriorModelConfig",
        "HashedDocumentFeatureExtractor",
        "JapanesePronunciationPolicy",
        "JapanesePronunciationTarget",
        "JointDocumentSemanticASRTranscriber",
        "JointPhoneMoraCTCHead",
        "JointPhoneticArtifact",
        "JointPhoneticHeadConfig",
        "JointPhoneticPosteriorExtractor",
        "LoadedMonoAudio",
        "LoadedSourceRecording",
        "MonoAudioLoader",
        "OverlapPolicy",
        "OverlapReceipt",
        "PhoneticDatasetResourcePolicy",
        "PhoneticEvaluationResult",
        "PhoneticFeatureExportConfig",
        "PhoneticFeatureExportResult",
        "PhoneticFeatureExporter",
        "PhoneticFeatureItem",
        "PhoneticFeatureManifest",
        "PhoneticFeatureReceipt",
        "PhoneticLabelInventory",
        "PhoneticOptimizationConfig",
        "PhoneticSequenceCalibration",
        "PhoneticSourceItem",
        "PhoneticSourceManifest",
        "PhoneticSourceResourcePolicy",
        "PhoneticSpanProviderConfig",
        "PhoneticTrainingHistory",
        "PhoneticTrainingManifest",
        "PhoneticValidationMetrics",
        "PosteriorBundle",
        "PosteriorLogits",
        "PosteriorResourcePolicy",
        "PromotionDecision",
        "SelectivePhoneticSpanProposalProvider",
        "SoundFileMonoAudioLoader",
        "SoundFileSourceAudioLoader",
        "SpanAudioReceipt",
        "SpanLexiconProvider",
        "SpanProposalFailure",
        "StaticSpanLexiconProvider",
        "TransformersAudioFeatureBackend",
        "TransformersCTCBackend",
        "WindowPathOption",
        "apply_document_promotion_gate",
        "apply_joint_document_deliberation",
        "build_joint_phonetic_artifact",
        "canonical_audio_sha256",
        "character_error_rate",
        "evaluate_document_deliberation",
        "evaluate_joint_phonetic_head",
        "file_sha256",
        "fit_document_ranker_calibration",
        "group_top1_accuracy",
        "japanese_pronunciation_target",
        "label_document_decision",
        "load_feature_array",
        "load_phonetic_feature_manifest",
        "load_phonetic_source_manifest",
        "minimum_ctc_frames",
        "paired_bootstrap_cer_delta",
        "pairwise_accuracy",
        "posterior_configs_from_artifact",
        "posterior_sequence_from_logits",
        "rank_input_from_candidate",
        "read_mono_wav",
        "resolve_window_overlap",
        "save_joint_phonetic_weights",
        "train_document_ranker",
        "train_joint_phonetic_head",
        "validate_phonetic_split_disjointness",
        "with_joint_document_deliberation",
        "write_labeled_groups",
    )
    extension = (
        "\n"
        + start
        + "\n__all__.extend(\n    name for name in "
        + repr(names)
        + " if name not in __all__\n)\n"
        + end
        + "\n"
    )
    version = '\n__version__ = "0.2.0"\n'
    if version not in text:
        raise RuntimeError("root API version marker missing")
    text = text.replace(version, extension + version, 1)
    path.write_text(text, encoding="utf-8")


def reconcile_ci() -> None:
    path = Path(".github/workflows/ci.yml")
    text = path.read_text(encoding="utf-8")
    cli_anchor = "          semantic-asr transcribe-v2 --help\n"
    commands = (
        "          python scripts/benchmark_document_deliberation.py --help\n",
        "          python scripts/train_document_ranker.py --help\n",
        "          python scripts/train_joint_phonetic_head.py --help\n",
        "          python scripts/export_phonetic_features.py --help\n",
    )
    if cli_anchor not in text:
        raise RuntimeError("permanent CI CLI anchor missing")
    insertion = "".join(command for command in commands if command.strip() not in text)
    if insertion:
        text = text.replace(cli_anchor, cli_anchor + insertion, 1)
    torch_anchor = (
        "          python -m pip install 'torch>=2.4' --index-url "
        "https://download.pytorch.org/whl/cpu\n"
    )
    dependencies = (
        "          python -m pip install 'numpy>=1.26' 'safetensors>=0.4' "
        "'soundfile>=0.12'\n"
    )
    if dependencies.strip() not in text:
        if torch_anchor not in text:
            raise RuntimeError("permanent CI torch install anchor missing")
        text = text.replace(torch_anchor, torch_anchor + dependencies, 1)
    original = (
        "python -m pytest -q tests/test_training_optional.py "
        "tests/test_acoustic_verifier_optional.py"
    )
    extended = (
        original
        + " tests/test_phonetic_heads_optional.py"
        + " tests/test_joint_phonetic_runtime_optional.py"
        + " tests/test_train_joint_phonetic_head_cli_optional.py"
        + " tests/test_phonetic_feature_export.py"
        + " tests/test_soundfile_source_audio_loader.py"
    )
    if "tests/test_soundfile_source_audio_loader.py" not in text:
        if original not in text:
            raise RuntimeError("permanent CI optional test anchor missing")
        text = text.replace(original, extended, 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    replace_ranker_dataset_test()
    reconcile_ranker_tamper_test()
    reconcile_root_api()
    reconcile_ci()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
