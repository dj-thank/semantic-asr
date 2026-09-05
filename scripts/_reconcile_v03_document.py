#!/usr/bin/env python3
"""One-shot source reconciler for the integrated v0.3 document stack."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def write_canonical_facades() -> None:
    compatibility = '''
    """Compatibility import for the canonical joint document-lattice engine."""

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

    __all__ = [name for name in globals() if not name.startswith("_")]
    '''
    Path("src/semantic_asr/document_joint_deliberation.py").write_text(
        textwrap.dedent(compatibility).lstrip(), encoding="utf-8"
    )
    facade = '''
    """Public research facade for document decoding, phonetic evidence, and evaluation."""

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
    from .phonetic_training import (
        JointPhoneticArtifact,
        JointPhoneticHeadConfig,
        PhoneticLabelInventory,
        PhoneticTrainingManifest,
        PhoneticValidationMetrics,
        posterior_configs_from_artifact,
    )

    __all__ = [name for name in globals() if not name.startswith("_")]
    '''
    Path("src/semantic_asr/document_deliberation.py").write_text(
        textwrap.dedent(facade).lstrip(), encoding="utf-8"
    )


def reconcile_document_engine() -> None:
    path = Path("src/semantic_asr/document_joint_engine.py")
    text = path.read_text(encoding="utf-8")
    if "maximum_windows: int" not in text:
        text = replace_once(
            text,
            "    maximum_changed_windows: int = 12\n    maximum_changed_ratio: float = 0.50",
            "    maximum_changed_windows: int = 12\n"
            "    maximum_windows: int = 512\n"
            "    maximum_total_local_options: int = 4_096\n"
            "    maximum_document_characters: int = 500_000\n"
            "    maximum_changed_ratio: float = 0.50",
            "document resource fields",
        )
        text = replace_once(
            text,
            '        for name in ("maximum_changed_windows",):',
            '        for name in (\n'
            '            "maximum_changed_windows",\n'
            '            "maximum_windows",\n'
            '            "maximum_total_local_options",\n'
            '            "maximum_document_characters",\n'
            '        ):',
            "document resource validation tuple",
        )
        marker = '''            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
'''
        replacement = '''            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.maximum_windows < 1:
            raise ValueError("maximum_windows must be positive")
        if self.maximum_total_local_options < 1:
            raise ValueError("maximum_total_local_options must be positive")
        if self.maximum_document_characters < 1:
            raise ValueError("maximum_document_characters must be positive")
        for name in (
'''
        text = replace_once(text, marker, replacement, "document resource positivity")

    if "proposal_kwargs = dict(kwargs)" not in text:
        text = replace_once(
            text,
            "            build = build_semantic_deliberation_lattice(**kwargs, proposals=proposals)",
            "            proposal_kwargs = dict(kwargs)\n"
            '            proposal_kwargs["proposals"] = proposals\n'
            "            build = build_semantic_deliberation_lattice(**proposal_kwargs)",
            "proposal keyword merge",
        )
    if "normalized_match == len(normalized_right)" not in text:
        text = replace_once(
            text,
            "        trim = right_map[normalized_match - 1]\n        if trim == len(right_text)",
            "        trim = right_map[normalized_match - 1]\n"
            "        if normalized_match == len(normalized_right):\n"
            "            trim = len(right_text)\n"
            "        if trim == len(right_text)",
            "full normalized overlap trim",
        )
    if "def _verify_first_pass_evidence(" not in text:
        marker = '''def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
'''
        helper = marker + '''

def _verify_first_pass_evidence(first_pass: LongformResult) -> None:
    expected = sha256_json(
        {
            "sourceAudioSha256": first_pass.source_audio_sha256,
            "durationMs": first_pass.duration_ms,
            "observedText": first_pass.observed_text,
            "normalizedText": first_pass.normalized_text,
            "segmentEvidence": [
                segment.observed.evidence_sha256 for segment in first_pass.segments
            ],
        }
    )
    if expected != first_pass.evidence_sha256:
        raise ValueError("first-pass long-form evidence hash mismatch")
'''
        text = replace_once(text, marker, helper, "first-pass evidence helper")
    pass_marker = '''        if self.first_pass.evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("pass-through result is linked to different first-pass evidence")
'''
    if pass_marker in text and pass_marker + "        _verify_first_pass_evidence(self.first_pass)\n" not in text:
        text = text.replace(
            pass_marker,
            pass_marker + "        _verify_first_pass_evidence(self.first_pass)\n",
            1,
        )
    result_marker = '''        if self.first_pass.evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("document result is linked to different first-pass evidence")
'''
    start = text.find("class DocumentDeliberatedResult")
    index = text.find(result_marker, start)
    if index >= 0 and "_verify_first_pass_evidence(self.first_pass)" not in text[
        index : index + len(result_marker) + 120
    ]:
        text = (
            text[:index]
            + result_marker
            + "        _verify_first_pass_evidence(self.first_pass)\n"
            + text[index + len(result_marker) :]
        )
    failure_marker = '''        if self.failure.first_pass_evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("failure receipt is linked to different first-pass evidence")
'''
    if "failure receipt is linked to different source audio" not in text:
        text = replace_once(
            text,
            failure_marker,
            failure_marker
            + '''        if self.failure.source_audio_sha256 != self.source_audio_sha256:
            raise ValueError("failure receipt is linked to different source audio")
''',
            "failure audio binding",
        )
    if "def _preserved_normalization(" not in text:
        marker = "\ndef _build_result(\n"
        helper = '''

def _preserved_normalization(
    first_pass_segment: LongformSegment,
    observed: DocumentObservedTranscript,
) -> DocumentNormalizedTranscript:
    return DocumentNormalizedTranscript(
        text=first_pass_segment.normalized.text,
        observed_evidence_sha256=observed.evidence_sha256,
        mode=getattr(first_pass_segment.normalized, "mode", "first-pass-linked"),
        normalizer_version=getattr(
            first_pass_segment.normalized,
            "normalizer_version",
            "first-pass-linked-v1",
        ),
    )
'''
        text = replace_once(text, marker, helper + marker, "preserved normalization helper")
    if "else _preserved_normalization(first_segment, observed)" not in text:
        text = replace_once(
            text,
            "                normalized=_normalization(observed),",
            "                normalized=(\n"
            "                    _normalization(observed)\n"
            "                    if decision.applied\n"
            "                    else _preserved_normalization(first_segment, observed)\n"
            "                ),",
            "unapplied normalization",
        )
    if '"localScore": self.local_score' not in text:
        marker = '''                    "emittedTextSha256": _text_sha256(self.text),
                },
            ),
        )
'''
        replacement = '''                    "emittedTextSha256": _text_sha256(self.text),
                    "localScore": self.local_score,
                    "overlapScore": self.overlap_score,
                    "meanAudioSupport": self.mean_audio_support,
                    "changedWindowCount": len(self.changed_window_indexes),
                    "generatedWindowCount": len(self.generated_window_indexes),
                    "ambiguousOverlapCount": len(self.ambiguous_overlap_indexes),
                    "windowCount": len(self.options),
                    "retainedPath": not self.changed_window_indexes,
                },
            ),
        )
'''
        text = replace_once(text, marker, replacement, "document scoring metadata")
    if "document exceeds maximum_windows" not in text:
        marker = '''    if config.require_document_scorer and document_scorer is None:
        raise ValueError("joint document deliberation requires an explicit document scorer")
    if not first_pass.segments:
        return first_pass
    try:
'''
        replacement = '''    if config.require_document_scorer and document_scorer is None:
        raise ValueError("joint document deliberation requires an explicit document scorer")
    if not first_pass.segments:
        return first_pass
    try:
        _verify_first_pass_evidence(first_pass)
        if len(first_pass.segments) > config.maximum_windows:
            raise ValueError("document exceeds maximum_windows")
        if sum(len(segment.observed.text) for segment in first_pass.segments) > (
            config.maximum_document_characters
        ):
            raise ValueError("document exceeds maximum_document_characters")
'''
        text = replace_once(text, marker, replacement, "document preflight guard")
    if "document exceeds maximum_total_local_options" not in text:
        marker = '''        options_by_window = tuple(
            _window_options(
                first_pass,
                index,
                config=config,
                build_config=build_config,
                local_policy=local_policy,
                proposal_provider=proposal_provider,
                declared_context=declared_context,
                audio_path=audio_path,
            )
            for index in range(len(first_pass.segments))
        )
        candidates = _expand_document_beam(options_by_window, config=config)
'''
        replacement = '''        options_by_window = tuple(
            _window_options(
                first_pass,
                index,
                config=config,
                build_config=build_config,
                local_policy=local_policy,
                proposal_provider=proposal_provider,
                declared_context=declared_context,
                audio_path=audio_path,
            )
            for index in range(len(first_pass.segments))
        )
        if sum(len(options) for options in options_by_window) > (
            config.maximum_total_local_options
        ):
            raise ValueError("document exceeds maximum_total_local_options")
        candidates = _expand_document_beam(options_by_window, config=config)
'''
        text = replace_once(text, marker, replacement, "document option guard")
    path.write_text(text, encoding="utf-8")


def reconcile_span_provider() -> None:
    path = Path("src/semantic_asr/phonetic_span_provider.py")
    text = path.read_text(encoding="utf-8")
    if "maximum_surface_length_delta" not in text:
        text = replace_once(
            text,
            "    proposals_per_span: int = 4\n    minimum_combined_utility: float = -0.20",
            "    proposals_per_span: int = 4\n"
            "    maximum_surface_length_delta: int = 2\n"
            "    maximum_lexicon_entries: int = 4_096\n"
            "    minimum_combined_utility: float = -0.20",
            "span provider resource fields",
        )
    elif "maximum_lexicon_entries" not in text:
        text = replace_once(
            text,
            "    maximum_surface_length_delta: int = 2\n    minimum_combined_utility: float = -0.20",
            "    maximum_surface_length_delta: int = 2\n"
            "    maximum_lexicon_entries: int = 4_096\n"
            "    minimum_combined_utility: float = -0.20",
            "span provider lexicon field",
        )
    if '"maximum_lexicon_entries"' not in text.split("def __post_init__", 1)[1].split(
        "@property", 1
    )[0]:
        text = replace_once(
            text,
            '            "maximum_surface_length_delta",\n        ):',
            '            "maximum_surface_length_delta",\n'
            '            "maximum_lexicon_entries",\n'
            '        ):',
            "span provider integer validation",
        )
    if "clip_sha256 = canonical_audio_sha256" not in text:
        text = replace_once(
            text,
            '''        clip = audio.samples[sample_start:sample_end]
        bundle = self.extractor.extract(
            clip,
            sample_rate=audio.sample_rate,
            source_audio_sha256=source_audio_sha256,
        )
''',
            '''        clip = audio.samples[sample_start:sample_end]
        clip_sha256 = canonical_audio_sha256(clip, audio.sample_rate)
        bundle = self.extractor.extract(
            clip,
            sample_rate=audio.sample_rate,
            source_audio_sha256=clip_sha256,
        )
''',
            "clip posterior binding",
        )
        text = replace_once(
            text,
            "            canonical_clip_sha256=canonical_audio_sha256(clip, audio.sample_rate),",
            "            canonical_clip_sha256=clip_sha256,",
            "clip receipt hash",
        )
    if "lexicon exceeds maximum_lexicon_entries" not in text:
        text = replace_once(
            text,
            "            lexicon = self.lexicon_provider(span=span, context=context, build=build)\n            proposals = propose_text_from_pronunciation(",
            "            lexicon = self.lexicon_provider(span=span, context=context, build=build)\n"
            "            if len(lexicon.entries) > self.config.maximum_lexicon_entries:\n"
            '                raise ValueError("lexicon exceeds maximum_lexicon_entries")\n'
            "            proposals = propose_text_from_pronunciation(",
            "span lexicon guard",
        )
    if "proposal posterior is bound to a different clip" not in text:
        text = replace_once(
            text,
            "            existing = {\n                arc.text: set(arc.independent_audio_channels) for arc in span.arcs\n            }",
            "            if any(\n"
            "                proposal.source_audio_sha256 != receipt.canonical_clip_sha256\n"
            "                for proposal in proposals\n"
            "            ):\n"
            '                raise ValueError("proposal posterior is bound to a different clip")\n'
            "            existing = {\n"
            "                arc.text: set(arc.independent_audio_channels) for arc in span.arcs\n"
            "            }",
            "proposal clip check",
        )
    if "posteriorClipSha256" not in text:
        text = replace_once(
            text,
            '                            "posteriorBundleDigest": bundle.digest,',
            '                            "posteriorBundleDigest": bundle.digest,\n'
            '                            "posteriorClipSha256": receipt.canonical_clip_sha256,\n'
            '                            "sourceRecordingSha256": source_audio_sha256,',
            "proposal dual provenance",
        )
    if "existing_lengths = [len(arc.text) for arc in span.arcs]" not in text:
        text = replace_once(
            text,
            "            verified = []\n            for proposal in proposals:\n",
            "            existing_lengths = [len(arc.text) for arc in span.arcs]\n"
            "            minimum_length = max(\n"
            "                0, min(existing_lengths) - self.config.maximum_surface_length_delta\n"
            "            )\n"
            "            maximum_length = (\n"
            "                max(existing_lengths) + self.config.maximum_surface_length_delta\n"
            "            )\n"
            "            verified = []\n"
            "            for proposal in proposals:\n"
            "                if not minimum_length <= len(proposal.text) <= maximum_length:\n"
            "                    continue\n",
            "span surface-length guard",
        )
    path.write_text(text, encoding="utf-8")


def reconcile_ranker() -> None:
    path = Path("src/semantic_asr/document_ranker.py")
    text = path.read_text(encoding="utf-8").replace(
        "from .contracts import canonical_json, sha256_json",
        "from .contracts import sha256_json",
    )
    old = '''        model_payload = dict(payload["model"])
        feature = DocumentFeatureConfig(**model_payload.pop("featureConfig"))
        model = DocumentLinearRanker(
            feature_config=feature,
            sparse_weights=tuple(tuple(row) for row in model_payload.pop("sparseWeights")),
            dense_means=tuple(model_payload.pop("denseMeans")),
            dense_scales=tuple(model_payload.pop("denseScales")),
            dense_weights=tuple(model_payload.pop("denseWeights")),
            epoch_losses=tuple(model_payload.pop("epochLosses")),
            **model_payload,
        )
'''
    if old in text:
        new = '''        model_payload = dict(payload["model"])
        expected_model_keys = {
            "featureConfig",
            "sparseWeights",
            "denseMeans",
            "denseScales",
            "denseWeights",
            "bias",
            "trainingConfigDigest",
            "trainingManifestSha256",
            "trainingExampleDigest",
            "epochLosses",
            "pairwiseAccuracy",
            "revision",
            "schemaVersion",
        }
        if set(model_payload) != expected_model_keys:
            raise ValueError("document ranker model schema is not exact")
        feature = DocumentFeatureConfig(**model_payload["featureConfig"])
        model = DocumentLinearRanker(
            feature_config=feature,
            sparse_weights=tuple(
                (int(row[0]), float(row[1])) for row in model_payload["sparseWeights"]
            ),
            dense_means=tuple(float(value) for value in model_payload["denseMeans"]),
            dense_scales=tuple(float(value) for value in model_payload["denseScales"]),
            dense_weights=tuple(float(value) for value in model_payload["denseWeights"]),
            bias=float(model_payload["bias"]),
            training_config_digest=str(model_payload["trainingConfigDigest"]),
            training_manifest_sha256=str(model_payload["trainingManifestSha256"]),
            training_example_digest=str(model_payload["trainingExampleDigest"]),
            epoch_losses=tuple(float(value) for value in model_payload["epochLosses"]),
            pairwise_accuracy=float(model_payload["pairwiseAccuracy"]),
            revision=str(model_payload["revision"]),
            schema_version=str(model_payload["schemaVersion"]),
        )
'''
        text = text.replace(old, new, 1)
    if "document feature index exceeds model dimension" not in text:
        marker = '''        if features.config_digest != self.feature_config.digest:
            raise ValueError("document features were created with a different feature config")
        sparse = self.sparse_weight_map
'''
        replacement = '''        if features.config_digest != self.feature_config.digest:
            raise ValueError("document features were created with a different feature config")
        if any(index >= self.feature_config.hash_dimension for index, _ in features.indices):
            raise ValueError("document feature index exceeds model dimension")
        sparse = self.sparse_weight_map
'''
        text = replace_once(text, marker, replacement, "ranker feature dimension guard")
    path.write_text(text, encoding="utf-8")

    path = Path("src/semantic_asr/document_ranker_dataset.py")
    text = path.read_text(encoding="utf-8").replace(
        '"referenceUse": "offline-label-only",',
        '"labelBoundary": "offline-only",',
    )
    path.write_text(text, encoding="utf-8")

    path = Path("scripts/train_document_ranker.py")
    text = path.read_text(encoding="utf-8")
    marker = '''    training_digest = manifest_sha256(args.train)
    calibration_digest = manifest_sha256(args.calibration)
    test_digest = manifest_sha256(args.test)
'''
    if "train, calibration, and test manifests must differ" not in text:
        text = replace_once(
            text,
            marker,
            marker
            + '''    if len({training_digest, calibration_digest, test_digest}) != 3:
        raise ValueError("train, calibration, and test manifests must differ")
''',
            "ranker distinct manifest guard",
        )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    write_canonical_facades()
    reconcile_document_engine()
    reconcile_span_provider()
    reconcile_ranker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
