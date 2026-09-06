from __future__ import annotations

import semantic_asr


def test_document_and_phonetic_research_surface_is_public() -> None:
    expected = {
        "DocumentDeliberatedResult",
        "DocumentDeliberationConfig",
        "DocumentEvaluationCase",
        "DocumentPassThroughResult",
        "DocumentPromotionGate",
        "DualPosteriorExtractor",
        "FrozenAudioPosteriorExtractor",
        "FrozenPosteriorModelConfig",
        "JointDocumentSemanticASRTranscriber",
        "JointPhoneticArtifact",
        "JointPhoneticHeadConfig",
        "OverlapPolicy",
        "PhoneticSpanProviderConfig",
        "SelectivePhoneticSpanProposalProvider",
        "SpanAudioReceipt",
        "apply_document_promotion_gate",
        "apply_joint_document_deliberation",
        "evaluate_document_deliberation",
        "with_joint_document_deliberation",
    }

    assert expected.issubset(set(semantic_asr.__all__))
    assert all(hasattr(semantic_asr, name) for name in expected)
