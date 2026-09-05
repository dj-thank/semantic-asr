from __future__ import annotations

import semantic_asr


def test_document_deliberation_research_surface_is_public() -> None:
    expected = {
        "DocumentBeamConfig",
        "DocumentDeliberatingTranscriber",
        "DocumentDeliberationDecision",
        "DocumentDeliberationPlan",
        "DocumentPathHypothesis",
        "FrozenWindowContext",
        "OverlapCompatibility",
        "WindowPathOption",
        "WindowPathSet",
        "apply_document_deliberation",
        "build_frozen_window_contexts",
        "plan_document_deliberation",
        "with_document_deliberation",
    }

    assert expected.issubset(set(semantic_asr.__all__))
    assert all(hasattr(semantic_asr, name) for name in expected)
