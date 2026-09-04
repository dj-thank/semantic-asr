from __future__ import annotations

import semantic_asr


def test_longform_deliberation_research_surface_is_public() -> None:
    expected = {
        "DeliberatedLongformResult",
        "DeliberatedObservedTranscript",
        "DocumentPromptFormat",
        "GlobalScoreNormalization",
        "LongformDeliberationConfig",
        "SemanticDeliberationConfig",
        "SequenceScorerGlobalAdapter",
        "SourcePath",
        "VerifiedSpanProposal",
        "apply_longform_deliberation",
        "build_semantic_deliberation_lattice",
        "with_global_deliberation",
    }

    assert expected.issubset(set(semantic_asr.__all__))
    assert all(hasattr(semantic_asr, name) for name in expected)
