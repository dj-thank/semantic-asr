from __future__ import annotations

import semantic_asr.context_phonetic_experiment as experiment


def test_context_phonetic_factorial_surface_is_public() -> None:
    expected = {
        "ContextPhoneticExperimentRegistration",
        "ContextPhoneticManifest",
        "ContextPhoneticPromotionPolicy",
        "ContextPhoneticProtocol",
        "FrozenContextSnapshot",
        "GlobalSequenceCandidateContextAdapter",
        "PreparedContextPhoneticExperiment",
        "prepare_context_phonetic_experiment",
        "run_registered_context_phonetic_experiment",
    }

    assert expected.issubset(set(experiment.__all__))
    assert all(hasattr(experiment, name) for name in expected)
