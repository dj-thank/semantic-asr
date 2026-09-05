"""Public API for preregistered context × phonetic factorial experiments."""

from .context_scorer import (
    CallableCandidateContextScorer,
    CandidateContextScorer,
    ContextCandidate,
    ContextCandidateScore,
    GlobalSequenceCandidateContextAdapter,
)
from .metrics import (
    ContextPhoneticArmAggregate,
    ContextPhoneticCaseMetrics,
    FactorialInteractionContrast,
    GroupedPairedContrast,
    aggregate_factorial_arm,
    evaluate_factorial_case_arm,
    grouped_factorial_interaction,
    grouped_paired_contrast,
)
from .planner import (
    ContextScoreSet,
    PreparedContextPhoneticCase,
    PreparedContextPhoneticExperiment,
    deterministic_context_derangement,
    prepare_context_phonetic_experiment,
)
from .promotion import (
    ContextPhoneticPromotionCheck,
    ContextPhoneticPromotionDecision,
    ContextPhoneticPromotionPolicy,
    evaluate_context_phonetic_promotion,
)
from .protocol import (
    ContextPhoneticArm,
    ContextPhoneticCase,
    ContextPhoneticManifest,
    ContextPhoneticProtocol,
    FrozenContextSnapshot,
)
from .registration import (
    ContextPhoneticExperimentRegistration,
    RegisteredContextPhoneticResult,
    run_registered_context_phonetic_experiment,
)
from .runner import (
    ContextPhoneticCaseResult,
    ContextPhoneticFactorialReport,
    evaluate_prepared_context_phonetic_experiment,
    run_context_phonetic_experiment,
)
from .selection import (
    ContextPhoneticDecision,
    ScoredContextPhoneticCandidate,
    select_context_phonetic_arm,
)

__all__ = [
    "CallableCandidateContextScorer",
    "CandidateContextScorer",
    "ContextCandidate",
    "ContextCandidateScore",
    "ContextPhoneticArm",
    "ContextPhoneticArmAggregate",
    "ContextPhoneticCase",
    "ContextPhoneticCaseMetrics",
    "ContextPhoneticCaseResult",
    "ContextPhoneticDecision",
    "ContextPhoneticExperimentRegistration",
    "ContextPhoneticFactorialReport",
    "ContextPhoneticManifest",
    "ContextPhoneticPromotionCheck",
    "ContextPhoneticPromotionDecision",
    "ContextPhoneticPromotionPolicy",
    "ContextPhoneticProtocol",
    "ContextScoreSet",
    "FactorialInteractionContrast",
    "FrozenContextSnapshot",
    "GlobalSequenceCandidateContextAdapter",
    "GroupedPairedContrast",
    "PreparedContextPhoneticCase",
    "PreparedContextPhoneticExperiment",
    "RegisteredContextPhoneticResult",
    "ScoredContextPhoneticCandidate",
    "aggregate_factorial_arm",
    "deterministic_context_derangement",
    "evaluate_context_phonetic_promotion",
    "evaluate_factorial_case_arm",
    "evaluate_prepared_context_phonetic_experiment",
    "grouped_factorial_interaction",
    "grouped_paired_contrast",
    "prepare_context_phonetic_experiment",
    "run_context_phonetic_experiment",
    "run_registered_context_phonetic_experiment",
    "select_context_phonetic_arm",
]
