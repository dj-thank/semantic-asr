"""Public API for reference-separated phonetic proposal experiments."""

from .metrics import (
    PairedErrorDelta,
    PhoneticArmAggregate,
    PhoneticCaseArmMetrics,
    aggregate_arm,
    edit_distance,
    evaluate_case_arm,
    paired_bootstrap_error_delta,
)
from .planner import (
    FrozenPhoneticCandidate,
    FrozenPhoneticCandidatePlanner,
    FrozenPhoneticCandidatePool,
    PlanningCaseView,
)
from .promotion import (
    PhoneticPromotionDecision,
    PhoneticPromotionPolicy,
    PromotionCheck,
    evaluate_phonetic_promotion,
)
from .protocol import (
    FirstPassSpanCandidate,
    FrozenSpanReference,
    PhoneticAblationArm,
    PhoneticAblationCase,
    PhoneticAblationManifest,
    PhoneticAblationProtocol,
)
from .runner import (
    PhoneticAblationCaseResult,
    PhoneticAblationReport,
    PreparedPhoneticAblation,
    evaluate_prepared_phonetic_ablation,
    prepare_phonetic_ablation,
    run_phonetic_ablation,
)
from .selection import (
    PhoneticAblationDecision,
    ScoredPhoneticCandidate,
    select_phonetic_arm,
)

__all__ = [
    "FirstPassSpanCandidate",
    "FrozenPhoneticCandidate",
    "FrozenPhoneticCandidatePlanner",
    "FrozenPhoneticCandidatePool",
    "FrozenSpanReference",
    "PairedErrorDelta",
    "PhoneticAblationArm",
    "PhoneticAblationCase",
    "PhoneticAblationCaseResult",
    "PhoneticAblationDecision",
    "PhoneticAblationManifest",
    "PhoneticAblationProtocol",
    "PhoneticAblationReport",
    "PhoneticArmAggregate",
    "PhoneticCaseArmMetrics",
    "PhoneticPromotionDecision",
    "PhoneticPromotionPolicy",
    "PlanningCaseView",
    "PreparedPhoneticAblation",
    "PromotionCheck",
    "ScoredPhoneticCandidate",
    "aggregate_arm",
    "edit_distance",
    "evaluate_case_arm",
    "evaluate_phonetic_promotion",
    "evaluate_prepared_phonetic_ablation",
    "paired_bootstrap_error_delta",
    "prepare_phonetic_ablation",
    "run_phonetic_ablation",
    "select_phonetic_arm",
]
