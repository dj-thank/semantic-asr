"""Paper-inspired discrete-unit surprisal and centroid-DTW evidence."""

from .discrete_unit_alignment import (
    CentroidDistanceTable,
    CentroidDTWFeatures,
    DTWAlignment,
    DTWConfig,
    ProjectionMode,
    TranscriptGuidedFeatures,
    align_collapsed_units,
    centroid_dtw_features,
    pronunciation_feature_vector,
    transcript_guided_features,
)
from .discrete_unit_ranker import (
    DiscreteUnitAcousticRanker,
    DiscreteUnitCandidateScore,
    StaticTextToDiscreteUnitEncoder,
    TextToDiscreteUnitEncoder,
)
from .discrete_units import (
    DISCRETE_SURPRISAL_PAPER_REVISION,
    CollapsedUnitSequence,
    DiscreteTokenLanguageModel,
    DiscreteUnitSequence,
    DiscreteUnitSpace,
    SurprisalProfile,
    SurprisalThreshold,
)

__all__ = [
    "DISCRETE_SURPRISAL_PAPER_REVISION",
    "CentroidDTWFeatures",
    "CentroidDistanceTable",
    "CollapsedUnitSequence",
    "DTWAlignment",
    "DTWConfig",
    "DiscreteTokenLanguageModel",
    "DiscreteUnitAcousticRanker",
    "DiscreteUnitCandidateScore",
    "DiscreteUnitSequence",
    "DiscreteUnitSpace",
    "ProjectionMode",
    "StaticTextToDiscreteUnitEncoder",
    "SurprisalProfile",
    "SurprisalThreshold",
    "TextToDiscreteUnitEncoder",
    "TranscriptGuidedFeatures",
    "align_collapsed_units",
    "centroid_dtw_features",
    "pronunciation_feature_vector",
    "transcript_guided_features",
]
