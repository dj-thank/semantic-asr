from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from .contracts import CandidateEvidence, sha256_json
from .discrete_unit_alignment import (
    CentroidDistanceTable,
    CentroidDTWFeatures,
    DTWConfig,
    TranscriptGuidedFeatures,
    centroid_dtw_features,
    transcript_guided_features,
)
from .discrete_units import (
    DISCRETE_SURPRISAL_PAPER_REVISION,
    DiscreteTokenLanguageModel,
    DiscreteUnitSequence,
    DiscreteUnitSpace,
    ensure_same_unit_space,
    validate_sha256,
)
from .score_types import EvidenceScore, ScoreSemantics


class TextToDiscreteUnitEncoder(Protocol):
    """Frozen text-to-unit adapter sharing the Audio2DUnit codebook exactly."""

    name: str
    revision: str
    configuration_digest: str
    space: DiscreteUnitSpace

    def encode(self, text: str) -> DiscreteUnitSequence: ...


class StaticTextToDiscreteUnitEncoder:
    """Deterministic fixture/reference adapter; not a learned Text2DUnit model."""

    def __init__(
        self,
        mapping: Mapping[str, Sequence[int]],
        *,
        space: DiscreteUnitSpace,
        revision: str,
        name: str = "static-text2dunit",
    ) -> None:
        if not mapping or not isinstance(name, str) or not name.strip():
            raise ValueError("a non-empty text-to-unit mapping and name are required")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("an immutable text encoder revision is required")
        if any(not isinstance(text, str) or not text for text in mapping):
            raise ValueError("text-to-unit mapping keys must be non-empty strings")
        normalized = {text: tuple(units) for text, units in mapping.items()}
        self.space = space
        self.name = name.strip()
        self.revision = revision.strip()
        for units in normalized.values():
            DiscreteUnitSequence(units=units, space=space)
        self.mapping = MappingProxyType(normalized)
        self.configuration_digest = sha256_json(
            {
                "name": self.name,
                "revision": self.revision,
                "unitSpaceDigest": space.digest,
                "mapping": self.mapping,
                "fixture": True,
            }
        )

    def encode(self, text: str) -> DiscreteUnitSequence:
        try:
            units = self.mapping[text]
        except KeyError as exc:
            text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            raise KeyError(
                f"no canonical discrete units are available for candidate sha256={text_digest}"
            ) from exc
        return DiscreteUnitSequence(
            units=units,
            space=self.space,
            source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


CandidateFeatureSet = CentroidDTWFeatures | TranscriptGuidedFeatures


@dataclass(frozen=True, slots=True)
class DiscreteUnitCandidateScore:
    candidate_id: str
    alignment_cost: EvidenceScore
    rank_score: EvidenceScore
    features: CandidateFeatureSet

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.alignment_cost.semantics != ScoreSemantics.COST:
            raise ValueError("alignment_cost must preserve COST semantics")
        if self.rank_score.semantics != ScoreSemantics.UNCALIBRATED_SCORE:
            raise ValueError("rank_score must remain uncalibrated")
        if not math.isclose(
            self.rank_score.value,
            -self.alignment_cost.value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("rank_score must be the negated DTW cost")
        if not math.isclose(
            self.alignment_cost.value,
            self.features.dtw_distance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("alignment_cost must match the attached DTW features")
        if self.alignment_cost.provenance.digest != self.rank_score.provenance.digest:
            raise ValueError("alignment_cost and rank_score must share identical provenance")
        if (
            self.alignment_cost.provenance.metadata.get("candidateFeatureDigest")
            != self.features.digest
        ):
            raise ValueError("score provenance must bind the attached candidate features")

    @property
    def includes_surprisal_features(self) -> bool:
        return isinstance(self.features, TranscriptGuidedFeatures)


class DiscreteUnitAcousticRanker:
    """Candidate-specific zero-shot ranker based only on canonical-unit DTW cost.

    The native token LM is optional. When supplied, transcript-guided surprisal
    features are recorded for held-out fitting, but the zero-shot rank score remains
    exactly ``-DTW distance``. Audio-only surprisal is never added to candidate
    scores because it is identical for every hypothesis of the same utterance.
    """

    def __init__(
        self,
        *,
        observed: DiscreteUnitSequence,
        distance_table: CentroidDistanceTable,
        text_encoder: TextToDiscreteUnitEncoder,
        token_lm: DiscreteTokenLanguageModel | None = None,
        alpha: float = 0.5,
        config: DTWConfig | None = None,
    ) -> None:
        self.observed = observed
        self.token_lm = token_lm
        self.distance_table = distance_table
        self.text_encoder = text_encoder
        if isinstance(alpha, bool):
            raise TypeError("alpha must be a real number")
        try:
            self.alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise TypeError("alpha must be a real number") from exc
        self.config = config or DTWConfig()
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if token_lm is not None:
            ensure_same_unit_space(observed.space, token_lm.space, name="token LM")
        ensure_same_unit_space(observed.space, distance_table.space, name="distance table")
        ensure_same_unit_space(observed.space, text_encoder.space, name="text encoder")
        if (
            not text_encoder.name.strip()
            or not text_encoder.revision.strip()
            or not text_encoder.configuration_digest
        ):
            raise ValueError(
                "text encoder name, immutable revision and configuration digest are required"
            )
        validate_sha256(
            text_encoder.configuration_digest,
            name="text encoder configuration_digest",
        )
        self.config_digest = sha256_json(
            {
                "schema": "discrete-unit-acoustic-ranker-v1",
                "paperRevision": DISCRETE_SURPRISAL_PAPER_REVISION,
                "unitSpaceDigest": observed.space.digest,
                "tokenLmDigest": token_lm.digest if token_lm is not None else None,
                "distanceTableDigest": distance_table.digest,
                "textEncoder": text_encoder.name,
                "textEncoderRevision": text_encoder.revision,
                "textEncoderConfigurationDigest": text_encoder.configuration_digest,
                "alpha": self.alpha,
                "dtwConfigDigest": self.config.digest,
                "zeroShotRankFeature": "negative-dtw-distance",
            }
        )
        self.name = f"discrete-unit-dtw:{self.config_digest[:16]}"
        self.model_name = self.text_encoder.name
        self.model_revision = self.text_encoder.revision

    def _candidate_features(self, canonical: DiscreteUnitSequence) -> CandidateFeatureSet:
        if self.token_lm is None:
            return centroid_dtw_features(
                self.observed,
                canonical,
                distance_table=self.distance_table,
                config=self.config,
            )
        return transcript_guided_features(
            self.observed,
            canonical,
            token_lm=self.token_lm,
            distance_table=self.distance_table,
            alpha=self.alpha,
            config=self.config,
        )

    def score_detailed(
        self,
        candidates: Sequence[CandidateEvidence],
    ) -> tuple[DiscreteUnitCandidateScore, ...]:
        if not candidates:
            raise ValueError("at least one candidate is required")
        identifiers = [candidate.candidate_id for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate IDs must be unique")
        output: list[DiscreteUnitCandidateScore] = []
        for candidate in candidates:
            canonical = self.text_encoder.encode(candidate.text)
            ensure_same_unit_space(self.observed.space, canonical.space, name="canonical candidate")
            candidate_text_sha256 = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
            if canonical.source_sha256 != candidate_text_sha256:
                raise ValueError(
                    "text encoder output source SHA-256 does not match the candidate text"
                )
            features = self._candidate_features(canonical)
            metadata: dict[str, object] = {
                "paperRevision": DISCRETE_SURPRISAL_PAPER_REVISION,
                "unitSpaceDigest": self.observed.space.digest,
                "tokenLmDigest": self.token_lm.digest if self.token_lm is not None else None,
                "distanceTableDigest": self.distance_table.digest,
                "textEncoderConfigurationDigest": self.text_encoder.configuration_digest,
                "candidateTextSha256": candidate_text_sha256,
                "candidateFeatureDigest": features.digest,
                "candidateIndependentSurprisalUsedForRanking": False,
                "candidateFeatureSet": (
                    "centroid-dtw-surprisal"
                    if isinstance(features, TranscriptGuidedFeatures)
                    else "centroid-dtw"
                ),
                "zeroShotRankFeature": "negative-dtw-distance",
                **features.as_dict(),
            }
            provenance = {
                "scorer": self.name,
                "model": self.text_encoder.name,
                "revision": self.text_encoder.revision,
                "runtime": "python-stdlib-centroid-dtw",
                "configuration_digest": self.config_digest,
                "input_evidence_digest": self.observed.digest,
                "metadata": metadata,
            }
            alignment_cost = EvidenceScore.raw(
                features.dtw_distance,
                semantics=ScoreSemantics.COST,
                **provenance,
            )
            rank_score = EvidenceScore.raw(
                -features.dtw_distance,
                semantics=ScoreSemantics.UNCALIBRATED_SCORE,
                **provenance,
            )
            output.append(
                DiscreteUnitCandidateScore(
                    candidate_id=candidate.candidate_id,
                    alignment_cost=alignment_cost,
                    rank_score=rank_score,
                    features=features,
                )
            )
        return tuple(output)

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        del context, consensus, contradiction
        return {row.candidate_id: row.rank_score.value for row in self.score_detailed(candidates)}
