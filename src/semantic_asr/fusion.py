from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import fmean

from .calibration import (
    CalibrationProfile,
    ScoreRankFeatures,
    calibrate_values,
    score_rank_confidence,
)
from .contracts import CandidateEvidence, EvidenceName, GateDecision, RankedCandidate

STREAMS: tuple[EvidenceName, ...] = (
    "acoustic",
    "mora",
    "lexical",
    "preservation",
    "cross_model",
)
ACOUSTIC_FAMILY = frozenset({"acoustic", "mora", "cross_model"})


@dataclass(frozen=True, slots=True)
class FusionConfig:
    priors: dict[EvidenceName, float] = field(
        default_factory=lambda: {
            "acoustic": 0.42,
            "mora": 0.23,
            "lexical": 0.08,
            "preservation": 0.12,
            "cross_model": 0.15,
        }
    )
    calibration_profiles: dict[EvidenceName, CalibrationProfile] = field(default_factory=dict)
    posterior_temperature: float = 0.14
    stream_temperature: float = 0.20
    acoustic_family_floor: float = 0.72
    missing_evidence_penalty: float = 0.10
    source_diversity_bonus: float = 0.018
    grammar_honeytrap_strength: float = 0.25
    grammar_honeytrap_deadband: float = 0.08
    relisten_entropy: float = 0.55
    relisten_disagreement: float = 0.22
    relisten_margin: float = 0.18
    max_selective_risk: float = 0.32
    minimum_evidence_coverage: float = 0.55
    acceptance_posterior: float = 0.64

    def __post_init__(self) -> None:
        if set(self.priors) != set(STREAMS):
            raise ValueError(f"priors must contain exactly {STREAMS}")
        if any(not math.isfinite(value) or value < 0 for value in self.priors.values()):
            raise ValueError("fusion priors must be finite and non-negative")
        if sum(self.priors.values()) <= 0:
            raise ValueError("at least one fusion prior must be positive")
        if self.posterior_temperature <= 0 or self.stream_temperature <= 0:
            raise ValueError("softmax temperatures must be positive")
        for value in (
            self.acoustic_family_floor,
            self.minimum_evidence_coverage,
            self.acceptance_posterior,
        ):
            if not 0 <= value <= 1:
                raise ValueError("probability thresholds must be in [0, 1]")


def _softmax(values: list[float], temperature: float) -> list[float]:
    maximum = max(values)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (value - maximum) / temperature))) for value in values
    ]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    raw = -sum(probability * math.log(probability + 1e-12) for probability in probabilities)
    return min(1.0, max(0.0, raw / math.log(len(probabilities))))


def _kl_divergence(left: list[float], right: list[float]) -> float:
    return sum(
        probability * math.log((probability + 1e-12) / (other + 1e-12))
        for probability, other in zip(left, right, strict=True)
        if probability > 0
    )


def _jensen_shannon(distributions: list[list[float]]) -> float:
    if len(distributions) <= 1:
        return 0.0
    mixture = [
        sum(distribution[index] for distribution in distributions) / len(distributions)
        for index in range(len(distributions[0]))
    ]
    raw = sum(_kl_divergence(distribution, mixture) for distribution in distributions)
    raw /= len(distributions)
    maximum = math.log(max(2, len(distributions[0])))
    return min(1.0, max(0.0, raw / maximum))


def _stream_reliability(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return 0.0
    coverage = len(finite) / len(values)
    spread = max(finite) - min(finite) if len(finite) > 1 else 0.0
    return coverage * min(1.0, 0.20 + 0.80 * spread)


def _enforce_family_floor(
    weights: dict[EvidenceName, float], floor: float
) -> dict[EvidenceName, float]:
    family = sum(weights[stream] for stream in ACOUSTIC_FAMILY)
    if family >= floor or family <= 0:
        return weights
    language_streams = [stream for stream in STREAMS if stream not in ACOUSTIC_FAMILY]
    language_total = sum(weights[stream] for stream in language_streams)
    if language_total <= 0:
        return weights
    output = dict(weights)
    family_scale = floor / family
    language_scale = (1.0 - floor) / language_total
    for stream in ACOUSTIC_FAMILY:
        output[stream] *= family_scale
    for stream in language_streams:
        output[stream] *= language_scale
    return output


def _calibration_digest(
    profiles: Mapping[EvidenceName, CalibrationProfile],
) -> str:
    payload = {
        stream: profiles[stream].digest if stream in profiles else "auto-robust-v1"
        for stream in STREAMS
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _beam_confidences(candidates: list[CandidateEvidence]) -> list[float | None]:
    ordered = sorted(
        (
            float(candidate.avg_logprob)
            for candidate in candidates
            if candidate.avg_logprob is not None
        ),
        reverse=True,
    )
    output: list[float | None] = []
    for candidate in candidates:
        if candidate.beam_confidence is not None:
            output.append(float(candidate.beam_confidence))
            continue
        if candidate.rank is None or candidate.hypothesis_count is None:
            output.append(None)
            continue
        margin = None
        if candidate.avg_logprob is not None and ordered:
            try:
                index = ordered.index(float(candidate.avg_logprob))
            except ValueError:
                index = -1
            if 0 <= index < len(ordered) - 1:
                margin = float(candidate.avg_logprob) - ordered[index + 1]
            elif len(ordered) == 1:
                margin = 1.0
        output.append(
            score_rank_confidence(
                ScoreRankFeatures(
                    rank=candidate.rank,
                    hypothesis_count=candidate.hypothesis_count,
                    avg_logprob=candidate.avg_logprob,
                    margin_to_next=margin,
                    token_count=len(candidate.token_ids) or None,
                )
            )
        )
    return output


def _source_bonus(candidate: CandidateEvidence, config: FusionConfig) -> float:
    return config.source_diversity_bonus * min(3, max(0, len(candidate.source_support) - 1))


def fuse_candidates(
    candidates: list[CandidateEvidence],
    config: FusionConfig | None = None,
) -> list[RankedCandidate]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    config = config or FusionConfig()
    all_degenerate = all(bool(candidate.metadata.get("degenerate")) for candidate in candidates)
    primary_candidates = [
        candidate for candidate in candidates if not candidate.metadata.get("secondEarCandidate")
    ]
    primary_all_degenerate = bool(primary_candidates) and all(
        bool(candidate.metadata.get("degenerate")) for candidate in primary_candidates
    )
    hard_degenerate_gate = all_degenerate or primary_all_degenerate

    calibrated: dict[EvidenceName, list[float | None]] = {
        stream: calibrate_values(
            [candidate.score(stream) for candidate in candidates],
            profile=config.calibration_profiles.get(stream),
            stream_name=stream,
        )
        for stream in STREAMS
    }
    for index, beam_confidence in enumerate(_beam_confidences(candidates)):
        if beam_confidence is None:
            continue
        acoustic = calibrated["acoustic"][index]
        calibrated["acoustic"][index] = (
            beam_confidence if acoustic is None else 0.76 * acoustic + 0.24 * beam_confidence
        )

    reliabilities = {stream: _stream_reliability(calibrated[stream]) for stream in STREAMS}
    raw_weights = {
        stream: config.priors[stream] * (0.30 + reliabilities[stream]) for stream in STREAMS
    }
    total = sum(raw_weights.values()) or 1.0
    weights = {stream: raw_weights[stream] / total for stream in STREAMS}
    weights = _enforce_family_floor(weights, config.acoustic_family_floor)

    stream_distributions: list[list[float]] = []
    for stream in STREAMS:
        values = calibrated[stream]
        if not any(value is not None for value in values):
            continue
        stream_distributions.append(
            _softmax(
                [-8.0 if value is None else float(value) for value in values],
                config.stream_temperature,
            )
        )
    disagreement = _jensen_shannon(stream_distributions)

    preliminary: list[
        tuple[
            CandidateEvidence,
            float,
            dict[str, float],
            float,
            float,
            float,
            float,
        ]
    ] = []
    for index, candidate in enumerate(candidates):
        parts = {
            stream: float(calibrated[stream][index])
            if calibrated[stream][index] is not None
            else 0.0
            for stream in STREAMS
        }
        present_weight = sum(
            weights[stream] for stream in STREAMS if calibrated[stream][index] is not None
        )
        missing_penalty = (1.0 - present_weight) * config.missing_evidence_penalty
        source_bonus = _source_bonus(candidate, config)
        score = sum(weights[stream] * parts[stream] for stream in STREAMS)
        score -= missing_penalty
        score += source_bonus

        family_weight = sum(weights[stream] for stream in ACOUSTIC_FAMILY)
        family_support = (
            sum(weights[stream] * parts[stream] for stream in ACOUSTIC_FAMILY) / family_weight
            if family_weight > 0
            else 0.0
        )
        honeytrap = 0.0
        if candidate.teacher is not None:
            unsupported = max(
                0.0,
                float(candidate.teacher) - family_support - config.grammar_honeytrap_deadband,
            )
            honeytrap = config.grammar_honeytrap_strength * unsupported
            score -= honeytrap
        preliminary.append(
            (
                candidate,
                score,
                parts,
                honeytrap,
                missing_penalty,
                source_bonus,
                present_weight,
            )
        )

    posterior = _softmax([item[1] for item in preliminary], config.posterior_temperature)
    order = sorted(
        range(len(preliminary)),
        key=lambda index: (
            -posterior[index],
            -preliminary[index][1],
            preliminary[index][0].candidate_id,
        ),
    )
    top_index = order[0]
    top_probability = posterior[top_index]
    second_probability = posterior[order[1]] if len(order) > 1 else 0.0
    margin = top_probability - second_probability
    entropy = _entropy(posterior)
    evidence_coverage = preliminary[top_index][6]
    selective_risk = min(
        1.0,
        max(
            0.0,
            0.56 * (1.0 - top_probability) + 0.24 * disagreement + 0.20 * (1.0 - evidence_coverage),
        ),
    )
    needs_relisten = (
        hard_degenerate_gate
        or entropy >= config.relisten_entropy
        or disagreement >= config.relisten_disagreement
        or margin <= config.relisten_margin
        or selective_risk >= config.max_selective_risk
        or evidence_coverage < config.minimum_evidence_coverage
    )
    abstain = hard_degenerate_gate or (
        top_probability < config.acceptance_posterior
        and (
            selective_risk >= config.max_selective_risk
            or evidence_coverage < config.minimum_evidence_coverage
            or disagreement >= config.relisten_disagreement
        )
    )
    reasons: list[str] = []
    if hard_degenerate_gate:
        reasons.append("all-candidates-degenerate")
    if primary_all_degenerate and not all_degenerate:
        reasons.append("all-primary-candidates-degenerate")
    if entropy >= config.relisten_entropy:
        reasons.append("high-candidate-entropy")
    if disagreement >= config.relisten_disagreement:
        reasons.append("evidence-stream-disagreement")
    if margin <= config.relisten_margin:
        reasons.append("small-posterior-margin")
    if selective_risk >= config.max_selective_risk:
        reasons.append("high-selective-risk")
    if evidence_coverage < config.minimum_evidence_coverage:
        reasons.append("low-evidence-coverage")
    if abstain:
        reasons.append("provisional-observation")

    posterior_map = {
        candidate.candidate_id: probability
        for candidate, probability in zip(candidates, posterior, strict=True)
    }
    gate = GateDecision(
        weights=weights,
        posterior=posterior_map,
        entropy=entropy,
        disagreement=disagreement,
        evidence_coverage=evidence_coverage,
        selective_risk=selective_risk,
        needs_relisten=needs_relisten,
        abstain=abstain,
        reasons=tuple(reasons),
        calibration_digest=_calibration_digest(config.calibration_profiles),
        uncertainty={
            "aleatoric": entropy,
            "epistemic": disagreement,
            "missingEvidence": 1.0 - evidence_coverage,
            "posteriorMargin": margin,
        },
    )
    ranked = [
        RankedCandidate(
            candidate=candidate,
            final_score=score,
            posterior=probability,
            calibrated_scores=parts,
            gate=gate,
            grammar_honeytrap_penalty=honeytrap,
            missing_evidence_penalty=missing_penalty,
            source_diversity_bonus=source_bonus,
        )
        for (
            candidate,
            score,
            parts,
            honeytrap,
            missing_penalty,
            source_bonus,
            _coverage,
        ), probability in zip(preliminary, posterior, strict=True)
    ]
    return sorted(
        ranked,
        key=lambda item: (-item.posterior, -item.final_score, item.candidate.candidate_id),
    )


def evidence_summary(ranked: list[RankedCandidate]) -> dict[str, object]:
    if not ranked:
        raise ValueError("ranked candidates are required")
    gate = ranked[0].gate
    return {
        "entropy": gate.entropy,
        "disagreement": gate.disagreement,
        "evidenceCoverage": gate.evidence_coverage,
        "selectiveRisk": gate.selective_risk,
        "needsRelisten": gate.needs_relisten,
        "abstain": gate.abstain,
        "reasons": list(gate.reasons),
        "topPosterior": ranked[0].posterior,
        "topScore": ranked[0].final_score,
        "meanScore": fmean(item.final_score for item in ranked),
        "calibrationDigest": gate.calibration_digest,
        "uncertainty": dict(gate.uncertainty),
    }
