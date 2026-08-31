from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from .contracts import CandidateEvidence


def logsumexp(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("logsumexp requires at least one finite value")
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def surface_key(text: str) -> str:
    """Canonicalize representation only; never grammar-correct observed text."""

    return unicodedata.normalize("NFC", str(text)).strip()


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _path_cumulative_logprob(candidate: CandidateEvidence) -> float | None:
    for key in ("aggregateCumulativeLogprob", "cumulativeLogprob"):
        value = _finite(candidate.metadata.get(key))
        if value is not None:
            return value
    if candidate.avg_logprob is not None:
        token_count = max(1, len(candidate.token_ids))
        return float(candidate.avg_logprob) * (token_count + 1)
    return None


def _score_domain(candidate: CandidateEvidence) -> str:
    explicit = candidate.metadata.get("scoreDomain")
    if explicit:
        return str(explicit)
    adapter = candidate.metadata.get("adapter") or candidate.evidence_source
    model = candidate.metadata.get("model") or "unknown-model"
    namespace = candidate.metadata.get("decodeNamespace") or "single-decode"
    start = candidate.metadata.get("decodeStartMs")
    end = candidate.metadata.get("decodeEndMs")
    return f"{adapter}|{model}|{namespace}|{start}:{end}"


def _score_or_floor(value: object) -> float:
    numeric = _finite(value)
    return -1e30 if numeric is None else numeric


def _strength(candidate: CandidateEvidence) -> tuple[float, float, float, float]:
    return (
        _score_or_floor(candidate.avg_logprob),
        _score_or_floor(candidate.acoustic),
        _score_or_floor(candidate.mora),
        _score_or_floor(candidate.sequence_score),
    )


def _aggregate_domain(rows: list[CandidateEvidence]) -> CandidateEvidence:
    representative = max(rows, key=lambda row: (_strength(row), row.candidate_id))
    cumulative = [value for row in rows if (value := _path_cumulative_logprob(row)) is not None]
    metadata = dict(representative.metadata)
    metadata.update(
        {
            "scoreDomain": _score_domain(representative),
            "pathCandidateIds": sorted(
                {
                    str(candidate_id)
                    for row in rows
                    for candidate_id in row.metadata.get("pathCandidateIds", [row.candidate_id])
                }
            ),
            "pathCount": sum(int(row.metadata.get("pathCount", 1)) for row in rows),
            "pathTokenIds": [
                list(token_ids)
                for row in rows
                for token_ids in row.metadata.get(
                    "pathTokenIds", [list(row.token_ids)] if row.token_ids else []
                )
            ],
            "pathSequenceScores": [
                value
                for row in rows
                for value in row.metadata.get(
                    "pathSequenceScores",
                    [row.sequence_score] if row.sequence_score is not None else [],
                )
            ],
            "pathAvgLogprobs": [
                value
                for row in rows
                for value in row.metadata.get(
                    "pathAvgLogprobs",
                    [row.avg_logprob] if row.avg_logprob is not None else [],
                )
            ],
        }
    )
    if cumulative:
        aggregate_cumulative = logsumexp(cumulative)
        representative_tokens = max(1, len(representative.token_ids))
        aggregate_average = aggregate_cumulative / (representative_tokens + 1)
        metadata["aggregateCumulativeLogprob"] = aggregate_cumulative
        metadata["pathProbabilityMassAggregated"] = True
        return replace(
            representative,
            acoustic=aggregate_average,
            avg_logprob=aggregate_average,
            metadata=metadata,
        )
    metadata["pathProbabilityMassAggregated"] = False
    return replace(representative, metadata=metadata)


def aggregate_surface_candidates(
    candidates: Iterable[CandidateEvidence],
    *,
    id_prefix: str = "surface",
) -> list[CandidateEvidence]:
    """Collapse equivalent surface strings without discarding decoder path mass.

    Path likelihoods are summed only inside an identical score domain. Scores
    from different models, spans, prompts, or decode namespaces are never added
    as if they came from one normalized distribution.
    """

    rows = list(candidates)
    if not rows:
        return []
    grouped: dict[str, list[CandidateEvidence]] = defaultdict(list)
    for candidate in rows:
        grouped[surface_key(candidate.text)].append(candidate)

    output: list[CandidateEvidence] = []
    for output_index, key in enumerate(sorted(grouped), 1):
        surface_rows = grouped[key]
        by_domain: dict[str, list[CandidateEvidence]] = defaultdict(list)
        for row in surface_rows:
            by_domain[_score_domain(row)].append(row)
        domain_rows = [_aggregate_domain(domain) for domain in by_domain.values()]
        representative = max(domain_rows, key=lambda row: (_strength(row), row.candidate_id))
        metadata = dict(representative.metadata)
        sources = {source for row in surface_rows for source in row.source_support if source}
        metadata["sourceSupport"] = sorted(sources)
        metadata["scoreDomains"] = sorted(by_domain)
        metadata["surfacePathCount"] = sum(
            int(row.metadata.get("pathCount", 1)) for row in domain_rows
        )
        metadata["surfaceCandidateIds"] = sorted(
            {
                str(candidate_id)
                for row in domain_rows
                for candidate_id in row.metadata.get("pathCandidateIds", [row.candidate_id])
            }
        )
        cross_model = representative.cross_model
        if len(sources) >= 2:
            consensus = min(1.0, 0.62 + 0.12 * (len(sources) - 2))
            cross_model = consensus if cross_model is None else max(cross_model, consensus)
        output.append(
            replace(
                representative,
                candidate_id=f"{id_prefix}:{output_index:04d}",
                text=key,
                cross_model=cross_model,
                metadata=metadata,
            )
        )
    return sorted(output, key=lambda row: (-_strength(row)[0], row.candidate_id))


def merge_candidate_pools(
    primary: Iterable[CandidateEvidence],
    additional: Iterable[CandidateEvidence],
    *,
    id_prefix: str = "merged",
) -> list[CandidateEvidence]:
    return aggregate_surface_candidates([*primary, *additional], id_prefix=id_prefix)
