"""Attach evidence the acoustic score does not already contain to candidate manifests.

Measured on 2026-09-02 (docs/RESEARCH_2026-09-02.md), the v0.2 linear rankers never
disagreed with the ASR rank because every feature was a function of the same decoder score.
This module adds two independent evidence streams to an existing candidate JSONL:

* ``cross_model`` — agreement between each candidate and an independent second-ear ASR
  hypothesis (for example Qwen3-ASR). Agreement is one minus the punctuation-insensitive
  character error rate between the two strings, clipped to ``[0, 1]``.
* ``lexical`` — a character/mora n-gram language-model score, min-max normalised inside the
  candidate set so the best-scoring surface receives 1.0. The raw average log-probability is
  kept in metadata.

Optionally the second-ear hypothesis itself is appended as a candidate. It is acoustically
grounded (it is an ASR output), so it may enter the observed-eligible pool, but it carries no
Whisper decoder score and is tagged with its own source and score domain.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .contracts import CandidateEvidence
from .evaluation import lenient_cer
from .ngram import NGramLanguageModel


def agreement(left: str, right: str) -> float:
    """Punctuation-insensitive string agreement in ``[0, 1]``."""

    value = lenient_cer(left, right)
    if value is None:
        return 0.0
    return max(0.0, 1.0 - min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class SecondEarHypothesis:
    sample_id: str
    texts: tuple[str, ...]
    source: str = "second-ear"
    seconds: float | None = None


def load_second_ear(
    path: str | Path, *, source: str = "second-ear"
) -> dict[str, SecondEarHypothesis]:
    """Load ``scripts/probe_second_ear.py`` output (one JSON object per line)."""

    output: dict[str, SecondEarHypothesis] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        sample_id = row.get("sampleId")
        if not sample_id:
            continue
        texts = tuple(str(text).strip() for text in row.get("hypotheses", []) if str(text).strip())
        output[str(sample_id)] = SecondEarHypothesis(
            sample_id=str(sample_id),
            texts=texts,
            source=source,
            seconds=row.get("seconds"),
        )
    return output


def _normalise_within_set(values: Sequence[float | None]) -> list[float | None]:
    present = [value for value in values if value is not None]
    if not present:
        return list(values)
    low, high = min(present), max(present)
    if high - low < 1e-12:
        return [None if value is None else 1.0 for value in values]
    return [None if value is None else (value - low) / (high - low) for value in values]


@dataclass(frozen=True, slots=True)
class EnrichmentConfig:
    add_second_ear_candidate: bool = False
    second_ear_source: str = "qwen3-asr"
    ngram_model: NGramLanguageModel | None = None
    ngram_name: str = "ngram"


def enrich_candidates(
    candidates: Sequence[CandidateEvidence],
    *,
    second_ear: SecondEarHypothesis | None,
    config: EnrichmentConfig,
) -> list[CandidateEvidence]:
    rows = list(candidates)
    second_text = second_ear.texts[0] if second_ear and second_ear.texts else None

    ngram_values: list[float | None] = []
    if config.ngram_model is not None:
        for candidate in rows:
            try:
                ngram_values.append(
                    config.ngram_model.score(candidate.text).average_log_probability
                )
            except ValueError:
                ngram_values.append(None)
    else:
        ngram_values = [None] * len(rows)
    lexical_values = _normalise_within_set(ngram_values)

    enriched: list[CandidateEvidence] = []
    for candidate, raw_ngram, lexical in zip(rows, ngram_values, lexical_values, strict=True):
        metadata: dict[str, Any] = dict(candidate.metadata)
        cross_model = candidate.cross_model
        if second_text is not None:
            score = agreement(candidate.text, second_text)
            metadata.update(
                {
                    "secondEarSource": config.second_ear_source,
                    "secondEarText": second_text,
                    "secondEarAgreement": score,
                }
            )
            cross_model = score
        if raw_ngram is not None:
            metadata.update(
                {
                    "ngramModel": config.ngram_name,
                    "ngramAverageLogProbability": raw_ngram,
                }
            )
        enriched.append(
            replace(
                candidate,
                cross_model=cross_model,
                lexical=lexical if raw_ngram is not None else candidate.lexical,
                metadata=metadata,
            )
        )

    if config.add_second_ear_candidate and second_text is not None:
        existing = {candidate.text for candidate in enriched}
        if second_text not in existing:
            ngram_value = None
            if config.ngram_model is not None:
                try:
                    ngram_value = config.ngram_model.score(second_text).average_log_probability
                except ValueError:
                    ngram_value = None
            enriched.append(
                CandidateEvidence(
                    candidate_id=f"{config.second_ear_source}:0001",
                    text=second_text,
                    cross_model=1.0,
                    rank=len(enriched) + 1,
                    hypothesis_count=len(enriched) + 1,
                    source=config.second_ear_source,
                    metadata={
                        "adapter": config.second_ear_source,
                        "scoreDomain": f"{config.second_ear_source}|second-ear",
                        "secondEarSource": config.second_ear_source,
                        "secondEarText": second_text,
                        "secondEarAgreement": 1.0,
                        "secondEarCandidate": True,
                        "ngramAverageLogProbability": ngram_value,
                    },
                )
            )
            enriched = [
                replace(candidate, hypothesis_count=len(enriched)) for candidate in enriched
            ]
    return enriched


def enrich_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    second_ear: Mapping[str, SecondEarHypothesis],
    config: EnrichmentConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        candidates = [CandidateEvidence.from_dict(dict(value)) for value in payload["candidates"]]
        enriched = enrich_candidates(
            candidates,
            second_ear=second_ear.get(str(payload.get("sampleId"))),
            config=config,
        )
        payload["candidates"] = [candidate.as_dict() for candidate in enriched]
        payload["enrichment"] = {
            "secondEar": config.second_ear_source if second_ear else None,
            "secondEarCandidateAdded": bool(
                config.add_second_ear_candidate
                and any(candidate.metadata.get("secondEarCandidate") for candidate in enriched)
            ),
            "ngram": config.ngram_name if config.ngram_model is not None else None,
        }
        output.append(payload)
    return output
