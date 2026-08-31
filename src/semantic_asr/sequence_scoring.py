"""Correct full-sequence causal-LM scoring for ASR candidates.

Every candidate token contributes its assigned next-token log probability. The
implementation does not use the maximum vocabulary probability at the final
step, and it never relabels an uncalibrated likelihood as a correctness
probability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .score_semantics import EvidenceScore, ScoreKind


@dataclass(frozen=True, slots=True)
class SequenceScore:
    candidate_id: str
    sum_logprob: float
    average_logprob: float
    token_count: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source:
            raise ValueError("candidate_id and source are required")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        if any(
            not math.isfinite(float(value))
            for value in (self.sum_logprob, self.average_logprob)
        ):
            raise ValueError("sequence log probabilities must be finite")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_evidence(self, *, average: bool = True) -> EvidenceScore:
        return EvidenceScore(
            value=self.average_logprob if average else self.sum_logprob,
            kind=ScoreKind.LOG_LIKELIHOOD,
            source=self.source,
            calibrated=False,
            calibration_digest=None,
            metadata={
                **dict(self.metadata),
                "candidateId": self.candidate_id,
                "tokenCount": self.token_count,
                "normalization": "per-token" if average else "sum",
            },
        )


def sequence_logprob_from_logits(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    ignore_index: int = -100,
    candidate_id: str = "candidate",
    source: str = "causal-lm",
) -> SequenceScore:
    """Score aligned next-token labels using a stable log-softmax."""

    if len(logits) != len(labels) or not logits:
        raise ValueError("logits and labels must have equal non-zero length")
    total = 0.0
    count = 0
    for row_values, label in zip(logits, labels, strict=True):
        if label == ignore_index:
            continue
        row = [float(value) for value in row_values]
        if not row or any(not math.isfinite(value) for value in row):
            raise ValueError("every logit row must contain finite values")
        if label < 0 or label >= len(row):
            raise ValueError("label is outside the vocabulary")
        maximum = max(row)
        log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in row))
        total += row[label] - log_normalizer
        count += 1
    if count < 1:
        raise ValueError("at least one candidate token must be scored")
    return SequenceScore(
        candidate_id=candidate_id,
        sum_logprob=total,
        average_logprob=total / count,
        token_count=count,
        source=source,
    )


def sequence_scores_to_preferences(
    scores: Sequence[SequenceScore],
    *,
    use_average: bool = True,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Convert comparable likelihoods to relative mass, not correctness probability."""

    if not scores:
        raise ValueError("sequence scores are required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    identifiers = [score.candidate_id for score in scores]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    values = {
        score.candidate_id: (
            score.average_logprob if use_average else score.sum_logprob
        )
        for score in scores
    }
    maximum = max(values.values())
    mass = {
        candidate_id: math.exp(
            max(-80.0, min(80.0, (value - maximum) / temperature))
        )
        for candidate_id, value in values.items()
    }
    total = sum(mass.values()) or 1.0
    return {candidate_id: value / total for candidate_id, value in mass.items()}


class TransformersCausalSequenceScorer:
    """Lazy scorer for a caller-supplied Hugging Face causal LM and tokenizer."""

    def __init__(self, model: Any, tokenizer: Any, *, source: str) -> None:
        if not source:
            raise ValueError("source is required")
        self.model = model
        self.tokenizer = tokenizer
        self.source = source

    def score(self, candidate_id: str, text: str, *, prefix: str = "") -> SequenceScore:
        if not text:
            raise ValueError("candidate text must not be empty")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers sequence scoring requires PyTorch") from exc

        prefix_ids = list(self.tokenizer.encode(prefix, add_special_tokens=True))
        candidate_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if not candidate_ids:
            raise ValueError("tokenizer produced no candidate tokens")
        if not prefix_ids:
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if bos is None:
                bos = getattr(getattr(self.model, "config", None), "bos_token_id", None)
            if bos is None:
                raise ValueError("a BOS token or non-empty prefix is required")
            prefix_ids = [int(bos)]
        input_ids = torch.tensor([prefix_ids + candidate_ids], dtype=torch.long)
        device = getattr(self.model, "device", None)
        if device is not None:
            input_ids = input_ids.to(device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
        logits = output.logits[0]
        start = len(prefix_ids) - 1
        candidate_logits = logits[start : start + len(candidate_ids)]
        log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
        labels = torch.tensor(candidate_ids, dtype=torch.long, device=log_probs.device)
        token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        total = float(token_logprobs.sum().item())
        return SequenceScore(
            candidate_id=candidate_id,
            sum_logprob=total,
            average_logprob=total / len(candidate_ids),
            token_count=len(candidate_ids),
            source=self.source,
            metadata={"prefixTokens": len(prefix_ids)},
        )
