from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import CandidateEvidence
from .mbr import semantic_loss
from .ranker_training import RankerExample


def _candidate(row: Mapping[str, Any]) -> CandidateEvidence:
    aliases = {
        "candidateId": "candidate_id",
        "tokenIds": "token_ids",
        "crossModel": "cross_model",
        "moraUnits": "mora_units",
        "hypothesisCount": "hypothesis_count",
        "sequenceScore": "sequence_score",
        "avgLogprob": "avg_logprob",
        "beamConfidence": "beam_confidence",
    }
    return CandidateEvidence.from_dict(
        {aliases.get(str(key), str(key)): value for key, value in row.items()}
    )


def ranker_example_from_row(
    row: Mapping[str, Any],
    *,
    line_number: int = 0,
    require_train_split: bool = True,
) -> RankerExample:
    split = str(row.get("split") or "train")
    if require_train_split and split != "train":
        raise ValueError(
            f"ranker training row {line_number} belongs to forbidden split {split!r}"
        )
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"ranker row {line_number} has no candidates array")
    candidates = tuple(_candidate(dict(value)) for value in raw_candidates)
    if len(candidates) < 2:
        raise ValueError("ranker training requires at least two candidates")
    identifiers = {candidate.candidate_id for candidate in candidates}
    if len(identifiers) != len(candidates):
        raise ValueError("ranker candidate IDs must be unique")

    raw_losses = row.get("losses")
    if isinstance(raw_losses, Mapping):
        losses = {str(key): float(value) for key, value in raw_losses.items()}
        if set(losses) != identifiers:
            raise ValueError("ranker losses must contain every candidate ID exactly once")
        if any(not math.isfinite(value) or value < 0 for value in losses.values()):
            raise ValueError("ranker losses must be finite and non-negative")
    else:
        reference_text = str(row.get("reference") or "").strip()
        if not reference_text:
            raise ValueError(
                f"ranker row {line_number} requires losses or a reference transcript"
            )
        reference = CandidateEvidence(
            candidate_id=f"reference:{line_number}",
            text=reference_text,
            reading=(str(row["referenceReading"]) if row.get("referenceReading") else None),
        )
        losses = {
            candidate.candidate_id: float(semantic_loss(candidate, reference)[0])
            for candidate in candidates
        }

    return RankerExample(
        example_id=str(row.get("exampleId") or row.get("example_id") or line_number),
        candidates=candidates,
        losses=losses,
        context=str(row.get("context") or ""),
    )


def load_ranker_examples(
    path: str | Path,
    *,
    require_train_split: bool = True,
) -> list[RankerExample]:
    output: list[RankerExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"ranker row {line_number} must be an object")
        output.append(
            ranker_example_from_row(
                payload,
                line_number=line_number,
                require_train_split=require_train_split,
            )
        )
    if not output:
        raise ValueError("ranker training dataset is empty")
    return output
