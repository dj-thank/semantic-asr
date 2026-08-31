from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import CandidateEvidence
from .learned_fusion import FusionTrainingExample, LearnedFusionResult
from .mbr import semantic_loss


def _candidate_from_row(row: Mapping[str, Any]) -> CandidateEvidence:
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


def fusion_example_from_row(
    row: Mapping[str, Any], *, line_number: int = 0
) -> FusionTrainingExample:
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"fusion row {line_number} has no candidates array")
    candidates = tuple(_candidate_from_row(dict(value)) for value in raw_candidates)
    raw_target = row.get("targetDistribution") or row.get("target_distribution")
    if isinstance(raw_target, Mapping):
        target = {str(key): float(value) for key, value in raw_target.items()}
    elif isinstance(row.get("reference"), str):
        reference = CandidateEvidence("__reference__", str(row["reference"]))
        losses = {
            candidate.candidate_id: semantic_loss(candidate, reference)[0]
            for candidate in candidates
        }
        minimum = min(losses.values())
        logits = {
            candidate_id: math.exp(-8.0 * max(0.0, loss - minimum))
            for candidate_id, loss in losses.items()
        }
        total = sum(logits.values()) or 1.0
        target = {
            candidate_id: value / total for candidate_id, value in logits.items()
        }
    else:
        raise ValueError(
            f"fusion row {line_number} requires targetDistribution or reference"
        )
    return FusionTrainingExample(
        example_id=str(row.get("exampleId") or row.get("example_id") or line_number),
        group_id=str(row.get("groupId") or row.get("group_id") or ""),
        candidates=candidates,
        target_distribution=target,
        split=str(row.get("split") or "train"),
    )


def load_fusion_examples(path: str | Path) -> list[FusionTrainingExample]:
    output: list[FusionTrainingExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"fusion row {line_number} must be an object")
        output.append(fusion_example_from_row(payload, line_number=line_number))
    if not output:
        raise ValueError("fusion training dataset is empty")
    return output


def write_learned_fusion_result(
    result: LearnedFusionResult, path: str | Path
) -> None:
    payload = {
        "schemaVersion": "learned-fusion-v1",
        "profile": {
            **asdict(result.profile),
            "digest": result.profile.digest,
        },
        "before": asdict(result.before),
        "after": asdict(result.after),
        "epochLosses": list(result.epoch_losses),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
