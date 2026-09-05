"""Replay stored scalar model evidence, without models, reference-conditioned inference or refit.

The eight checked fixtures include every changed held-out selection and two controls.
They are regression fixtures, NOT a replacement for the complete two-wave evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from semantic_asr.candidate_pool import lenient_surface_key
from semantic_asr.evaluation import edit_distance
from semantic_asr.phonetic_refinement import FrozenPhoneContextPolicy, PhoneContextCandidate


def replay(root: Path) -> int:
    items = json.loads((root / "decision-fixtures.json").read_text(encoding="utf-8"))
    for item in items:
        frozen = json.loads((root / f"wave{item['wave']}-policy.json").read_text(encoding="utf-8"))
        policy = FrozenPhoneContextPolicy(**frozen["policy"])
        if policy.digest != frozen["policy_digest"]:
            raise ValueError("frozen policy digest mismatch")
        candidates = [
            PhoneContextCandidate(
                row["id"],
                row["text"],
                tuple(row["phones"].split()),
                row["phone_score"],
                row["language_score"],
                policy.profile_digest,
                item["source_audio_sha256"],
                item["posterior_digest"],
                hashlib.sha256(row["text"].encode()).hexdigest(),
            )
            for row in item["candidates"]
        ]
        decision = policy.select(candidates, baseline_id=item["baseline_id"])
        if decision.selected_id != item["expected_selected_id"]:
            raise ValueError(f"selection regression: wave{item['wave']}/{item['sample_id']}")
        # Reference enters only this post-selection evaluation step.
        errors = edit_distance(
            lenient_surface_key(item["reference"]), lenient_surface_key(decision.text)
        )
        if errors != item["expected_errors"]:
            raise ValueError("reference/evaluation normalization regression")
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("research/phonetic-20260905"))
    args = parser.parse_args()
    print(f"replayed {replay(args.study)} stored-evidence regression decisions")


if __name__ == "__main__":
    main()
