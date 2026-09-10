"""Select from four captured hypotheses; no model execution or default promotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from run_real_audio_pipeline import ensure_safe_output_dir

from semantic_asr.contracts import CandidateEvidence, sha256_json
from semantic_asr.frozen_consensus import POLICY_ID, ROLES, select_frozen_consensus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-local-research", action="store_true")
    args = parser.parse_args()
    if not args.allow_local_research:
        parser.error("captured transcript export requires --allow-local-research")
    out = ensure_safe_output_dir(args.output_dir)
    if out.exists():
        parser.error("output exists; choose a new directory")
    if args.input.stat().st_size > 1024 * 1024:
        parser.error("one-window input exceeds 1 MiB")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(ROLES):
        parser.error("input must contain only the four engine roles")
    allowed = {field.name for field in fields(CandidateEvidence)}
    for role in ROLES:
        if not isinstance(raw[role], dict) or set(raw[role]) - allowed:
            parser.error("candidate fields must follow CandidateEvidence; no reference channel")
    candidates = {role: CandidateEvidence.from_dict(raw[role]) for role in ROLES}
    selected = select_frozen_consensus(candidates)
    # This is a capture/selection receipt, not a RankedCandidate or accepted observation.
    receipt = {
        "policy": POLICY_ID,
        "decision": "provisional",
        "newInference": False,
        "promotion": False,
        "selectedCandidateId": selected.candidate_id,
        "selected": selected.as_dict(),
        "candidates": {role: candidate.as_dict() for role, candidate in candidates.items()},
        "inputSha256": sha256_json(raw),
    }
    receipt["evidenceSha256"] = sha256_json(receipt)
    out.mkdir(parents=True, exist_ok=False)
    (out / "selection.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "selected.txt").write_text(selected.text + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "evidenceSha256": receipt["evidenceSha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
