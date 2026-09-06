"""Retrospective paired audit of the full frozen two-wave public-speech decision bundle.

No models, reference-conditioned inference, fitting or promotion. Input SHA-256
is mandatory; previously observed test rows remain exposed regression data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import unicodedata
from dataclasses import asdict
from pathlib import Path

from semantic_asr import evaluation, experiment
from semantic_asr.evaluation import (
    edit_distance,
    normalize_characters,
    normalize_characters_lenient,
)
from semantic_asr.experiment import (
    PairedErrorCounts,
    SampleResult,
    paired_bootstrap_comparison,
    paired_error_rate_comparison,
)
from semantic_asr.outputs import atomic_write
from semantic_asr.phonetic_refinement import FrozenPhoneContextPolicy, PhoneContextCandidate


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(path: Path, *, expected_sha256: str, policies: Path, iterations: int = 2000) -> dict:
    raw = path.read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("frozen input SHA-256 mismatch")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not items:
        raise ValueError("empty decision cohort")
    cached_policies = {}
    policy_hashes = {}
    grouped = {}
    seen = set()
    identities = set()
    for item in items:
        wave, split, sample = item["wave"], item["split"], item["sample_id"]
        if type(wave) is not int or wave not in (1, 2) or split not in {"validation", "test"}:
            raise ValueError("unrecognized frozen experiment wave or split")
        key = (wave, split, sample)
        if key in seen:
            raise ValueError("duplicate experiment sample")
        seen.add(key)
        identities.add(item["source_audio_sha256"])
        if wave not in cached_policies:
            data = (policies / f"wave{wave}-policy.json").read_bytes()
            frozen = json.loads(data)
            policy = FrozenPhoneContextPolicy(**frozen["policy"])
            if policy.digest != frozen["policy_digest"]:
                raise ValueError("frozen policy digest mismatch")
            cached_policies[wave] = policy
            policy_hashes[str(wave)] = _sha(data)
        policy = cached_policies[wave]
        candidates = [
            PhoneContextCandidate(
                c["id"],
                c["text"],
                tuple(c["phones"].split()),
                c["phone_score"],
                c["language_score"],
                policy.profile_digest,
                item["source_audio_sha256"],
                item["posterior_digest"],
                _sha(c["text"].encode("utf-8")),
            )
            for c in item["candidates"]
        ]
        decision = policy.select(candidates, baseline_id=item["baseline_id"])
        if decision.selected_id != item["expected_selected_id"]:
            raise ValueError("stored decision no longer reproduces")
        baseline = next(c.text for c in candidates if c.candidate_id == item["baseline_id"])
        # Only now do references enter evaluation. Selection receives candidates only.
        for name, normalize in (
            ("strict_cer", normalize_characters),
            ("lenient_cer", normalize_characters_lenient),
        ):
            reference = normalize(item["reference"])
            base_errors = edit_distance(reference, normalize(baseline))
            selected_errors = edit_distance(reference, normalize(decision.text))
            if name == "lenient_cer" and selected_errors != item["expected_errors"]:
                raise ValueError("stored error count no longer reproduces")
            if not reference:
                raise ValueError("this historical audit requires non-empty references")
            counts = PairedErrorCounts(
                sample, str(item["source_id"]), len(reference), base_errors, selected_errors
            )
            grouped.setdefault((wave, split, name), []).append((counts, decision.changed))

    reports = []
    for (wave, split, metric), rows in sorted(grouped.items()):
        counts = [row for row, _ in rows]
        expected = sorted(row.sample_id for row in counts)
        corpus = paired_error_rate_comparison(
            counts,
            baseline_system="frozen-baseline",
            candidate_system="frozen-selection",
            metric=metric,
            iterations=iterations,
            seed=17,
            expected_sample_ids=expected,
        )
        scalar = [
            SampleResult(row.sample_id, system, {metric: errors / row.reference_units}, 0.0)
            for row in counts
            for system, errors in (
                ("frozen-baseline", row.baseline_errors),
                ("frozen-selection", row.candidate_errors),
            )
        ]
        macro = paired_bootstrap_comparison(
            scalar,
            baseline_system="frozen-baseline",
            candidate_system="frozen-selection",
            metric=metric,
            iterations=iterations,
            seed=17,
            group_ids={row.sample_id: row.group_id for row in counts},
            expected_sample_ids=expected,
        )
        exact = sum(row.baseline_errors == 0 for row in counts)
        false_corrections = sum(
            row.baseline_errors == 0 and row.candidate_errors > 0 for row in counts
        )
        reports.append(
            {
                "wave": wave,
                "original_split": split,
                "role": "exposed-regression",
                "metric": metric,
                "samples": len(counts),
                "reference_units": sum(row.reference_units for row in counts),
                "baseline_errors": sum(row.baseline_errors for row in counts),
                "candidate_errors": sum(row.candidate_errors for row in counts),
                "improved": sum(row.candidate_errors < row.baseline_errors for row in counts),
                "harmed": sum(row.candidate_errors > row.baseline_errors for row in counts),
                "tied": sum(row.candidate_errors == row.baseline_errors for row in counts),
                "changed_text": sum(changed for _, changed in rows),
                "baseline_exact": exact,
                "false_corrections": false_corrections,
                "false_correction_rate": None if not exact else false_corrections / exact,
                "corpus_comparison": asdict(corpus),
                "utterance_mean_comparison": asdict(macro),
            }
        )
    return {
        "schema": "semantic-asr-retrospective-paired-audit-v1",
        "input_sha256": expected_sha256,
        "policy_file_sha256": policy_hashes,
        "decision_count": len(items),
        "unique_source_audio_hashes": len(identities),
        "iterations": iterations,
        "seed": 17,
        "confidence": 0.95,
        "resampling_assumption": "clip groups; speaker/session independence NOT verified",
        "bootstrap_fraction_note": "not a posterior correctness probability or a p-value",
        "prediction_changes_from_replay": 0,
        "new_model_inference": False,
        "new_weight_training": False,
        "promotion": "not-evaluated",
        "fresh_publication_test": False,
        "missing_promotion_evidence": [
            "unseen cohort",
            "gold semantic/phonetic labels",
            "latency",
            "speaker/session independence",
        ],
        "normalization": (
            "existing NFKC/no-whitespace strict; lenient also removes punctuation/symbols"
        ),
        "code_sha256": {
            name: _sha(Path(module.__file__).read_bytes())
            for name, module in (("evaluation", evaluation), ("experiment", experiment))
        },
        "script_sha256": _sha(Path(__file__).read_bytes()),
        "python": platform.python_version(),
        "unicode": unicodedata.unidata_version,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--policies", type=Path, default=Path("research/phonetic-20260905"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.decisions, expected_sha256=args.sha256, policies=args.policies)
    atomic_write(
        args.output, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(f"replayed and paired-audited {report['decision_count']} exposed decisions; no promotion")


if __name__ == "__main__":
    main()
