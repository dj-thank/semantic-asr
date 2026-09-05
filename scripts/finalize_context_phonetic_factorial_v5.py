#!/usr/bin/env python3
"""One-shot idempotent hardening migration for context × phonetic factorial tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def save(path: str, text: str) -> None:
    (ROOT / path).write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_protocol() -> None:
    path = "src/semantic_asr/context_phonetic_experiment/protocol.py"
    text = load(path)
    if "require_different_context_group_for_shuffle" not in text:
        text = text.replace(
            "    require_different_source_for_shuffle: bool = True\n",
            "    require_different_source_for_shuffle: bool = True\n"
            "    require_different_context_group_for_shuffle: bool = True\n",
            1,
        )
        text = text.replace(
            '            "require_different_source_for_shuffle",\n',
            '            "require_different_source_for_shuffle",\n'
            '            "require_different_context_group_for_shuffle",\n',
            1,
        )
        text = text.replace(
            '                "requireDifferentSourceForShuffle": (\n'
            "                    self.require_different_source_for_shuffle\n"
            "                ),\n",
            '                "requireDifferentSourceForShuffle": (\n'
            "                    self.require_different_source_for_shuffle\n"
            "                ),\n"
            '                "requireDifferentContextGroupForShuffle": (\n'
            "                    self.require_different_context_group_for_shuffle\n"
            "                ),\n",
            1,
        )
    save(path, text)


def patch_planner() -> None:
    path = "src/semantic_asr/context_phonetic_experiment/planner.py"
    text = load(path)
    text = text.replace(
        "        if latency < 0.0:\n"
        '            raise ValueError("scoring_latency_ms must be non-negative")\n',
        "        if not math.isfinite(latency) or latency < 0.0:\n"
        "            raise ValueError(\n"
        '                "scoring_latency_ms must be finite and non-negative"\n'
        "            )\n",
    )
    if "import math\n" not in text:
        text = text.replace("import hashlib\n", "import hashlib\nimport math\n", 1)
    if "require_different_context_group_for_shuffle" not in text:
        text = text.replace(
            "    if protocol.require_different_source_for_shuffle and left.source_id == right.source_id:\n"
            "        return False\n",
            "    if protocol.require_different_source_for_shuffle and left.source_id == right.source_id:\n"
            "        return False\n"
            "    if (\n"
            "        protocol.require_different_context_group_for_shuffle\n"
            "        and receiver.context_group_id == donor.context_group_id\n"
            "    ):\n"
            "        return False\n",
            1,
        )
    pattern = re.compile(
        r"def deterministic_context_derangement\(\n"
        r"    manifest: ContextPhoneticManifest,\n"
        r"    protocol: ContextPhoneticProtocol,\n"
        r"\) -> dict\[str, ContextPhoneticCase\]:\n"
        r".*?\n\ndef _context_candidates",
        re.DOTALL,
    )
    replacement = '''def deterministic_context_derangement(
    manifest: ContextPhoneticManifest,
    protocol: ContextPhoneticProtocol,
) -> dict[str, ContextPhoneticCase]:
    """Find a deterministic perfect donor matching under all registered exclusions."""

    receivers = tuple(
        sorted(
            manifest.cases,
            key=lambda case: hashlib.sha256(
                f"{protocol.shuffle_seed}:receiver:{case.case_id}".encode("utf-8")
            ).hexdigest(),
        )
    )
    if len(receivers) < 2:
        raise ValueError("shuffled-context control requires at least two cases")
    donors_by_receiver = {
        receiver.case_id: tuple(
            sorted(
                (
                    donor
                    for donor in manifest.cases
                    if _shuffle_compatible(receiver, donor, protocol)
                ),
                key=lambda donor: hashlib.sha256(
                    (
                        f"{protocol.shuffle_seed}:donor:{receiver.case_id}:"
                        f"{donor.case_id}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
        )
        for receiver in receivers
    }
    if any(not donors for donors in donors_by_receiver.values()):
        raise ValueError(
            "no deterministic context derangement satisfies the registered "
            "speaker/session/source/context-group exclusions"
        )
    assigned_receiver_by_donor: dict[str, str] = {}
    assigned_donor_by_receiver: dict[str, ContextPhoneticCase] = {}

    def augment(receiver: ContextPhoneticCase, visited: set[str]) -> bool:
        for donor in donors_by_receiver[receiver.case_id]:
            if donor.case_id in visited:
                continue
            visited.add(donor.case_id)
            previous_receiver_id = assigned_receiver_by_donor.get(donor.case_id)
            if previous_receiver_id is None:
                assigned_receiver_by_donor[donor.case_id] = receiver.case_id
                assigned_donor_by_receiver[receiver.case_id] = donor
                return True
            previous_receiver = next(
                row for row in receivers if row.case_id == previous_receiver_id
            )
            if augment(previous_receiver, visited):
                assigned_receiver_by_donor[donor.case_id] = receiver.case_id
                assigned_donor_by_receiver[receiver.case_id] = donor
                return True
        return False

    for receiver in receivers:
        if not augment(receiver, set()):
            raise ValueError(
                "no deterministic context derangement satisfies the registered "
                "speaker/session/source/context-group exclusions"
            )
    if len(assigned_donor_by_receiver) != len(receivers):
        raise ValueError("context derangement did not assign every receiver")
    return assigned_donor_by_receiver


def _context_candidates'''
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1 and "Find a deterministic perfect donor matching" not in text:
        raise RuntimeError("could not replace deterministic context derangement")
    save(path, text)


def patch_context_scorer() -> None:
    path = "src/semantic_asr/context_phonetic_experiment/context_scorer.py"
    text = load(path)
    if "path_digest," not in text:
        text = text.replace(
            "from ..deliberation_lattice import DocumentContext, LatticeArc\n",
            "from ..deliberation_lattice import DocumentContext, LatticeArc, path_digest\n",
            1,
        )
    manual = """            if row.path_digest != sha256_json(
                [
                    {
                        "arcId": path[0].arc_id,
                        "spanId": path[0].span_id,
                        "text": path[0].text,
                        "arcDigest": path[0].digest,
                    }
                ]
            ):
"""
    if manual in text:
        text = text.replace(
            manual,
            "            if row.path_digest != path_digest(path):\n",
            1,
        )
    save(path, text)


def patch_metrics() -> None:
    path = "src/semantic_asr/context_phonetic_experiment/metrics.py"
    text = load(path)
    paired_start = text.find("\ndef _paired_groups(\n")
    paired_end = text.find("\ndef _grouped_bootstrap(\n")
    if paired_start >= 0 and paired_end > paired_start:
        text = text[:paired_start] + text[paired_end:]
    if (
        "class ContextPhoneticArmAggregate" in text
        and "def __post_init__(self)"
        not in text[
            text.find("class ContextPhoneticArmAggregate") : text.find(
                "class GroupedPairedContrast"
            )
        ]
    ):
        marker = "    mean_selection_latency_ms: float\n\n"
        addition = """    def __post_init__(self) -> None:
        if not self.arm_name or not self.phonetic_arm_name:
            raise ValueError("factorial aggregate requires arm identities")
        if self.context_condition not in {"none", "ordered", "shuffled"}:
            raise ValueError("factorial aggregate context condition is invalid")
        count_fields = (
            "case_count",
            "exact_count",
            "proposed_exact_count",
            "first_pass_exact_count",
            "oracle_count",
            "outside_first_pass_case_count",
            "outside_first_pass_recovery_count",
            "false_correction_count",
            "corrected_first_pass_count",
            "critical_case_count",
            "critical_exact_count",
            "accepted_count",
            "changed_effective_count",
            "total_reference_characters",
            "total_first_pass_edits",
            "total_effective_edits",
            "total_introduced_error_characters",
            "total_corrected_error_characters",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.case_count < 1 or self.total_reference_characters < 1:
            raise ValueError("factorial aggregate requires cases and reference characters")
        bounded_counts = (
            self.exact_count,
            self.proposed_exact_count,
            self.first_pass_exact_count,
            self.oracle_count,
            self.false_correction_count,
            self.corrected_first_pass_count,
            self.critical_case_count,
            self.critical_exact_count,
            self.accepted_count,
            self.changed_effective_count,
        )
        if any(value > self.case_count for value in bounded_counts):
            raise ValueError("factorial aggregate count exceeds case_count")
        if self.outside_first_pass_recovery_count > self.outside_first_pass_case_count:
            raise ValueError("outside-first-pass recovery exceeds eligible cases")
        if self.false_correction_count > self.first_pass_exact_count:
            raise ValueError("false corrections exceed first-pass exact cases")
        if self.critical_exact_count > self.critical_case_count:
            raise ValueError("critical exact count exceeds critical cases")
        for name in (
            "mean_margin",
            "mean_pool_generation_latency_ms",
            "mean_context_scoring_latency_ms",
            "mean_selection_latency_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

"""
        if marker not in text:
            raise RuntimeError("aggregate validation insertion marker missing")
        text = text.replace(marker, marker + addition, 1)
    percentile = """    return point, values[lower_index], values[upper_index], len(group_ids)
"""
    replacement = """    lower = min(point, values[lower_index])
    upper = max(point, values[upper_index])
    return point, lower, upper, len(group_ids)
"""
    text = text.replace(percentile, replacement)
    save(path, text)


def patch_protocol_tests() -> None:
    path = "tests/test_context_phonetic_protocol.py"
    text = load(path)
    if "require_different_context_group_for_shuffle" not in text:
        text = text.replace(
            '    assert protocol.arm("phone+mora:shuffled").phonetic_arm_name == "phone+mora"\n',
            '    assert protocol.arm("phone+mora:shuffled").phonetic_arm_name == "phone+mora"\n'
            "    assert protocol.require_different_context_group_for_shuffle\n",
            1,
        )
    save(path, text)


def main() -> int:
    patch_protocol()
    patch_planner()
    patch_context_scorer()
    patch_metrics()
    patch_protocol_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
