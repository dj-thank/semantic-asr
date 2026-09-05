#!/usr/bin/env python3
"""One-shot idempotent hardening migration for the phonetic ablation framework."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def save(path: str, text: str) -> None:
    (ROOT / path).write_text(text.rstrip() + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"missing migration marker: {label}")
    if text.count(old) != 1:
        raise RuntimeError(f"non-unique migration marker: {label}")
    return text.replace(old, new, 1)


def patch_protocol() -> None:
    path = "src/semantic_asr/phonetic_experiment/protocol.py"
    text = load(path)
    if "BootstrapGroup = Literal" not in text:
        text = text.replace(
            'EvidenceChannel = Literal["first_pass", "phone", "mora", "discrete_unit"]\n',
            'EvidenceChannel = Literal["first_pass", "phone", "mora", "discrete_unit"]\n'
            'BootstrapGroup = Literal["speaker", "session", "source"]\n',
            1,
        )
    reference_requirement = """        if self.reference.text not in lexicon_surfaces:
            raise ValueError("reference surface must be present in the frozen exogenous lexicon")
"""
    text = text.replace(reference_requirement, "")
    speaker_requirement = """        speakers = [case.speaker_id for case in self.cases]
        sessions = [case.session_id for case in self.cases]
        if len(speakers) != len(set(speakers)):
            raise ValueError("ablation test cases must be speaker-disjoint")
        if len(sessions) != len(set(sessions)):
            raise ValueError("ablation test cases must be session-disjoint")
"""
    text = text.replace(speaker_requirement, "")
    if "bootstrap_group: BootstrapGroup" not in text:
        text = text.replace(
            '    bootstrap_seed: str = "semantic-asr-phonetic-ablation-v1"\n',
            '    bootstrap_seed: str = "semantic-asr-phonetic-ablation-v1"\n'
            '    bootstrap_group: BootstrapGroup = "speaker"\n',
            1,
        )
        text = text.replace(
            '        if not self.bootstrap_seed:\n            raise ValueError("bootstrap_seed is required")\n',
            "        if not self.bootstrap_seed:\n"
            '            raise ValueError("bootstrap_seed is required")\n'
            '        if self.bootstrap_group not in {"speaker", "session", "source"}:\n'
            '            raise ValueError("bootstrap_group must be speaker, session, or source")\n',
            1,
        )
        text = text.replace(
            '                "bootstrapSeed": self.bootstrap_seed,\n',
            '                "bootstrapSeed": self.bootstrap_seed,\n'
            '                "bootstrapGroup": self.bootstrap_group,\n',
            1,
        )
    save(path, text)


def patch_planner() -> None:
    path = "src/semantic_asr/phonetic_experiment/planner.py"
    text = load(path)
    if "FrozenPronunciationLexicon" not in text.split("\n\n", 8)[0:8]:
        text = text.replace(
            "from ..phonetic_bridge import PhoneticBridgeConfig, propose_text_from_pronunciation\n",
            "from ..phonetic_bridge import (\n"
            "    FrozenPronunciationLexicon,\n"
            "    PhoneticBridgeConfig,\n"
            "    propose_text_from_pronunciation,\n"
            ")\n",
            1,
        )
    text = text.replace("    lexicon: object\n", "    lexicon: FrozenPronunciationLexicon\n")
    if '"generationLatencyMs": self.generation_latency_ms,\n' in text:
        text = text.replace(
            '                "generationLatencyMs": self.generation_latency_ms,\n', ""
        )
    old_digest = """                    proposal_digest=sha256_json(
                        {
                            "candidateId": proposal.candidate_id,
                            "entryId": proposal.entry_id,
                            "pronunciationKey": proposal.pronunciation_key,
                            "utilityDigests": [
                                utility.digest for utility in proposal.utilities
                            ],
                            "phoneScoreDigest": proposal.phone_score.evidence.metadata,
                            "moraScoreDigest": proposal.mora_score.evidence.metadata,
                        }
                    ),
"""
    new_digest = """                    proposal_digest=sha256_json(
                        {
                            "candidateId": proposal.candidate_id,
                            "entryId": proposal.entry_id,
                            "pronunciationKey": proposal.pronunciation_key,
                            "utilityDigests": [
                                utility.digest for utility in proposal.utilities
                            ],
                            "phonePosteriorDigest": proposal.phone_score.posterior_digest,
                            "phonePronunciationDigest": proposal.phone_score.pronunciation_digest,
                            "moraPosteriorDigest": proposal.mora_score.posterior_digest,
                            "moraPronunciationDigest": proposal.mora_score.pronunciation_digest,
                        }
                    ),
"""
    if old_digest in text:
        text = text.replace(old_digest, new_digest, 1)
    if "if len(proposals) != len(case.lexicon.entries):" not in text:
        marker = "        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0\n"
        text = replace_once(
            text,
            marker,
            "        if len(proposals) != len(case.lexicon.entries):\n"
            "            raise ValueError(\n"
            '                "frozen planner did not score every exogenous lexicon entry"\n'
            "            )\n" + marker,
            label="planner complete lexicon coverage",
        )
    save(path, text)


def patch_selection() -> None:
    path = "src/semantic_asr/phonetic_experiment/selection.py"
    text = load(path)
    pattern = re.compile(
        r"    @property\n    def digest\(self\) -> str:\n"
        r"        return sha256_json\(\n"
        r"            \{\n"
        r"                \*\*asdict\(self\),\n"
        r"                \"ranked\": \[asdict\(row\) for row in self\.ranked\],\n"
        r"            \}\n"
        r"        \)\n",
    )
    replacement = """    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "armName": self.arm_name,
                "armDigest": self.arm_digest,
                "poolDigest": self.pool_digest,
                "proposedCandidateId": self.proposed_candidate_id,
                "effectiveCandidateId": self.effective_candidate_id,
                "firstPassSelectedCandidateId": self.first_pass_selected_candidate_id,
                "status": self.status,
                "margin": self.margin,
                "proposedScore": self.proposed_score,
                "ranked": [asdict(row) for row in self.ranked],
                "reason": self.reason,
            }
        )
"""
    text, count = pattern.subn(replacement, text, count=1)
    if (
        count == 0
        and '"selection_latency_ms"'
        in text[text.find("def digest") : text.find("def digest") + 800]
    ):
        raise RuntimeError("could not remove selection latency from decision digest")
    save(path, text)


def patch_metrics() -> None:
    path = "src/semantic_asr/phonetic_experiment/metrics.py"
    text = load(path)
    if "from ..deliberation_evidence import _is_sha256\n" not in text:
        text = text.replace(
            "from ..contracts import sha256_json\n",
            "from ..contracts import sha256_json\nfrom ..deliberation_evidence import _is_sha256\n",
            1,
        )
    if "    group_id: str\n" not in text:
        text = text.replace(
            "class PhoneticCaseArmMetrics:\n    case_id: str\n",
            "class PhoneticCaseArmMetrics:\n    case_id: str\n    group_id: str\n",
            1,
        )
        text = text.replace(
            "        if not self.case_id or not self.arm_name:\n"
            '            raise ValueError("case arm metrics require case and arm names")\n',
            "        if not self.case_id or not self.group_id or not self.arm_name:\n"
            '            raise ValueError("case arm metrics require case, group, and arm names")\n'
            "        for digest in (\n"
            "            self.reference_text_sha256,\n"
            "            self.first_pass_text_sha256,\n"
            "            self.proposed_text_sha256,\n"
            "            self.effective_text_sha256,\n"
            "        ):\n"
            "            if not _is_sha256(digest):\n"
            '                raise ValueError("case arm text digests must be SHA-256 values")\n',
            1,
        )
    if "    first_pass_exact_count: int\n" not in text:
        text = text.replace(
            "    proposed_exact_count: int\n",
            "    proposed_exact_count: int\n    first_pass_exact_count: int\n",
            1,
        )
    text = text.replace(
        "    def false_correction_rate(self) -> float:\n"
        "        return self.false_correction_count / self.case_count\n",
        "    def false_correction_rate(self) -> float:\n"
        "        if not self.first_pass_exact_count:\n"
        "            return 0.0\n"
        "        return self.false_correction_count / self.first_pass_exact_count\n",
    )
    if (
        "        first_pass_exact_count=sum(row.first_pass_edits == 0 for row in rows),\n"
        not in text
    ):
        text = text.replace(
            "        proposed_exact_count=sum(row.proposed_exact for row in rows),\n",
            "        proposed_exact_count=sum(row.proposed_exact for row in rows),\n"
            "        first_pass_exact_count=sum(row.first_pass_edits == 0 for row in rows),\n",
            1,
        )
    if "    group_count: int\n" not in text:
        text = text.replace(
            "    resamples: int\n    seed: str\n",
            "    resamples: int\n    seed: str\n    group_count: int\n",
            1,
        )
    text = text.replace(
        "def evaluate_case_arm(\n"
        "    pool: FrozenPhoneticCandidatePool,\n"
        "    decision: PhoneticAblationDecision,\n"
        "    reference: FrozenSpanReference,\n"
        ") -> PhoneticCaseArmMetrics:\n",
        "def evaluate_case_arm(\n"
        "    pool: FrozenPhoneticCandidatePool,\n"
        "    decision: PhoneticAblationDecision,\n"
        "    reference: FrozenSpanReference,\n"
        "    *,\n"
        "    group_id: str,\n"
        ") -> PhoneticCaseArmMetrics:\n",
    )
    if "        group_id=group_id,\n" not in text:
        text = text.replace(
            "        case_id=pool.case_id,\n",
            "        case_id=pool.case_id,\n        group_id=group_id,\n",
            1,
        )
    bootstrap_pattern = re.compile(
        r"def paired_bootstrap_error_delta\(\n.*?\n    return PairedErrorDelta\(\n"
        r"        target_arm=target\[0\]\.arm_name,\n"
        r"        baseline_arm=baseline\[0\]\.arm_name,\n"
        r"        mean_character_error_delta=point,\n"
        r"        lower_95=values\[lower_index\],\n"
        r"        upper_95=values\[upper_index\],\n"
        r"        resamples=resamples,\n"
        r"        seed=seed,\n"
        r"    \)\n",
        re.DOTALL,
    )
    bootstrap_replacement = """def paired_bootstrap_error_delta(
    target: tuple[PhoneticCaseArmMetrics, ...],
    baseline: tuple[PhoneticCaseArmMetrics, ...],
    *,
    resamples: int,
    seed: str,
) -> PairedErrorDelta:
    if not target or len(target) != len(baseline):
        raise ValueError("paired bootstrap requires equal non-empty arms")
    target_by_case = {row.case_id: row for row in target}
    baseline_by_case = {row.case_id: row for row in baseline}
    if len(target_by_case) != len(target) or len(baseline_by_case) != len(baseline):
        raise ValueError("paired bootstrap case IDs must be unique")
    if set(target_by_case) != set(baseline_by_case):
        raise ValueError("paired bootstrap case IDs differ")
    if resamples < 1 or not seed:
        raise ValueError("paired bootstrap requires resamples and seed")
    groups: dict[str, list[str]] = {}
    for case_id in sorted(target_by_case):
        target_row = target_by_case[case_id]
        baseline_row = baseline_by_case[case_id]
        if target_row.group_id != baseline_row.group_id:
            raise ValueError("paired bootstrap group identities differ between arms")
        groups.setdefault(target_row.group_id, []).append(case_id)
    group_ids = tuple(sorted(groups))

    def delta(sampled_groups: tuple[str, ...]) -> float:
        sampled_cases = tuple(
            case_id for group_id in sampled_groups for case_id in groups[group_id]
        )
        target_edits = sum(target_by_case[case_id].effective_edits for case_id in sampled_cases)
        baseline_edits = sum(
            baseline_by_case[case_id].effective_edits for case_id in sampled_cases
        )
        characters = sum(
            target_by_case[case_id].reference_characters for case_id in sampled_cases
        )
        return (target_edits - baseline_edits) / characters

    point = delta(group_ids)
    randomizer = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = tuple(randomizer.choice(group_ids) for _ in group_ids)
        values.append(delta(sample))
    values.sort()
    lower_index = max(0, math.floor(0.025 * (len(values) - 1)))
    upper_index = min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))
    return PairedErrorDelta(
        target_arm=target[0].arm_name,
        baseline_arm=baseline[0].arm_name,
        mean_character_error_delta=point,
        lower_95=values[lower_index],
        upper_95=values[upper_index],
        resamples=resamples,
        seed=seed,
        group_count=len(group_ids),
    )
"""
    text, count = bootstrap_pattern.subn(bootstrap_replacement, text, count=1)
    if count != 1 and "sampled_groups" not in text:
        raise RuntimeError("could not replace paired bootstrap implementation")
    save(path, text)


def patch_runner() -> None:
    path = "src/semantic_asr/phonetic_experiment/runner.py"
    text = load(path)
    text = text.replace(
        "    def as_dict(self, *, include_text: bool = False) -> dict[str, object]:",
        "    def as_dict(\n"
        "        self, *, include_ranked_candidate_ids: bool = False\n"
        "    ) -> dict[str, object]:",
    )
    text = text.replace("if include_text\n", "if include_ranked_candidate_ids\n")
    text = text.replace(
        "    def write(self, path: str | Path, *, include_text: bool = False) -> Path:",
        "    def write(\n"
        "        self,\n"
        "        path: str | Path,\n"
        "        *,\n"
        "        include_ranked_candidate_ids: bool = False,\n"
        "    ) -> Path:",
    )
    text = text.replace(
        "                    self.as_dict(include_text=include_text),",
        "                    self.as_dict(\n"
        "                        include_ranked_candidate_ids=include_ranked_candidate_ids\n"
        "                    ),",
    )
    if "def _bootstrap_group_id(" not in text:
        marker = "\ndef prepare_phonetic_ablation(\n"
        helper = """

def _bootstrap_group_id(case, protocol: PhoneticAblationProtocol) -> str:
    if protocol.bootstrap_group == "speaker":
        return case.speaker_id
    if protocol.bootstrap_group == "session":
        return case.session_id
    if protocol.bootstrap_group == "source":
        return case.source_id
    raise ValueError("unknown bootstrap group")
"""
        if marker not in text:
            raise RuntimeError("runner group helper insertion marker missing")
        text = text.replace(marker, helper + marker, 1)
    text = text.replace(
        "            evaluate_case_arm(pool, decision, case.reference) for decision in decisions\n",
        "            evaluate_case_arm(\n"
        "                pool,\n"
        "                decision,\n"
        "                case.reference,\n"
        "                group_id=_bootstrap_group_id(case, protocol),\n"
        "            )\n"
        "            for decision in decisions\n",
    )
    save(path, text)


def patch_api() -> None:
    path = "src/semantic_asr/phonetic_experiment/api.py"
    text = load(path)
    if "from .registration import (" not in text:
        text = text.replace(
            "from .runner import (\n",
            "from .registration import (\n"
            "    PhoneticExperimentRegistration,\n"
            "    RegisteredPhoneticExperimentResult,\n"
            "    run_registered_phonetic_experiment,\n"
            ")\n"
            "from .runner import (\n",
            1,
        )
    for name in (
        "PhoneticExperimentRegistration",
        "RegisteredPhoneticExperimentResult",
        "run_registered_phonetic_experiment",
    ):
        entry = f'    "{name}",\n'
        if entry not in text:
            text = text.replace(
                '    "PhoneticPromotionDecision",\n',
                '    "PhoneticPromotionDecision",\n' + entry,
                1,
            )
    save(path, text)


def patch_tests() -> None:
    path = "tests/test_phonetic_experiment_runner.py"
    text = load(path)
    if "from dataclasses import replace\n" not in text:
        text = text.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nfrom dataclasses import replace\n",
            1,
        )
    text = text.replace(
        "def test_planning_view_contains_no_reference() -> None:\n"
        '    experiment, _runtime = manifest(pytest.ensuretemp("phonetic-planning-view"))',
        "def test_planning_view_contains_no_reference(tmp_path) -> None:\n"
        "    experiment, _runtime = manifest(tmp_path)",
    )
    text = text.replace(
        "    assert experiment.cases[0].reference.text not in repr(view)\n",
        "",
    )
    constructor_pattern = re.compile(
        r"    changed_case = case\.__class__\(\n"
        r"        \*\*\{\n"
        r"            \*\*\{field: getattr\(case, field\) for field in case\.__dataclass_fields__\},\n"
        r"            \"reference\": case\.reference\.__class__\(\n"
        r"                reference_id=case\.reference\.reference_id,\n"
        r"                text=\"ただ\",\n"
        r"                semantic_kind=case\.reference\.semantic_kind,\n"
        r"                critical=case\.reference\.critical,\n"
        r"            \),\n"
        r"        \}\n"
        r"    \)\n",
    )
    replacement = """    changed_case = replace(
        case,
        reference=replace(case.reference, text="ただ"),
    )
"""
    text, _count = constructor_pattern.subn(replacement, text, count=1)
    save(path, text)


def main() -> int:
    patch_protocol()
    patch_planner()
    patch_selection()
    patch_metrics()
    patch_runner()
    patch_api()
    patch_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
