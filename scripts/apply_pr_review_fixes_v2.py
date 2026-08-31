#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing patch anchor for {label}: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def patch_planner() -> None:
    path = ROOT / "src/semantic_asr/planner.py"
    replace_once(
        path,
        """    reasons: tuple[str, ...]\n    affects_observed_decision: bool\n\n\n@dataclass(frozen=True, slots=True)\n""",
        """    reasons: tuple[str, ...]\n    affects_observed_decision: bool\n    hypotheses: tuple[str, ...] = ()\n\n\n@dataclass(frozen=True, slots=True)\n""",
        label="EvidenceAction local hypotheses",
    )
    replace_once(
        path,
        """        duration = _duration_ms(island)\n        base_gain = island.expected_information_gain * (0.55 + 0.45 * global_uncertainty)\n        for kind in (\n""",
        """        duration = _duration_ms(island)\n        base_gain = island.expected_information_gain * (0.55 + 0.45 * global_uncertainty)\n        hypotheses = tuple(\n            dict.fromkeys(\n                \"\".join(alternative.units).strip()\n                for alternative in island.alternatives\n                if \"\".join(alternative.units).strip()\n            )\n        )\n        for kind in (\n""",
        label="derive contradiction-local hypotheses",
    )
    replace_once(
        path,
        """                    affects_observed_decision=kind\n                    not in {\n                        \"local-teacher\",\n                        \"lexicon-lookup\",\n                    },\n                )\n""",
        """                    affects_observed_decision=kind\n                    not in {\n                        \"local-teacher\",\n                        \"lexicon-lookup\",\n                    },\n                    hypotheses=hypotheses,\n                )\n""",
        label="store contradiction-local hypotheses",
    )


def patch_longform() -> None:
    path = ROOT / "src/semantic_asr/longform.py"
    replace_once(
        path,
        """                aligned = self.forced_aligner.align(request, text=ranked[0].candidate.text)\n                alignment_rows.extend(\n                    asdict(row) if hasattr(row, \"__dataclass_fields__\") else dict(row)\n                    for row in aligned\n                )\n""",
        """                for hypothesis in action.hypotheses:\n                    aligned = self.forced_aligner.align(request, text=hypothesis)\n                    alignment_rows.append(\n                        {\n                            \"actionId\": action.action_id,\n                            \"hypothesis\": hypothesis,\n                            \"startMs\": action.start_ms,\n                            \"endMs\": action.end_ms,\n                            \"tokens\": [\n                                asdict(row)\n                                if hasattr(row, \"__dataclass_fields__\")\n                                else dict(row)\n                                for row in aligned\n                            ],\n                        }\n                    )\n""",
        label="align local hypotheses rather than full-window transcript",
    )


def patch_teachers() -> None:
    path = ROOT / "src/semantic_asr/teachers.py"
    replace_once(path, "import socket\n", "", label="remove DNS lookup dependency")
    replace_once(
        path,
        """    try:\n        return ipaddress.ip_address(host.strip(\"[]\")).is_loopback\n    except ValueError:\n        try:\n            addresses = socket.getaddrinfo(host, None)\n        except socket.gaierror:\n            return False\n        return bool(addresses) and all(\n            ipaddress.ip_address(row[4][0]).is_loopback for row in addresses\n        )\n""",
        """    try:\n        return ipaddress.ip_address(host.strip(\"[]\")).is_loopback\n    except ValueError:\n        # Do not resolve arbitrary hostnames. A name that resolves to loopback now\n        # can be rebound later; local teachers therefore accept only localhost or\n        # a literal loopback address.\n        return False\n""",
        label="fail closed on DNS hostnames",
    )
    replace_once(
        path,
        """    ) -> None:\n        self.model = model\n        self.endpoint = validate_openai_endpoint(endpoint)\n""",
        """    ) -> None:\n        lowered = model.lower()\n        if \":cloud\" in lowered or lowered.startswith(\"cloud/\"):\n            raise ValueError(\"cloud-routed model names are disabled\")\n        self.model = model\n        self.endpoint = validate_openai_endpoint(endpoint)\n""",
        label="reject cloud-routed OpenAI-compatible model names",
    )


def patch_tests() -> None:
    planner = ROOT / "tests/test_lattice_planner.py"
    replace_once(
        planner,
        """    assert all(action.utility > 0 for action in plan.selected)\n    assert any(\"currency\" in action.reasons for action in plan.selected)\n""",
        """    assert all(action.utility > 0 for action in plan.selected)\n    assert any(\"currency\" in action.reasons for action in plan.selected)\n    forced_plan = plan_evidence(\n        ranked,\n        lattice,\n        budget=EvidenceBudget(\n            total_cost_ms=2_500,\n            max_actions=1,\n            minimum_utility=0,\n        ),\n        enabled=(\"forced-align\",),\n    )\n    forced = forced_plan.selected[0]\n    assert set(forced.hypotheses) == {\"三\", \"二\"}\n    assert all(\"万円です\" not in hypothesis for hypothesis in forced.hypotheses)\n""",
        label="planner local-hypothesis regression",
    )

    longform = ROOT / "tests/test_longform.py"
    replace_once(
        longform,
        """class FakeTeacher:\n""",
        """class FakeForcedAligner:\n    name = \"fake-forced-aligner\"\n    model_name = \"fixture-aligner\"\n\n    def __init__(self) -> None:\n        self.calls: list[tuple[DecodeRequest, str]] = []\n\n    def align(self, request: DecodeRequest, *, text: str):\n        self.calls.append((request, text))\n        return [\n            {\n                \"text\": text,\n                \"start_ms\": request.start_ms,\n                \"end_ms\": request.end_ms,\n            }\n        ]\n\n\nclass FakeTeacher:\n""",
        label="fake forced aligner",
    )
    replace_once(
        longform,
        """def test_teacher_changes_only_normalized_layer() -> None:\n""",
        """def test_forced_aligner_receives_local_island_hypotheses() -> None:\n    with tempfile.TemporaryDirectory() as directory:\n        audio = Path(directory) / \"fixture.wav\"\n        audio.write_bytes(b\"fixture\")\n        aligner = FakeForcedAligner()\n        result = SemanticASRTranscriber(\n            FakeAdapter(),\n            forced_aligner=aligner,\n        ).transcribe(audio, duration_ms=1_000)\n        assert aligner.calls\n        full_sentences = {\n            \"昨日学校を行きました\",\n            \"昨日学校に行きました\",\n        }\n        hypotheses = {text for _request, text in aligner.calls}\n        assert not hypotheses & full_sentences\n        assert {\"を\", \"に\"} <= hypotheses\n        rows = result.segments[0].diagnostics[\"forcedAlignment\"]\n        assert {row[\"hypothesis\"] for row in rows} == hypotheses\n        assert all(row[\"tokens\"] for row in rows)\n\n\ndef test_teacher_changes_only_normalized_layer() -> None:\n""",
        label="forced-align local-hypothesis regression",
    )

    teachers = ROOT / "tests/test_teachers.py"
    replace_once(
        teachers,
        """    with pytest.raises(ValueError):\n        validate_openai_endpoint(\"http://example.com/v1/chat/completions\")\n""",
        """    with pytest.raises(ValueError):\n        validate_openai_endpoint(\"http://example.com/v1/chat/completions\")\n    with pytest.raises(ValueError):\n        validate_openai_endpoint(\n            \"http://localhost.localdomain:8000/v1/chat/completions\"\n        )\n""",
        label="DNS loopback hostname rejection regression",
    )
    replace_once(
        teachers,
        """    assert client.endpoint.endswith(\"/v1/chat/completions\")\n\n\ndef test_delayed_policy_only_queries_ambiguous_sets() -> None:\n""",
        """    assert client.endpoint.endswith(\"/v1/chat/completions\")\n    with pytest.raises(ValueError):\n        OpenAICompatibleRanker(model=\"cloud/Qwen3.8\")\n    with pytest.raises(ValueError):\n        OpenAICompatibleRanker(model=\"Qwen3.8:cloud\")\n\n\ndef test_delayed_policy_only_queries_ambiguous_sets() -> None:\n""",
        label="cloud-routed model rejection regression",
    )


def main() -> int:
    patch_planner()
    patch_longform()
    patch_teachers()
    patch_tests()
    for script in (
        "apply_pr_review_fixes.py",
        "apply_pr_review_fixes_v2.py",
        "repair_pr_review_test.py",
    ):
        (ROOT / "scripts" / script).unlink(missing_ok=True)
    print("PR review fixes applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
