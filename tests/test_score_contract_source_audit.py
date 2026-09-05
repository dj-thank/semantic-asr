from __future__ import annotations

from pathlib import Path


def test_repository_defines_one_evidence_score_class() -> None:
    source_root = Path(__file__).parents[1] / "src" / "semantic_asr"
    definitions: list[tuple[str, int]] = []
    for source_path in sorted(source_root.glob("*.py")):
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if line.startswith("class EvidenceScore"):
                definitions.append((source_path.name, line_number))

    assert len(definitions) == 1, definitions
    assert definitions[0][0] == "_score_contract_base.py"
