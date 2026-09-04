from __future__ import annotations

import json

import pytest

from semantic_asr.context_catalog import ContextCatalog, ContextEntry, load_context_catalog


def _catalog() -> ContextCatalog:
    return ContextCatalog(
        name="meeting-people",
        revision="crm-export-2026-09-04",
        entries=(
            ContextEntry(
                entry_id="person:moriwaki",
                phrase="森脇翔太",
                aliases=("森脇さん",),
                reading="モリワキショウタ",
                tags=("person", "project-a"),
                priority=2.0,
            ),
            ContextEntry(
                entry_id="product:semantic-asr",
                phrase="Semantic ASR",
                aliases=("セマンティックASR",),
                tags=("product",),
            ),
        ),
    )


def test_catalog_digest_is_order_independent_and_revision_bound() -> None:
    first = _catalog()
    reversed_catalog = ContextCatalog(
        name=first.name,
        revision=first.revision,
        entries=tuple(reversed(first.entries)),
    )
    assert first.digest == reversed_catalog.digest
    changed = ContextCatalog(
        name=first.name,
        revision="crm-export-2026-09-05",
        entries=first.entries,
    )
    assert first.digest != changed.digest


def test_selection_uses_aliases_tags_and_explicit_abstention() -> None:
    catalog = _catalog()
    selection = catalog.select(
        "森脇さんと次回の収録について相談",
        required_tags=("person",),
    )
    assert selection.hotwords == ("森脇翔太",)
    assert selection.abstained is False
    assert selection.matches[0].reasons

    distractor = catalog.select("明日の天気と電車", required_tags=("person",))
    assert distractor.hotwords == ()
    assert distractor.abstained is True
    assert distractor.reason == "no-match"

    empty = catalog.select("")
    assert empty.abstained is True
    assert empty.reason == "empty-query"


def test_phonetic_selection_uses_caller_supplied_reading() -> None:
    selection = _catalog().select("モリワキショウタ", required_tags=("person",))
    assert selection.hotwords == ("森脇翔太",)
    assert any(reason.startswith("phonetic") for reason in selection.matches[0].reasons)


def test_receipt_does_not_retain_raw_query_or_phrase() -> None:
    query = "森脇さんとの会議"
    selection = _catalog().select(query)
    payload = json.dumps(selection.receipt(), ensure_ascii=False)
    assert query not in payload
    assert "森脇翔太" not in payload
    assert selection.catalog_digest in payload
    assert selection.query_sha256 in payload


def test_json_loader_fails_closed_on_schema_and_duplicates(tmp_path) -> None:
    payload = _catalog().as_dict()
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = load_context_catalog(source)
    assert loaded is not None and loaded.digest == _catalog().digest

    payload["schemaVersion"] = 2
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schemaVersion"):
        load_context_catalog(source)

    with pytest.raises(ValueError, match="IDs"):
        ContextCatalog(
            name="bad",
            revision="1",
            entries=(
                ContextEntry("same", "A"),
                ContextEntry("same", "B"),
            ),
        )
    with pytest.raises(ValueError, match="canonical phrases"):
        ContextCatalog(
            name="bad",
            revision="1",
            entries=(
                ContextEntry("a", "Semantic ASR"),
                ContextEntry("b", "Semantic-ASR"),
            ),
        )


def test_selection_controls_are_bounded() -> None:
    catalog = _catalog()
    with pytest.raises(ValueError, match="limit"):
        catalog.select("森脇", limit=0)
    with pytest.raises(ValueError, match="minimum_score"):
        catalog.select("森脇", minimum_score=float("nan"))


def test_context_query_length_is_bounded() -> None:
    with pytest.raises(ValueError, match="context query"):
        _catalog().select("長" * 1_025)
