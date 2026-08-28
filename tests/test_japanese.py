from semantic_asr.japanese import (
    deterministic_normalize,
    join_japanese_fragments,
    merge_character_alignment,
    mora_sequence,
    split_mora,
    to_katakana,
)


def test_kana_normalization_and_foreign_mora() -> None:
    assert to_katakana("きゃ ﾃｨ ふぁ") == "キャ ティ ファ"
    assert mora_sequence("きゃく") == ["キャ", "ク"]
    assert mora_sequence("ティファール") == ["ティ", "ファ", "ー", "ル"]


def test_special_mora_are_independent() -> None:
    assert mora_sequence("がっこう") == ["ガ", "ッ", "コ", "ウ"]
    assert mora_sequence("しんぶん") == ["シ", "ン", "ブ", "ン"]
    assert mora_sequence("スーパー") == ["ス", "ー", "パ", "ー"]


def test_offsets_are_normalized_codepoint_offsets() -> None:
    units = split_mora("きゃく")
    assert units[0].kana == "キャ"
    assert (units[0].char_start, units[0].char_end) == (0, 2)
    assert (units[1].char_start, units[1].char_end) == (2, 3)


def test_character_ctc_rows_merge_into_one_timed_mora() -> None:
    units = merge_character_alignment(
        [
            {"char": "き", "startMs": 10, "endMs": 20, "confidence": 0.92},
            {"char": "ゃ", "startMs": 20, "endMs": 32, "confidence": 0.81},
            {"char": "く", "startMs": 32, "endMs": 50, "confidence": 0.90},
        ]
    )
    assert [unit.kana for unit in units] == ["キャ", "ク"]
    assert (units[0].start_ms, units[0].end_ms) == (10, 32)
    assert units[0].confidence == 0.81


def test_japanese_joining_removes_window_overlap_without_spaces() -> None:
    assert join_japanese_fragments(["今日は学校へ", "学校へ行きます。"]) == "今日は学校へ行きます。"
    assert join_japanese_fragments(["OpenAI", "API"]) == "OpenAI API"


def test_deterministic_normalization_is_readability_only() -> None:
    assert deterministic_normalize("ＡＩ  、便利!!!!!") == "AI、便利！！"
