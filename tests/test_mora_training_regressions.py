import pytest

from semantic_asr.japanese import merge_character_alignment, mora_sequence


def test_small_kana_never_attaches_across_separator_or_invalid_base():
    assert mora_sequence("き、ゃ") == ["キ", "ャ"]
    assert mora_sequence("き ゃ") == ["キ", "ャ"]
    assert mora_sequence("かぃ") == ["カ", "ィ"]
    assert mora_sequence("キャャ") == ["キャ", "ャ"]


def test_timed_kana_respects_explicit_pause_rows():
    units = merge_character_alignment(
        [
            {"char": "キ", "startMs": 0, "endMs": 10},
            {"char": "、", "startMs": 10, "endMs": 100},
            {"char": "ャ", "startMs": 100, "endMs": 110},
        ]
    )
    assert [u.kana for u in units] == ["キ", "ャ"]


def test_ctc_training_rejects_silently_zeroed_impossible_repetitions():
    torch = pytest.importorskip("torch")
    from semantic_asr.training import _ctc_loss

    with pytest.raises(ValueError, match="repeat|alignment"):
        _ctc_loss(
            torch.zeros(1, 2, 3),
            torch.tensor([[1, 1]]),
            input_lengths=torch.tensor([2]),
            blank_id=0,
            name="mora",
        )


@pytest.mark.parametrize("labels", [[[1.2, 2.0]], [[-2, 1]], [[9, 1]]])
def test_ctc_training_rejects_invalid_target_ids(labels):
    torch = pytest.importorskip("torch")
    from semantic_asr.training import _ctc_loss

    with pytest.raises((ValueError, TypeError)):
        _ctc_loss(
            torch.zeros(1, 5, 3),
            torch.tensor(labels),
            input_lengths=torch.tensor([5]),
            blank_id=0,
            name="phone",
        )
