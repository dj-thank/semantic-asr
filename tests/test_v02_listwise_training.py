from __future__ import annotations

from semantic_asr.listwise_training import (
    ListwiseTrainingConfig,
    train_listwise_semantic_mwer,
)
from semantic_asr.ranker_training import RankerExample
from semantic_asr.synthetic import synthetic_ranker_example


def _rotated_example(reference: str, example_id: str, offset: int) -> RankerExample:
    base = synthetic_ranker_example(
        reference,
        example_id=example_id,
        maximum_negatives=6,
        seed=offset + 7,
    )
    candidates = list(base.candidates)
    offset %= len(candidates)
    rotated = tuple(candidates[offset:] + candidates[:offset])
    return RankerExample(
        example_id=base.example_id,
        candidates=rotated,
        losses=base.losses,
        context=base.context,
    )


def test_listwise_semantic_mwer_reduces_expected_loss() -> None:
    examples = [
        _rotated_example(
            "えっと明日は行きません。料金は3000円です。",
            "one",
            2,
        ),
        _rotated_example("学校へ行って切符を買います。", "two", 3),
        _rotated_example("スーパーでしんぶんを買った。", "three", 1),
        _rotated_example("きょうは東京へ行きます。", "four", 4),
    ]
    result = train_listwise_semantic_mwer(
        examples,
        config=ListwiseTrainingConfig(
            epochs=220,
            learning_rate=0.05,
            l2=0.001,
            seed=31,
        ),
    )
    assert result.after.mean_expected_loss < result.before.mean_expected_loss
    assert result.after.mean_rank_regret <= result.before.mean_rank_regret
    assert result.after.top1_oracle_rate >= result.before.top1_oracle_rate
    assert result.profile.version == "listwise-mwer-1"
    assert result.profile.training_manifest_sha256 == result.training_manifest_sha256
