from __future__ import annotations

import pytest

from semantic_asr.phonetic_dataset import PhoneticFeatureItem


def kwargs():
    return {
        "utterance_id": "u1",
        "split": "train",
        "feature_path": "features/u1.npy",
        "feature_sha256": "a" * 64,
        "frame_count": 2,
        "feature_dimension": 4,
        "feature_dtype": "float32",
        "phone_targets": (1, 1),
        "mora_targets": (1,),
        "phone_inventory_digest": "b" * 64,
        "mora_inventory_digest": "c" * 64,
        "speaker_id": "speaker-1",
        "source_id": "source-1",
        "source_audio_sha256": "d" * 64,
        "feature_revision": "features-r1",
        "rights_decision": "allow",
        "license_id": "license",
    }


def test_repeated_ctc_labels_require_an_intervening_blank_frame() -> None:
    with pytest.raises(ValueError, match="phone target requires more CTC frames"):
        PhoneticFeatureItem(**kwargs())


def test_sufficient_frame_count_accepts_repeated_targets() -> None:
    values = kwargs()
    values["frame_count"] = 3
    item = PhoneticFeatureItem(**values)

    assert item.frame_count == 3
