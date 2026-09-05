from __future__ import annotations

import json
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from semantic_asr.phonetic_dataset import (  # noqa: E402
    file_sha256,
    load_phonetic_feature_manifest,
)
from semantic_asr.phonetic_heads_optional import (  # noqa: E402
    JointPhoneMoraCTCHead,
)
from semantic_asr.phonetic_trainer_optional import (  # noqa: E402
    PhoneticSequenceCalibration,
    evaluate_joint_phonetic_head,
)
from semantic_asr.phonetic_training import (  # noqa: E402
    JointPhoneticHeadConfig,
    PhoneticLabelInventory,
    PhoneticValidationMetrics,
)


def inventory(kind: str, marker: str) -> PhoneticLabelInventory:
    labels = ("<blk>", "a", "k") if kind == "phone" else ("<blk>", "ア", "カ")
    return PhoneticLabelInventory(
        kind=kind,
        labels=labels,
        blank_symbol="<blk>",
        revision=f"{kind}-r1",
        source_manifest_sha256=marker * 64,
    )


PHONE = inventory("phone", "a")
MORA = inventory("mora", "b")


def config() -> JointPhoneticHeadConfig:
    return JointPhoneticHeadConfig(
        input_dimension=4,
        hidden_dimension=6,
        phone_inventory=PHONE,
        mora_inventory=MORA,
        encoder_id="test-encoder",
        encoder_revision="1" * 40,
        encoder_artifact_sha256="c" * 64,
        dropout=0.0,
    )


def test_model_loss_rejects_repeated_target_without_blank_frame() -> None:
    model = JointPhoneMoraCTCHead(config())
    output = model(torch.zeros((1, 2, 4), dtype=torch.float32))

    with pytest.raises(ValueError, match="phone target requires more CTC frames"):
        model.loss(
            output,
            input_lengths=torch.tensor([2], dtype=torch.long),
            phone_targets=torch.tensor([1, 1], dtype=torch.long),
            phone_target_lengths=torch.tensor([2], dtype=torch.long),
            mora_targets=torch.tensor([1], dtype=torch.long),
            mora_target_lengths=torch.tensor([1], dtype=torch.long),
        )


def test_error_rates_may_exceed_one_but_auc_and_far_may_not() -> None:
    metrics = PhoneticValidationMetrics(
        phone_error_rate=1.4,
        mora_error_rate=1.1,
        phone_candidate_auc=0.7,
        mora_candidate_auc=0.8,
        critical_false_accept_rate=0.1,
        validation_sample_count=10,
    )

    assert metrics.phone_error_rate == 1.4
    with pytest.raises(ValueError, match="phone_candidate_auc"):
        PhoneticValidationMetrics(
            phone_error_rate=0.1,
            mora_error_rate=0.1,
            phone_candidate_auc=1.1,
            mora_candidate_auc=0.8,
            critical_false_accept_rate=0.1,
            validation_sample_count=10,
        )


def manifest(tmp_path: Path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    values = numpy.zeros((6, 4), dtype=numpy.float32)
    feature = feature_dir / "item.npy"
    numpy.save(feature, values, allow_pickle=False)
    row = {
        "schemaVersion": "1",
        "utteranceId": "test-item",
        "split": "test",
        "featurePath": "features/item.npy",
        "featureSha256": file_sha256(feature),
        "frameCount": 6,
        "featureDimension": 4,
        "featureDtype": "float32",
        "phoneTargets": [1, 2],
        "moraTargets": [1, 2],
        "phoneInventoryDigest": PHONE.digest,
        "moraInventoryDigest": MORA.digest,
        "speakerId": "speaker-test",
        "sourceId": "source-test",
        "sourceAudioSha256": "d" * 64,
        "featureRevision": "feature-r1",
        "rightsDecision": "allow",
        "licenseId": "license",
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return load_phonetic_feature_manifest(
        path,
        split="test",
        phone_inventory=PHONE,
        mora_inventory=MORA,
    )


def test_locked_test_uses_supplied_calibration_without_refitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = manifest(tmp_path)
    model = JointPhoneMoraCTCHead(config())
    calibration = PhoneticSequenceCalibration(
        phone_threshold=-10.0,
        mora_threshold=-10.0,
        target_true_accept_rate=0.95,
        calibration_manifest_sha256="e" * 64,
        revision="cal-r1",
        phone_false_accept_rate=0.2,
        mora_false_accept_rate=0.2,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("locked test attempted to refit its threshold")

    monkeypatch.setattr(
        "semantic_asr.phonetic_trainer_optional._threshold",
        forbidden,
    )
    result = evaluate_joint_phonetic_head(
        model,
        data,
        device="cpu",
        fit_calibration=False,
        calibration=calibration,
    )

    assert result.calibration == calibration
    assert result.manifest_sha256 == data.manifest_sha256
