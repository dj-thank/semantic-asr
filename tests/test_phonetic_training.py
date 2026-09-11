from __future__ import annotations

import pytest

from semantic_asr.phonetic_training import (
    JointPhoneticArtifact,
    JointPhoneticHeadConfig,
    PhoneticLabelInventory,
    PhoneticTrainingManifest,
    PhoneticValidationMetrics,
    posterior_configs_from_artifact,
)


def inventory(kind: str, labels: tuple[str, ...], digest: str) -> PhoneticLabelInventory:
    return PhoneticLabelInventory(
        kind=kind,
        labels=labels,
        blank_symbol="<blk>",
        revision=f"{kind}-labels-r1",
        source_manifest_sha256=digest,
    )


def head_config() -> JointPhoneticHeadConfig:
    return JointPhoneticHeadConfig(
        input_dimension=256,
        hidden_dimension=192,
        phone_inventory=inventory("phone", ("<blk>", "a", "k"), "a" * 64),
        mora_inventory=inventory("mora", ("<blk>", "ア", "カ"), "b" * 64),
        encoder_id="frozen-ja-encoder",
        encoder_revision="1" * 40,
        encoder_artifact_sha256="c" * 64,
    )


def training_manifest() -> PhoneticTrainingManifest:
    return PhoneticTrainingManifest(
        training_manifest_sha256="1" * 64,
        calibration_manifest_sha256="2" * 64,
        test_manifest_sha256="3" * 64,
        speaker_disjoint=True,
        source_disjoint=True,
        rights_registry_sha256="4" * 64,
        feature_revision="features-r1",
        random_seed=7,
    )


def artifact(config: JointPhoneticHeadConfig) -> JointPhoneticArtifact:
    return JointPhoneticArtifact(
        config_digest=config.digest,
        training_manifest_digest=training_manifest().digest,
        weights_sha256="d" * 64,
        serialization="safetensors",
        metrics=PhoneticValidationMetrics(
            phone_error_rate=0.1,
            mora_error_rate=0.08,
            phone_candidate_auc=0.8,
            mora_candidate_auc=0.82,
            critical_false_accept_rate=0.01,
            validation_sample_count=500,
        ),
        framework="torch",
        framework_version="2.x",
        revision="joint-phone-mora-r1",
    )


def test_artifact_exports_distinct_phone_and_mora_runtime_configs() -> None:
    config = head_config()

    phone, mora = posterior_configs_from_artifact(
        config,
        artifact(config),
        sample_rate=16_000,
        frame_stride_ms=20.0,
    )

    assert phone.kind == "phone"
    assert mora.kind == "mora"
    assert phone.vocabulary == config.phone_inventory.labels
    assert mora.vocabulary == config.mora_inventory.labels
    assert phone.artifact_sha256 == mora.artifact_sha256 == "d" * 64
    assert phone.digest != mora.digest


def test_training_splits_must_be_speaker_and_source_disjoint() -> None:
    with pytest.raises(ValueError, match="speaker and source disjoint"):
        PhoneticTrainingManifest(
            training_manifest_sha256="1" * 64,
            calibration_manifest_sha256="2" * 64,
            test_manifest_sha256="3" * 64,
            speaker_disjoint=False,
            source_disjoint=True,
            rights_registry_sha256="4" * 64,
            feature_revision="features-r1",
            random_seed=7,
        )


def test_pickle_style_weight_serialization_is_rejected() -> None:
    config = head_config()

    with pytest.raises(ValueError, match="safe serialization"):
        JointPhoneticArtifact(
            config_digest=config.digest,
            training_manifest_digest=training_manifest().digest,
            weights_sha256="d" * 64,
            serialization="pickle",
            metrics=artifact(config).metrics,
            framework="torch",
            framework_version="2.x",
            revision="bad",
        )
