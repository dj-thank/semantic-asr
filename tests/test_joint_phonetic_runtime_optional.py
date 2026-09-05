from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from semantic_asr.joint_phonetic_runtime_optional import (  # noqa: E402
    FrozenAudioFeatureConfig,
    FrozenFeatureMatrix,
    JointPhoneticPosteriorExtractor,
)
from semantic_asr.phonetic_heads_optional import JointPhoneMoraCTCHead  # noqa: E402
from semantic_asr.phonetic_trainer_optional import save_joint_phonetic_weights  # noqa: E402
from semantic_asr.phonetic_training import (  # noqa: E402
    JointPhoneticArtifact,
    JointPhoneticHeadConfig,
    PhoneticLabelInventory,
    PhoneticTrainingManifest,
    PhoneticValidationMetrics,
)

AUDIO = "a" * 64
ENCODER_REVISION = "1" * 40
ENCODER_ARTIFACT = "b" * 64


def inventory(kind: str, labels: tuple[str, ...], marker: str):
    return PhoneticLabelInventory(
        kind=kind,
        labels=labels,
        blank_symbol="<blk>",
        revision=f"{kind}-r1",
        source_manifest_sha256=marker * 64,
    )


def head_config() -> JointPhoneticHeadConfig:
    return JointPhoneticHeadConfig(
        input_dimension=4,
        hidden_dimension=6,
        phone_inventory=inventory("phone", ("<blk>", "a", "k"), "c"),
        mora_inventory=inventory("mora", ("<blk>", "ア", "カ"), "d"),
        encoder_id="frozen-test-encoder",
        encoder_revision=ENCODER_REVISION,
        encoder_artifact_sha256=ENCODER_ARTIFACT,
        dropout=0.0,
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
        random_seed=0,
    )


def artifact(config: JointPhoneticHeadConfig, weights_sha256: str):
    return JointPhoneticArtifact(
        config_digest=config.digest,
        training_manifest_digest=training_manifest().digest,
        weights_sha256=weights_sha256,
        serialization="safetensors",
        metrics=PhoneticValidationMetrics(
            phone_error_rate=0.2,
            mora_error_rate=0.2,
            phone_candidate_auc=0.7,
            mora_candidate_auc=0.7,
            critical_false_accept_rate=0.05,
            validation_sample_count=20,
        ),
        framework="torch",
        framework_version="test",
        revision="joint-r1",
    )


class FakeFeatureBackend:
    def __init__(self, config: FrozenAudioFeatureConfig) -> None:
        self.config = config
        self.calls = 0

    def extract_features(self, samples, *, sample_rate, source_audio_sha256):
        self.calls += 1
        assert sample_rate == self.config.sample_rate
        assert samples
        return FrozenFeatureMatrix(
            values=(
                (0.1, 0.2, 0.3, 0.4),
                (0.2, 0.3, 0.4, 0.5),
                (0.3, 0.4, 0.5, 0.6),
            ),
            source_audio_sha256=source_audio_sha256,
            feature_config_digest=self.config.digest,
        )


def feature_config() -> FrozenAudioFeatureConfig:
    return FrozenAudioFeatureConfig(
        model_id="frozen-test-encoder",
        model_revision=ENCODER_REVISION,
        model_artifact_sha256=ENCODER_ARTIFACT,
        layer_index=3,
        sample_rate=16_000,
        feature_dimension=4,
        frame_stride_ms=20.0,
    )


def test_runtime_uses_one_feature_pass_for_phone_and_mora(tmp_path: Path) -> None:
    config = head_config()
    model = JointPhoneMoraCTCHead(config)
    weights = tmp_path / "joint.safetensors"
    weights_sha256 = save_joint_phonetic_weights(
        model,
        weights,
        config_digest=config.digest,
    )
    backend = FakeFeatureBackend(feature_config())
    runtime = JointPhoneticPosteriorExtractor(
        feature_backend=backend,
        head_config=config,
        artifact=artifact(config, weights_sha256),
        weights_path=weights,
    )

    bundle = runtime.extract(
        (0.0, 0.1, -0.1),
        sample_rate=16_000,
        source_audio_sha256=AUDIO,
    )

    assert backend.calls == 1
    assert bundle.source_audio_sha256 == AUDIO
    assert bundle.phone is not None and bundle.phone.kind == "phone"
    assert bundle.mora is not None and bundle.mora.kind == "mora"
    assert len(bundle.phone.frames) == len(bundle.mora.frames) == 3
    assert all(
        sum(value for _, value in frame.probabilities) == pytest.approx(1.0)
        for frame in (*bundle.phone.frames, *bundle.mora.frames)
    )


def test_runtime_rejects_tampered_weights(tmp_path: Path) -> None:
    config = head_config()
    model = JointPhoneMoraCTCHead(config)
    weights = tmp_path / "joint.safetensors"
    digest = save_joint_phonetic_weights(model, weights, config_digest=config.digest)
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="weights SHA-256 mismatch"):
        JointPhoneticPosteriorExtractor(
            feature_backend=FakeFeatureBackend(feature_config()),
            head_config=config,
            artifact=artifact(config, digest),
            weights_path=weights,
        )


def test_runtime_rejects_encoder_revision_mismatch(tmp_path: Path) -> None:
    config = head_config()
    model = JointPhoneMoraCTCHead(config)
    weights = tmp_path / "joint.safetensors"
    digest = save_joint_phonetic_weights(model, weights, config_digest=config.digest)
    wrong = FrozenAudioFeatureConfig(
        model_id="frozen-test-encoder",
        model_revision="2" * 40,
        model_artifact_sha256=ENCODER_ARTIFACT,
        layer_index=3,
        sample_rate=16_000,
        feature_dimension=4,
        frame_stride_ms=20.0,
    )

    with pytest.raises(ValueError, match="revision"):
        JointPhoneticPosteriorExtractor(
            feature_backend=FakeFeatureBackend(wrong),
            head_config=config,
            artifact=artifact(config, digest),
            weights_path=weights,
        )
