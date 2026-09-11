from __future__ import annotations

import math

import pytest

from semantic_asr.audio_posterior_adapters import (
    DualPosteriorExtractor,
    FrozenAudioPosteriorExtractor,
    FrozenPosteriorModelConfig,
    PosteriorBundle,
    PosteriorLogits,
    PosteriorResourcePolicy,
    canonical_audio_sha256,
    posterior_sequence_from_logits,
)

REVISION = "1" * 40


def config(kind: str) -> FrozenPosteriorModelConfig:
    vocabulary = ("<blk>", "a", "b") if kind == "phone" else ("<blk>", "ア", "ブ")
    return FrozenPosteriorModelConfig(
        kind=kind,  # type: ignore[arg-type]
        model_id=f"test-{kind}-ctc",
        model_revision=REVISION,
        vocabulary=vocabulary,
        blank_symbol="<blk>",
        sample_rate=16_000,
        frame_stride_ms=20.0,
    )


class FakeBackend:
    def __init__(self, model_config: FrozenPosteriorModelConfig) -> None:
        self.config = model_config
        self.calls = 0

    def infer_logits(self, samples, *, sample_rate, source_audio_sha256):
        self.calls += 1
        return PosteriorLogits(
            values=((4.0, 0.0, -1.0), (0.0, 4.0, -1.0), (4.0, 0.0, -1.0)),
            source_audio_sha256=source_audio_sha256,
            model_config_digest=self.config.digest,
        )


def test_canonical_audio_digest_is_deterministic_and_rate_bound() -> None:
    samples = (0.0, 0.5, -0.25)

    assert canonical_audio_sha256(samples, 16_000) == canonical_audio_sha256(samples, 16_000)
    assert canonical_audio_sha256(samples, 16_000) != canonical_audio_sha256(samples, 48_000)


def test_frozen_backend_emits_candidate_independent_posterior() -> None:
    backend = FakeBackend(config("phone"))
    extractor = FrozenAudioPosteriorExtractor(backend)
    samples = tuple(0.0 for _ in range(1_600))

    posterior = extractor.extract(samples, sample_rate=16_000)

    assert backend.calls == 1
    assert posterior.kind == "phone"
    assert posterior.encoder == "test-phone-ctc"
    assert posterior.encoder_revision == REVISION
    assert posterior.frames[1].probability("a") > 0.95
    assert all(
        math.isclose(sum(value for _, value in frame.probabilities), 1.0, abs_tol=1e-6)
        for frame in posterior.frames
    )


def test_resampling_must_be_explicit() -> None:
    extractor = FrozenAudioPosteriorExtractor(FakeBackend(config("phone")))

    with pytest.raises(ValueError, match="resampling must be explicit"):
        extractor.extract((0.0,) * 100, sample_rate=8_000)


def test_logits_and_model_config_are_digest_bound() -> None:
    phone = config("phone")
    logits = PosteriorLogits(
        values=((1.0, 0.0, -1.0),),
        source_audio_sha256="a" * 64,
        model_config_digest="b" * 64,
    )

    with pytest.raises(ValueError, match="different model configuration"):
        posterior_sequence_from_logits(logits, phone)


def test_resource_policy_rejects_unbounded_frames() -> None:
    phone = config("phone")
    logits = PosteriorLogits(
        values=((1.0, 0.0, -1.0), (1.0, 0.0, -1.0)),
        source_audio_sha256="a" * 64,
        model_config_digest=phone.digest,
    )

    with pytest.raises(ValueError, match="frame count"):
        posterior_sequence_from_logits(
            logits,
            phone,
            resources=PosteriorResourcePolicy(maximum_frames=1),
        )


def test_dual_extractor_binds_phone_and_mora_to_one_audio() -> None:
    phone = FrozenAudioPosteriorExtractor(FakeBackend(config("phone")))
    mora = FrozenAudioPosteriorExtractor(FakeBackend(config("mora")))
    extractor = DualPosteriorExtractor(phone=phone, mora=mora)
    samples = tuple(0.0 for _ in range(1_600))

    bundle = extractor.extract(samples, sample_rate=16_000)

    assert isinstance(bundle, PosteriorBundle)
    assert bundle.phone is not None
    assert bundle.mora is not None
    assert bundle.phone.source_audio_sha256 == bundle.source_audio_sha256
    assert bundle.mora.source_audio_sha256 == bundle.source_audio_sha256


def test_exact_commit_revision_is_mandatory_by_default() -> None:
    with pytest.raises(ValueError, match="40-character"):
        FrozenPosteriorModelConfig(
            kind="phone",
            model_id="mutable-model",
            model_revision="main",
            vocabulary=("<blk>", "a"),
            blank_symbol="<blk>",
            sample_rate=16_000,
            frame_stride_ms=20.0,
        )
