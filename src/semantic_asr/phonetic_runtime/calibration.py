"""Held-out utility normalization for candidate-specific phone or mora CTC evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import UtilityCalibrationProfile, _is_sha256
from ..phonetic_evidence import (
    CandidatePronunciation,
    PosteriorSequence,
    ctc_pronunciation_score,
)


@dataclass(frozen=True, slots=True)
class PhoneticCalibrationCandidate:
    candidate_id: str
    text: str
    symbols: tuple[str, ...]
    correct: bool

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text or not self.symbols:
            raise ValueError("calibration candidate requires ID, text, and symbols")
        if not isinstance(self.correct, bool):
            raise TypeError("correct must be a boolean")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "candidateId": self.candidate_id,
                "textSha256": sha256_json({"text": self.text}),
                "symbols": self.symbols,
                "correct": self.correct,
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticCalibrationExample:
    example_id: str
    posterior: PosteriorSequence
    candidates: tuple[PhoneticCalibrationCandidate, ...]

    def __post_init__(self) -> None:
        if not self.example_id or len(self.candidates) < 2:
            raise ValueError("calibration example requires ID and at least two candidates")
        if sum(candidate.correct for candidate in self.candidates) != 1:
            raise ValueError("calibration example requires exactly one correct candidate")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("calibration candidate IDs must be unique within an example")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "exampleId": self.example_id,
                "posteriorDigest": self.posterior.digest,
                "candidateDigests": [candidate.digest for candidate in self.candidates],
            }
        )


@dataclass(frozen=True, slots=True)
class CTCUtilityCalibrationReport:
    channel: str
    example_count: int
    candidate_count: int
    pairwise_correct_examples: int
    raw_score_mean: float
    raw_score_standard_deviation: float
    profile: UtilityCalibrationProfile
    held_out_manifest_sha256: str
    example_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.channel not in {"phone", "mora"}:
            raise ValueError("CTC utility calibration channel must be phone or mora")
        if self.example_count < 1 or self.candidate_count < self.example_count * 2:
            raise ValueError("CTC calibration counts are invalid")
        if not 0 <= self.pairwise_correct_examples <= self.example_count:
            raise ValueError("pairwise_correct_examples is invalid")
        if not math.isfinite(self.raw_score_mean):
            raise ValueError("raw_score_mean must be finite")
        if not math.isfinite(self.raw_score_standard_deviation):
            raise ValueError("raw_score_standard_deviation must be finite")
        if not _is_sha256(self.held_out_manifest_sha256):
            raise ValueError("held_out_manifest_sha256 must be a SHA-256 value")
        if self.profile.channel != self.channel:
            raise ValueError("calibration profile is assigned to the wrong channel")

    @property
    def pairwise_accuracy(self) -> float:
        return self.pairwise_correct_examples / self.example_count

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "profile": asdict(self.profile),
                "profileDigest": self.profile.digest,
                "pairwiseAccuracy": self.pairwise_accuracy,
            }
        )


def fit_ctc_utility_calibration(
    examples: tuple[PhoneticCalibrationExample, ...],
    *,
    held_out_manifest_sha256: str,
    revision: str,
    minimum_scale: float = 1e-4,
) -> CTCUtilityCalibrationReport:
    """Fit a bounded utility scale; this does not estimate correctness probability."""

    if not examples:
        raise ValueError("CTC utility calibration requires held-out examples")
    if not _is_sha256(held_out_manifest_sha256):
        raise ValueError("held_out_manifest_sha256 must be a SHA-256 value")
    if not revision:
        raise ValueError("calibration revision is required")
    if isinstance(minimum_scale, bool):
        raise TypeError("minimum_scale must be a real number")
    minimum_scale = float(minimum_scale)
    if not math.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be finite and positive")
    kinds = {example.posterior.kind for example in examples}
    if len(kinds) != 1:
        raise ValueError("one calibration profile cannot mix phone and mora posteriors")
    kind = next(iter(kinds))
    values: list[float] = []
    source: str | None = None
    correct_examples = 0
    candidate_count = 0
    for example in examples:
        scored: list[tuple[float, bool]] = []
        for candidate in example.candidates:
            pronunciation = CandidatePronunciation.create(
                candidate_id=candidate.candidate_id,
                text=candidate.text,
                kind=kind,
                symbols=candidate.symbols,
                producer="held-out-phonetic-calibration",
                producer_revision=revision,
            )
            score = ctc_pronunciation_score(example.posterior, pronunciation)
            source = source or score.evidence.source
            if score.evidence.source != source:
                raise ValueError("calibration examples mix incompatible CTC score sources")
            value = score.mean_frame_log_likelihood
            values.append(value)
            scored.append((value, candidate.correct))
            candidate_count += 1
        correct_value = next(value for value, correct in scored if correct)
        incorrect_values = [value for value, correct in scored if not correct]
        correct_examples += int(correct_value > max(incorrect_values))
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    scale = max(math.sqrt(variance), minimum_scale)
    assert source is not None
    profile = UtilityCalibrationProfile(
        channel=kind,
        score_source=source,
        score_kind=next(
            example
            for example in (
                ctc_pronunciation_score(
                    examples[0].posterior,
                    CandidatePronunciation.create(
                        candidate_id=examples[0].candidates[0].candidate_id,
                        text=examples[0].candidates[0].text,
                        kind=kind,
                        symbols=examples[0].candidates[0].symbols,
                        producer="held-out-phonetic-calibration",
                        producer_revision=revision,
                    ),
                ).evidence.kind,
            )
        ),
        center=center,
        scale=scale,
        fitted_manifest_sha256=held_out_manifest_sha256,
        revision=revision,
        higher_is_better=True,
    )
    return CTCUtilityCalibrationReport(
        channel=kind,
        example_count=len(examples),
        candidate_count=candidate_count,
        pairwise_correct_examples=correct_examples,
        raw_score_mean=center,
        raw_score_standard_deviation=scale,
        profile=profile,
        held_out_manifest_sha256=held_out_manifest_sha256,
        example_digests=tuple(example.digest for example in examples),
    )
