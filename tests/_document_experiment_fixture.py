from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from semantic_asr.contracts import (
    CandidateEvidence,
    GateDecision,
    NormalizedTranscript,
    ObservedTranscript,
    RankedCandidate,
    sha256_json,
)
from semantic_asr.longform import LongformResult, LongformSegment, Window

AUDIO = "a" * 64
SPLIT = "b" * 64
RIGHTS = "c" * 64


def candidate(candidate_id: str, text: str, score: float) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        text=text,
        acoustic=score,
        mora=score - 0.02,
        lexical=score - 0.04,
        preservation=score - 0.01,
        cross_model=score - 0.03,
        source="fixture-asr",
    )


def observed(
    rows: tuple[CandidateEvidence, ...],
    selected_id: str,
    posterior: dict[str, float],
) -> ObservedTranscript:
    gate = GateDecision(
        weights={"acoustic": 1.0},
        posterior=posterior,
        entropy=0.5,
        disagreement=0.0,
        evidence_coverage=1.0,
        selective_risk=0.1,
        needs_relisten=False,
        abstain=False,
    )
    ranked = tuple(
        RankedCandidate(
            candidate=row,
            final_score=posterior[row.candidate_id],
            posterior=posterior[row.candidate_id],
            calibrated_scores={"acoustic": posterior[row.candidate_id]},
            gate=gate,
        )
        for row in rows
    )
    selected = next(row for row in ranked if row.candidate.candidate_id == selected_id)
    return ObservedTranscript.create(
        selected=selected,
        ranked=ranked,
        uncertainty_spans=[],
        source_audio_sha256=AUDIO,
    )


def first_pass() -> LongformResult:
    first_rows = (
        candidate("mata", "レビュー完了まではまたマージしません。", 0.70),
        candidate("mada", "レビュー完了まではまだマージしません。", 0.68),
    )
    first_observed = observed(first_rows, "mata", {"mata": 0.55, "mada": 0.45})
    first_normalized = NormalizedTranscript.attach(
        first_observed,
        text=first_observed.text,
        mode="deterministic",
    )
    second_rows = (candidate("approved", "承認後に統合します。", 0.90),)
    second_observed = observed(second_rows, "approved", {"approved": 1.0})
    second_normalized = NormalizedTranscript.attach(
        second_observed,
        text=second_observed.text,
        mode="deterministic",
    )
    segments = (
        LongformSegment(
            window=Window(index=0, start_ms=0, end_ms=1_000),
            observed=first_observed,
            normalized=first_normalized,
            diagnostics={"topPosterior": 0.55},
        ),
        LongformSegment(
            window=Window(index=1, start_ms=1_000, end_ms=2_000),
            observed=second_observed,
            normalized=second_normalized,
            diagnostics={"topPosterior": 1.0},
        ),
    )
    observed_text = "".join(segment.observed.text for segment in segments)
    normalized_text = "".join(segment.normalized.text for segment in segments)
    evidence = sha256_json(
        {
            "sourceAudioSha256": AUDIO,
            "durationMs": 2_000,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in segments],
        }
    )
    return LongformResult(
        source_name="fixture.wav",
        source_audio_sha256=AUDIO,
        duration_ms=2_000,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=segments,
        evidence_sha256=evidence,
        diagnostics={"provisionalWindowCount": 0},
    )


@dataclass(frozen=True, slots=True)
class FakeOption:
    option_id: str
    text: str
    start_ms: int
    end_ms: int
    generated: bool = False

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "optionId": self.option_id,
                "text": self.text,
                "startMs": self.start_ms,
                "endMs": self.end_ms,
                "generated": self.generated,
            }
        )


@dataclass(frozen=True, slots=True)
class FakePath:
    options: tuple[FakeOption, ...]
    base_score: float
    mean_audio_support: float

    @property
    def text(self) -> str:
        return "".join(option.text for option in self.options)

    @property
    def generated_window_count(self) -> int:
        return sum(option.generated for option in self.options)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "optionDigests": [option.digest for option in self.options],
                "baseScore": self.base_score,
                "meanAudioSupport": self.mean_audio_support,
            }
        )


def fake_paths() -> tuple[FakePath, FakePath, FakePath]:
    retained = FakePath(
        options=(
            FakeOption("retained-0", "レビュー完了まではまたマージしません。", 0, 1_000),
            FakeOption("retained-1", "承認後に統合します。", 1_000, 2_000),
        ),
        base_score=1.00,
        mean_audio_support=0.75,
    )
    corrected = FakePath(
        options=(
            FakeOption("corrected-0", "レビュー完了まではまだマージしません。", 0, 1_000),
            FakeOption("retained-1", "承認後に統合します。", 1_000, 2_000),
        ),
        base_score=0.98,
        mean_audio_support=0.73,
    )
    harmful = FakePath(
        options=(
            FakeOption("harmful-0", "レビュー完了まではただマージしません。", 0, 1_000),
            FakeOption("harmful-1", "承認前に統合します。", 1_000, 2_000),
        ),
        base_score=0.90,
        mean_audio_support=0.60,
    )
    return retained, corrected, harmful


def fake_plan():
    retained, corrected, harmful = fake_paths()
    decision = SimpleNamespace(
        alternatives=(retained, corrected, harmful),
        retained=retained,
    )
    return SimpleNamespace(
        decision=decision,
        digest=sha256_json(
            {"paths": [retained.digest, corrected.digest, harmful.digest]}
        ),
    )
