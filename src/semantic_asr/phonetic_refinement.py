"""Frozen, opt-in phone/context candidate selection with auditable abstention.

The selector has no reference-text input. Training/evaluation live in a separate
module. Scores remain likelihood/preferences; selected corrections are provisional.
It never invents candidates, edits first-pass evidence or adds phone-derived morae
as an independent acoustic channel.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .contracts import sha256_json
from .evaluation import edit_distance


def _sha(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True, slots=True)
class PhoneContextCandidate:
    candidate_id: str
    text: str
    phones: tuple[str, ...]
    phone_score: float
    language_score: float
    profile_digest: str
    source_audio_sha256: str
    posterior_digest: str
    text_sha256: str

    def __post_init__(self):
        if not self.candidate_id or not self.text or not self.phones:
            raise ValueError("candidate ID, text and pronunciation are required")
        if not isinstance(self.phones, tuple) or any(
            not isinstance(p, str) or not p for p in self.phones
        ):
            raise TypeError("candidate phones must be immutable non-empty symbols")
        for name in ("phone_score", "language_score"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("phone and language scores must be finite reals")
        if hashlib.sha256(self.text.encode()).hexdigest() != self.text_sha256:
            raise ValueError("candidate text digest mismatch")
        if not all(
            _sha(v) for v in (self.profile_digest, self.source_audio_sha256, self.posterior_digest)
        ):
            raise ValueError("score provenance requires SHA-256 identities")


@dataclass(frozen=True, slots=True)
class PhoneContextDecision:
    baseline_id: str
    selected_id: str
    text: str
    changed: bool
    status: str
    reason: str
    phone_delta: float
    language_delta: float
    policy_digest: str
    candidate_evidence_digest: str

    @property
    def digest(self):
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class FrozenPhoneContextPolicy:
    """Non-negative linear preference with an independent acoustic-retention guard.

    ``language_weight`` and thresholds are fitted on development data only. A
    likelihood improvement is not a probability of a correct transcript. Positive
    language preference cannot override ``maximum_phone_regression``. Equal scores
    retain the baseline, especially for homophones not resolved by language evidence.
    """

    profile_digest: str
    development_manifest_sha256: str
    language_weight: float = 0.0
    minimum_gain: float = 0.0
    maximum_phone_regression: float = 0.0
    maximum_edit_ratio: float = 0.35
    require_language_agreement: bool = True
    schema: str = "semantic-asr-phone-context-policy-v1"

    def __post_init__(self):
        if not _sha(self.profile_digest) or not _sha(self.development_manifest_sha256):
            raise ValueError("policy must bind score profile and development manifest")
        if self.schema != "semantic-asr-phone-context-policy-v1":
            raise ValueError("unsupported phone/context policy")
        for name in (
            "language_weight",
            "minimum_gain",
            "maximum_phone_regression",
            "maximum_edit_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.maximum_edit_ratio <= 1:
            raise ValueError("maximum_edit_ratio must be in [0,1]")
        if not isinstance(self.require_language_agreement, bool):
            raise TypeError("require_language_agreement must be boolean")

    @property
    def digest(self):
        return sha256_json(asdict(self))

    def select(
        self, candidates: Sequence[PhoneContextCandidate], *, baseline_id: str
    ) -> PhoneContextDecision:
        rows = tuple(candidates)
        if not rows or len({c.candidate_id for c in rows}) != len(rows):
            raise ValueError("candidate IDs must be non-empty and unique")
        if {c.profile_digest for c in rows} != {self.profile_digest}:
            raise ValueError("candidate score profile differs from frozen policy")
        if len({(c.source_audio_sha256, c.posterior_digest) for c in rows}) != 1:
            raise ValueError(
                "candidate acoustic evidence belongs to different recordings or windows"
            )
        baseline = next((c for c in rows if c.candidate_id == baseline_id), None)
        if baseline is None:
            raise ValueError("baseline must remain in the candidate pool")
        best = baseline
        best_gain = self.minimum_gain
        for candidate in sorted(rows, key=lambda c: c.candidate_id):
            phone_delta = candidate.phone_score - baseline.phone_score
            language_delta = candidate.language_score - baseline.language_score
            if phone_delta < -self.maximum_phone_regression - 1e-12:
                continue
            if self.require_language_agreement and language_delta < -1e-12:
                continue
            if (
                edit_distance(baseline.text, candidate.text) / max(1, len(baseline.text))
                > self.maximum_edit_ratio
            ):
                continue
            gain = phone_delta + self.language_weight * language_delta
            if gain > best_gain + 1e-12:
                best, best_gain = candidate, gain
        changed = best.text != baseline.text
        return PhoneContextDecision(
            baseline_id,
            best.candidate_id,
            best.text,
            changed,
            "provisional" if changed else "retained",
            (
                "context-resolved-homophone"
                if best.phones == baseline.phones
                else "phone-context-preference"
            )
            if changed
            else "retained-first-pass",
            best.phone_score - baseline.phone_score,
            best.language_score - baseline.language_score,
            self.digest,
            sha256_json([asdict(c) for c in rows]),
        )
