"""Frozen, opt-in phone/context candidate selection with auditable abstention.

The selector has no reference-text input. Training/evaluation live in a separate
module. Scores remain likelihood/preferences; selected corrections are provisional.
It never invents candidates, edits first-pass evidence or adds phone-derived morae
as an independent acoustic channel.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

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


def guard_decoder_fallback(
    baseline: PhoneContextCandidate,
    proposal: PhoneContextCandidate | None,
    *,
    maximum_phone_regression: float = 0.0,
    minimum_baseline_phone_score: float | None = None,
    require_same_phone_language_agreement: bool = False,
    language_weight: float = 0.0,
) -> PhoneContextDecision:
    """Veto an experimental decoder fallback when its acoustic score regresses.

    This is an opt-in retention guard, not a new ranker or spelling verifier.
    Both candidates must have the existing, identical full-window score profile
    and posterior. Language scores are unused unless the optional same-phone
    veto or joint preference is enabled; then their producer must also belong
    to that score profile. A positive language weight requires joint preference
    against the decoder before an acoustic veto, preserving unresolved conflicts
    such as a G2P misreading of a correctly spelled foreign proper name.
    A tie, including a G2P homophone,
    means only that this guard did not veto the decoder; it is not acoustic proof
    of the proposed spelling. Positive CTC gain cannot prove that a repetition,
    negation, numeral or proper name was preserved. Callers must retain the raw
    first pass separately and keep accepted proposals provisional.

    The optional score floor declines to veto when the phone model fits even the
    baseline poorly. Both thresholds are profile-specific development parameters,
    not calibrated correctness probabilities; freeze them before evaluation.

    Missing proposal evidence retains the baseline. Invalid or mixed evidence
    raises instead of silently falling back to an unscored decoder output.
    """
    for name, value in (
        ("maximum_phone_regression", maximum_phone_regression),
        ("minimum_baseline_phone_score", minimum_baseline_phone_score),
        ("language_weight", language_weight),
    ):
        if name == "minimum_baseline_phone_score" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("guard thresholds must be finite reals")
    if maximum_phone_regression < 0 or language_weight < 0:
        raise ValueError("maximum phone regression and language weight must be non-negative")
    if not isinstance(require_same_phone_language_agreement, bool):
        raise TypeError("same-phone language agreement flag must be boolean")
    rows = [baseline] if proposal is None else [baseline, proposal]
    if proposal is not None:
        if baseline.candidate_id == proposal.candidate_id:
            raise ValueError("baseline and proposal candidate IDs must differ")
        if baseline.profile_digest != proposal.profile_digest:
            raise ValueError("candidate score profiles differ")
        if (baseline.source_audio_sha256, baseline.posterior_digest) != (
            proposal.source_audio_sha256,
            proposal.posterior_digest,
        ):
            raise ValueError("candidate recordings or posterior windows differ")
    delta = 0.0 if proposal is None else proposal.phone_score - baseline.phone_score
    poor_fit = (
        minimum_baseline_phone_score is not None
        and baseline.phone_score < minimum_baseline_phone_score
    )
    language_delta = 0.0 if proposal is None else proposal.language_score - baseline.language_score
    acoustic_regression = not poor_fit and delta < -maximum_phone_regression - 1e-12
    joint_regression = delta + language_weight * language_delta < -maximum_phone_regression - 1e-12
    phone_veto = acoustic_regression and joint_regression

    def lexical_text(text):
        return "".join(
            c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P")
        )

    language_veto = (
        proposal is not None
        and require_same_phone_language_agreement
        and baseline.phones == proposal.phones
        and lexical_text(baseline.text) != lexical_text(proposal.text)
        and proposal.language_score < baseline.language_score - 1e-12
    )
    veto = proposal is None or phone_veto or language_veto
    selected = baseline if veto else proposal
    assert selected is not None
    changed = selected.text != baseline.text
    return PhoneContextDecision(
        baseline.candidate_id,
        selected.candidate_id,
        selected.text,
        changed,
        "provisional" if changed else "retained",
        "decoder-score-unavailable"
        if proposal is None
        else "decoder-phone-regression"
        if phone_veto
        else "decoder-same-phone-language-regression"
        if language_veto
        else "decoder-unresolved-low-baseline-fit"
        if poor_fit
        else "decoder-unresolved-phone-language-conflict"
        if acoustic_regression
        else "decoder-not-vetoed",
        selected.phone_score - baseline.phone_score,
        selected.language_score - baseline.language_score
        if require_same_phone_language_agreement or language_weight
        else 0.0,
        sha256_json(
            {
                "schema": "semantic-asr-decoder-acoustic-veto-v1",
                "profile_digest": baseline.profile_digest,
                "maximum_phone_regression": maximum_phone_regression,
                "minimum_baseline_phone_score": minimum_baseline_phone_score,
                "numerical_tolerance": 1e-12,
                "require_same_phone_language_agreement": require_same_phone_language_agreement,
                "language_weight": language_weight,
            }
        ),
        sha256_json([asdict(c) for c in rows]),
    )


def project_decoder_display_edits(baseline: str, proposal: str) -> str:
    """Keep limited display edits after vetoing a decoder's lexical replacement.

    The result is a display suggestion, never a replacement acoustic observation
    or a new scored candidate. Preserve baseline lexical content except equivalent
    numeral notation checked by the existing conservative Japanese ITN. Do not
    delete signs, decimal points, quotation boundaries, repetitions or negation.
    Prose layout edits cannot join/split numbers or ASCII alphanumeric tokens.
    """
    from .hayamimi_itn import convert

    numeral_chars = "〇零一二三四五六七八九十百千"
    numeral = re.compile(r"[0-9〇零一二三四五六七八九十百千]+")

    def protected(character):
        return (
            character.isdecimal()
            or character in numeral_chars
            or (character.isascii() and character.isalnum())
        )

    def separates_tokens(text, start, end):
        return start > 0 and end < len(text) and protected(text[start - 1]) and protected(text[end])

    parts = []
    for tag, a, b, c, d in SequenceMatcher(None, baseline, proposal, autojunk=False).get_opcodes():
        old, new = baseline[a:b], proposal[c:d]
        layout = all(ch in "、。 \t\n" for ch in old + new) and not (
            separates_tokens(baseline, a, b) or separates_tokens(proposal, c, d)
        )
        equal_number = bool(
            numeral.fullmatch(old)
            and numeral.fullmatch(new)
            and convert(old, "ja") == convert(new, "ja")
        )
        parts.append(new if tag == "equal" or layout or equal_number else old)
    return "".join(parts)
