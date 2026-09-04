"""Bridge independent phone/mora posteriors into acoustically verified text arcs.

A frozen pronunciation lexicon is the dependency-free baseline for phoneme-to-grapheme proposal.
Neural P2G adapters can implement the same output contract later. The bridge never promotes raw
CTC likelihoods directly: each score must pass a held-out utility calibration profile first.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .contracts import sha256_json
from .multilevel_lattice import (
    BoundedUtility,
    LatticeArc,
    UtilityCalibrationProfile,
)
from .phonetic_evidence import (
    CandidatePronunciation,
    CTCPronunciationScore,
    PosteriorSequence,
    ctc_pronunciation_score,
)


@dataclass(frozen=True, slots=True)
class PronunciationLexiconEntry:
    entry_id: str
    text: str
    phone_symbols: tuple[str, ...] = ()
    mora_symbols: tuple[str, ...] = ()
    reading: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entry_id or not self.text:
            raise ValueError("lexicon entries require entry_id and text")
        if not self.phone_symbols and not self.mora_symbols:
            raise ValueError("a lexicon entry requires phone or mora symbols")
        if any(not symbol for symbol in (*self.phone_symbols, *self.mora_symbols)):
            raise ValueError("lexicon symbols must not be empty")
        object.__setattr__(self, "tags", tuple(dict.fromkeys(self.tags)))

    @property
    def pronunciation_key(self) -> str:
        return sha256_json(
            {
                "phones": self.phone_symbols,
                "moras": self.mora_symbols,
                "reading": self.reading,
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenPronunciationLexicon:
    name: str
    revision: str
    entries: tuple[PronunciationLexiconEntry, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("pronunciation lexicon requires name and revision")
        if not self.entries:
            raise ValueError("pronunciation lexicon requires at least one entry")
        if len({entry.entry_id for entry in self.entries}) != len(self.entries):
            raise ValueError("pronunciation lexicon entry IDs must be unique")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "entries": self.entries,
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticBridgeConfig:
    top_k: int = 8
    phone_weight: float = 1.0
    mora_weight: float = 1.0
    minimum_combined_utility: float = -1.0

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool):
            raise TypeError("top_k must be an integer")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        for name in ("phone_weight", "mora_weight", "minimum_combined_utility"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, numeric)
        if self.phone_weight < 0 or self.mora_weight < 0:
            raise ValueError("phonetic bridge weights must be non-negative")
        if self.phone_weight == 0 and self.mora_weight == 0:
            raise ValueError("at least one phonetic bridge weight must be positive")
        if not -1.0 <= self.minimum_combined_utility <= 1.0:
            raise ValueError("minimum_combined_utility must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class PhoneticTextProposal:
    candidate_id: str
    entry_id: str
    text: str
    pronunciation_key: str
    utilities: tuple[BoundedUtility, ...]
    combined_utility: float
    phone_score: CTCPronunciationScore | None
    mora_score: CTCPronunciationScore | None
    lexicon_digest: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.entry_id or not self.text:
            raise ValueError("phonetic proposal requires candidate, entry and text IDs")
        if not self.phone_score and not self.mora_score:
            raise ValueError("phonetic proposal requires phone or mora evidence")
        if not -1.0 <= self.combined_utility <= 1.0:
            raise ValueError("combined_utility must be in [-1, 1]")
        if len(self.pronunciation_key) != 64 or len(self.lexicon_digest) != 64:
            raise ValueError("phonetic proposal digests must be SHA-256 values")
        channels = {utility.channel for utility in self.utilities}
        if self.phone_score is not None and "phone" not in channels:
            raise ValueError("phone score requires a calibrated phone utility")
        if self.mora_score is not None and "mora" not in channels:
            raise ValueError("mora score requires a calibrated mora utility")

    def as_lattice_arc(
        self,
        *,
        span_id: str,
        arc_id: str | None = None,
        source_candidate_ids: tuple[str, ...] = (),
        observed_eligible: bool = True,
    ) -> LatticeArc:
        identifier = arc_id or f"phonetic-{self.candidate_id}"
        return LatticeArc(
            arc_id=identifier,
            span_id=span_id,
            text=self.text,
            origin="phonetic-proposal",
            utilities=self.utilities,
            observed_eligible=observed_eligible,
            pronunciation_key=self.pronunciation_key,
            source_candidate_ids=source_candidate_ids,
            metadata={
                "entryId": self.entry_id,
                "lexiconDigest": self.lexicon_digest,
                "combinedPhoneticUtility": self.combined_utility,
            },
        )


def _proposal_score(
    utilities: tuple[BoundedUtility, ...],
    config: PhoneticBridgeConfig,
) -> float:
    weighted = []
    for utility in utilities:
        if utility.channel == "phone" and config.phone_weight > 0:
            weighted.append((config.phone_weight, utility.value))
        elif utility.channel == "mora" and config.mora_weight > 0:
            weighted.append((config.mora_weight, utility.value))
    if not weighted:
        raise ValueError("no enabled calibrated phone or mora utilities were produced")
    total = sum(weight for weight, _ in weighted)
    return sum(weight * value for weight, value in weighted) / total


def propose_text_from_pronunciation(
    lexicon: FrozenPronunciationLexicon,
    *,
    phone_posterior: PosteriorSequence | None = None,
    mora_posterior: PosteriorSequence | None = None,
    phone_calibration: UtilityCalibrationProfile | None = None,
    mora_calibration: UtilityCalibrationProfile | None = None,
    config: PhoneticBridgeConfig | None = None,
) -> tuple[PhoneticTextProposal, ...]:
    """Rank lexicon text proposals against independent audio posteriorgrams."""

    config = config or PhoneticBridgeConfig()
    if phone_posterior is None and mora_posterior is None:
        raise ValueError("phone_posterior or mora_posterior is required")
    if phone_posterior is not None:
        if phone_posterior.kind != "phone" or phone_calibration is None:
            raise ValueError("phone posterior requires a phone calibration profile")
        if phone_calibration.channel != "phone":
            raise ValueError("phone calibration must produce the phone utility channel")
    if mora_posterior is not None:
        if mora_posterior.kind != "mora" or mora_calibration is None:
            raise ValueError("mora posterior requires a mora calibration profile")
        if mora_calibration.channel != "mora":
            raise ValueError("mora calibration must produce the mora utility channel")
    if (
        phone_posterior is not None
        and mora_posterior is not None
        and phone_posterior.source_audio_sha256 != mora_posterior.source_audio_sha256
    ):
        raise ValueError("phone and mora posteriorgrams must come from the same audio")

    proposals: list[PhoneticTextProposal] = []
    for entry in lexicon.entries:
        candidate_id = hashlib.sha256(
            f"{lexicon.digest}:{entry.entry_id}".encode()
        ).hexdigest()[:24]
        phone_score: CTCPronunciationScore | None = None
        mora_score: CTCPronunciationScore | None = None
        utilities: list[BoundedUtility] = []
        if phone_posterior is not None and entry.phone_symbols:
            pronunciation = CandidatePronunciation.create(
                candidate_id=candidate_id,
                text=entry.text,
                kind="phone",
                symbols=entry.phone_symbols,
                producer=f"lexicon:{lexicon.name}",
                producer_revision=lexicon.revision,
            )
            phone_score = ctc_pronunciation_score(phone_posterior, pronunciation)
            assert phone_calibration is not None
            utilities.append(phone_calibration.transform(phone_score.evidence))
        if mora_posterior is not None and entry.mora_symbols:
            pronunciation = CandidatePronunciation.create(
                candidate_id=candidate_id,
                text=entry.text,
                kind="mora",
                symbols=entry.mora_symbols,
                producer=f"lexicon:{lexicon.name}",
                producer_revision=lexicon.revision,
            )
            mora_score = ctc_pronunciation_score(mora_posterior, pronunciation)
            assert mora_calibration is not None
            utilities.append(mora_calibration.transform(mora_score.evidence))
        if not utilities:
            continue
        ordered_utilities = tuple(sorted(utilities, key=lambda utility: utility.channel))
        combined = _proposal_score(ordered_utilities, config)
        if combined < config.minimum_combined_utility:
            continue
        proposals.append(
            PhoneticTextProposal(
                candidate_id=candidate_id,
                entry_id=entry.entry_id,
                text=entry.text,
                pronunciation_key=entry.pronunciation_key,
                utilities=ordered_utilities,
                combined_utility=combined,
                phone_score=phone_score,
                mora_score=mora_score,
                lexicon_digest=lexicon.digest,
            )
        )
    proposals.sort(key=lambda row: (-row.combined_utility, row.entry_id))
    return tuple(proposals[: config.top_k])
