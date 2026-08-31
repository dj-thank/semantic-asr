from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Literal

ErrorType = Literal[
    "long-vowel",
    "geminate",
    "moraic-nasal",
    "particle",
    "number",
    "negation",
    "filler",
    "repetition",
    "lexicon-neighbor",
]

_DIGIT_PATTERN = re.compile(r"\d")
_FILLERS = ("えー", "ええと", "えっと", "あの", "その", "まあ", "うーん", "んー")
_NEGATIONS = ("ない", "なかった", "ません", "ませんでした", "ではない", "じゃない", "ず", "ぬ")
_PARTICLE_SWAPS = {
    "は": ("が", "わ"),
    "が": ("は", "か"),
    "を": ("お", "に"),
    "に": ("へ", "を"),
    "へ": ("え", "に"),
}


@dataclass(frozen=True, slots=True)
class HardNegative:
    negative_id: str
    source_text: str
    text: str
    error_type: ErrorType
    criticality: float
    source_start: int
    source_end: int
    replacement: str
    generator_version: str = "ja-hard-negative-v1"

    def __post_init__(self) -> None:
        if not self.negative_id or not self.source_text or not self.text:
            raise ValueError("negative ID and texts are required")
        if self.text == self.source_text:
            raise ValueError("hard negative must change the source")
        if not 0 <= self.criticality <= 1:
            raise ValueError("criticality must be in [0, 1]")
        if not 0 <= self.source_start <= self.source_end <= len(self.source_text):
            raise ValueError("negative span is outside source text")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def _negative(
    source: str,
    *,
    start: int,
    end: int,
    replacement: str,
    error_type: ErrorType,
    criticality: float,
) -> HardNegative | None:
    text = source[:start] + replacement + source[end:]
    if not text or text == source:
        return None
    payload = {
        "source": source,
        "text": text,
        "type": error_type,
        "start": start,
        "end": end,
        "replacement": replacement,
    }
    identifier = (
        "neg-"
        + hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
    )
    return HardNegative(
        negative_id=identifier,
        source_text=source,
        text=text,
        error_type=error_type,
        criticality=criticality,
        source_start=start,
        source_end=end,
        replacement=replacement,
    )


def generate_hard_negatives(
    text: str,
    *,
    seed: int = 0,
    maximum: int = 32,
    lexicon_neighbors: dict[str, tuple[str, ...]] | None = None,
) -> tuple[HardNegative, ...]:
    """Generate labelled corruption candidates, never replacement transcripts.

    The generator is intentionally conservative and deterministic. Its outputs are
    training negatives only; they are not assumed to model the true ASR error
    distribution and must be mixed with real N-best competitors.
    """

    source = str(text or "")
    if not source:
        raise ValueError("text is required")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    rng = random.Random(seed)
    output: list[HardNegative] = []

    def add(candidate: HardNegative | None) -> None:
        if candidate is None:
            return
        if candidate.text in {source, *(row.text for row in output)}:
            return
        output.append(candidate)

    # Long-vowel mark deletion and vowel-length contraction.
    for index, character in enumerate(source):
        if character == "ー":
            add(
                _negative(
                    source,
                    start=index,
                    end=index + 1,
                    replacement="",
                    error_type="long-vowel",
                    criticality=0.45,
                )
            )
    for match in re.finditer(r"([ぁ-ゖァ-ヶ])([あいうえおアイウエオ])", source):
        add(
            _negative(
                source,
                start=match.start(2),
                end=match.end(2),
                replacement="",
                error_type="long-vowel",
                criticality=0.42,
            )
        )

    # Sokuon deletion and a bounded insertion before common obstruent kana.
    for index, character in enumerate(source):
        if character in {"っ", "ッ"}:
            add(
                _negative(
                    source,
                    start=index,
                    end=index + 1,
                    replacement="",
                    error_type="geminate",
                    criticality=0.55,
                )
            )
    insertion_sites = [
        index
        for index, character in enumerate(source)
        if index > 0 and character in "かきくけこさしすせそたちつてとカキクケコサシスセソタチツテト"
    ]
    rng.shuffle(insertion_sites)
    for index in insertion_sites[:2]:
        replacement = "ッ" if "ァ" <= source[index] <= "ヶ" else "っ"
        add(
            _negative(
                source,
                start=index,
                end=index,
                replacement=replacement,
                error_type="geminate",
                criticality=0.45,
            )
        )

    # Moraic nasal deletion/substitution.
    for index, character in enumerate(source):
        if character in {"ん", "ン"}:
            add(
                _negative(
                    source,
                    start=index,
                    end=index + 1,
                    replacement="",
                    error_type="moraic-nasal",
                    criticality=0.52,
                )
            )

    # Particle confusions are generated only at simple bounded positions. They are
    # labels for robustness training, not a morphological analysis claim.
    for index, character in enumerate(source):
        replacements = _PARTICLE_SWAPS.get(character)
        if replacements is None:
            continue
        for replacement in replacements:
            add(
                _negative(
                    source,
                    start=index,
                    end=index + 1,
                    replacement=replacement,
                    error_type="particle",
                    criticality=0.58,
                )
            )

    # Digit substitution: change one digit while preserving formatting.
    for match in _DIGIT_PATTERN.finditer(source):
        original = int(match.group())
        replacement = str((original + rng.choice((1, 2, 5, 9))) % 10)
        add(
            _negative(
                source,
                start=match.start(),
                end=match.end(),
                replacement=replacement,
                error_type="number",
                criticality=1.0,
            )
        )

    # Negation deletion is high impact. A limited insertion is added only when no
    # explicit negation is present and a copular ending is found.
    found_negation = False
    for negation in _NEGATIONS:
        start = 0
        while True:
            index = source.find(negation, start)
            if index < 0:
                break
            found_negation = True
            add(
                _negative(
                    source,
                    start=index,
                    end=index + len(negation),
                    replacement="",
                    error_type="negation",
                    criticality=1.0,
                )
            )
            start = index + len(negation)
    if not found_negation:
        for ending in ("です", "ます"):
            index = source.rfind(ending)
            if index >= 0:
                add(
                    _negative(
                        source,
                        start=index,
                        end=index,
                        replacement="ない",
                        error_type="negation",
                        criticality=1.0,
                    )
                )
                break

    # Filler deletion and local repetition removal.
    for filler in _FILLERS:
        start = 0
        while True:
            index = source.find(filler, start)
            if index < 0:
                break
            add(
                _negative(
                    source,
                    start=index,
                    end=index + len(filler),
                    replacement="",
                    error_type="filler",
                    criticality=0.50,
                )
            )
            start = index + len(filler)
    for match in re.finditer(r"(.{1,6})\1", source):
        add(
            _negative(
                source,
                start=match.start(),
                end=match.start() + len(match.group(1)),
                replacement="",
                error_type="repetition",
                criticality=0.56,
            )
        )

    # User/rights-gated lexicon neighbors can supply homophones or domain near misses.
    for phrase, neighbors in (lexicon_neighbors or {}).items():
        start = source.find(phrase)
        if start < 0:
            continue
        for replacement in neighbors:
            if replacement and replacement != phrase:
                add(
                    _negative(
                        source,
                        start=start,
                        end=start + len(phrase),
                        replacement=replacement,
                        error_type="lexicon-neighbor",
                        criticality=0.82,
                    )
                )

    # Critical examples are kept first, then deterministic ID order.
    output.sort(key=lambda row: (-row.criticality, row.error_type, row.negative_id))
    return tuple(output[:maximum])
