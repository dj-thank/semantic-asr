"""Frozen, provenance-bound context retrieval for ASR hotword biasing.

A catalog is caller-owned context that exists before an audio evaluation.  Selection is
deterministic and may abstain.  Only selected phrases become decoder hotwords; the audit
receipt stores IDs, hashes and scores rather than raw names or the raw query.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_pool import lenient_surface_key
from .contracts import canonical_json, sha256_json
from .japanese import mora_sequence, to_katakana

CONTEXT_CATALOG_SCHEMA_VERSION = 1
_MAX_ENTRY_COUNT = 100_000
_MAX_PHRASE_CHARACTERS = 160
_MAX_QUERY_CHARACTERS = 1_024


def _text(value: Any, *, name: str, maximum: int | None = None) -> str:
    output = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not output:
        raise ValueError(f"{name} is required")
    if maximum is not None and len(output) > maximum:
        raise ValueError(f"{name} must not exceed {maximum} characters")
    return output


def _tag(value: Any) -> str:
    return _text(value, name="tag", maximum=80).casefold()


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array of strings")
    return tuple(str(item) for item in value)


def _surface_key(value: str) -> str:
    return lenient_surface_key(unicodedata.normalize("NFKC", value)).casefold()


def _ngrams(value: str, order: int = 2) -> set[str]:
    if not value:
        return set()
    if len(value) < order:
        return {value}
    return {value[index : index + order] for index in range(len(value) - order + 1)}


def _dice(left: str, right: str) -> float:
    left_grams = _ngrams(left)
    right_grams = _ngrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def _edit_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _similarity(query: str, candidate: str, *, phonetic: bool = False) -> tuple[float, str]:
    if not query or not candidate:
        return 0.0, "none"
    prefix = "phonetic" if phonetic else "surface"
    if query == candidate:
        return 1.0, f"{prefix}-exact"
    if candidate in query:
        return (0.96 if phonetic else 0.98), f"{prefix}-contained"
    if len(query) >= 2 and query in candidate:
        coverage = len(query) / len(candidate)
        return min(0.94, 0.72 + 0.22 * coverage), f"{prefix}-query-contained"
    fuzzy = max(_dice(query, candidate), _edit_similarity(query, candidate))
    multiplier = 0.82 if phonetic else 0.72
    return multiplier * fuzzy, f"{prefix}-fuzzy"


def _reading_key(value: str) -> str:
    return "".join(mora_sequence(to_katakana(value)))


@dataclass(frozen=True, slots=True)
class ContextEntry:
    """One immutable phrase and its caller-supplied retrieval aliases."""

    entry_id: str
    phrase: str
    aliases: tuple[str, ...] = ()
    reading: str | None = None
    tags: tuple[str, ...] = ()
    priority: float = 1.0

    def __post_init__(self) -> None:
        entry_id = _text(self.entry_id, name="entry_id", maximum=200)
        phrase = _text(
            self.phrase,
            name=f"phrase for {entry_id!r}",
            maximum=_MAX_PHRASE_CHARACTERS,
        )
        aliases = tuple(
            dict.fromkeys(
                _text(
                    value,
                    name=f"alias for {entry_id!r}",
                    maximum=_MAX_PHRASE_CHARACTERS,
                )
                for value in self.aliases
            )
        )
        aliases = tuple(value for value in aliases if value != phrase)
        reading = None
        if self.reading is not None and str(self.reading).strip():
            reading = to_katakana(
                _text(
                    self.reading,
                    name=f"reading for {entry_id!r}",
                    maximum=_MAX_PHRASE_CHARACTERS,
                )
            )
        tags = tuple(sorted({_tag(value) for value in self.tags}))
        priority = float(self.priority)
        if not math.isfinite(priority) or not 0.0 <= priority <= 100.0:
            raise ValueError("priority must be finite and lie in [0, 100]")
        object.__setattr__(self, "entry_id", entry_id)
        object.__setattr__(self, "phrase", phrase)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "reading", reading)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "priority", priority)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContextEntry:
        return cls(
            entry_id=str(value.get("id", "")),
            phrase=str(value.get("phrase", "")),
            aliases=_string_tuple(value.get("aliases"), name="aliases"),
            reading=(None if value.get("reading") is None else str(value["reading"])),
            tags=_string_tuple(value.get("tags"), name="tags"),
            priority=float(value.get("priority", 1.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry_id,
            "phrase": self.phrase,
            "aliases": list(self.aliases),
            "reading": self.reading,
            "tags": list(self.tags),
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class ContextMatch:
    entry_id: str
    phrase: str
    score: float
    matched_surface: str
    reasons: tuple[str, ...]
    priority: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("context match score must lie in [0, 1]")
        if not self.reasons:
            raise ValueError("context match requires at least one reason")


@dataclass(frozen=True, slots=True)
class ContextSelection:
    catalog_name: str
    catalog_revision: str
    catalog_digest: str
    query_sha256: str
    matches: tuple[ContextMatch, ...]
    requested_tags: tuple[str, ...]
    minimum_score: float
    limit: int
    abstained: bool
    reason: str

    @property
    def hotwords(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(match.phrase for match in self.matches))

    def receipt(self) -> dict[str, Any]:
        """Return an audit record without retaining the raw query or selected phrases."""

        return {
            "schemaVersion": CONTEXT_CATALOG_SCHEMA_VERSION,
            "catalogName": self.catalog_name,
            "catalogRevision": self.catalog_revision,
            "catalogDigest": self.catalog_digest,
            "querySha256": self.query_sha256,
            "requestedTags": list(self.requested_tags),
            "minimumScore": self.minimum_score,
            "limit": self.limit,
            "abstained": self.abstained,
            "reason": self.reason,
            "selected": [
                {
                    "entryId": match.entry_id,
                    "phraseSha256": hashlib.sha256(match.phrase.encode("utf-8")).hexdigest(),
                    "score": round(match.score, 6),
                    "reasons": list(match.reasons),
                }
                for match in self.matches
            ],
        }

    @property
    def cache_context(self) -> str:
        """Stable context binding for decode cache identity."""

        return canonical_json(self.receipt())


@dataclass(frozen=True, slots=True)
class ContextCatalog:
    """A caller-frozen catalog; entries are canonicalized into stable ID order."""

    name: str
    revision: str
    entries: tuple[ContextEntry, ...]
    schema_version: int = CONTEXT_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != CONTEXT_CATALOG_SCHEMA_VERSION:
            raise ValueError(f"unsupported context catalog schemaVersion: {self.schema_version}")
        name = _text(self.name, name="catalog name", maximum=200)
        revision = _text(self.revision, name="catalog revision", maximum=200)
        entries = tuple(sorted(tuple(self.entries), key=lambda row: row.entry_id))
        if len(entries) > _MAX_ENTRY_COUNT:
            raise ValueError(f"context catalog exceeds {_MAX_ENTRY_COUNT} entries")
        identifiers = [entry.entry_id for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("context catalog entry IDs must be unique")
        phrase_keys = [_surface_key(entry.phrase) for entry in entries]
        if any(not key for key in phrase_keys):
            raise ValueError("context catalog phrases must contain searchable characters")
        if len(set(phrase_keys)) != len(phrase_keys):
            raise ValueError("context catalog canonical phrases must be unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "schema_version", CONTEXT_CATALOG_SCHEMA_VERSION)

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "name": self.name,
            "revision": self.revision,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContextCatalog:
        schema = value.get("schemaVersion")
        if schema != CONTEXT_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"context catalog schemaVersion must be exactly {CONTEXT_CATALOG_SCHEMA_VERSION}"
            )
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("context catalog entries must be an array")
        entries: list[ContextEntry] = []
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, Mapping):
                raise ValueError(f"context catalog entry {index} must be an object")
            entries.append(ContextEntry.from_mapping(raw))
        return cls(
            name=str(value.get("name", "")),
            revision=str(value.get("revision", "")),
            entries=tuple(entries),
            schema_version=int(schema),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ContextCatalog:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("context catalog root must be an object")
        return cls.from_mapping(payload)

    def select(
        self,
        query: str,
        *,
        limit: int = 8,
        minimum_score: float = 0.55,
        required_tags: Iterable[str] = (),
    ) -> ContextSelection:
        if not 1 <= int(limit) <= 64:
            raise ValueError("context selection limit must lie in [1, 64]")
        threshold = float(minimum_score)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_score must be finite and lie in [0, 1]")
        tags = tuple(sorted({_tag(value) for value in required_tags}))
        raw_query = unicodedata.normalize("NFKC", str(query or "")).strip()
        if len(raw_query) > _MAX_QUERY_CHARACTERS:
            raise ValueError(f"context query must not exceed {_MAX_QUERY_CHARACTERS} characters")
        query_sha256 = hashlib.sha256(raw_query.encode("utf-8")).hexdigest()
        query_key = _surface_key(raw_query)
        if not query_key:
            return ContextSelection(
                catalog_name=self.name,
                catalog_revision=self.revision,
                catalog_digest=self.digest,
                query_sha256=query_sha256,
                matches=(),
                requested_tags=tags,
                minimum_score=threshold,
                limit=int(limit),
                abstained=True,
                reason="empty-query",
            )

        query_reading = _reading_key(raw_query)
        matches: list[ContextMatch] = []
        required = set(tags)
        for entry in self.entries:
            if required and not required.issubset(entry.tags):
                continue
            best_score = 0.0
            best_surface = entry.phrase
            reasons: list[str] = []
            for surface in (entry.phrase, *entry.aliases):
                score, reason = _similarity(query_key, _surface_key(surface))
                if score > best_score:
                    best_score = score
                    best_surface = surface
                    reasons = [reason]
                elif score == best_score and score > 0.0 and reason not in reasons:
                    reasons.append(reason)

            reading_surfaces = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        entry.reading,
                        *(_reading_key(surface) for surface in entry.aliases),
                        _reading_key(entry.phrase),
                    )
                    if value
                )
            )
            for reading in reading_surfaces:
                score, reason = _similarity(query_reading, reading, phonetic=True)
                if score > best_score:
                    best_score = score
                    best_surface = entry.reading or best_surface
                    reasons = [reason]
                elif score == best_score and score > 0.0 and reason not in reasons:
                    reasons.append(reason)

            if best_score >= threshold:
                matches.append(
                    ContextMatch(
                        entry_id=entry.entry_id,
                        phrase=entry.phrase,
                        score=min(1.0, best_score),
                        matched_surface=best_surface,
                        reasons=tuple(reasons or ("threshold",)),
                        priority=entry.priority,
                    )
                )

        ordered = tuple(
            sorted(
                matches,
                key=lambda match: (-match.score, -match.priority, match.entry_id),
            )[: int(limit)]
        )
        return ContextSelection(
            catalog_name=self.name,
            catalog_revision=self.revision,
            catalog_digest=self.digest,
            query_sha256=query_sha256,
            matches=ordered,
            requested_tags=tags,
            minimum_score=threshold,
            limit=int(limit),
            abstained=not ordered,
            reason="selected" if ordered else "no-match",
        )


def load_context_catalog(
    value: ContextCatalog | str | Path | None,
) -> ContextCatalog | None:
    if value is None:
        return None
    if isinstance(value, ContextCatalog):
        return value
    return ContextCatalog.from_json(value)
