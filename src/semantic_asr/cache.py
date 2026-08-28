from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import CandidateEvidence, canonical_json

SCHEMA_VERSION = "1"


def _digest(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheKey:
    namespace: str
    audio_sha256: str
    start_ms: int
    end_ms: int
    adapter: str
    model: str
    language: str | None
    beam_size: int
    hypotheses: int
    prompt_sha256: str | None = None
    hotwords_sha256: str | None = None
    context_sha256: str | None = None
    calibration_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.namespace or not self.adapter or not self.model:
            raise ValueError("cache namespace, adapter, and model are required")
        digest = self.audio_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("audio_sha256 must be a SHA-256 hex digest")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("cache span is invalid")
        if self.beam_size < 1 or self.hypotheses < 1:
            raise ValueError("beam_size and hypotheses must be positive")

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        audio_sha256: str,
        start_ms: int,
        end_ms: int,
        adapter: str,
        model: str,
        language: str | None,
        beam_size: int,
        hypotheses: int,
        prompt: str | None = None,
        hotwords: Iterable[str] = (),
        context: str | None = None,
        calibration_digest: str | None = None,
    ) -> CacheKey:
        hotword_text = "\u241f".join(str(value) for value in hotwords if str(value))
        return cls(
            namespace=namespace,
            audio_sha256=audio_sha256.lower(),
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            adapter=adapter,
            model=model,
            language=None if language in {None, "", "auto"} else str(language),
            beam_size=int(beam_size),
            hypotheses=int(hypotheses),
            prompt_sha256=_digest(prompt),
            hotwords_sha256=_digest(hotword_text),
            context_sha256=_digest(context),
            calibration_sha256=calibration_digest,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherCacheEntry:
    probabilities: dict[str, float]
    abstained: bool
    entropy: float
    model: str
    protocol: str

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("teacher probabilities must not be empty")
        if any(
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in self.probabilities.values()
        ):
            raise ValueError("teacher probabilities must be finite and in [0, 1]")
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("teacher probabilities must sum to one")
        if not math.isfinite(self.entropy):
            raise ValueError("teacher entropy must be finite")


class EvidenceCache:
    """Versioned SQLite cache for model evidence; raw audio is never stored."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._migrate()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_cache (
                cache_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                key_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS evidence_cache_namespace_idx
                ON evidence_cache(namespace);
            """
        )
        row = self.connection.execute(
            "SELECT value FROM cache_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO cache_metadata(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
        elif row[0] != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported cache schema {row[0]!r}; expected {SCHEMA_VERSION!r}")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EvidenceCache:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.close()

    def put_json(self, key: CacheKey, payload: Mapping[str, Any]) -> None:
        key_json = canonical_json(asdict(key))
        payload_json = canonical_json(dict(payload))
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO evidence_cache(
                    cache_key, namespace, key_json, payload_json, payload_sha256, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    namespace=excluded.namespace,
                    key_json=excluded.key_json,
                    payload_json=excluded.payload_json,
                    payload_sha256=excluded.payload_sha256,
                    created_at=excluded.created_at
                """,
                (
                    key.digest,
                    key.namespace,
                    key_json,
                    payload_json,
                    payload_sha256,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_json(self, key: CacheKey) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT key_json, payload_json, payload_sha256 FROM evidence_cache WHERE cache_key=?",
            (key.digest,),
        ).fetchone()
        if row is None:
            return None
        key_json, payload_json, payload_sha256 = row
        if key_json != canonical_json(asdict(key)):
            raise RuntimeError("cache key digest collision or corruption")
        if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != payload_sha256:
            raise RuntimeError("cache payload was modified")
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise RuntimeError("cache payload is not an object")
        return payload

    def put_candidates(self, key: CacheKey, candidates: Iterable[CandidateEvidence]) -> None:
        rows = [candidate.as_dict() for candidate in candidates]
        if not rows:
            raise ValueError("candidate cache rows must not be empty")
        self.put_json(key, {"kind": "candidates", "rows": rows})

    def get_candidates(self, key: CacheKey) -> list[CandidateEvidence] | None:
        payload = self.get_json(key)
        if payload is None:
            return None
        if payload.get("kind") != "candidates" or not isinstance(payload.get("rows"), list):
            raise RuntimeError("cache entry is not a candidate set")
        return [CandidateEvidence.from_dict(row) for row in payload["rows"]]

    def put_teacher(self, key: CacheKey, entry: TeacherCacheEntry) -> None:
        self.put_json(
            key,
            {
                "kind": "teacher",
                "probabilities": entry.probabilities,
                "abstained": entry.abstained,
                "entropy": entry.entropy,
                "model": entry.model,
                "protocol": entry.protocol,
            },
        )

    def get_teacher(self, key: CacheKey) -> TeacherCacheEntry | None:
        payload = self.get_json(key)
        if payload is None:
            return None
        if payload.get("kind") != "teacher":
            raise RuntimeError("cache entry is not a teacher result")
        return TeacherCacheEntry(
            probabilities={
                str(candidate_id): float(probability)
                for candidate_id, probability in payload["probabilities"].items()
            },
            abstained=bool(payload["abstained"]),
            entropy=float(payload["entropy"]),
            model=str(payload["model"]),
            protocol=str(payload["protocol"]),
        )

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            row = self.connection.execute("SELECT COUNT(*) FROM evidence_cache").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM evidence_cache WHERE namespace=?", (namespace,)
            ).fetchone()
        return int(row[0])
