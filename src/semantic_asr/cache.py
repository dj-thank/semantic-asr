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
from .revisions import validate_artifact_sha256

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
    model_revision: str | None = None
    runtime_revision: str | None = None
    model_artifact_sha256: str | None = None
    decode_config_sha256: str | None = None
    score_domain: str | None = None

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
        for name, value in (
            ("model_revision", self.model_revision),
            ("runtime_revision", self.runtime_revision),
            ("score_domain", self.score_domain),
        ):
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must not be empty when present")
        for name, value in (
            ("model_artifact_sha256", self.model_artifact_sha256),
            ("decode_config_sha256", self.decode_config_sha256),
        ):
            if value is not None:
                normalized = validate_artifact_sha256(value, identifier=name)
                if normalized != value:
                    raise ValueError(f"{name} must be lowercase SHA-256 hex")

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
        model_revision: str | None = None,
        runtime_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        decode_settings: Mapping[str, Any] | None = None,
        decode_config: Mapping[str, Any] | None = None,
        decode_config_sha256: str | None = None,
        artifact_sha256: str | None = None,
        score_domain: str | None = None,
    ) -> CacheKey:
        hotword_text = "\u241f".join(str(value) for value in hotwords if str(value))
        if (
            decode_settings is not None
            and decode_config is not None
            and canonical_json(dict(decode_settings)) != canonical_json(dict(decode_config))
        ):
            raise ValueError("decode_settings and decode_config disagree")
        decode_settings = decode_settings if decode_settings is not None else decode_config
        if (
            model_artifact_sha256 is not None
            and artifact_sha256 is not None
            and model_artifact_sha256.lower() != artifact_sha256.lower()
        ):
            raise ValueError("model_artifact_sha256 and artifact_sha256 disagree")
        model_artifact_sha256 = model_artifact_sha256 or artifact_sha256
        if decode_settings is not None:
            settings_digest = hashlib.sha256(
                canonical_json(dict(decode_settings)).encode("utf-8")
            ).hexdigest()
            if decode_config_sha256 is not None:
                supplied_digest = validate_artifact_sha256(
                    decode_config_sha256,
                    identifier="decode_config_sha256",
                )
                if supplied_digest != settings_digest:
                    raise ValueError("decode settings and decode_config_sha256 disagree")
            decode_config_sha256 = settings_digest
        if model_artifact_sha256 is not None:
            model_artifact_sha256 = validate_artifact_sha256(
                model_artifact_sha256,
                identifier="model_artifact_sha256",
            )
        if decode_config_sha256 is not None:
            decode_config_sha256 = validate_artifact_sha256(
                decode_config_sha256,
                identifier="decode_config_sha256",
            )
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
            model_revision=None if model_revision is None else str(model_revision),
            runtime_revision=None if runtime_revision is None else str(runtime_revision),
            model_artifact_sha256=model_artifact_sha256,
            decode_config_sha256=decode_config_sha256,
            score_domain=None if score_domain is None else str(score_domain),
        )

    @property
    def decode_settings_sha256(self) -> str | None:
        """Backward-compatible alias for the decode configuration digest."""

        return self.decode_config_sha256

    @property
    def artifact_sha256(self) -> str | None:
        """Backward-compatible alias for the model artifact digest."""

        return self.model_artifact_sha256

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
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA busy_timeout=30000")
            self._migrate()
        except BaseException:
            self.connection.close()
            raise

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
