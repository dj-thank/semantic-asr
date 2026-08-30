import sqlite3
import tempfile
from pathlib import Path

import pytest

from semantic_asr.cache import CacheKey, EvidenceCache, TeacherCacheEntry
from semantic_asr.contracts import CandidateEvidence, MoraUnit


def make_key(**overrides):
    values = {
        "namespace": "base",
        "audio_sha256": "a" * 64,
        "start_ms": 0,
        "end_ms": 1_000,
        "adapter": "fake",
        "model": "fixture",
        "language": "ja",
        "beam_size": 5,
        "hypotheses": 2,
        "prompt": "技術会議",
        "hotwords": ("MoraWeave", "Qwen"),
        "context": "文脈",
        "calibration_digest": "b" * 64,
    }
    values.update(overrides)
    return CacheKey.create(**values)


def test_candidate_roundtrip_preserves_all_evidence() -> None:
    candidate = CandidateEvidence(
        "a",
        "きょうです",
        token_ids=(1, 2),
        acoustic=-0.2,
        mora=0.7,
        cross_model=0.8,
        rank=1,
        hypothesis_count=2,
        sequence_score=-0.4,
        avg_logprob=-0.2,
        beam_confidence=0.8,
        source="fake",
        reading="キョウデス",
        mora_units=(MoraUnit(0, "キョ"), MoraUnit(1, "ウ")),
        metadata={"adapter": "fake", "sourceSupport": ["fake"]},
    )
    with (
        tempfile.TemporaryDirectory() as directory,
        EvidenceCache(Path(directory) / "cache.sqlite3") as cache,
    ):
        cache.put_candidates(make_key(), [candidate])
        assert cache.get_candidates(make_key()) == [candidate]
        assert cache.count("base") == 1


def test_cache_key_changes_with_context_hotwords_and_calibration() -> None:
    keys = {
        make_key(context="A").digest,
        make_key(context="B").digest,
        make_key(hotwords=("東京",)).digest,
        make_key(calibration_digest="c" * 64).digest,
    }
    assert len(keys) == 4


def test_teacher_abstention_roundtrip() -> None:
    entry = TeacherCacheEntry(
        probabilities={"a": 0.5, "b": 0.5},
        abstained=True,
        entropy=1.0,
        model="local-qwen",
        protocol="ollama",
    )
    with (
        tempfile.TemporaryDirectory() as directory,
        EvidenceCache(Path(directory) / "cache.sqlite3") as cache,
    ):
        cache.put_teacher(make_key(namespace="teacher"), entry)
        assert cache.get_teacher(make_key(namespace="teacher")) == entry


def test_unknown_cache_schema_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cache.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO cache_metadata VALUES('schema_version', '999')")
        connection.commit()
        connection.close()
        with pytest.raises(RuntimeError):
            EvidenceCache(path)


@pytest.mark.parametrize("failure_stage", ["pragma", "migration"])
def test_cache_initialization_failure_closes_connection(
    tmp_path, monkeypatch, failure_stage
) -> None:
    connections = []

    class TrackedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if failure_stage == "pragma" and sql == "PRAGMA journal_mode=WAL":
                raise sqlite3.OperationalError("synthetic pragma failure")
            return super().execute(sql, *args, **kwargs)

    original_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, factory=TrackedConnection, **kwargs)
        connections.append(connection)
        return connection

    def reject_migration(self):
        raise RuntimeError("synthetic migration failure")

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    if failure_stage == "migration":
        monkeypatch.setattr(EvidenceCache, "_migrate", reject_migration)

    expected_error = sqlite3.OperationalError if failure_stage == "pragma" else RuntimeError
    try:
        with pytest.raises(expected_error, match=f"synthetic {failure_stage} failure"):
            EvidenceCache(tmp_path / "rejected.sqlite3")
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connections[0].execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()
