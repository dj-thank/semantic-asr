"""Rights-gated, speaker/session-disjoint manifests for dual CTC training and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .contracts import PhoneticInventory

SplitName = Literal["train", "calibration", "test"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_string(value: Any, *, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"line {line_number}: {name} must be a non-empty string")
    return value


def _string_tuple(value: Any, *, name: str, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"line {line_number}: {name} must be a non-empty array")
    if any(not isinstance(row, str) or not row for row in value):
        raise TypeError(f"line {line_number}: {name} must contain non-empty strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class PhoneticManifestRow:
    utterance_id: str
    audio_path: Path
    source_audio_sha256: str
    sample_rate: int
    phone_symbols: tuple[str, ...]
    mora_symbols: tuple[str, ...]
    speaker_id: str
    session_id: str
    source_id: str
    license_id: str
    rights_decision: str
    split: SplitName
    transcript_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "utterance_id",
            "speaker_id",
            "session_id",
            "source_id",
            "license_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.rights_decision != "allow":
            raise ValueError("phonetic manifest rows require rights_decision='allow'")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("phonetic manifest split is invalid")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        if self.transcript_sha256 is not None and not _is_sha256(self.transcript_sha256):
            raise ValueError("transcript_sha256 must be a SHA-256 value")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if not self.audio_path.is_absolute():
            raise ValueError("audio_path must be absolute")
        if not self.phone_symbols or not self.mora_symbols:
            raise ValueError("phone and mora targets must be non-empty")
        if any(not value for value in (*self.phone_symbols, *self.mora_symbols)):
            raise ValueError("phone and mora symbols must not be empty")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "audio_path": str(self.audio_path),
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticSplitManifest:
    name: str
    revision: str
    rows: tuple[PhoneticManifestRow, ...]
    source_manifest_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.rows:
            raise ValueError("phonetic split manifest requires name, revision, and rows")
        if not _is_sha256(self.source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be a SHA-256 value")
        if len({row.utterance_id for row in self.rows}) != len(self.rows):
            raise ValueError("utterance IDs must be unique")
        if len({row.source_audio_sha256 for row in self.rows}) != len(self.rows):
            raise ValueError("audio hashes must be unique within a split manifest")

    def rows_for(self, split: SplitName) -> tuple[PhoneticManifestRow, ...]:
        return tuple(row for row in self.rows if row.split == split)

    def validate_inventories(
        self,
        phone_inventory: PhoneticInventory,
        mora_inventory: PhoneticInventory,
    ) -> None:
        for row in self.rows:
            phone_inventory.encode(row.phone_symbols)
            mora_inventory.encode(row.mora_symbols)
            if row.sample_rate != phone_inventory_sample_rate(self):
                raise ValueError("phonetic manifest mixes sample rates")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "rowDigests": [row.digest for row in self.rows],
                "sourceManifestSha256": self.source_manifest_sha256,
            }
        )


def phone_inventory_sample_rate(manifest: PhoneticSplitManifest) -> int:
    values = {row.sample_rate for row in manifest.rows}
    if len(values) != 1:
        raise ValueError("phonetic manifest must use one sample rate")
    return next(iter(values))


def validate_split_isolation(manifest: PhoneticSplitManifest) -> None:
    rows_by_split = {split: manifest.rows_for(split) for split in ("train", "calibration", "test")}
    if not rows_by_split["train"] or not rows_by_split["calibration"]:
        raise ValueError("phonetic manifest requires non-empty train and calibration splits")
    speakers = {split: {row.speaker_id for row in rows} for split, rows in rows_by_split.items()}
    sessions = {split: {row.session_id for row in rows} for split, rows in rows_by_split.items()}
    sources = {split: {row.source_id for row in rows} for split, rows in rows_by_split.items()}
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if speakers[left].intersection(speakers[right]):
            raise ValueError(f"speaker leakage between {left} and {right}")
        if sessions[left].intersection(sessions[right]):
            raise ValueError(f"session leakage between {left} and {right}")
        if sources[left].intersection(sources[right]):
            raise ValueError(f"source leakage between {left} and {right}")


def load_phonetic_manifest(path: str | Path) -> PhoneticSplitManifest:
    source = Path(path).resolve(strict=True)
    rows: list[PhoneticManifestRow] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"line {line_number}: manifest row must be a JSON object")
            audio_path_value = _required_string(
                payload.get("audioPath"),
                name="audioPath",
                line_number=line_number,
            )
            audio_path = Path(audio_path_value)
            if not audio_path.is_absolute():
                raise ValueError(f"line {line_number}: audioPath must be absolute")
            rows.append(
                PhoneticManifestRow(
                    utterance_id=_required_string(
                        payload.get("utteranceId"),
                        name="utteranceId",
                        line_number=line_number,
                    ),
                    audio_path=audio_path,
                    source_audio_sha256=_required_string(
                        payload.get("sourceAudioSha256"),
                        name="sourceAudioSha256",
                        line_number=line_number,
                    ),
                    sample_rate=payload.get("sampleRate"),
                    phone_symbols=_string_tuple(
                        payload.get("phoneSymbols"),
                        name="phoneSymbols",
                        line_number=line_number,
                    ),
                    mora_symbols=_string_tuple(
                        payload.get("moraSymbols"),
                        name="moraSymbols",
                        line_number=line_number,
                    ),
                    speaker_id=_required_string(
                        payload.get("speakerId"),
                        name="speakerId",
                        line_number=line_number,
                    ),
                    session_id=_required_string(
                        payload.get("sessionId"),
                        name="sessionId",
                        line_number=line_number,
                    ),
                    source_id=_required_string(
                        payload.get("sourceId"),
                        name="sourceId",
                        line_number=line_number,
                    ),
                    license_id=_required_string(
                        payload.get("licenseId"),
                        name="licenseId",
                        line_number=line_number,
                    ),
                    rights_decision=_required_string(
                        payload.get("rightsDecision"),
                        name="rightsDecision",
                        line_number=line_number,
                    ),
                    split=_required_string(
                        payload.get("split"),
                        name="split",
                        line_number=line_number,
                    ),
                    transcript_sha256=(
                        None
                        if payload.get("transcriptSha256") is None
                        else _required_string(
                            payload.get("transcriptSha256"),
                            name="transcriptSha256",
                            line_number=line_number,
                        )
                    ),
                )
            )
    if not rows:
        raise ValueError("phonetic manifest contains no rows")
    metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise ValueError("phonetic manifest metadata sidecar is required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("phonetic manifest metadata must be a JSON object")
    manifest = PhoneticSplitManifest(
        name=_required_string(metadata.get("name"), name="name", line_number=0),
        revision=_required_string(metadata.get("revision"), name="revision", line_number=0),
        rows=tuple(rows),
        source_manifest_sha256=_sha256_file(source),
    )
    expected_sha = metadata.get("manifestSha256")
    if expected_sha is not None and expected_sha != manifest.source_manifest_sha256:
        raise ValueError("phonetic manifest SHA-256 does not match metadata sidecar")
    validate_split_isolation(manifest)
    return manifest
