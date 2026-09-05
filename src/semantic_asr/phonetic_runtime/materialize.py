"""Rights-gated materialization of Japanese reading labels and source-audio manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..contracts import sha256_json
from .audio import sha256_file
from .japanese_labels import JapanesePhoneticLabelProfile
from .manifest import (
    PhoneticManifestRow,
    PhoneticSplitManifest,
    validate_split_isolation,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_string(value: Any, *, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"line {line_number}: {name} must be a non-empty string")
    return value


def _inventory_payload(inventory) -> dict[str, object]:
    return {
        "kind": inventory.kind,
        "symbols": list(inventory.symbols),
        "blankSymbol": inventory.blank_symbol,
        "unknownSymbol": inventory.unknown_symbol,
        "language": inventory.language,
        "revision": inventory.revision,
    }


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"materialization destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _read_input(path: Path) -> tuple[dict[str, Any], ...]:
    required = {
        "utteranceId",
        "audioPath",
        "reading",
        "speakerId",
        "sessionId",
        "sourceId",
        "licenseId",
        "rightsDecision",
        "split",
    }
    optional = {"transcript"}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"line {line_number}: input row must be a JSON object")
            missing = required - set(payload)
            unknown = set(payload) - required - optional
            if missing or unknown:
                raise ValueError(
                    f"line {line_number}: input keys mismatch; "
                    f"missing={sorted(missing)}, unknown={sorted(unknown)}"
                )
            normalized = dict(payload)
            for name in required:
                normalized[name] = _strict_string(
                    payload[name],
                    name=name,
                    line_number=line_number,
                )
            if payload.get("transcript") is not None:
                normalized["transcript"] = _strict_string(
                    payload["transcript"],
                    name="transcript",
                    line_number=line_number,
                )
            rows.append(normalized)
    if not rows:
        raise ValueError("phonetic materialization input contains no rows")
    return tuple(rows)


def _wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("phonetic materialization requires uncompressed PCM16 WAV")
        if wav.getnchannels() not in {1, 2}:
            raise ValueError("phonetic materialization supports mono or stereo WAV")
        if wav.getnframes() < 1:
            raise ValueError("phonetic materialization input WAV is empty")
        return wav.getframerate()


@dataclass(frozen=True, slots=True)
class JapanesePhoneticMaterialization:
    output_directory: Path
    manifest_path: Path
    metadata_path: Path
    phone_inventory_path: Path
    mora_inventory_path: Path
    manifest: PhoneticSplitManifest
    label_profile_digest: str
    input_manifest_sha256: str
    phone_inventory_digest: str
    mora_inventory_digest: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.output_directory.is_absolute():
            raise ValueError("materialization output directory must be absolute")
        for path in (
            self.manifest_path,
            self.metadata_path,
            self.phone_inventory_path,
            self.mora_inventory_path,
        ):
            if path.parent != self.output_directory:
                raise ValueError("materialization artifacts must share one output directory")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "manifestDigest": self.manifest.digest,
                "labelProfileDigest": self.label_profile_digest,
                "inputManifestSha256": self.input_manifest_sha256,
                "phoneInventoryDigest": self.phone_inventory_digest,
                "moraInventoryDigest": self.mora_inventory_digest,
                "manifestSha256": _sha256_file(self.manifest_path),
                "metadataSha256": _sha256_file(self.metadata_path),
                "phoneInventorySha256": _sha256_file(self.phone_inventory_path),
                "moraInventorySha256": _sha256_file(self.mora_inventory_path),
            }
        )


def materialize_japanese_phonetic_manifest(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    name: str,
    revision: str,
    profile: JapanesePhoneticLabelProfile | None = None,
) -> JapanesePhoneticMaterialization:
    source = Path(input_path).resolve(strict=True)
    destination = Path(output_directory).resolve()
    repository = Path(__file__).resolve().parents[3]
    if destination == repository or repository in destination.parents:
        raise ValueError("phonetic materialization output must be outside the repository checkout")
    if destination.exists():
        raise FileExistsError(f"materialization output already exists: {destination}")
    if not name or not revision:
        raise ValueError("materialization name and revision are required")
    profile = profile or JapanesePhoneticLabelProfile()
    phone_inventory = profile.phone_inventory()
    mora_inventory = profile.mora_inventory()
    input_rows = _read_input(source)
    rows: list[PhoneticManifestRow] = []
    serialized_rows: list[dict[str, object]] = []
    for input_row in input_rows:
        if input_row["rightsDecision"] != "allow":
            raise ValueError("all materialized rows require rightsDecision='allow'")
        audio_path = Path(input_row["audioPath"])
        if not audio_path.is_absolute():
            raise ValueError("audioPath must be absolute")
        audio_path = audio_path.resolve(strict=True)
        if not audio_path.is_file():
            raise ValueError("audioPath must identify a regular file")
        labels = profile.label(input_row["reading"])
        audio_sha256 = sha256_file(audio_path)
        sample_rate = _wav_sample_rate(audio_path)
        transcript = input_row.get("transcript")
        transcript_sha256 = (
            None if transcript is None else hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        )
        row = PhoneticManifestRow(
            utterance_id=input_row["utteranceId"],
            audio_path=audio_path,
            source_audio_sha256=audio_sha256,
            sample_rate=sample_rate,
            phone_symbols=labels.phones,
            mora_symbols=labels.moras,
            speaker_id=input_row["speakerId"],
            session_id=input_row["sessionId"],
            source_id=input_row["sourceId"],
            license_id=input_row["licenseId"],
            rights_decision="allow",
            split=input_row["split"],
            transcript_sha256=transcript_sha256,
        )
        rows.append(row)
        serialized_rows.append(
            {
                "utteranceId": row.utterance_id,
                "audioPath": str(row.audio_path),
                "sourceAudioSha256": row.source_audio_sha256,
                "sampleRate": row.sample_rate,
                "phoneSymbols": list(row.phone_symbols),
                "moraSymbols": list(row.mora_symbols),
                "speakerId": row.speaker_id,
                "sessionId": row.session_id,
                "sourceId": row.source_id,
                "licenseId": row.license_id,
                "rightsDecision": row.rights_decision,
                "split": row.split,
                **(
                    {"transcriptSha256": transcript_sha256} if transcript_sha256 is not None else {}
                ),
            }
        )
    provisional = PhoneticSplitManifest(
        name=name,
        revision=revision,
        rows=tuple(rows),
        source_manifest_sha256="0" * 64,
    )
    validate_split_isolation(provisional)
    provisional.validate_inventories(phone_inventory, mora_inventory)

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    try:
        manifest_path = temporary / "manifest.jsonl"
        phone_path = temporary / "phone_inventory.json"
        mora_path = temporary / "mora_inventory.json"
        manifest_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in serialized_rows
        )
        manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
        manifest_sha256 = _sha256_file(manifest_path)
        phone_path.write_text(
            json.dumps(
                _inventory_payload(phone_inventory), ensure_ascii=False, sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        mora_path.write_text(
            json.dumps(
                _inventory_payload(mora_inventory), ensure_ascii=False, sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        metadata_path = temporary / "manifest.jsonl.metadata.json"
        metadata_payload = {
            "name": name,
            "revision": revision,
            "manifestSha256": manifest_sha256,
            "inputManifestSha256": _sha256_file(source),
            "labelProfile": asdict(profile),
            "labelProfileDigest": profile.digest,
            "mappingDigest": profile.mapping_digest,
            "phoneInventoryDigest": phone_inventory.digest,
            "moraInventoryDigest": mora_inventory.digest,
            "rowDigests": [row.digest for row in rows],
            "rawTranscriptStored": False,
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for path in (manifest_path, phone_path, mora_path, metadata_path):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    manifest_path = destination / "manifest.jsonl"
    metadata_path = destination / "manifest.jsonl.metadata.json"
    phone_path = destination / "phone_inventory.json"
    mora_path = destination / "mora_inventory.json"
    manifest = PhoneticSplitManifest(
        name=name,
        revision=revision,
        rows=tuple(rows),
        source_manifest_sha256=_sha256_file(manifest_path),
    )
    return JapanesePhoneticMaterialization(
        output_directory=destination,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        phone_inventory_path=phone_path,
        mora_inventory_path=mora_path,
        manifest=manifest,
        label_profile_digest=profile.digest,
        input_manifest_sha256=_sha256_file(source),
        phone_inventory_digest=phone_inventory.digest,
        mora_inventory_digest=mora_inventory.digest,
    )
