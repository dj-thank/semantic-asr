from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

PermissionState = Literal["allow", "deny", "review"]
Operation = Literal[
    "train",
    "evaluate",
    "derive_features",
    "publish_text",
    "publish_features",
    "publish_weights",
    "redistribute_audio",
    # Backward-compatible aliases retained while v1 registries migrate.
    "redistribute_raw",
    "export_speaker_id",
]
_PERMISSION_STATES = frozenset(("allow", "deny", "review"))
_OPERATIONS = frozenset(
    (
        "train",
        "evaluate",
        "derive_features",
        "publish_text",
        "publish_features",
        "publish_weights",
        "redistribute_audio",
        "redistribute_raw",
        "export_speaker_id",
    )
)
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_SUPPORTED_SCHEMA_VERSIONS = frozenset(("1.0.0", "2.0.0"))
_V2_ROOT_KEYS = frozenset(("schemaVersion", "registryDigest", "assets"))
_V2_ASSET_KEYS = frozenset(
    (
        "assetId",
        "sourceName",
        "sourceUrl",
        "version",
        "licenseName",
        "licenseUrl",
        "train",
        "evaluate",
        "deriveFeatures",
        "publishText",
        "publishFeatures",
        "publishWeights",
        "redistributeAudio",
        "redistributeRaw",
        "exportSpeakerId",
        "attribution",
        "reviewedAt",
        "acquiredAt",
        "licenseTextSha256",
        "consentRecord",
        "notes",
    )
)
_MISSING = object()


def _require_state(value: object, *, name: str) -> PermissionState:
    if not isinstance(value, str) or value not in _PERMISSION_STATES:
        raise ValueError(f"{name} must be allow, deny, or review")
    return cast(PermissionState, value)


def _require_nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _require_optional_sha256(value: object, *, name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")


def _required_value(row: dict[str, object], key: str) -> object:
    if key not in row:
        raise ValueError(f"rights field {key} is required")
    return row[key]


def _required_text(row: dict[str, object], key: str) -> str:
    return _require_nonempty(_required_value(row, key), name=key)


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return _require_nonempty(value, name=key)


def _notes_field(row: dict[str, object]) -> str:
    value = row.get("notes", "")
    if not isinstance(value, str):
        raise ValueError("notes must be a string")
    return value


def _state_field(
    row: dict[str, object],
    key: str,
    *,
    default: object = _MISSING,
) -> PermissionState:
    value = _required_value(row, key) if default is _MISSING else row.get(key, default)
    return _require_state(value, name=key)


@dataclass(frozen=True, slots=True)
class RightsRecord:
    asset_id: str
    source_name: str
    source_url: str
    version: str
    license_name: str
    license_url: str
    train: PermissionState
    derive_features: PermissionState
    redistribute_raw: PermissionState
    export_speaker_id: PermissionState
    attribution: str
    reviewed_at: str
    notes: str = ""
    evaluate: PermissionState = "review"
    publish_text: PermissionState = "review"
    publish_features: PermissionState = "review"
    publish_weights: PermissionState = "review"
    redistribute_audio: PermissionState | None = None
    acquired_at: str | None = None
    license_text_sha256: str | None = None
    consent_record: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "asset_id",
            "source_name",
            "source_url",
            "version",
            "license_name",
            "license_url",
            "attribution",
            "reviewed_at",
        ):
            _require_nonempty(getattr(self, name), name=name)
        for name in (
            "train",
            "evaluate",
            "derive_features",
            "publish_text",
            "publish_features",
            "publish_weights",
            "redistribute_raw",
            "export_speaker_id",
        ):
            _require_state(getattr(self, name), name=name)
        if self.redistribute_audio is None:
            # v1's redistributeRaw covered the raw audio payload. Preserve that
            # explicit decision rather than inventing a broader permission.
            object.__setattr__(self, "redistribute_audio", self.redistribute_raw)
        else:
            _require_state(self.redistribute_audio, name="redistribute_audio")
        if self.acquired_at is None:
            object.__setattr__(self, "acquired_at", self.reviewed_at)
        else:
            _require_nonempty(self.acquired_at, name="acquired_at")
        _require_optional_sha256(self.license_text_sha256, name="license_text_sha256")
        if self.consent_record is not None:
            _require_nonempty(self.consent_record, name="consent_record")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

    def permission(self, operation: Operation) -> PermissionState:
        if operation not in _OPERATIONS:
            raise ValueError(f"unknown rights operation: {operation!r}")
        if operation == "redistribute_raw":
            return self.redistribute_raw
        value = getattr(self, operation)
        return _require_state(value, name=operation)

    def as_dict(self) -> dict[str, object]:
        return {
            "assetId": self.asset_id,
            "sourceName": self.source_name,
            "sourceUrl": self.source_url,
            "version": self.version,
            "licenseName": self.license_name,
            "licenseUrl": self.license_url,
            "train": self.train,
            "evaluate": self.evaluate,
            "deriveFeatures": self.derive_features,
            "publishText": self.publish_text,
            "publishFeatures": self.publish_features,
            "publishWeights": self.publish_weights,
            "redistributeAudio": self.redistribute_audio,
            "redistributeRaw": self.redistribute_raw,
            "exportSpeakerId": self.export_speaker_id,
            "attribution": self.attribution,
            "reviewedAt": self.reviewed_at,
            "acquiredAt": self.acquired_at,
            "licenseTextSha256": self.license_text_sha256,
            "consentRecord": self.consent_record,
            "notes": self.notes,
        }


class RightsRegistry:
    def __init__(self, records: list[RightsRecord]) -> None:
        if not records:
            raise ValueError("rights registry must contain at least one asset")
        if len({record.asset_id for record in records}) != len(records):
            raise ValueError("rights asset IDs must be unique")
        self.records = {record.asset_id: record for record in records}

    @classmethod
    def load(cls, path: str | Path) -> RightsRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("rights registry must contain a JSON object")
        raw_schema_version = payload.get("schemaVersion")
        if raw_schema_version is None:
            schema_version = "1.0.0"
        elif not isinstance(raw_schema_version, str):
            raise ValueError("schemaVersion must be a string")
        else:
            schema_version = raw_schema_version
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported rights registry schemaVersion: {schema_version!r}")
        strict_v2 = schema_version == "2.0.0"
        if strict_v2:
            unknown_root = sorted(set(payload) - _V2_ROOT_KEYS)
            if unknown_root:
                raise ValueError(f"unknown rights registry field(s): {', '.join(unknown_root)}")

        rows = payload.get("assets")
        if not isinstance(rows, list):
            raise ValueError("rights registry must contain an assets array")
        records: list[RightsRecord] = []
        for index, raw_row in enumerate(rows):
            if not isinstance(raw_row, dict):
                raise ValueError("rights registry assets must be objects")
            row: dict[str, object] = raw_row
            if strict_v2:
                unknown_fields = sorted(set(row) - _V2_ASSET_KEYS)
                if unknown_fields:
                    raise ValueError(
                        "unknown rights asset field(s) at index "
                        f"{index}: {', '.join(unknown_fields)}"
                    )
            raw_permission = _state_field(row, "redistributeRaw")
            records.append(
                RightsRecord(
                    asset_id=_required_text(row, "assetId"),
                    source_name=_required_text(row, "sourceName"),
                    source_url=_required_text(row, "sourceUrl"),
                    version=_required_text(row, "version"),
                    license_name=_required_text(row, "licenseName"),
                    license_url=_required_text(row, "licenseUrl"),
                    train=_state_field(row, "train"),
                    derive_features=_state_field(row, "deriveFeatures"),
                    redistribute_raw=raw_permission,
                    export_speaker_id=_state_field(row, "exportSpeakerId"),
                    attribution=_required_text(row, "attribution"),
                    reviewed_at=_required_text(row, "reviewedAt"),
                    notes=_notes_field(row),
                    evaluate=_state_field(
                        row,
                        "evaluate",
                        default=_MISSING if strict_v2 else "review",
                    ),
                    publish_text=_state_field(
                        row,
                        "publishText",
                        default=_MISSING if strict_v2 else "review",
                    ),
                    publish_features=_state_field(
                        row,
                        "publishFeatures",
                        default=_MISSING if strict_v2 else "review",
                    ),
                    publish_weights=_state_field(
                        row,
                        "publishWeights",
                        default=_MISSING if strict_v2 else "review",
                    ),
                    redistribute_audio=_state_field(
                        row,
                        "redistributeAudio",
                        default=_MISSING if strict_v2 else raw_permission,
                    ),
                    acquired_at=(
                        _required_text(row, "acquiredAt")
                        if strict_v2
                        else _optional_text(row, "acquiredAt")
                    ),
                    license_text_sha256=_optional_text(row, "licenseTextSha256"),
                    consent_record=_optional_text(row, "consentRecord"),
                )
            )
        registry = cls(records)
        declared_digest = payload.get("registryDigest")
        if declared_digest is not None:
            _require_optional_sha256(declared_digest, name="registryDigest")
            if declared_digest != registry.digest:
                raise ValueError("rights registry digest does not match its asset contents")
        return registry

    @property
    def digest(self) -> str:
        payload = {
            "schemaVersion": "2.0.0",
            "assets": [self.records[asset_id].as_dict() for asset_id in sorted(self.records)],
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": "2.0.0",
            "registryDigest": self.digest,
            "assets": [self.records[asset_id].as_dict() for asset_id in sorted(self.records)],
        }

    def require(self, asset_id: str, operation: Operation) -> RightsRecord:
        if operation not in _OPERATIONS:
            raise ValueError(f"unknown rights operation: {operation!r}")
        record = self.records.get(asset_id)
        if record is None:
            raise PermissionError(f"unknown rights asset: {asset_id}")
        state = record.permission(operation)
        if state != "allow":
            raise PermissionError(f"operation {operation} is {state} for asset {asset_id}")
        return record


def pseudonymize_speaker(identifier: str, secret: bytes) -> str:
    if len(secret) < 16:
        raise ValueError("speaker pseudonymization secret must be at least 16 bytes")
    digest = hmac.new(secret, identifier.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"speaker-{digest[:24]}"
