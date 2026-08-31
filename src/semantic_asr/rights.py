from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PermissionState = Literal["allow", "deny", "review"]
Operation = Literal[
    "train",
    "derive_features",
    "redistribute_raw",
    "export_speaker_id",
]


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

    def permission(self, operation: Operation) -> PermissionState:
        return getattr(self, operation)


class RightsRegistry:
    def __init__(self, records: list[RightsRecord]) -> None:
        if len({record.asset_id for record in records}) != len(records):
            raise ValueError("rights asset IDs must be unique")
        self.records = {record.asset_id: record for record in records}

    @classmethod
    def load(cls, path: str | Path) -> RightsRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("assets") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("rights registry must contain an assets array")
        return cls(
            [
                RightsRecord(
                    asset_id=str(row["assetId"]),
                    source_name=str(row["sourceName"]),
                    source_url=str(row["sourceUrl"]),
                    version=str(row["version"]),
                    license_name=str(row["licenseName"]),
                    license_url=str(row["licenseUrl"]),
                    train=row["train"],
                    derive_features=row["deriveFeatures"],
                    redistribute_raw=row["redistributeRaw"],
                    export_speaker_id=row["exportSpeakerId"],
                    attribution=str(row["attribution"]),
                    reviewed_at=str(row["reviewedAt"]),
                    notes=str(row.get("notes", "")),
                )
                for row in rows
            ]
        )

    def require(self, asset_id: str, operation: Operation) -> RightsRecord:
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
