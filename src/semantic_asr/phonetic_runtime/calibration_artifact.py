"""Canonical phone/mora utility profiles bound to one frozen dual CTC runtime."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..contracts import sha256_json
from ..deliberation_evidence import UtilityCalibrationProfile, _is_sha256
from ..score_semantics import ScoreKind
from .calibration import CTCUtilityCalibrationReport


def _exact_keys(value: dict[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _profile_from_dict(value: dict[str, Any]) -> UtilityCalibrationProfile:
    _exact_keys(
        value,
        {
            "channel",
            "score_source",
            "score_kind",
            "center",
            "scale",
            "fitted_manifest_sha256",
            "revision",
            "higher_is_better",
            "schema_version",
        },
        name="utility calibration profile",
    )
    return UtilityCalibrationProfile(
        channel=value["channel"],
        score_source=value["score_source"],
        score_kind=ScoreKind(value["score_kind"]),
        center=value["center"],
        scale=value["scale"],
        fitted_manifest_sha256=value["fitted_manifest_sha256"],
        revision=value["revision"],
        higher_is_better=value["higher_is_better"],
        schema_version=value["schema_version"],
    )


@dataclass(frozen=True, slots=True)
class DualCTCUtilityArtifact:
    name: str
    revision: str
    runtime_profile_digest: str
    held_out_manifest_sha256: str
    phone_profile: UtilityCalibrationProfile
    mora_profile: UtilityCalibrationProfile
    phone_pairwise_accuracy: float
    mora_pairwise_accuracy: float
    phone_example_digests: tuple[str, ...]
    mora_example_digests: tuple[str, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("utility artifact requires name and revision")
        if not _is_sha256(self.runtime_profile_digest):
            raise ValueError("runtime_profile_digest must be a SHA-256 value")
        if not _is_sha256(self.held_out_manifest_sha256):
            raise ValueError("held_out_manifest_sha256 must be a SHA-256 value")
        if self.phone_profile.channel != "phone" or self.mora_profile.channel != "mora":
            raise ValueError("utility profiles are assigned to the wrong channels")
        for profile in (self.phone_profile, self.mora_profile):
            if profile.fitted_manifest_sha256 != self.held_out_manifest_sha256:
                raise ValueError("utility profile belongs to a different held-out manifest")
        for name in ("phone_pairwise_accuracy", "mora_pairwise_accuracy"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        if not self.phone_example_digests or not self.mora_example_digests:
            raise ValueError("utility artifact requires phone and mora calibration examples")
        if any(
            not _is_sha256(value)
            for value in (*self.phone_example_digests, *self.mora_example_digests)
        ):
            raise ValueError("calibration example digests must be SHA-256 values")

    @classmethod
    def from_reports(
        cls,
        phone: CTCUtilityCalibrationReport,
        mora: CTCUtilityCalibrationReport,
        *,
        name: str,
        revision: str,
        runtime_profile_digest: str,
    ) -> DualCTCUtilityArtifact:
        if phone.held_out_manifest_sha256 != mora.held_out_manifest_sha256:
            raise ValueError("phone and mora reports use different held-out manifests")
        return cls(
            name=name,
            revision=revision,
            runtime_profile_digest=runtime_profile_digest,
            held_out_manifest_sha256=phone.held_out_manifest_sha256,
            phone_profile=phone.profile,
            mora_profile=mora.profile,
            phone_pairwise_accuracy=phone.pairwise_accuracy,
            mora_pairwise_accuracy=mora.pairwise_accuracy,
            phone_example_digests=phone.example_digests,
            mora_example_digests=mora.example_digests,
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "name": self.name,
            "revision": self.revision,
            "runtimeProfileDigest": self.runtime_profile_digest,
            "heldOutManifestSha256": self.held_out_manifest_sha256,
            "phoneProfile": {
                **asdict(self.phone_profile),
                "score_kind": self.phone_profile.score_kind.value,
            },
            "moraProfile": {
                **asdict(self.mora_profile),
                "score_kind": self.mora_profile.score_kind.value,
            },
            "phonePairwiseAccuracy": self.phone_pairwise_accuracy,
            "moraPairwiseAccuracy": self.mora_pairwise_accuracy,
            "phoneExampleDigests": self.phone_example_digests,
            "moraExampleDigests": self.mora_example_digests,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.payload)

    def as_dict(self) -> dict[str, object]:
        return {**self.payload, "artifactDigest": self.digest}

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.as_dict(), stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return destination

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DualCTCUtilityArtifact:
        _exact_keys(
            value,
            {
                "schemaVersion",
                "name",
                "revision",
                "runtimeProfileDigest",
                "heldOutManifestSha256",
                "phoneProfile",
                "moraProfile",
                "phonePairwiseAccuracy",
                "moraPairwiseAccuracy",
                "phoneExampleDigests",
                "moraExampleDigests",
                "artifactDigest",
            },
            name="dual CTC utility artifact",
        )
        phone_value = value["phoneProfile"]
        mora_value = value["moraProfile"]
        if not isinstance(phone_value, dict) or not isinstance(mora_value, dict):
            raise TypeError("utility profiles must be objects")
        artifact = cls(
            name=value["name"],
            revision=value["revision"],
            runtime_profile_digest=value["runtimeProfileDigest"],
            held_out_manifest_sha256=value["heldOutManifestSha256"],
            phone_profile=_profile_from_dict(phone_value),
            mora_profile=_profile_from_dict(mora_value),
            phone_pairwise_accuracy=value["phonePairwiseAccuracy"],
            mora_pairwise_accuracy=value["moraPairwiseAccuracy"],
            phone_example_digests=tuple(value["phoneExampleDigests"]),
            mora_example_digests=tuple(value["moraExampleDigests"]),
            schema_version=value["schemaVersion"],
        )
        if value["artifactDigest"] != artifact.digest:
            raise ValueError("dual CTC utility artifact digest mismatch")
        return artifact

    @classmethod
    def read(cls, path: str | Path) -> DualCTCUtilityArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("dual CTC utility artifact must be a JSON object")
        return cls.from_dict(payload)
