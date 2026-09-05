"""Canonical, tamper-checked artifacts for the dependency-free document scorer."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .ngram_scorer import (
    BidirectionalCharacterNgramScorer,
    FrozenCharacterNgramModel,
    NgramScoreNormalization,
)


def _exact_keys(value: dict[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, unknown={unknown}")


def _model_from_dict(value: dict[str, Any]) -> FrozenCharacterNgramModel:
    _exact_keys(
        value,
        {
            "order",
            "alpha",
            "vocabulary",
            "rows",
            "training_manifest_sha256",
            "revision",
            "reversed_text",
            "schema_version",
        },
        name="character n-gram model",
    )
    rows = tuple(
        (
            str(context),
            tuple((str(symbol), int(count)) for symbol, count in counts),
        )
        for context, counts in value["rows"]
    )
    return FrozenCharacterNgramModel(
        order=value["order"],
        alpha=value["alpha"],
        vocabulary=tuple(str(row) for row in value["vocabulary"]),
        rows=rows,
        training_manifest_sha256=str(value["training_manifest_sha256"]),
        revision=str(value["revision"]),
        reversed_text=value["reversed_text"],
        schema_version=str(value["schema_version"]),
    )


def _normalization_from_dict(value: dict[str, Any]) -> NgramScoreNormalization:
    _exact_keys(
        value,
        {
            "center",
            "scale",
            "calibration_manifest_sha256",
            "sample_count",
            "revision",
            "schema_version",
        },
        name="n-gram normalization",
    )
    return NgramScoreNormalization(
        center=value["center"],
        scale=value["scale"],
        calibration_manifest_sha256=str(value["calibration_manifest_sha256"]),
        sample_count=value["sample_count"],
        revision=str(value["revision"]),
        schema_version=str(value["schema_version"]),
    )


@dataclass(frozen=True, slots=True)
class BidirectionalNgramArtifact:
    forward: FrozenCharacterNgramModel
    backward: FrozenCharacterNgramModel
    normalization: NgramScoreNormalization
    name: str
    revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("n-gram artifact requires name and revision")
        BidirectionalCharacterNgramScorer(
            self.forward,
            self.backward,
            self.normalization,
            source=f"artifact:{self.name}",
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "revision": self.revision,
            "forward": asdict(self.forward),
            "backward": asdict(self.backward),
            "normalization": asdict(self.normalization),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.payload)

    def as_dict(self) -> dict[str, object]:
        return {**self.payload, "artifact_digest": self.digest}

    def scorer(self) -> BidirectionalCharacterNgramScorer:
        return BidirectionalCharacterNgramScorer(
            self.forward,
            self.backward,
            self.normalization,
            source=f"artifact:{self.name}@{self.revision}",
        )

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
                json.dump(self.as_dict(), stream, ensure_ascii=False, indent=2)
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
    def from_dict(cls, value: dict[str, Any]) -> BidirectionalNgramArtifact:
        _exact_keys(
            value,
            {
                "schema_version",
                "name",
                "revision",
                "forward",
                "backward",
                "normalization",
                "artifact_digest",
            },
            name="bidirectional n-gram artifact",
        )
        artifact_digest = str(value["artifact_digest"])
        if not _is_sha256(artifact_digest):
            raise ValueError("artifact_digest must be a SHA-256 value")
        artifact = cls(
            forward=_model_from_dict(dict(value["forward"])),
            backward=_model_from_dict(dict(value["backward"])),
            normalization=_normalization_from_dict(dict(value["normalization"])),
            name=str(value["name"]),
            revision=str(value["revision"]),
            schema_version=str(value["schema_version"]),
        )
        if artifact.digest != artifact_digest:
            raise ValueError("bidirectional n-gram artifact digest mismatch")
        return artifact

    @classmethod
    def read(cls, path: str | Path) -> BidirectionalNgramArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("bidirectional n-gram artifact must be a JSON object")
        return cls.from_dict(payload)
