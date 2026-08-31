from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .adapters import ASRAdapter, DecodeRequest
from .candidate_pool import aggregate_surface_candidates
from .contracts import CandidateEvidence, canonical_json

DatasetSplit = Literal["train", "calibration", "test"]
RightsDecision = Literal["allow", "deny", "review"]


@dataclass(frozen=True, slots=True)
class AudioManifestRecord:
    sample_id: str
    group_id: str
    source_id: str
    split: DatasetSplit
    audio_path: str
    reference: str
    domain: str = "unknown"
    near_duplicate_id: str | None = None
    rights_decision: RightsDecision = "review"
    license_id: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id or not self.source_id:
            raise ValueError("sample, group, and source IDs are required")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("unknown audio-manifest split")
        if not self.audio_path or not self.reference:
            raise ValueError("audio path and reference are required")
        if self.rights_decision not in {"allow", "deny", "review"}:
            raise ValueError("unknown rights decision")


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    language: str | None = "ja"
    beam_size: int = 12
    hypotheses: int = 12
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    return_timestamps: bool = False
    fail_on_non_allow_rights: bool = True
    model_revision: str | None = None
    runtime_revision: str | None = None

    def __post_init__(self) -> None:
        if self.beam_size < 1 or self.hypotheses < 1:
            raise ValueError("beam size and hypothesis count must be positive")
        if self.hypotheses > self.beam_size:
            raise ValueError("hypothesis count cannot exceed beam size")

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratedCandidateRecord:
    sample_id: str
    group_id: str
    source_id: str
    split: DatasetSplit
    reference: str
    domain: str
    near_duplicate_id: str | None
    audio_sha256: str
    candidates: tuple[CandidateEvidence, ...]
    adapter: str
    model: str
    generation_config_sha256: str
    elapsed_ms: int
    license_id: str | None

    def __post_init__(self) -> None:
        if len(self.audio_sha256) != 64:
            raise ValueError("audio digest must be SHA-256 hex")
        if len(self.generation_config_sha256) != 64:
            raise ValueError("generation configuration digest must be SHA-256 hex")
        if not self.candidates:
            raise ValueError("generated record requires candidates")
        if self.elapsed_ms < 0:
            raise ValueError("generation elapsed time must be non-negative")

    def as_benchmark_row(self) -> dict[str, Any]:
        return {
            "sampleId": self.sample_id,
            "groupId": self.group_id,
            "sourceId": self.source_id,
            "split": self.split,
            "reference": self.reference,
            "domain": self.domain,
            "nearDuplicateId": self.near_duplicate_id,
            "audioSha256": self.audio_sha256,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "generation": {
                "adapter": self.adapter,
                "model": self.model,
                "configSha256": self.generation_config_sha256,
                "elapsedMs": self.elapsed_ms,
                "licenseId": self.license_id,
            },
        }

    def as_ranker_row(self) -> dict[str, Any]:
        return {
            "exampleId": self.sample_id,
            "groupId": self.group_id,
            "split": self.split,
            "reference": self.reference,
            "context": "",
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "audioSha256": self.audio_sha256,
        }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_isolation(records: Sequence[AudioManifestRecord]) -> None:
    if not records:
        raise ValueError("audio manifest is empty")
    seen_samples: set[str] = set()
    dimensions: dict[str, dict[str, set[str]]] = {
        "group": {},
        "source": {},
        "near-duplicate": {},
    }
    for record in records:
        if record.sample_id in seen_samples:
            raise ValueError(f"duplicate sample ID: {record.sample_id}")
        seen_samples.add(record.sample_id)
        for kind, identifier in (
            ("group", record.group_id),
            ("source", record.source_id),
            ("near-duplicate", record.near_duplicate_id),
        ):
            if not identifier:
                continue
            dimensions[kind].setdefault(identifier, set()).add(record.split)
    for kind, values in dimensions.items():
        leaking = {
            identifier: sorted(splits)
            for identifier, splits in values.items()
            if len(splits) > 1
        }
        if leaking:
            identifier = sorted(leaking)[0]
            raise ValueError(
                f"{kind} leakage across splits: {identifier} -> {leaking[identifier]}"
            )


def generate_candidates(
    record: AudioManifestRecord,
    adapter: ASRAdapter,
    *,
    config: CandidateGenerationConfig | None = None,
) -> GeneratedCandidateRecord:
    config = config or CandidateGenerationConfig()
    if config.fail_on_non_allow_rights and record.rights_decision != "allow":
        raise PermissionError(
            f"sample {record.sample_id} rights decision is {record.rights_decision!r}"
        )
    source = Path(record.audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    audio_sha256 = sha256_file(source)
    started = time.perf_counter()
    candidates = adapter.decode(
        DecodeRequest(
            audio_path=str(source),
            language=config.language,
            beam_size=config.beam_size,
            hypotheses=config.hypotheses,
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
            return_timestamps=config.return_timestamps,
        )
    )
    pooled = aggregate_surface_candidates(candidates, id_prefix="experiment")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    model = str(getattr(adapter, "model_name", adapter.name))
    annotated: list[CandidateEvidence] = []
    for candidate in pooled:
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "experimentSampleId": record.sample_id,
                "audioSha256": audio_sha256,
                "generationConfigSha256": config.digest,
                "modelRevision": config.model_revision,
                "runtimeRevision": config.runtime_revision,
            }
        )
        annotated.append(
            CandidateEvidence.from_dict({**candidate.as_dict(), "metadata": metadata})
        )
    return GeneratedCandidateRecord(
        sample_id=record.sample_id,
        group_id=record.group_id,
        source_id=record.source_id,
        split=record.split,
        reference=record.reference,
        domain=record.domain,
        near_duplicate_id=record.near_duplicate_id,
        audio_sha256=audio_sha256,
        candidates=tuple(annotated),
        adapter=adapter.name,
        model=model,
        generation_config_sha256=config.digest,
        elapsed_ms=elapsed_ms,
        license_id=record.license_id,
    )


def generate_manifest(
    records: Sequence[AudioManifestRecord],
    adapter: ASRAdapter,
    *,
    config: CandidateGenerationConfig | None = None,
) -> list[GeneratedCandidateRecord]:
    verify_manifest_isolation(records)
    return [generate_candidates(record, adapter, config=config) for record in records]


def audio_record_from_row(
    row: Mapping[str, Any], *, line_number: int = 0
) -> AudioManifestRecord:
    return AudioManifestRecord(
        sample_id=str(row.get("sampleId") or row.get("sample_id") or line_number),
        group_id=str(row.get("groupId") or row.get("group_id") or ""),
        source_id=str(row.get("sourceId") or row.get("source_id") or ""),
        split=str(row.get("split") or "train"),
        audio_path=str(row.get("audioPath") or row.get("audio_path") or ""),
        reference=str(row.get("reference") or ""),
        domain=str(row.get("domain") or "unknown"),
        near_duplicate_id=(
            str(row.get("nearDuplicateId") or row.get("near_duplicate_id"))
            if row.get("nearDuplicateId") or row.get("near_duplicate_id")
            else None
        ),
        rights_decision=str(
            row.get("rightsDecision") or row.get("rights_decision") or "review"
        ),
        license_id=(
            str(row.get("licenseId") or row.get("license_id"))
            if row.get("licenseId") or row.get("license_id")
            else None
        ),
    )


def load_audio_manifest(path: str | Path) -> list[AudioManifestRecord]:
    output: list[AudioManifestRecord] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"audio manifest row {line_number} must be an object")
        output.append(audio_record_from_row(payload, line_number=line_number))
    verify_manifest_isolation(output)
    return output


def write_generated_manifests(
    records: Iterable[GeneratedCandidateRecord],
    *,
    benchmark_path: str | Path,
    ranker_path: str | Path | None = None,
) -> None:
    rows = list(records)
    benchmark = Path(benchmark_path)
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text(
        "\n".join(
            json.dumps(record.as_benchmark_row(), ensure_ascii=False, separators=(",", ":"))
            for record in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if ranker_path is not None:
        ranker = Path(ranker_path)
        ranker.parent.mkdir(parents=True, exist_ok=True)
        ranker.write_text(
            "\n".join(
                json.dumps(record.as_ranker_row(), ensure_ascii=False, separators=(",", ":"))
                for record in rows
            )
            + ("\n" if rows else ""),
            encoding="utf-8",
        )
