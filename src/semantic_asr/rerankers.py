from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .contracts import CandidateEvidence
from .mbr import critical_units
from .revisions import resolve_hugging_face_revision, verify_artifact_sha256


class CandidateRanker(Protocol):
    name: str

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]: ...


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


FEATURE_ALIASES: dict[str, str] = {
    "rank_fraction": "relative_rank",
    "logprob": "avg_logprob",
    "length": "log_text_length",
    "critical_count": "critical_unit_count",
}


FEATURE_NAMES: tuple[str, ...] = (
    "acoustic",
    "mora",
    "lexical",
    "preservation",
    "cross_model",
    "avg_logprob",
    "sequence_score",
    "reciprocal_rank",
    "relative_rank",
    "log_path_count",
    "source_diversity",
    "log_text_length",
    "critical_unit_count",
    "context_overlap",
)


def _finite(value: float | None, default: float = 0.0) -> float:
    if value is None or not math.isfinite(float(value)):
        return default
    return float(value)


def _resolve_ranker_identity(
    model: str,
    model_revision: str | None,
    model_artifact_sha256: str | None,
    artifact_sha256: str | None,
) -> tuple[str | None, str | None]:
    """Resolve an immutable remote revision or verify a local ranker artifact.

    A ranker is loaded by the user-facing v0.2 transcriber, so an unpinned Hub
    identifier would make both generation and its cache non-reproducible.  Existing
    local directories/files are accepted only with a digest of the exact bytes that
    the loader will consume; the two identity forms are intentionally exclusive.
    """

    if (
        model_artifact_sha256 is not None
        and artifact_sha256 is not None
        and model_artifact_sha256.lower() != artifact_sha256.lower()
    ):
        raise ValueError("model_artifact_sha256 and artifact_sha256 disagree")
    supplied_artifact = model_artifact_sha256 or artifact_sha256
    local = Path(model).expanduser()
    is_local = local.is_file() or local.is_dir()
    if is_local:
        if model_revision is not None:
            raise ValueError(
                "a local ranker artifact cannot claim a Hub revision; provide its verified SHA-256"
            )
        if supplied_artifact is None:
            raise ValueError("a verified local ranker artifact SHA-256 is required")
        return None, verify_artifact_sha256(
            local,
            supplied_artifact,
            identifier="local ranker artifact",
        )
    if supplied_artifact is not None:
        raise ValueError("ranker artifact SHA-256 is only valid for a local model")
    return (
        resolve_hugging_face_revision(model, model_revision, {}),
        None,
    )


def _character_ngrams(text: str, size: int = 2) -> set[str]:
    compact = "".join(str(text or "").split())
    if not compact:
        return set()
    if len(compact) < size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def context_overlap(text: str, context: str) -> float:
    left = _character_ngrams(text)
    right = _character_ngrams(context)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def candidate_features(
    candidate: CandidateEvidence,
    *,
    context: str = "",
) -> dict[str, float]:
    hypothesis_count = candidate.hypothesis_count or max(1, candidate.rank or 1)
    rank = candidate.rank or hypothesis_count
    path_count = max(1, int(candidate.metadata.get("pathCount", 1)))
    return {
        "acoustic": _finite(candidate.acoustic),
        "mora": _finite(candidate.mora),
        "lexical": _finite(candidate.lexical),
        "preservation": _finite(candidate.preservation),
        "cross_model": _finite(candidate.cross_model),
        "avg_logprob": _finite(candidate.avg_logprob),
        "sequence_score": _finite(candidate.sequence_score),
        "reciprocal_rank": 1.0 / max(1, rank),
        "relative_rank": (
            1.0
            if hypothesis_count <= 1
            else (hypothesis_count - rank) / max(1, hypothesis_count - 1)
        ),
        "log_path_count": math.log1p(path_count),
        "source_diversity": float(len(candidate.source_support)),
        "log_text_length": math.log1p(len(candidate.text)),
        "critical_unit_count": float(len(critical_units(candidate.text))),
        "context_overlap": context_overlap(candidate.text, context),
    }


@dataclass(frozen=True, slots=True)
class LinearRankerProfile:
    name: str
    weights: dict[str, float]
    bias: float = 0.0
    feature_mean: dict[str, float] | None = None
    feature_scale: dict[str, float] | None = None
    training_manifest_sha256: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ranker profile name is required")
        unknown = set(self.weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown ranker features: {sorted(unknown)}")
        values = [self.bias, *self.weights.values()]
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("ranker weights must be finite")
        if self.feature_scale is not None:
            for name, value in self.feature_scale.items():
                if name not in FEATURE_NAMES or not math.isfinite(value) or value <= 0:
                    raise ValueError("feature scales must be positive finite known features")
        if self.feature_mean is not None:
            for name, value in self.feature_mean.items():
                if name not in FEATURE_NAMES or not math.isfinite(value):
                    raise ValueError("feature means must be finite known features")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "digest": self.digest}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> LinearRankerProfile:
        values = dict(row)
        values.pop("digest", None)

        def canonical_features(raw: Mapping[str, Any]) -> dict[str, float]:
            output: dict[str, float] = {}
            for raw_name, raw_value in raw.items():
                name = FEATURE_ALIASES.get(str(raw_name), str(raw_name))
                if name in output:
                    raise ValueError(f"duplicate ranker feature after aliasing: {name}")
                output[name] = float(raw_value)
            return output

        values["weights"] = canonical_features(dict(values["weights"]))
        if values.get("feature_mean") is not None:
            values["feature_mean"] = canonical_features(dict(values["feature_mean"]))
        if values.get("feature_scale") is not None:
            values["feature_scale"] = canonical_features(dict(values["feature_scale"]))
        return cls(**values)


class LinearCandidateRanker:
    def __init__(self, profile: LinearRankerProfile) -> None:
        self.profile = profile
        self.name = f"linear:{profile.name}:{profile.digest[:12]}"
        self.model_name = profile.name
        self.config_digest = profile.digest

    def _score_one(self, candidate: CandidateEvidence, *, context: str) -> float:
        features = candidate_features(candidate, context=context)
        score = float(self.profile.bias)
        means = self.profile.feature_mean or {}
        scales = self.profile.feature_scale or {}
        for name, weight in self.profile.weights.items():
            normalized = (features[name] - means.get(name, 0.0)) / scales.get(name, 1.0)
            score += float(weight) * normalized
        return score

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        augmented_context = "\n".join(
            value for value in (context, consensus, contradiction) if value
        )
        return {
            candidate.candidate_id: self._score_one(candidate, context=augmented_context)
            for candidate in candidates
        }


class StaticCandidateRanker:
    """Deterministic fixture ranker used by tests and integrations."""

    name = "static"

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = {str(key): float(value) for key, value in scores.items()}

    def score(self, candidates: Sequence[CandidateEvidence], **_: Any) -> Mapping[str, float]:
        return {
            candidate.candidate_id: self.scores[candidate.candidate_id] for candidate in candidates
        }


class CrossEncoderCandidateRanker:
    """Optional raw-logit CrossEncoder ranker.

    The constructor uses an Identity activation so pairwise/listwise logits are
    not collapsed by a sigmoid before held-out calibration.
    """

    def __init__(
        self,
        model: str,
        *,
        model_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        artifact_sha256: str | None = None,
        runtime_revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int | None = None,
        instruction: str = (
            "日本語ASRの候補を、文法的な自然さだけでなく、"
            "与えられた文脈と既存の音響候補集合に整合する順に評価してください。"
        ),
    ) -> None:
        model = str(model)
        model_revision, model_artifact_sha256 = _resolve_ranker_identity(
            model,
            model_revision,
            model_artifact_sha256,
            artifact_sha256,
        )
        try:
            import torch.nn as nn
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install semantic-asr with the 'rerank' extra") from exc
        kwargs: dict[str, Any] = {
            "device": device,
            "num_labels": 1,
            "activation_fn": nn.Identity(),
        }
        if model_revision is not None:
            kwargs["revision"] = model_revision
        if max_length is not None:
            kwargs["max_length"] = int(max_length)
        self.model_name = model
        self.name = f"cross-encoder:{model}"
        self.batch_size = int(batch_size)
        self.instruction = instruction
        self.model_revision = model_revision
        self.model_artifact_sha256 = model_artifact_sha256
        self.runtime_revision = (
            None if runtime_revision is None else str(runtime_revision).strip() or None
        )
        self.config_digest = _digest(
            {
                "schema": "semantic-asr-ranker-config-v1",
                "backend": "cross-encoder",
                "model": self.model_name,
                "modelRevision": self.model_revision,
                "modelArtifactSha256": self.model_artifact_sha256,
                "runtimeRevision": self.runtime_revision,
                "device": str(device),
                "batchSize": self.batch_size,
                "maxLength": None if max_length is None else int(max_length),
                "instruction": self.instruction,
            }
        )
        self.model = CrossEncoder(model, **kwargs)

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        query = "\n".join(
            [
                self.instruction,
                f"文脈: {context}",
                f"固定済み共通部分: {consensus}",
                f"矛盾部分: {contradiction}",
            ]
        )
        pairs = [(query, candidate.text) for candidate in candidates]
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = raw.reshape(-1).tolist() if hasattr(raw, "reshape") else list(raw)
        if len(values) != len(candidates):
            raise RuntimeError("CrossEncoder returned a mismatched score count")
        return {
            candidate.candidate_id: float(value)
            for candidate, value in zip(candidates, values, strict=True)
        }


class Qwen3CandidateRanker:
    """Optional Qwen3-Reranker adapter returning the raw yes-vs-no logit margin."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        model_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        artifact_sha256: str | None = None,
        runtime_revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "auto",
        batch_size: int = 8,
        max_length: int = 8192,
        instruction: str = (
            "Given a Japanese speech-recognition context and one hypothesis, "
            "judge whether the hypothesis is the best acoustically plausible candidate. "
            "Do not reward grammar correction unsupported by the candidate evidence."
        ),
    ) -> None:
        model = str(model)
        model_revision, model_artifact_sha256 = _resolve_ranker_identity(
            model,
            model_revision,
            model_artifact_sha256,
            artifact_sha256,
        )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install semantic-asr with the 'rerank' extra") from exc
        self._torch = torch
        self.model_name = model
        self.name = f"qwen3-reranker:{model}"
        self.model_revision = model_revision
        self.model_artifact_sha256 = model_artifact_sha256
        self.runtime_revision = (
            None if runtime_revision is None else str(runtime_revision).strip() or None
        )
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.instruction = instruction
        self.device_map = str(device_map)
        self.dtype = str(dtype)
        revision_kwargs = {} if model_revision is None else {"revision": model_revision}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            padding_side="left",
            **revision_kwargs,
        )
        model_kwargs: dict[str, Any] = {"device_map": self.device_map, **revision_kwargs}
        if dtype != "auto":
            aliases = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if dtype.lower() not in aliases:
                raise ValueError(f"unsupported dtype: {dtype}")
            model_kwargs["torch_dtype"] = aliases[dtype.lower()]
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs).eval()
        self.config_digest = _digest(
            {
                "schema": "semantic-asr-ranker-config-v1",
                "backend": "qwen3",
                "model": self.model_name,
                "modelRevision": self.model_revision,
                "modelArtifactSha256": self.model_artifact_sha256,
                "runtimeRevision": self.runtime_revision,
                "deviceMap": self.device_map,
                "dtype": self.dtype,
                "batchSize": self.batch_size,
                "maxLength": self.max_length,
                "instruction": self.instruction,
            }
        )
        self._yes_id = self._single_token_id("yes")
        self._no_id = self._single_token_id("no")

    def _single_token_id(self, value: str) -> int:
        token_ids = self.tokenizer.encode(value, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"reranker tokenizer cannot encode {value!r}")
        return int(token_ids[-1])

    def _prompt(
        self,
        candidate: CandidateEvidence,
        *,
        context: str,
        consensus: str,
        contradiction: str,
    ) -> str:
        return (
            "<|im_start|>system\n"
            "Judge whether the candidate satisfies the instruction. "
            "Answer only yes or no.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {self.instruction}\n"
            f"<Context>: {context}\n"
            f"<Consensus>: {consensus}\n"
            f"<Contradiction>: {contradiction}\n"
            f"<Document>: {candidate.text}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        output: dict[str, float] = {}
        torch = self._torch
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            prompts = [
                self._prompt(
                    candidate,
                    context=context,
                    consensus=consensus,
                    contradiction=contradiction,
                )
                for candidate in batch
            ]
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits[:, -1, :].float()
            margins = logits[:, self._yes_id] - logits[:, self._no_id]
            for candidate, value in zip(batch, margins.tolist(), strict=True):
                output[candidate.candidate_id] = float(value)
        return output


def apply_reranker_scores(
    candidates: Sequence[CandidateEvidence],
    ranker: CandidateRanker,
    *,
    context: str = "",
    consensus: str = "",
    contradiction: str = "",
) -> list[CandidateEvidence]:
    scores = dict(
        ranker.score(
            candidates,
            context=context,
            consensus=consensus,
            contradiction=contradiction,
        )
    )
    identifiers = {candidate.candidate_id for candidate in candidates}
    if set(scores) != identifiers:
        raise ValueError("ranker must return exactly one score for every candidate ID")
    output: list[CandidateEvidence] = []
    for candidate in candidates:
        score = float(scores[candidate.candidate_id])
        if not math.isfinite(score):
            raise ValueError("ranker scores must be finite")
        metadata = dict(candidate.metadata)
        score_rows = list(metadata.get("evidenceScores", []))
        score_rows.append(
            {
                "source": ranker.name,
                "kind": "logit",
                "value": score,
                "calibrated": False,
            }
        )
        metadata["evidenceScores"] = score_rows
        metadata["rerankerSource"] = ranker.name
        metadata["rerankerRawLogit"] = score
        metadata["rerankerModel"] = getattr(ranker, "model_name", None)
        metadata["rerankerModelRevision"] = getattr(ranker, "model_revision", None)
        metadata["rerankerModelArtifactSha256"] = getattr(
            ranker,
            "model_artifact_sha256",
            None,
        )
        metadata["rerankerRuntimeRevision"] = getattr(ranker, "runtime_revision", None)
        metadata["rerankerConfigDigest"] = getattr(
            ranker,
            "config_digest",
            getattr(ranker, "configuration_digest", None),
        )
        output.append(replace(candidate, metadata=metadata))
    return output
