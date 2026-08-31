from __future__ import annotations

import ipaddress
import json
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from .score_types import EvidenceScore, ScoreSemantics


@dataclass(frozen=True, slots=True)
class TextCandidate:
    candidate_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text:
            raise ValueError("candidate ID and text are required")


@dataclass(frozen=True, slots=True)
class SequenceScoreResult:
    candidate_id: str
    cumulative: EvidenceScore
    average: EvidenceScore
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count < 1:
            raise ValueError("token_count must be positive")


class SequenceScorer(Protocol):
    name: str

    def score(self, candidates: list[TextCandidate], *, context: str = "") -> list[SequenceScoreResult]: ...


@dataclass(frozen=True, slots=True)
class CausalScoringConfig:
    model_name: str
    model_revision: str | None = None
    add_bos: bool = True
    batch_size: int = 8
    maximum_candidate_tokens: int = 512
    length_normalization_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name is required")
        if self.batch_size < 1 or self.maximum_candidate_tokens < 1:
            raise ValueError("batch_size and maximum_candidate_tokens must be positive")
        if not math.isfinite(self.length_normalization_alpha) or self.length_normalization_alpha <= 0:
            raise ValueError("length_normalization_alpha must be finite and positive")


class TransformersCausalSequenceScorer:
    """Teacher-forced candidate sequence scorer.

    The scorer consumes already-loaded model/tokenizer objects so model download,
    trust policy, quantization and device placement stay explicit in the caller. It
    never calls ``generate`` and never substitutes a maximum next-token probability
    for candidate sequence likelihood.
    """

    name = "transformers-causal-sequence-loglikelihood"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: CausalScoringConfig,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("causal sequence scoring requires PyTorch") from exc
        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    def _ids(self, text: str, *, add_special_tokens: bool) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _encoded_candidate(self, context: str, candidate: TextCandidate) -> tuple[list[int], int]:
        context_ids = self._ids(context, add_special_tokens=self.config.add_bos) if context else []
        candidate_ids = self._ids(candidate.text, add_special_tokens=not context_ids and self.config.add_bos)
        if context_ids and candidate_ids and candidate_ids[0] in {
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "cls_token_id", None),
        }:
            candidate_ids = candidate_ids[1:]
        if len(candidate_ids) > self.config.maximum_candidate_tokens:
            raise ValueError(
                f"candidate {candidate.candidate_id} exceeds maximum_candidate_tokens"
            )
        if not candidate_ids:
            raise ValueError(f"candidate {candidate.candidate_id} tokenized to an empty sequence")
        input_ids = context_ids + candidate_ids
        if len(input_ids) < 2:
            # A single token has no preceding position in a causal model. Prefix the
            # explicit BOS token when possible rather than inventing a probability.
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if bos is None:
                raise ValueError("causal scoring requires at least two tokens or a BOS token")
            input_ids = [int(bos), *input_ids]
            context_ids = [int(bos)]
        return input_ids, len(context_ids)

    def score(
        self,
        candidates: list[TextCandidate],
        *,
        context: str = "",
    ) -> list[SequenceScoreResult]:
        if not candidates:
            raise ValueError("at least one candidate is required")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be unique")
        torch = self._torch
        rows = [self._encoded_candidate(context, candidate) for candidate in candidates]
        output: list[SequenceScoreResult] = []
        try:
            device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            device = getattr(self.model, "device", "cpu")
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id")

        for batch_start in range(0, len(candidates), self.config.batch_size):
            batch_candidates = candidates[batch_start : batch_start + self.config.batch_size]
            batch_rows = rows[batch_start : batch_start + self.config.batch_size]
            maximum = max(len(ids) for ids, _context_length in batch_rows)
            input_ids: list[list[int]] = []
            attention: list[list[int]] = []
            for ids, _context_length in batch_rows:
                padding = maximum - len(ids)
                input_ids.append([*ids, *([int(pad_id)] * padding)])
                attention.append([*([1] * len(ids)), *([0] * padding)])
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            attention_tensor = torch.tensor(attention, dtype=torch.long, device=device)
            with torch.inference_mode():
                model_output = self.model(
                    input_ids=input_tensor,
                    attention_mask=attention_tensor,
                    use_cache=False,
                )
            logits = model_output.logits[:, :-1, :].float()
            labels = input_tensor[:, 1:]
            token_log_probs = logits.log_softmax(dim=-1).gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)

            for row_index, (candidate, (ids, context_length)) in enumerate(
                zip(batch_candidates, batch_rows, strict=True)
            ):
                # labels position p predicts input token p+1. Candidate tokens start at
                # input index context_length, so the first scored label index is
                # max(0, context_length - 1).
                score_start = max(0, context_length - 1)
                score_end = len(ids) - 1
                values = token_log_probs[row_index, score_start:score_end]
                token_count = int(values.numel())
                if token_count < 1:
                    raise RuntimeError(
                        f"candidate {candidate.candidate_id} has no scoreable tokens"
                    )
                cumulative = float(values.sum().item())
                denominator = token_count ** self.config.length_normalization_alpha
                average = cumulative / denominator
                common = {
                    "scorer": self.name,
                    "model": self.config.model_name,
                    "revision": self.config.model_revision,
                    "runtime": "transformers",
                    "metadata": {
                        "contextTokenCount": context_length,
                        "candidateTokenCount": token_count,
                        "lengthNormalizationAlpha": self.config.length_normalization_alpha,
                    },
                }
                output.append(
                    SequenceScoreResult(
                        candidate_id=candidate.candidate_id,
                        cumulative=EvidenceScore.raw(
                            cumulative,
                            semantics=ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
                            **common,
                        ),
                        average=EvidenceScore.raw(
                            average,
                            semantics=ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
                            **common,
                        ),
                        token_count=token_count,
                    )
                )
        return output


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        try:
            addresses = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return False
        return bool(addresses) and all(
            ipaddress.ip_address(address[4][0]).is_loopback for address in addresses
        )


def validate_local_rerank_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not _is_loopback(parsed.hostname):
        raise ValueError("reranker endpoint must be loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("credentials, query and fragment are not allowed")
    path = parsed.path.rstrip("/")
    if path in {"", "/v1"}:
        path = "/v1/rerank"
    if path != "/v1/rerank":
        raise ValueError("only /v1/rerank is supported")
    host = parsed.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return f"http://{host}{port}{path}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "redirect blocked", headers, fp)


@dataclass(frozen=True, slots=True)
class RerankScoreResult:
    candidate_id: str
    score: EvidenceScore
    rank: int


class LocalRerankEndpointScorer:
    """Dedicated loopback reranker client using numeric server scores.

    Unlike the chat-teacher adapter, this class does not ask a generative model to
    write probabilities. Returned scores are typed as uncalibrated until a held-out
    calibrator is applied.
    """

    name = "local-v1-rerank"

    def __init__(
        self,
        *,
        model: str,
        endpoint: str = "http://127.0.0.1:8012/v1/rerank",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.endpoint = validate_local_rerank_endpoint(endpoint)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )

    def score(
        self,
        candidates: list[TextCandidate],
        *,
        context: str = "",
    ) -> list[RerankScoreResult]:
        if not candidates:
            raise ValueError("at least one candidate is required")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be unique")
        payload = json.dumps(
            {
                "model": self.model,
                "query": context or "日本語音声認識候補として音響文脈に整合する順に評価する",
                "top_n": len(candidates),
                "documents": [candidate.text for candidate in candidates],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                outer = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local reranker request failed: {exc}") from exc
        rows = outer.get("results") if isinstance(outer, dict) else outer
        if not isinstance(rows, list):
            raise ValueError("reranker response has no results array")
        output: list[RerankScoreResult] = []
        seen: set[int] = set()
        for rank, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError("reranker result must be an object")
            index = int(row.get("index"))
            if index < 0 or index >= len(candidates) or index in seen:
                raise ValueError("reranker result index is invalid or duplicated")
            seen.add(index)
            raw_score = row.get("relevance_score", row.get("score"))
            value = float(raw_score)
            if not math.isfinite(value):
                raise ValueError("reranker score must be finite")
            candidate = candidates[index]
            output.append(
                RerankScoreResult(
                    candidate_id=candidate.candidate_id,
                    score=EvidenceScore.raw(
                        value,
                        semantics=ScoreSemantics.UNCALIBRATED_SCORE,
                        scorer=self.name,
                        model=self.model,
                        runtime="llama.cpp-compatible-rerank-endpoint",
                        metadata={"endpointOrigin": self.endpoint.rsplit("/v1/rerank", 1)[0]},
                    ),
                    rank=rank,
                )
            )
        if seen != set(range(len(candidates))):
            raise ValueError("reranker response must contain every candidate exactly once")
        return output
