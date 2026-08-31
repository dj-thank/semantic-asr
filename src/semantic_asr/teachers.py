from __future__ import annotations

import ipaddress
import json
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from .contracts import CandidateEvidence, RankedCandidate


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
            ipaddress.ip_address(row[4][0]).is_loopback for row in addresses
        )


def _validate_endpoint(endpoint: str, *, allowed_path: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not _is_loopback(parsed.hostname):
        raise ValueError("local teacher endpoint must be loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("credentials, query, and fragment are not allowed")
    path = parsed.path.rstrip("/")
    if path in {"", "/api", "/v1"}:
        path = allowed_path
    if path != allowed_path:
        raise ValueError(f"only {allowed_path} is supported")
    host = parsed.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return f"http://{host}{port}{path}"


def validate_ollama_endpoint(endpoint: str) -> str:
    return _validate_endpoint(endpoint, allowed_path="/api/chat")


def validate_openai_endpoint(endpoint: str) -> str:
    return _validate_endpoint(endpoint, allowed_path="/v1/chat/completions")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "redirect blocked", headers, fp)


def _safe_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )


@dataclass(frozen=True, slots=True)
class TeacherResult:
    probabilities: dict[str, float]
    model: str
    endpoint_origin: str
    protocol: str
    entropy: float
    abstained: bool


@dataclass(frozen=True, slots=True)
class DelayedTeacherPolicy:
    minimum_entropy: float = 0.45
    maximum_posterior_margin: float = 0.40
    minimum_disagreement: float = 0.18

    def should_query(self, ranked: list[RankedCandidate]) -> bool:
        if len(ranked) < 2:
            return False
        gate = ranked[0].gate
        margin = gate.uncertainty.get("posteriorMargin", 1.0)
        return (
            gate.entropy >= self.minimum_entropy
            or margin <= self.maximum_posterior_margin
            or gate.disagreement >= self.minimum_disagreement
        )


def _schema(candidate_count: int) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["probabilities", "abstain"],
        "properties": {
            "probabilities": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "p"],
                    "properties": {
                        "id": {"type": "string"},
                        "p": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
            },
            "abstain": {"type": "boolean"},
        },
    }


def _prompt(
    candidates: list[CandidateEvidence],
    *,
    context: str,
    locked_consensus: str = "",
    contradiction: str = "",
) -> str:
    return (
        "あなたは日本語ASR候補の局所ランキング教師です。"
        "新しい文章、候補集合外の語、思考過程を出力してはいけません。"
        "文法的な自然さだけで、言い間違い・フィラー・助詞誤りを消してはいけません。"
        "音響証拠が与えられていないため、判断できない場合はabstain=trueにしてください。"
        "probabilitiesは入力候補IDだけに割り当て、合計を1にしてください。\n"
        f"固定済み共通部分: {locked_consensus}\n"
        f"矛盾区間: {contradiction}\n"
        f"文脈: {context}\n"
        "候補: "
        + json.dumps(
            [
                {
                    "id": candidate.candidate_id,
                    "text": candidate.text,
                    "source": candidate.evidence_source,
                }
                for candidate in candidates
            ],
            ensure_ascii=False,
        )
    )


def _candidate_ids(candidates: list[CandidateEvidence]) -> list[str]:
    if len(candidates) < 2:
        raise ValueError("teacher comparison requires at least two candidates")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    return identifiers


def _validate_response(
    payload: object, candidate_ids: list[str]
) -> tuple[dict[str, float], bool, float]:
    rows = payload.get("probabilities") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("teacher response has no probabilities array")
    actual_ids = [str(row.get("id")) for row in rows if isinstance(row, dict)]
    if len(actual_ids) != len(candidate_ids) or set(actual_ids) != set(candidate_ids):
        raise ValueError("teacher response must contain every candidate ID exactly once")
    probabilities = {str(row["id"]): float(row["p"]) for row in rows}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("teacher probability is invalid")
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("teacher probabilities sum to zero")
    probabilities = {
        candidate_id: probability / total for candidate_id, probability in probabilities.items()
    }
    entropy = (
        -sum(probability * math.log(probability + 1e-12) for probability in probabilities.values())
        / math.log(len(probabilities))
        if len(probabilities) > 1
        else 0.0
    )
    abstained = bool(payload.get("abstain", False)) if isinstance(payload, dict) else False
    return probabilities, abstained, entropy


class OllamaRanker:
    protocol = "ollama"

    def __init__(
        self,
        *,
        model: str,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: float = 90.0,
    ) -> None:
        lowered = model.lower()
        if ":cloud" in lowered or lowered.startswith("cloud/"):
            raise ValueError("cloud-routed model names are disabled")
        self.model = model
        self.endpoint = validate_ollama_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self._opener = _safe_opener()

    def probabilities(
        self,
        candidates: list[CandidateEvidence],
        *,
        context: str = "",
        locked_consensus: str = "",
        contradiction: str = "",
    ) -> TeacherResult:
        identifiers = _candidate_ids(candidates)
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": _schema(len(identifiers)),
                "options": {"temperature": 0},
                "messages": [
                    {
                        "role": "user",
                        "content": _prompt(
                            candidates,
                            context=context,
                            locked_consensus=locked_consensus,
                            contradiction=contradiction,
                        ),
                    }
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                outer = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local teacher request failed: {exc}") from exc
        content = outer.get("message", {}).get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError as exc:
            raise ValueError("teacher returned invalid structured JSON") from exc
        probabilities, abstained, entropy = _validate_response(parsed, identifiers)
        return TeacherResult(
            probabilities=probabilities,
            model=self.model,
            endpoint_origin=self.endpoint.rsplit("/api/chat", 1)[0],
            protocol=self.protocol,
            entropy=entropy,
            abstained=abstained,
        )


class OpenAICompatibleRanker:
    """Loopback-only ranker for locally served Qwen3.8 or compatible models."""

    protocol = "openai-compatible"

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3.8-Flash-Next",
        endpoint: str = "http://127.0.0.1:8000/v1/chat/completions",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.endpoint = validate_openai_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self._opener = _safe_opener()

    def probabilities(
        self,
        candidates: list[CandidateEvidence],
        *,
        context: str = "",
        locked_consensus: str = "",
        contradiction: str = "",
    ) -> TeacherResult:
        identifiers = _candidate_ids(candidates)
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": _prompt(
                            candidates,
                            context=context,
                            locked_consensus=locked_consensus,
                            contradiction=contradiction,
                        ),
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "semantic_asr_rank",
                        "strict": True,
                        "schema": _schema(len(identifiers)),
                    },
                },
                "chat_template_kwargs": {"preserve_thinking": False},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                outer = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible teacher failed: {exc}") from exc
        content = outer.get("choices", [{}])[0].get("message", {}).get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError as exc:
            raise ValueError("teacher returned invalid structured JSON") from exc
        probabilities, abstained, entropy = _validate_response(parsed, identifiers)
        return TeacherResult(
            probabilities=probabilities,
            model=self.model,
            endpoint_origin=self.endpoint.rsplit("/v1/chat/completions", 1)[0],
            protocol=self.protocol,
            entropy=entropy,
            abstained=abstained,
        )
