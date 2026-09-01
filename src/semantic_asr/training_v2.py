from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("semantic_asr.training_v2 requires the 'train' extra") from exc


@dataclass(slots=True)
class RerankerOutput:
    logits: Tensor
    probabilities: Tensor
    router_probabilities: Tensor
    selected_experts: Tensor
    hidden_states: Tensor


@dataclass(slots=True)
class RankingLossOutput:
    loss: Tensor
    expected_task_loss: Tensor
    listwise_loss: Tensor
    pairwise_loss: Tensor
    critical_loss: Tensor
    distillation_loss: Tensor


@dataclass(slots=True)
class VerifierOutput:
    logits: Tensor
    probabilities: Tensor
    global_similarity: Tensor
    local_alignment: Tensor
    attended_audio: Tensor


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if logits.shape != mask.shape:
        raise ValueError("masked softmax logits and mask must have the same shape")
    if torch.any(~mask.any(dim=dim)):
        raise ValueError("every masked-softmax row must contain a valid item")
    minimum = torch.finfo(logits.dtype).min
    values = logits.masked_fill(~mask, minimum)
    probabilities = F.softmax(values.float(), dim=dim).to(dtype=logits.dtype)
    return probabilities.masked_fill(~mask, 0)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denominator


class ResidualMLP(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size < 1 or expansion < 1:
            raise ValueError("hidden_size and expansion must be positive")
        inner = hidden_size * expansion
        self.norm = nn.LayerNorm(hidden_size)
        self.up = nn.Linear(hidden_size, inner * 2)
        self.down = nn.Linear(inner, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.norm(hidden)
        value, gate = self.up(normalized).chunk(2, dim=-1)
        update = self.down(F.silu(gate) * value)
        return hidden + self.dropout(update)


class SparseEvidenceReranker(nn.Module):
    """Small listwise reranker with sparse specialist routing.

    General LLM ideas are translated at the decision level: a router selects a
    bounded set of specialist evidence experts, while an explicit acoustic branch
    retains a minimum contribution. This is not a transformer-MoE reproduction.
    """

    def __init__(
        self,
        *,
        feature_size: int,
        state_size: int,
        hidden_size: int = 128,
        expert_count: int = 6,
        top_k_experts: int = 2,
        depth: int = 2,
        acoustic_feature_indices: tuple[int, ...] = (0, 1, 2, 3),
        acoustic_floor: float = 0.55,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if min(feature_size, state_size, hidden_size, expert_count, top_k_experts, depth) < 1:
            raise ValueError("reranker dimensions must be positive")
        if top_k_experts > expert_count:
            raise ValueError("top_k_experts cannot exceed expert_count")
        if not acoustic_feature_indices or any(
            index < 0 or index >= feature_size for index in acoustic_feature_indices
        ):
            raise ValueError("acoustic feature indices are invalid")
        if not 0 <= acoustic_floor <= 1:
            raise ValueError("acoustic_floor must be in [0, 1]")
        self.feature_size = feature_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.expert_count = expert_count
        self.top_k_experts = top_k_experts
        self.acoustic_feature_indices = acoustic_feature_indices
        self.acoustic_floor = acoustic_floor

        self.input = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
        )
        self.shared = nn.Sequential(
            *[ResidualMLP(hidden_size, expansion=2, dropout=dropout) for _ in range(depth)]
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(expert_count)
            ]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, expert_count),
        )
        self.acoustic = nn.Sequential(
            nn.LayerNorm(len(acoustic_feature_indices)),
            nn.Linear(len(acoustic_feature_indices), hidden_size // 2 or 1),
            nn.SiLU(),
            nn.Linear(hidden_size // 2 or 1, 1),
        )
        self.final_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        *,
        candidate_features: Tensor,
        state_features: Tensor,
        candidate_mask: Tensor,
        hard_routing: bool | None = None,
    ) -> RerankerOutput:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must be [batch, candidates, features]")
        batch, candidates, features = candidate_features.shape
        if features != self.feature_size:
            raise ValueError("candidate feature size mismatch")
        if state_features.shape != (batch, self.state_size):
            raise ValueError("state_features shape mismatch")
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask shape mismatch")
        if torch.any(~candidate_mask.bool().any(dim=1)):
            raise ValueError("every batch item needs at least one candidate")

        hidden = self.shared(self.input(candidate_features))
        expert_scores = torch.cat([expert(hidden) for expert in self.experts], dim=-1)
        router_logits = self.router(state_features)
        router_probabilities = F.softmax(router_logits.float(), dim=-1).to(hidden.dtype)
        use_hard = (not self.training) if hard_routing is None else bool(hard_routing)
        selected = torch.topk(router_probabilities, k=self.top_k_experts, dim=-1).indices
        sparse_mask = torch.zeros_like(router_probabilities).scatter_(1, selected, 1.0)
        sparse_probabilities = router_probabilities * sparse_mask
        sparse_probabilities = sparse_probabilities / sparse_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        if self.training and not use_hard:
            # Soft routing supplies gradient to all experts during early training.
            routed = torch.einsum("bke,be->bk", expert_scores, router_probabilities)
        else:
            routed = torch.einsum("bke,be->bk", expert_scores, sparse_probabilities)

        acoustic_features = candidate_features[:, :, list(self.acoustic_feature_indices)]
        acoustic_score = self.acoustic(acoustic_features).squeeze(-1)
        logits = self.acoustic_floor * acoustic_score + (1.0 - self.acoustic_floor) * routed
        temperature = self.final_temperature.abs().clamp_min(0.05)
        logits = logits / temperature
        probabilities = _masked_softmax(logits, candidate_mask.bool(), dim=1)
        minimum = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~candidate_mask.bool(), minimum)
        return RerankerOutput(
            logits=logits,
            probabilities=probabilities,
            router_probabilities=router_probabilities,
            selected_experts=selected,
            hidden_states=hidden,
        )


class MultiObjectiveRankingLoss(nn.Module):
    """MWER/listwise/pairwise/critical/distillation objective."""

    def __init__(
        self,
        *,
        expected_task_weight: float = 0.45,
        listwise_weight: float = 0.25,
        pairwise_weight: float = 0.12,
        critical_weight: float = 0.13,
        distillation_weight: float = 0.05,
        target_temperature: float = 0.15,
        teacher_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        weights = (
            expected_task_weight,
            listwise_weight,
            pairwise_weight,
            critical_weight,
            distillation_weight,
        )
        if any(value < 0 or not math.isfinite(value) for value in weights) or sum(weights) <= 0:
            raise ValueError("loss weights must be finite, non-negative and non-zero")
        if target_temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("temperatures must be positive")
        total = sum(weights)
        self.weights = tuple(value / total for value in weights)
        self.target_temperature = target_temperature
        self.teacher_temperature = teacher_temperature

    def forward(
        self,
        *,
        logits: Tensor,
        candidate_mask: Tensor,
        task_losses: Tensor,
        critical_losses: Tensor,
        teacher_logits: Tensor | None = None,
    ) -> RankingLossOutput:
        if logits.shape != candidate_mask.shape or logits.shape != task_losses.shape:
            raise ValueError("logits, mask and task_losses must share shape")
        if critical_losses.shape != logits.shape:
            raise ValueError("critical_losses must match logits")
        mask = candidate_mask.bool()
        if torch.any(~mask.any(dim=1)):
            raise ValueError("each row requires a valid candidate")
        if torch.any(mask & ((task_losses < 0) | (task_losses > 1))):
            raise ValueError("task losses must be in [0, 1]")
        if torch.any(mask & ((critical_losses < 0) | (critical_losses > 1))):
            raise ValueError("critical losses must be in [0, 1]")

        probabilities = _masked_softmax(logits, mask, dim=1)
        expected_task = (probabilities * task_losses.masked_fill(~mask, 0)).sum(dim=1).mean()
        expected_critical = (
            (probabilities * critical_losses.masked_fill(~mask, 0)).sum(dim=1).mean()
        )

        target_logits = -task_losses / self.target_temperature
        target_probabilities = _masked_softmax(target_logits, mask, dim=1)
        log_probabilities = torch.log(probabilities.clamp_min(1e-12))
        listwise = -(target_probabilities * log_probabilities).sum(dim=1).mean()

        pairwise_terms: list[Tensor] = []
        for row in range(logits.shape[0]):
            valid = torch.where(mask[row])[0]
            for left_index in range(len(valid)):
                for right_index in range(left_index + 1, len(valid)):
                    left = valid[left_index]
                    right = valid[right_index]
                    left_loss = task_losses[row, left]
                    right_loss = task_losses[row, right]
                    if torch.isclose(left_loss, right_loss):
                        continue
                    preferred, other = (left, right) if left_loss < right_loss else (right, left)
                    gap = (task_losses[row, preferred] - task_losses[row, other]).abs().detach()
                    pairwise_terms.append(
                        gap * F.softplus(-(logits[row, preferred] - logits[row, other]))
                    )
        pairwise = torch.stack(pairwise_terms).mean() if pairwise_terms else logits.sum() * 0.0

        distillation = logits.sum() * 0.0
        if teacher_logits is not None:
            if teacher_logits.shape != logits.shape:
                raise ValueError("teacher_logits must match logits")
            student_log = torch.log(
                _masked_softmax(logits / self.teacher_temperature, mask, dim=1).clamp_min(1e-12)
            )
            teacher = _masked_softmax(
                teacher_logits / self.teacher_temperature,
                mask,
                dim=1,
            ).detach()
            distillation = F.kl_div(student_log, teacher, reduction="batchmean") * (
                self.teacher_temperature**2
            )

        weighted = (
            self.weights[0] * expected_task
            + self.weights[1] * listwise
            + self.weights[2] * pairwise
            + self.weights[3] * expected_critical
            + self.weights[4] * distillation
        )
        return RankingLossOutput(
            loss=weighted,
            expected_task_loss=expected_task,
            listwise_loss=listwise,
            pairwise_loss=pairwise,
            critical_loss=expected_critical,
            distillation_loss=distillation,
        )


class AcousticTextVerifier(nn.Module):
    """Compact draft/verify model for generated or high-risk candidates.

    The verifier combines a global text-conditioned acoustic pool with a local
    frame/token compatibility score. It predicts compatibility; it does not decode
    text and therefore cannot silently rewrite observed evidence.
    """

    def __init__(
        self,
        *,
        audio_size: int,
        text_size: int,
        projection_size: int = 192,
        hidden_size: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if min(audio_size, text_size, projection_size, hidden_size) < 1:
            raise ValueError("verifier dimensions must be positive")
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(audio_size),
            nn.Linear(audio_size, projection_size),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, projection_size),
        )
        self.query = nn.Sequential(
            nn.Linear(projection_size, projection_size),
            nn.SiLU(),
            nn.Linear(projection_size, projection_size),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_size * 3 + 3),
            nn.Linear(projection_size * 3 + 3, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        *,
        audio_hidden: Tensor,
        audio_mask: Tensor,
        text_hidden: Tensor,
        text_mask: Tensor,
        duration_features: Tensor | None = None,
    ) -> VerifierOutput:
        if audio_hidden.ndim != 3:
            raise ValueError("audio_hidden must be [batch, frames, hidden]")
        if text_hidden.ndim != 4:
            raise ValueError("text_hidden must be [batch, candidates, tokens, hidden]")
        batch, frames, _audio_size = audio_hidden.shape
        text_batch, candidates, tokens, _text_size = text_hidden.shape
        if text_batch != batch:
            raise ValueError("audio/text batch size mismatch")
        if audio_mask.shape != (batch, frames):
            raise ValueError("audio_mask shape mismatch")
        if text_mask.shape != (batch, candidates, tokens):
            raise ValueError("text_mask shape mismatch")
        if torch.any(~audio_mask.bool().any(dim=1)) or torch.any(~text_mask.bool().any(dim=2)):
            raise ValueError("audio and every candidate need valid tokens")

        audio = F.normalize(self.audio_projection(audio_hidden), dim=-1)
        text = F.normalize(self.text_projection(text_hidden), dim=-1)
        text_summary = _masked_mean(text, text_mask.bool(), dim=2)
        query = F.normalize(self.query(text_summary), dim=-1)

        attention_logits = torch.einsum("bkd,btd->bkt", query, audio)
        expanded_audio_mask = audio_mask[:, None, :].expand(batch, candidates, frames)
        attention = _masked_softmax(attention_logits, expanded_audio_mask, dim=2)
        attended_audio = torch.einsum("bkt,btd->bkd", attention, audio)
        global_similarity = F.cosine_similarity(attended_audio, text_summary, dim=-1)

        # Local compatibility: for every text token, use its best acoustically valid
        # frame, then average over valid candidate tokens.
        local_matrix = torch.einsum("bkud,btd->bkut", text, audio)
        local_matrix = local_matrix.masked_fill(
            ~audio_mask[:, None, None, :].bool(),
            torch.finfo(local_matrix.dtype).min,
        )
        best_frame = local_matrix.max(dim=-1).values
        local_alignment = _masked_mean(best_frame, text_mask.bool(), dim=2)

        acoustic_mean = _masked_mean(audio, audio_mask.bool(), dim=1)
        acoustic_mean = acoustic_mean[:, None, :].expand(batch, candidates, -1)
        if duration_features is None:
            duration_features = torch.zeros(
                batch,
                candidates,
                3,
                dtype=audio.dtype,
                device=audio.device,
            )
        if duration_features.shape != (batch, candidates, 3):
            raise ValueError("duration_features must be [batch, candidates, 3]")
        classifier_input = torch.cat(
            (
                attended_audio,
                text_summary,
                acoustic_mean,
                duration_features.to(dtype=audio.dtype),
            ),
            dim=-1,
        )
        logits = self.classifier(classifier_input).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return VerifierOutput(
            logits=logits,
            probabilities=probabilities,
            global_similarity=global_similarity,
            local_alignment=local_alignment,
            attended_audio=attended_audio,
        )


def verifier_loss(
    output: VerifierOutput,
    labels: Tensor,
    *,
    candidate_mask: Tensor | None = None,
    positive_weight: float = 1.0,
) -> Tensor:
    if labels.shape != output.logits.shape:
        raise ValueError("verifier labels must match logits")
    valid = (
        torch.ones_like(labels, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask.bool()
    )
    if valid.shape != labels.shape:
        raise ValueError("candidate_mask must match labels")
    if torch.any(valid & ((labels < 0) | (labels > 1))):
        raise ValueError("verifier labels must be in [0, 1]")
    if not torch.any(valid):
        return output.logits.sum() * 0.0
    weight = torch.tensor(
        float(positive_weight),
        dtype=output.logits.dtype,
        device=output.logits.device,
    )
    return F.binary_cross_entropy_with_logits(
        output.logits.masked_select(valid),
        labels.to(dtype=output.logits.dtype).masked_select(valid),
        pos_weight=weight,
    )
