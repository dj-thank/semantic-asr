from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("semantic_asr.acoustic_verifier requires the 'train' extra") from exc


@dataclass(slots=True)
class CandidateVerifierOutput:
    loss: Tensor | None
    ranking_loss: Tensor | None
    balance_loss: Tensor
    logits: Tensor
    attention: Tensor
    branch_gates: Tensor
    candidate_embeddings: Tensor
    selected_acoustic_embeddings: Tensor


class QuerySelectedAcousticVerifier(nn.Module):
    """Small candidate-conditioned acoustic verifier.

    Each candidate mora sequence forms a query over acoustic frames. Three
    bounded branches are mixed by learned gates:

    1. query-selected acoustic evidence,
    2. global/local acoustic context,
    3. candidate-internal mora evidence.

    This is an ASR orchestration translation of sparse selection and gated
    residual ideas. It is not an implementation of QSA, GDN, mHC, or AttnRes.
    """

    def __init__(
        self,
        *,
        acoustic_hidden_size: int,
        mora_vocab_size: int,
        model_size: int = 192,
        mora_padding_id: int = 0,
        dropout: float = 0.10,
        balance_weight: float = 0.02,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if acoustic_hidden_size < 1 or mora_vocab_size < 2 or model_size < 8:
            raise ValueError("invalid verifier dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if balance_weight < 0 or temperature <= 0:
            raise ValueError("invalid verifier regularization")
        self.mora_padding_id = int(mora_padding_id)
        self.balance_weight = float(balance_weight)
        self.temperature = float(temperature)

        self.acoustic_projection = nn.Sequential(
            nn.LayerNorm(acoustic_hidden_size),
            nn.Linear(acoustic_hidden_size, model_size),
            nn.GELU(),
        )
        self.mora_embedding = nn.Embedding(
            mora_vocab_size,
            model_size,
            padding_idx=self.mora_padding_id,
        )
        self.mora_encoder = nn.GRU(
            model_size,
            model_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.query_projection = nn.Linear(model_size, model_size, bias=False)
        self.key_projection = nn.Linear(model_size, model_size, bias=False)
        self.value_projection = nn.Linear(model_size, model_size, bias=False)

        self.selected_branch = nn.Sequential(
            nn.Linear(model_size * 4, model_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size, model_size),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(model_size * 3, model_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size, model_size),
        )
        self.mora_branch = nn.Sequential(
            nn.Linear(model_size, model_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size, model_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(model_size * 3, model_size),
            nn.GELU(),
            nn.Linear(model_size, 3),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_size),
            nn.Linear(model_size, model_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_size // 2, 1),
        )

    def _candidate_embeddings(
        self,
        candidate_mora_ids: Tensor,
        candidate_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_mora_ids.ndim != 3:
            raise ValueError("candidate_mora_ids must be [batch, candidates, mora]")
        batch, candidates, length = candidate_mora_ids.shape
        flat = candidate_mora_ids.reshape(batch * candidates, length)
        if candidate_mask is None:
            mask = flat.ne(self.mora_padding_id)
        else:
            if candidate_mask.shape != candidate_mora_ids.shape:
                raise ValueError("candidate_mask must match candidate_mora_ids")
            mask = candidate_mask.reshape(batch * candidates, length).bool()
        lengths = mask.sum(dim=1).clamp_min(1)
        embedded = self.mora_embedding(flat)
        encoded, _state = self.mora_encoder(embedded)
        mask_value = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * mask_value).sum(dim=1) / lengths.unsqueeze(-1)
        return pooled.reshape(batch, candidates, -1), mask.reshape(batch, candidates, length)

    def forward(
        self,
        *,
        acoustic_hidden: Tensor,
        candidate_mora_ids: Tensor,
        acoustic_mask: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> CandidateVerifierOutput:
        if acoustic_hidden.ndim != 3:
            raise ValueError("acoustic_hidden must be [batch, frames, hidden]")
        batch, frames, _hidden = acoustic_hidden.shape
        if candidate_mora_ids.shape[0] != batch:
            raise ValueError("acoustic and candidate batch sizes must match")
        candidates = candidate_mora_ids.shape[1]
        if candidates < 1:
            raise ValueError("at least one candidate is required")

        acoustic = self.acoustic_projection(acoustic_hidden)
        candidate, _resolved_candidate_mask = self._candidate_embeddings(
            candidate_mora_ids, candidate_mask
        )
        if acoustic_mask is None:
            frame_mask = torch.ones((batch, frames), dtype=torch.bool, device=acoustic.device)
        else:
            if acoustic_mask.shape != (batch, frames):
                raise ValueError("acoustic_mask must be [batch, frames]")
            frame_mask = acoustic_mask.bool()
        if torch.any(frame_mask.sum(dim=1) == 0):
            raise ValueError("every audio item must contain at least one valid frame")

        query = self.query_projection(candidate)
        keys = self.key_projection(acoustic)
        values = self.value_projection(acoustic)
        attention_logits = torch.einsum("bnd,btd->bnt", query, keys)
        attention_logits /= max(1.0, query.shape[-1] ** 0.5) * self.temperature
        attention_logits = attention_logits.masked_fill(
            ~frame_mask[:, None, :], torch.finfo(attention_logits.dtype).min
        )
        attention = F.softmax(attention_logits, dim=-1)
        selected = torch.einsum("bnt,btd->bnd", attention, values)

        frame_weights = frame_mask.to(dtype=acoustic.dtype)
        global_mean = (acoustic * frame_weights.unsqueeze(-1)).sum(dim=1)
        global_mean /= frame_weights.sum(dim=1, keepdim=True).clamp_min(1)
        masked_acoustic = acoustic.masked_fill(
            ~frame_mask.unsqueeze(-1), torch.finfo(acoustic.dtype).min
        )
        global_max = masked_acoustic.amax(dim=1)
        global_context = 0.5 * (global_mean + global_max)
        global_expanded = global_context[:, None, :].expand(-1, candidates, -1)

        selected_features = torch.cat(
            [
                selected,
                candidate,
                selected * candidate,
                torch.abs(selected - candidate),
            ],
            dim=-1,
        )
        context_features = torch.cat(
            [
                global_expanded,
                candidate,
                torch.abs(global_expanded - candidate),
            ],
            dim=-1,
        )
        selected_branch = self.selected_branch(selected_features)
        context_branch = self.context_branch(context_features)
        mora_branch = self.mora_branch(candidate)
        gate_input = torch.cat([selected, global_expanded, candidate], dim=-1)
        gates = F.softmax(self.gate(gate_input), dim=-1)
        stacked = torch.stack([selected_branch, context_branch, mora_branch], dim=-2)
        mixed = (stacked * gates.unsqueeze(-1)).sum(dim=-2)
        logits = self.output(mixed).squeeze(-1)

        mean_gate = gates.mean(dim=(0, 1))
        uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
        balance_loss = F.mse_loss(mean_gate, uniform)
        ranking_loss = None
        total_loss = self.balance_weight * balance_loss
        if targets is not None:
            if targets.shape != (batch,):
                raise ValueError("targets must contain one candidate index per batch")
            if torch.any(targets < 0) or torch.any(targets >= candidates):
                raise ValueError("target candidate index is out of range")
            ranking_loss = F.cross_entropy(logits, targets.long())
            total_loss = ranking_loss + self.balance_weight * balance_loss

        return CandidateVerifierOutput(
            loss=total_loss,
            ranking_loss=ranking_loss,
            balance_loss=balance_loss,
            logits=logits,
            attention=attention,
            branch_gates=gates,
            candidate_embeddings=candidate,
            selected_acoustic_embeddings=selected,
        )
