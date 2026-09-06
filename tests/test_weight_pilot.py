"""Offline feasibility-script contracts; these are not recognition benchmarks."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pilot_module():
    spec = importlib.util.spec_from_file_location(
        "weight_pilot", ROOT / "scripts/train_public_weight_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenizerFixture:
    def __init__(self):
        self.inputs = []

    def encode(self, text, **kwargs):
        self.inputs.append(text)
        return [ord(c) for c in text]


def test_pilot_candidate_input_does_not_read_reference_fields():
    module = pilot_module()
    tokenizer = TokenizerFixture()
    row = {"text": "候補文", "kana": "コウホブン", "acoustic": -0.1, "reference": "SECRET"}
    tokens, boundary = module.candidate_tokens(tokenizer, row)
    assert "SECRET" not in "".join(tokenizer.inputs)
    assert tokens[boundary:] == [ord(c) for c in row["text"]]
    assert boundary > 0


@pytest.mark.parametrize("text", ["", "あ" * 257])
def test_pilot_never_silently_truncates_targets(text):
    with pytest.raises(ValueError):
        pilot_module().candidate_tokens(
            TokenizerFixture(), {"text": text, "kana": "ア", "acoustic": -0.1}
        )


def test_pilot_weight_updates_and_safe_head_reload(tmp_path):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    module = pilot_module()
    torch.manual_seed(17)
    vocab = {"phone": {"blank": 0, "a": 1, "i": 2}, "mora": {"blank": 0, "ア": 1, "イ": 2}}
    model = module.acoustic_model(4, vocab)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    features = torch.randn(1, 8, 4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    output = model(
        input_features=features,
        phone_labels=torch.tensor([[1, 2]]),
        mora_labels=torch.tensor([[1, 2]]),
    )
    module.finite_step(output.loss, model, optimizer)
    current = model.state_dict()
    updated = {k for k in before if not torch.equal(before[k], current[k])}
    assert updated and all(k.startswith(("phone_head.", "mora_head.")) for k in updated)
    state = {k: v for k, v in current.items() if k.startswith(("phone_head.", "mora_head."))}
    path = tmp_path / "heads.safetensors"
    safe.save_file(state, str(path))
    other = module.acoustic_model(4, vocab)
    other.load_state_dict(safe.load_file(path), strict=False)
    assert module.tensor_digest(state.items()) == module.tensor_digest(safe.load_file(path).items())
    with torch.no_grad():
        a, b = model(input_features=features), other(input_features=features)
    assert torch.equal(a.phone_logits, b.phone_logits)
    assert torch.equal(a.mora_logits, b.mora_logits)


def test_pilot_rejects_nonfinite_loss_before_optimizer_update():
    torch = pytest.importorskip("torch")
    module = pilot_module()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = module.tensor_digest(model.named_parameters())
    with pytest.raises(ValueError, match="non-finite"):
        module.finite_step(torch.tensor(float("nan")), model, optimizer)
    assert before == module.tensor_digest(model.named_parameters())


def test_pilot_result_keeps_feasibility_and_quality_claims_separate():
    result = json.loads((ROOT / "research/weight-pilot-20260905/result.json").read_text())
    assert result["real_weights_trained"] and result["fresh_process_reload"]
    assert result["promotion_approved"] is False
    assert result["heldout_publication_test"] is False
    assert result["acoustic"]["comparison"] == "random-initialized-heads"
    assert result["lora"]["development_errors_before"] == result["lora"]["development_errors_after"]
    assert len(set(result["new_source_ids"])) == 40
    exclusions = json.loads((ROOT / "research/weight-pilot-20260905/exclusions.json").read_text())
    assert len(exclusions["source_ids"]) == 72
    assert not set(result["new_source_ids"]) & set(exclusions["source_ids"])
