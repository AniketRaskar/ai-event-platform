"""NeuMF shape, branch and unseen-event behavior tests."""
import pytest
import torch
from ml.models.neumf import NeuMF

def _small_model(**overrides):
    """Create a tiny model with deterministic side-feature tensors."""
    options = {"n_users": 3, "n_events": 4,
               "event_content": torch.randn(4, 384),
               "user_side_features": torch.randn(3, 13),
               "seen_event_mask": torch.tensor([True, True, False, False])}
    options.update(overrides)
    return NeuMF(**options)

def test_hybrid_model_scores_and_backpropagates():
    """The full hybrid architecture returns one trainable logit per pair."""
    model = _small_model(id_dropout=1.0)
    logits = model(torch.tensor([0, 2]), torch.tensor([0, 3]))
    assert logits.shape == (2,)
    logits.sum().backward()
    assert model.unknown_event_embedding.grad is not None

def test_unseen_event_uses_unknown_id_vector_at_inference():
    """An event marked unseen ignores its learned ID embedding in eval mode."""
    model = _small_model()
    model.eval()
    before = model(torch.tensor([0]), torch.tensor([2])).detach()
    with torch.no_grad():
        model.event_embedding.weight[2].add_(100)
    after = model(torch.tensor([0]), torch.tensor([2])).detach()
    assert torch.allclose(before, after)

def test_at_least_one_branch_is_required():
    """Reject a model with no scoring branch."""
    with pytest.raises(ValueError):
        _small_model(use_gmf=False, use_mlp=False)
