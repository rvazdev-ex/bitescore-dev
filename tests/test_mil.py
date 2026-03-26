"""Tests for the MIL (Multiple Instance Learning) digestibility model.

Tests that don't require PyTorch use mocking. Integration tests that need
torch are marked with pytest.mark.skipif.
"""

import numpy as np
import pytest

from bitescore.ml.mil import (
    _check_torch,
    FoodBag,
    MILConfig,
)

HAS_TORCH = _check_torch()


def _make_food_bags(n_foods=6, n_proteins=3, n_features=10):
    """Create synthetic food bags for testing."""
    rng = np.random.RandomState(42)
    bags = []
    for i in range(n_foods):
        features = rng.randn(n_proteins, n_features).astype(np.float32)
        abundance = np.ones(n_proteins) / n_proteins
        label = rng.uniform(0.3, 0.9)
        bags.append(FoodBag(
            food_id=f"food_{i}",
            protein_ids=[f"prot_{i}_{j}" for j in range(n_proteins)],
            features=features,
            abundance=abundance,
            label=label,
        ))
    return bags


class TestFoodBag:
    def test_creation(self):
        bag = FoodBag(
            food_id="test",
            protein_ids=["p1", "p2"],
            features=np.zeros((2, 5)),
            abundance=np.array([0.6, 0.4]),
            label=0.75,
        )
        assert bag.food_id == "test"
        assert bag.features.shape == (2, 5)
        assert np.isclose(bag.abundance.sum(), 1.0)


class TestMILConfig:
    def test_defaults(self):
        cfg = MILConfig()
        assert cfg.hidden_dim == 256
        assert cfg.attention_dim == 128
        assert cfg.n_epochs == 200
        assert cfg.label_scale == 140.0

    def test_custom(self):
        cfg = MILConfig(hidden_dim=64, n_epochs=10)
        assert cfg.hidden_dim == 64
        assert cfg.n_epochs == 10


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestMILModel:
    def test_build_model(self):
        from bitescore.ml.mil import _build_model
        cfg = MILConfig(hidden_dim=32, attention_dim=16)
        model = _build_model(input_dim=10, cfg=cfg)
        assert model is not None

    def test_forward_pass(self):
        import torch
        from bitescore.ml.mil import _build_model
        cfg = MILConfig(hidden_dim=32, attention_dim=16)
        model = _build_model(input_dim=10, cfg=cfg)
        model.eval()
        features = torch.randn(5, 10)
        abundance = torch.ones(5) / 5
        score, attn = model(features, abundance)
        assert score.shape == ()
        assert 0 <= score.item() <= 1
        assert attn.shape == (5,)
        assert np.isclose(attn.sum().item(), 1.0, atol=1e-5)

    def test_protein_scores(self):
        import torch
        from bitescore.ml.mil import _build_model
        cfg = MILConfig(hidden_dim=32, attention_dim=16)
        model = _build_model(input_dim=10, cfg=cfg)
        model.eval()
        features = torch.randn(5, 10)
        scores = model.protein_scores(features)
        assert scores.shape == (5,)
        assert all(0 <= s <= 1 for s in scores)

    def test_train_mil_model(self):
        from bitescore.ml.mil import train_mil_model
        bags = _make_food_bags(n_foods=6, n_proteins=3, n_features=10)
        cfg = MILConfig(hidden_dim=16, attention_dim=8, n_epochs=5, patience=3)
        model, history = train_mil_model(bags, cfg=cfg, val_fraction=0.3)
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

    def test_train_mil_too_few_bags(self):
        from bitescore.ml.mil import train_mil_model
        bags = _make_food_bags(n_foods=1)
        with pytest.raises(ValueError, match="at least 2"):
            train_mil_model(bags)

    def test_predict_food_digestibility(self):
        from bitescore.ml.mil import _build_model, predict_food_digestibility
        cfg = MILConfig(hidden_dim=16, attention_dim=8)
        model = _build_model(input_dim=10, cfg=cfg)
        features = np.random.randn(5, 10).astype(np.float32)
        abundance = np.ones(5) / 5
        diaas, attn = predict_food_digestibility(model, features, abundance)
        assert 0 <= diaas <= 140
        assert attn.shape == (5,)

    def test_predict_protein_scores(self):
        from bitescore.ml.mil import _build_model, predict_protein_scores
        cfg = MILConfig(hidden_dim=16, attention_dim=8)
        model = _build_model(input_dim=10, cfg=cfg)
        features = np.random.randn(5, 10).astype(np.float32)
        scores = predict_protein_scores(model, features)
        assert scores.shape == (5,)
        assert all(0 <= s <= 140 for s in scores)

    def test_save_load_model(self, tmp_path):
        from bitescore.ml.mil import _build_model, save_mil_model, load_mil_model
        cfg = MILConfig(hidden_dim=16, attention_dim=8)
        model = _build_model(input_dim=10, cfg=cfg)
        path = tmp_path / "model.pt"
        save_mil_model(model, cfg, path)
        loaded_model, loaded_cfg = load_mil_model(path, input_dim=10)
        assert loaded_cfg.hidden_dim == 16
        assert loaded_cfg.attention_dim == 8
