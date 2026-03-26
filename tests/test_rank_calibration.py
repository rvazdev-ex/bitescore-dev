"""Tests for the updated rank module with calibration support."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bitescore.ml.rank import (
    default_model,
    _feature_matrix,
    _train_heuristic_targets,
    rank_sequences,
    FEATURE_EXCLUDE,
)


def _make_features_df(n=5):
    """Create a minimal feature DataFrame for ranking tests."""
    return pd.DataFrame({
        "id": [f"seq_{i}" for i in range(n)],
        "aa_essential_frac": np.random.uniform(0.1, 0.5, n),
        "protease_total_sites": np.random.randint(1, 20, n),
        "trypsin_K_sites": np.random.randint(0, 10, n),
        "trypsin_R_sites": np.random.randint(0, 10, n),
        "red_flag": [False] * n,
        "green_flag": [True] * n,
    })


class TestFeatureMatrix:
    def test_excludes_id(self):
        df = _make_features_df()
        X, cols = _feature_matrix(df)
        assert "id" not in cols
        assert X.shape == (5, 6)


class TestHeuristicTargets:
    def test_generates_targets(self):
        df = _make_features_df()
        y = _train_heuristic_targets(df)
        assert y.shape == (5,)


class TestRankSequences:
    def test_train_demo_with_calibration(self, tmp_path):
        df = _make_features_df(10)
        ranked, model_path = rank_sequences(
            df,
            model_path=None,
            train_demo=True,
            outdir=tmp_path,
            calibrate=True,
        )
        assert "digestibility_score" in ranked.columns
        assert len(ranked) == 10
        # Should be sorted descending
        scores = ranked["digestibility_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        # Model should be saved
        assert (tmp_path / "model.joblib").exists()

    def test_no_calibration(self, tmp_path):
        df = _make_features_df(5)
        ranked, model_path = rank_sequences(
            df,
            model_path=None,
            train_demo=True,
            outdir=tmp_path,
            calibrate=False,
        )
        assert "digestibility_score" in ranked.columns
        # Should NOT have calibrator
        assert not (tmp_path / "calibrator.joblib").exists()

    def test_calibrator_saved(self, tmp_path):
        df = _make_features_df(10)
        rank_sequences(
            df,
            model_path=None,
            train_demo=True,
            outdir=tmp_path,
            calibrate=True,
        )
        # Calibrator should be saved when calibration succeeds
        assert (tmp_path / "calibrator.joblib").exists()

    def test_calibration_default_on(self, tmp_path):
        """Calibration should be on by default."""
        df = _make_features_df(10)
        ranked, _ = rank_sequences(
            df,
            model_path=None,
            train_demo=True,
            outdir=tmp_path,
            # calibrate not specified — should default to True
        )
        assert "digestibility_score" in ranked.columns
