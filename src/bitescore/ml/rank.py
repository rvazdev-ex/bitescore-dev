from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_EXCLUDE = {"id"}


def default_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])


def _feature_matrix(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    X = df[cols].values
    return X, cols


def _train_heuristic_targets(features_df: pd.DataFrame) -> np.ndarray:
    """Generate ad-hoc training targets (legacy demo mode)."""
    _zero = pd.Series(0.0, index=features_df.index)
    return (
        0.6 * features_df.get("aa_essential_frac", _zero)
        + 0.2 * features_df.get("trypsin_K_sites", _zero)
        + 0.2 * features_df.get("trypsin_R_sites", _zero)
    ).values


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _build_reference_scorer(model, feature_columns):
    """Return a function that scores a reference protein given an sklearn model.

    The scorer computes handcrafted features for the reference protein,
    builds a single-row DataFrame matching the model's training columns,
    and returns the model's prediction.
    """
    from ..features.extract import (
        compute_aa_features,
        compute_regsite_features,
        compute_structure_feature_table,
        compute_function_features,
        merge_feature_frames,
    )
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    def _score(protein_id: str, sequence: str) -> float:
        rec = SeqRecord(Seq(sequence), id=protein_id, description="")
        records = [rec]
        aa_df = compute_aa_features(records)
        reg_df = compute_regsite_features(records)
        struct_df = compute_structure_feature_table(
            records, structure_enabled=True, alphafold_enabled=False,
        )
        func_df = compute_function_features(records)
        combined = merge_feature_frames([aa_df, reg_df, struct_df, func_df])
        # Align columns with model training columns
        for col in feature_columns:
            if col not in combined.columns:
                combined[col] = 0
        row = combined[feature_columns].values
        return float(model.predict(row)[0])

    return _score


def _run_default_calibration(model, feature_columns):
    """Fit and return a calibrator using the built-in reference data."""
    from .calibrate import calibrate_with_reference

    scorer = _build_reference_scorer(model, feature_columns)
    calibrator = calibrate_with_reference(scorer, method="isotonic")
    return calibrator


# ---------------------------------------------------------------------------
# MIL-based ranking
# ---------------------------------------------------------------------------

def _try_mil_ranking(
    features_df: pd.DataFrame,
    mil_model_path: Optional[str],
    calibrate: bool,
    outdir: Path,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Attempt MIL-based protein scoring.  Returns None if not available."""
    if mil_model_path is None:
        return None

    mil_path = Path(mil_model_path)
    if not mil_path.exists():
        logger.warning("MIL model not found at %s; falling back to RF.", mil_path)
        return None

    try:
        from .mil import load_mil_model, predict_protein_scores
        from .calibrate import load_calibrator
    except ImportError:
        logger.warning("PyTorch not available; falling back to RF ranking.")
        return None

    X, cols = _feature_matrix(features_df)
    model, mil_cfg = load_mil_model(mil_path, input_dim=len(cols))

    scores = predict_protein_scores(model, X, label_scale=mil_cfg.label_scale)

    if calibrate:
        cal_path = outdir / "calibrator.joblib"
        if cal_path.exists():
            calibrator = load_calibrator(cal_path)
            scores = calibrator.transform(scores)

    ranked = features_df.copy()
    ranked["digestibility_score"] = scores
    ranked = ranked.sort_values("digestibility_score", ascending=False).reset_index(drop=True)
    return ranked, str(mil_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def rank_sequences(
    features_df: pd.DataFrame,
    model_path: str | None,
    train_demo: bool,
    outdir: Path,
    calibrate: bool = True,
    mil_model_path: str | None = None,
):
    """Score and rank protein sequences by predicted digestibility.

    When *calibrate* is True (the default), the raw model scores are mapped
    to the DIAAS scale using the built-in reference proteins as calibration
    anchors.

    Parameters
    ----------
    features_df : DataFrame with protein features and an ``id`` column
    model_path : path to a pre-trained sklearn model (.joblib)
    train_demo : if True, train a demo model with heuristic targets
    outdir : output directory for saving models/calibrators
    calibrate : if True, calibrate scores to DIAAS scale (default)
    mil_model_path : optional path to a trained MIL model (.pt)

    Returns
    -------
    (ranked_df, model_path_str) — ranked DataFrame and path to used model
    """
    # --- Try MIL model first ---
    mil_result = _try_mil_ranking(features_df, mil_model_path, calibrate, outdir)
    if mil_result is not None:
        return mil_result

    # --- Fallback: sklearn Random Forest ---
    model = None
    if model_path and Path(model_path).exists():
        model = joblib.load(model_path)

    X, cols = _feature_matrix(features_df)

    if model is None:
        model = default_model()
        if train_demo:
            y = _train_heuristic_targets(features_df)
        else:
            y = np.zeros(len(features_df))
        model.fit(X, y)
        mp = outdir / "model.joblib"
        joblib.dump(model, mp)
        model_path = str(mp)

    scores = model.predict(X)

    # --- Calibration (on by default) ---
    if calibrate:
        try:
            calibrator = _run_default_calibration(model, cols)
            from .calibrate import save_calibrator
            cal_path = outdir / "calibrator.joblib"
            save_calibrator(calibrator, cal_path)
            scores = calibrator.transform(scores)
            logger.info("Scores calibrated to DIAAS scale using reference anchors.")
        except Exception as exc:
            logger.warning(
                "Calibration failed (%s); using raw scores.", exc,
            )

    ranked = features_df.copy()
    ranked["digestibility_score"] = scores
    ranked = ranked.sort_values("digestibility_score", ascending=False).reset_index(drop=True)
    return ranked, model_path
