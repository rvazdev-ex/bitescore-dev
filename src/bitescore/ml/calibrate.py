"""Calibration of raw digestibility scores against experimental reference data.

This module fits a monotonic mapping from raw model predictions to the
experimentally validated DIAAS scale using reference proteins/foods as
calibration anchors.

Two calibration strategies are provided:

1. **Isotonic regression** (default): A non-parametric, monotone-increasing
   step function fitted to (predicted, observed) pairs from reference foods.
   Robust to non-linear score distributions.

2. **Linear calibration**: A simple affine mapping y = a*x + b fitted by
   least squares.  Suitable when the relationship is approximately linear.

The calibrator is always fitted on the built-in reference data by default,
ensuring that output scores are in a meaningful, interpretable DIAAS-scale
range even when no user-supplied reference data is provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Outcome of fitting a calibrator."""
    method: str
    n_anchors: int
    residual_rmse: float
    anchor_ids: List[str]
    predicted_raw: np.ndarray
    observed_diaas: np.ndarray


class DigestibilityCalibrator:
    """Maps raw digestibility scores to the DIAAS scale.

    Parameters
    ----------
    method : "isotonic" or "linear"
    clip_min : minimum output value
    clip_max : maximum output value
    """

    def __init__(
        self,
        method: str = "isotonic",
        clip_min: float = 0.0,
        clip_max: float = 150.0,
    ):
        if method not in ("isotonic", "linear"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._fitted = False

        if method == "isotonic":
            self._model = IsotonicRegression(
                y_min=clip_min, y_max=clip_max, out_of_bounds="clip"
            )
        else:
            self._model = LinearRegression()

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        predicted_raw: np.ndarray,
        observed_diaas: np.ndarray,
        anchor_ids: Optional[List[str]] = None,
    ) -> CalibrationResult:
        """Fit the calibrator from raw predictions and observed DIAAS values.

        Parameters
        ----------
        predicted_raw : array of raw model scores for reference items
        observed_diaas : array of experimental DIAAS values
        anchor_ids : optional list of reference item identifiers

        Returns
        -------
        CalibrationResult with fit diagnostics
        """
        predicted_raw = np.asarray(predicted_raw, dtype=np.float64)
        observed_diaas = np.asarray(observed_diaas, dtype=np.float64)

        if len(predicted_raw) != len(observed_diaas):
            raise ValueError("predicted_raw and observed_diaas must have same length")
        if len(predicted_raw) < 2:
            raise ValueError("Need at least 2 calibration anchors")

        if self.method == "linear":
            self._model.fit(predicted_raw.reshape(-1, 1), observed_diaas)
        else:
            self._model.fit(predicted_raw, observed_diaas)
        self._fitted = True

        calibrated = self.transform(predicted_raw)
        residuals = calibrated - observed_diaas
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        logger.info(
            "Calibrator fitted (%s) on %d anchors — RMSE: %.2f DIAAS",
            self.method, len(predicted_raw), rmse,
        )

        return CalibrationResult(
            method=self.method,
            n_anchors=len(predicted_raw),
            residual_rmse=rmse,
            anchor_ids=anchor_ids or [],
            predicted_raw=predicted_raw,
            observed_diaas=observed_diaas,
        )

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated DIAAS values.

        If not yet fitted, returns raw scores unchanged with a warning.
        """
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        if not self._fitted:
            logger.warning("Calibrator not fitted; returning raw scores.")
            return raw_scores

        calibrated = self._model.predict(raw_scores.reshape(-1, 1) if self.method == "linear" else raw_scores)

        if self.method == "linear":
            calibrated = np.clip(calibrated, self.clip_min, self.clip_max)

        return calibrated

    def fit_transform(
        self,
        predicted_raw: np.ndarray,
        observed_diaas: np.ndarray,
        anchor_ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, CalibrationResult]:
        """Fit and then transform in one step."""
        result = self.fit(predicted_raw, observed_diaas, anchor_ids)
        calibrated = self.transform(predicted_raw)
        return calibrated, result


def save_calibrator(calibrator: DigestibilityCalibrator, path: Path):
    """Persist a fitted calibrator to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)
    logger.info("Calibrator saved to %s", path)


def load_calibrator(path: Path) -> DigestibilityCalibrator:
    """Load a previously saved calibrator."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Convenience: calibrate using built-in reference data
# ---------------------------------------------------------------------------

def calibrate_with_reference(
    score_fn,
    method: str = "isotonic",
) -> DigestibilityCalibrator:
    """Fit a calibrator by scoring built-in reference proteins.

    Parameters
    ----------
    score_fn : callable(protein_id, sequence) -> float
        A function that returns a raw digestibility score for a protein.
    method : calibration method ("isotonic" or "linear")

    Returns
    -------
    Fitted DigestibilityCalibrator
    """
    from ..data.reference_proteins import REFERENCE_FOODS

    food_ids = []
    raw_scores = []
    observed = []

    for food in REFERENCE_FOODS:
        protein_scores = []
        weights = []
        for prot in food.proteins:
            score = score_fn(prot.protein_id, prot.sequence)
            protein_scores.append(score)
            weights.append(prot.abundance_fraction)

        # Weighted average of protein scores for the food
        weights = np.array(weights)
        weights = weights / weights.sum()
        food_score = float(np.dot(protein_scores, weights))

        food_ids.append(food.food_id)
        raw_scores.append(food_score)
        observed.append(food.diaas)

    calibrator = DigestibilityCalibrator(method=method)
    calibrator.fit(
        np.array(raw_scores),
        np.array(observed),
        anchor_ids=food_ids,
    )
    return calibrator
