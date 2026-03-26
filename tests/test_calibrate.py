"""Tests for the digestibility calibration module."""

import numpy as np
import pytest
from pathlib import Path

from bitescore.ml.calibrate import (
    DigestibilityCalibrator,
    CalibrationResult,
    save_calibrator,
    load_calibrator,
)


class TestDigestibilityCalibrator:
    def test_isotonic_fit_transform(self):
        cal = DigestibilityCalibrator(method="isotonic")
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = np.array([20.0, 50.0, 80.0, 100.0, 115.0])
        result = cal.fit(predicted, observed)

        assert cal.fitted
        assert result.method == "isotonic"
        assert result.n_anchors == 5
        assert result.residual_rmse >= 0

    def test_linear_fit_transform(self):
        cal = DigestibilityCalibrator(method="linear")
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = np.array([20.0, 50.0, 80.0, 100.0, 115.0])
        result = cal.fit(predicted, observed)

        assert cal.fitted
        assert result.method == "linear"

    def test_transform_before_fit_warns(self):
        cal = DigestibilityCalibrator()
        raw = np.array([0.5, 0.6])
        result = cal.transform(raw)
        np.testing.assert_array_equal(result, raw)

    def test_isotonic_monotonic(self):
        cal = DigestibilityCalibrator(method="isotonic")
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = np.array([20.0, 50.0, 80.0, 100.0, 115.0])
        cal.fit(predicted, observed)

        test_inputs = np.linspace(0.0, 1.0, 20)
        calibrated = cal.transform(test_inputs)
        # Isotonic regression output should be non-decreasing
        for i in range(1, len(calibrated)):
            assert calibrated[i] >= calibrated[i - 1] - 1e-10

    def test_clipping(self):
        cal = DigestibilityCalibrator(method="linear", clip_min=0, clip_max=150)
        predicted = np.array([0.1, 0.5, 0.9])
        observed = np.array([20.0, 80.0, 115.0])
        cal.fit(predicted, observed)
        extreme = np.array([-10.0, 100.0])
        calibrated = cal.transform(extreme)
        assert all(0 <= v <= 150 for v in calibrated)

    def test_fit_requires_minimum_anchors(self):
        cal = DigestibilityCalibrator()
        with pytest.raises(ValueError, match="at least 2"):
            cal.fit(np.array([0.5]), np.array([80.0]))

    def test_fit_requires_equal_lengths(self):
        cal = DigestibilityCalibrator()
        with pytest.raises(ValueError, match="same length"):
            cal.fit(np.array([0.5, 0.6]), np.array([80.0]))

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            DigestibilityCalibrator(method="cubic")

    def test_fit_transform(self):
        cal = DigestibilityCalibrator(method="isotonic")
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = np.array([20.0, 50.0, 80.0, 100.0, 115.0])
        calibrated, result = cal.fit_transform(predicted, observed)
        assert len(calibrated) == 5
        assert result.n_anchors == 5


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        cal = DigestibilityCalibrator(method="isotonic")
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = np.array([20.0, 50.0, 80.0, 100.0, 115.0])
        cal.fit(predicted, observed)

        path = tmp_path / "calibrator.joblib"
        save_calibrator(cal, path)
        assert path.exists()

        loaded = load_calibrator(path)
        assert loaded.fitted
        test_input = np.array([0.4])
        np.testing.assert_allclose(
            cal.transform(test_input),
            loaded.transform(test_input),
        )
