"""Tests for ESM-2 embedding feature extraction.

These tests verify the module's graceful degradation when torch/esm are not
installed, and the DataFrame output format.  Full embedding tests require
torch and fair-esm to be installed.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bitescore.features.esm import (
    _check_dependencies,
    _truncate_sequence,
    esm_embeddings_to_dataframe,
    compute_esm_feature_table,
    ESM_MODELS,
    _ESM_MAX_SEQ_LEN,
)


def _make_records(n=3, length=50):
    records = []
    for i in range(n):
        seq = "ACDEFGHIKLMNPQRSTVWY" * (length // 20 + 1)
        seq = seq[:length]
        records.append(SeqRecord(Seq(seq), id=f"seq_{i}", description=""))
    return records


class TestTruncation:
    def test_short_sequence_unchanged(self):
        seq = "ACDEF"
        assert _truncate_sequence(seq) == seq

    def test_long_sequence_truncated(self):
        seq = "A" * (_ESM_MAX_SEQ_LEN + 100)
        result = _truncate_sequence(seq)
        assert len(result) == _ESM_MAX_SEQ_LEN


class TestEmbeddingsToDataFrame:
    def test_empty_embeddings(self):
        df = esm_embeddings_to_dataframe({})
        assert "id" in df.columns
        assert len(df) == 0

    def test_normal_embeddings(self):
        embeddings = {
            "seq_0": np.random.randn(320),
            "seq_1": np.random.randn(320),
        }
        df = esm_embeddings_to_dataframe(embeddings, prefix="esm_")
        assert len(df) == 2
        assert "id" in df.columns
        assert "esm_0" in df.columns
        assert "esm_319" in df.columns
        assert df.iloc[0]["id"] == "seq_0"

    def test_custom_prefix(self):
        embeddings = {"s1": np.array([1.0, 2.0, 3.0])}
        df = esm_embeddings_to_dataframe(embeddings, prefix="emb_")
        assert "emb_0" in df.columns
        assert "emb_2" in df.columns


class TestComputeEsmFeatureTable:
    def test_returns_id_only_when_deps_missing(self):
        records = _make_records(3)
        with patch("bitescore.features.esm._check_dependencies", return_value=False):
            df = compute_esm_feature_table(records)
        assert "id" in df.columns
        assert len(df) == 3
        # Only id column when deps are missing
        assert len(df.columns) == 1

    def test_all_records_present_in_output(self):
        """Even when embeddings fail, all records should appear."""
        records = _make_records(5)
        with patch("bitescore.features.esm._check_dependencies", return_value=False):
            df = compute_esm_feature_table(records)
        assert set(df["id"]) == {f"seq_{i}" for i in range(5)}


class TestESMModels:
    def test_model_registry(self):
        assert "esm2_t6_8M_UR50D" in ESM_MODELS
        assert "esm2_t33_650M_UR50D" in ESM_MODELS
        assert ESM_MODELS["esm2_t6_8M_UR50D"] == 320
        assert ESM_MODELS["esm2_t33_650M_UR50D"] == 1280
