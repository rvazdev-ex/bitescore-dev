"""Tests for structure-aware proxy features and AlphaFold summary statistics."""

import math
from pathlib import Path

import numpy as np
import pytest

from bitescore.features import structure
from bitescore.features.structure import (
    sequence_structural_proxies,
    plddt_summary_statistics,
    structural_geometry_metrics,
    _mean_property,
    _shannon_entropy,
    _charge_features,
    _estimated_mass_kda,
    HYDROPHOBICITY,
    DISORDER_PROPENSITY,
    HELIX_PROPENSITY,
    PLDDT_VERY_LOW,
    PLDDT_LOW,
    PLDDT_CONFIDENT,
)
from bitescore.features.cleavage import cleavage_site_positions


AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def _linear_pdb(seq: str, path: Path, spacing: float, bfactor: float) -> Path:
    lines = []
    for index, aa in enumerate(seq, start=1):
        resname = AA3[aa]
        x = spacing * index
        line = (
            f"ATOM  {index:5d}  CA  {resname} A{index:4d}"
            f"{x:11.3f}{0.0:8.3f}{0.0:8.3f}  1.00{bfactor:6.2f}           C"
        )
        lines.append(line)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _variable_plddt_pdb(seq: str, path: Path, spacing: float, bfactors: list[float]) -> Path:
    """Create a PDB where each residue has a different B-factor (pLDDT)."""
    lines = []
    for index, (aa, bf) in enumerate(zip(seq, bfactors), start=1):
        resname = AA3[aa]
        x = spacing * index
        line = (
            f"ATOM  {index:5d}  CA  {resname} A{index:4d}"
            f"{x:11.3f}{0.0:8.3f}{0.0:8.3f}  1.00{bf:6.2f}           C"
        )
        lines.append(line)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


# -----------------------------------------------------------------------
# Sequence-only proxy tests
# -----------------------------------------------------------------------

class TestSequenceStructuralProxies:
    def test_returns_all_expected_keys(self):
        result = sequence_structural_proxies("ACDEFGHIKLMNPQRSTVWY")
        expected_keys = {
            "plddt_proxy",
            "hydrophobicity_mean", "hydrophobicity_max_window", "hydrophobicity_min_window",
            "disorder_propensity_mean", "disorder_prone_frac",
            "helix_propensity_mean", "sheet_propensity_mean", "coil_propensity_proxy",
            "surface_accessibility_proxy", "burial_propensity_mean",
            "aa_entropy", "sequence_complexity",
            "net_charge", "charge_density", "pos_charge_frac", "neg_charge_frac",
            "mass_kda", "pro_gly_frac", "cys_frac",
        }
        assert expected_keys == set(result.keys())

    def test_hydrophobic_sequence_has_positive_hydrophobicity(self):
        hydrophobic = "IIIIIVVVVVLLLLLAAAAAA"
        result = sequence_structural_proxies(hydrophobic)
        assert result["hydrophobicity_mean"] > 0

    def test_charged_sequence_has_negative_hydrophobicity(self):
        charged = "KKKRRRDDDEEEKKKRRRDDD"
        result = sequence_structural_proxies(charged)
        assert result["hydrophobicity_mean"] < 0

    def test_disorder_prone_sequence(self):
        # E, K, P, Q, S are disorder-promoting
        disordered = "EKPQSEKPQSEKPQSEKPQS"
        result = sequence_structural_proxies(disordered)
        assert result["disorder_propensity_mean"] > 0.2
        assert result["disorder_prone_frac"] > 0.5

    def test_ordered_sequence(self):
        # I, F, W, V, C are order-promoting
        ordered = "IFWVCIFWVCIFWVCIFWVCI"
        result = sequence_structural_proxies(ordered)
        assert result["disorder_propensity_mean"] < 0
        assert result["disorder_prone_frac"] < 0.2

    def test_helix_promoting_sequence(self):
        # A, E, L, M are strong helix formers
        helical = "AELMAELMAELMAELMAELM"
        result = sequence_structural_proxies(helical)
        assert result["helix_propensity_mean"] > 1.3

    def test_sheet_promoting_sequence(self):
        # V, I, Y, F are strong sheet formers
        sheet = "VIYFVIYFVIYFVIYFVIYF"
        result = sequence_structural_proxies(sheet)
        assert result["sheet_propensity_mean"] > 1.4

    def test_high_complexity_diverse_sequence(self):
        diverse = "ACDEFGHIKLMNPQRSTVWY"
        result = sequence_structural_proxies(diverse)
        assert result["sequence_complexity"] > 0.9

    def test_low_complexity_repetitive_sequence(self):
        repetitive = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        result = sequence_structural_proxies(repetitive)
        assert result["sequence_complexity"] < 0.1

    def test_cysteine_rich(self):
        cys_rich = "CCCCACCCCA"
        result = sequence_structural_proxies(cys_rich)
        assert result["cys_frac"] == 0.8

    def test_proline_glycine_fraction(self):
        pg = "PPGGPPGGAA"
        result = sequence_structural_proxies(pg)
        assert math.isclose(result["pro_gly_frac"], 0.8)

    def test_charge_features(self):
        seq = "KKRRDDEE"
        result = sequence_structural_proxies(seq)
        assert result["net_charge"] == 0.0  # 4 positive, 4 negative
        assert result["charge_density"] == 1.0

    def test_positive_charge_dominant(self):
        seq = "KKKKRRRR"
        result = sequence_structural_proxies(seq)
        assert result["net_charge"] > 0
        assert result["pos_charge_frac"] == 1.0
        assert result["neg_charge_frac"] == 0.0

    def test_mass_kda_reasonable(self):
        # Average AA mass is ~110 Da, so 100 residues ~ 11 kDa
        seq = "A" * 100
        result = sequence_structural_proxies(seq)
        assert 5 < result["mass_kda"] < 15

    def test_empty_sequence(self):
        result = sequence_structural_proxies("")
        assert result["plddt_proxy"] == 50.0
        assert result["hydrophobicity_mean"] == 0.0
        assert result["mass_kda"] == 0.0


# -----------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------

class TestHelperFunctions:
    def test_mean_property(self):
        assert math.isclose(_mean_property("A", HYDROPHOBICITY), 1.8)
        assert _mean_property("", HYDROPHOBICITY) == 0.0

    def test_shannon_entropy_uniform(self):
        # All 20 amino acids once → max entropy = log2(20) ≈ 4.32
        seq = "ACDEFGHIKLMNPQRSTVWY"
        ent = _shannon_entropy(seq)
        assert math.isclose(ent, math.log2(20), rel_tol=0.01)

    def test_shannon_entropy_single_aa(self):
        assert _shannon_entropy("AAAAAAA") == 0.0

    def test_shannon_entropy_empty(self):
        assert _shannon_entropy("") == 0.0

    def test_charge_features_neutral(self):
        result = _charge_features("AAAA")
        assert result["net_charge"] == 0.0
        assert result["charge_density"] == 0.0

    def test_estimated_mass_kda(self):
        mass = _estimated_mass_kda("ACDE")
        assert 0.3 < mass < 0.6  # ~4 residues

    def test_estimated_mass_kda_empty(self):
        assert _estimated_mass_kda("") == 0.0


# -----------------------------------------------------------------------
# pLDDT summary statistics tests
# -----------------------------------------------------------------------

class TestPlddtSummaryStatistics:
    def test_uniform_high_confidence(self):
        residues = [{"plddt": 95.0, "coord": [0, 0, 0]} for _ in range(10)]
        stats = plddt_summary_statistics(residues)
        assert stats["plddt_mean"] == 95.0
        assert stats["plddt_std"] == 0.0
        assert stats["plddt_frac_very_high"] == 1.0
        assert stats["plddt_frac_disordered"] == 0.0
        assert stats["plddt_n_residues"] == 10

    def test_uniform_low_confidence(self):
        residues = [{"plddt": 30.0, "coord": [0, 0, 0]} for _ in range(10)]
        stats = plddt_summary_statistics(residues)
        assert stats["plddt_mean"] == 30.0
        assert stats["plddt_frac_disordered"] == 1.0
        assert stats["plddt_frac_very_high"] == 0.0

    def test_mixed_confidence(self):
        # 5 disordered (pLDDT=30), 3 confident (pLDDT=80), 2 very high (pLDDT=95)
        residues = (
            [{"plddt": 30.0, "coord": [0, 0, 0]} for _ in range(5)]
            + [{"plddt": 80.0, "coord": [0, 0, 0]} for _ in range(3)]
            + [{"plddt": 95.0, "coord": [0, 0, 0]} for _ in range(2)]
        )
        stats = plddt_summary_statistics(residues)
        assert stats["plddt_n_residues"] == 10
        assert math.isclose(stats["plddt_frac_disordered"], 0.5)
        assert math.isclose(stats["plddt_frac_confident"], 0.3)
        assert math.isclose(stats["plddt_frac_very_high"], 0.2)
        assert stats["plddt_min"] == 30.0
        assert stats["plddt_max"] == 95.0

    def test_boundary_values(self):
        # Test exact boundary values
        residues = [
            {"plddt": 50.0, "coord": [0, 0, 0]},   # low (50 ≤ x < 70)
            {"plddt": 70.0, "coord": [0, 0, 0]},   # confident (70 ≤ x < 90)
            {"plddt": 90.0, "coord": [0, 0, 0]},   # very_high (≥ 90)
            {"plddt": 49.9, "coord": [0, 0, 0]},   # very_low (< 50)
        ]
        stats = plddt_summary_statistics(residues)
        assert stats["plddt_frac_disordered"] == 0.25  # 49.9
        assert stats["plddt_frac_low"] == 0.25          # 50.0
        assert stats["plddt_frac_confident"] == 0.25    # 70.0
        assert stats["plddt_frac_very_high"] == 0.25    # 90.0

    def test_empty_residues(self):
        stats = plddt_summary_statistics([])
        assert math.isnan(stats["plddt_mean"])
        assert stats["plddt_n_residues"] == 0


# -----------------------------------------------------------------------
# Structural geometry metrics tests
# -----------------------------------------------------------------------

class TestStructuralGeometryMetrics:
    def test_linear_structure(self):
        # Linearly spaced residues with 10 Å spacing
        residues = [
            {"plddt": 90.0, "coord": np.array([i * 10.0, 0.0, 0.0])}
            for i in range(10)
        ]
        metrics = structural_geometry_metrics(residues)
        assert metrics["radius_of_gyration"] > 0
        assert metrics["max_distance"] > 0
        assert metrics["contact_density"] >= 0

    def test_compact_structure(self):
        # All residues very close together
        residues = [
            {"plddt": 90.0, "coord": np.array([float(i % 2), float(i // 2 % 2), float(i // 4)])}
            for i in range(8)
        ]
        metrics = structural_geometry_metrics(residues)
        assert metrics["radius_of_gyration"] < 5  # Very compact
        assert metrics["contact_density"] > 0

    def test_empty_residues(self):
        metrics = structural_geometry_metrics([])
        assert math.isnan(metrics["radius_of_gyration"])
        assert math.isnan(metrics["contact_density"])

    def test_too_few_residues(self):
        residues = [{"plddt": 90.0, "coord": np.array([0.0, 0.0, 0.0])}]
        metrics = structural_geometry_metrics(residues)
        assert math.isnan(metrics["radius_of_gyration"])


# -----------------------------------------------------------------------
# Integration: structure_features with localcolabfold mock
# -----------------------------------------------------------------------

class TestStructureFeaturesIntegration:
    def test_full_features_with_structure(self, monkeypatch, tmp_path):
        seq = "AKRACAKAA"
        pdb_path = _linear_pdb(seq, tmp_path / "wide.pdb", spacing=6.0, bfactor=90.0)

        monkeypatch.setattr(
            structure,
            "predict_structure",
            lambda sequence, seq_id, cache_dir, threads=None: pdb_path,
        )

        result = structure.structure_features(
            seq, "seq1", alphafold_enabled=False, cache_dir=tmp_path,
        )

        # Sequence-only proxies present
        assert "hydrophobicity_mean" in result
        assert "disorder_propensity_mean" in result
        assert "helix_propensity_mean" in result
        assert "sequence_complexity" in result

        # pLDDT stats present (from PDB)
        assert result["plddt_mean"] == 90.0
        assert result["plddt_frac_very_high"] == 1.0
        assert result["plddt_n_residues"] == len(seq)

        # Geometry metrics present
        assert result["radius_of_gyration"] > 0
        assert result["contact_density"] >= 0

        # Cleavage accessibility present
        total_sites = len(cleavage_site_positions(seq))
        assert result["cleavage_sites_with_structure"] == total_sites
        assert result["structure_source"] == "localcolabfold"

    def test_features_without_structure(self, monkeypatch, tmp_path):
        seq = "AKRACAKAA"
        monkeypatch.setattr(
            structure,
            "predict_structure",
            lambda sequence, seq_id, cache_dir, threads=None: None,
        )

        result = structure.structure_features(
            seq, "seq2", alphafold_enabled=False, cache_dir=tmp_path,
        )

        # Sequence-only proxies always present
        assert "hydrophobicity_mean" in result
        assert "disorder_propensity_mean" in result
        assert result["plddt_proxy"] > 0

        # pLDDT stats are NaN when no structure
        assert math.isnan(result["plddt_mean"])
        assert result["plddt_n_residues"] == 0

        # Geometry metrics are NaN when no structure
        assert math.isnan(result["radius_of_gyration"])

        # Cleavage accessibility is zero
        assert result["cleavage_sites_with_structure"] == 0
        assert result["structure_source"] == "none"

    def test_variable_plddt_structure(self, monkeypatch, tmp_path):
        seq = "AKRACAKAA"
        # 4 disordered (pLDDT=30), 5 confident (pLDDT=80)
        bfactors = [30.0, 30.0, 30.0, 30.0, 80.0, 80.0, 80.0, 80.0, 80.0]
        pdb_path = _variable_plddt_pdb(seq, tmp_path / "mixed.pdb", spacing=6.0, bfactors=bfactors)

        monkeypatch.setattr(
            structure,
            "predict_structure",
            lambda sequence, seq_id, cache_dir, threads=None: pdb_path,
        )

        result = structure.structure_features(
            seq, "seq3", alphafold_enabled=False, cache_dir=tmp_path,
        )

        assert result["plddt_n_residues"] == 9
        assert math.isclose(result["plddt_frac_disordered"], 4 / 9, rel_tol=0.01)
        assert result["plddt_mean"] > 30 and result["plddt_mean"] < 80

    def test_caching_works(self, monkeypatch, tmp_path):
        seq = "AKRACAKAA"
        call_count = [0]

        def mock_predict(sequence, seq_id, cache_dir, threads=None):
            call_count[0] += 1
            return None

        monkeypatch.setattr(structure, "predict_structure", mock_predict)

        # First call
        result1 = structure.structure_features(
            seq, "seq4", alphafold_enabled=False, cache_dir=tmp_path,
        )
        assert call_count[0] == 1

        # Second call should use cache
        result2 = structure.structure_features(
            seq, "seq4", alphafold_enabled=False, cache_dir=tmp_path,
        )
        assert call_count[0] == 1  # Not incremented
        assert result1["struct_hash"] == result2["struct_hash"]


# -----------------------------------------------------------------------
# Backward compatibility: ensure old test still passes
# -----------------------------------------------------------------------

def test_structure_features_localcolabfold_accessibility(monkeypatch, tmp_path):
    """Backward-compatible test matching the original test_features_structure.py."""
    seq = "AKRACAKAA"
    pdb_path = _linear_pdb(seq, tmp_path / "wide.pdb", spacing=6.0, bfactor=90.0)

    monkeypatch.setattr(
        structure,
        "predict_structure",
        lambda sequence, seq_id, cache_dir, threads=None: pdb_path,
    )

    result = structure.structure_features(
        seq, "seq1", alphafold_enabled=False, cache_dir=tmp_path,
    )

    total_sites = len(cleavage_site_positions(seq))
    assert result["cleavage_sites_with_structure"] == total_sites
    assert math.isclose(result["cleavage_sites_accessible"], total_sites)
    assert math.isclose(result["cleavage_site_accessible_fraction"], 1.0)
    assert result["structure_source"] == "localcolabfold"


def test_structure_features_no_structure(monkeypatch, tmp_path):
    """Backward-compatible test matching the original test_features_structure.py."""
    seq = "AKRACAKAA"
    monkeypatch.setattr(
        structure,
        "predict_structure",
        lambda sequence, seq_id, cache_dir, threads=None: None,
    )

    result = structure.structure_features(
        seq, "seq2", alphafold_enabled=False, cache_dir=tmp_path,
    )

    assert result["cleavage_sites_with_structure"] == 0
    assert result["cleavage_site_accessible_fraction"] == 0.0
    assert result["structure_source"] == "none"
