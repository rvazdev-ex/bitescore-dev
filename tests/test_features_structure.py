import math
from pathlib import Path

from bitescore.features import structure
from bitescore.features.cleavage import cleavage_site_positions


AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
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


def test_structure_features_localcolabfold_accessibility(monkeypatch, tmp_path):
    seq = "AKRACAKAA"
    pdb_path = _linear_pdb(seq, tmp_path / "wide.pdb", spacing=6.0, bfactor=90.0)

    monkeypatch.setattr(
        structure,
        "predict_structure",
        lambda sequence, seq_id, cache_dir, threads=None: pdb_path,
    )

    result = structure.structure_features(
        seq,
        "seq1",
        alphafold_enabled=False,
        cache_dir=tmp_path,
    )

    total_sites = len(cleavage_site_positions(seq))
    assert result["cleavage_sites_with_structure"] == total_sites
    assert math.isclose(result["cleavage_sites_accessible"], total_sites)
    assert math.isclose(result["cleavage_site_accessible_fraction"], 1.0)
    assert result["structure_source"] == "localcolabfold"


def test_structure_features_no_structure(monkeypatch, tmp_path):
    seq = "AKRACAKAA"
    monkeypatch.setattr(
        structure,
        "predict_structure",
        lambda sequence, seq_id, cache_dir, threads=None: None,
    )

    result = structure.structure_features(
        seq,
        "seq2",
        alphafold_enabled=False,
        cache_dir=tmp_path,
    )

    assert result["cleavage_sites_with_structure"] == 0
    assert result["cleavage_site_accessible_fraction"] == 0.0
    assert result["structure_source"] == "none"
