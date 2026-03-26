from __future__ import annotations

from collections import Counter
from math import isclose

ESSENTIAL_AA = set("HILKMFWTV")
STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

# Average residue molecular weights in Daltons (g/mol) for proteinogenic amino acids
AA_MOLECULAR_WEIGHTS = {
    "A": 89.0935,
    "C": 121.1590,
    "D": 133.1027,
    "E": 147.1293,
    "F": 165.1900,
    "G": 75.0669,
    "H": 155.1546,
    "I": 131.1736,
    "K": 146.1882,
    "L": 131.1736,
    "M": 149.2124,
    "N": 132.1179,
    "P": 115.1310,
    "Q": 146.1445,
    "R": 174.2017,
    "S": 105.0930,
    "T": 119.1197,
    "V": 117.1469,
    "W": 204.2262,
    "Y": 181.1894,
}

# FAO/WHO/UNU 2007 indispensable amino acid requirements (adults, mg per g protein)
FAO_WHO_REQUIREMENTS_MG_PER_G = {
    "histidine": 15.0,
    "isoleucine": 30.0,
    "leucine": 59.0,
    "lysine": 45.0,
    "sulfur_aa": 22.0,  # Methionine + Cysteine
    "aromatic_aa": 38.0,  # Phenylalanine + Tyrosine
    "threonine": 23.0,
    "tryptophan": 6.0,
    "valine": 39.0,
}

EAA_GROUPS = {
    "histidine": ("H",),
    "isoleucine": ("I",),
    "leucine": ("L",),
    "lysine": ("K",),
    "sulfur_aa": ("M", "C"),
    "aromatic_aa": ("F", "Y"),
    "threonine": ("T",),
    "tryptophan": ("W",),
    "valine": ("V",),
}


def essential_aa_content(seq: str) -> dict[str, float | int | str | bool | None]:
    seq = (seq or "").replace("*", "").upper()
    counts = Counter(seq)
    total_residues = len(seq)
    length_denom = total_residues or 1

    # Counts, fractions and mass contribution for all standard amino acids
    aa_counts = {aa: counts.get(aa, 0) for aa in STANDARD_AMINO_ACIDS}
    essential_count = sum(aa_counts[aa] for aa in ESSENTIAL_AA)
    mass_contrib = {
        aa: aa_counts[aa] * AA_MOLECULAR_WEIGHTS[aa]
        for aa in STANDARD_AMINO_ACIDS
    }
    total_mass = sum(mass_contrib.values())
    mass_denom = total_mass or 1.0

    result: dict[str, float | int | str | bool | None] = {
        "aa_essential_frac": essential_count / length_denom,
    }

    for aa in STANDARD_AMINO_ACIDS:
        count = aa_counts[aa]
        result[f"aa_{aa}_count"] = count
        result[f"aa_{aa}_frac"] = count / length_denom
        result[f"aa_{aa}_mg_g"] = (mass_contrib[aa] / mass_denom) * 1000.0

    # Grouped essential amino acid metrics against FAO/WHO requirements
    scores = {}
    for group, residues in EAA_GROUPS.items():
        content_mg_g = sum(result[f"aa_{aa}_mg_g"] for aa in residues)
        requirement = FAO_WHO_REQUIREMENTS_MG_PER_G[group]
        score = content_mg_g / requirement if requirement else 0.0
        meets = score >= 1.0 or isclose(score, 1.0, rel_tol=1e-9, abs_tol=1e-9)
        result[f"aa_{group}_mg_g"] = content_mg_g
        result[f"aa_score_{group}"] = score
        result[f"aa_{group}_meets_requirement"] = meets
        scores[group] = score

    limiting_group = None
    limiting_score = 0.0
    if scores:
        limiting_group, limiting_score = min(scores.items(), key=lambda item: item[1])

    result["aa_limiting_group"] = limiting_group
    result["aa_limiting_score"] = limiting_score
    result["aa_limiting_meets_requirement"] = (
        limiting_group is not None
        and (limiting_score >= 1.0 or isclose(limiting_score, 1.0, rel_tol=1e-9, abs_tol=1e-9))
    )

    return result

def physchem(seq: str):
    length = len(seq)
    aromatics = sum(seq.count(a) for a in "FWY") / max(length,1)
    charge_proxy = (seq.count("K")+seq.count("R")-seq.count("D")-seq.count("E"))/max(length,1)
    glyco_sites = seq.count("NXT")+seq.count("NXS")
    return {
        "aromatic_frac": aromatics,
        "charge_proxy": charge_proxy,
        "glyco_site_proxy": glyco_sites
    }
