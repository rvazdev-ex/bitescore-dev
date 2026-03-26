from math import ceil

import pytest

from bitescore.features.aa import (
    AA_MOLECULAR_WEIGHTS,
    EAA_GROUPS,
    FAO_WHO_REQUIREMENTS_MG_PER_G,
    STANDARD_AMINO_ACIDS,
    essential_aa_content,
)


def test_essential_aa_content_counts_and_mass_sum():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    result = essential_aa_content(seq)

    for aa in STANDARD_AMINO_ACIDS:
        assert result[f"aa_{aa}_count"] == 1
        assert result[f"aa_{aa}_frac"] == pytest.approx(1 / len(seq))

    mg_total = sum(result[f"aa_{aa}_mg_g"] for aa in STANDARD_AMINO_ACIDS)
    assert mg_total == pytest.approx(1000.0)
    assert result["aa_essential_frac"] == pytest.approx(len(EAA_GROUPS) / len(seq))


def test_fao_scoring_and_limiting_metrics():
    seq = "M" * 10 + "C" * 10 + "K" * 10
    result = essential_aa_content(seq)

    assert result["aa_M_count"] == 10
    assert result["aa_C_count"] == 10
    assert result["aa_K_count"] == 10

    sulfur_mg = result["aa_sulfur_aa_mg_g"]
    expected_sulfur_mg = result["aa_M_mg_g"] + result["aa_C_mg_g"]
    assert sulfur_mg == pytest.approx(expected_sulfur_mg)

    expected_score = sulfur_mg / FAO_WHO_REQUIREMENTS_MG_PER_G["sulfur_aa"]
    assert result["aa_score_sulfur_aa"] == pytest.approx(expected_score)

    assert result["aa_lysine_meets_requirement"] is True
    assert result["aa_limiting_group"] == "histidine"
    assert result["aa_limiting_score"] == pytest.approx(0.0)
    assert result["aa_limiting_meets_requirement"] is False


def test_limiting_requirement_pass_when_all_scores_above_one():
    scale = 100
    counts: dict[str, int] = {aa: 0 for aa in STANDARD_AMINO_ACIDS}

    for group, residues in EAA_GROUPS.items():
        requirement = FAO_WHO_REQUIREMENTS_MG_PER_G[group]
        representative = residues[0]
        residue_mass = AA_MOLECULAR_WEIGHTS[representative]
        counts[representative] += ceil((requirement * scale) / residue_mass)

    seq = "".join(aa * counts[aa] for aa in STANDARD_AMINO_ACIDS)
    result = essential_aa_content(seq)

    for group in EAA_GROUPS:
        assert result[f"aa_{group}_meets_requirement"] is True
        assert result[f"aa_score_{group}"] > 1.0

    assert result["aa_limiting_meets_requirement"] is True
    assert result["aa_limiting_score"] > 1.0
