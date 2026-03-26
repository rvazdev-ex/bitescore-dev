import math

import pytest

from bitescore.features.cleavage import (
    DEFAULT_PROTEASES,
    ProteaseRule,
    cleavage_accessibility_scores,
    protease_cleavage_counts,
)


def test_protease_cleavage_counts_basic():
    seq = "AKRPFPD"

    counts = protease_cleavage_counts(seq)

    assert counts["protease_trypsin_sites"] == 1
    assert counts["protease_chymotrypsin_sites"] == 0
    assert counts["protease_arg_c_sites"] == 0
    assert counts["protease_total_sites"] == sum(
        v for k, v in counts.items() if k.startswith("protease_") and k.endswith("_sites") and k != "protease_total_sites"
    )


def test_protease_counts_custom_rule():
    rule = ProteaseRule(name="custom", cleavage_side="C", residues=frozenset("A"))
    seq = "AAA"

    counts = protease_cleavage_counts(seq, proteases=[rule])

    # bonds AA -> AA: two cleavage opportunities
    assert counts == {"protease_custom_sites": 2, "protease_total_sites": 2}


@pytest.mark.parametrize(
    "seq, expected_trypsin, expected_chymo, expected_glu_c",
    [
        ("AKRPFPD", (1, 0), 0, 0),
        ("MDEKLPFYWAA", (1, 0), 3, 1),
    ],
)
def test_cleavage_accessibility_scores(seq, expected_trypsin, expected_chymo, expected_glu_c):
    scores = cleavage_accessibility_scores(seq)

    assert scores["trypsin_K_sites"] == expected_trypsin[0]
    assert scores["trypsin_R_sites"] == expected_trypsin[1]
    assert scores["chymotrypsin_sites"] == expected_chymo
    assert scores["protease_glu_c_sites"] == expected_glu_c

    assert math.isclose(
        scores["cleavage_accessibility_proxy"],
        0.5 * scores["exposure_avg"] + 0.5 * scores["flexibility_win7"],
    )


def test_default_proteases_are_unique():
    names = [rule.name for rule in DEFAULT_PROTEASES]
    assert len(names) == len(set(names))
