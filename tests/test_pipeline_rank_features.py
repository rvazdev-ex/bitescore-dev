from pathlib import Path

import pandas as pd

from bitescore.pipeline import (
    assemble_ranking_features,
    path_features_aa,
    path_features_regsite,
    path_features_structure,
    path_features_function,
)


def test_assemble_ranking_features(tmp_path: Path):
    outdir = tmp_path
    aa_df = pd.DataFrame([
        {"id": "seq1", "aa_essential_frac": 0.25, "other": 1.0},
        {"id": "seq2", "aa_essential_frac": 0.5, "other": 2.0},
    ])
    aa_df.to_csv(path_features_aa(outdir), index=False)

    reg_df = pd.DataFrame([
        {"id": "seq1", "protease_total_sites": 4, "trypsin_K_sites": 2},
        {"id": "seq2", "protease_total_sites": 6, "trypsin_R_sites": 3},
    ])
    reg_df.to_csv(path_features_regsite(outdir), index=False)

    struct_df = pd.DataFrame([
        {"id": "seq1"},
        {"id": "seq2", "cleavage_site_accessible_fraction": 0.75},
    ])
    struct_df.to_csv(path_features_structure(outdir), index=False)

    func_df = pd.DataFrame([
        {"id": "seq1", "red_flag": True, "green_flag": False},
        {"id": "seq2", "red_flag": False, "green_flag": True},
    ])
    func_df.to_csv(path_features_function(outdir), index=False)

    rank_df = assemble_ranking_features(outdir)

    assert set(rank_df.columns) >= {
        "id",
        "aa_essential_frac",
        "protease_total_sites",
        "trypsin_K_sites",
        "trypsin_R_sites",
        "cleavage_site_accessible_fraction",
        "red_flag",
        "green_flag",
    }

    seq1 = rank_df.set_index("id").loc["seq1"]
    seq2 = rank_df.set_index("id").loc["seq2"]

    assert seq1["aa_essential_frac"] == 0.25
    assert seq2["trypsin_K_sites"] == 0  # missing column default
    assert seq1["cleavage_site_accessible_fraction"] == 0  # default when absent
    assert bool(seq1["red_flag"]) is True and bool(seq1["green_flag"]) is False
    assert bool(seq2["red_flag"]) is False and bool(seq2["green_flag"]) is True

