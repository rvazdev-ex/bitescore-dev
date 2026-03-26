"""Tests for the digestibility reference data management module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from bitescore.ml.digestibility_ref import (
    load_user_reference_csv,
    get_combined_reference_foods,
    score_reference_proteins_dataframe,
)
from bitescore.data.reference_proteins import REFERENCE_FOODS


class TestGetCombinedReferenceFoods:
    def test_builtin_only(self):
        foods = get_combined_reference_foods()
        assert len(foods) == len(REFERENCE_FOODS)

    def test_user_override(self, tmp_path):
        ref_csv = tmp_path / "ref.csv"
        ref_csv.write_text(
            "food_id,food_name,diaas,pdcaas\n"
            "whole_milk,Custom Milk,120,1.0\n"
            "new_food,Novel Protein,95,0.95\n"
        )
        foods = get_combined_reference_foods(user_ref_path=ref_csv)
        # Built-in whole_milk overridden + new_food added
        assert len(foods) == len(REFERENCE_FOODS) + 1
        milk = [f for f in foods if f.food_id == "whole_milk"][0]
        assert milk.diaas == 120  # overridden value
        novel = [f for f in foods if f.food_id == "new_food"]
        assert len(novel) == 1


class TestLoadUserReferenceCSV:
    def test_minimal_csv(self, tmp_path):
        ref_csv = tmp_path / "ref.csv"
        ref_csv.write_text(
            "food_id,food_name,diaas\n"
            "test_food,Test,88\n"
        )
        foods = load_user_reference_csv(ref_csv)
        assert len(foods) == 1
        assert foods[0].food_id == "test_food"
        assert foods[0].diaas == 88

    def test_missing_required_columns(self, tmp_path):
        ref_csv = tmp_path / "ref.csv"
        ref_csv.write_text("food_id,diaas\n" "test,88\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_user_reference_csv(ref_csv)

    def test_with_fasta_and_composition(self, tmp_path):
        fasta = tmp_path / "proteome.faa"
        fasta.write_text(">prot_A\nACDEFGHIKL\n>prot_B\nMNPQRSTVWY\n")

        ref_csv = tmp_path / "ref.csv"
        ref_csv.write_text(
            f"food_id,food_name,diaas,proteome_fasta\n"
            f"test_food,Test,88,{fasta}\n"
        )

        comp_csv = tmp_path / "comp.csv"
        comp_csv.write_text(
            "food_id,protein_id,abundance_fraction\n"
            "test_food,prot_A,0.7\n"
            "test_food,prot_B,0.3\n"
        )

        foods = load_user_reference_csv(ref_csv, composition_path=comp_csv)
        assert len(foods) == 1
        assert len(foods[0].proteins) == 2
        assert foods[0].proteins[0].abundance_fraction == 0.7


class TestScoreReferenceProteinsDataframe:
    def test_produces_dataframe(self):
        df = score_reference_proteins_dataframe(REFERENCE_FOODS)
        assert isinstance(df, pd.DataFrame)
        assert "protein_id" in df.columns
        assert "food_id" in df.columns
        assert "experimental_diaas" in df.columns
        assert "sequence" in df.columns
        assert len(df) > 0

    def test_all_foods_represented(self):
        df = score_reference_proteins_dataframe(REFERENCE_FOODS)
        food_ids = set(df["food_id"])
        expected = {f.food_id for f in REFERENCE_FOODS}
        assert food_ids == expected
