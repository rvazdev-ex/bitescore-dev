"""Digestibility reference data management.

Handles loading and preparation of reference food digestibility data for both
model training (MIL) and score calibration.

Data sources:
    1. Built-in reference foods (always available via ``data.reference_proteins``)
    2. User-supplied CSV files with custom reference data

User-supplied CSVs
------------------
**digestibility_reference.csv** — food-level experimental values::

    food_id,food_name,diaas,pdcaas,proteome_fasta
    my_food,My Custom Food,95,0.96,path/to/proteome.faa

**food_protein_composition.csv** — protein abundance per food::

    food_id,protein_id,abundance_fraction
    my_food,prot_A,0.6
    my_food,prot_B,0.4

When user CSVs are provided, they are merged with the built-in reference data
to form the complete training/calibration set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

from ..data.reference_proteins import (
    REFERENCE_FOODS,
    ReferenceFood,
    ReferenceProtein,
    get_all_reference_proteins,
    reference_proteins_as_seqrecords,
)

logger = logging.getLogger(__name__)


def load_user_reference_csv(
    ref_path: str | Path,
    composition_path: str | Path | None = None,
) -> List[ReferenceFood]:
    """Load user-supplied reference food digestibility data.

    Parameters
    ----------
    ref_path : path to digestibility_reference.csv
    composition_path : optional path to food_protein_composition.csv

    Returns
    -------
    List of ReferenceFood objects with proteins populated from FASTA files
    and composition table.
    """
    ref_df = pd.read_csv(ref_path)
    required_cols = {"food_id", "food_name", "diaas"}
    missing = required_cols - set(ref_df.columns)
    if missing:
        raise ValueError(f"Reference CSV missing columns: {missing}")

    # Load composition if provided
    comp_df = None
    if composition_path is not None:
        comp_df = pd.read_csv(composition_path)
        comp_required = {"food_id", "protein_id", "abundance_fraction"}
        comp_missing = comp_required - set(comp_df.columns)
        if comp_missing:
            raise ValueError(f"Composition CSV missing columns: {comp_missing}")

    foods: List[ReferenceFood] = []
    for _, row in ref_df.iterrows():
        food_id = row["food_id"]
        diaas = float(row["diaas"])
        pdcaas = float(row.get("pdcaas", diaas / 100.0))
        food_name = row["food_name"]

        proteins: List[ReferenceProtein] = []

        # Load proteins from FASTA if provided
        fasta_path = row.get("proteome_fasta")
        if pd.notna(fasta_path) and Path(fasta_path).exists():
            records = list(SeqIO.parse(str(fasta_path), "fasta"))

            # Get abundance fractions from composition table
            abundances: Dict[str, float] = {}
            if comp_df is not None:
                food_comp = comp_df[comp_df["food_id"] == food_id]
                abundances = dict(
                    zip(food_comp["protein_id"], food_comp["abundance_fraction"])
                )

            # Default to equal abundance if not specified
            default_abundance = 1.0 / max(len(records), 1)

            for rec in records:
                proteins.append(ReferenceProtein(
                    protein_id=rec.id,
                    uniprot_accession=rec.id,
                    sequence=str(rec.seq),
                    abundance_fraction=abundances.get(rec.id, default_abundance),
                ))

        foods.append(ReferenceFood(
            food_id=food_id,
            food_name=food_name,
            diaas=diaas,
            pdcaas=pdcaas,
            proteins=proteins,
        ))

    return foods


def get_combined_reference_foods(
    user_ref_path: str | Path | None = None,
    user_comp_path: str | Path | None = None,
) -> List[ReferenceFood]:
    """Combine built-in and user-supplied reference foods.

    Built-in foods are always included. User-supplied foods with the same
    food_id override the built-in entry.
    """
    foods_by_id: Dict[str, ReferenceFood] = {}

    # Start with built-in
    for food in REFERENCE_FOODS:
        foods_by_id[food.food_id] = food

    # Merge user-supplied
    if user_ref_path is not None:
        user_foods = load_user_reference_csv(user_ref_path, user_comp_path)
        for food in user_foods:
            foods_by_id[food.food_id] = food
            logger.info("Added/overrode reference food: %s", food.food_id)

    return list(foods_by_id.values())


def prepare_food_bags(
    reference_foods: List[ReferenceFood],
    feature_fn,
    label_scale: float = 140.0,
):
    """Convert reference foods into MIL FoodBag instances.

    Parameters
    ----------
    reference_foods : list of ReferenceFood
    feature_fn : callable(protein_id, sequence) -> np.ndarray
        Function that returns a feature vector for a protein.
    label_scale : normalization factor for DIAAS labels

    Returns
    -------
    List of FoodBag instances ready for MIL training
    """
    from .mil import FoodBag

    bags: List[FoodBag] = []

    for food in reference_foods:
        if not food.proteins:
            logger.warning("Skipping food %s: no proteins", food.food_id)
            continue

        protein_ids = []
        feature_rows = []
        abundances = []

        for prot in food.proteins:
            feats = feature_fn(prot.protein_id, prot.sequence)
            if feats is not None:
                protein_ids.append(prot.protein_id)
                feature_rows.append(feats)
                abundances.append(prot.abundance_fraction)

        if not feature_rows:
            logger.warning("Skipping food %s: no features computed", food.food_id)
            continue

        feature_matrix = np.stack(feature_rows)
        abundance_array = np.array(abundances)
        abundance_array = abundance_array / abundance_array.sum()  # normalize

        bags.append(FoodBag(
            food_id=food.food_id,
            protein_ids=protein_ids,
            features=feature_matrix,
            abundance=abundance_array,
            label=food.diaas / label_scale,
        ))

    return bags


def score_reference_proteins_dataframe(
    reference_foods: List[ReferenceFood],
) -> pd.DataFrame:
    """Create a DataFrame of reference proteins suitable for pipeline scoring.

    Returns a DataFrame with columns: protein_id, food_id, abundance_fraction,
    experimental_diaas, sequence.
    """
    rows = []
    for food in reference_foods:
        for prot in food.proteins:
            rows.append({
                "protein_id": prot.protein_id,
                "food_id": food.food_id,
                "food_name": food.food_name,
                "abundance_fraction": prot.abundance_fraction,
                "experimental_diaas": food.diaas,
                "experimental_pdcaas": food.pdcaas,
                "sequence": prot.sequence,
            })
    return pd.DataFrame(rows)
