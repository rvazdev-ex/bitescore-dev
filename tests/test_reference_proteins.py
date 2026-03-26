"""Tests for the built-in reference protein data module."""

from bitescore.data.reference_proteins import (
    REFERENCE_FOODS,
    ReferenceFood,
    ReferenceProtein,
    get_all_reference_proteins,
    get_reference_food_by_id,
    reference_proteins_as_seqrecords,
)


def test_reference_foods_non_empty():
    assert len(REFERENCE_FOODS) >= 5


def test_each_food_has_proteins():
    for food in REFERENCE_FOODS:
        assert isinstance(food, ReferenceFood)
        assert len(food.proteins) >= 1
        assert food.diaas > 0
        assert 0 < food.pdcaas <= 1.0 or food.diaas > 100


def test_abundance_fractions_sum():
    for food in REFERENCE_FOODS:
        total = sum(p.abundance_fraction for p in food.proteins)
        # Allow partial (some proteins may not be included)
        assert 0.0 < total <= 1.01, f"{food.food_id}: abundances sum to {total}"


def test_protein_sequences_are_amino_acids():
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for prot in get_all_reference_proteins():
        seq_chars = set(prot.sequence.upper())
        invalid = seq_chars - valid_aa
        assert not invalid, f"{prot.protein_id}: invalid chars {invalid}"
        assert len(prot.sequence) > 20, f"{prot.protein_id}: too short"


def test_get_all_reference_proteins_unique():
    prots = get_all_reference_proteins()
    ids = [p.protein_id for p in prots]
    assert len(ids) == len(set(ids)), "Duplicate protein IDs"


def test_get_reference_food_by_id():
    food = get_reference_food_by_id("whole_milk")
    assert food is not None
    assert food.food_name == "Whole milk"
    assert get_reference_food_by_id("nonexistent") is None


def test_reference_proteins_as_seqrecords():
    records = reference_proteins_as_seqrecords()
    assert len(records) > 0
    for rec in records:
        assert len(str(rec.seq)) > 20
        assert "ref|" in rec.description


def test_diaas_range():
    """DIAAS values should be in a realistic range."""
    for food in REFERENCE_FOODS:
        assert 10 <= food.diaas <= 150, f"{food.food_id}: DIAAS={food.diaas} out of range"
