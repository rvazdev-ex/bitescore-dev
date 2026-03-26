import json

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bitescore.features.extract import compute_function_features


def _single_record_df(sequence: str, **kwargs):
    record = SeqRecord(Seq(sequence), id="test_protein")
    return compute_function_features([record], **kwargs)


def test_uniprot_curated_mapping(tmp_path):
    sequence = "MKAILVVLLYTFATANADNKD"
    go_terms = [
        {
            "go_id": "GO:0005509",
            "name": "calcium ion binding",
            "aspect": "MF",
            "evidence_code": "EXP",
            "source": "PMID:12345",
        },
        {
            "go_id": "GO:0005576",
            "name": "extracellular region",
            "aspect": "CC",
            "evidence_code": "IDA",
        },
    ]
    go_map = tmp_path / "go_map.tsv"
    go_map.write_text(f"P00001\t{sequence}\tSwiss-Prot\t{json.dumps(go_terms)}\n")

    df = _single_record_df(sequence, go_map_path=str(go_map))
    row = df.iloc[0]

    assert row["best_uniprot_accession"] == "P00001"
    assert row["best_uniprot_identity"] == 100.0
    assert bool(row["green_flag"]) is True
    assert bool(row["red_flag"]) is False
    assert row["functional_label"].startswith("Secreted/extracellular")

    terms = {entry["go_id"]: entry for entry in json.loads(row["go_terms_json"])}
    assert terms["GO:0005509"]["confidence_score"] == 0.95
    assert terms["GO:0005509"]["source_method"] == "UniProt"
    assert row["top_mf_go_id"] == "GO:0005509"


def test_interpro_red_flag_annotation():
    sequence = "MKKLLLLVVAVVLALLAKUNITZCG"
    df = _single_record_df(sequence)
    row = df.iloc[0]

    assert bool(row["red_flag"]) is True
    assert "⚠️" in row["functional_label"]

    red_terms = json.loads(row["red_flag_terms"])
    assert "GO:0004866" in red_terms

    terms = {entry["go_id"]: entry for entry in json.loads(row["go_terms_json"])}
    assert terms["GO:0004866"]["source_method"] == "InterPro2GO"
    assert terms["GO:0004866"]["confidence_score"] == 0.65


def test_blast_consensus_annotation():
    sequence = "MSTRTKQLTAALREKLEELAAALKKA"
    df = _single_record_df(sequence)
    row = df.iloc[0]

    assert bool(row["red_flag"]) is False
    assert bool(row["green_flag"]) is False
    assert "BLAST2GO" in row["annotation_methods"].split(",")

    terms = {entry["go_id"]: entry for entry in json.loads(row["go_terms_json"])}
    assert "GO:0003677" in terms
    assert terms["GO:0003677"]["confidence_score"] == 0.55
    assert terms["GO:0003677"]["source_method"] == "BLAST2GO"

    assert "DNA binding" in row["functional_label"]
