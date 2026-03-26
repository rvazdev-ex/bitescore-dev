from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bitescore.app.chatbot import _ensure_record_ids


def test_ensure_record_ids_generates_missing_ids_and_descriptions():
    rec1 = SeqRecord(Seq("AAA"), id="", description="")
    rec2 = SeqRecord(Seq("CCC"), id="  seq2  ", description="   ")
    rec3 = SeqRecord(Seq("GGG"), id="seq3", description="seq3 description")

    records, updated = _ensure_record_ids([rec1, rec2, rec3])

    assert updated is True
    assert records[0].id == "sequence_1"
    assert records[0].name == "sequence_1"
    assert records[0].description == "sequence_1"

    assert records[1].id == "seq2"
    assert records[1].name == "seq2"
    assert records[1].description == "seq2"

    assert records[2].id == "seq3"
    assert records[2].name == "seq3"
    assert records[2].description == "seq3 description"

    # Calling again should not mark records as updated
    second_pass, second_updated = _ensure_record_ids(records)
    assert second_pass is records
    assert second_updated is False
