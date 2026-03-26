from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

STARTS = {"ATG"}
STOPS = {"TAA","TAG","TGA"}

def translate_orf(nt: str) -> str:
    pep = Seq(nt).translate(to_stop=True)
    return str(pep)

def simple_orf_caller(records: List[SeqRecord], min_len: int = 90) -> List[SeqRecord]:
    aa_records: List[SeqRecord] = []
    for rec in records:
        seq = str(rec.seq).upper().replace("U","T")
        for frame in range(3):
            i = frame
            while i+3 <= len(seq):
                codon = seq[i:i+3]
                if codon in STARTS:
                    j = i
                    while j+3 <= len(seq):
                        scod = seq[j:j+3]
                        if scod in STOPS:
                            orf = seq[i:j+3]
                            if len(orf) >= min_len:
                                pep = translate_orf(orf)
                                if pep:
                                    aa_records.append(SeqRecord(Seq(pep), id=f"{rec.id}_f{frame}_{i}-{j+3}", description="ORF"))
                            i = j+3
                            break
                        j += 3
                i += 3
    return aa_records
