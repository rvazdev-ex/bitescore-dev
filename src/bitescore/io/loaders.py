from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def load_inputs(path: str, input_type: str):
    p = Path(path)
    if input_type in {"proteome","genome"}:
        return list(SeqIO.parse(p, "fasta"))
    elif input_type == "sequences":
        recs = []
        with open(p) as fh:
            for i, line in enumerate(fh, 1):
                s = line.strip()
                if not s: continue
                recs.append(SeqRecord(Seq(s), id=f"seq_{i}", description=""))
        return recs
    elif input_type in {"genomes","metagenome"}:
        if p.is_dir():
            recs = []
            for fp in sorted(p.glob("*.fa*")):
                recs.extend(list(SeqIO.parse(fp, "fasta")))
            return recs
        else:
            return list(SeqIO.parse(p, "fasta"))
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
