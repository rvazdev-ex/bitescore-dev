from __future__ import annotations
import subprocess, shutil, tempfile
from typing import List
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def segmask(records: List[SeqRecord]) -> List[SeqRecord]:
    exe = shutil.which("segmasker")
    if exe is None:
        return records
    fa = tempfile.NamedTemporaryFile(delete=False, suffix=".faa"); fa.close()
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".faa"); out.close()
    SeqIO.write(records, fa.name, "fasta")
    try:
        cmd = [exe, "-in", fa.name, "-out", out.name, "-outfmt", "fasta"]
        subprocess.run(cmd, check=True, capture_output=True)
        masked = list(SeqIO.parse(out.name, "fasta"))
        return masked if masked else records
    except subprocess.CalledProcessError:
        return records
