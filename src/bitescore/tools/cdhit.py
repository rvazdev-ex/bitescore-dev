from __future__ import annotations
from pathlib import Path
import subprocess, shutil, tempfile
from typing import List
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def cdhit_cluster(records: List[SeqRecord], ident: float = 0.95) -> List[SeqRecord]:
    if shutil.which("cd-hit") is None:
        return records
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".faa"); tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".faa"); tmp_out.close()
    SeqIO.write(records, tmp_in.name, "fasta")
    try:
        cmd = ["cd-hit", "-i", tmp_in.name, "-o", tmp_out.name, "-c", str(ident), "-n", "5"]
        subprocess.run(cmd, check=True, capture_output=True)
        reps = list(SeqIO.parse(tmp_out.name, "fasta"))
        return reps if reps else records
    except subprocess.CalledProcessError:
        return records
