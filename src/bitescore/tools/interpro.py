from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import shutil
import tempfile
from typing import Callable, Dict, List, Optional

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def _log(logger: Callable[[str], None] | None, message: str):
    if logger:
        logger(message)


def _log_outputs(
    logger: Callable[[str], None] | None,
    tool: str,
    returncode: int,
    stdout: Optional[str],
    stderr: Optional[str],
):
    if logger is None:
        return
    logger(f"[{tool}] return code: {returncode}")
    if stdout:
        logger(f"[{tool}] stdout:\n{stdout.strip()}")
    else:
        logger(f"[{tool}] stdout: <empty>")
    if stderr:
        logger(f"[{tool}] stderr:\n{stderr.strip()}")
    else:
        logger(f"[{tool}] stderr: <empty>")


def interproscan(
    records: List[SeqRecord],
    logger: Callable[[str], None] | None = None,
) -> Optional[Dict[str, int]]:
    exe = shutil.which("interproscan.sh")
    if exe is None:
        _log(logger, "interproscan executable not found; skipping InterProScan run.")
        return None

    faa = tempfile.NamedTemporaryFile(delete=False, suffix=".faa")
    out_tsv = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    faa.close()
    out_tsv.close()
    SeqIO.write(records, faa.name, "fasta")

    cmd = [exe, "-i", faa.name, "-f", "TSV", "-o", out_tsv.name]
    _log(logger, "[interproscan] command: " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _log_outputs(logger, "interproscan", result.returncode, result.stdout, result.stderr)
        features: Dict[str, int] = {}
        with open(out_tsv.name, newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if len(row) < 12:
                    continue
                db = row[3]
                ipr = row[11]
                key = f"db_{db}_count"
                features[key] = features.get(key, 0) + 1
                if ipr:
                    features["interpro_count"] = features.get("interpro_count", 0) + 1
        return features if features else None
    except subprocess.CalledProcessError as exc:
        _log_outputs(logger, "interproscan", exc.returncode, exc.stdout, exc.stderr)
        return None
    finally:
        try:
            Path(faa.name).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            Path(out_tsv.name).unlink(missing_ok=True)
        except OSError:
            pass
