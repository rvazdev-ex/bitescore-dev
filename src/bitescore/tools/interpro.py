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
        _cleanup(Path(faa.name), Path(out_tsv.name))


def _cleanup(*paths):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def interproscan_detailed(
    records: List[SeqRecord],
    logger: Callable[[str], None] | None = None,
) -> Optional[Dict[str, List[Dict[str, object]]]]:
    """Run InterProScan returning per-query annotation results with GO terms.

    Parses the standard InterProScan TSV output (11+ columns) extracting:
    - analysis database and signature accession/description
    - InterPro accession and description (if available)
    - GO terms (column 14, pipe-separated)

    Returns a dict mapping query ID -> list of annotation dicts.
    """
    exe = shutil.which("interproscan.sh")
    if exe is None:
        _log(logger, "interproscan.sh not found; skipping InterProScan detailed run.")
        return None

    faa = tempfile.NamedTemporaryFile(delete=False, suffix=".faa")
    out_tsv = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    faa.close()
    out_tsv.close()
    SeqIO.write(records, faa.name, "fasta")

    cmd = [exe, "-i", faa.name, "-f", "TSV", "-o", out_tsv.name, "-goterms"]
    _log(logger, "[interproscan] command: " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _log_outputs(logger, "interproscan", result.returncode, result.stdout, result.stderr)

        hits_by_query: Dict[str, List[Dict[str, object]]] = {}
        with open(out_tsv.name, newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if len(row) < 11:
                    continue
                # InterProScan TSV columns:
                # 0: protein accession
                # 1: sequence MD5
                # 2: sequence length
                # 3: analysis (database)
                # 4: signature accession
                # 5: signature description
                # 6: start location
                # 7: stop location
                # 8: score (e-value)
                # 9: status (T=true positive)
                # 10: date
                # 11: InterPro accession (optional)
                # 12: InterPro description (optional)
                # 13: GO annotations (pipe-separated, optional)
                # 14: Pathways annotations (optional)
                query_id = row[0]
                analysis_db = row[3]
                sig_acc = row[4]
                sig_desc = row[5]

                try:
                    evalue = float(row[8]) if row[8] and row[8] != "-" else 1.0
                except ValueError:
                    evalue = 1.0

                ipr_acc = row[11] if len(row) > 11 and row[11] else ""
                ipr_desc = row[12] if len(row) > 12 and row[12] else ""

                # Parse GO terms from column 14 (pipe-separated)
                go_terms = []
                if len(row) > 13 and row[13]:
                    go_terms = [
                        g.strip() for g in row[13].split("|")
                        if g.strip().startswith("GO:")
                    ]

                hits_by_query.setdefault(query_id, []).append({
                    "analysis_db": analysis_db,
                    "signature_accession": sig_acc,
                    "signature_description": sig_desc,
                    "evalue": evalue,
                    "ipr_accession": ipr_acc,
                    "ipr_description": ipr_desc,
                    "go_terms": go_terms,
                })

        return hits_by_query if hits_by_query else None
    except subprocess.CalledProcessError as exc:
        _log_outputs(logger, "interproscan", exc.returncode, exc.stdout, exc.stderr)
        return None
    finally:
        _cleanup(Path(faa.name), Path(out_tsv.name))
