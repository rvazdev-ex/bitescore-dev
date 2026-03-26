from __future__ import annotations

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


def hmmscan_domains(
    records: List[SeqRecord],
    pfam_hmms: str,
    logger: Callable[[str], None] | None = None,
) -> Optional[dict]:
    exe = shutil.which("hmmscan")
    if exe is None:
        _log(logger, "hmmscan executable not found; skipping Pfam domain search.")
        return None

    faa = tempfile.NamedTemporaryFile(delete=False, suffix=".faa")
    tbl = tempfile.NamedTemporaryFile(delete=False, suffix=".tbl")
    faa.close()
    tbl.close()
    SeqIO.write(records, faa.name, "fasta")

    cmd = [exe, "--tblout", tbl.name, pfam_hmms, faa.name]
    _log(logger, "[hmmscan] command: " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _log_outputs(logger, "hmmscan", result.returncode, result.stdout, result.stderr)
        dom_counts: dict[str, int] = {}
        with open(tbl.name) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                hmm = parts[0]
                dom_counts[hmm] = dom_counts.get(hmm, 0) + 1
        return dom_counts if dom_counts else None
    except subprocess.CalledProcessError as exc:
        _log_outputs(logger, "hmmscan", exc.returncode, exc.stdout, exc.stderr)
        return None
    finally:
        _cleanup(Path(faa.name), Path(tbl.name))


def _cleanup(*paths: Path):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def hmmscan_detailed(
    records: List[SeqRecord],
    pfam_hmms: str,
    evalue: float = 1e-5,
    logger: Callable[[str], None] | None = None,
) -> Optional[Dict[str, List[Dict[str, object]]]]:
    """Run hmmscan returning per-query domain hits with scores and e-values.

    Returns a dict mapping query ID -> list of domain hit dicts containing:
    target_name, target_accession, evalue, score, description.
    Uses --domtblout for per-domain statistics.
    """
    exe = shutil.which("hmmscan")
    if exe is None:
        _log(logger, "hmmscan executable not found; skipping Pfam detailed search.")
        return None

    faa = tempfile.NamedTemporaryFile(delete=False, suffix=".faa")
    domtbl = tempfile.NamedTemporaryFile(delete=False, suffix=".domtbl")
    faa.close()
    domtbl.close()
    SeqIO.write(records, faa.name, "fasta")

    cmd = [
        exe,
        "--domtblout", domtbl.name,
        "-E", str(evalue),
        "--noali",
        pfam_hmms,
        faa.name,
    ]
    _log(logger, "[hmmscan] command: " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _log_outputs(logger, "hmmscan", result.returncode, result.stdout, result.stderr)

        hits_by_query: Dict[str, List[Dict[str, object]]] = {}
        with open(domtbl.name) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 23:
                    continue
                # domtblout columns:
                # 0: target name, 1: target accession, 2: tlen
                # 3: query name, 4: query accession, 5: qlen
                # 6: full E-value, 7: full score, 8: full bias
                # 9: domain #, 10: domain of, 11: c-Evalue, 12: i-Evalue
                # 13: dom score, 14: dom bias
                # 15: hmm from, 16: hmm to, 17: ali from, 18: ali to
                # 19: env from, 20: env to, 21: acc
                # 22+: description
                target_name = parts[0]
                target_acc = parts[1]
                query_name = parts[3]
                dom_evalue = float(parts[12])  # independent E-value
                dom_score = float(parts[13])
                description = " ".join(parts[22:]) if len(parts) > 22 else ""

                hits_by_query.setdefault(query_name, []).append({
                    "target_name": target_name,
                    "target_accession": target_acc,
                    "evalue": dom_evalue,
                    "score": dom_score,
                    "description": description,
                })

        return hits_by_query if hits_by_query else None
    except subprocess.CalledProcessError as exc:
        _log_outputs(logger, "hmmscan", exc.returncode, exc.stdout, exc.stderr)
        return None
    finally:
        _cleanup(Path(faa.name), Path(domtbl.name))
