from __future__ import annotations

from pathlib import Path
import subprocess
import shutil
import tempfile
from typing import Callable, List, Optional

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
        try:
            Path(faa.name).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            Path(tbl.name).unlink(missing_ok=True)
        except OSError:
            pass
