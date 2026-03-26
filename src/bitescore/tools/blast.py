from __future__ import annotations

from pathlib import Path
import subprocess
import shutil
import tempfile
from typing import Callable, Optional, List

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def _write_temp_faa(records: List[SeqRecord]) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".faa")
    tmp.close()
    SeqIO.write(records, tmp.name, "fasta")
    return Path(tmp.name)


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


def diamond_top_hits(
    records: List[SeqRecord],
    db: str,
    max_targets: int = 1,
    logger: Callable[[str], None] | None = None,
) -> Optional[list[tuple[str, float]]]:
    exe = shutil.which("diamond")
    if exe is None:
        _log(logger, "diamond executable not found; skipping DIAMOND search.")
        return None

    faa = _write_temp_faa(records)
    out = Path(faa.parent) / (faa.stem + ".m8")
    cmd = [
        exe,
        "blastp",
        "-q",
        str(faa),
        "-d",
        str(db),
        "-o",
        str(out),
        "-f",
        "6",
        "qseqid",
        "sseqid",
        "bitscore",
        "-k",
        str(max_targets),
    ]
    _log(logger, "[diamond] command: " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _log_outputs(logger, "diamond", result.returncode, result.stdout, result.stderr)
        hits: list[tuple[str, float]] = []
        for line in out.read_text().splitlines():
            q, s, bit = line.split("\t")
            hits.append((s, float(bit)))
        return hits if hits else None
    except subprocess.CalledProcessError as exc:
        _log_outputs(logger, "diamond", exc.returncode, exc.stdout, exc.stderr)
        return None
    finally:
        try:
            faa.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass


def blastp_top_hits(
    records: List[SeqRecord],
    db: str,
    max_targets: int = 1,
    logger: Callable[[str], None] | None = None,
) -> Optional[list[tuple[str, float]]]:
    exe = shutil.which("blastp")
    if exe is None:
        _log(logger, "blastp executable not found; skipping BLAST search.")
        return None

    faa = _write_temp_faa(records)
    out = Path(faa.parent) / (faa.stem + ".m8")
    cmd = [
        exe,
        "-query",
        str(faa),
        "-db",
        str(db),
        "-outfmt",
        "6 qseqid sseqid bitscore",
        "-max_target_seqs",
        str(max_targets),
        "-out",
        str(out),
    ]
    _log(logger, "[blastp] command: " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _log_outputs(logger, "blastp", result.returncode, result.stdout, result.stderr)
        hits: list[tuple[str, float]] = []
        for line in out.read_text().splitlines():
            q, s, bit = line.split("\t")
            hits.append((s, float(bit)))
        return hits if hits else None
    except subprocess.CalledProcessError as exc:
        _log_outputs(logger, "blastp", exc.returncode, exc.stdout, exc.stderr)
        return None
    finally:
        try:
            faa.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
