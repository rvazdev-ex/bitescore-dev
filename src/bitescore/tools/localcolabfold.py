"""Lightweight wrapper for optional localcolabfold predictions."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def _which_localcolabfold() -> str | None:
    env_override = os.environ.get("LOCALCOLABFOLD_BIN")
    if env_override:
        return env_override
    return "localcolabfold"


def predict_structure(
    sequence: str,
    seq_id: str,
    cache_dir: Path,
    threads: int | None = None,
    logger: Callable[[str], None] | None = None,
) -> Path | None:
    """Predict a structure using localcolabfold when available.

    Parameters
    ----------
    sequence:
        Amino acid sequence.
    seq_id:
        Sequence identifier used for logging; not used by the predictor.
    cache_dir:
        Directory used to store prediction results.
    threads:
        Optional number of CPU threads to request from localcolabfold.
    logger:
        Optional logging callback.

    Returns
    -------
    pathlib.Path | None
        Path to a predicted PDB file, or ``None`` if prediction was not
        possible.
    """

    binary = _which_localcolabfold()
    if shutil.which(binary) is None and not Path(binary).exists():
        if logger:
            logger("localcolabfold binary not found; skipping structure prediction.")
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    work_dir = cache_dir / seq_hash
    pdb_targets = [work_dir / name for name in ("ranked_0.pdb", "result_model_1_ptm_pred_0.pdb")]
    for target in pdb_targets:
        if target.exists():
            return target

    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "query.fasta"
    fasta_path.write_text(f">{seq_id}\n{sequence}\n")

    cmd = [binary, str(fasta_path), str(work_dir)]
    if threads:
        cmd.extend(["--cpu", str(int(threads))])

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        if logger:
            logger("localcolabfold executable missing; ensure it is installed.")
        return None
    except subprocess.CalledProcessError as exc:
        if logger:
            logger(f"localcolabfold failed with exit code {exc.returncode}.")
        return None

    for target in pdb_targets:
        if target.exists():
            return target

    pdb_files = list(work_dir.glob("*.pdb"))
    return pdb_files[0] if pdb_files else None


__all__ = ["predict_structure"]
