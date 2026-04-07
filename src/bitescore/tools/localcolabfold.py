"""Lightweight wrapper for optional localcolabfold predictions."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


def _find_output_pdb(work_dir: Path) -> Path | None:
    """Find a predicted structure PDB from common localcolabfold layouts."""
    preferred_names = ("ranked_0.pdb", "result_model_1_ptm_pred_0.pdb")
    for name in preferred_names:
        path = work_dir / name
        if path.exists():
            return path
    for name in preferred_names:
        matches = list(work_dir.rglob(name))
        if matches:
            return matches[0]
    pdb_files = sorted(work_dir.rglob("*.pdb"))
    return pdb_files[0] if pdb_files else None


def _which_localcolabfold() -> str | None:
    env_override = os.environ.get("LOCALCOLABFOLD_BIN")
    if env_override:
        return env_override
    return "localcolabfold"


def localcolabfold_status(binary_override: str | None = None) -> dict[str, Any]:
    """Return runtime availability information for localcolabfold."""
    env_override = os.environ.get("LOCALCOLABFOLD_BIN")
    configured_binary = binary_override or env_override or "localcolabfold"
    resolved = shutil.which(configured_binary)
    available = bool(resolved) or Path(configured_binary).exists()
    return {
        "available": bool(available),
        "binary": configured_binary,
        "resolved_path": str(resolved) if resolved else (configured_binary if Path(configured_binary).exists() else None),
        "env_override": env_override,
    }


def predict_structure(
    sequence: str,
    seq_id: str,
    cache_dir: Path,
    threads: int | None = None,
    logger: Callable[[str], None] | None = None,
    timeout_seconds: int | None = None,
    binary: str | None = None,
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

    executable = binary or _which_localcolabfold()
    if shutil.which(executable) is None and not Path(executable).exists():
        if logger:
            logger(f"localcolabfold binary not found ({executable}); skipping structure prediction.")
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    work_dir = cache_dir / seq_hash
    existing = _find_output_pdb(work_dir)
    if existing:
        return existing

    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "query.fasta"
    fasta_path.write_text(f">{seq_id}\n{sequence}\n")

    cmd = [executable, str(fasta_path), str(work_dir)]
    executable_name = Path(executable).name.lower()
    supports_cpu_flag = executable_name == "localcolabfold" or executable_name.startswith("localcolabfold.")
    if threads and supports_cpu_flag:
        cmd.extend(["--cpu", str(int(threads))])
    elif threads and logger:
        logger(f"Skipping --cpu flag for '{executable_name}' (not supported by this binary).")
    if executable_name == "colabfold_batch":
        # Stable defaults for recent GPU+WSL setups where default settings
        # can crash with segmentation faults.
        cmd.extend(["--num-recycle", "1", "--num-models", "1", "--disable-unified-memory"])

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
    except FileNotFoundError:
        if logger:
            logger("localcolabfold executable missing; ensure it is installed.")
        return None
    except subprocess.TimeoutExpired:
        if logger:
            logger(
                "localcolabfold timed out"
                + (f" after {int(timeout_seconds)}s." if timeout_seconds else ".")
            )
        return None
    except subprocess.CalledProcessError as exc:
        if logger:
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            if stderr:
                logger(f"localcolabfold failed with exit code {exc.returncode}: {stderr}")
            else:
                logger(f"localcolabfold failed with exit code {exc.returncode}.")
        return None

    return _find_output_pdb(work_dir)


__all__ = ["predict_structure", "localcolabfold_status"]
