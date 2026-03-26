from typing import Callable, List, Optional, Union
from pathlib import Path
import tempfile
import subprocess
import shutil
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def _write_temp_fasta(records: List[SeqRecord], suffix: str = ".fna") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    SeqIO.write(records, tmp.name, "fasta")
    return Path(tmp.name)


LogFn = Optional[Callable[[str], None]]


def _log_stream(logger: LogFn, stream_name: str, content: Optional[Union[str, bytes]]):
    if logger is None or not content:
        return
    if isinstance(content, bytes):
        try:
            content = content.decode()
        except Exception:
            content = content.decode(errors="replace")
    stripped = content.strip()
    if not stripped:
        return
    logger(f"Prodigal {stream_name}:")
    for line in stripped.splitlines():
        logger(f"  {line}")


def call_prodigal(records: List[SeqRecord], logger: LogFn = None) -> Optional[List[SeqRecord]]:
    if shutil.which("prodigal") is None:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_fa = tmp_path / "input.fna"
        out_prot = tmp_path / "prodigal.faa"
        SeqIO.write(records, str(in_fa), "fasta")
        cmd = ["prodigal", "-i", str(in_fa), "-a", str(out_prot), "-p", "meta"]
        if logger is not None:
            logger(f"Running Prodigal: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            _log_stream(logger, "stdout", res.stdout)
            _log_stream(logger, "stderr", res.stderr)
        except subprocess.CalledProcessError as exc:
            if logger is not None:
                logger(f"Prodigal failed with return code {exc.returncode}")
                _log_stream(logger, "stdout", exc.stdout)
                _log_stream(logger, "stderr", exc.stderr)
            return None
        if not out_prot.exists():
            return None
        aa = list(SeqIO.parse(str(out_prot), "fasta"))
        return aa if aa else None

def call_augustus(records: List[SeqRecord]) -> Optional[List[SeqRecord]]:
    if shutil.which("augustus") is None:
        return None
    import os
    species = os.environ.get("BITESCORE_AUG_SPECIES", "human")
    in_fa = _write_temp_fasta(records, ".fna")
    out_faa = Path(in_fa.parent) / (in_fa.stem + ".augustus.faa")
    try:
        res2 = subprocess.run(["augustus", "--protein=on", f"--species={species}", str(in_fa)], check=True, capture_output=True, text=True)
        lines = res2.stdout.splitlines()
        fa_lines = []
        keep = False
        for line in lines:
            if line.startswith(">"):
                keep = True
            if keep:
                fa_lines.append(line)
        if fa_lines:
            out_faa.write_text("\n".join(fa_lines) + "\n")
            aa = list(SeqIO.parse(out_faa, "fasta"))
            return aa if aa else None
    except subprocess.CalledProcessError:
        return None
    finally:
        try: in_fa.unlink(missing_ok=True)
        except: pass
    return None
