from pathlib import Path
from datetime import datetime

def log(outdir: Path, message: str):
    ts = datetime.utcnow().isoformat()
    (outdir / "pipeline.log").open("a").write(f"[{ts}] {message}\n")
