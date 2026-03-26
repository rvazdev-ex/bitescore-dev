from pathlib import Path
from datetime import datetime, timezone

def log(outdir: Path, message: str):
    ts = datetime.now(timezone.utc).isoformat()
    with (outdir / "pipeline.log").open("a") as fh:
        fh.write(f"[{ts}] {message}\n")
