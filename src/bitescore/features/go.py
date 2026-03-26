from __future__ import annotations
from pathlib import Path

def _load_go_map(go_map_path: str | None):
    if not go_map_path:
        return {}
    m = {}
    p = Path(go_map_path)
    if not p.exists():
        return {}
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            acc, gos = parts[0], parts[1]
            m[acc] = [g for g in gos.split(";") if g]
    return m

def map_go_terms(seq: str, acc: str | None = None, go_map_path: str | None = None) -> dict:
    if go_map_path and acc:
        m = _load_go_map(go_map_path)
        gos = m.get(acc, [])
        d = {"go_term_count": len(gos)}
        for g in gos[:50]:
            d[f"go_{g}"] = 1
        return d
    return {"go_term_count": 0}
