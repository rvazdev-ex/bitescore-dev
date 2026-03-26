"""Gene Ontology term mapping utilities.

Supports multiple mapping sources:
- UniProt accession -> GO terms (id2go.tsv)
- Pfam accession -> GO terms (pfam2go)
- InterPro accession -> GO terms (interpro2go)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def _load_go_map(go_map_path: str | None) -> Dict[str, List[str]]:
    """Load a simple accession -> GO terms TSV mapping."""
    if not go_map_path:
        return {}
    m: Dict[str, List[str]] = {}
    p = Path(go_map_path)
    if not p.exists():
        return {}
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            acc, gos = parts[0], parts[1]
            m[acc] = [g for g in gos.split(";") if g]
    return m


def map_go_terms(
    seq: str,
    acc: str | None = None,
    go_map_path: str | None = None,
) -> dict:
    """Map a protein to GO terms via its accession."""
    if go_map_path and acc:
        m = _load_go_map(go_map_path)
        gos = m.get(acc, [])
        d: dict = {"go_term_count": len(gos)}
        for g in gos[:50]:
            d[f"go_{g}"] = 1
        return d
    return {"go_term_count": 0}


def load_pfam2go(path: Optional[str]) -> Dict[str, List[str]]:
    """Load a Pfam2GO mapping file.

    Supports:
    - Standard GO consortium pfam2go format::

        Pfam:PF00014 Kunitz_BPTI > GO:serine-type endopeptidase inhibitor activity ; GO:0004866

    - Simple TSV format::

        PF00014\tGO:0004866;GO:0005576
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    mapping: Dict[str, List[str]] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        # Standard pfam2go format
        if line.startswith("Pfam:"):
            parts = line.split()
            if len(parts) < 2:
                continue
            pfam_id = parts[0].replace("Pfam:", "")
            if "; " in line:
                go_id = line.rsplit("; ", 1)[-1].strip()
                if go_id.startswith("GO:"):
                    mapping.setdefault(pfam_id, []).append(go_id)
            continue

        # Simple TSV
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                pfam_id = parts[0].strip()
                go_ids = [g.strip() for g in parts[1].split(";") if g.strip().startswith("GO:")]
                if go_ids:
                    mapping.setdefault(pfam_id, []).extend(go_ids)

    return mapping


def load_interpro2go(path: Optional[str]) -> Dict[str, List[str]]:
    """Load an InterPro2GO mapping file.

    Supports:
    - Standard GO consortium interpro2go format::

        InterPro:IPR000001 Kringle > GO:calcium ion binding ; GO:0005509

    - Simple TSV format::

        IPR000001\tGO:0005509;GO:0004866
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    mapping: Dict[str, List[str]] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        # Standard interpro2go format
        if line.startswith("InterPro:"):
            parts = line.split()
            if len(parts) < 2:
                continue
            ipr_id = parts[0].replace("InterPro:", "")
            if "; " in line:
                go_id = line.rsplit("; ", 1)[-1].strip()
                if go_id.startswith("GO:"):
                    mapping.setdefault(ipr_id, []).append(go_id)
            continue

        # Simple TSV
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                ipr_id = parts[0].strip()
                go_ids = [g.strip() for g in parts[1].split(";") if g.strip().startswith("GO:")]
                if go_ids:
                    mapping.setdefault(ipr_id, []).extend(go_ids)

    return mapping


def resolve_go_terms_for_accession(
    accession: str,
    go_map: Dict[str, List[str]],
    pfam2go: Optional[Dict[str, List[str]]] = None,
    interpro2go: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Resolve GO terms for an accession from all available mapping sources."""
    terms: List[str] = []

    # Direct accession -> GO mapping
    terms.extend(go_map.get(accession, []))

    # Pfam accession -> GO mapping
    if pfam2go:
        terms.extend(pfam2go.get(accession, []))
        # Try without version suffix
        if "." in accession:
            terms.extend(pfam2go.get(accession.split(".")[0], []))

    # InterPro accession -> GO mapping
    if interpro2go:
        terms.extend(interpro2go.get(accession, []))

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique
