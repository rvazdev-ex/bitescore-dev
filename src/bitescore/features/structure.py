from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from typing import Dict, Iterable

import numpy as np
from Bio.PDB import PDBParser
import requests

from .cleavage import DEFAULT_PROTEASES, cleavage_site_positions
from ..tools.localcolabfold import predict_structure

LOCALCOLABFOLD_CACHE = "localcolabfold"
CA_RADIUS = 8.0
CONTACT_THRESHOLD = 18
PLDDT_THRESHOLD = 70.0

UNIPROT_RE = re.compile(r"^[A-NR-Z0-9]{6,10}$")

def _cache_read(cache_file: Path):
    if cache_file.exists():
        try: return json.loads(cache_file.read_text())
        except: return None
    return None
def _cache_write(cache_file: Path, data: Dict):
    try: cache_file.write_text(json.dumps(data))
    except: pass

def _alphafold_by_uniprot(acc: str):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            js = r.json()
            hit = js[0] if isinstance(js, list) and js else (js if isinstance(js, dict) else None)
            if not hit: return None
            return {
                "af_uniprot": acc,
                "af_model_created": hit.get("modelCreatedDate"),
                "af_plddt_avg": hit.get("pLDDT"),
                "af_citation_count": hit.get("citationCount"),
            }
    except Exception:
        return None
    return None

def _load_residue_table(pdb_path: Path) -> list[dict]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("model", str(pdb_path))
    except Exception:
        return []

    residues: list[dict] = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                if "CA" not in res:
                    continue
                atom = res["CA"]
                residues.append(
                    {
                        "chain": chain.id,
                        "coord": atom.coord.copy(),
                        "plddt": float(atom.bfactor),
                    }
                )
    return residues


def _contact_numbers(coords: np.ndarray, radius: float = CA_RADIUS) -> np.ndarray:
    if coords.size == 0:
        return np.array([], dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    contacts = (dists <= radius).sum(axis=1) - 1
    return contacts.astype(float)


def _cleavage_availability(
    seq: str,
    residues: Iterable[dict],
    proteases=DEFAULT_PROTEASES,
    contact_threshold: int = CONTACT_THRESHOLD,
    plddt_threshold: float = PLDDT_THRESHOLD,
) -> dict:
    cleavage_positions = cleavage_site_positions(seq, proteases=proteases)
    if not residues or not cleavage_positions:
        return {
            "cleavage_sites_with_structure": 0,
            "cleavage_sites_accessible": 0,
            "cleavage_site_accessible_fraction": 0.0,
        }

    coords = np.array([entry["coord"] for entry in residues], dtype=float)
    contacts = _contact_numbers(coords)

    accessible = 0
    observed = 0
    for pos in cleavage_positions:
        if pos >= len(residues):
            continue
        observed += 1
        if contacts[pos] <= contact_threshold and residues[pos]["plddt"] >= plddt_threshold:
            accessible += 1

    fraction = float(accessible / observed) if observed else 0.0
    return {
        "cleavage_sites_with_structure": observed,
        "cleavage_sites_accessible": accessible,
        "cleavage_site_accessible_fraction": fraction,
    }


def structure_features(
    seq: str,
    seq_id: str,
    alphafold_enabled: bool,
    cache_dir: Path,
    threads: int | None = None,
) -> dict:
    h = hashlib.sha256(seq.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"{h}.json"
    cached = _cache_read(cache_file)
    if cached: return cached
    plddt_proxy = 50 + 50*(seq.count('H') + seq.count('P'))/max(len(seq),1)
    data: dict[str, object] = {
        "struct_hash": h,
        "plddt_proxy": float(plddt_proxy),
        "structure_source": "none",
    }
    if alphafold_enabled:
        parts = re.split(r"[|\s]", str(seq_id))
        candidates = [p for p in parts if UNIPROT_RE.match(p)]
        for acc in candidates:
            af = _alphafold_by_uniprot(acc)
            if af:
                data.update(af)
                data["structure_source"] = "alphafold"
                break

    local_cache = cache_dir / LOCALCOLABFOLD_CACHE
    pdb_path = predict_structure(seq, seq_id, local_cache, threads=threads)
    residues = []
    if pdb_path is not None:
        residues = _load_residue_table(pdb_path)
        if residues:
            data["structure_source"] = "localcolabfold"
            data["predicted_structure_path"] = str(pdb_path)

    data.update(_cleavage_availability(seq, residues))
    _cache_write(cache_file, data)
    return data
