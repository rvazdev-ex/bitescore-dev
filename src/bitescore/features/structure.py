from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
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

# ---------------------------------------------------------------------------
# pLDDT confidence interpretation thresholds (AlphaFold convention)
# ---------------------------------------------------------------------------
PLDDT_VERY_LOW = 50.0   # likely disordered
PLDDT_LOW = 70.0         # low confidence
PLDDT_CONFIDENT = 90.0   # confident
# Bins: <50 very_low, 50-70 low, 70-90 confident, >=90 very_high

# ---------------------------------------------------------------------------
# Amino-acid property scales for sequence-only structural proxies
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# TOP-IDP disorder propensity scale (Campen et al., 2008)
# Higher values → more disordered
DISORDER_PROPENSITY = {
    "A": 0.06, "R": 0.18, "N": 0.01, "D": 0.19, "C": -0.20,
    "Q": 0.20, "E": 0.74, "G": 0.17, "H": -0.08, "I": -0.49,
    "L": -0.34, "K": 0.39, "M": -0.10, "F": -0.42, "P": 0.41,
    "S": 0.34, "T": 0.01, "W": -0.49, "Y": -0.31, "V": -0.46,
}

# Chou-Fasman propensity: helix, sheet, coil (P_alpha, P_beta, P_turn)
HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

SHEET_PROPENSITY = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Relative solvent accessibility (Janin, 1979) – fraction buried
BURIAL_PROPENSITY = {
    "A": 0.74, "R": 0.64, "N": 0.63, "D": 0.62, "C": 0.91,
    "Q": 0.62, "E": 0.62, "G": 0.72, "H": 0.78, "I": 0.88,
    "L": 0.85, "K": 0.52, "M": 0.85, "F": 0.88, "P": 0.64,
    "S": 0.66, "T": 0.70, "W": 0.85, "Y": 0.76, "V": 0.86,
}

# Amino acid molecular weights (Da) for rough mass calculation
AA_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_read(cache_file: Path):
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _cache_write(cache_file: Path, data: Dict):
    try:
        cache_file.write_text(json.dumps(data))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# AlphaFold EBI API lookup
# ---------------------------------------------------------------------------

def _alphafold_by_uniprot(acc: str):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            js = r.json()
            hit = js[0] if isinstance(js, list) and js else (js if isinstance(js, dict) else None)
            if not hit:
                return None
            return {
                "af_uniprot": acc,
                "af_model_created": hit.get("modelCreatedDate"),
                "af_plddt_avg": hit.get("pLDDT"),
                "af_citation_count": hit.get("citationCount"),
            }
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# PDB residue table loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Contact numbers
# ---------------------------------------------------------------------------

def _contact_numbers(coords: np.ndarray, radius: float = CA_RADIUS) -> np.ndarray:
    if coords.size == 0:
        return np.array([], dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    contacts = (dists <= radius).sum(axis=1) - 1
    return contacts.astype(float)


# ---------------------------------------------------------------------------
# Cleavage availability (structure-informed)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sequence-only structural proxies (computed for every sequence)
# ---------------------------------------------------------------------------

def _mean_property(seq: str, scale: dict, default: float = 0.0) -> float:
    """Mean of a per-residue property scale over the sequence."""
    if not seq:
        return 0.0
    vals = [scale.get(aa, default) for aa in seq]
    return sum(vals) / len(vals)


def _shannon_entropy(seq: str) -> float:
    """Shannon entropy (bits) of amino acid composition."""
    if not seq:
        return 0.0
    length = len(seq)
    counts: dict[str, int] = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _charge_features(seq: str) -> dict:
    """Net charge and charge-related features at pH 7."""
    if not seq:
        return {"net_charge": 0.0, "charge_density": 0.0, "pos_charge_frac": 0.0, "neg_charge_frac": 0.0}
    pos = sum(1 for aa in seq if aa in "KRH")
    neg = sum(1 for aa in seq if aa in "DE")
    length = len(seq)
    return {
        "net_charge": float(pos - neg),
        "charge_density": float(pos + neg) / length,
        "pos_charge_frac": float(pos) / length,
        "neg_charge_frac": float(neg) / length,
    }


def _window_score(seq: str, scale: dict, window: int = 9, default: float = 0.0) -> tuple[float, float]:
    """Return (max, min) of a sliding window average over the property scale."""
    if not seq:
        return (0.0, 0.0)
    if len(seq) <= window:
        avg = _mean_property(seq, scale, default)
        return (avg, avg)
    scores = []
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        scores.append(sum(scale.get(aa, default) for aa in w) / window)
    return (max(scores), min(scores))


def _estimated_mass_kda(seq: str) -> float:
    """Rough molecular mass in kDa from amino acid composition."""
    if not seq:
        return 0.0
    # Sum of residue masses minus (n-1) water molecules lost in peptide bonds
    water = 18.015
    total = sum(AA_MASS.get(aa, 110.0) for aa in seq)
    total -= (len(seq) - 1) * water if len(seq) > 1 else 0
    return total / 1000.0


def sequence_structural_proxies(seq: str) -> dict:
    """Compute lightweight structure-aware proxy features from sequence alone.

    These features are always available (no external tools needed) and serve as
    proxies for structural properties like disorder, compactness, and secondary
    structure content.  They are cached alongside other structure features.
    """
    length = max(len(seq), 1)

    # Basic composition proxy (legacy)
    plddt_proxy = 50 + 50 * (seq.count("H") + seq.count("P")) / length

    # Hydrophobicity
    hydro_mean = _mean_property(seq, HYDROPHOBICITY, 0.0)
    hydro_max_win, hydro_min_win = _window_score(seq, HYDROPHOBICITY, window=9)

    # Disorder propensity
    disorder_mean = _mean_property(seq, DISORDER_PROPENSITY, 0.0)
    disorder_frac = sum(1 for aa in seq if DISORDER_PROPENSITY.get(aa, 0.0) > 0.1) / length

    # Secondary structure propensities
    helix_propensity_mean = _mean_property(seq, HELIX_PROPENSITY, 1.0)
    sheet_propensity_mean = _mean_property(seq, SHEET_PROPENSITY, 1.0)
    coil_propensity = 1.0 / max(helix_propensity_mean + sheet_propensity_mean, 0.01)

    # Burial / surface accessibility proxy
    burial_mean = _mean_property(seq, BURIAL_PROPENSITY, 0.7)
    surface_accessibility_proxy = 1.0 - burial_mean

    # Sequence complexity
    aa_entropy = _shannon_entropy(seq)
    max_possible_entropy = math.log2(min(20, length)) if length > 0 else 1.0
    complexity_score = aa_entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0

    # Charge features
    charge = _charge_features(seq)

    # Molecular mass proxy
    mass_kda = _estimated_mass_kda(seq)

    # Proline and glycine content (flexibility indicators)
    pro_gly_frac = (seq.count("P") + seq.count("G")) / length

    # Cysteine content (disulfide bond potential → compactness)
    cys_frac = seq.count("C") / length

    return {
        "plddt_proxy": float(plddt_proxy),
        "hydrophobicity_mean": round(hydro_mean, 4),
        "hydrophobicity_max_window": round(hydro_max_win, 4),
        "hydrophobicity_min_window": round(hydro_min_win, 4),
        "disorder_propensity_mean": round(disorder_mean, 4),
        "disorder_prone_frac": round(disorder_frac, 4),
        "helix_propensity_mean": round(helix_propensity_mean, 4),
        "sheet_propensity_mean": round(sheet_propensity_mean, 4),
        "coil_propensity_proxy": round(coil_propensity, 4),
        "surface_accessibility_proxy": round(surface_accessibility_proxy, 4),
        "burial_propensity_mean": round(burial_mean, 4),
        "aa_entropy": round(aa_entropy, 4),
        "sequence_complexity": round(complexity_score, 4),
        "net_charge": round(charge["net_charge"], 4),
        "charge_density": round(charge["charge_density"], 4),
        "pos_charge_frac": round(charge["pos_charge_frac"], 4),
        "neg_charge_frac": round(charge["neg_charge_frac"], 4),
        "mass_kda": round(mass_kda, 3),
        "pro_gly_frac": round(pro_gly_frac, 4),
        "cys_frac": round(cys_frac, 4),
    }


# ---------------------------------------------------------------------------
# AlphaFold / PDB summary statistics
# ---------------------------------------------------------------------------

def plddt_summary_statistics(residues: list[dict]) -> dict:
    """Compute pLDDT distribution statistics from a resolved structure.

    Works with both AlphaFold and LocalColabFold PDB outputs where per-residue
    pLDDT values are stored in the B-factor column.

    Returns a dict of summary statistics that help interpret disorder and
    structural confidence at the sequence level.
    """
    if not residues:
        return _empty_plddt_stats()

    plddt_values = np.array([r["plddt"] for r in residues], dtype=float)
    n = len(plddt_values)

    # Confidence bins (AlphaFold convention)
    n_very_low = int(np.sum(plddt_values < PLDDT_VERY_LOW))
    n_low = int(np.sum((plddt_values >= PLDDT_VERY_LOW) & (plddt_values < PLDDT_LOW)))
    n_confident = int(np.sum((plddt_values >= PLDDT_LOW) & (plddt_values < PLDDT_CONFIDENT)))
    n_very_high = int(np.sum(plddt_values >= PLDDT_CONFIDENT))

    return {
        "plddt_mean": round(float(np.mean(plddt_values)), 2),
        "plddt_std": round(float(np.std(plddt_values)), 2),
        "plddt_median": round(float(np.median(plddt_values)), 2),
        "plddt_min": round(float(np.min(plddt_values)), 2),
        "plddt_max": round(float(np.max(plddt_values)), 2),
        "plddt_frac_disordered": round(n_very_low / n, 4),
        "plddt_frac_low": round(n_low / n, 4),
        "plddt_frac_confident": round(n_confident / n, 4),
        "plddt_frac_very_high": round(n_very_high / n, 4),
        "plddt_n_residues": n,
    }


def _empty_plddt_stats() -> dict:
    """Return empty pLDDT statistics when no structure is available."""
    return {
        "plddt_mean": float("nan"),
        "plddt_std": float("nan"),
        "plddt_median": float("nan"),
        "plddt_min": float("nan"),
        "plddt_max": float("nan"),
        "plddt_frac_disordered": float("nan"),
        "plddt_frac_low": float("nan"),
        "plddt_frac_confident": float("nan"),
        "plddt_frac_very_high": float("nan"),
        "plddt_n_residues": 0,
    }


# ---------------------------------------------------------------------------
# Structural metrics from 3D coordinates
# ---------------------------------------------------------------------------

def structural_geometry_metrics(residues: list[dict]) -> dict:
    """Compute structural geometry metrics from a resolved structure.

    These metrics capture global shape/compactness from the CA coordinates.
    """
    if not residues or len(residues) < 3:
        return {
            "radius_of_gyration": float("nan"),
            "contact_density": float("nan"),
            "max_distance": float("nan"),
            "mean_contact_number": float("nan"),
        }

    coords = np.array([r["coord"] for r in residues], dtype=float)
    n = len(coords)

    # Radius of gyration
    centroid = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))

    # Contact density
    contacts = _contact_numbers(coords, CA_RADIUS)
    mean_contacts = float(np.mean(contacts))
    contact_density = mean_contacts / max(n, 1)

    # Max pairwise distance (end-to-end spread)
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    max_dist = float(np.max(dists))

    return {
        "radius_of_gyration": round(rg, 3),
        "contact_density": round(contact_density, 4),
        "max_distance": round(max_dist, 3),
        "mean_contact_number": round(mean_contacts, 2),
    }


# ---------------------------------------------------------------------------
# Main entry point: structure_features()
# ---------------------------------------------------------------------------

def structure_features(
    seq: str,
    seq_id: str,
    alphafold_enabled: bool,
    cache_dir: Path,
    threads: int | None = None,
) -> dict:
    """Compute all structure-related features for a single sequence.

    This function produces three layers of features:
    1. Sequence-only structural proxies (always computed)
    2. AlphaFold summary statistics (when AlphaFold is enabled and a hit exists)
    3. Full 3D structure analysis (when localcolabfold is available)

    All results are cached by sequence hash.
    """
    h = hashlib.sha256(seq.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"{h}.json"
    cached = _cache_read(cache_file)
    if cached:
        return cached

    # Layer 1: Sequence-only structural proxies (always available)
    data: dict[str, object] = {
        "struct_hash": h,
        "structure_source": "none",
    }
    data.update(sequence_structural_proxies(seq))

    # Layer 2: AlphaFold DB lookup (optional)
    if alphafold_enabled:
        parts = re.split(r"[|\s]", str(seq_id))
        candidates = [p for p in parts if UNIPROT_RE.match(p)]
        for acc in candidates:
            af = _alphafold_by_uniprot(acc)
            if af:
                data.update(af)
                data["structure_source"] = "alphafold"
                break

    # Layer 3: LocalColabFold structure prediction
    local_cache = cache_dir / LOCALCOLABFOLD_CACHE
    pdb_path = predict_structure(seq, seq_id, local_cache, threads=threads)
    residues = []
    if pdb_path is not None:
        residues = _load_residue_table(pdb_path)
        if residues:
            data["structure_source"] = "localcolabfold"
            data["predicted_structure_path"] = str(pdb_path)

    # AlphaFold summary statistics from PDB (works for both AF and ColabFold)
    plddt_stats = plddt_summary_statistics(residues)
    data.update(plddt_stats)

    # Structural geometry metrics from 3D coordinates
    geom = structural_geometry_metrics(residues)
    data.update(geom)

    # Cleavage accessibility (structure-informed)
    data.update(_cleavage_availability(seq, residues))

    _cache_write(cache_file, data)
    return data
