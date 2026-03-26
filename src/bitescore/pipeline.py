from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from .io.loaders import load_inputs
from .gene_callers.call import call_genes_if_needed
from .features.extract import (
    compute_aa_features,
    compute_regsite_features,
    compute_structure_feature_table,
    compute_function_features,
    merge_feature_frames,
)
from .ml.rank import rank_sequences
from .utils.logging import log
from .tools.cdhit import cdhit_cluster
from .tools.low_complexity import segmask

GENOME_INPUT_TYPES = {"genome", "genomes", "metagenome"}
AA_INPUT_TYPES = {"proteome", "sequences"}
ALL_INPUT_TYPES = ["genome", "genomes", "metagenome", "proteome", "sequences"]


def _loaded_suffix(input_type: str) -> str:
    if input_type in GENOME_INPUT_TYPES:
        return ".fna"
    if input_type in AA_INPUT_TYPES:
        return ".faa"
    raise ValueError(f"Unsupported input_type: {input_type}")


def path_loaded(outdir: Path, input_type: str) -> Path:
    return outdir / f"loaded{_loaded_suffix(input_type)}"
def path_called(outdir: Path) -> Path: return outdir / "called.faa"
def path_clustered(outdir: Path) -> Path: return outdir / "clustered.faa"
def path_masked(outdir: Path) -> Path: return outdir / "masked.faa"
def path_features(outdir: Path) -> Path: return outdir / "features.csv"
def path_features_aa(outdir: Path) -> Path: return outdir / "features_aa.csv"
def path_features_regsite(outdir: Path) -> Path: return outdir / "features_regsite.csv"
def path_features_structure(outdir: Path) -> Path: return outdir / "features_structure.csv"
def path_features_function(outdir: Path) -> Path: return outdir / "features_function.csv"
def path_ranked(outdir: Path) -> Path: return outdir / "ranked.csv"

def _write_fasta(records: List[SeqRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(path), "fasta")


def _read_fasta(path: Path) -> List[SeqRecord]:
    return list(SeqIO.parse(str(path), "fasta"))


def _load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature table: {path}")
    return pd.read_csv(path)


def _resolve_input_type(outdir: Path, configured: str | None) -> str:
    if configured:
        return configured
    for candidate in ALL_INPUT_TYPES:
        candidate_path = path_loaded(outdir, candidate)
        if candidate_path.exists():
            return candidate
    raise ValueError(
        "Unable to determine input_type. Provide it via config/CLI or run 'bitescore load' first."
    )

def step_load(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Loading inputs")
    input_type = cfg["input_type"]
    recs = load_inputs(cfg["input_path"], input_type)
    dest = path_loaded(outdir, input_type)
    _write_fasta(recs, dest)
    log(outdir, f"Loaded {len(recs)} records -> {dest}")

def step_call_genes(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Gene calling step")
    loaded_override = cfg.get("loaded_path")
    if loaded_override:
        src = Path(loaded_override)
        if not src.exists():
            raise FileNotFoundError(f"Configured loaded_path does not exist: {src}")
        input_type = cfg.get("input_type")
        if input_type is None:
            raise ValueError("When specifying a custom loaded_path you must also provide input_type via the CLI or config.")
    else:
        input_type = _resolve_input_type(outdir, cfg.get("input_type"))
        src = path_loaded(outdir, input_type)
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}. Run 'bitescore load' first or the full pipeline.")
    if input_type not in GENOME_INPUT_TYPES:
        log(outdir, f"Input type '{input_type}' does not require gene calling; skipping.")
        return
    recs = _read_fasta(src)
    pipeline_logger = lambda message: log(outdir, message)
    aa = call_genes_if_needed(recs, input_type, cfg.get("organism"), logger=pipeline_logger)
    dest = path_called(outdir)
    _write_fasta(aa, dest)
    log(outdir, f"Gene calling -> {len(aa)} AA sequences -> {dest}")

def _feature_base_path(outdir: Path, input_type: str) -> Path:
    """Return the baseline FASTA path for feature generation."""
    called = path_called(outdir)
    if input_type in GENOME_INPUT_TYPES:
        if not called.exists():
            raise FileNotFoundError(
                f"Missing {called}. Gene calling outputs are required for genome-like inputs."
            )
        return called
    src = path_loaded(outdir, input_type)
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run previous steps or full pipeline.")
    return src


def _feature_sequences_for_extraction(cfg: dict, outdir: Path, input_type: str | None) -> Path:
    override = cfg.get("feature_sequences")
    if override:
        override_path = Path(override)
        if not override_path.exists():
            raise FileNotFoundError(f"Configured feature_sequences path does not exist: {override_path}")
        return override_path
    if input_type is None:
        input_type = _resolve_input_type(outdir, cfg.get("input_type"))
    mask = path_masked(outdir)
    if mask.exists():
        return mask
    clustered = path_clustered(outdir)
    if clustered.exists():
        return clustered
    return _feature_base_path(outdir, input_type)


def step_features_cluster(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "CD-HIT clustering step")
    input_type = _resolve_input_type(outdir, cfg.get("input_type"))
    src = _feature_base_path(outdir, input_type)
    recs = _read_fasta(src)
    thr = cfg.get("cdhit_threshold") or 0.95
    recs = cdhit_cluster(recs, ident=thr)
    dest = path_clustered(outdir)
    _write_fasta(recs, dest)
    mask_path = path_masked(outdir)
    if mask_path.exists():
        mask_path.unlink()
        log(outdir, f"Removed stale masked sequences: {mask_path}")
    log(outdir, f"CD-HIT clustering -> {len(recs)} representatives -> {dest}")


def step_features_mask(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Low-complexity masking step")
    input_type = _resolve_input_type(outdir, cfg.get("input_type"))
    clustered = path_clustered(outdir)
    if clustered.exists():
        src = clustered
    else:
        src = _feature_base_path(outdir, input_type)
    recs = _read_fasta(src)
    recs = segmask(recs)
    dest = path_masked(outdir)
    _write_fasta(recs, dest)
    log(outdir, f"Low-complexity masking applied -> {dest}")


def step_features_extract(cfg: dict):
    raise NotImplementedError("step_features_extract has been replaced by specific feature steps")


def assemble_feature_tables(outdir: Path) -> pd.DataFrame:
    outdir = Path(outdir)
    required = {
        "features-aa": path_features_aa(outdir),
        "features-regsite": path_features_regsite(outdir),
        "features-structure": path_features_structure(outdir),
        "features-function": path_features_function(outdir),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing feature tables: " + ", ".join(missing) + ". Run the corresponding feature steps first."
        )
    frames = [pd.read_csv(path) for path in required.values()]
    combined = merge_feature_frames(frames)
    combined.to_csv(path_features(outdir), index=False)
    return combined


def _update_combined_features_if_ready(outdir: Path) -> bool:
    try:
        assemble_feature_tables(outdir)
    except FileNotFoundError:
        return False
    return True


def assemble_ranking_features(outdir: Path, overrides: dict | None = None) -> pd.DataFrame:
    """Join the feature tables needed for ranking.

    The ranking stage focuses on nutritional quality and digestibility
    proxies, so only a subset of the extracted feature columns are needed.
    """

    outdir = Path(outdir)
    overrides = overrides or {}

    def _feature_path(name: str, default_path: Path) -> Path:
        override_value = overrides.get(f"features_{name}_path")
        if override_value:
            return Path(override_value)
        return default_path

    tables: Dict[str, tuple[pd.DataFrame, Sequence[str]]] = {}
    path_map: Dict[str, Path] = {
        "aa": _feature_path("aa", path_features_aa(outdir)),
        "regsite": _feature_path("regsite", path_features_regsite(outdir)),
        "structure": _feature_path("structure", path_features_structure(outdir)),
        "function": _feature_path("function", path_features_function(outdir)),
    }

    missing = [name for name, path in path_map.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    tables = {
        "aa": (_load_feature_table(path_map["aa"]), ["aa_essential_frac"]),
        "regsite": (
            _load_feature_table(path_map["regsite"]),
            ["protease_total_sites", "trypsin_K_sites", "trypsin_R_sites"],
        ),
        "structure": (
            _load_feature_table(path_map["structure"]),
            ["cleavage_site_accessible_fraction"],
        ),
        "function": (
            _load_feature_table(path_map["function"]),
            ["red_flag", "green_flag"],
        ),
    }

    merged: pd.DataFrame | None = None
    for name, (df, required_cols) in tables.items():
        if "id" not in df.columns:
            raise ValueError(f"Feature table '{name}' is missing an 'id' column")
        frame = df.set_index("id").copy()
        present = [c for c in required_cols if c in frame.columns]
        missing = [c for c in required_cols if c not in frame.columns]
        subset = frame[present].copy() if present else pd.DataFrame(index=frame.index)
        for col in missing:
            subset[col] = 0
        if merged is None:
            merged = subset
        else:
            merged = merged.join(subset, how="outer")

    if merged is None:
        return pd.DataFrame(columns=["id"])

    merged = merged.fillna(0)
    for boolean_col in ("red_flag", "green_flag"):
        if boolean_col in merged.columns:
            merged[boolean_col] = merged[boolean_col].astype(bool)

    return merged.reset_index().rename(columns={"index": "id"})


def step_features_aa(cfg: dict, assemble: bool = True) -> pd.DataFrame:
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Amino acid composition features")
    input_type = cfg.get("input_type")
    src = _feature_sequences_for_extraction(cfg, outdir, input_type)
    recs = _read_fasta(src)
    aa_df = compute_aa_features(recs)
    aa_df.to_csv(path_features_aa(outdir), index=False)
    log(outdir, f"AA features saved: {path_features_aa(outdir)}")
    if assemble and _update_combined_features_if_ready(outdir):
        log(outdir, f"Combined features saved: {path_features(outdir)}")
    return aa_df


def step_features_regsite(cfg: dict, assemble: bool = True) -> pd.DataFrame:
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Protease recognition site features")
    input_type = cfg.get("input_type")
    src = _feature_sequences_for_extraction(cfg, outdir, input_type)
    recs = _read_fasta(src)
    reg_df = compute_regsite_features(recs)
    reg_df.to_csv(path_features_regsite(outdir), index=False)
    log(outdir, f"Recognition site features saved: {path_features_regsite(outdir)}")
    if assemble and _update_combined_features_if_ready(outdir):
        log(outdir, f"Combined features saved: {path_features(outdir)}")
    return reg_df


def step_features_structure(cfg: dict, assemble: bool = True) -> pd.DataFrame:
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Cleavage accessibility structure features")
    input_type = cfg.get("input_type")
    src = _feature_sequences_for_extraction(cfg, outdir, input_type)
    recs = _read_fasta(src)
    structure_enabled = cfg.get("structure_enabled", True)
    struct_df = compute_structure_feature_table(
        recs,
        structure_enabled=structure_enabled,
        alphafold_enabled=cfg.get("alphafold_enabled", False),
        cache_dir=outdir / "cache",
        threads=cfg.get("threads"),
    )
    struct_df.to_csv(path_features_structure(outdir), index=False)
    if structure_enabled:
        log(outdir, f"Structure features saved: {path_features_structure(outdir)}")
    else:
        log(outdir, f"Structure features disabled; placeholders saved: {path_features_structure(outdir)}")
    if assemble and _update_combined_features_if_ready(outdir):
        log(outdir, f"Combined features saved: {path_features(outdir)}")
    return struct_df


def step_features_function(cfg: dict, assemble: bool = True) -> pd.DataFrame:
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Protein function features")
    input_type = cfg.get("input_type")
    src = _feature_sequences_for_extraction(cfg, outdir, input_type)
    recs = _read_fasta(src)
    pipeline_logger = lambda message: log(outdir, message)
    func_df = compute_function_features(
        recs,
        go_map_path=cfg.get("go_map"),
        diamond_db=cfg.get("diamond_db"),
        blast_db=cfg.get("blast_db"),
        pfam_hmms=cfg.get("pfam_hmms"),
        run_interpro=cfg.get("interpro", False),
        logger=pipeline_logger,
    )
    func_df.to_csv(path_features_function(outdir), index=False)
    log(outdir, f"Functional features saved: {path_features_function(outdir)}")
    if assemble and _update_combined_features_if_ready(outdir):
        log(outdir, f"Combined features saved: {path_features(outdir)}")
    return func_df


def step_features(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    if cfg.get("cluster_cdhit"):
        step_features_cluster(cfg)
    if cfg.get("low_complexity"):
        step_features_mask(cfg)

    feature_funcs = {
        "aa": step_features_aa,
        "regsite": step_features_regsite,
        "structure": step_features_structure,
        "function": step_features_function,
    }

    results: Dict[str, pd.DataFrame] = {}
    configured_workers = cfg.get("feature_workers") or len(feature_funcs)
    try:
        max_workers = int(configured_workers)
    except (TypeError, ValueError):
        max_workers = len(feature_funcs)
    max_workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(func, cfg, False): name
            for name, func in feature_funcs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    ordered = [results[key] for key in ("aa", "regsite", "structure", "function") if key in results]
    combined = merge_feature_frames(ordered)
    combined.to_csv(path_features(outdir), index=False)
    log(outdir, f"Combined features saved: {path_features(outdir)}")

def step_rank(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    try:
        rank_df = assemble_ranking_features(outdir, cfg)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing required feature table(s) for ranking: "
            f"{str(exc)}. Run the relevant feature commands or provide the paths via CLI."
        ) from exc

    if rank_df.empty:
        raise ValueError("No features available for ranking. Run feature extraction first.")

    feats = path_features(outdir)
    if not feats.exists() and _update_combined_features_if_ready(outdir):
        log(outdir, f"Combined features saved: {feats}")

    df = rank_df
    ranked_df, model_path = rank_sequences(
        df,
        model_path=cfg.get("model_path"),
        train_demo=cfg.get("train_demo", False),
        outdir=outdir
    )
    ranked_df.to_csv(path_ranked(outdir), index=False)
    log(outdir, f"Ranking saved: {path_ranked(outdir)}")
    if model_path:
        log(outdir, f"Model saved: {model_path}")

def run_pipeline(cfg: dict):
    outdir = Path(cfg["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    log(outdir, "Starting bitescore pipeline")
    step_load(cfg)
    if cfg["input_type"] in GENOME_INPUT_TYPES:
        step_call_genes(cfg)
    step_features(cfg)
    step_rank(cfg)
    log(outdir, "Pipeline completed")
