from __future__ import annotations
from pathlib import Path
import yaml
from copy import deepcopy

DEFAULTS = {
    "input_type": None,
    "organism": None,
    "structure_enabled": True,
    "alphafold_enabled": False,
    "outdir": "results",
    "train_demo": False,
    "go_map": None,
    "diamond_db": None,
    "blast_db": None,
    "pfam_hmms": None,
    "interpro": False,
    "pfam2go": None,
    "interpro2go": None,
    "diamond_evalue": 1e-5,
    "diamond_max_targets": 5,
    "blast_evalue": 1e-5,
    "blast_max_targets": 5,
    "pfam_evalue": 1e-5,
    "annotation_hooks": None,
    "cluster_cdhit": False,
    "cdhit_threshold": 0.95,
    "low_complexity": False,
    "tmhmm": False,
    "signalp": False,
    "threads": 1,
    "model_path": None,
    "loaded_path": None,
    "feature_sequences": None,
    "features_aa_path": None,
    "features_regsite_path": None,
    "features_structure_path": None,
    "features_function_path": None,
    "features_esm_path": None,
    # --- Deep learning & calibration ---
    "esm_enabled": False,
    "esm_model": "esm2_t6_8M_UR50D",
    "esm_batch_size": 8,
    "calibrate": True,
    "calibration_method": "isotonic",
    "mil_model_path": None,
    "mil_train": False,
    "mil_hidden_dim": 256,
    "mil_attention_dim": 128,
    "mil_epochs": 200,
    "mil_lr": 1e-3,
    "digestibility_ref": None,
    "food_composition": None,
}

def load_config(yaml_path: str | None, cli_overrides: dict) -> dict:
    cfg = deepcopy(DEFAULTS)
    if yaml_path:
        p = Path(yaml_path)
        if p.exists():
            with p.open() as fh:
                y = yaml.safe_load(fh) or {}
            cfg.update({k:v for k,v in y.items() if v is not None})
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["outdir"] = str(Path(cfg["outdir"]).resolve())
    return cfg
