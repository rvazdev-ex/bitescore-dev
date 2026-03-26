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
