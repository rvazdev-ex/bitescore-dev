from pathlib import Path
from typing import Callable, List
import os
import pandas as pd
from Bio.SeqRecord import SeqRecord

from .aa import essential_aa_content, physchem
from .function import annotation_row, load_uniprot_records
from .cleavage import cleavage_accessibility_scores
from .structure import structure_features
from ..tools.blast import diamond_top_hits, blastp_top_hits
from ..tools.hmmer import hmmscan_domains
from ..tools.interpro import interproscan


def compute_aa_features(records: List[SeqRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        seq = str(rec.seq)
        row = {"id": rec.id, "length": len(seq)}
        row.update(essential_aa_content(seq))
        row.update(physchem(seq))
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    template = {"id": [], "length": []}
    template.update({k: [] for k in essential_aa_content("").keys()})
    template.update({k: [] for k in physchem("").keys()})
    return pd.DataFrame(template)


def compute_regsite_features(records: List[SeqRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        seq = str(rec.seq)
        row = {"id": rec.id}
        row.update(cleavage_accessibility_scores(seq))
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    template = {"id": []}
    template.update({k: [] for k in cleavage_accessibility_scores("").keys()})
    return pd.DataFrame(template)


def compute_structure_feature_table(
    records: List[SeqRecord],
    structure_enabled: bool = True,
    alphafold_enabled: bool = False,
    cache_dir: Path | None = None,
    threads: int | None = None,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir or ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not structure_enabled:
        return pd.DataFrame({"id": [rec.id for rec in records]})

    rows = []
    for rec in records:
        seq = str(rec.seq)
        row = {"id": rec.id}
        row.update(
            structure_features(
                seq,
                rec.id,
                alphafold_enabled,
                cache_dir,
                threads=threads,
            )
        )
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["id"])


def compute_function_features(
    records: List[SeqRecord],
    go_map_path: str | None = None,
    diamond_db: str | None = None,
    blast_db: str | None = None,
    pfam_hmms: str | None = None,
    run_interpro: bool = False,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    rows = []
    go_map_path = go_map_path or os.environ.get("BITESCORE_GO_MAP")
    go_records = load_uniprot_records(go_map_path)

    for rec in records:
        seq = str(rec.seq)
        accession_hint = None
        if (diamond_db or blast_db) and go_records:
            top_hits = None
            if diamond_db:
                top_hits = diamond_top_hits([rec], diamond_db, max_targets=1, logger=logger)
            if top_hits is None and blast_db:
                top_hits = blastp_top_hits([rec], blast_db, max_targets=1, logger=logger)
            if top_hits:
                accession_hint = top_hits[0][0].split("|")[-1]

        row = annotation_row(rec.id, seq, go_records, accession_hint=accession_hint)

        if pfam_hmms:
            doms = hmmscan_domains([rec], pfam_hmms, logger=logger)
            if doms:
                row.update({f"pfam_{k}_count": v for k, v in doms.items()})

        if run_interpro:
            ipr = interproscan([rec], logger=logger)
            if ipr:
                row.update(ipr)

        rows.append(row)

    if rows:
        return pd.DataFrame(rows)

    template = {
        "id": [],
        "functional_label": [],
        "go_term_count": [],
        "go_terms_json": [],
        "annotation_methods": [],
        "red_flag": [],
        "red_flag_terms": [],
        "green_flag": [],
        "green_flag_terms": [],
        "top_mf_go_id": [],
        "top_mf_term": [],
        "top_bp_go_id": [],
        "top_bp_term": [],
        "top_cc_go_id": [],
        "top_cc_term": [],
        "best_uniprot_accession": [],
        "best_uniprot_identity": [],
        "best_uniprot_coverage": [],
        "best_uniprot_entry_type": [],
    }
    return pd.DataFrame(template)


def merge_feature_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if df is not None]
    if not frames:
        return pd.DataFrame()

    def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
        if "id" not in df.columns:
            df = df.assign(id=pd.Series(dtype=str))
        return df

    merged = _ensure_id_column(frames[0]).set_index("id")
    for frame in frames[1:]:
        frame = _ensure_id_column(frame)
        merged = merged.join(frame.set_index("id"), how="outer")

    return merged.reset_index().fillna(0)


def compute_features(
    records: List[SeqRecord],
    structure_enabled: bool = True,
    alphafold_enabled: bool = False,
    cache_dir: Path | None = None,
    go_map_path: str | None = None,
    diamond_db: str | None = None,
    blast_db: str | None = None,
    pfam_hmms: str | None = None,
    run_interpro: bool = False,
    threads: int | None = None,
) -> pd.DataFrame:
    aa_df = compute_aa_features(records)
    reg_df = compute_regsite_features(records)
    struct_df = compute_structure_feature_table(
        records,
        structure_enabled=structure_enabled,
        alphafold_enabled=alphafold_enabled,
        cache_dir=cache_dir,
        threads=threads,
    )
    func_df = compute_function_features(
        records,
        go_map_path=go_map_path,
        diamond_db=diamond_db,
        blast_db=blast_db,
        pfam_hmms=pfam_hmms,
        run_interpro=run_interpro,
    )
    return merge_feature_frames([aa_df, reg_df, struct_df, func_df])
