from pathlib import Path
from typing import Callable, List, Optional
import os
import pandas as pd
from Bio.SeqRecord import SeqRecord

from .aa import essential_aa_content, physchem
from .function import annotation_row, load_uniprot_records
from .cleavage import cleavage_accessibility_scores
from .hooks import hooks_to_evidence, run_annotation_hooks, HookResult
from .structure import structure_features
from .esm import compute_esm_feature_table
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
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Compute functional annotation features for protein records.

    When *cfg* is provided, annotation hooks are run in batch mode against
    all records at once (more efficient for external tools).  The hooks
    produce per-query evidence that is merged with the built-in annotation
    logic (UniProt lookup, motif scanning, BLAST reference DB).

    The legacy per-record tool calls (``diamond_top_hits``, ``hmmscan_domains``,
    ``interproscan``) remain as a fallback when hooks are not configured or
    when the external tool is unavailable.
    """
    rows = []
    go_map_path = go_map_path or os.environ.get("BITESCORE_GO_MAP")
    go_records = load_uniprot_records(go_map_path)

    # --- Run annotation hooks in batch (if cfg provided) ---
    hook_results: dict[str, HookResult] = {}
    if cfg is not None:
        hook_cfg = dict(cfg)
        hook_cfg.setdefault("diamond_db", diamond_db)
        hook_cfg.setdefault("blast_db", blast_db)
        hook_cfg.setdefault("pfam_hmms", pfam_hmms)
        hook_cfg.setdefault("interpro", run_interpro)
        hook_results = run_annotation_hooks(records, hook_cfg, logger=logger)

    has_hook_hits = any(r.available for r in hook_results.values())

    for rec in records:
        seq = str(rec.seq)
        accession_hint = None

        # Resolve best accession hint from hooks or legacy tool calls
        if has_hook_hits:
            # Prefer DIAMOND hits, fall back to BLAST
            for hook_name in ("diamond", "blast"):
                hr = hook_results.get(hook_name)
                if hr and hr.available:
                    hits = hr.hits_by_query.get(rec.id, [])
                    if hits:
                        accession_hint = _extract_accession_from_subject(hits[0].subject_id)
                        break
        elif (diamond_db or blast_db) and go_records:
            top_hits = None
            if diamond_db:
                top_hits = diamond_top_hits([rec], diamond_db, max_targets=1, logger=logger)
            if top_hits is None and blast_db:
                top_hits = blastp_top_hits([rec], blast_db, max_targets=1, logger=logger)
            if top_hits:
                accession_hint = top_hits[0][0].split("|")[-1]

        # Convert hook results into evidence dicts for this query
        hook_evidence = None
        if has_hook_hits:
            hook_evidence = hooks_to_evidence(rec.id, hook_results, go_records=go_records)

        row = annotation_row(
            rec.id, seq, go_records,
            accession_hint=accession_hint,
            hook_evidence=hook_evidence,
        )

        # --- Legacy per-record Pfam domain scan (fallback) ---
        pfam_hook = hook_results.get("pfam")
        if pfam_hmms and (pfam_hook is None or pfam_hook.skipped):
            doms = hmmscan_domains([rec], pfam_hmms, logger=logger)
            if doms:
                row.update({f"pfam_{k}_count": v for k, v in doms.items()})
        elif pfam_hook and pfam_hook.available:
            # Aggregate domain counts from hook results
            hits = pfam_hook.hits_by_query.get(rec.id, [])
            dom_counts: dict[str, int] = {}
            for hit in hits:
                dom_counts[hit.domain_name] = dom_counts.get(hit.domain_name, 0) + 1
            if dom_counts:
                row.update({f"pfam_{k}_count": v for k, v in dom_counts.items()})

        # --- Legacy per-record InterProScan (fallback) ---
        ipr_hook = hook_results.get("interpro")
        if run_interpro and (ipr_hook is None or ipr_hook.skipped):
            ipr = interproscan([rec], logger=logger)
            if ipr:
                row.update(ipr)
        elif ipr_hook and ipr_hook.available:
            # Aggregate InterPro counts from hook results
            hits = ipr_hook.hits_by_query.get(rec.id, [])
            ipr_counts: dict[str, int] = {}
            for hit in hits:
                db_key = f"db_{hit.database}_count"
                ipr_counts[db_key] = ipr_counts.get(db_key, 0) + 1
                if hit.ipr_accession:
                    ipr_counts["interpro_count"] = ipr_counts.get("interpro_count", 0) + 1
            if ipr_counts:
                row.update(ipr_counts)

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


def _extract_accession_from_subject(subject_id: str) -> str:
    """Extract UniProt accession from subject ID (e.g. sp|P00001|NAME -> P00001)."""
    parts = subject_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return subject_id.split()[0]


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


def compute_esm_features(
    records: List[SeqRecord],
    esm_model: str = "esm2_t6_8M_UR50D",
    batch_size: int = 8,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute ESM-2 embedding features for a list of SeqRecords.

    Delegates to :func:`esm.compute_esm_feature_table`.  Returns a DataFrame
    with id + embedding dimension columns.  When torch/esm are not installed,
    returns an id-only DataFrame so the rest of the pipeline can proceed.
    """
    return compute_esm_feature_table(
        records,
        model_name=esm_model,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )


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
    cfg: dict | None = None,
    esm_enabled: bool = False,
    esm_model: str = "esm2_t6_8M_UR50D",
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
        cfg=cfg,
    )
    frames = [aa_df, reg_df, struct_df, func_df]
    if esm_enabled:
        esm_df = compute_esm_features(
            records,
            esm_model=esm_model,
            cache_dir=cache_dir,
        )
        frames.append(esm_df)
    return merge_feature_frames(frames)
