"""Functional annotation hooks.

This module provides a pluggable hook system for connecting external
bioinformatics tools (DIAMOND/BLAST, HMMER/Pfam, InterProScan) to the
functional annotation pipeline.  Each hook receives protein sequences and
optional database paths, runs an external tool (or skips gracefully when
the tool is unavailable), and returns structured ``HookResult`` objects
that feed into the evidence-combination logic in ``function.py``.

Hooks are registered in the ``HOOK_REGISTRY`` and executed by
``run_annotation_hooks`` which is called from ``extract.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from Bio.SeqRecord import SeqRecord


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class HitDetail:
    """A single similarity/domain hit returned by an external tool."""

    query_id: str
    subject_id: str
    identity_percent: float = 0.0
    query_coverage_percent: float = 0.0
    subject_coverage_percent: float = 0.0
    evalue: float = 1.0
    bitscore: float = 0.0
    description: str = ""
    go_terms: List[str] = field(default_factory=list)
    ipr_accession: str = ""
    domain_name: str = ""
    database: str = ""


@dataclass
class HookResult:
    """Aggregated result from a single annotation hook."""

    source_method: str
    hits_by_query: Dict[str, List[HitDetail]] = field(default_factory=dict)
    go_mapping: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.skipped


# ---------------------------------------------------------------------------
# Hook type alias
# ---------------------------------------------------------------------------

HookFn = Callable[
    [List[SeqRecord], dict, Optional[Callable[[str], None]]],
    HookResult,
]


# ---------------------------------------------------------------------------
# Hook implementations
# ---------------------------------------------------------------------------

def diamond_hook(
    records: List[SeqRecord],
    cfg: dict,
    logger: Optional[Callable[[str], None]] = None,
) -> HookResult:
    """Run DIAMOND blastp and return per-query top hits with alignment stats."""
    from ..tools.blast import diamond_blastp_detailed

    db = cfg.get("diamond_db")
    if not db:
        return HookResult(source_method="DIAMOND", skipped=True, skip_reason="no diamond_db configured")

    evalue_cutoff = cfg.get("diamond_evalue", 1e-5)
    max_targets = cfg.get("diamond_max_targets", 5)
    threads = cfg.get("threads", 1)

    hits_by_query = diamond_blastp_detailed(
        records, db,
        max_targets=max_targets,
        evalue=evalue_cutoff,
        threads=threads,
        logger=logger,
    )

    if hits_by_query is None:
        return HookResult(source_method="DIAMOND", skipped=True, skip_reason="diamond not installed or search failed")

    result = HookResult(source_method="DIAMOND")
    for qid, hits in hits_by_query.items():
        result.hits_by_query[qid] = [
            HitDetail(
                query_id=qid,
                subject_id=h["subject_id"],
                identity_percent=h["identity_percent"],
                query_coverage_percent=h["query_coverage_percent"],
                evalue=h["evalue"],
                bitscore=h["bitscore"],
                database="UniProtKB",
            )
            for h in hits
        ]
    return result


def blast_hook(
    records: List[SeqRecord],
    cfg: dict,
    logger: Optional[Callable[[str], None]] = None,
) -> HookResult:
    """Run NCBI blastp and return per-query top hits with alignment stats."""
    from ..tools.blast import blastp_detailed

    db = cfg.get("blast_db")
    if not db:
        return HookResult(source_method="BLAST", skipped=True, skip_reason="no blast_db configured")

    evalue_cutoff = cfg.get("blast_evalue", 1e-5)
    max_targets = cfg.get("blast_max_targets", 5)
    threads = cfg.get("threads", 1)

    hits_by_query = blastp_detailed(
        records, db,
        max_targets=max_targets,
        evalue=evalue_cutoff,
        threads=threads,
        logger=logger,
    )

    if hits_by_query is None:
        return HookResult(source_method="BLAST", skipped=True, skip_reason="blastp not installed or search failed")

    result = HookResult(source_method="BLAST")
    for qid, hits in hits_by_query.items():
        result.hits_by_query[qid] = [
            HitDetail(
                query_id=qid,
                subject_id=h["subject_id"],
                identity_percent=h["identity_percent"],
                query_coverage_percent=h["query_coverage_percent"],
                evalue=h["evalue"],
                bitscore=h["bitscore"],
                database="UniProtKB",
            )
            for h in hits
        ]
    return result


def pfam_hook(
    records: List[SeqRecord],
    cfg: dict,
    logger: Optional[Callable[[str], None]] = None,
) -> HookResult:
    """Run hmmscan against Pfam-A HMMs and return per-query domain hits."""
    from ..tools.hmmer import hmmscan_detailed

    pfam_hmms = cfg.get("pfam_hmms")
    if not pfam_hmms:
        return HookResult(source_method="Pfam", skipped=True, skip_reason="no pfam_hmms configured")

    evalue_cutoff = cfg.get("pfam_evalue", 1e-5)

    hits_by_query = hmmscan_detailed(
        records, pfam_hmms,
        evalue=evalue_cutoff,
        logger=logger,
    )

    if hits_by_query is None:
        return HookResult(source_method="Pfam", skipped=True, skip_reason="hmmscan not installed or search failed")

    # Load Pfam2GO mapping if available
    pfam2go = _load_pfam2go(cfg.get("pfam2go"))

    result = HookResult(source_method="Pfam")
    for qid, domains in hits_by_query.items():
        hit_details = []
        for dom in domains:
            pfam_acc = dom["target_accession"]
            go_terms = pfam2go.get(pfam_acc, [])
            # Also try without version suffix (PF00014.22 -> PF00014)
            if not go_terms and "." in pfam_acc:
                go_terms = pfam2go.get(pfam_acc.split(".")[0], [])
            hit_details.append(
                HitDetail(
                    query_id=qid,
                    subject_id=pfam_acc,
                    evalue=dom["evalue"],
                    bitscore=dom["score"],
                    domain_name=dom["target_name"],
                    description=dom.get("description", ""),
                    go_terms=go_terms,
                    database="Pfam",
                )
            )
        result.hits_by_query[qid] = hit_details

    return result


def interpro_hook(
    records: List[SeqRecord],
    cfg: dict,
    logger: Optional[Callable[[str], None]] = None,
) -> HookResult:
    """Run InterProScan and return per-query annotation hits with GO terms."""
    from ..tools.interpro import interproscan_detailed

    if not cfg.get("interpro", False):
        return HookResult(source_method="InterProScan", skipped=True, skip_reason="interpro not enabled")

    hits_by_query = interproscan_detailed(records, logger=logger)

    if hits_by_query is None:
        return HookResult(
            source_method="InterProScan", skipped=True,
            skip_reason="interproscan.sh not installed or search failed",
        )

    result = HookResult(source_method="InterProScan")
    for qid, annotations in hits_by_query.items():
        hit_details = []
        for ann in annotations:
            hit_details.append(
                HitDetail(
                    query_id=qid,
                    subject_id=ann.get("ipr_accession", ann.get("signature_accession", "")),
                    evalue=ann.get("evalue", 1.0),
                    domain_name=ann.get("signature_description", ""),
                    description=ann.get("ipr_description", ""),
                    go_terms=ann.get("go_terms", []),
                    ipr_accession=ann.get("ipr_accession", ""),
                    database=ann.get("analysis_db", "InterPro"),
                )
            )
        result.hits_by_query[qid] = hit_details

    return result


# ---------------------------------------------------------------------------
# Pfam2GO mapping loader
# ---------------------------------------------------------------------------

def _load_pfam2go(path: Optional[str]) -> Dict[str, List[str]]:
    """Load a Pfam2GO mapping file.

    Supports two formats:
    - Standard Gene Ontology pfam2go format:
        Pfam:PF00014 Kunitz_BPTI > GO:serine-type endopeptidase inhibitor activity ; GO:0004866
    - Simple TSV format:
        PF00014\tGO:0004866;GO:0005576
    """
    if not path:
        return {}

    from pathlib import Path as P
    p = P(path)
    if not p.exists():
        return {}

    mapping: Dict[str, List[str]] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        # Standard pfam2go format from Gene Ontology
        if line.startswith("Pfam:"):
            parts = line.split()
            if len(parts) < 2:
                continue
            pfam_id = parts[0].replace("Pfam:", "")
            # GO term is the last token after "; "
            if "; " in line:
                go_id = line.rsplit("; ", 1)[-1].strip()
                if go_id.startswith("GO:"):
                    mapping.setdefault(pfam_id, []).append(go_id)
            continue

        # Simple TSV format
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                pfam_id = parts[0].strip()
                go_ids = [g.strip() for g in parts[1].split(";") if g.strip().startswith("GO:")]
                if go_ids:
                    mapping.setdefault(pfam_id, []).extend(go_ids)

    return mapping


# ---------------------------------------------------------------------------
# Hook registry and runner
# ---------------------------------------------------------------------------

HOOK_REGISTRY: Dict[str, HookFn] = {
    "diamond": diamond_hook,
    "blast": blast_hook,
    "pfam": pfam_hook,
    "interpro": interpro_hook,
}


def run_annotation_hooks(
    records: List[SeqRecord],
    cfg: dict,
    logger: Optional[Callable[[str], None]] = None,
    hooks: Optional[Sequence[str]] = None,
) -> Dict[str, HookResult]:
    """Execute annotation hooks and return results keyed by hook name.

    Parameters
    ----------
    records : list of SeqRecord
        Protein sequences to annotate.
    cfg : dict
        Pipeline configuration (database paths, thresholds, flags).
    logger : callable, optional
        Logging callback.
    hooks : sequence of str, optional
        Subset of hook names to run.  If ``None``, all registered hooks
        whose configuration prerequisites are met will be attempted.

    Returns
    -------
    dict mapping hook name to ``HookResult``
    """
    selected = hooks if hooks is not None else list(HOOK_REGISTRY.keys())
    results: Dict[str, HookResult] = {}

    for name in selected:
        hook_fn = HOOK_REGISTRY.get(name)
        if hook_fn is None:
            continue

        if logger:
            logger(f"[hooks] Running annotation hook: {name}")

        result = hook_fn(records, cfg, logger)
        results[name] = result

        if logger:
            if result.skipped:
                logger(f"[hooks] Hook '{name}' skipped: {result.skip_reason}")
            else:
                total_hits = sum(len(h) for h in result.hits_by_query.values())
                logger(f"[hooks] Hook '{name}' returned {total_hits} hits for {len(result.hits_by_query)} queries")

    return results


def hooks_to_evidence(
    query_id: str,
    hook_results: Dict[str, HookResult],
    go_records: Optional[dict] = None,
) -> List[Dict[str, object]]:
    """Convert hook results for a single query into evidence dicts.

    This bridges the hook system with the existing evidence-combination
    logic in ``function.py``.  Each hit that carries GO terms produces
    one evidence entry per GO term.

    Parameters
    ----------
    query_id : str
        The protein record ID to look up in each hook's hits.
    hook_results : dict
        Output of ``run_annotation_hooks``.
    go_records : dict, optional
        UniProt GO record map for resolving accessions from DIAMOND/BLAST
        hits to GO terms.

    Returns
    -------
    list of evidence dicts compatible with ``_combine_evidence``.
    """
    from .function import _go_metadata, _make_evidence

    evidence: List[Dict[str, object]] = []

    for hook_name, result in hook_results.items():
        if result.skipped:
            continue

        hits = result.hits_by_query.get(query_id, [])

        for hit in hits:
            # Determine GO terms: from hit itself or from GO records lookup
            go_terms_for_hit = list(hit.go_terms)

            # For DIAMOND/BLAST hits, resolve accession -> GO terms via go_records
            if not go_terms_for_hit and go_records and hit.database == "UniProtKB":
                accession = _extract_accession(hit.subject_id)
                record = go_records.get(accession)
                if record:
                    go_terms_for_hit = [t.go_id for t in record.go_terms]

            if not go_terms_for_hit:
                continue

            # Determine evidence code and confidence based on source and stats
            source_method, evidence_code, confidence = _assess_evidence(
                hook_name, hit, result.source_method,
            )

            provenance = {
                "source_method": source_method,
                "database_ids": [f"{hit.database}:{hit.subject_id}"] if hit.subject_id else [],
                "hit_accessions": [hit.subject_id] if hit.subject_id else [],
                "evalue": hit.evalue,
                "identity_percent": hit.identity_percent,
                "query_coverage_percent": hit.query_coverage_percent,
                "subject_coverage_percent": hit.subject_coverage_percent,
                "bitscore": hit.bitscore,
                "domain_name": hit.domain_name or None,
                "ipr_accession": hit.ipr_accession or None,
            }

            for go_id in go_terms_for_hit:
                meta = _go_metadata(go_id)
                evidence.append(
                    _make_evidence(
                        go_id,
                        meta["name"],
                        meta["aspect"],
                        evidence_code,
                        source_method,
                        confidence,
                        provenance,
                    )
                )

    return evidence


def _extract_accession(subject_id: str) -> str:
    """Extract UniProt accession from a DIAMOND/BLAST subject ID.

    Handles formats like:
    - sp|P00001|NAME_HUMAN -> P00001
    - tr|A0A000|NAME_HUMAN -> A0A000
    - P00001 -> P00001
    """
    parts = subject_id.split("|")
    if len(parts) >= 2:
        return parts[1]
    return subject_id.split()[0]


def _assess_evidence(
    hook_name: str,
    hit: HitDetail,
    source_method: str,
) -> tuple[str, str, float]:
    """Determine source_method label, evidence code, and confidence score.

    Returns (source_method, evidence_code, confidence_score).
    """
    if hook_name in ("diamond", "blast"):
        if hit.identity_percent >= 90 and hit.query_coverage_percent >= 80:
            return source_method, "ISS", 0.90
        if hit.identity_percent >= 60 and hit.query_coverage_percent >= 60:
            return source_method, "ISS", 0.70
        if hit.identity_percent >= 40 and hit.query_coverage_percent >= 50:
            return source_method, "IEA", 0.55
        return source_method, "IEA", 0.40

    if hook_name == "pfam":
        if hit.evalue <= 1e-20:
            return "Pfam2GO", "IEA", 0.80
        if hit.evalue <= 1e-10:
            return "Pfam2GO", "IEA", 0.70
        return "Pfam2GO", "IEA", 0.60

    if hook_name == "interpro":
        if hit.ipr_accession:
            return "InterPro2GO", "IEA", 0.75
        return "InterPro2GO", "IEA", 0.60

    return source_method, "IEA", 0.50
