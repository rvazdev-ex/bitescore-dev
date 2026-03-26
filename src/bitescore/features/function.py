from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


# Minimal term metadata used for pretty-printing and aspect lookup in tests.
GO_TERM_METADATA: Dict[str, Dict[str, str]] = {
    "GO:0004866": {"name": "serine-type endopeptidase inhibitor activity", "aspect": "MF"},
    "GO:0030414": {"name": "peptidase inhibitor activity", "aspect": "MF"},
    "GO:0030246": {"name": "carbohydrate binding", "aspect": "MF"},
    "GO:0090729": {"name": "toxin activity", "aspect": "MF"},
    "GO:0005576": {"name": "extracellular region", "aspect": "CC"},
    "GO:0005618": {"name": "cell wall", "aspect": "CC"},
    "GO:0005509": {"name": "calcium ion binding", "aspect": "MF"},
    "GO:0005524": {"name": "ATP binding", "aspect": "MF"},
    "GO:0004674": {"name": "protein serine/threonine kinase activity", "aspect": "MF"},
    "GO:0006468": {"name": "protein phosphorylation", "aspect": "BP"},
    "GO:0003677": {"name": "DNA binding", "aspect": "MF"},
    "GO:0006355": {"name": "regulation of transcription, DNA-templated", "aspect": "BP"},
    "GO:0005634": {"name": "nucleus", "aspect": "CC"},
}


CURATED_EVIDENCE_CODES = {"EXP", "TAS", "IDA", "IMP", "IGI", "IC"}

RED_FLAG_GOS = {
    "GO:0030414": "protease inhibitor",
    "GO:0004866": "protease inhibitor",
    "GO:0030246": "lectin",
    "GO:0090729": "toxin",
}

GREEN_FLAG_GOS = {
    "GO:0005576": "extracellular",
    "GO:0005618": "extracellular",
}


@dataclass
class UniProtGO:
    go_id: str
    name: str
    aspect: str
    evidence_code: str
    original_source: Optional[str] = None


@dataclass
class UniProtRecord:
    accession: str
    sequence: Optional[str]
    entry_type: str
    go_terms: List[UniProtGO]


INTERPRO_RULES: List[Dict[str, object]] = [
    {
        "name": "Kunitz-type protease inhibitor",
        "motifs": ["KUNITZ"],
        "database_ids": ["InterPro:IPR002223", "Pfam:PF00014"],
        "go_terms": ["GO:0004866", "GO:0005576"],
    },
    {
        "name": "Secreted lectin repeat",
        "motifs": ["LECTIN", "QXW"],
        "database_ids": ["InterPro:IPR001304", "Pfam:PF00139"],
        "go_terms": ["GO:0030246", "GO:0005576"],
    },
]


BLAST_REFERENCE: List[Dict[str, object]] = [
    {
        "accession": "REF_DNA_BINDING_A",
        "sequence": "MSTRTKQLTAALREKLEELAAALKKA",
        "go_terms": ["GO:0003677", "GO:0006355", "GO:0005634"],
    },
    {
        "accession": "REF_DNA_BINDING_B",
        "sequence": "MSTRTRQLTAALREKLEELAASLKKQ",
        "go_terms": ["GO:0003677", "GO:0006355", "GO:0005634"],
    },
    {
        "accession": "REF_KINASE",
        "sequence": "MGCGTGGGGIGTVYRDLKPENILLDVK",
        "go_terms": ["GO:0005524", "GO:0004674", "GO:0006468"],
    },
]


def _go_metadata(go_id: str) -> Dict[str, str]:
    meta = GO_TERM_METADATA.get(go_id, {})
    return {
        "go_id": go_id,
        "name": meta.get("name", go_id),
        "aspect": meta.get("aspect", "NA"),
    }


def load_uniprot_records(go_map_path: Optional[str]) -> Dict[str, UniProtRecord]:
    if not go_map_path:
        return {}

    path = Path(go_map_path)
    if not path.exists():
        return {}

    records: Dict[str, UniProtRecord] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        accession = parts[0]
        sequence = parts[1] or None
        entry_type = parts[2] if len(parts) >= 3 and parts[2] else "Swiss-Prot"
        raw_terms = parts[3] if len(parts) >= 4 else ""

        go_terms: List[UniProtGO] = []
        payload: List[dict]
        try:
            payload = json.loads(raw_terms) if raw_terms else []
            if not isinstance(payload, list):
                payload = []
        except json.JSONDecodeError:
            payload = []

        if payload:
            for item in payload:
                go_id = item.get("go_id")
                if not go_id:
                    continue
                meta = _go_metadata(go_id)
                go_terms.append(
                    UniProtGO(
                        go_id=go_id,
                        name=item.get("name") or meta["name"],
                        aspect=item.get("aspect") or meta["aspect"],
                        evidence_code=item.get("evidence_code", "IEA"),
                        original_source=item.get("source"),
                    )
                )
        else:
            for go_id in [token for token in raw_terms.split(";") if token]:
                meta = _go_metadata(go_id)
                go_terms.append(
                    UniProtGO(
                        go_id=go_id,
                        name=meta["name"],
                        aspect=meta["aspect"],
                        evidence_code="IEA",
                    )
                )

        records[accession] = UniProtRecord(
            accession=accession,
            sequence=sequence,
            entry_type=entry_type,
            go_terms=go_terms,
        )

    return records


def _alignment_stats(seq: str, ref: str) -> Dict[str, float]:
    if not seq or not ref:
        return {"identity": 0.0, "query_coverage": 0.0, "subject_coverage": 0.0}
    length = min(len(seq), len(ref))
    matches = sum(1 for a, b in zip(seq[:length], ref[:length]) if a == b)
    identity = matches / length if length else 0.0
    query_cov = length / len(seq) if seq else 0.0
    subject_cov = length / len(ref) if ref else 0.0
    return {"identity": identity, "query_coverage": query_cov, "subject_coverage": subject_cov}


def _identity_to_percent(value: float) -> float:
    return round(value * 100.0, 2)


def _best_uniprot_match(seq: str, records: Iterable[UniProtRecord]) -> Optional[Dict[str, object]]:
    best: Optional[Dict[str, object]] = None
    for record in records:
        if not record.sequence:
            continue
        stats = _alignment_stats(seq, record.sequence)
        if best is None or stats["identity"] > best["identity"]:
            best = {
                "record": record,
                "identity": stats["identity"],
                "query_coverage": stats["query_coverage"],
                "subject_coverage": stats["subject_coverage"],
            }
    return best


def _make_evidence(
    go_id: str,
    name: str,
    aspect: str,
    evidence_code: str,
    source_method: str,
    confidence: float,
    provenance: Dict[str, object],
) -> Dict[str, object]:
    return {
        "go_id": go_id,
        "name": name,
        "aspect": aspect,
        "evidence_code": evidence_code,
        "source_method": source_method,
        "confidence_score": confidence,
        "provenance": provenance,
    }


def _uniprot_confidence(entry_type: str, identity: float, coverage: float, original: str) -> tuple[str, float]:
    entry_type = entry_type.lower()
    if (
        entry_type == "swiss-prot"
        and identity >= 0.98
        and coverage >= 0.95
        and original in CURATED_EVIDENCE_CODES
    ):
        return "ISS", 0.95
    if entry_type == "swiss-prot" and (identity >= 0.90 or coverage >= 0.80):
        return "ISS", 0.80
    return "IEA", 0.55


def _uniprot_evidence(seq: str, match: Dict[str, object]) -> List[Dict[str, object]]:
    record: UniProtRecord = match["record"]
    identity = match["identity"]
    coverage = match["query_coverage"]
    evidence: List[Dict[str, object]] = []

    for term in record.go_terms:
        meta = _go_metadata(term.go_id)
        evidence_code, confidence = _uniprot_confidence(
            record.entry_type, identity, coverage, term.evidence_code
        )
        provenance = {
            "source_method": "UniProt",
            "database_ids": [f"UniProtKB:{record.accession}"],
            "hit_accessions": [record.accession],
            "evalue": 0.0,
            "identity_percent": _identity_to_percent(identity),
            "query_coverage_percent": _identity_to_percent(coverage),
            "subject_coverage_percent": _identity_to_percent(match["subject_coverage"]),
            "original_evidence_code": term.evidence_code,
            "entry_type": record.entry_type,
        }
        if term.original_source:
            provenance["source_reference"] = term.original_source

        evidence.append(
            _make_evidence(
                term.go_id,
                term.name or meta["name"],
                term.aspect or meta["aspect"],
                evidence_code,
                "UniProt",
                confidence,
                provenance,
            )
        )

    return evidence


def _interpro_evidence(seq: str) -> List[Dict[str, object]]:
    annotations: List[Dict[str, object]] = []
    upper_seq = seq.upper()
    for rule in INTERPRO_RULES:
        if any(motif in upper_seq for motif in rule["motifs"]):
            for go_id in rule["go_terms"]:
                meta = _go_metadata(go_id)
                provenance = {
                    "source_method": "InterPro2GO",
                    "database_ids": rule["database_ids"],
                    "hit_accessions": [],
                    "evalue": None,
                    "identity_percent": None,
                    "query_coverage_percent": None,
                    "subject_coverage_percent": None,
                    "rule": rule["name"],
                }
                annotations.append(
                    _make_evidence(
                        go_id,
                        meta["name"],
                        meta["aspect"],
                        "IEA",
                        "InterPro2GO",
                        0.65,
                        provenance,
                    )
                )
    return annotations


def _blast_evidence(seq: str) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    for ref in BLAST_REFERENCE:
        stats = _alignment_stats(seq, ref["sequence"])
        identity = stats["identity"]
        coverage = stats["query_coverage"]
        if identity >= 0.40 and coverage >= 0.70:
            score = identity * coverage
            evalue = max(1e-40, 1e-5 * (1 - score))
            hits.append(
                {
                    "accession": ref["accession"],
                    "identity": identity,
                    "coverage": coverage,
                    "subject_coverage": stats["subject_coverage"],
                    "evalue": evalue,
                    "go_terms": ref["go_terms"],
                }
            )

    if not hits:
        return []

    hits.sort(key=lambda item: item["identity"], reverse=True)
    support: Dict[str, Dict[str, object]] = {}
    for hit in hits:
        for go_id in hit["go_terms"]:
            entry = support.setdefault(
                go_id,
                {
                    "hits": [],
                    "name": _go_metadata(go_id)["name"],
                    "aspect": _go_metadata(go_id)["aspect"],
                },
            )
            entry["hits"].append(hit)

    annotations: List[Dict[str, object]] = []
    for go_id, entry in support.items():
        hits_for_term = entry["hits"]
        confidence = 0.55 if len(hits_for_term) >= 2 else 0.45
        best_hit = max(hits_for_term, key=lambda item: item["identity"])
        provenance = {
            "source_method": "BLAST2GO",
            "database_ids": [],
            "hit_accessions": [hit["accession"] for hit in hits_for_term],
            "evalue": min(hit["evalue"] for hit in hits_for_term),
            "identity_percent": _identity_to_percent(best_hit["identity"]),
            "query_coverage_percent": _identity_to_percent(best_hit["coverage"]),
            "subject_coverage_percent": _identity_to_percent(best_hit["subject_coverage"]),
        }
        annotations.append(
            _make_evidence(
                go_id,
                entry["name"],
                entry["aspect"],
                "ISS" if confidence >= 0.55 else "IEA",
                "BLAST2GO",
                confidence,
                provenance,
            )
        )

    return annotations


def _combine_evidence(evidence: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    combined: Dict[str, Dict[str, object]] = {}
    for ev in evidence:
        entry = combined.setdefault(
            ev["go_id"],
            {
                "go_id": ev["go_id"],
                "name": ev["name"],
                "aspect": ev["aspect"],
                "confidence_score": ev["confidence_score"],
                "evidence_code": ev["evidence_code"],
                "source_method": ev["source_method"],
                "sources": [],
            },
        )
        entry["sources"].append(
            {
                **ev["provenance"],
                "confidence_score": ev["confidence_score"],
                "evidence_code": ev["evidence_code"],
            }
        )
        if ev["confidence_score"] > entry["confidence_score"]:
            entry["confidence_score"] = ev["confidence_score"]
            entry["evidence_code"] = ev["evidence_code"]
            entry["source_method"] = ev["source_method"]
            entry["name"] = ev["name"]
            entry["aspect"] = ev["aspect"]
    return combined


def _select_best(terms: Dict[str, Dict[str, object]], aspect: str) -> Optional[Dict[str, object]]:
    candidates = [term for term in terms.values() if term["aspect"] == aspect]
    if not candidates:
        return None
    candidates.sort(key=lambda term: (term["confidence_score"], term["name"]), reverse=True)
    return candidates[0]


def _flag_terms(terms: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    red_terms = [term for term in terms.values() if term["go_id"] in RED_FLAG_GOS]
    green_terms = [term for term in terms.values() if term["go_id"] in GREEN_FLAG_GOS]
    return {
        "red_terms": red_terms,
        "green_terms": green_terms,
        "red_ids": [term["go_id"] for term in red_terms],
        "green_ids": [term["go_id"] for term in green_terms],
    }


def _summarise_label(terms: Dict[str, Dict[str, object]], flags: Dict[str, object]) -> str:
    top_mf = _select_best(terms, "MF")
    top_bp = _select_best(terms, "BP")
    top_cc = _select_best(terms, "CC")

    def _short(text: str) -> str:
        return text if len(text) <= 120 else text[:117] + "..."

    red_terms = flags["red_terms"]
    green_terms = flags["green_terms"]

    if red_terms:
        kind = RED_FLAG_GOS.get(red_terms[0]["go_id"], "risk")
        names = ", ".join(term["name"] for term in red_terms[:2])
        return _short(f"⚠️ Potential {kind}: {names}")

    if green_terms:
        mf = top_mf["name"] if top_mf else "unknown activity"
        bp = top_bp["name"] if top_bp else "unknown process"
        return _short(f"Secreted/extracellular {mf}: {bp}")

    mf = top_mf["name"] if top_mf else "Uncharacterized activity"
    bp = top_bp["name"] if top_bp else "no biological process"
    cc = top_cc["name"] if top_cc else "cellular component unknown"
    return _short(f"{mf}; likely {bp}; {cc}")


def annotate_sequence(
    seq: str,
    go_records: Dict[str, UniProtRecord],
    accession_hint: Optional[str] = None,
) -> Dict[str, object]:
    evidence: List[Dict[str, object]] = []

    best_match: Optional[Dict[str, object]] = None
    if accession_hint and accession_hint in go_records:
        record = go_records[accession_hint]
        if record.sequence:
            stats = _alignment_stats(seq, record.sequence)
        else:
            stats = {"identity": 1.0, "query_coverage": 1.0, "subject_coverage": 1.0}
        best_match = {"record": record, **stats}
    else:
        best_match = _best_uniprot_match(seq, go_records.values())

    if best_match:
        evidence.extend(_uniprot_evidence(seq, best_match))

    reliable_uniprot = bool(
        best_match
        and best_match["identity"] >= 0.98
        and best_match["query_coverage"] >= 0.95
        and best_match["record"].entry_type.lower() == "swiss-prot"
    )

    if not evidence or not reliable_uniprot:
        evidence.extend(_interpro_evidence(seq))
        evidence.extend(_blast_evidence(seq))

    combined = _combine_evidence(evidence)
    flags = _flag_terms(combined)
    label = _summarise_label(combined, flags)

    return {
        "combined_terms": combined,
        "flags": flags,
        "functional_label": label,
        "top_mf": _select_best(combined, "MF"),
        "top_bp": _select_best(combined, "BP"),
        "top_cc": _select_best(combined, "CC"),
        "best_uniprot_match": best_match,
    }


def annotation_row(
    record_id: str,
    seq: str,
    go_map: Dict[str, UniProtRecord],
    accession_hint: Optional[str] = None,
) -> Dict[str, object]:
    details = annotate_sequence(seq, go_map, accession_hint=accession_hint)
    combined = details["combined_terms"]
    top_mf = details["top_mf"] or {}
    top_bp = details["top_bp"] or {}
    top_cc = details["top_cc"] or {}
    best_match = details["best_uniprot_match"]

    if best_match:
        record = best_match["record"]
        best_accession = record.accession
        identity = _identity_to_percent(best_match["identity"])
        coverage = _identity_to_percent(best_match["query_coverage"])
        entry_type = record.entry_type
    else:
        best_accession = ""
        identity = 0.0
        coverage = 0.0
        entry_type = ""

    annotation_methods = sorted({term["source_method"] for term in combined.values()})

    return {
        "id": record_id,
        "functional_label": details["functional_label"],
        "go_term_count": len(combined),
        "go_terms_json": json.dumps(list(combined.values())),
        "annotation_methods": ",".join(annotation_methods),
        "red_flag": bool(details["flags"]["red_terms"]),
        "red_flag_terms": json.dumps(details["flags"]["red_ids"]),
        "green_flag": bool(details["flags"]["green_terms"]),
        "green_flag_terms": json.dumps(details["flags"]["green_ids"]),
        "top_mf_go_id": top_mf.get("go_id", ""),
        "top_mf_term": top_mf.get("name", ""),
        "top_bp_go_id": top_bp.get("go_id", ""),
        "top_bp_term": top_bp.get("name", ""),
        "top_cc_go_id": top_cc.get("go_id", ""),
        "top_cc_term": top_cc.get("name", ""),
        "best_uniprot_accession": best_accession,
        "best_uniprot_identity": identity,
        "best_uniprot_coverage": coverage,
        "best_uniprot_entry_type": entry_type,
    }
