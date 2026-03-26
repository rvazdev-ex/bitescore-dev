"""Tests for functional annotation hooks.

Covers:
- Hook data types (HitDetail, HookResult)
- Individual hook implementations with mocked tool outputs
- Pfam2GO / InterPro2GO mapping loaders
- Evidence conversion (hooks_to_evidence)
- Integration with compute_function_features via cfg
- Accession extraction from various subject ID formats
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bitescore.features.hooks import (
    HitDetail,
    HookResult,
    _assess_evidence,
    _extract_accession,
    _load_pfam2go,
    diamond_hook,
    blast_hook,
    pfam_hook,
    interpro_hook,
    hooks_to_evidence,
    run_annotation_hooks,
    HOOK_REGISTRY,
)
from bitescore.features.go import (
    load_pfam2go,
    load_interpro2go,
    resolve_go_terms_for_accession,
)
from bitescore.features.extract import compute_function_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records():
    return [
        SeqRecord(Seq("MSTRTKQLTAALREKLEELAAALKKA"), id="prot_A"),
        SeqRecord(Seq("MKAILVVLLYTFATANADNKD"), id="prot_B"),
    ]


@pytest.fixture
def single_record():
    return [SeqRecord(Seq("MSTRTKQLTAALREKLEELAAALKKA"), id="test_protein")]


# ---------------------------------------------------------------------------
# HitDetail and HookResult data types
# ---------------------------------------------------------------------------

class TestHitDetail:
    def test_defaults(self):
        hit = HitDetail(query_id="q1", subject_id="s1")
        assert hit.identity_percent == 0.0
        assert hit.evalue == 1.0
        assert hit.go_terms == []
        assert hit.database == ""

    def test_with_go_terms(self):
        hit = HitDetail(
            query_id="q1", subject_id="PF00014",
            evalue=1e-30, bitscore=120.5,
            go_terms=["GO:0004866", "GO:0005576"],
            database="Pfam",
            domain_name="Kunitz_BPTI",
        )
        assert len(hit.go_terms) == 2
        assert hit.domain_name == "Kunitz_BPTI"


class TestHookResult:
    def test_available_when_not_skipped(self):
        result = HookResult(source_method="DIAMOND")
        assert result.available is True
        assert result.skipped is False

    def test_unavailable_when_skipped(self):
        result = HookResult(source_method="DIAMOND", skipped=True, skip_reason="no db")
        assert result.available is False

    def test_hits_by_query(self):
        result = HookResult(source_method="Pfam")
        result.hits_by_query["prot_A"] = [
            HitDetail(query_id="prot_A", subject_id="PF00014", evalue=1e-20),
        ]
        assert len(result.hits_by_query["prot_A"]) == 1


# ---------------------------------------------------------------------------
# Accession extraction
# ---------------------------------------------------------------------------

class TestExtractAccession:
    def test_uniprot_swissprot_format(self):
        assert _extract_accession("sp|P00001|NAME_HUMAN") == "P00001"

    def test_uniprot_trembl_format(self):
        assert _extract_accession("tr|A0A000|NAME_MOUSE") == "A0A000"

    def test_plain_accession(self):
        assert _extract_accession("P00001") == "P00001"

    def test_with_whitespace(self):
        assert _extract_accession("P00001 some description") == "P00001"


# ---------------------------------------------------------------------------
# Evidence assessment
# ---------------------------------------------------------------------------

class TestAssessEvidence:
    def test_diamond_high_identity(self):
        hit = HitDetail(query_id="q", subject_id="s", identity_percent=95, query_coverage_percent=90)
        method, code, conf = _assess_evidence("diamond", hit, "DIAMOND")
        assert code == "ISS"
        assert conf == 0.90

    def test_diamond_medium_identity(self):
        hit = HitDetail(query_id="q", subject_id="s", identity_percent=70, query_coverage_percent=65)
        method, code, conf = _assess_evidence("diamond", hit, "DIAMOND")
        assert code == "ISS"
        assert conf == 0.70

    def test_diamond_low_identity(self):
        hit = HitDetail(query_id="q", subject_id="s", identity_percent=45, query_coverage_percent=55)
        method, code, conf = _assess_evidence("diamond", hit, "DIAMOND")
        assert code == "IEA"
        assert conf == 0.55

    def test_pfam_high_confidence(self):
        hit = HitDetail(query_id="q", subject_id="PF00014", evalue=1e-30)
        method, code, conf = _assess_evidence("pfam", hit, "Pfam")
        assert method == "Pfam2GO"
        assert conf == 0.80

    def test_pfam_medium_confidence(self):
        hit = HitDetail(query_id="q", subject_id="PF00014", evalue=1e-15)
        method, code, conf = _assess_evidence("pfam", hit, "Pfam")
        assert conf == 0.70

    def test_interpro_with_ipr(self):
        hit = HitDetail(query_id="q", subject_id="IPR000001", ipr_accession="IPR000001")
        method, code, conf = _assess_evidence("interpro", hit, "InterProScan")
        assert method == "InterPro2GO"
        assert conf == 0.75


# ---------------------------------------------------------------------------
# Pfam2GO / InterPro2GO mapping loaders
# ---------------------------------------------------------------------------

class TestLoadPfam2go:
    def test_standard_format(self, tmp_path):
        mapping_file = tmp_path / "pfam2go"
        mapping_file.write_text(
            "!version: 2024-01-01\n"
            "Pfam:PF00014 Kunitz_BPTI > GO:serine-type endopeptidase inhibitor activity ; GO:0004866\n"
            "Pfam:PF00014 Kunitz_BPTI > GO:extracellular region ; GO:0005576\n"
            "Pfam:PF00139 Lectin_legB > GO:carbohydrate binding ; GO:0030246\n"
        )
        mapping = load_pfam2go(str(mapping_file))
        assert "PF00014" in mapping
        assert "GO:0004866" in mapping["PF00014"]
        assert "GO:0005576" in mapping["PF00014"]
        assert "GO:0030246" in mapping["PF00139"]

    def test_tsv_format(self, tmp_path):
        mapping_file = tmp_path / "pfam2go.tsv"
        mapping_file.write_text("PF00014\tGO:0004866;GO:0005576\n")
        mapping = load_pfam2go(str(mapping_file))
        assert len(mapping["PF00014"]) == 2

    def test_missing_file(self):
        assert load_pfam2go("/nonexistent/file") == {}

    def test_none_path(self):
        assert load_pfam2go(None) == {}


class TestLoadInterpro2go:
    def test_standard_format(self, tmp_path):
        mapping_file = tmp_path / "interpro2go"
        mapping_file.write_text(
            "InterPro:IPR000001 Kringle > GO:calcium ion binding ; GO:0005509\n"
            "InterPro:IPR002223 Kunitz > GO:serine-type endopeptidase inhibitor activity ; GO:0004866\n"
        )
        mapping = load_interpro2go(str(mapping_file))
        assert "GO:0005509" in mapping["IPR000001"]
        assert "GO:0004866" in mapping["IPR002223"]

    def test_tsv_format(self, tmp_path):
        mapping_file = tmp_path / "interpro2go.tsv"
        mapping_file.write_text("IPR000001\tGO:0005509;GO:0005576\n")
        mapping = load_interpro2go(str(mapping_file))
        assert len(mapping["IPR000001"]) == 2


class TestResolveGoTerms:
    def test_direct_mapping(self):
        go_map = {"P00001": ["GO:0005509", "GO:0005576"]}
        terms = resolve_go_terms_for_accession("P00001", go_map)
        assert "GO:0005509" in terms
        assert "GO:0005576" in terms

    def test_pfam2go_fallback(self):
        go_map = {}
        pfam2go = {"PF00014": ["GO:0004866"]}
        terms = resolve_go_terms_for_accession("PF00014", go_map, pfam2go=pfam2go)
        assert "GO:0004866" in terms

    def test_version_suffix_stripping(self):
        go_map = {}
        pfam2go = {"PF00014": ["GO:0004866"]}
        terms = resolve_go_terms_for_accession("PF00014.22", go_map, pfam2go=pfam2go)
        assert "GO:0004866" in terms

    def test_deduplication(self):
        go_map = {"P00001": ["GO:0005509"]}
        pfam2go = {"P00001": ["GO:0005509", "GO:0005576"]}
        terms = resolve_go_terms_for_accession("P00001", go_map, pfam2go=pfam2go)
        assert terms.count("GO:0005509") == 1


# ---------------------------------------------------------------------------
# Hook implementations (mocked tools)
# ---------------------------------------------------------------------------

class TestDiamondHook:
    def test_skips_without_db(self, sample_records):
        result = diamond_hook(sample_records, {})
        assert result.skipped is True
        assert "no diamond_db" in result.skip_reason

    @patch("bitescore.tools.blast.diamond_blastp_detailed")
    def test_returns_hits(self, mock_diamond, sample_records):
        mock_diamond.return_value = {
            "prot_A": [
                {"subject_id": "sp|P00001|NAME", "identity_percent": 95.0,
                 "query_coverage_percent": 88.0, "evalue": 1e-30, "bitscore": 200.0},
            ],
        }
        cfg = {"diamond_db": "/path/to/db.dmnd"}
        result = diamond_hook(sample_records, cfg)
        assert result.available is True
        assert "prot_A" in result.hits_by_query
        hit = result.hits_by_query["prot_A"][0]
        assert hit.identity_percent == 95.0
        assert hit.database == "UniProtKB"

    @patch("bitescore.tools.blast.diamond_blastp_detailed")
    def test_handles_tool_failure(self, mock_diamond, sample_records):
        mock_diamond.return_value = None
        cfg = {"diamond_db": "/path/to/db.dmnd"}
        result = diamond_hook(sample_records, cfg)
        assert result.skipped is True


class TestBlastHook:
    def test_skips_without_db(self, sample_records):
        result = blast_hook(sample_records, {})
        assert result.skipped is True

    @patch("bitescore.tools.blast.blastp_detailed")
    def test_returns_hits(self, mock_blast, sample_records):
        mock_blast.return_value = {
            "prot_A": [
                {"subject_id": "P00001", "identity_percent": 80.0,
                 "query_coverage_percent": 75.0, "evalue": 1e-20, "bitscore": 150.0},
            ],
        }
        cfg = {"blast_db": "/path/to/blastdb"}
        result = blast_hook(sample_records, cfg)
        assert result.available is True
        assert len(result.hits_by_query["prot_A"]) == 1


class TestPfamHook:
    def test_skips_without_hmms(self, sample_records):
        result = pfam_hook(sample_records, {})
        assert result.skipped is True

    @patch("bitescore.tools.hmmer.hmmscan_detailed")
    def test_returns_domain_hits(self, mock_hmmscan, sample_records, tmp_path):
        mock_hmmscan.return_value = {
            "prot_A": [
                {"target_name": "Kunitz_BPTI", "target_accession": "PF00014",
                 "evalue": 1e-25, "score": 100.0, "description": "Kunitz domain"},
            ],
        }
        # Create a pfam2go file
        pfam2go_file = tmp_path / "pfam2go"
        pfam2go_file.write_text("PF00014\tGO:0004866;GO:0005576\n")

        cfg = {"pfam_hmms": "/path/to/Pfam-A.hmm", "pfam2go": str(pfam2go_file)}
        result = pfam_hook(sample_records, cfg)
        assert result.available is True
        hit = result.hits_by_query["prot_A"][0]
        assert hit.domain_name == "Kunitz_BPTI"
        assert "GO:0004866" in hit.go_terms
        assert "GO:0005576" in hit.go_terms

    @patch("bitescore.tools.hmmer.hmmscan_detailed")
    def test_without_pfam2go(self, mock_hmmscan, sample_records):
        mock_hmmscan.return_value = {
            "prot_A": [
                {"target_name": "Kunitz_BPTI", "target_accession": "PF00014",
                 "evalue": 1e-25, "score": 100.0},
            ],
        }
        cfg = {"pfam_hmms": "/path/to/Pfam-A.hmm"}
        result = pfam_hook(sample_records, cfg)
        assert result.available is True
        hit = result.hits_by_query["prot_A"][0]
        assert hit.go_terms == []


class TestInterproHook:
    def test_skips_when_disabled(self, sample_records):
        result = interpro_hook(sample_records, {"interpro": False})
        assert result.skipped is True

    @patch("bitescore.tools.interpro.interproscan_detailed")
    def test_returns_annotations(self, mock_ipr, sample_records):
        mock_ipr.return_value = {
            "prot_A": [
                {
                    "analysis_db": "Pfam",
                    "signature_accession": "PF00014",
                    "signature_description": "Kunitz BPTI",
                    "evalue": 1e-20,
                    "ipr_accession": "IPR002223",
                    "ipr_description": "Kunitz-type protease inhibitor",
                    "go_terms": ["GO:0004866"],
                },
            ],
        }
        cfg = {"interpro": True}
        result = interpro_hook(sample_records, cfg)
        assert result.available is True
        hit = result.hits_by_query["prot_A"][0]
        assert hit.ipr_accession == "IPR002223"
        assert "GO:0004866" in hit.go_terms


# ---------------------------------------------------------------------------
# Hook runner
# ---------------------------------------------------------------------------

class TestRunAnnotationHooks:
    def test_runs_all_hooks(self, sample_records):
        # Without any DBs configured, all hooks should skip gracefully
        results = run_annotation_hooks(sample_records, {})
        assert "diamond" in results
        assert "blast" in results
        assert "pfam" in results
        assert "interpro" in results
        assert all(r.skipped for r in results.values())

    def test_runs_selected_hooks(self, sample_records):
        results = run_annotation_hooks(sample_records, {}, hooks=["diamond", "pfam"])
        assert "diamond" in results
        assert "pfam" in results
        assert "blast" not in results

    def test_registry_contains_all_hooks(self):
        assert set(HOOK_REGISTRY.keys()) == {"diamond", "blast", "pfam", "interpro"}


# ---------------------------------------------------------------------------
# hooks_to_evidence conversion
# ---------------------------------------------------------------------------

class TestHooksToEvidence:
    def test_converts_pfam_hit_with_go(self):
        hook_results = {
            "pfam": HookResult(
                source_method="Pfam",
                hits_by_query={
                    "prot_A": [
                        HitDetail(
                            query_id="prot_A",
                            subject_id="PF00014",
                            evalue=1e-25,
                            bitscore=100.0,
                            domain_name="Kunitz_BPTI",
                            go_terms=["GO:0004866"],
                            database="Pfam",
                        ),
                    ],
                },
            ),
        }
        evidence = hooks_to_evidence("prot_A", hook_results)
        assert len(evidence) == 1
        ev = evidence[0]
        assert ev["go_id"] == "GO:0004866"
        assert ev["source_method"] == "Pfam2GO"
        assert ev["confidence_score"] == 0.80  # evalue <= 1e-20

    def test_resolves_diamond_hit_via_go_records(self):
        from bitescore.features.function import UniProtRecord, UniProtGO

        go_records = {
            "P00001": UniProtRecord(
                accession="P00001",
                sequence=None,
                entry_type="Swiss-Prot",
                go_terms=[
                    UniProtGO(go_id="GO:0005509", name="calcium ion binding",
                              aspect="MF", evidence_code="EXP"),
                ],
            ),
        }
        hook_results = {
            "diamond": HookResult(
                source_method="DIAMOND",
                hits_by_query={
                    "prot_A": [
                        HitDetail(
                            query_id="prot_A",
                            subject_id="sp|P00001|NAME",
                            identity_percent=92.0,
                            query_coverage_percent=85.0,
                            evalue=1e-50,
                            bitscore=300.0,
                            database="UniProtKB",
                        ),
                    ],
                },
            ),
        }
        evidence = hooks_to_evidence("prot_A", hook_results, go_records=go_records)
        assert len(evidence) == 1
        ev = evidence[0]
        assert ev["go_id"] == "GO:0005509"
        assert ev["source_method"] == "DIAMOND"
        assert ev["evidence_code"] == "ISS"
        assert ev["confidence_score"] == 0.90

    def test_skips_hits_without_go(self):
        hook_results = {
            "diamond": HookResult(
                source_method="DIAMOND",
                hits_by_query={
                    "prot_A": [
                        HitDetail(
                            query_id="prot_A",
                            subject_id="UNKNOWN",
                            identity_percent=95.0,
                            database="UniProtKB",
                        ),
                    ],
                },
            ),
        }
        evidence = hooks_to_evidence("prot_A", hook_results)
        assert len(evidence) == 0

    def test_skips_skipped_hooks(self):
        hook_results = {
            "diamond": HookResult(source_method="DIAMOND", skipped=True),
        }
        evidence = hooks_to_evidence("prot_A", hook_results)
        assert len(evidence) == 0

    def test_missing_query_returns_empty(self):
        hook_results = {
            "pfam": HookResult(
                source_method="Pfam",
                hits_by_query={"prot_B": [HitDetail(query_id="prot_B", subject_id="PF00014", go_terms=["GO:0004866"], database="Pfam")]},
            ),
        }
        evidence = hooks_to_evidence("prot_A", hook_results)
        assert len(evidence) == 0


# ---------------------------------------------------------------------------
# Integration with compute_function_features
# ---------------------------------------------------------------------------

class TestComputeFunctionFeaturesWithHooks:
    def test_without_cfg_uses_legacy_path(self, single_record):
        """Without cfg, should work the same as before (legacy path)."""
        df = compute_function_features(single_record)
        assert "id" in df.columns
        assert len(df) == 1

    def test_with_empty_cfg_hooks_skip_gracefully(self, single_record):
        """With cfg but no DBs configured, hooks skip and legacy fallback runs."""
        cfg = {"outdir": "/tmp/test"}
        df = compute_function_features(single_record, cfg=cfg)
        assert "id" in df.columns
        assert len(df) == 1

    @patch("bitescore.features.extract.run_annotation_hooks")
    def test_hook_evidence_merged_into_annotation(self, mock_hooks, single_record):
        """When hooks return GO terms, they appear in the combined annotation."""
        mock_hooks.return_value = {
            "pfam": HookResult(
                source_method="Pfam",
                hits_by_query={
                    "test_protein": [
                        HitDetail(
                            query_id="test_protein",
                            subject_id="PF00014",
                            evalue=1e-25,
                            bitscore=100.0,
                            domain_name="Kunitz_BPTI",
                            go_terms=["GO:0004866", "GO:0005576"],
                            database="Pfam",
                        ),
                    ],
                },
            ),
            "diamond": HookResult(source_method="DIAMOND", skipped=True),
            "blast": HookResult(source_method="BLAST", skipped=True),
            "interpro": HookResult(source_method="InterProScan", skipped=True),
        }
        cfg = {"pfam_hmms": "/path/to/Pfam-A.hmm"}
        df = compute_function_features(single_record, cfg=cfg)
        row = df.iloc[0]

        terms = {e["go_id"]: e for e in json.loads(row["go_terms_json"])}
        # GO:0004866 should be present from Pfam hook evidence
        assert "GO:0004866" in terms
        assert bool(row["red_flag"]) is True  # protease inhibitor is a red flag

    def test_empty_records_returns_template(self):
        df = compute_function_features([], cfg={})
        assert "id" in df.columns
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Pfam2GO loader in hooks module
# ---------------------------------------------------------------------------

class TestHooksPfam2goLoader:
    def test_standard_go_format(self, tmp_path):
        f = tmp_path / "pfam2go"
        f.write_text(
            "!version: 2024-01\n"
            "Pfam:PF00014 Kunitz_BPTI > GO:endopeptidase inhibitor ; GO:0004866\n"
        )
        mapping = _load_pfam2go(str(f))
        assert "GO:0004866" in mapping["PF00014"]

    def test_tsv_format(self, tmp_path):
        f = tmp_path / "pfam2go.tsv"
        f.write_text("PF00139\tGO:0030246;GO:0005576\n")
        mapping = _load_pfam2go(str(f))
        assert len(mapping["PF00139"]) == 2
