from __future__ import annotations

import asyncio
import json
import os
import tempfile
import traceback
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..pipeline import (
    GENOME_INPUT_TYPES,
    run_pipeline,
    path_ranked,
    path_features_aa,
    path_features_regsite,
    path_features_structure,
    path_features_function,
    path_loaded,
    path_masked,
    path_clustered,
    path_called,
)
from ..utils.config import load_config
from .schemas import (
    AnalysisResult,
    AnalysisStatus,
    ExampleInfo,
    ProgressUpdate,
    SequenceDetail,
    SequenceSummary,
)

ASSETS_DIR = Path(__file__).parent / "assets"
EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend" / "dist"

INPUT_TYPE_MAP: Dict[str, str] = {
    "proteomic": "proteome",
    "metaproteomic": "proteome",
    "genomic": "genome",
    "metagenomic": "metagenome",
    "sequence": "sequences",
    "sequences": "sequences",
    "proteome": "proteome",
    "genome": "genome",
    "metagenome": "metagenome",
}

SINGLE_SEQUENCE_EXAMPLE = """>sp|Q9XYZ1|EXAMPLE1 Example protein 1
MSTNPKPQRITKRRVVYAAFVVLLVLTALLASSSKRRRYYYAA
"""

MULTI_SEQUENCE_EXAMPLE = """>sp|Q9XYZ1|EXAMPLE1 Example protein 1
MSTNPKPQRITKRRVVYAAFVVLLVLTALLASSSKRRRYYYAA
>sp|P12345|EXAMPLE2 Example protein 2
MKKLLPTAAAGLLLLAAQPAMARRRKKKYYFWYVVVVTTTTAA
"""

# In-memory job store
_jobs: Dict[str, Dict[str, Any]] = {}
_ws_connections: Dict[str, List[WebSocket]] = {}

app = FastAPI(title="BiteScore API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_input_type(label: str | None) -> str:
    if not label:
        return "proteome"
    key = str(label).strip().lower()
    return INPUT_TYPE_MAP.get(key, "proteome")


def _save_sequences_to_fasta(text: str, dest: Path) -> Path:
    text = text.strip()
    if not text:
        return dest
    if text.startswith(">") or "\n>" in text:
        dest.write_text(text)
        return dest
    recs: list[SeqRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        recs.append(SeqRecord(Seq(s), id=f"userseq_{i}", description=f"userseq_{i}"))
    SeqIO.write(recs, str(dest), "fasta")
    return dest


def _ensure_record_ids(records: list[SeqRecord]) -> list[SeqRecord]:
    for idx, rec in enumerate(records, start=1):
        original_id = rec.id or ""
        normalized_id = original_id.strip()
        if not normalized_id:
            normalized_id = f"sequence_{idx}"
        rec.id = normalized_id
        rec.name = normalized_id
        if not (rec.description or "").strip():
            rec.description = normalized_id
    return records


def _feature_sequence_records(outdir: Path, input_type: str) -> list[SeqRecord]:
    candidates: list[Path] = []
    mask = path_masked(outdir)
    if mask.exists():
        candidates.append(mask)
    clustered = path_clustered(outdir)
    if clustered.exists():
        candidates.append(clustered)
    if input_type in GENOME_INPUT_TYPES:
        called = path_called(outdir)
        if called.exists():
            candidates.append(called)
    loaded = path_loaded(outdir, input_type)
    if loaded.exists():
        candidates.append(loaded)
    for candidate in candidates:
        try:
            records = list(SeqIO.parse(str(candidate), "fasta"))
        except OSError:
            continue
        if records:
            return records
    return []


def _collect_pipeline_outputs(outdir: Path, input_type: str) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    ranked_path = path_ranked(outdir)
    if ranked_path.exists():
        outputs["ranked_path"] = ranked_path
        outputs["ranked"] = pd.read_csv(ranked_path)
    feature_paths = {
        "aa": path_features_aa(outdir),
        "regsite": path_features_regsite(outdir),
        "structure": path_features_structure(outdir),
        "function": path_features_function(outdir),
    }
    feature_tables: Dict[str, pd.DataFrame] = {}
    for name, path in feature_paths.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "id" in table.columns:
            table = table.set_index("id")
        feature_tables[name] = table
    if feature_tables:
        outputs["features"] = feature_tables
    sequence_records = _feature_sequence_records(outdir, input_type)
    if sequence_records:
        outputs["sequences"] = {rec.id: str(rec.seq) for rec in sequence_records}
    return outputs


def _run_pipeline_sync(
    input_path: Path,
    input_type: str,
    organism: str | None,
    outdir: Path,
    opts: dict,
) -> Dict[str, Any]:
    overrides = dict(
        input_path=str(input_path),
        input_type=input_type,
        organism=organism,
        outdir=str(outdir),
        structure_enabled=not opts.get("no_structure", False),
        alphafold_enabled=bool(opts.get("alphafold", False)),
        go_map=opts.get("go_map"),
        diamond_db=opts.get("diamond_db"),
        blast_db=opts.get("blast_db"),
        pfam_hmms=opts.get("pfam_hmms"),
        interpro=bool(opts.get("interpro", False)),
        cluster_cdhit=bool(opts.get("cluster_cdhit", False)),
        cdhit_threshold=opts.get("cdhit_threshold"),
        low_complexity=bool(opts.get("low_complexity", False)),
        train_demo=True,
    )
    cfg = load_config(None, overrides)
    run_pipeline(cfg)
    resolved_input_type = str(cfg.get("input_type") or input_type or "proteome")
    return _collect_pipeline_outputs(Path(cfg["outdir"]), resolved_input_type)


async def _broadcast_progress(job_id: str, percent: float, description: str, status: AnalysisStatus):
    update = ProgressUpdate(
        job_id=job_id, status=status, percent=percent, description=description
    )
    payload = update.model_dump_json()
    for ws in _ws_connections.get(job_id, []):
        try:
            await ws.send_text(payload)
        except Exception:
            pass


def _format_sequence_text(seq_id: str, sequence: str | None, width: int = 60) -> str:
    if not sequence:
        return ""
    seq = sequence.replace("\n", "").strip()
    if not seq:
        return ""
    lines = [f">{seq_id}"]
    for start in range(0, len(seq), width):
        lines.append(seq[start : start + width])
    return "\n".join(lines)


def _blastp_url(seq_id: str, sequence: str | None) -> str | None:
    if not sequence:
        return None
    fasta_text = _format_sequence_text(seq_id, sequence)
    if not fasta_text:
        return None
    encoded = urllib.parse.quote(fasta_text)
    return (
        "https://blast.ncbi.nlm.nih.gov/Blast.cgi?"
        f"PAGE=Proteins&PROGRAM=blastp&QUERY={encoded}"
    )


def _safe_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if pd.isna(f):
            return None
        return f
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _df_row_to_feature_list(df: pd.DataFrame | None, seq_id: str | None) -> list[dict[str, str]]:
    if df is None or seq_id is None:
        return []
    try:
        row = df.loc[seq_id]
    except (KeyError, TypeError):
        return []
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return []
        row = row.iloc[0]
    result = []
    for key, value in row.items():
        if key == "id":
            continue
        v = _safe_value(value)
        if v is None:
            display = "—"
        elif isinstance(v, bool):
            display = "Yes" if v else "No"
        elif isinstance(v, int):
            display = str(v)
        elif isinstance(v, float):
            display = f"{v:.4f}" if not v.is_integer() else str(int(v))
        else:
            display = str(v)
            if len(display) > 160:
                display = display[:157] + "..."
        result.append({"metric": str(key), "value": display})
    return result


def _structure_pdb_text(features: dict, seq_id: str) -> str | None:
    structure_df = features.get("structure")
    if structure_df is None or seq_id is None:
        return None
    try:
        entry = structure_df.loc[seq_id]
    except (KeyError, TypeError):
        return None
    if isinstance(entry, pd.DataFrame):
        if entry.empty:
            return None
        entry = entry.iloc[0]
    pdb_path = entry.get("predicted_structure_path")
    if not pdb_path:
        return None
    path = Path(pdb_path)
    if not path.exists():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/api/examples")
async def get_examples():
    examples = [
        ExampleInfo(
            name="Single protein",
            description="A single example protein sequence for quick testing",
            sequences=SINGLE_SEQUENCE_EXAMPLE.strip(),
            input_type="proteome",
        ),
        ExampleInfo(
            name="Multiple proteins",
            description="Two example protein sequences to demonstrate batch analysis",
            sequences=MULTI_SEQUENCE_EXAMPLE.strip(),
            input_type="proteome",
        ),
    ]
    genome_path = EXAMPLES_DIR / "GCF_000005845.2_ASM584v2_genomic.fna"
    if genome_path.exists():
        examples.append(
            ExampleInfo(
                name="E. coli genome",
                description="E. coli K-12 reference genome (GCF_000005845.2)",
                file_path=str(genome_path),
                input_type="genome",
            )
        )
    return examples


@app.post("/api/analyze")
async def start_analysis(
    input_type: str = Form("proteome"),
    organism: Optional[str] = Form(None),
    organisms: str = Form("[]"),
    sequences: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    options: str = Form("{}"),
):
    job_id = uuid4().hex[:12]
    tmpdir = Path(tempfile.mkdtemp(prefix="bitescore_api_"))
    outdir = tmpdir / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    input_fasta = tmpdir / "input.faa"

    if file:
        content = await file.read()
        input_fasta.write_bytes(content)
    elif sequences and sequences.strip():
        _save_sequences_to_fasta(sequences, input_fasta)
    else:
        raise HTTPException(status_code=400, detail="No sequences or file provided")

    if not input_fasta.exists() or input_fasta.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="No valid sequences found")

    records = list(SeqIO.parse(str(input_fasta), "fasta"))
    records = _ensure_record_ids(records)
    if records:
        SeqIO.write(records, str(input_fasta), "fasta")
    if not records:
        raise HTTPException(status_code=400, detail="No valid sequences found in the provided input")

    pipeline_input_type = _normalize_input_type(input_type)
    if pipeline_input_type in {"genome", "genomes", "metagenome"} and not organism:
        raise HTTPException(
            status_code=400,
            detail="For genome-like inputs, set organism to 'prok' or 'euk'",
        )

    try:
        parsed_organisms = json.loads(organisms)
    except (json.JSONDecodeError, TypeError):
        parsed_organisms = []

    try:
        parsed_options = json.loads(options)
    except (json.JSONDecodeError, TypeError):
        parsed_options = {}

    _jobs[job_id] = {
        "status": AnalysisStatus.pending,
        "input_fasta": input_fasta,
        "input_type": pipeline_input_type,
        "organism": organism,
        "organisms": parsed_organisms,
        "outdir": outdir,
        "tmpdir": tmpdir,
        "options": parsed_options,
        "records": records,
        "result": None,
        "error": None,
    }

    asyncio.get_event_loop().run_in_executor(None, _run_job, job_id)

    return {"job_id": job_id, "status": "pending"}


def _run_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return
    job["status"] = AnalysisStatus.running
    try:
        pipeline_outputs = _run_pipeline_sync(
            job["input_fasta"],
            job["input_type"],
            job["organism"],
            job["outdir"],
            job["options"],
        )
        job["result"] = pipeline_outputs
        job["status"] = AnalysisStatus.completed
    except Exception as exc:
        job["error"] = str(exc)
        job["status"] = AnalysisStatus.failed
        traceback.print_exc()


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response: Dict[str, Any] = {
        "job_id": job_id,
        "status": job["status"].value,
        "input_type": job["input_type"],
        "organisms": job.get("organisms", []),
        "sequence_count": len(job.get("records", [])),
    }

    if job["status"] == AnalysisStatus.failed:
        response["error"] = job.get("error", "Unknown error")

    if job["status"] == AnalysisStatus.completed and job.get("result"):
        outputs = job["result"]
        ranked_df = outputs.get("ranked")
        if ranked_df is not None and not ranked_df.empty:
            ranked_df = ranked_df.copy().reset_index(drop=True)
            features = outputs.get("features", {}) or {}
            summaries = []
            for i, row in ranked_df.iterrows():
                seq_id = str(row.get("id", ""))
                length = None
                if "aa" in features and "length" in features["aa"].columns:
                    try:
                        length = int(features["aa"].loc[seq_id, "length"])
                    except (KeyError, TypeError, ValueError):
                        pass
                summaries.append(
                    SequenceSummary(
                        id=seq_id,
                        rank=i + 1,
                        length=length,
                        digestibility_score=_safe_value(row.get("digestibility_score")),
                        aa_essential_frac=_safe_value(row.get("aa_essential_frac")),
                    ).model_dump()
                )
            response["ranked"] = summaries

        ranked_path = outputs.get("ranked_path")
        if ranked_path:
            response["download_url"] = f"/api/jobs/{job_id}/download"

    return response


@app.get("/api/jobs/{job_id}/sequence/{seq_id}")
async def get_sequence_detail(job_id: str, seq_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != AnalysisStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")

    outputs = job.get("result", {})
    ranked_df = outputs.get("ranked")
    features = outputs.get("features", {}) or {}
    sequences = outputs.get("sequences", {}) or {}

    if ranked_df is None or ranked_df.empty:
        raise HTTPException(status_code=404, detail="No results available")

    row = ranked_df[ranked_df["id"] == seq_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Sequence '{seq_id}' not found")
    row = row.iloc[0]
    rank_idx = ranked_df.index.get_loc(row.name) + 1

    metrics: Dict[str, Any] = {}
    ranked_columns = [
        "aa_essential_frac", "protease_total_sites", "trypsin_K_sites",
        "trypsin_R_sites", "cleavage_site_accessible_fraction",
        "disorder_propensity_mean", "plddt_mean",
        "red_flag", "green_flag", "digestibility_score",
    ]
    for col in ranked_columns:
        if col in row:
            v = _safe_value(row[col])
            if v is not None:
                metrics[col] = v

    feature_data: Dict[str, list] = {}
    for section in ("aa", "regsite", "structure", "function"):
        feature_data[section] = _df_row_to_feature_list(features.get(section), seq_id)

    sequence_text = sequences.get(seq_id)
    blastp = _blastp_url(seq_id, sequence_text)

    structure_pdb = _structure_pdb_text(features, seq_id)

    return SequenceDetail(
        id=seq_id,
        rank=rank_idx,
        sequence=sequence_text,
        digestibility_score=_safe_value(row.get("digestibility_score")),
        metrics=metrics,
        features=feature_data,
        structure_available=structure_pdb is not None,
        blastp_url=blastp,
    ).model_dump()


@app.get("/api/jobs/{job_id}/sequence/{seq_id}/structure")
async def get_sequence_structure(job_id: str, seq_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != AnalysisStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")

    features = job.get("result", {}).get("features", {}) or {}
    pdb_text = _structure_pdb_text(features, seq_id)
    if pdb_text is None:
        raise HTTPException(status_code=404, detail="No structure available")
    return JSONResponse(content={"pdb": pdb_text})


@app.get("/api/jobs/{job_id}/download")
async def download_results(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != AnalysisStatus.completed:
        raise HTTPException(status_code=400, detail="Job not yet completed")
    ranked_path = job.get("result", {}).get("ranked_path")
    if not ranked_path or not ranked_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    return FileResponse(
        ranked_path,
        media_type="text/csv",
        filename=f"bitescore_ranked_{job_id}.csv",
    )


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in _ws_connections:
        _ws_connections[job_id] = []
    _ws_connections[job_id].append(websocket)
    try:
        while True:
            job = _jobs.get(job_id)
            if job:
                status = job["status"]
                update = ProgressUpdate(
                    job_id=job_id,
                    status=status,
                    percent=100.0 if status in (AnalysisStatus.completed, AnalysisStatus.failed) else 50.0,
                    description="Analysis complete" if status == AnalysisStatus.completed
                    else "Analysis failed" if status == AnalysisStatus.failed
                    else "Running analysis...",
                )
                await websocket.send_text(update.model_dump_json())
                if status in (AnalysisStatus.completed, AnalysisStatus.failed):
                    break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        conns = _ws_connections.get(job_id, [])
        if websocket in conns:
            conns.remove(websocket)


# Serve logo
@app.get("/api/assets/logo.png")
async def get_logo():
    logo_path = ASSETS_DIR / "logo.png"
    if not logo_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(logo_path, media_type="image/png")


@app.get("/api/assets/team.jpg")
async def get_team_photo():
    photo_path = ASSETS_DIR / "IMG_0021.JPG"
    if not photo_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(photo_path, media_type="image/jpeg")


# Serve frontend static files (after build)
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="frontend-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
