from __future__ import annotations

import base64
import html
import json
import os
import tempfile
import urllib.parse
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict
from uuid import uuid4

import gradio as gr
from gradio.events import SelectData
import pandas as pd
import numpy as np
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
    path_masked,
    path_clustered,
    path_called,
    path_loaded,
)
from ..utils.config import load_config

APP_TITLE = "BiteScore"
APP_DESC = "Upload a FASTA or paste sequences."

INPUT_TYPE_MAP: Dict[str, str] = {
    "proteomic": "proteome",
    "metaproteomic": "proteome",
    "genomic": "genome",
    "metagenomic": "metagenome",
    "sequence": "sequences",
    "sequences": "sequences",
}

GENOME_ORGANISM_LABELS: Dict[str, str] = {
    "prok": "Prokaryotic",
    "euk": "Eukaryotic",
}

FEATURE_SECTION_TITLES: Dict[str, str] = {
    "aa": "Amino Acid Composition",
    "regsite": "Protease Recognition Sites",
    "structure": "Cleavage Accessibility",
    "function": "Functional Annotations",
}

FEATURE_TABLE_COLUMNS = ("aa", "regsite", "structure", "function")

STRUCTURE_PLACEHOLDER_HTML = """
<div style="padding: 18px; border-radius: 14px; background: linear-gradient(135deg, #ecf2ff, #f8faff); border: 1px solid #c9dffb; color: #27446b; font-size: 14px; box-shadow: 0 14px 32px rgba(46, 94, 152, 0.08);">
    Select a sequence to preview its predicted structure. The viewer will appear here after running the analysis.
</div>
"""


def _analysis_progress_html(percent: float, description: str) -> str:
    value = 0 if percent is None else float(percent)
    if value <= 1:
        value *= 100
    clamped = max(0, min(100, int(round(value))))
    safe_description = description or "Processing"
    return f"""
<div class=\"analysis-progress__wrapper\">
    <div class=\"analysis-progress__label\">{safe_description}</div>
    <div class=\"analysis-progress__bar\">
        <div class=\"analysis-progress__fill\" style=\"width: {clamped}%\"></div>
    </div>
    <div class=\"analysis-progress__percent\">{clamped}%</div>
</div>
"""


def _analysis_progress_update(percent: float, description: str, visible: bool = True) -> gr.Update:
    return gr.update(value=_analysis_progress_html(percent, description), visible=visible)


def _format_extension_error_popup(expected_ext: str, actual_ext: str | None, input_type_label: str) -> str:
    expected_safe = html.escape(expected_ext)
    if actual_ext:
        actual_safe = html.escape(actual_ext)
    else:
        actual_safe = "no extension"
    type_safe = html.escape(input_type_label or "the selected")
    return f"""
<div class=\"format-error-overlay\" onclick=\"this.style.display='none'\">
    <div class=\"format-error-modal\">
        <div class=\"format-error-title\">File format mismatch</div>
        <p>For <strong>{type_safe}</strong> inputs, upload FASTA files ending with <code>{expected_safe}</code>.</p>
        <p>Uploaded file extension: <code>{actual_safe}</code></p>
        <p class=\"format-error-dismiss\">Click anywhere to dismiss this message and adjust the input type or file.</p>
    </div>
</div>
"""

SINGLE_SEQUENCE_EXAMPLE = """>sp|Q9XYZ1|EXAMPLE1 Example protein 1
MSTNPKPQRITKRRVVYAAFVVLLVLTALLASSSKRRRYYYAA
"""

MULTI_SEQUENCE_EXAMPLE = """>sp|Q9XYZ1|EXAMPLE1 Example protein 1
MSTNPKPQRITKRRVVYAAFVVLLVLTALLASSSKRRRYYYAA
>sp|P12345|EXAMPLE2 Example protein 2
MKKLLPTAAAGLLLLAAQPAMARRRKKKYYFWYVVVVTTTTAA
"""

# Set static paths for assets
ASSETS_DIR = Path(__file__).parent / "assets"
EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
BACTERIAL_GENOME_EXAMPLE_PATH = EXAMPLES_DIR / "GCF_000005845.2_ASM584v2_genomic.fna"


def _register_asset_path(directory: Path, alias: str = "") -> str:
    """Register a static asset directory and return the alias actually used."""

    if alias:
        try:
            gr.set_static_paths(paths=[(str(directory), alias)])
        except (TypeError, ValueError):
            # Older Gradio releases only accept bare string paths.
            gr.set_static_paths(paths=[str(directory)])
            return ""
        else:
            return alias
    gr.set_static_paths(paths=[str(directory)])
    return ""


ASSETS_ALIAS = _register_asset_path(ASSETS_DIR, alias="app-assets")
LOGO_PATH = ASSETS_DIR / "logo.png"
TEAM_PHOTO_PATH = ASSETS_DIR / "IMG_0021.JPG"


def _load_asset_source(
    asset_path: Path,
    fallback_name: str,
    mime_type: str,
    *,
    max_inline_bytes: int = 400_000,
    assets_alias: str = "",
) -> str:
    """Return a source attribute for an asset, inlining only small files.

    Large assets noticeably slow initial page load when base64 encoded directly
    into the HTML. Keeping them as static file references allows the browser to
    fetch them lazily without blocking Gradio's loading spinner.
    """

    def _fallback() -> str:
        if assets_alias:
            return f"file={assets_alias}/{fallback_name}"
        return f"file={fallback_name}"

    try:
        size = asset_path.stat().st_size
    except OSError:
        return _fallback()

    if size is None or size > max_inline_bytes:
        return _fallback()

    try:
        data = asset_path.read_bytes()
    except OSError:
        return _fallback()

    encoded = base64.b64encode(data).decode("ascii").strip()
    if not encoded:
        return _fallback()

    return f"data:{mime_type};base64,{encoded}"


LOGO_SRC = _load_asset_source(
    LOGO_PATH,
    "logo.png",
    "image/png",
    assets_alias=ASSETS_ALIAS,
)
TEAM_PHOTO_SRC = _load_asset_source(
    TEAM_PHOTO_PATH,
    "IMG_0021.JPG",
    "image/jpeg",
    max_inline_bytes=200_000,
    assets_alias=ASSETS_ALIAS,
)


class PipelineOutputs(TypedDict, total=False):
    ranked: pd.DataFrame
    ranked_path: Path
    features: Dict[str, pd.DataFrame]
    sequences: Dict[str, str]


def _normalize_input_type(label: str | None) -> str:
    if not label:
        return "proteome"
    key = str(label).strip().lower()
    return INPUT_TYPE_MAP.get(key, key if key in INPUT_TYPE_MAP.values() else "proteome")


def _normalize_genome_organism(choice: str | None) -> str | None:
    if not choice:
        return None
    label = str(choice).strip().lower()
    if label in {"prok", "prokaryote", "prokaryotic"}:
        return "prok"
    if label in {"euk", "eukaryote", "eukaryotic"}:
        return "euk"
    return None


def _format_genome_organism(choice: str | None) -> str:
    if not choice:
        return ""
    normalized = _normalize_genome_organism(choice)
    if not normalized:
        return str(choice)
    return GENOME_ORGANISM_LABELS.get(normalized, normalized)


def _normalize_messages(messages: Iterable[Any] | None) -> List[dict[str, str]]:
    normalized: List[dict[str, str]] = []
    if not messages:
        return normalized
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", "")
        if role is None:
            continue
        normalized.append({
            "role": str(role),
            "content": "" if content is None else str(content),
        })
    return normalized

def _ensure_record_ids(records: List[SeqRecord]) -> Tuple[List[SeqRecord], bool]:
    """Ensure every record has a stable identifier and minimal description.

    Returns the mutated record list along with a boolean indicating whether any
    changes were applied. When identifiers are missing or contain surrounding
    whitespace they are normalised to ``sequence_{index}`` style values so that
    downstream tables always contain a usable key.
    """

    updated = False
    for idx, rec in enumerate(records, start=1):
        original_id = rec.id or ""
        normalized_id = original_id.strip()
        if not normalized_id:
            normalized_id = f"sequence_{idx}"
        if normalized_id != rec.id:
            rec.id = normalized_id
            updated = True
        if rec.name != normalized_id:
            rec.name = normalized_id
            updated = True
        description = rec.description or ""
        if not description.strip():
            rec.description = normalized_id
            if description != normalized_id:
                updated = True
    return records, updated


def _save_sequences_to_fasta(text: str, dest: Path) -> Path:
    recs: List[SeqRecord] = []
    text = text.strip()
    if not text:
        return dest
    if text.startswith(">") or "\n>" in text:
        dest.write_text(text)
        return dest
    else:
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            recs.append(SeqRecord(Seq(s), id=f"userseq_{i}", description=f"userseq_{i}"))
        SeqIO.write(recs, str(dest), "fasta")
        return dest


def _feature_sequence_records(outdir: Path, input_type: str) -> List[SeqRecord]:
    """Return the amino-acid records used for feature extraction.

    The pipeline may generate several intermediate FASTA files. We follow the
    same precedence as :func:`bitescore.pipeline._feature_sequences_for_extraction`
    to locate the most processed set of sequences so that the UI can display
    the exact proteins that were ranked (including gene-calling outputs for
    genome inputs).
    """

    candidates: List[Path] = []
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


def _collect_pipeline_outputs(outdir: Path, input_type: str) -> PipelineOutputs:
    outputs: PipelineOutputs = {}
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


def _run_pipeline_helper(input_path: Path, input_type: str, organism: str | None, outdir: Path, opts: dict):
    overrides = dict(
        input_path=str(input_path), input_type=input_type, organism=organism, outdir=str(outdir),
        structure_enabled=not opts.get("no_structure", False),
        alphafold_enabled=bool(opts.get("alphafold", False)),
        go_map=opts.get("go_map"), diamond_db=opts.get("diamond_db"),
        blast_db=opts.get("blast_db"), pfam_hmms=opts.get("pfam_hmms"),
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


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "—"
    if isinstance(value, Number):
        if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer()):
            return str(int(value))
        return f"{float(value):.4f}"
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(v) for v in value)
    text = str(value)
    if not text:
        return "—"
    if len(text) > 160:
        return text[:157] + "..."
    return text


def _series_to_markdown(series: pd.Series) -> str:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[0]
    data = []
    for key, value in series.items():
        if key == "id":
            continue
        formatted = _format_metric_value(value if not (isinstance(value, float) and pd.isna(value)) else None)
        data.append({"Metric": key, "Value": formatted})
    if not data:
        return "_No data available._"
    display = pd.DataFrame(data)
    return display.to_markdown(index=False)


def _series_to_dataframe(series: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(series, pd.DataFrame):
        if series.empty:
            return pd.DataFrame(columns=["Metric", "Value"])
        series = series.iloc[0]
    rows: List[Dict[str, str]] = []
    for key, value in series.items():
        if key == "id":
            continue
        formatted = _format_metric_value(
            value if not (isinstance(value, float) and pd.isna(value)) else None
        )
        rows.append({"Metric": key, "Value": formatted})
    if not rows:
        return pd.DataFrame(columns=["Metric", "Value"])
    return pd.DataFrame(rows)


def _feature_row_dataframe(
    feature_df: Optional[pd.DataFrame], seq_id: Optional[str]
) -> pd.DataFrame:
    if feature_df is None or seq_id is None:
        return pd.DataFrame(columns=["Metric", "Value"])
    try:
        series = feature_df.loc[seq_id]
    except (KeyError, TypeError):
        return pd.DataFrame(columns=["Metric", "Value"])
    return _series_to_dataframe(series)


def _prepare_feature_tables(
    features: Dict[str, pd.DataFrame], seq_id: Optional[str]
) -> Tuple[Dict[str, pd.DataFrame], bool]:
    tables: Dict[str, pd.DataFrame] = {}
    has_any = False
    for key in FEATURE_TABLE_COLUMNS:
        table = _feature_row_dataframe(features.get(key), seq_id)
        tables[key] = table
        if not table.empty:
            has_any = True
    return tables, has_any


def _structure_placeholder(message: str | None = None) -> str:
    text = message or "No predicted structure is available for the selected sequence."
    return (
        "<div style=\"padding: 18px; border-radius: 14px; background: linear-gradient(135deg, #edf3ff, #f9fbff);"
        " border: 1px solid #c9dffb; color: #27446b; font-size: 14px; box-shadow: 0 14px 32px rgba(46, 94, 152, 0.08);\">"
        f"{text}"
        "</div>"
    )


def _structure_view_content(
    structure_df: Optional[pd.DataFrame], seq_id: Optional[str]
) -> Tuple[str, bool]:
    if structure_df is None or seq_id is None:
        return STRUCTURE_PLACEHOLDER_HTML, False
    try:
        entry = structure_df.loc[seq_id]
    except (KeyError, TypeError):
        return _structure_placeholder(), False
    if isinstance(entry, pd.DataFrame):
        if entry.empty:
            return _structure_placeholder(), False
        entry = entry.iloc[0]
    pdb_path = entry.get("predicted_structure_path")
    if not pdb_path:
        return _structure_placeholder(
            "Structure prediction was skipped for this sequence."
        ), False
    path = Path(pdb_path)
    if not path.exists():
        return _structure_placeholder(
            "Predicted structure file could not be found."
        ), False
    try:
        pdb_text = path.read_text()
    except OSError:
        return _structure_placeholder(
            "Unable to read the predicted structure file."
        ), False

    container_id = f"structure-viewer-{abs(hash(seq_id)) % (36 ** 8):08x}"
    pdb_json = json.dumps(pdb_text)
    html = f"""
<div style=\"margin-top: 12px;\">
  <div style=\"display: flex; justify-content: space-between; align-items: center;\">
    <h3 style=\"margin: 0; font-size: 16px; color: #1f3b64;\">Predicted structure</h3>
    <span style=\"font-size: 13px; color: #5a7dac;\">Source: {path.name}</span>
  </div>
  <div id=\"{container_id}\" style=\"width: 100%; height: 420px; border: 1px solid #b6cff7; border-radius: 14px; margin-top: 12px; background: linear-gradient(135deg, #f4f8ff, #ffffff); box-shadow: 0 16px 32px rgba(36, 82, 142, 0.12);\"></div>
</div>
<script>
(function() {{
    const mount = document.getElementById("{container_id}");
    if (!mount) {{
        return;
    }}
    const renderViewer = () => {{
        if (typeof $3Dmol === "undefined") {{
            return;
        }}
        try {{
            const viewer = $3Dmol.createViewer(mount, {{ backgroundColor: "white" }});
            viewer.addModel({pdb_json}, "pdb");
            viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum" }} }});
            viewer.zoomTo();
            viewer.render();
        }} catch (err) {{
            mount.innerHTML = '<div style=\"padding: 16px; color: #c0392b;\">Failed to render structure: ' + err + '</div>';
        }}
    }};
    if (typeof $3Dmol !== "undefined") {{
        renderViewer();
        return;
    }}
    let script = document.querySelector('script[data-bitescore-3dmol]');
    if (!script) {{
        script = document.createElement('script');
        script.src = 'https://3dmol.org/build/3Dmol-min.js';
        script.dataset.bitescore3dmol = '1';
        script.onload = renderViewer;
        document.head.appendChild(script);
    }} else {{
        script.addEventListener('load', renderViewer, {{ once: true }});
    }}
}})();
</script>
"""
    return html, True

def _build_ranking_table(ranked: pd.DataFrame, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    ranked = ranked.copy().reset_index(drop=True)
    if "Rank" in ranked.columns:
        ranked = ranked.drop(columns=["Rank"])
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    desired_columns = ["Rank", "id", "aa_essential_frac", "digestibility_score", "length"]
    existing_columns = [col for col in desired_columns if col in ranked.columns]
    table = ranked[existing_columns].copy()
    if "length" not in table.columns and "aa" in features:
        aa_table = features["aa"]
        if "length" in aa_table.columns:
            lengths = aa_table["length"].to_dict()
            table["length"] = table["id"].map(lengths)
    table = table.rename(columns={
        "id": "Sequence ID",
        "aa_essential_frac": "aa_essential_fraction",
        "digestibility_score": "Digestibility Score",
        "length": "Length",
    })
    if "Digestibility Score" in table.columns:
        table["Digestibility Score"] = table["Digestibility Score"].apply(
            lambda v: float(v) if pd.notna(v) else v
        )
    if "aa_essential_fraction" in table.columns:
        table["aa_essential_fraction"] = table["aa_essential_fraction"].apply(
            lambda v: float(v) if pd.notna(v) else v
        )
    return table


def _build_sequence_card(
    seq_id: str, ranked: pd.DataFrame, _features: Dict[str, pd.DataFrame]
) -> str:
    if not seq_id:
        return "<p>No sequence selected.</p>"
    if ranked is None or ranked.empty:
        return "<p>No ranking data available.</p>"
    row = ranked[ranked["id"] == seq_id]
    if row.empty:
        safe_seq = html.escape(seq_id)
        return f"<p>Sequence <code>{safe_seq}</code> not found in results.</p>"
    row = row.iloc[0]

    ranked_columns = [
        "aa_essential_frac",
        "protease_total_sites",
        "trypsin_K_sites",
        "trypsin_R_sites",
        "cleavage_site_accessible_fraction",
        "red_flag",
        "green_flag",
        "digestibility_score",
    ]

    int_like_columns = {
        "protease_total_sites",
        "trypsin_K_sites",
        "trypsin_R_sites",
    }

    bool_like_columns = {"red_flag", "green_flag"}

    column_labels = {
        "aa_essential_frac": "aa_essential_fraction",
    }

    chart_labels: List[str] = []
    chart_values: List[float] = []
    chart_display_values: List[str] = []
    value_items: List[str] = []

    for column in ranked_columns:
        if column not in row:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue

        label = column_labels.get(column, column)
        display_value: str
        numeric_value: Optional[float] = None

        if isinstance(value, (bool, np.bool_)) or column in bool_like_columns:
            bool_value = bool(value)
            display_value = "True" if bool_value else "False"
            numeric_value = 1.0 if bool_value else 0.0
        elif isinstance(value, (float, np.floating)):
            numeric_value = float(value)
            if column in int_like_columns:
                display_value = str(int(round(numeric_value)))
            else:
                display_value = f"{numeric_value:.4f}"
        elif isinstance(value, (int, np.integer)):
            numeric_value = float(value)
            display_value = str(int(value))
        elif isinstance(value, Number):
            numeric_value = float(value)
            display_value = str(value)
        else:
            display_value = str(value)

        if numeric_value is None:
            continue

        chart_labels.append(label)
        chart_values.append(numeric_value)
        chart_display_values.append(display_value)
        value_items.append(
            f"<li><strong>{html.escape(label)}:</strong> {html.escape(display_value)}</li>"
        )

    safe_seq_id = html.escape(seq_id)
    if not chart_labels:
        return (
            f"<div class=\"sequence-card\"><h2>Results for <code>{safe_seq_id}</code></h2>"
            "<p>No metrics available.</p></div>"
        )

    chart_id = f"sequence-card-chart-{uuid4().hex}"
    labels_json = json.dumps(chart_labels)
    values_json = json.dumps(chart_values)
    display_json = json.dumps(chart_display_values)

    list_html = "".join(value_items)

    return f"""
<div class=\"sequence-card\">
  <h2>Results for <code>{safe_seq_id}</code></h2>
  <div class=\"sequence-card__chart\">
    <canvas id=\"{chart_id}\" aria-label=\"Radar chart for {safe_seq_id}\"></canvas>
  </div>
  <ul class=\"sequence-card__metrics\">{list_html}</ul>
</div>
<script>
(function() {{
  const labels = {labels_json};
  const values = {values_json};
  const rawValues = {display_json};
  const chartId = "{chart_id}";

  function renderChart() {{
    const canvas = document.getElementById(chartId);
    if (!canvas || !window.Chart) {{
      return;
    }}
    window.bitescoreSequenceCardCharts = window.bitescoreSequenceCardCharts || {{}};
    const existing = window.bitescoreSequenceCardCharts[chartId];
    if (existing) {{
      existing.destroy();
    }}
    window.bitescoreSequenceCardCharts[chartId] = new Chart(canvas, {{
      type: 'radar',
      data: {{
        labels: labels,
        datasets: [{{
          data: values,
          fill: true,
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgba(54, 162, 235, 1)',
          pointBackgroundColor: 'rgba(54, 162, 235, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(54, 162, 235, 1)'
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: function(context) {{
                const idx = context.dataIndex;
                const label = labels[idx] || '';
                const raw = rawValues[idx] || context.formattedValue;
                return label ? `${{label}}: ${{raw}}` : context.formattedValue;
              }}
            }}
          }}
        }},
        scales: {{
          r: {{
            beginAtZero: true,
            ticks: {{
              precision: 2
            }}
          }}
        }}
      }}
    }});
  }}

  function ensureChartJs() {{
    if (window.Chart) {{
      renderChart();
      return;
    }}
    if (!window.bitescoreChartJsPromise) {{
      window.bitescoreChartJsPromise = new Promise(function(resolve, reject) {{
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js';
        script.onload = function() {{ resolve(); }};
        script.onerror = function(err) {{ reject(err); }};
        document.head.appendChild(script);
      }}).catch(function(err) {{
        console.error('Failed to load Chart.js', err);
      }});
    }}
    window.bitescoreChartJsPromise.then(renderChart);
  }}

  ensureChartJs();
}})();
</script>
"""


def _format_sequence_text(seq_id: str, sequence: Optional[str], width: int = 60) -> str:
    if not sequence:
        return ""
    seq = sequence.replace("\n", "").strip()
    if not seq:
        return ""
    lines = [f">{seq_id}"]
    for start in range(0, len(seq), width):
        lines.append(seq[start : start + width])
    return "\n".join(lines)


def _blastp_link_html(seq_id: str, sequence: Optional[str]) -> str:
    if not sequence:
        return ""
    fasta_text = _format_sequence_text(seq_id, sequence)
    if not fasta_text:
        return ""
    encoded = urllib.parse.quote(fasta_text)
    url = (
        "https://blast.ncbi.nlm.nih.gov/Blast.cgi?"
        "PAGE=Proteins&PROGRAM=blastp&QUERY="
        f"{encoded}"
    )
    return (
        "<a class=\"blastp-link\" href=\"{url}\" target=\"_blank\" "
        "rel=\"noopener noreferrer\">Run BLASTp on NCBI</a>"
    ).format(url=url)


def _analysis_summary(input_type_label: str, pipeline_type: str, organisms: List[str], sequence_count: int, mode: str) -> str:
    org_text = ", ".join(organisms) if organisms else "Not specified"
    summary = [
        "## Analysis Complete ✅",
        f"- **Sequences analyzed:** {sequence_count}",
        f"- **Mode:** {'Single sequence' if mode == 'single' else 'Multiple sequences'}",
        f"- **Input type:** {input_type_label} ({pipeline_type})",
        f"- **Organisms:** {org_text}",
    ]
    if mode == "multi":
        summary.append("- Select a sequence in the ranked table to view its feature breakdown.")
    else:
        summary.append("- Detailed feature breakdown shown below.")
    return "\n".join(summary)


def chat_fn(
    message: str,
    history: Iterable[Any] | None,
    files,
    input_type,
    organism,
    options,
) -> List[dict[str, str]]:
    history = _normalize_messages(history)
    out_messages: List[dict[str, str]] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="bitescore_chat_"))
    outdir = tmpdir / "results"; outdir.mkdir(parents=True, exist_ok=True)
    input_fasta = tmpdir / "input.faa"
    if files:
        fpath = Path(files[-1])
        input_fasta.write_bytes(Path(fpath).read_bytes())
    else:
        _save_sequences_to_fasta(message or "", input_fasta)

    if not input_fasta.exists() or input_fasta.stat().st_size == 0:
        out_messages.append({"role": "assistant", "content": "Please upload a FASTA or paste sequences to run the pipeline."})
        return out_messages
    pipeline_input_type = _normalize_input_type(input_type)
    if pipeline_input_type in {"genome", "genomes", "metagenome"} and not organism:
        out_messages.append({"role": "assistant", "content": "For genome-like inputs, set **organism** to `prok` or `euk`."})
        return out_messages

    opts = {
        "no_structure": options.get("no_structure", False),
        "alphafold": options.get("alphafold", False),
        "cluster_cdhit": options.get("cluster_cdhit", False),
        "cdhit_threshold": float(options.get("cdhit_threshold", 0.95)) if options.get("cdhit_threshold") else None,
        "low_complexity": options.get("low_complexity", False),
        "interpro": options.get("interpro", False),
        "go_map": options.get("go_map") or None,
        "diamond_db": options.get("diamond_db") or None,
        "blast_db": options.get("blast_db") or None,
        "pfam_hmms": options.get("pfam_hmms") or None,
    }
    out_messages.append({"role": "assistant", "content": "Running pipeline…"})
    pipeline_outputs = _run_pipeline_helper(input_fasta, pipeline_input_type, organism or None, outdir, opts)

    ranked_df = pipeline_outputs.get("ranked")
    ranked_path = pipeline_outputs.get("ranked_path")
    if ranked_df is not None and not ranked_df.empty:
        df = ranked_df.head(10)
        table_md = "### Top ranked proteins\n\n" + df.to_markdown(index=False)
        out_messages.append({"role": "assistant", "content": table_md})
        if ranked_path:
            out_messages.append({"role": "assistant", "content": f"Full CSV: {ranked_path}"})
    else:
        out_messages.append({"role": "assistant", "content": "No results produced. Please check logs."})
    return out_messages

def build_ui():
    with gr.Blocks(title=APP_TITLE, css="""
        :root {
            --soft-blue-50: #f4f8ff;
            --soft-blue-75: #edf2ff;
            --soft-blue-100: #d6e6ff;
            --soft-blue-400: #4a90e2;
            --soft-blue-500: #377dd5;
            --slate-700: #1f2f4a;
        }
        .format-error-overlay {
            position: fixed;
            inset: 0;
            background: rgba(17, 34, 64, 0.45);
            backdrop-filter: blur(2px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 24px;
            cursor: pointer;
        }
        .format-error-modal {
            max-width: 420px;
            width: 100%;
            background: linear-gradient(135deg, #ffffff, #eff4ff);
            border-radius: 18px;
            box-shadow: 0 18px 36px rgba(23, 53, 87, 0.18);
            border: 1px solid rgba(74, 144, 226, 0.35);
            padding: 28px 30px;
            color: #1f2f4a;
            cursor: pointer;
        }
        .format-error-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #b22222;
        }
        .format-error-modal p {
            margin: 0 0 10px;
            line-height: 1.45;
        }
        .format-error-dismiss {
            font-size: 12px;
            color: #4a90e2;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 12px;
        }
        body, .gradio-container {
            background: linear-gradient(180deg, var(--soft-blue-50) 0%, #e6f0ff 100%) !important;
            color: var(--slate-700) !important;
            font-family: "Inter", "Segoe UI", sans-serif !important;
        }
        .gradio-container .gradio-container {
            background: transparent !important;
        }
        #app-header {
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.18), rgba(223, 235, 255, 0.88));
            border: 1px solid rgba(74, 144, 226, 0.35);
            border-radius: 22px;
            padding: 20px 28px;
            margin-bottom: 18px;
            box-shadow: 0 12px 32px rgba(36, 76, 125, 0.12);
        }
        #app-header h1 {
            color: #1d3557;
            letter-spacing: 0.3px;
        }
        #app-header p {
            color: #4c6992;
        }
        #page-wrapper {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(74, 144, 226, 0.2);
            border-radius: 22px;
            padding: 32px 28px;
            margin-bottom: 24px;
            box-shadow: 0 20px 48px rgba(15, 54, 96, 0.08);
        }
        #page-wrapper .gr-markdown h3,
        #page-wrapper .gr-markdown h4 {
            color: #25436b;
        }
        #page-wrapper .gr-markdown p {
            color: #365378;
        }
        .nav-row {
            display: flex !important;
            flex-direction: row !important;
            gap: 6px !important;
            align-items: center !important;
            justify-content: flex-end !important;
            flex-wrap: wrap !important;
            min-width: 0 !important;
            overflow: visible !important;
        }
        .nav-btn {
            display: flex !important;
        }
        .nav-btn > button {
            min-width: 120px !important;
            width: auto !important;
            height: 44px !important;
            font-size: 16px !important;
            border-radius: 999px !important;
            margin: 2px !important;
            padding: 10px 22px !important;
            flex-shrink: 1 !important;
            white-space: nowrap !important;
            border: 1px solid rgba(74, 144, 226, 0.45) !important;
            color: #1f2f4a !important;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(229, 240, 255, 0.92)) !important;
            box-shadow: 0 10px 22px rgba(52, 104, 170, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
        }
        .nav-btn > button:hover {
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.95), rgba(102, 163, 232, 0.95)) !important;
            color: white !important;
            box-shadow: 0 14px 28px rgba(58, 123, 213, 0.24) !important;
            transform: translateY(-1px);
        }
        .nav-btn > button:focus-visible {
            outline: 3px solid rgba(74, 144, 226, 0.45) !important;
            outline-offset: 2px;
        }
        @media (max-width: 768px) {
            .nav-btn > button {
                min-width: 100px !important;
                font-size: 14px !important;
                padding: 8px 16px !important;
            }
            .nav-row {
                gap: 3px !important;
            }
        }
        @media (max-width: 480px) {
            .nav-btn > button {
                min-width: 88px !important;
                font-size: 12px !important;
                padding: 6px 12px !important;
            }
        }
        #analysis-progress {
            margin-top: 12px;
        }
        #analysis-progress .analysis-progress__wrapper {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 18px 20px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.12), rgba(223, 235, 255, 0.9));
            border: 1px solid rgba(74, 144, 226, 0.25);
            box-shadow: 0 14px 32px rgba(39, 82, 146, 0.12);
        }
        #analysis-progress .analysis-progress__label {
            font-weight: 600;
            color: #1d3557;
            font-size: 16px;
        }
        #analysis-progress .analysis-progress__bar {
            width: 100%;
            height: 14px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.6);
            overflow: hidden;
            border: 1px solid rgba(74, 144, 226, 0.45);
        }
        #analysis-progress .analysis-progress__fill {
            height: 100%;
            background: linear-gradient(135deg, #4f8fde, #2f6fce);
            transition: width 0.3s ease;
        }
        #analysis-progress .analysis-progress__percent {
            align-self: flex-end;
            font-weight: 600;
            color: #2b4c7f;
            font-size: 14px;
        }
        #run-analysis-btn button {
            position: relative;
            background: linear-gradient(135deg, #4f8fde, #2f6fce) !important;
            border: 1px solid #2f6fce !important;
            box-shadow: 0 22px 36px rgba(47, 111, 206, 0.25);
        }
        #run-analysis-btn button:hover {
            background: linear-gradient(135deg, #2f6fce, #2559b2) !important;
            box-shadow: 0 26px 40px rgba(37, 89, 178, 0.3);
        }
        #run-analysis-btn button[aria-busy="true"] {
            padding-left: 2.5em !important;
        }
        #run-analysis-btn button[aria-busy="true"]::before {
            content: "";
            position: absolute;
            left: 1rem;
            top: 50%;
            width: 1rem;
            height: 1rem;
            margin-top: -0.5rem;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.4);
            border-top-color: rgba(255, 255, 255, 0.95);
            animation: run-analysis-spin 0.8s linear infinite;
        }
        #run-analysis-btn .nav-btn[aria-busy="true"]::before {
            left: 0.75rem;
        }
        @keyframes run-analysis-spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        .blastp-link {
            display: inline-block;
            margin-top: 8px;
            padding: 10px 16px;
            background: linear-gradient(135deg, #5c9ded, #3b7bd4);
            color: #fff !important;
            border-radius: 999px;
            text-decoration: none !important;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            box-shadow: 0 12px 28px rgba(59, 123, 212, 0.22);
        }
        .blastp-link:hover {
            background: linear-gradient(135deg, #3b7bd4, #2d67be);
            box-shadow: 0 16px 32px rgba(45, 103, 190, 0.28);
        }
        .sequence-card {
            margin-top: 12px;
            padding: 18px 20px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(236, 243, 255, 0.95), rgba(255, 255, 255, 0.95));
            border: 1px solid rgba(74, 144, 226, 0.25);
            box-shadow: 0 14px 32px rgba(39, 82, 146, 0.12);
        }
        .sequence-card h2 {
            margin-top: 0;
            margin-bottom: 16px;
            font-size: 18px;
            color: #1d3557;
        }
        .sequence-card__chart {
            position: relative;
            width: 100%;
            min-height: 280px;
            max-height: 360px;
            margin-bottom: 16px;
        }
        .sequence-card__chart canvas {
            width: 100% !important;
            height: 100% !important;
        }
        .sequence-card__metrics {
            list-style: none;
            padding: 0;
            margin: 0;
            display: grid;
            gap: 8px;
        }
        .sequence-card__metrics li {
            display: flex;
            gap: 6px;
            align-items: baseline;
            color: #365378;
        }
        #organism-list h4 {
            color: #2b4c7f;
            margin-bottom: 6px;
        }
        #organism-items span {
            box-shadow: 0 6px 16px rgba(82, 141, 214, 0.18);
        }
    """) as demo:
        # State to track current page
        current_page = gr.State("main")
        
        # Header with logo and navigation
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=1):
                gr.HTML(
                    f"""
                    <div style="display: flex; align-items: center; gap: 18px;">
                        <div style="width: 60px; height: 60px; border-radius: 18px; background: linear-gradient(135deg, rgba(74, 144, 226, 0.85), rgba(173, 209, 255, 0.9)); display: flex; align-items: center; justify-content: center; box-shadow: 0 14px 30px rgba(43, 97, 164, 0.28);">
                            <img src=\"{LOGO_SRC}\" alt=\"BiteScore Logo\" style=\"width: 40px; height: 40px; border-radius: 12px; background: rgba(255, 255, 255, 0.65); padding: 4px;\">
                        </div>
                        <div>
                            <h1 style=\"margin: 0; color: #1d3557; font-size: 30px; font-weight: 700; letter-spacing: 0.4px;\">BiteScore</h1>
                            <p style=\"margin: 6px 0 0 0; color: #4c6ea8; font-size: 15px;\">Protein Digestibility Intelligence Suite</p>
                        </div>
                    </div>
                    """
                )
            with gr.Column(scale=2):
                gr.HTML("")  # Spacer
            with gr.Column(scale=1):
                # All buttons in a single row with CSS to prevent wrapping
                with gr.Row(elem_classes="nav-row"):
                    about_btn = gr.Button("About", variant="outline", size="sm", elem_classes="nav-btn")
                    measure_btn = gr.Button("Predict", variant="outline", size="sm", elem_classes="nav-btn")

        
        # Page content container
        with gr.Row(elem_id="page-wrapper") as page_container:
            with gr.Column():
                # Main page content (now empty, will be removed)
                with gr.Row(visible=False) as main_page:
                    with gr.Column():
                        gr.Markdown("")
                        
                # About page content
                with gr.Row(visible=False) as about_page:
                    with gr.Column():
                        gr.Markdown(f"""
                        # About BiteScore Project

                        **BiteScore** is a comprehensive protein digestibility analysis to analyze and rank proteins ased on their digestibility potential.

                        ## What it is
                        
                        BiteScore screens individual proteins for nutritional quality and predicts their digestibility from sequence. It analyzes essential-amino-acid balance and enzyme-cleavage accessibility to deliver an explainable, per-protein digestibility score.
                        
                        ## Why it matters
                        
                        By pinpointing proteins with strong essential-amino-acid profiles and high predicted digestibility, BiteScore helps teams choose candidates to overexpress in a host organism for production. Single, well-characterized proteins can often follow clearer, simpler regulatory paths than complex multi-ingredient foods—accelerating R&D and time-to-market.*
                        
                        *Regulatory pathways vary by jurisdiction; this is not legal advice.

                        ## Our Team

                        ### Team Picture
                        <div style="display: flex; justify-content: center; margin: 16px 0;">
                            <img src="{TEAM_PHOTO_SRC}" alt="BiteScore team" style="width: 100%; max-width: 520px; border-radius: 18px; box-shadow: 0 18px 42px rgba(30, 64, 120, 0.16); object-fit: cover;" />
                        </div>

                        ### Team Members
                        - **Ricardo** - Computational Biology, Bioinformatics
                        - **Miranda** - Computer Science, Biochemical Engineering
                        - **Jimmy** - Biomedical Engineering, Machine Learning
                        - **Alexia** - Molecular Biology, Data Visualization
                        
                        """)
                
                # Predict page content (now includes the home page content)
                with gr.Row(visible=True) as measure_page:
                    with gr.Column():
                        gr.HTML(
                            """
                            <div style="margin-bottom: 18px; padding: 14px 18px; border-radius: 16px; background: linear-gradient(135deg, rgba(74, 144, 226, 0.12), rgba(173, 209, 255, 0.2)); border: 1px solid rgba(74, 144, 226, 0.25); color: #1d3557; box-shadow: 0 10px 24px rgba(39, 82, 146, 0.08);">
                                <strong>For a quick start, click one of the example buttons at the bottom and press Run Analysis.</strong>
                            </div>
                            """
                        )
                        # Step-by-step interface

                        # Step 1: Input Type Selection
                        gr.Markdown("### Step 1: Select Input Type")
                        input_type = gr.Dropdown(
                            choices=["Proteomic", "Metaproteomic", "Genomic", "Metagenomic"],
                            value="Proteomic",
                            label="Input Type",
                            info="Choose the type of data you're analyzing",
                            interactive=True,
                            allow_custom_value=False
                        )
                        
                        # Step 2: Organism Selection
                        gr.Markdown("### Step 2: Select Organism(s)")
                        organism_input = gr.Textbox(
                            label="Add Organism",
                            placeholder="Type organism name and press Enter (e.g., Escherichia coli, Homo sapiens)",
                            info="Press Enter to add organism to the list below",
                        )

                        def render_organism_list_html(current_organisms: List[str]) -> str:
                            if not current_organisms:
                                return """
                                <div id=\"organism-list\" style=\"margin-top: 12px; padding: 16px 18px; border-radius: 16px; border: 1px solid rgba(74, 144, 226, 0.25); background: linear-gradient(135deg, rgba(236, 243, 255, 0.95), rgba(255, 255, 255, 0.95)); box-shadow: 0 12px 28px rgba(52, 104, 170, 0.08);\">
                                    <h4>Selected Organisms:</h4>
                                    <div id=\"organism-items\" style=\"display: flex; flex-wrap: wrap; gap: 8px;\">
                                        <p style=\"color: #5b77a6; font-style: italic; margin: 0;\">No organisms selected yet</p>
                                    </div>
                                </div>
                                """
                            colors = ['#5e9ded', '#6fb1f3', '#4c88d8', '#88c0f7', '#3f75c6', '#9ccaf8', '#5ba3e3', '#79b5f0']
                            organism_tags = ""
                            for i, org in enumerate(current_organisms):
                                color = colors[i % len(colors)]
                                organism_tags += f"""
                                <span style=\"background: linear-gradient(135deg, {color}, rgba(255, 255, 255, 0.2)); color: white; padding: 6px 12px; border-radius: 999px; font-size: 12px; margin: 2px; display: inline-block; letter-spacing: 0.3px;\">
                                    {org}
                                </span>
                                """
                            return f"""
                            <div id=\"organism-list\" style=\"margin-top: 12px; padding: 16px 18px; border-radius: 16px; border: 1px solid rgba(74, 144, 226, 0.25); background: linear-gradient(135deg, rgba(236, 243, 255, 0.95), rgba(255, 255, 255, 0.95)); box-shadow: 0 12px 28px rgba(52, 104, 170, 0.08);\">
                                <h4>Selected Organisms:</h4>
                                <div id=\"organism-items\" style=\"display: flex; flex-wrap: wrap; gap: 8px;\">
                                    {organism_tags}
                                </div>
                            </div>
                            """

                        organism_list = gr.HTML(render_organism_list_html([]))

                        organism_state = gr.State([])

                        genome_notice = gr.Markdown(
                            "For genome or metagenome analyses, choose the organism type below before running the pipeline.",
                            visible=False,
                        )
                        genome_organism_selector = gr.Radio(
                            label="Organism type (required for genome/metagenome inputs)",
                            choices=["Prokaryotic", "Eukaryotic"],
                            value="Prokaryotic",
                            interactive=True,
                            visible=False,
                        )

                        add_organism_btn = gr.Button("Add Organism", variant="secondary", size="sm")

                        def add_organism(organism_name, current_organisms):
                            if organism_name and organism_name.strip():
                                organism_name = organism_name.strip()
                                if organism_name not in current_organisms:
                                    current_organisms.append(organism_name)
                                organism_html = render_organism_list_html(current_organisms)
                                return "", current_organisms, organism_html
                            return organism_name, current_organisms, gr.update()

                        def clear_all_organisms(current_organisms):
                            current_organisms.clear()
                            organism_html = render_organism_list_html(current_organisms)
                            return current_organisms, organism_html

                        clear_organisms_btn = gr.Button("Clear All Organisms", variant="secondary", size="sm")

                        genome_disabled_html = """
                        <div id=\"organism-list\" style=\"margin-top: 12px; padding: 16px 18px; border-radius: 16px; border: 1px solid rgba(74, 144, 226, 0.25); background: linear-gradient(135deg, rgba(236, 243, 255, 0.95), rgba(255, 255, 255, 0.95)); box-shadow: 0 12px 28px rgba(52, 104, 170, 0.08);\">
                            <h4>Selected Organisms:</h4>
                            <div id=\"organism-items\" style=\"display: flex; flex-wrap: wrap; gap: 8px;\">
                                <p style=\"color: #5b77a6; font-style: italic; margin: 0;\">Genome-style analyses use the selector below.</p>
                            </div>
                        </div>
                        """

                        def handle_input_type_change(selected_type, current_organisms, current_genome_choice):
                            pipeline_type = _normalize_input_type(selected_type)
                            if pipeline_type in {"genome", "genomes", "metagenome"}:
                                selected_label = current_genome_choice or "Prokaryotic"
                                return (
                                    gr.update(
                                        value="",
                                        interactive=False,
                                        info="Organism list entry is disabled for genome inputs. Use the selector below.",
                                    ),
                                    gr.update(interactive=False),
                                    gr.update(interactive=False),
                                    gr.update(value=genome_disabled_html),
                                    gr.update(visible=True),
                                    gr.update(visible=True, value=selected_label),
                                )
                            html = render_organism_list_html(current_organisms or [])
                            return (
                                gr.update(
                                    interactive=True,
                                    info="Press Enter to add organism to the list below",
                                    placeholder="Type organism name and press Enter (e.g., Escherichia coli, Homo sapiens)",
                                ),
                                gr.update(interactive=True),
                                gr.update(interactive=True),
                                gr.update(value=html),
                                gr.update(visible=False),
                                gr.update(visible=False),
                            )

                        add_organism_btn.click(
                            fn=add_organism,
                            inputs=[organism_input, organism_state],
                            outputs=[organism_input, organism_state, organism_list]
                        )

                        organism_input.submit(
                            fn=add_organism,
                            inputs=[organism_input, organism_state],
                            outputs=[organism_input, organism_state, organism_list]
                        )

                        clear_organisms_btn.click(
                            fn=clear_all_organisms,
                            inputs=[organism_state],
                            outputs=[organism_state, organism_list]
                        )

                        input_type.change(
                            fn=handle_input_type_change,
                            inputs=[input_type, organism_state, genome_organism_selector],
                            outputs=[
                                organism_input,
                                add_organism_btn,
                                clear_organisms_btn,
                                organism_list,
                                genome_notice,
                                genome_organism_selector,
                            ],
                        )

                        # Step 3: FASTA Input
                        gr.Markdown("### Step 3: Insert FASTA or Paste Sequences")
                        files = gr.File(label="Upload FASTA (optional)", file_count="multiple", type="filepath")
                        sequence_input = gr.Textbox(
                            label="Or paste sequences directly",
                            placeholder="Paste protein sequences here (FASTA format or plain text)",
                            lines=5,
                            info="You can paste sequences in FASTA format or as plain text"
                        )

                        with gr.Row():
                            single_example_btn = gr.Button("Single-sequence example", variant="secondary")
                            multi_example_btn = gr.Button("Multi-sequence example", variant="secondary")
                            genome_example_btn = gr.Button("Example bacterial genome", variant="secondary")

                        # Run Analysis Button
                        run_btn = gr.Button("Run Analysis", variant="primary", size="lg", elem_id="run-analysis-btn")

                        format_error_popup = gr.HTML(value="", visible=False, elem_id="format-error-popup")

                        # Results display area placeholder
                        gr.Markdown("Complete the steps above and click 'Run Analysis' to start the protein analysis.")

                # Results page content (static)
                with gr.Row(visible=False) as results_page:
                    with gr.Column():
                        gr.Markdown("""
                        # Analysis Results
                        
                        ## Recent Analyses
                        View and download results from your protein digestibility analyses.
                        
                        ## Results Overview
                        - **Analysis History**: Track all your previous analyses
                        
                        ## Analysis Details
                        - **Digestibility Scores**: Per-protein digestibility ratings
                        - **Ranking Information**: Top-performing proteins
                        - **Statistical Summary**: Analysis metrics and statistics
                        - **Quality Metrics**: Confidence scores and reliability indicators
                        
                        ## Data Management
                        - **Save Results**: Store analysis results for future reference
                        - **Share Analysis**: Export results for collaboration
                        - **Compare Analyses**: Compare different protein sets
                        """)

                # NEW Analysis Results page (dynamic)
                with gr.Row(visible=False) as analysis_results_page:
                    with gr.Column():
                        gr.Markdown("# Analysis Results")

                        # Back to Measure button
                        back_to_measure_btn = gr.Button("← Back to Analysis", variant="secondary", size="sm")

                        # Results display components
                        analysis_status = gr.Markdown("## Awaiting analysis")
                        analysis_progress = gr.HTML(
                            value=_analysis_progress_html(0, "Preparing analysis..."),
                            visible=False,
                            elem_id="analysis-progress",
                        )
                        with gr.Row():
                            with gr.Column(scale=2):
                                analysis_dataframe = gr.Dataframe(
                                    visible=False,
                                    interactive=False,
                                    label="Ranked Sequences",
                                )
                                download_file = gr.File(
                                    visible=False,
                                    label="Download Full Results",
                                )
                            with gr.Column(scale=1):
                                sequence_card = gr.HTML(visible=False)
                                sequence_viewer = gr.Textbox(
                                    label="Selected Sequence",
                                    interactive=False,
                                    lines=10,
                                    visible=False,
                                    show_copy_button=True,
                                )
                                blastp_link = gr.HTML(visible=False)

                        with gr.Row():
                            with gr.Column(scale=2):
                                feature_tabs = gr.Tabs(visible=False)
                                with feature_tabs:
                                    with gr.TabItem("features-aa · Amino acid composition"):
                                        aa_table = gr.Dataframe(
                                            headers=["Metric", "Value"],
                                            datatype=["str", "str"],
                                            interactive=False,
                                            visible=False,
                                        )
                                    with gr.TabItem("features-regsite · Protease recognition"):
                                        regsite_table = gr.Dataframe(
                                            headers=["Metric", "Value"],
                                            datatype=["str", "str"],
                                            interactive=False,
                                            visible=False,
                                        )
                                    with gr.TabItem("features-structure · Cleavage accessibility"):
                                        structure_table = gr.Dataframe(
                                            headers=["Metric", "Value"],
                                            datatype=["str", "str"],
                                            interactive=False,
                                            visible=False,
                                        )
                                    with gr.TabItem("features-function · Functional annotations"):
                                        function_table = gr.Dataframe(
                                            headers=["Metric", "Value"],
                                            datatype=["str", "str"],
                                            interactive=False,
                                            visible=False,
                                        )
                            with gr.Column(scale=1):
                                structure_view = gr.HTML(
                                    value=STRUCTURE_PLACEHOLDER_HTML,
                                    visible=False,
                                )

                analysis_state = gr.State({})
                selected_sequence = gr.State(None)
                
                # Account page content
                with gr.Row(visible=False) as account_page:
                    with gr.Column():
                        gr.Markdown("""
                        # Account Settings & Project Management
                        
                        ## User Profile
                        - **Username**: Current user session
                        - **Project Affiliation**: Research institution or organization
                        - **Analysis History**: View previous digestibility analyses
                        - **Saved Results**: Access downloaded reports and data
                        - **Research Interests**: Protein types and analysis preferences
                        
                        ## Project Preferences
                        - **Default Settings**: Configure default analysis parameters
                        - **Organism Preferences**: Set preferred organism types for analysis
                        - **Output Format**: Choose preferred result formats (CSV, JSON, PDF)
                        - **Notification Settings**: Email and system notifications for analysis completion
                        - **Data Retention**: Manage analysis data storage and privacy settings
                        
                        ## Research Collaboration
                        - **Project Sharing**: Share analysis results with collaborators
                        - **Team Access**: Manage team member permissions
                        - **Version Control**: Track analysis versions and changes
                        - **Export Options**: Export data for external analysis tools
                        
                        ## Support & Resources
                        - **Help Center**: Frequently asked questions about protein digestibility analysis
                        - **Tutorials**: Step-by-step guides for using BiteScore
                        - **Contact Support**: Get help with technical issues
                        - **Feature Requests**: Suggest new features for protein analysis
                        - **Research Papers**: Access to relevant scientific literature
                        
                        ## Data Management
                        - **Privacy Settings**: Control data sharing and privacy
                        - **Backup Options**: Automatic backup of analysis results
                        - **Data Export**: Export analysis data in various formats
                        - **Compliance**: Ensure data handling meets research standards
                        """)
        
        # Analysis function
        def run_analysis(
            input_type_label,
            organism_list,
            genome_choice,
            files,
            sequence_input,
            progress=gr.Progress(track_tqdm=True),
        ):
            page_updates_full = show_analysis_results_page()
            page_updates = page_updates_full[:-1]
            measure_page_updates_full = show_measure_page()
            measure_page_updates = measure_page_updates_full[:-1]

            blank_table = pd.DataFrame(columns=["Metric", "Value"])
            empty_feature_outputs = (
                gr.update(visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=STRUCTURE_PLACEHOLDER_HTML, visible=False),
            )
            (
                feature_tabs_reset,
                aa_reset,
                regsite_reset,
                structure_table_reset,
                function_reset,
                structure_view_reset,
            ) = empty_feature_outputs
            empty_sequence_updates = (
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
            )
            sequence_view_reset, blastp_reset = empty_sequence_updates
            empty_dataframe_update = gr.update(value=pd.DataFrame(), visible=False)
            empty_download_update = gr.update(value=None, visible=False)
            empty_card_update = gr.update(value="", visible=False)
            default_popup_update = (
                page_updates_full[-1]
                if len(page_updates_full) > 6
                else gr.update(value="", visible=False)
            )
            error_popup_reset = default_popup_update

            def response(
                status_update,
                progress_update,
                dataframe_update,
                download_update,
                card_update,
                sequence_update,
                blastp_update,
                feature_tabs_update,
                aa_update,
                regsite_update,
                structure_table_update,
                function_update,
                structure_view_update,
                state_value,
                selected_value,
                popup_update=error_popup_reset,
                page_updates_override=None,
            ):
                pages = page_updates_override if page_updates_override is not None else page_updates
                return (
                    *pages,
                    popup_update,
                    status_update,
                    progress_update,
                    dataframe_update,
                    download_update,
                    card_update,
                    sequence_update,
                    blastp_update,
                    feature_tabs_update,
                    aa_update,
                    regsite_update,
                    structure_table_update,
                    function_update,
                    structure_view_update,
                    state_value,
                    selected_value,
                )

            def progress_response(
                percent: float,
                desc: str,
                status_text: str = "## Preparing analysis...",
            ):
                return response(
                    gr.update(value=status_text, visible=True),
                    _analysis_progress_update(percent, desc),
                    empty_dataframe_update,
                    empty_download_update,
                    empty_card_update,
                    sequence_view_reset,
                    blastp_reset,
                    feature_tabs_reset,
                    aa_reset,
                    regsite_reset,
                    structure_table_reset,
                    function_reset,
                    structure_view_reset,
                    {},
                    None,
                    error_popup_reset,
                )

            pipeline_input_type = _normalize_input_type(input_type_label)

            try:
                progress(0.01, desc="Preparing analysis...")
                yield progress_response(5, "Preparing analysis...")

                tmpdir = Path(tempfile.mkdtemp(prefix="bitescore_analysis_"))
                outdir = tmpdir / "results"
                outdir.mkdir(parents=True, exist_ok=True)
                input_fasta = tmpdir / "input.faa"

                if files and len(files) > 0:
                    progress(0.1, desc="Reading uploaded sequences...")
                    fpath = Path(files[-1])
                    expected_extension = None
                    if pipeline_input_type == "proteome":
                        expected_extension = ".faa"
                    elif pipeline_input_type in {"genome", "genomes", "metagenome", "metagenomes"}:
                        expected_extension = ".fna"
                    if expected_extension is not None:
                        lower_name = fpath.name.lower()
                        if not lower_name.endswith(expected_extension):
                            actual_ext = None
                            if "." in lower_name:
                                actual_ext = lower_name[lower_name.index(".") :]
                            yield response(
                                gr.update(value="## Awaiting analysis", visible=False),
                                _analysis_progress_update(0, "Awaiting input", visible=False),
                                empty_dataframe_update,
                                empty_download_update,
                                empty_card_update,
                                sequence_view_reset,
                                blastp_reset,
                                feature_tabs_reset,
                                aa_reset,
                                regsite_reset,
                                structure_table_reset,
                                function_reset,
                                structure_view_reset,
                                {},
                                None,
                                popup_update=gr.update(
                                    value=_format_extension_error_popup(
                                        expected_extension,
                                        actual_ext,
                                        input_type_label,
                                    ),
                                    visible=True,
                                ),
                                page_updates_override=measure_page_updates,
                            )
                            return
                    input_fasta.write_bytes(fpath.read_bytes())
                    yield progress_response(25, "Reading uploaded sequences...")
                elif sequence_input and sequence_input.strip():
                    progress(0.1, desc="Processing pasted sequences...")
                    _save_sequences_to_fasta(sequence_input, input_fasta)
                    yield progress_response(25, "Processing pasted sequences...")
                else:
                    yield response(
                        gr.update(
                            value="## Error\nPlease upload a FASTA file or paste sequences."
                        ),
                        _analysis_progress_update(100, "Analysis cancelled", visible=False),
                        empty_dataframe_update,
                        empty_download_update,
                        empty_card_update,
                        sequence_view_reset,
                        blastp_reset,
                        feature_tabs_reset,
                        aa_reset,
                        regsite_reset,
                        structure_table_reset,
                        function_reset,
                        structure_view_reset,
                        {},
                        None,
                    )
                    return

                if not input_fasta.exists() or input_fasta.stat().st_size == 0:
                    yield response(
                        gr.update(
                            value="## Error\nNo valid sequences found. Please check your input."
                        ),
                        _analysis_progress_update(100, "Analysis cancelled", visible=False),
                        empty_dataframe_update,
                        empty_download_update,
                        empty_card_update,
                        sequence_view_reset,
                        blastp_reset,
                        feature_tabs_reset,
                        aa_reset,
                        regsite_reset,
                        structure_table_reset,
                        function_reset,
                        structure_view_reset,
                        {},
                        None,
                    )
                    return

                progress(0.2, desc="Parsing FASTA records...")
                records = list(SeqIO.parse(str(input_fasta), "fasta"))
                records, normalized = _ensure_record_ids(records)
                if normalized:
                    SeqIO.write(records, str(input_fasta), "fasta")
                if not records:
                    yield response(
                        gr.update(
                            value="## Error\nNo valid sequences found in the provided FASTA."
                        ),
                        _analysis_progress_update(100, "Analysis cancelled", visible=False),
                        empty_dataframe_update,
                        empty_download_update,
                        empty_card_update,
                        sequence_view_reset,
                        blastp_reset,
                        feature_tabs_reset,
                        aa_reset,
                        regsite_reset,
                        structure_table_reset,
                        function_reset,
                        structure_view_reset,
                        {},
                        None,
                    )
                    return

                yield progress_response(45, "Parsing FASTA records...", "## Parsing sequences...")

                organisms = list(organism_list or [])
                summary_organisms = organisms.copy()
                organism_value: Optional[str] = None
                if pipeline_input_type in {"genome", "genomes", "metagenome", "metagenomes"}:
                    normalized_choice = _normalize_genome_organism(genome_choice)
                    if not normalized_choice:
                        yield response(
                            gr.update(
                                value="## Error\nFor genome or metagenome analyses, select whether the organism is prokaryotic or eukaryotic before running."
                            ),
                            _analysis_progress_update(100, "Analysis cancelled", visible=False),
                            empty_dataframe_update,
                            empty_download_update,
                            empty_card_update,
                            sequence_view_reset,
                            blastp_reset,
                            feature_tabs_reset,
                            aa_reset,
                            regsite_reset,
                            structure_table_reset,
                            function_reset,
                            structure_view_reset,
                            {},
                            None,
                        )
                        return
                    organism_value = normalized_choice
                    summary_organisms = [_format_genome_organism(normalized_choice)]
                else:
                    organism_value = organisms[0] if organisms else None
                opts: Dict[str, Any] = {}

                yield progress_response(60, "Preparing analysis pipeline...", "## Preparing analysis...")

                pipeline_outputs = _run_pipeline_helper(
                    input_fasta, pipeline_input_type, organism_value, outdir, opts
                )
                ranked_df = pipeline_outputs.get("ranked")
                features = pipeline_outputs.get("features", {}) or {}
                ranked_path = pipeline_outputs.get("ranked_path")

                progress(0.6, desc="Analyzing sequences...")
                yield progress_response(75, "Analyzing sequences...", "## Analyzing sequences...")

                if ranked_df is None or ranked_df.empty:
                    yield response(
                        gr.update(
                            value="## Analysis Failed\nNo results were produced. Please check your input and try again."
                        ),
                        _analysis_progress_update(100, "Analysis cancelled", visible=False),
                        empty_dataframe_update,
                        empty_download_update,
                        empty_card_update,
                        sequence_view_reset,
                        blastp_reset,
                        feature_tabs_reset,
                        aa_reset,
                        regsite_reset,
                        structure_table_reset,
                        function_reset,
                        structure_view_reset,
                        {},
                        None,
                    )
                    return

                ranked_for_state = ranked_df.copy().reset_index(drop=True)
                if "Rank" not in ranked_for_state.columns:
                    ranked_for_state.insert(0, "Rank", range(1, len(ranked_for_state) + 1))

                table_df = _build_ranking_table(ranked_for_state, features)
                result_count = len(ranked_for_state)
                mode = "single" if result_count == 1 else "multi"
                if mode == "single":
                    selected_id = str(ranked_for_state.iloc[0]["id"])
                elif not table_df.empty:
                    selected_id = str(table_df.iloc[0]["Sequence ID"])
                else:
                    selected_id = str(ranked_for_state.iloc[0]["id"])

                sequences = {rec.id: str(rec.seq) for rec in records}
                sequences.update(pipeline_outputs.get("sequences", {}))

                card_html = _build_sequence_card(selected_id, ranked_for_state, features)
                summary_md = _analysis_summary(
                    input_type_label,
                    pipeline_input_type,
                    summary_organisms,
                    len(records),
                    mode,
                )

                progress(0.8, desc="Compiling results...")
                yield progress_response(90, "Compiling results...", "## Compiling results...")

                table_update = gr.update(
                    value=table_df, visible=(mode == "multi" and not table_df.empty)
                )
                if ranked_path:
                    download_update = gr.update(value=str(ranked_path), visible=True)
                else:
                    download_update = gr.update(value=None, visible=False)
                card_update = gr.update(value=card_html, visible=True)
                sequence_text = _format_sequence_text(selected_id or "", sequences.get(selected_id))
                sequence_update = gr.update(value=sequence_text, visible=bool(sequence_text))
                blastp_html = _blastp_link_html(selected_id or "", sequences.get(selected_id))
                blastp_update = gr.update(value=blastp_html, visible=bool(blastp_html))

                feature_tables, has_feature_tables = _prepare_feature_tables(features, selected_id)
                feature_tabs_update = gr.update(visible=has_feature_tables)
                aa_update = gr.update(
                    value=feature_tables["aa"],
                    visible=not feature_tables["aa"].empty,
                )
                regsite_update = gr.update(
                    value=feature_tables["regsite"],
                    visible=not feature_tables["regsite"].empty,
                )
                structure_table_update = gr.update(
                    value=feature_tables["structure"],
                    visible=not feature_tables["structure"].empty,
                )
                function_update = gr.update(
                    value=feature_tables["function"],
                    visible=not feature_tables["function"].empty,
                )
                structure_html, _ = _structure_view_content(features.get("structure"), selected_id)
                structure_view_update = gr.update(value=structure_html, visible=True)

                state_value = {
                    "mode": mode,
                    "ranked": ranked_for_state,
                    "features": features,
                    "table": table_df,
                    "selected": selected_id,
                    "sequences": sequences,
                }

                progress(1.0, desc="Analysis complete")
                yield response(
                    gr.update(value=summary_md),
                    _analysis_progress_update(100, "Analysis complete", visible=False),
                    table_update,
                    download_update,
                    card_update,
                    sequence_update,
                    blastp_update,
                    feature_tabs_update,
                    aa_update,
                    regsite_update,
                    structure_table_update,
                    function_update,
                    structure_view_update,
                    state_value,
                    selected_id,
                )
            except Exception as exc:
                yield response(
                    gr.update(
                        value=f"## Analysis Error\nAn error occurred during analysis: {exc}"
                    ),
                    _analysis_progress_update(100, "Analysis error", visible=False),
                    empty_dataframe_update,
                    empty_download_update,
                    empty_card_update,
                    sequence_view_reset,
                    blastp_reset,
                    feature_tabs_reset,
                    aa_reset,
                    regsite_reset,
                    structure_table_reset,
                    function_reset,
                    structure_view_reset,
                    {},
                    None,
                )

        def handle_sequence_selection(state, current_selected, evt: SelectData):
            blank_table = pd.DataFrame(columns=["Metric", "Value"])
            empty_feature_outputs = (
                gr.update(visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=blank_table, visible=False),
                gr.update(value=STRUCTURE_PLACEHOLDER_HTML, visible=True),
            )
            empty_sequence_updates = (
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
            )

            if not state:
                return (
                    gr.update(value="Select a sequence to view details.", visible=False),
                    *empty_sequence_updates,
                    *empty_feature_outputs,
                    current_selected,
                )
            ranked = state.get("ranked")
            features = state.get("features", {}) or {}
            table = state.get("table")
            sequences = state.get("sequences", {}) or {}
            seq_id = current_selected or state.get("selected")
            if evt is not None:
                index = getattr(evt, "index", None)
                if isinstance(index, tuple):
                    row_idx = index[0]
                elif isinstance(index, list):
                    row_idx = index[0] if index else None
                else:
                    row_idx = index
                if row_idx is not None and table is not None and 0 <= row_idx < len(table):
                    seq_column = "Sequence ID" if "Sequence ID" in table.columns else "id"
                    seq_id = str(table.iloc[row_idx][seq_column])
            if not seq_id or ranked is None:
                return (
                    gr.update(value="Select a sequence to view details.", visible=False),
                    *empty_sequence_updates,
                    *empty_feature_outputs,
                    current_selected,
                )
            state["selected"] = seq_id
            card_html = _build_sequence_card(seq_id, ranked, features)
            feature_tables, _ = _prepare_feature_tables(features, seq_id)
            structure_html, _ = _structure_view_content(features.get("structure"), seq_id)
            sequence_text = _format_sequence_text(seq_id, sequences.get(seq_id))
            sequence_update = gr.update(value=sequence_text, visible=bool(sequence_text))
            blastp_html = _blastp_link_html(seq_id, sequences.get(seq_id))
            blastp_update = gr.update(value=blastp_html, visible=bool(blastp_html))
            return (
                gr.update(value=card_html, visible=True),
                sequence_update,
                blastp_update,
                gr.update(visible=any(not tbl.empty for tbl in feature_tables.values())),
                gr.update(value=feature_tables["aa"], visible=not feature_tables["aa"].empty),
                gr.update(value=feature_tables["regsite"], visible=not feature_tables["regsite"].empty),
                gr.update(value=feature_tables["structure"], visible=not feature_tables["structure"].empty),
                gr.update(value=feature_tables["function"], visible=not feature_tables["function"].empty),
                gr.update(value=structure_html, visible=True),
                seq_id,
            )

        def load_single_example():
            return gr.update(value=None), SINGLE_SEQUENCE_EXAMPLE.strip() + "\n"

        def load_multi_example():
            return gr.update(value=None), MULTI_SEQUENCE_EXAMPLE.strip() + "\n"

        def load_bacterial_genome_example():
            if not BACTERIAL_GENOME_EXAMPLE_PATH.exists():
                return gr.update(value=None), ""
            return gr.update(value=[str(BACTERIAL_GENOME_EXAMPLE_PATH)]), ""

        # Navigation functions
        def show_about_page():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def show_measure_page():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def show_results_page():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def show_account_page():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def show_analysis_results_page():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        # Event handlers
        about_btn.click(
            fn=show_about_page,
            outputs=[main_page, about_page, measure_page, results_page, account_page, analysis_results_page, format_error_popup]
        )

        measure_btn.click(
            fn=show_measure_page,
            outputs=[main_page, about_page, measure_page, results_page, account_page, analysis_results_page, format_error_popup]
        )
        
        single_example_btn.click(
            fn=load_single_example,
            outputs=[files, sequence_input]
        )

        multi_example_btn.click(
            fn=load_multi_example,
            outputs=[files, sequence_input]
        )

        genome_example_btn.click(
            fn=load_bacterial_genome_example,
            outputs=[files, sequence_input]
        )

        run_btn.click(
            fn=run_analysis,
            inputs=[input_type, organism_state, genome_organism_selector, files, sequence_input],
            outputs=[
                main_page,
                about_page,
                measure_page,
                results_page,
                account_page,
                analysis_results_page,
                format_error_popup,
                analysis_status,
                analysis_progress,
                analysis_dataframe,
                download_file,
                sequence_card,
                sequence_viewer,
                blastp_link,
                feature_tabs,
                aa_table,
                regsite_table,
                structure_table,
                function_table,
                structure_view,
                analysis_state,
                selected_sequence,
            ],
            show_progress="full",
        )

        analysis_dataframe.select(
            fn=handle_sequence_selection,
            inputs=[analysis_state, selected_sequence],
            outputs=[
                sequence_card,
                sequence_viewer,
                blastp_link,
                feature_tabs,
                aa_table,
                regsite_table,
                structure_table,
                function_table,
                structure_view,
                selected_sequence,
            ]
        )

        back_to_measure_btn.click(
            fn=show_measure_page,
            outputs=[main_page, about_page, measure_page, results_page, account_page, analysis_results_page, format_error_popup]
        )
    
    return demo

def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860)), 
        share=True
    )

if __name__ == "__main__":
    main()
