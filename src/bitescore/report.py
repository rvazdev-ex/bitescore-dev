from pathlib import Path
import pandas as pd
import numpy as np


def _structural_context_section(outdir: Path) -> str:
    """Generate an HTML section summarising structure-aware proxy features."""
    struct_path = outdir / "features_structure.csv"
    if not struct_path.exists():
        return ""

    df = pd.read_csv(struct_path)
    if df.empty:
        return ""

    parts = ['<h2>Structural Context Summary</h2>']
    parts.append('<p>Structure-aware proxy features computed for every sequence, '
                 'with AlphaFold summary statistics where available.</p>')

    # Summary statistics table
    proxy_cols = [
        ("disorder_propensity_mean", "Disorder Propensity (mean)"),
        ("disorder_prone_frac", "Disorder-Prone Fraction"),
        ("hydrophobicity_mean", "Hydrophobicity (mean)"),
        ("surface_accessibility_proxy", "Surface Accessibility Proxy"),
        ("helix_propensity_mean", "Helix Propensity (mean)"),
        ("sheet_propensity_mean", "Sheet Propensity (mean)"),
        ("sequence_complexity", "Sequence Complexity"),
    ]

    rows = []
    for col, label in proxy_cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                rows.append(
                    f"<tr><td>{label}</td>"
                    f"<td>{vals.mean():.3f}</td>"
                    f"<td>{vals.std():.3f}</td>"
                    f"<td>{vals.min():.3f}</td>"
                    f"<td>{vals.max():.3f}</td></tr>"
                )

    if rows:
        parts.append(
            '<table><thead><tr><th>Proxy Feature</th><th>Mean</th><th>Std</th>'
            '<th>Min</th><th>Max</th></tr></thead><tbody>'
        )
        parts.extend(rows)
        parts.append('</tbody></table>')

    # AlphaFold / pLDDT statistics (only when structures were resolved)
    plddt_col = "plddt_mean"
    if plddt_col in df.columns:
        resolved = df[df[plddt_col].notna() & (df.get("plddt_n_residues", pd.Series(dtype=float)) > 0)]
        if not resolved.empty:
            parts.append('<h3>AlphaFold / Structure Confidence (pLDDT)</h3>')
            parts.append(f'<p>{len(resolved)} of {len(df)} sequences have resolved structures.</p>')

            plddt_stats = [
                ("plddt_mean", "Mean pLDDT"),
                ("plddt_frac_disordered", "Fraction Disordered (pLDDT < 50)"),
                ("plddt_frac_confident", "Fraction Confident (70 ≤ pLDDT < 90)"),
                ("plddt_frac_very_high", "Fraction Very High (pLDDT ≥ 90)"),
                ("radius_of_gyration", "Radius of Gyration (Å)"),
                ("contact_density", "Contact Density"),
            ]

            stat_rows = []
            for col, label in plddt_stats:
                if col in resolved.columns:
                    vals = resolved[col].dropna()
                    if len(vals) > 0:
                        stat_rows.append(
                            f"<tr><td>{label}</td>"
                            f"<td>{vals.mean():.3f}</td>"
                            f"<td>{vals.std():.3f}</td></tr>"
                        )

            if stat_rows:
                parts.append(
                    '<table><thead><tr><th>Metric</th><th>Mean</th><th>Std</th>'
                    '</tr></thead><tbody>'
                )
                parts.extend(stat_rows)
                parts.append('</tbody></table>')

    # Structure source breakdown
    if "structure_source" in df.columns:
        counts = df["structure_source"].value_counts()
        parts.append('<h3>Structure Sources</h3><ul>')
        for src, cnt in counts.items():
            parts.append(f'<li><strong>{src}</strong>: {cnt} sequences</li>')
        parts.append('</ul>')

    return "\n".join(parts)


def make_report(outdir: Path):
    feats = Path(outdir) / "features.csv"
    ranked = Path(outdir) / "ranked.csv"
    html = Path(outdir) / "report.html"
    parts = []
    if feats.exists():
        df = pd.read_csv(feats).head(50)
        parts.append("<h2>Feature preview (top 50)</h2>" + df.to_html(index=False))

    structural_section = _structural_context_section(outdir)
    if structural_section:
        parts.append(structural_section)

    if ranked.exists():
        df2 = pd.read_csv(ranked).head(50)
        parts.append("<h2>Top 50 ranked proteins</h2>" + df2.to_html(index=False))
    html.write_text(f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>bitescore report</title>
<style>body{{font-family:Arial, sans-serif; margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd; padding:4px}} h3{{margin-top:1.5em}}</style>
</head><body>
<h1>bitescore report</h1>
{''.join(parts) if parts else '<p>No outputs found.</p>'}
</body></html>
""")
    return html
