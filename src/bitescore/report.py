from pathlib import Path
import pandas as pd

def make_report(outdir: Path):
    feats = Path(outdir) / "features.csv"
    ranked = Path(outdir) / "ranked.csv"
    html = Path(outdir) / "report.html"
    parts = []
    if feats.exists():
        df = pd.read_csv(feats).head(50)
        parts.append("<h2>Feature preview (top 50)</h2>" + df.to_html(index=False))
    if ranked.exists():
        df2 = pd.read_csv(ranked).head(50)
        parts.append("<h2>Top 50 ranked proteins</h2>" + df2.to_html(index=False))
    html.write_text(f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>bitescore report</title>
<style>body{{font-family:Arial, sans-serif; margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd; padding:4px}}</style>
</head><body>
<h1>bitescore report</h1>
{''.join(parts) if parts else '<p>No outputs found.</p>'}
</body></html>
""")
    return html
