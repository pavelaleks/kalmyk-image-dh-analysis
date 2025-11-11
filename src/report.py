"""
Generate an HTML report summarising the Kalmyk Image DH analysis outputs.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
import pandas as pd


LOGGER = logging.getLogger(__name__)
REPORT_PATH = Path("output") / "report.html"


def _render_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "<p>No data available.</p>"
    limited = df.head(max_rows)
    return limited.to_html(index=False, escape=False, classes="table table-sm")


def add_summary_block(df: pd.DataFrame) -> str:
    sem = df["semantic_label"].value_counts().to_dict()
    att = df["attitude"].value_counts().to_dict()
    total = len(df)
    html_block = f"""
    <h2>Statistical Summary</h2>
    <ul>
        <li><strong>Total contexts:</strong> {total}</li>
        <li><strong>Semantic label distribution:</strong> {sem}</li>
        <li><strong>Attitude distribution:</strong> {att}</li>
    </ul>
    """
    return html_block


def generate_report(contexts: pd.DataFrame, output_path: Path | str = REPORT_PATH) -> None:
    """
    Produce an HTML dashboard linking together the analysis artefacts.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_contexts = len(contexts)
    unique_authors = contexts["author"].nunique(dropna=True)
    time_span = (
        int(contexts["year"].min()) if pd.notnull(contexts["year"].min()) else None,
        int(contexts["year"].max()) if pd.notnull(contexts["year"].max()) else None,
    )

    summary_table = contexts[
        [
            "author",
            "year",
            "title",
            "ethnonym",
            "semantic_label",
            "semantic_label_ru",
            "attitude",
            "attitude_ru",
            "summary_en",
            "summary_ru",
        ]
    ].copy() if not contexts.empty else pd.DataFrame()

    for col in ["semantic_label_ru", "attitude_ru", "summary_ru"]:
        if col in summary_table.columns:
            summary_table[col] = (
                summary_table[col]
                .fillna("")
                .astype(str)
                .str.replace(r"\n+", "<br>", regex=True)
            )

    contexts_table = contexts[
        [
            "author",
            "year",
            "ethnonym",
            "context",
            "semantic_label",
            "attitude",
        ]
    ].copy() if not contexts.empty else pd.DataFrame()

    df_display = summary_table

    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Kalmyk Image DH Analysis — Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/mini.css/3.0.1/mini-default.min.css">
    <style>
        body {{ margin: 2rem; }}
        h1, h2, h3 {{ margin-top: 2rem; }}
        img.figure {{ max-width: 100%; height: auto; margin-bottom: 1.5rem; }}
        .table-sm td, .table-sm th {{ padding: 0.35rem; }}
        .context-block {{
            background:#f7f7f7;
            border-radius:6px;
            padding:1rem;
            margin-bottom:1rem;
        }}
    </style>
</head>
<body>
    <h1>Kalmyk Image DH Analysis</h1>
    <p>Generated on {timestamp}</p>

    <h2>Project Overview</h2>
    <p>This report documents automated processing of English-language travelogues about Siberia and Altai (1864–1919) focusing on representations of Kalmyks.</p>
    <ul>
        <li>Total contexts analysed: <strong>{total_contexts}</strong></li>
        <li>Unique authors: <strong>{unique_authors}</strong></li>
        <li>Temporal coverage: <strong>{time_span[0] or 'N/A'} – {time_span[1] or 'N/A'}</strong></li>
    </ul>
"""
    html += """
    <h2>DeepSeek Semantic Overview</h2>
    <div style="max-height:600px; overflow:auto; border:1px solid #ccc; padding:0.5rem;">
"""
    if not df_display.empty:
        html += df_display.to_html(
            classes="dataframe table table-sm",
            index=False,
            escape=False,
            justify="left",
        )
    else:
        html += "<p>No data available.</p>"
    html += "</div>"

    html += f"""
    <h2>Sample Contexts</h2>
    {_render_table(contexts_table, max_rows=15)}

    <h2>Visualisations</h2>
    <div>
        <img class="figure" src="figures/mentions_by_year.png" alt="Mentions by year">
        <img class="figure" src="figures/wordcloud.png" alt="Word cloud">
        <img class="figure" src="figures/network.png" alt="Author–Ethnonym–Place network">
    </div>

    <h3>Semantic and Sentiment Statistics</h3>
    <div>
        <img class="figure" src="figures/semantic_distribution.png" alt="Semantic distribution">
        <img class="figure" src="figures/sentiment_by_author.png" alt="Sentiment by author">
    </div>
"""

    html += add_summary_block(contexts)

    html += """
    <p>All artefacts are reproducible via the pipeline defined in <code>main.py</code>.</p>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)


__all__ = ["generate_report"]

