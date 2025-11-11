"""
Generate an HTML report summarising the Kalmyk Image DH analysis outputs.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from src.deepseek_module import interpret_cached


LOGGER = logging.getLogger(__name__)
REPORT_PATH = Path("output") / "report.html"

pd.set_option("display.max_colwidth", 100)


def add_summary_block(df: pd.DataFrame) -> str:
    sem = df["semantic_label"].value_counts().to_dict()
    att = df["attitude"].value_counts().to_dict()
    total = len(df)
    html_block = f"""
    <h2>Статистическое резюме (Statistical Summary)</h2>
    <ul>
        <li><strong>Всего контекстов:</strong> {total}</li>
        <li><strong>Распределение образов:</strong> {sem}</li>
        <li><strong>Распределение тональностей:</strong> {att}</li>
    </ul>
    """
    return html_block


def shorten_text(text: str, limit: int = 200) -> str:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "..."
    return text


def render_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        table_html = "<p>No data available.</p>"
    else:
        table_html = df.to_html(
            classes="table table-striped table-sm",
            index=False,
            escape=False,
            justify="left",
            col_space=120,
        )
    return f"""
    <h3>{title}</h3>
    <div style="max-height:500px; overflow:auto; border:1px solid #ccc; padding:0.5rem; margin-bottom:1rem;">
        {table_html}
    </div>
    """


def interpret_visual_or_table(title: str, description_hint: str, sample_text: str = "") -> str:
    explanation = interpret_cached(title, description_hint, sample_text)
    if not explanation or explanation.lower() in {"unavailable", "error"}:
        explanation = "(Interpretation unavailable — API error.)"
    return f"<div style='font-style:italic; margin:0.5rem 0 1.5rem 0;'>{explanation}</div>"


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

    summary_table = (
        contexts[
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
        ].copy()
        if not contexts.empty
        else pd.DataFrame()
    )

    for col in ["semantic_label_ru", "attitude_ru", "summary_ru"]:
        if col in summary_table.columns:
            summary_table[col] = (
                summary_table[col]
                .fillna("")
                .astype(str)
                .str.replace(r"\n+", "<br>", regex=True)
                .apply(shorten_text)
            )

    contexts_table = (
        contexts[
            [
                "author",
                "year",
                "ethnonym",
                "context",
                "semantic_label",
                "attitude",
            ]
        ].copy()
        if not contexts.empty
        else pd.DataFrame()
    )

    df_display = summary_table

    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>Анализ образа калмыков (Kalmyk Image DH Analysis)</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/mini.css/3.0.1/mini-default.min.css">
    <style>
        body {{ margin: 2rem; }}
        h1, h2, h3 {{ margin-top: 2rem; }}
        img.figure {{ max-width: 100%; height: auto; margin-bottom: 1.5rem; }}
        .table-sm td, .table-sm th {{ padding: 0.35rem; }}
    </style>
</head>
<body>
    <h1>Анализ образа калмыков (Kalmyk Image DH Analysis)</h1>
    <p>Отчёт сгенерирован {timestamp}</p>

    <h2>Обзор проекта (Project Overview)</h2>
    <p>Документ фиксирует автоматизированный анализ англоязычных травелогов о Сибири и Алтае (1864–1919), посвящённых образу калмыков.</p>
    <ul>
        <li>Количество проанализированных контекстов: <strong>{total_contexts}</strong></li>
        <li>Число уникальных авторов: <strong>{unique_authors}</strong></li>
        <li>Хронологическое покрытие: <strong>{time_span[0] or 'N/A'} – {time_span[1] or 'N/A'}</strong></li>
    </ul>
"""

    html += render_table(df_display, "Сводная таблица аннотаций (DeepSeek Semantic Overview)")
    html += interpret_visual_or_table(
        "Сводная таблица аннотаций (DeepSeek Semantic Overview)",
        "Содержит семантические метки, оценки тональности и двуязычные резюме для каждого контекста.",
        df_display.head(10).to_string(),
    )

    html += render_table(contexts_table.head(50), "Примеры контекстов (Sample Contexts)")
    html += interpret_visual_or_table(
        "Примеры контекстов (Sample Contexts)",
        "Показывает характерные выдержки, где упоминаются калмыки, вместе с назначенными категориями и тональностью.",
        contexts_table.head(10).to_string(),
    )

    html += """
    <h2>Визуализации (Visualisations)</h2>
    <h3>Частота упоминаний по годам (Mentions by Year)</h3>
    <div>
        <img class="figure" src="figures/mentions_by_year.png" alt="Mentions by year">
    </div>
"""
    html += interpret_visual_or_table(
        "Частота упоминаний по годам (Mentions by Year)",
        "Отражает, в какие годы корпус фиксирует наибольшее количество ссылок на калмыков.",
    )

    html += """
    <h3>Облако слов (Word Cloud)</h3>
    <div>
        <img class="figure" src="figures/wordcloud.png" alt="Word cloud">
    </div>
"""
    html += interpret_visual_or_table(
        "Облако слов (Word Cloud)",
        "Визуализирует наиболее частотные лексемы в контекстах о калмыках, выделяя доминирующие темы и эпитеты.",
    )

    html += """
    <h3>Сеть автор – этноним – топоним (Author–Ethnonym–Place Network)</h3>
    <div>
        <img class="figure" src="figures/network.png" alt="Author–Ethnonym–Place network">
    </div>
"""
    html += interpret_visual_or_table(
        "Сеть автор – этноним – топоним (Author–Ethnonym–Place Network)",
        "Демонстрирует, какие авторы связывают калмыков с определёнными топонимами, показывая географическое воображение.",
    )

    html += """
    <h3>Распределение типов образов (Distribution of Kalmyk Image Types)</h3>
    <div>
        <img class="figure" src="figures/semantic_distribution.png" alt="Semantic distribution">
    </div>
"""
    html += interpret_visual_or_table(
        "Распределение типов образов (Distribution of Kalmyk Image Types)",
        "Показывает, какие смысловые категории (этнографическая, функциональная, оценочная и др.) доминируют в корпусе.",
    )

    html += """
    <h3>Тональность по авторам (Sentiment by Author)</h3>
    <div>
        <img class="figure" src="figures/sentiment_by_author.png" alt="Sentiment by author">
    </div>
"""
    html += interpret_visual_or_table(
        "Тональность по авторам (Sentiment by Author)",
        "Отражает пропорции позитивных, нейтральных и негативных описаний у каждого автора.",
    )

    html += add_summary_block(contexts)
    html += interpret_visual_or_table(
        "Статистическое резюме (Statistical Summary)",
        "Кратко объясняет, какие образы и тональности преобладают и как это соотносится с британской ориенталистской традицией.",
        contexts.head(10).to_string(),
    )

    html += """
    <p>Все артефакты воспроизводимы и описаны в сценарии <code>main.py</code>.</p>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    LOGGER.info("Report written to %s", output_path)


__all__ = ["generate_report"]

