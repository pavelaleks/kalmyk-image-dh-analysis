"""
Generate PIRO (Place, Identity, Representation, Otherness) annotation tables
from enriched context data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


LOGGER = logging.getLogger(__name__)
PIRO_OUTPUT_PATH = Path("output") / "piro_table.xlsx"


def _stringify_list(values: Any, separator: str = "; ") -> str:
    if isinstance(values, (list, tuple, set)):
        return separator.join(str(item) for item in values if item)
    return str(values) if values not in (None, float("nan")) else ""


def make_piro_table(
    contexts: pd.DataFrame,
    output_path: Path | str = PIRO_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Construct a PIRO table combining metadata and DeepSeek annotations.

    Parameters
    ----------
    contexts:
        DataFrame returned from `extract_ethnic_contexts`, augmented with the
        DeepSeek columns (semantic_label, attitude, summary_en) and linguistic
        annotations (toponyms, etc.).
    output_path:
        Destination Excel file path.
    """
    if contexts.empty:
        LOGGER.warning("Empty contexts dataframe; PIRO table will not be created.")
        return contexts

    records: List[Dict[str, Any]] = []
    for row in contexts.to_dict(orient="records"):
        records.append(
            {
                "context_id": row.get("context_id"),
                "author": row.get("author"),
                "year": row.get("year"),
                "title": row.get("title"),
                "source": row.get("source"),
                "ethnonym": row.get("ethnonym"),
                "Place": _stringify_list(row.get("toponyms", [])),
                "Identity": row.get("ethnonym_normalised") or row.get("ethnonym"),
                "Representation": row.get("semantic_label", ""),
                "Representation_ru": row.get("semantic_label_ru", ""),
                "Otherness": row.get("attitude", ""),
                "Otherness_ru": row.get("attitude_ru", ""),
                "Summary_en": row.get("summary_en", ""),
                "Summary_ru": row.get("summary_ru", ""),
                "Context": row.get("context", ""),
            }
        )

    table = pd.DataFrame.from_records(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(output_path, index=False)

    LOGGER.info("PIRO table saved to %s (%d rows)", output_path, len(table))
    return table


__all__ = ["make_piro_table"]

