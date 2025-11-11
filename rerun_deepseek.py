from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.deepseek_module import (
    classify_context,
    detect_sentiment,
    summarize_context,
    translate_to_russian,
)


LOGGER = logging.getLogger("rerun-deepseek")
CONTEXTS_CANDIDATES = [
    Path("output") / "contexts_full.csv",
    Path("output") / "contexts.csv",
]


def _needs_update(value: str | float | None, force: bool) -> bool:
    if force:
        return True
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "unavailable"}


def _resolve_contexts_path() -> Path:
    for candidate in CONTEXTS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No contexts file found. Run main.py to generate output/contexts_full.csv."
    )


def _ensure_column(df: pd.DataFrame, column: str, force: bool, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column].apply(lambda v: _needs_update(v, force))
    df[column] = default
    return pd.Series(True, index=df.index)


def rerun_deepseek(force: bool = False) -> None:
    contexts_path = _resolve_contexts_path()
    contexts = pd.read_csv(contexts_path)
    if "context" not in contexts.columns:
        raise ValueError("contexts.csv does not contain a 'context' column.")

    missing_semantic = _ensure_column(contexts, "semantic_label", force)
    missing_attitude = _ensure_column(contexts, "attitude", force)
    missing_summary = _ensure_column(contexts, "summary_en", force)
    missing_semantic_ru = _ensure_column(contexts, "semantic_label_ru", force, "нет данных")
    missing_attitude_ru = _ensure_column(contexts, "attitude_ru", force, "нет данных")
    missing_summary_ru = _ensure_column(contexts, "summary_ru", force, "нет данных")

    to_update = (
        missing_semantic
        | missing_attitude
        | missing_summary
        | missing_semantic_ru
        | missing_attitude_ru
        | missing_summary_ru
    )
    total = int(to_update.sum())
    if total == 0:
        LOGGER.info("No contexts require DeepSeek updates (use --force to reprocess all).")
        return

    LOGGER.info("Updating DeepSeek annotations for %d/%d contexts.", total, len(contexts))

    for idx in tqdm(to_update[to_update].index, desc="DeepSeek rerun"):
        text = contexts.at[idx, "context"]
        if missing_semantic.loc[idx]:
            contexts.at[idx, "semantic_label"] = classify_context(text)
        if missing_attitude.loc[idx]:
            contexts.at[idx, "attitude"] = detect_sentiment(text)
        if missing_summary.loc[idx]:
            contexts.at[idx, "summary_en"] = summarize_context(text)

        if missing_semantic_ru.loc[idx] or missing_semantic.loc[idx]:
            contexts.at[idx, "semantic_label_ru"] = translate_to_russian(
                contexts.at[idx, "semantic_label"]
            )
        if missing_attitude_ru.loc[idx] or missing_attitude.loc[idx]:
            contexts.at[idx, "attitude_ru"] = translate_to_russian(contexts.at[idx, "attitude"])
        if missing_summary_ru.loc[idx] or missing_summary.loc[idx]:
            contexts.at[idx, "summary_ru"] = translate_to_russian(contexts.at[idx, "summary_en"])

    contexts.to_csv(contexts_path, index=False)
    if contexts_path != CONTEXTS_CANDIDATES[0]:
        contexts.to_csv(CONTEXTS_CANDIDATES[0], index=False)
    LOGGER.info(
        "Contexts file updated with refreshed DeepSeek annotations at %s.", contexts_path
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run DeepSeek annotations for contexts.csv."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all contexts regardless of existing values.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    rerun_deepseek(force=args.force)

