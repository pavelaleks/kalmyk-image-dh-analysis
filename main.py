from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

from src.deepseek_module import (
    classify_context,
    detect_sentiment,
    interpret_table,
    request_commentary,
    summarize_context,
    translate_to_russian,
)
from src.extract_contexts import extract_ethnic_contexts
from src.linguistic_analysis import analyze_contexts
from src.piro_table import make_piro_table
from src.report import generate_report
from src.utils import load_ethnonyms, load_texts
from src.visualization import create_visuals


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("kalmyk-analysis")


def clean_contexts(df):
    df["context"] = df["context"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["word_count"] = df["context"].apply(lambda x: len(x.split()))
    filtered = df[df["word_count"] >= 20].copy()
    print(
        f"🧹 Filtered contexts: {len(filtered)} of {len(df)} "
        f"(removed {len(df) - len(filtered)})"
    )
    return filtered


def normalize_label(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    for cat in ["ethnographic", "functional", "evaluative", "religious", "imperial"]:
        if cat in text_lower:
            return cat
    return "other"


def normalize_attitude(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    return text.strip().split()[0].lower()


def main() -> None:
    tqdm.pandas()

    texts = load_texts(Path("data") / "texts")
    if texts.empty:
        LOGGER.warning("No texts found in data/texts. Exiting.")
        return
    LOGGER.info("Loaded %d documents.", len(texts))

    ethnonyms = load_ethnonyms(Path("data") / "ethnonyms.txt")
    LOGGER.info("Loaded %d ethnonym variants.", len(ethnonyms))

    contexts = extract_ethnic_contexts(texts, ethnonyms)
    if contexts.empty:
        LOGGER.warning("No contexts identified. Exiting.")
        return
    LOGGER.info("Extracted %d contexts.", len(contexts))

    contexts = clean_contexts(contexts)
    if contexts.empty:
        LOGGER.warning("No contexts remain after filtering. Exiting.")
        return

    # DeepSeek-анализ
    tqdm.pandas(desc="Classifying contexts")
    contexts["semantic_label"] = contexts["context"].progress_apply(classify_context)
    tqdm.pandas(desc="Detecting sentiment")
    contexts["attitude"] = contexts["context"].progress_apply(detect_sentiment)
    tqdm.pandas(desc="Summarising contexts")
    contexts["summary_en"] = contexts["context"].progress_apply(summarize_context)

    # Переводы на русский язык
    tqdm.pandas(desc="Translating summaries (RU)")
    contexts["summary_ru"] = contexts["summary_en"].progress_apply(translate_to_russian)
    tqdm.pandas(desc="Translating sentiment labels (RU)")
    contexts["attitude_ru"] = contexts["attitude"].progress_apply(translate_to_russian)
    tqdm.pandas(desc="Translating semantic labels (RU)")
    contexts["semantic_label_ru"] = contexts["semantic_label"].progress_apply(
        translate_to_russian
    )

    contexts["semantic_label"] = contexts["semantic_label"].apply(normalize_label)
    contexts["attitude"] = contexts["attitude"].apply(normalize_attitude)

    contexts = analyze_contexts(contexts)
    make_piro_table(contexts)
    create_visuals(contexts)
    generate_report(contexts)

    # Persist enriched contexts for downstream checks.
    output_path = Path("output") / "contexts_full.csv"
    contexts.to_csv(output_path, index=False)
    LOGGER.info("Updated enriched contexts at %s", output_path)

    stats = {
        "total_contexts": len(contexts),
        "semantic_label_counts": contexts["semantic_label"].value_counts().to_dict(),
        "attitude_counts": contexts["attitude"].value_counts().to_dict(),
        "authors": contexts["author"].nunique(),
    }
    summary_path = Path("output") / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("📊 Saved summary to output/summary.json")

    summary_txt_path = Path("output") / "summary.txt"
    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("=== Summary of Kalmyk Image Analysis ===\n")
        f.write(f"Total contexts: {len(contexts)}\n")
        f.write("Semantic label distribution:\n")
        f.write(json.dumps(contexts["semantic_label"].value_counts().to_dict(), indent=2))
        f.write("\n\nAttitude distribution:\n")
        f.write(json.dumps(contexts["attitude"].value_counts().to_dict(), indent=2))
        f.write("\n\nInterpretive notes:\n")
        commentary = request_commentary(
            "Сформулируй краткое (5–6 предложений) научное резюме общих тенденций представления калмыков во всех травелогах. Пиши по-русски, придерживаясь академического стиля."
        )
        f.write(commentary)
    print("🧾 Saved detailed summary to output/summary.txt")
    print("✅ Saved full bilingual contexts to output/contexts_full.csv")


if __name__ == "__main__":
    main()