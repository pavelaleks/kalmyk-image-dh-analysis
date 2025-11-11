from __future__ import annotations

import logging
from pathlib import Path

from tqdm.auto import tqdm

from src.deepseek_module import (
    classify_context,
    detect_sentiment,
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

    contexts = analyze_contexts(contexts)
    make_piro_table(contexts)
    create_visuals(contexts)
    generate_report(contexts)

    # Persist enriched contexts for downstream checks.
    output_path = Path("output") / "contexts_full.csv"
    contexts.to_csv(output_path, index=False)
    LOGGER.info("Updated enriched contexts at %s", output_path)

    print("\n[SUMMARY]")
    print("Semantic categories:", contexts["semantic_label"].value_counts().to_dict())
    print("Sentiment distribution:", contexts["attitude"].value_counts().to_dict())
    print(f"Contexts total: {len(contexts)}")
    print("✅ Saved full bilingual contexts to output/contexts_full.csv")


if __name__ == "__main__":
    main()

