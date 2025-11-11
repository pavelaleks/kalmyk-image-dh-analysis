"""
Utilities for locating ethnonym mentions in the corpus and exporting their
surrounding contexts for downstream analysis.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from src.utils import hash_text, tokenize_sentences


LOGGER = logging.getLogger(__name__)
CONTEXT_WINDOW = 3
CONTEXT_OUTPUT_PATH = Path("output") / "contexts.csv"


def _compile_ethnonym_pattern(ethnonyms: Iterable[str]) -> re.Pattern[str]:
    escaped = [re.escape(term) for term in ethnonyms]
    group = "|".join(escaped)
    return re.compile(rf"\b({group})\b", re.IGNORECASE)


def extract_ethnic_contexts(
    texts: pd.DataFrame,
    ethnonyms: List[str],
    window: int = CONTEXT_WINDOW,
    output_path: Path | str = CONTEXT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Identify occurrences of Kalmyk ethnonyms and extract +/- `window` sentences.

    Parameters
    ----------
    texts:
        DataFrame from `load_texts` with at least columns:
        document_id, text, author, year, title, source, filename.
    ethnonyms:
        Iterable of ethnonym spellings (lowercase recommended).
    window:
        Number of sentences before and after the target sentence to include.
    output_path:
        Path where the aggregated contexts CSV will be written.
    """
    if texts.empty:
        LOGGER.warning("No texts provided for context extraction.")
        return pd.DataFrame()

    pattern = _compile_ethnonym_pattern(ethnonyms)
    records = []

    progress = tqdm(
        texts.itertuples(index=False),
        total=len(texts),
        desc="Extracting ethnonym contexts",
    )

    for doc in progress:
        sentences = tokenize_sentences(doc.text)
        occurrence_counter = 0
        for sentence_idx, sentence in enumerate(sentences):
            matches = list(pattern.finditer(sentence))
            if not matches:
                continue
            for match in matches:
                occurrence_counter += 1
                start = max(sentence_idx - window, 0)
                stop = min(sentence_idx + window + 1, len(sentences))
                context_sentences = sentences[start:stop]
                pre_context = " ".join(sentences[start:sentence_idx])
                post_context = " ".join(sentences[sentence_idx + 1:stop])

                context_text = " ".join(context_sentences)
                context_hash = hash_text(
                    f"{doc.document_id}|{sentence_idx}|{match.group(0)}|{context_text}"
                )

                records.append(
                    {
                        "context_id": context_hash,
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "author": doc.author,
                        "year": doc.year,
                        "title": doc.title,
                        "source": doc.source,
                        "ethnonym": match.group(0),
                        "ethnonym_normalised": match.group(1).lower(),
                        "sentence_index": sentence_idx,
                        "occurrence_index": occurrence_counter,
                        "target_sentence": sentence,
                        "context": context_text,
                        "pre_context": pre_context,
                        "post_context": post_context,
                        "context_sentence_count": len(context_sentences),
                    }
                )

    contexts = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(output_path, index=False)
    LOGGER.info("Saved %d contexts to %s", len(contexts), output_path)
    return contexts


__all__ = ["extract_ethnic_contexts"]

