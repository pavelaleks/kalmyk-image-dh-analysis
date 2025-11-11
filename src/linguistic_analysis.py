"""
Linguistic analysis routines powered by spaCy for part-of-speech tagging,
lemma extraction, and toponym identification.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm

from src.utils import hash_text, load_stopwords


LOGGER = logging.getLogger(__name__)
DEFAULT_STOPWORDS_PATH = Path("data") / "stopwords_en.txt"
COLLOCATIONS_OUTPUT_PATH = Path("output") / "collocations.csv"
SPACY_MODEL = "en_core_web_sm"

_NLP: Language | None = None


def _get_nlp() -> Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(SPACY_MODEL)
        except OSError as exc:  # pragma: no cover
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL}' is not installed. "
                "Install it via `python -m spacy download en_core_web_sm`."
            ) from exc
    return _NLP


def _normalise_context_id(context_row: pd.Series) -> str:
    if "context_id" in context_row and isinstance(context_row["context_id"], str):
        return context_row["context_id"]
    return hash_text(context_row.get("context", ""))


def analyze_contexts(
    contexts: pd.DataFrame,
    stopwords_path: Path | str = DEFAULT_STOPWORDS_PATH,
    collocations_output: Path | str = COLLOCATIONS_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Annotate contexts with linguistic features and export collocation statistics.

    This function mutates the provided DataFrame by adding:
        - adjectives: ordered list of adjective lemmas and frequencies
        - verbs: ordered list of verb lemmas and frequencies
        - toponyms: list of distinct GPE entities detected by spaCy
    """
    if contexts.empty:
        LOGGER.warning("Empty contexts dataframe supplied to analyze_contexts.")
        return contexts

    nlp = _get_nlp()
    stopwords = load_stopwords(stopwords_path)

    collocation_records: List[Dict[str, object]] = []
    adjective_map: Dict[str, List[tuple[str, int]]] = {}
    verb_map: Dict[str, List[tuple[str, int]]] = {}
    toponym_map: Dict[str, List[str]] = {}

    iterable = tqdm(
        contexts.to_dict(orient="records"),
        desc="Running spaCy analysis",
        total=len(contexts),
    )

    for row in iterable:
        context_id = row.get("context_id") or hash_text(row.get("context", ""))
        doc = nlp(row.get("context", ""))

        adj_counter: Counter[str] = Counter()
        verb_counter: Counter[str] = Counter()
        gpe_entities: List[str] = []

        ethnonym = str(row.get("ethnonym_normalised") or row.get("ethnonym", "")).lower()

        for token in doc:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower()
            if not lemma or lemma in stopwords:
                continue
            if token.pos_ == "ADJ":
                adj_counter[lemma] += 1
                collocation_records.append(
                    {
                        "context_id": context_id,
                        "lemma": lemma,
                        "pos": "ADJ",
                        "ethnonym": ethnonym,
                        "author": row.get("author", ""),
                        "year": row.get("year"),
                        "count": 1,
                    }
                )
            elif token.pos_ == "VERB":
                verb_counter[lemma] += 1
                collocation_records.append(
                    {
                        "context_id": context_id,
                        "lemma": lemma,
                        "pos": "VERB",
                        "ethnonym": ethnonym,
                        "author": row.get("author", ""),
                        "year": row.get("year"),
                        "count": 1,
                    }
                )

        for ent in doc.ents:
            if ent.label_ == "GPE":
                gpe_entities.append(ent.text)

        adjective_map[context_id] = adj_counter.most_common()
        verb_map[context_id] = verb_counter.most_common()
        toponym_map[context_id] = sorted(set(gpe_entities))

    collocations_output = Path(collocations_output)
    collocations_output.parent.mkdir(parents=True, exist_ok=True)

    if collocation_records:
        collocations_df = (
            pd.DataFrame(collocation_records)
            .groupby(["context_id", "lemma", "pos", "ethnonym", "author", "year"])
            .agg({"count": "sum"})
            .reset_index()
            .sort_values(["year", "author", "ethnonym", "pos", "count"], ascending=[True, True, True, True, False])
        )
        collocations_df.to_csv(collocations_output, index=False)
        LOGGER.info(
            "Saved collocation statistics to %s (%d rows)",
            collocations_output,
            len(collocations_df),
        )
    else:
        LOGGER.info("No collocations identified; skipping CSV export.")

    contexts.loc[:, "context_id"] = contexts.apply(_normalise_context_id, axis=1)
    contexts.loc[:, "adjectives"] = contexts["context_id"].map(
        lambda cid: adjective_map.get(cid, [])
    )
    contexts.loc[:, "verbs"] = contexts["context_id"].map(
        lambda cid: verb_map.get(cid, [])
    )
    contexts.loc[:, "toponyms"] = contexts["context_id"].map(
        lambda cid: toponym_map.get(cid, [])
    )

    return contexts


__all__ = ["analyze_contexts"]

