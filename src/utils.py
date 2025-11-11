"""
Utility functions for loading and preprocessing travelog texts used in the
Kalmyk Image DH Analysis project.

This module centralises text IO, cleaning, tokenisation, and helper utilities
that are shared across the processing pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from nltk import download as nltk_download
from nltk import sent_tokenize
from nltk.data import find as nltk_find


LOGGER = logging.getLogger(__name__)
DEFAULT_METADATA_PATH = Path("data") / "metadata.csv"
WHITESPACE_RE = re.compile(r"\s+")


def ensure_nltk_resources() -> None:
    """
    Make sure required NLTK resources are available at runtime.

    Currently only the Punkt sentence tokenizer is required. The download call
    is safe to re-run and will exit early if the resource already exists.
    """
    try:
        nltk_find("tokenizers/punkt")
    except LookupError:
        LOGGER.info("Downloading NLTK resource: punkt")
        nltk_download("punkt")


def clean_text(text: str) -> str:
    """
    Perform lightweight cleanup on the raw OCR text.

    The aim is to normalise whitespace while leaving the original casing and
    punctuation intact for downstream linguistic analysis.
    """
    if not isinstance(text, str):
        return ""
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _normalise_identifier(value: Optional[str]) -> str:
    """
    Collapse a string into a comparable identifier by removing punctuation and
    collapsing whitespace. Used to link metadata rows and filenames.
    """
    if not value:
        return ""
    return re.sub(r"\W+", "", value).lower()


FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "cp1251", "windows-1252", "latin-1"]


def _read_metadata_csv(metadata_path: Path) -> pd.DataFrame:
    raw_bytes = metadata_path.read_bytes()
    last_error: Optional[Exception] = None
    for encoding in FALLBACK_ENCODINGS:
        try:
            text = raw_bytes.decode(encoding)
            try:
                return pd.read_csv(StringIO(text), sep=None, engine="python")
            except Exception as exc:
                last_error = exc
                continue
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise UnicodeDecodeError(
        "metadata",
        getattr(last_error, "encoding", "utf-8"),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 0),
        f"Unable to decode metadata.csv using encodings: {', '.join(FALLBACK_ENCODINGS)}",
    )


def load_metadata(metadata_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load metadata describing the travelog texts.

    Expected columns:
        - author
        - year
        - title
        - source (e.g. bibliographic reference)
        - filename (optional) or document_id (optional)
    """
    metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = _read_metadata_csv(metadata_path)
    metadata.columns = metadata.columns.str.strip().str.lower()

    required = {"author", "year", "title"}
    missing = required.difference(set(metadata.columns))
    if missing:
        raise ValueError(
            f"Metadata file {metadata_path} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    metadata["year"] = pd.to_numeric(metadata["year"], errors="coerce")
    metadata["source"] = metadata.get("source", "")

    if "document_id" not in metadata.columns:
        if "filename" in metadata.columns:
            metadata["document_id"] = metadata["filename"].map(
                lambda value: Path(str(value)).stem
            )
        else:
            # As a fallback, construct a slug from author, year, title.
            metadata["document_id"] = metadata.apply(
                lambda row: "_".join(
                    filter(
                        None,
                        [
                            str(row.get("author", "")).strip().replace(" ", ""),
                            str(int(row["year"])) if pd.notnull(row["year"]) else "",
                            str(row.get("title", "")).strip().replace(" ", ""),
                        ],
                    )
                ),
                axis=1,
            )

    metadata["__norm_id"] = metadata["document_id"].map(_normalise_identifier)
    return metadata


def load_texts(
    text_directory: Path | str,
    metadata_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Load and clean all travelog texts from the given directory.

    Parameters
    ----------
    text_directory:
        Directory containing plaintext files (UTF-8 encoded).
    metadata_path:
        Optional path to the metadata CSV. Defaults to `data/metadata.csv`.

    Returns
    -------
    pandas.DataFrame
        One row per file with columns:
            - document_id
            - filename
            - author / year / title / source
            - text (cleaned)
            - raw_text (original)
    """
    ensure_nltk_resources()

    text_dir = Path(text_directory)
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")

    metadata: Optional[pd.DataFrame] = None
    if metadata_path is not None or DEFAULT_METADATA_PATH.exists():
        meta_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
        if meta_path.exists():
            metadata = load_metadata(meta_path)

    rows = []
    for text_path in sorted(text_dir.glob("*.txt")):
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)
        document_id = text_path.stem
        norm_id = _normalise_identifier(document_id)

        meta_row = {"author": "", "year": None, "title": "", "source": ""}
        if metadata is not None:
            match = metadata.loc[
                metadata["__norm_id"] == norm_id
            ]
            if match.empty and "filename" in metadata.columns:
                alt_norm = _normalise_identifier(text_path.name)
                match = metadata.loc[
                    metadata["filename"].map(_normalise_identifier) == alt_norm
                ]
            if not match.empty:
                meta = match.iloc[0]
                meta_row = {
                    "author": meta.get("author", ""),
                    "year": int(meta["year"]) if pd.notnull(meta["year"]) else None,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                }
        rows.append(
            {
                "document_id": document_id,
                "filename": text_path.name,
                "author": meta_row["author"],
                "year": meta_row["year"],
                "title": meta_row["title"],
                "source": meta_row["source"],
                "raw_text": raw_text,
                "text": cleaned,
            }
        )

    if not rows:
        LOGGER.warning("No .txt files found in %s", text_dir)

    df = pd.DataFrame(rows)

    if metadata is not None and not df.empty:
        matched = df["author"].astype(str).str.strip().ne("").sum()
        unmatched_docs = df.loc[df["author"].astype(str).str.strip() == "", "document_id"].tolist()
        LOGGER.info(
            "Metadata matched for %d/%d documents.",
            matched,
            len(df),
        )
        if unmatched_docs:
            LOGGER.warning(
                "Missing metadata for documents: %s",
                ", ".join(unmatched_docs[:10]) + ("..." if len(unmatched_docs) > 10 else ""),
            )

    return df


def tokenize_sentences(text: str) -> List[str]:
    """Split a cleaned text string into sentences using NLTK."""
    ensure_nltk_resources()
    return sent_tokenize(text)


def load_ethnonyms(path: Path | str) -> List[str]:
    """
    Load ethnonym variants (one per line) and normalise them to lowercase.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ethnonym list not found: {path}")
    ethnonyms = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(set(ethnonyms))


def load_stopwords(path: Path | str) -> set[str]:
    """
    Load stopwords from a newline-delimited file.
    """
    path = Path(path)
    if not path.exists():
        LOGGER.warning("Stopword file not found at %s", path)
        return set()
    words = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return words


def hash_text(text: str) -> str:
    """
    Produce a reproducible SHA256 hash for caching and provenance tracking.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "clean_text",
    "ensure_nltk_resources",
    "hash_text",
    "load_ethnonyms",
    "load_metadata",
    "load_stopwords",
    "load_texts",
    "tokenize_sentences",
]

