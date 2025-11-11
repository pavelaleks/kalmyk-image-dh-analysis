from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("metadata-generator")

TEXTS_DIR = Path("data") / "texts"
METADATA_PATH = Path("data") / "metadata.csv"
BACKUP_SUFFIX = ".bak"

FILENAME_PATTERN = re.compile(r"^([A-Za-z0-9'’\-\.]+)_(\d{4})_(.+)$")


@dataclass
class MetadataEntry:
    author: str
    year: str
    title: str
    source: str = ""

    def to_csv_row(self) -> List[str]:
        return [self.author, self.year, self.title, self.source]


def parse_filename(stem: str) -> MetadataEntry | None:
    match = FILENAME_PATTERN.match(stem)
    if not match:
        LOGGER.warning("Unable to parse filename: %s", stem)
        return None
    author, year, title = match.groups()
    title = title.replace("_", " ").strip()
    author = author.replace("_", " ").strip()
    return MetadataEntry(author=author, year=year, title=title)


def iter_text_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Text directory not found: {directory}")
    yield from sorted(directory.glob("*.txt"))


def backup_metadata(path: Path) -> None:
    if path.exists():
        backup_path = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        backup_path.write_bytes(path.read_bytes())
        LOGGER.info("Existing metadata backed up to %s", backup_path)


def write_metadata(entries: List[MetadataEntry], path: Path) -> None:
    backup_metadata(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["author", "year", "title", "source"])
        for entry in entries:
            writer.writerow(entry.to_csv_row())
    LOGGER.info("Metadata written to %s (%d rows)", path, len(entries))


def main() -> None:
    entries: List[MetadataEntry] = []
    for file_path in iter_text_files(TEXTS_DIR):
        entry = parse_filename(file_path.stem)
        if entry:
            entries.append(entry)
    if not entries:
        LOGGER.warning("No metadata entries generated. Check filenames in %s", TEXTS_DIR)
        return
    write_metadata(entries, METADATA_PATH)


if __name__ == "__main__":
    main()

