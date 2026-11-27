"""Strict local CSV loading and deterministic text cleaning."""
import csv
import hashlib
import html
import re
from pathlib import Path


def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text))).strip()


def load_reviews(path, text_column=None, label_column=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset missing: {path}. Add the instructor's imdb_top_500.csv or imdb_balanced_10k.csv to data/.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        by_lower = {c.lower().strip(): c for c in columns}
        def choose(explicit, candidates):
            if explicit:
                if explicit not in columns:
                    raise ValueError(f"Column {explicit!r} absent. Available columns: {columns}")
                return explicit
            matches = [by_lower[c] for c in candidates if c in by_lower]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify one column from {candidates}. Available: {columns}. Specify --text-column / --label-column.")
            return matches[0]
        text_column = choose(text_column, ["review", "text", "content"])
        label_column = choose(label_column, ["sentiment", "label", "target"])
        labels = {"positive": 1, "pos": 1, "1": 1, "negative": 0, "neg": 0, "0": 0}
        unique = {}
        duplicates = 0
        for row_number, row in enumerate(reader, 2):
            text = clean_text(row.get(text_column) or "")
            raw = (row.get(label_column) or "").lower().strip()
            if not text or raw not in labels:
                raise ValueError(f"Invalid review or binary label on CSV row {row_number}; accepted labels: positive/negative or 1/0.")
            key = text.casefold()
            if key in unique:
                if unique[key][1] != labels[raw]:
                    raise ValueError(f"Conflicting labels for duplicate review on CSV row {row_number}.")
                duplicates += 1
            else:
                unique[key] = (text, labels[raw])
    rows = list(unique.values())
    counts = {str(k): sum(y == k for _, y in rows) for k in [0, 1]}
    if min(counts.values()) < 5:
        raise ValueError(f"At least 5 unique reviews per class are required for stratified train/validation/test splits; found {counts}.")
    return [t for t, _ in rows], [y for _, y in rows], {
        "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "text_column": text_column, "label_column": label_column,
        "unique_reviews": len(rows), "duplicates_removed": duplicates, "class_counts": counts,
    }
