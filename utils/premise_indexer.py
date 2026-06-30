"""Sentence-level indexing utilities for the h_question pipeline."""

import logging

logger = logging.getLogger("control")


def _ensure_nltk() -> None:
    import nltk
    for resource, path in [("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                nltk.download("punkt", quiet=True)


def number_sentences(premise: str) -> tuple[list[tuple[int, str]], str]:
    """Split premise into sentences and return (indexed list, numbered string).

    indexed list  : [(0, sentence_0), (1, sentence_1), ...]
    numbered string: "[0] sentence_0\n[1] sentence_1\n..."
    """
    _ensure_nltk()
    import nltk
    sentences = nltk.sent_tokenize(premise)
    indexed = [(i, s.strip()) for i, s in enumerate(sentences) if s.strip()]
    numbered_str = "\n".join(f"[{i}] {s}" for i, s in indexed)
    return indexed, numbered_str


def pull_by_indices(
    indexed_sentences: list[tuple[int, str]],
    indices: list[int],
) -> list[str]:
    """Return sentence texts for the given indices.

    Out-of-range indices are silently dropped; duplicates are deduplicated.
    """
    idx_map = {i: s for i, s in indexed_sentences}
    seen: set[int] = set()
    result: list[str] = []
    for idx in indices:
        if idx in idx_map and idx not in seen:
            result.append(idx_map[idx])
            seen.add(idx)
    return result
