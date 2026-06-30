"""POS-based keyphrase extractor — pure Python, no LLM call."""

import logging

logger = logging.getLogger("control")

_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "must", "can",
}


def _ensure_nltk() -> None:
    import nltk
    for resource, path in [
        ("punkt_tab",                  "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                # Fallback to older resource names
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)


def extract_keyphrases(hypothesis: str) -> dict:
    """Extract noun phrases, main verbs, and numerals from hypothesis via POS tagging.

    Returns {"keyphrases": [...], "claim_from_H": str}.
    Pure Python — no LLM call. Safe to unit-test in isolation.
    """
    _ensure_nltk()
    import nltk
    from nltk import word_tokenize, pos_tag, RegexpParser

    tokens = word_tokenize(hypothesis)
    tagged = pos_tag(tokens)

    # Chunk NPs: optional DT, optional JJ*/CD, one or more NN*
    parser = RegexpParser(r"NP: {<DT>?<JJ.*|CD>*<NN.*>+}")
    tree = parser.parse(tagged)

    keyphrases: list[str] = []

    # 1. Noun phrases
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(w for w, _ in subtree.leaves())
        if phrase.lower() not in ("a", "an", "the"):
            keyphrases.append(phrase)

    # 2. Main verbs (non-auxiliary)
    for word, tag in tagged:
        if tag.startswith("VB") and word.lower() not in _AUXILIARIES:
            keyphrases.append(word)

    # 3. Standalone numerals / dates not already inside a captured NP
    np_words = {w.lower() for kp in keyphrases for w in kp.split()}
    for word, tag in tagged:
        if tag == "CD" and word.lower() not in np_words:
            keyphrases.append(word)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kp in keyphrases:
        key = kp.lower()
        if key not in seen:
            seen.add(key)
            unique.append(kp)

    return {"keyphrases": unique, "claim_from_H": hypothesis.strip()}
