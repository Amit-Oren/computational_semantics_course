"""
Dependency-SVO seeder — extracts subject-verb-object triples via spaCy.

Walks the dependency tree: for each ROOT verb, collects nsubj/nsubjpass
and dobj/pobj/attr children with their full subtrees, and formats each
triple as a natural-language phrase.

Academic grounding: DAE — Goyal & Durrett (2020-2021), Dependency Arc
Entailment; relation-level verification catches errors that sentence-level
checks miss.

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging

logger = logging.getLogger("control")

_SKIP_DEPS = {"det", "punct", "cc"}

_NLP = None  # loaded once on first use


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
        except ImportError:
            raise ImportError(
                "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"
            )
    return _NLP


def _subtree_text(token) -> str:
    """Reconstruct the full text span for a token's subtree, skipping punctuation."""
    return " ".join(t.text for t in token.subtree if t.dep_ not in _SKIP_DEPS).strip()


class SVOSeeder:
    name = "svo"

    def __init__(self, model=None, params=None):
        pass  # no LLM needed

    def seed(self, text: str) -> list[str]:
        """
        Return SVO triples as natural-language phrases.

        Each output string is one of:
          "{subject} {verb} {object}"   — transitive triple
          "{subject} {verb}"            — intransitive (no direct object found)

        Runs once per sentence; collects the ROOT verb's primary
        subject and object. Multiple subjects or objects in the same
        sentence produce separate triples.
        """
        try:
            nlp = _get_nlp()
        except ImportError as exc:
            logger.warning(f"SVOSeeder unavailable: {exc}")
            return []

        doc = nlp(text)
        seeds: list[str] = []
        seen: set[str] = set()

        for sent in doc.sents:
            for token in sent:
                if token.dep_ != "ROOT" or token.pos_ not in ("VERB", "AUX"):
                    continue

                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects  = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr", "oprd")]

                verb = token.lemma_

                for subj in subjects or [None]:
                    s_text = _subtree_text(subj) if subj else ""
                    for obj in objects or [None]:
                        o_text = _subtree_text(obj) if obj else ""
                        if s_text and o_text:
                            triple = f"{s_text} {verb} {o_text}"
                        elif s_text:
                            triple = f"{s_text} {verb}"
                        elif o_text:
                            triple = f"{verb} {o_text}"
                        else:
                            continue
                        key = triple.lower()
                        if key not in seen:
                            seen.add(key)
                            seeds.append(triple)

        return seeds
