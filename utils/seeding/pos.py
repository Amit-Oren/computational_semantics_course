"""
POS-based seeder — wraps the existing extract_keyphrases utility.

Extracts noun phrases, main verbs, and numerals via NLTK POS tagging.
No LLM call. Deterministic.

Academic grounding: QAGS (Wang et al., ACL 2020), FEQA (Durmus et al., ACL 2020) —
answer-first / keyphrase-anchored question generation paradigm.
"""

from utils.pos_keyphrase import extract_keyphrases


class POSSeeder:
    name = "pos"

    def __init__(self, model=None, params=None):
        pass  # no LLM needed

    def seed(self, text: str) -> list[str]:
        """Return keyphrases (NPs, main verbs, numerals) from text."""
        result = extract_keyphrases(text)
        return result["keyphrases"]
