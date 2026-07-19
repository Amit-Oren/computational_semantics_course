"""
SRL-based seeder — extracts predicate-argument structures via LLM.

Prompts the lab LLM to identify PropBank-style semantic roles for each
predicate in the text, returning one seed phrase per role-argument pair.

Academic grounding:
  QA-SRL — FitzGerald et al. (2018)
  "Asking It All: Generating Contextualized Questions for any Semantic Role"
  — Pyatkin et al. (2021)

LLM-based SRL extraction is equivalent in output to classical SRL models
(BERT fine-tuned on CoNLL-2012/PropBank) for the purpose of generating
seed phrases — the key property is role coverage, not the extraction method.

Uses the existing lab LLM (no additional model downloads required).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

from config.config import get_structured_llm, DEFAULT_PARAMS
from utils.retry import call_with_retry

logger = logging.getLogger("control")

_SRL_SYSTEM = """\
You are a semantic role labeler. Given a text, extract predicate-argument structures.

For each main verb (predicate) in the text, extract its semantic roles following PropBank conventions:
- ARG0 (agent): who performs the action
- ARG1 (theme/patient): what is acted upon
- ARG2 (recipient/destination): to whom / to where
- ARGM-TMP (time): when
- ARGM-LOC (location): where
- ARGM-MNR (manner): how
- ARGM-EXT (extent): by how much
- ARGM-CAU (cause): why/because
- ARGM-NEG (negation): if negated

Only include roles that are explicitly present in the text. Return exact spans from the text.

Output JSON only, no extra text. Use EXACTLY this format — one object per verb:
{"roles": [{"predicate": "reduced", "ARG0": "The government", "ARG1": "corporate taxes", "ARGM-TMP": "in 2019"}, {"predicate": "fell", "ARG1": "prices", "ARGM-LOC": "in Europe"}]}\
"""

_SRL_USER = """\
Extract predicate-argument structures from this text.

Text: {text}

Output JSON only, no extra text.\
"""

_ROLE_READABLE: dict[str, str] = {
    "ARG0":     "agent",
    "ARG1":     "theme",
    "ARG2":     "recipient",
    "ARG3":     "starting point",
    "ARG4":     "ending point",
    "ARGM-TMP": "time",
    "ARGM-LOC": "location",
    "ARGM-MNR": "manner",
    "ARGM-EXT": "extent",
    "ARGM-CAU": "cause",
    "ARGM-PRP": "purpose",
    "ARGM-NEG": "negation",
    "ARGM-MOD": "modal",
    "ARGM-DIR": "direction",
}

_ROLE_KEYS = frozenset(_ROLE_READABLE) | {"ARG3", "ARG4"}


class _SRLOutput(BaseModel):
    """Flexible schema that accepts several formats the model may return.

    The model inconsistently returns:
      Format A (flat per verb):   {"predicate": "verb", "ARG0": "span", "ARG1": "span"}
      Format B (nested roles):    {"predicate": "verb", "roles": {"ARG0": "span"}}
      Format C (old role-centric): {"predicate": "verb", "role": "ARG0", "span": "span"}

    We accept all three via a list of raw dicts and normalise in seed().
    """
    roles: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_list_at_root(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"roles": data}
        return data


def _extract_seeds(entries: list[dict]) -> list[str]:
    """Parse verb-centric dicts into seed strings, handling all model formats."""
    seeds: list[str] = []
    seen: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        verb = entry.get("predicate") or entry.get("verb") or "unknown"

        # Format C: role-centric {"predicate", "role", "span"}
        if "role" in entry and "span" in entry:
            role = str(entry["role"]).upper()
            span = str(entry["span"]).strip()
            readable = _ROLE_READABLE.get(role, role.lower())
            seed = f"{readable}: {span}  [predicate: {verb}]"
            if seed.lower() not in seen:
                seen.add(seed.lower())
                seeds.append(seed)
            continue

        # Format B: nested {"predicate", "roles": {"ARG0": "span", ...}}
        nested = entry.get("roles")
        if isinstance(nested, dict):
            role_items = nested.items()
        else:
            # Format A: flat {"predicate", "ARG0": "span", "ARG1": "span", ...}
            role_items = [
                (k, v) for k, v in entry.items()
                if k not in ("verb", "predicate") and (
                    k.upper() in _ROLE_KEYS or k.upper().startswith("ARG")
                )
            ]

        for role, span in role_items:
            if not span or not isinstance(span, str):
                continue
            role = role.upper()
            readable = _ROLE_READABLE.get(role, role.lower())
            seed = f"{readable}: {span.strip()}  [predicate: {verb}]"
            if seed.lower() not in seen:
                seen.add(seed.lower())
                seeds.append(seed)

    return seeds


class SRLSeeder:
    name = "srl"

    def __init__(self, model: Optional[str] = None, params: Optional[dict] = None):
        self.model  = model
        self.params = params or DEFAULT_PARAMS

    def seed(self, text: str) -> list[str]:
        """Return SRL role-argument pairs as seed phrases.

        Output format: "{readable_role}: {span}  [predicate: {verb}]"
        Falls back to empty list if the LLM call fails.
        """
        if self.model is None:
            logger.warning("SRLSeeder: no model set.")
            return []

        messages = [
            SystemMessage(content=_SRL_SYSTEM),
            HumanMessage(content=_SRL_USER.format(text=text)),
        ]
        try:
            llm = get_structured_llm(self.model, _SRLOutput, self.params)
            output: _SRLOutput = call_with_retry(llm.invoke, messages)
        except Exception as exc:
            logger.warning(f"SRLSeeder LLM call failed: {exc}")
            return []

        if output is None or not output.roles:
            return []

        return _extract_seeds(output.roles)
