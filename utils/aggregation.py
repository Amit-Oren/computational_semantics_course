"""
Aggregation modes for NLI classification over Q/A evidence pairs.

Three interchangeable modes — only the aggregation strategy changes;
the classifier prompt, schema, and evidence format are shared:

  aggregated     (default) — all Q/A pairs in one block, one LLM call.
                             Classifier sees all evidence simultaneously.
                             Matches the QAGS/FEQA/SummaC paradigm.

  sequential_cot — evidence fed one Q/A at a time; each step receives
                   the running verdict and produces an updated verdict +
                   one-sentence reason. Explicit incremental reasoning
                   chain — closer to a human reading through evidence.
                   Grounded in Khot et al. 2023 (Decomposed Prompting).

  voting         — one LLM call per Q/A pair, majority label wins.
                   Each call is independent — no cross-pair interference.
                   Ties broken by priority: Contradiction > Entailment > Neutral.

Public entry point:
    label = aggregate(mode, model, params, qa_pairs, hypothesis)
"""

from __future__ import annotations

import logging
from collections import Counter

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config.config import ClassifyOutput, get_structured_llm, logger
from prompts.shared_classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    classify_evidence,
    _classify,
    build_evidence_block,
)
from utils.retry import call_with_retry

FALLBACK_LABEL = "Neutral"
LABEL_PRIORITY = ["Contradiction", "Entailment", "Neutral"]
AGGREGATION_MODES = ("aggregated", "sequential_cot", "voting")


# ── Sequential CoT schemas ────────────────────────────────────────────────────

class _StepOutput(BaseModel):
    label: str = Field(description="Updated NLI verdict: Entailment, Contradiction, or Neutral")
    reasoning: str = Field(description="One sentence explaining why this evidence updates the verdict")


_SEQ_SYSTEM = """\
You are a strict NLI classifier reasoning incrementally through evidence.

You receive evidence one piece at a time. For each piece, you update your
running verdict based on all evidence seen so far plus this new piece.

Apply the same strict priority rules:
1. CONTRADICTION — evidence makes the hypothesis impossible or highly unlikely.
2. ENTAILMENT — evidence directly supports or strongly implies the hypothesis.
3. NEUTRAL — evidence genuinely does not address the hypothesis.

A decisive prior verdict (Contradiction or Entailment) should only change if
the new evidence is clearly stronger and contradicts it.

Output JSON only: {"label": "Entailment|Contradiction|Neutral", "reasoning": "one sentence"}\
"""

_SEQ_USER = """\
Hypothesis: "{hypothesis}"

New evidence (step {step}/{total}):
Q: {question}
A: {answer}

Running verdict before this step: {running_verdict}

Given this new evidence, what is the updated verdict?\
"""


def _sequential_cot(
    model: str, params: dict, qa_pairs: list[dict], hypothesis: str
) -> str:
    """Run classifier step-by-step, passing running verdict forward.

    Returns the final label after all Q/A pairs are processed.
    Falls back to FALLBACK_LABEL if all LLM calls fail.
    """
    llm = get_structured_llm(model, _StepOutput, params)
    running_verdict = "None yet"
    total = len(qa_pairs)

    for step, pair in enumerate(qa_pairs, start=1):
        messages = [
            SystemMessage(content=_SEQ_SYSTEM),
            HumanMessage(content=_SEQ_USER.format(
                hypothesis=hypothesis,
                step=step,
                total=total,
                question=pair["question"],
                answer=pair["answer"],
                running_verdict=running_verdict,
            )),
        ]
        try:
            out: _StepOutput | None = call_with_retry(llm.invoke, messages)
        except Exception as exc:
            logger.warning(f"sequential_cot step {step} failed: {exc}")
            continue

        if out is None:
            continue

        label = out.label.strip()
        if label not in ("Entailment", "Contradiction", "Neutral"):
            logger.warning(f"sequential_cot step {step}: unexpected label '{label}', keeping {running_verdict}")
            continue

        logger.debug(f"sequential_cot step {step}/{total}: {label} — {out.reasoning}")
        running_verdict = label

    return running_verdict if running_verdict != "None yet" else FALLBACK_LABEL


def _voting(
    model: str, params: dict, qa_pairs: list[dict], hypothesis: str
) -> str:
    """Run one classifier call per Q/A pair, return majority label.

    Ties are broken by LABEL_PRIORITY: Contradiction > Entailment > Neutral.
    """
    votes: list[str] = []
    for pair in qa_pairs:
        evidence_block = f"Q: {pair['question']}\nA: {pair['answer']}"
        label = _classify(model, params, evidence_block, hypothesis)
        votes.append(label)
        logger.debug(f"voting — '{pair['question'][:60]}…' → {label}")

    if not votes:
        return FALLBACK_LABEL

    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [lbl for lbl in LABEL_PRIORITY if counts.get(lbl, 0) == max_count]
    return winners[0]


# ── Public entry point ────────────────────────────────────────────────────────

def aggregate(
    mode: str,
    model: str,
    params: dict,
    qa_pairs: list[dict],
    hypothesis: str,
) -> str:
    """Classify hypothesis against Q/A evidence using the specified aggregation mode.

    Args:
        mode: one of "aggregated", "sequential_cot", "voting"
        model: LLM model name
        params: LLM params dict (temperature, max_tokens, …)
        qa_pairs: answerable Q/A pairs from locate_and_answer
        hypothesis: the NLI hypothesis string

    Returns:
        label string: "Entailment", "Contradiction", or "Neutral"
    """
    if not qa_pairs:
        return FALLBACK_LABEL

    if mode == "aggregated":
        return classify_evidence(model, params, qa_pairs, hypothesis)

    if mode == "sequential_cot":
        return _sequential_cot(model, params, qa_pairs, hypothesis)

    if mode == "voting":
        return _voting(model, params, qa_pairs, hypothesis)

    raise ValueError(f"Unknown aggregation mode '{mode}'; choose from {AGGREGATION_MODES}")
