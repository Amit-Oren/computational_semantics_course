"""Shared Locator + Answer Extractor for evidence-finding.

This is the single evidence-finding implementation used by q2_pipeline,
p_question, and h_question (full locate + answer per question), and by
h_multihop for its locate step only (h_multihop's own hop answerer keeps its
running-context logic and is not routed through here).

    [numbered P + question] → Locator (LLM)        → {indices}
    [located sentences + question] → Extractor (LLM) → {answer, answerable}
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import LocateOutput, AnswerOutput, get_structured_llm, logger
from prompts.shared_answering import (
    LOCATE_SYSTEM_PROMPT,
    LOCATE_USER_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
)
from utils.premise_indexer import number_sentences, pull_by_indices
from utils.retry import call_with_retry

NOT_ANSWERABLE = "NOT_ANSWERABLE"


def _locate(model: str, params: dict, numbered_premise: str, question: str) -> LocateOutput | None:
    messages = [
        SystemMessage(content=LOCATE_SYSTEM_PROMPT),
        HumanMessage(content=LOCATE_USER_PROMPT.format(
            numbered_premise=numbered_premise, question=question,
        )),
    ]
    llm = get_structured_llm(model, LocateOutput, params)
    return call_with_retry(llm.invoke, messages)


def _extract_answer(model: str, params: dict, question: str, sentences: list[str]) -> AnswerOutput | None:
    sentences_block = "\n".join(f"- {s}" for s in sentences)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=ANSWER_USER_PROMPT.format(
            question=question, sentences_block=sentences_block,
        )),
    ]
    llm = get_structured_llm(model, AnswerOutput, params)
    return call_with_retry(llm.invoke, messages)


def locate_and_answer(
    model: str,
    params: dict,
    question: str,
    *,
    premise: str | None = None,
    indexed_sentences: list[tuple[int, str]] | None = None,
    numbered_premise: str | None = None,
) -> dict:
    """Locate up to 5 relevant premise sentences, then answer from them alone.

    Pass either `premise` (sentences get indexed internally) or a precomputed
    `(indexed_sentences, numbered_premise)` pair — callers answering multiple
    questions against the same premise should index once and reuse it.

    Returns {"question": str, "answer": str, "answerable": bool,
    "located_indices": list[int]}.
    """
    if indexed_sentences is None or numbered_premise is None:
        if premise is None:
            raise ValueError("Provide either `premise` or `indexed_sentences` + `numbered_premise`.")
        indexed_sentences, numbered_premise = number_sentences(premise)

    located_indices:   list[int] = []
    locate_reasoning:  str       = ""
    try:
        loc = _locate(model, params, numbered_premise, question)
        if loc and loc.indices:
            located_indices  = loc.indices[:5]
            locate_reasoning = loc.reasoning
    except Exception as exc:
        logger.warning(f"Locator failed for '{question[:60]}': {exc}")

    extracted = pull_by_indices(indexed_sentences, located_indices) if located_indices else []

    answer = NOT_ANSWERABLE
    answerable = False
    if extracted:
        try:
            ans = _extract_answer(model, params, question, extracted)
            if ans:
                answer     = ans.answer
                answerable = ans.answerable
        except Exception as exc:
            logger.warning(f"Answer extractor failed for '{question[:60]}': {exc}")

    return {
        "question":         question,
        "answer":           answer,
        "answerable":       answerable,
        "located_indices":  located_indices,
        "locate_reasoning": locate_reasoning,
    }


def locate_only(
    model: str,
    params: dict,
    query: str,
    *,
    premise: str | None = None,
    indexed_sentences: list[tuple[int, str]] | None = None,
    numbered_premise: str | None = None,
) -> dict:
    """Locate up to 5 relevant premise sentences for `query` — Stage 2 only,
    no Answer Extractor call. Used by retrieve_then_classify to isolate the
    Locator's contribution: `query` there is the hypothesis itself, not a
    generated question.

    Same premise-indexing contract as locate_and_answer: pass either
    `premise` or a precomputed `(indexed_sentences, numbered_premise)` pair.

    Returns {"located_indices": list[int], "located_sentences": list[str]}.
    """
    if indexed_sentences is None or numbered_premise is None:
        if premise is None:
            raise ValueError("Provide either `premise` or `indexed_sentences` + `numbered_premise`.")
        indexed_sentences, numbered_premise = number_sentences(premise)

    located_indices: list[int] = []
    try:
        loc = _locate(model, params, numbered_premise, query)
        if loc and loc.indices:
            located_indices = loc.indices[:5]
    except Exception as exc:
        logger.warning(f"Locator failed for '{query[:60]}': {exc}")

    located_sentences = pull_by_indices(indexed_sentences, located_indices) if located_indices else []

    return {
        "located_indices":   located_indices,
        "located_sentences": located_sentences,
    }
