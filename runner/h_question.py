"""
H-Question Pipeline Runner — Hypothesis Interrogation for NLI
=============================================================
Four-stage pipeline:
  Stage 0:  Extract keyphrases from H (POS tagging, no LLM)
  Stage 1:  H + keyphrases → 1-2 probe questions              (LLM)
  Stage 2a: Numbered premise + question → sentence indices     (LLM)
  Stage 2b: Extracted sentences + question → concise answer   (LLM)
  Stage 3:  Answer + claim from H → per-question NLI label    (LLM)

Aggregation rule:
  any "Contradiction"        → Contradiction
  all "Entailment"           → Entailment
  otherwise (neutral / NOT_ANSWERABLE / mix) → Neutral

Dataset note: ConTRoL JSONL contains uid/premise/hypothesis/label only —
gold evidence sentence indices are not available, so located_indices vs
gold_evidence_indices comparison cannot be done on this dataset.
"""

from __future__ import annotations

import time
from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    HQuestionsOutput,
    HLocateOutput,
    HAnswerOutput,
    HCompareOutput,
    get_structured_llm,
    logger,
)
from prompts.h_question import (
    H_QUESTION_GEN_SYSTEM_PROMPT,
    H_QUESTION_GEN_USER_PROMPT,
    H_LOCATE_SYSTEM_PROMPT,
    H_LOCATE_USER_PROMPT,
    H_ANSWER_SYSTEM_PROMPT,
    H_ANSWER_USER_PROMPT,
    H_COMPARE_SYSTEM_PROMPT,
    H_COMPARE_USER_PROMPT,
)
from utils.pos_keyphrase import extract_keyphrases
from utils.premise_indexer import number_sentences, pull_by_indices

_RETRY_WAIT  = 30
_MAX_RETRIES = 5
FALLBACK_LABEL = "Neutral"
NOT_ANSWERABLE = "NOT_ANSWERABLE"


# ── Retry helper (mirrors q2_pipeline / p_question) ───────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_capacity = any(k in msg for k in ("capacity", "rate limit", "429", "503", "overloaded"))
            if is_capacity and attempt < _MAX_RETRIES:
                logger.warning(
                    f"Provider at capacity (attempt {attempt}/{_MAX_RETRIES}) "
                    f"— retrying in {_RETRY_WAIT}s"
                )
                time.sleep(_RETRY_WAIT)
            else:
                raise
    return None


# ── Label aggregation ─────────────────────────────────────────────────────────

def _aggregate_labels(labels: list[str]) -> str:
    if not labels:
        return FALLBACK_LABEL
    if any(l == "Contradiction" for l in labels):
        return "Contradiction"
    if all(l == "Entailment" for l in labels):
        return "Entailment"
    return FALLBACK_LABEL


# ── Pipeline class ────────────────────────────────────────────────────────────

class HQuestionPipeline:
    """Hypothesis-interrogation NLI pipeline."""

    def __init__(self, model: str, params: dict):
        self.model  = model
        self.params = params

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def stage1_generate_questions(
        self, hypothesis: str, keyphrases: list[str]
    ) -> HQuestionsOutput | None:
        kp_str = ", ".join(keyphrases) if keyphrases else hypothesis
        messages = [
            SystemMessage(content=H_QUESTION_GEN_SYSTEM_PROMPT),
            HumanMessage(content=H_QUESTION_GEN_USER_PROMPT.format(
                hypothesis=hypothesis, keyphrases=kp_str,
            )),
        ]
        llm = get_structured_llm(self.model, HQuestionsOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 2a ──────────────────────────────────────────────────────────────

    def stage2a_locate(
        self, numbered_premise: str, question: str
    ) -> HLocateOutput | None:
        messages = [
            SystemMessage(content=H_LOCATE_SYSTEM_PROMPT),
            HumanMessage(content=H_LOCATE_USER_PROMPT.format(
                numbered_premise=numbered_premise, question=question,
            )),
        ]
        llm = get_structured_llm(self.model, HLocateOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 2b ──────────────────────────────────────────────────────────────

    def stage2b_answer(
        self, question: str, sentences: list[str]
    ) -> HAnswerOutput | None:
        sentences_block = "\n".join(f"- {s}" for s in sentences)
        messages = [
            SystemMessage(content=H_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=H_ANSWER_USER_PROMPT.format(
                question=question, sentences_block=sentences_block,
            )),
        ]
        llm = get_structured_llm(self.model, HAnswerOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def stage3_compare(
        self, question: str, answer: str, claim: str
    ) -> HCompareOutput | None:
        messages = [
            SystemMessage(content=H_COMPARE_SYSTEM_PROMPT),
            HumanMessage(content=H_COMPARE_USER_PROMPT.format(
                question=question, answer=answer, claim=claim,
            )),
        ]
        llm = get_structured_llm(self.model, HCompareOutput, self.params)
        return _call_with_retry(llm.invoke, messages)

    # ── Full sample ───────────────────────────────────────────────────────────

    def run_sample(self, sample: dict) -> dict | None:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings: list[str] = []

        # Stage 0 — keyphrases (no LLM)
        kp_result  = extract_keyphrases(hypothesis)
        keyphrases = kp_result["keyphrases"]
        claim_from_H = kp_result["claim_from_H"]

        # Index premise sentences once
        indexed_sentences, numbered_premise = number_sentences(premise)

        # Stage 1 — generate questions
        q_output = None
        try:
            q_output = self.stage1_generate_questions(hypothesis, keyphrases)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        if q_output is None or not q_output.questions:
            warnings.append("Stage 1 returned no questions; defaulting to Neutral.")
            return self._make_result(
                sample, keyphrases, [], [], [], [], [], FALLBACK_LABEL, warnings,
            )

        questions = q_output.questions
        located_indices_list:    list[list[int]] = []
        extracted_sentences_list: list[list[str]] = []
        answers:                 list[str]       = []
        per_question_labels:     list[str]       = []

        for question in questions:
            # Stage 2a — locate
            indices: list[int] = []
            try:
                loc = self.stage2a_locate(numbered_premise, question)
                if loc and loc.indices:
                    indices = loc.indices[:5]
            except Exception as exc:
                logger.warning(f"Stage 2a failed for '{question[:60]}': {exc}")
            located_indices_list.append(indices)

            # Pull sentences
            extracted = pull_by_indices(indexed_sentences, indices) if indices else []
            extracted_sentences_list.append(extracted)

            # Stage 2b — answer
            answer = NOT_ANSWERABLE
            if extracted:
                try:
                    ans = self.stage2b_answer(question, extracted)
                    if ans:
                        answer = ans.answer
                except Exception as exc:
                    logger.warning(f"Stage 2b failed for '{question[:60]}': {exc}")
            answers.append(answer)

            # Stage 3 — compare
            label = FALLBACK_LABEL
            if answer != NOT_ANSWERABLE:
                try:
                    cmp = self.stage3_compare(question, answer, claim_from_H)
                    if cmp:
                        label = cmp.label
                    else:
                        warnings.append(f"Stage 3 returned None for '{question[:60]}'; defaulting to Neutral.")
                except Exception as exc:
                    logger.warning(f"Stage 3 failed for '{question[:60]}': {exc}")
                    warnings.append(f"Stage 3 error: {exc}")
            per_question_labels.append(label)

        final_label = _aggregate_labels(per_question_labels)
        return self._make_result(
            sample, keyphrases, questions, located_indices_list,
            extracted_sentences_list, answers, per_question_labels,
            final_label, warnings,
        )

    @staticmethod
    def _make_result(
        sample:                   dict,
        keyphrases:               list[str],
        gen_questions:            list[str],
        located_indices_list:     list[list[int]],
        extracted_sentences_list: list[list[str]],
        answers:                  list[str],
        per_question_labels:      list[str],
        final_label:              str,
        warnings:                 list[str],
    ) -> dict:
        per_question_details = [
            {
                "question":            gen_questions[i],
                "located_indices":     located_indices_list[i],
                "gold_evidence_indices": None,  # not in ConTRoL JSONL
                "extracted_sentences": extracted_sentences_list[i],
                "answer_from_P":       answers[i],
                "answerable_flag":     answers[i] != "NOT_ANSWERABLE",
                "per_question_label":  per_question_labels[i],
            }
            for i in range(len(gen_questions))
        ]
        return {
            "id":                    sample.get("id"),
            "premise":               sample["premise"],
            "hypothesis":            sample["hypothesis"],
            "label":                 sample["label"],
            "prediction":            final_label,
            "keyphrases":            keyphrases,
            "gen_questions":         gen_questions,
            "per_question_details":  per_question_details,
            "warnings":              warnings,
        }


# ── Public entry point (matches runner contract used by main.py) ──────────────

def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    pipeline = HQuestionPipeline(model, params)

    logger.info("=" * 60)
    logger.info("Experiment : h_question")
    logger.info(f"Model      : {model}")
    logger.info(f"Temperature: {params.get('temperature')}")
    logger.info(f"Max tokens : {params.get('max_tokens')}")
    logger.info(f"Samples    : {len(samples)}")
    logger.info("=" * 60)

    results = []
    skipped = 0

    for i, sample in enumerate(samples):
        sample_id = sample.get("id")
        try:
            result = pipeline.run_sample(sample)
        except Exception as exc:
            logger.error(f"[{i+1}/{len(samples)}] id={sample_id} | unexpected error: {exc}")
            skipped += 1
            continue

        if result is None:
            logger.warning(
                f"[{i+1}/{len(samples)}] id={sample_id} | pipeline returned None, skipping"
            )
            skipped += 1
            continue

        results.append(result)
        warn_str = f" | warnings={result['warnings']}" if result["warnings"] else ""
        logger.info(
            f"[{i+1}/{len(samples)}] id={sample_id} "
            f"| gold={result['label']} | pred={result['prediction']}{warn_str}"
        )

    if results:
        correct  = sum(r["label"] == r["prediction"] for r in results)
        accuracy = correct / len(results)
        logger.info("=" * 60)
        logger.info(f"Processed : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy  : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info("=" * 60)

    return results
