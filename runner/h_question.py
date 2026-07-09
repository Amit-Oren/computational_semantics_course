"""
H-Question Pipeline Runner — Hypothesis Interrogation for NLI
=============================================================
Three-stage pipeline:
  Stage 0: Extract keyphrases from H (POS tagging, no LLM)
  Stage 1: H + keyphrases → 1-2 probe questions                      (LLM)
  Stage 2: locate_and_answer per question, then classify_evidence
           over the surviving (question, answer) pairs               (shared, LLM)

Evidence-finding (Locator + Answer Extractor) and final classification are
shared with q2_pipeline, p_question, and h_multihop — see
utils/locator_extractor.py and prompts/shared_classifier.py. Only Stage 1
(question generation from H) is method-specific.

Dataset note: ConTRoL JSONL contains uid/premise/hypothesis/label only —
gold evidence sentence indices are not available, so located_indices vs
gold_evidence_indices comparison cannot be done on this dataset.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from config.config import (
    DEFAULT_PARAMS,
    HQuestionsOutput,
    get_structured_llm,
    logger,
)
from prompts.h_question import (
    H_QUESTION_GEN_SYSTEM_PROMPT,
    H_QUESTION_GEN_USER_PROMPT,
)
from prompts.shared_classifier import classify_evidence
from utils.locator_extractor import locate_and_answer
from utils.pos_keyphrase import extract_keyphrases
from utils.premise_indexer import number_sentences
from utils.retry import call_with_retry

METHOD = "h_question"
FALLBACK_LABEL = "Neutral"


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
        return call_with_retry(llm.invoke, messages)

    # ── Full sample ───────────────────────────────────────────────────────────

    def run_sample(self, sample: dict) -> dict:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings: list[str] = []

        # Stage 0 — keyphrases (no LLM)
        kp_result  = extract_keyphrases(hypothesis)
        keyphrases = kp_result["keyphrases"]

        # Stage 1 — generate questions
        q_output = None
        try:
            q_output = self.stage1_generate_questions(hypothesis, keyphrases)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        if q_output is None or not q_output.questions:
            warnings.append("Stage 1 returned no questions; defaulting to Neutral.")
            return self._make_result(sample, keyphrases, [], FALLBACK_LABEL, warnings)

        questions = q_output.questions

        # Stage 2 — locate + answer each question, then classify the survivors
        indexed_sentences, numbered_premise = number_sentences(premise)
        qa_pairs = []
        for question in questions:
            result = locate_and_answer(
                self.model, self.params, question,
                indexed_sentences=indexed_sentences, numbered_premise=numbered_premise,
            )
            if result["answerable"]:
                qa_pairs.append(result)

        if not qa_pairs:
            warnings.append("All questions unanswerable; defaulting to Neutral.")
            return self._make_result(sample, keyphrases, qa_pairs, FALLBACK_LABEL, warnings)

        prediction = classify_evidence(self.model, self.params, qa_pairs, hypothesis)
        return self._make_result(sample, keyphrases, qa_pairs, prediction, warnings)

    @staticmethod
    def _make_result(
        sample:      dict,
        keyphrases:  list[str],
        qa_pairs:    list[dict],
        prediction:  str,
        warnings:    list[str],
    ) -> dict:
        return {
            "id":         sample.get("id"),
            "premise":    sample["premise"],
            "hypothesis": sample["hypothesis"],
            "gold_label": sample["label"],
            "prediction": prediction,
            "qa_pairs":   qa_pairs,
            "method":     METHOD,
            "keyphrases": keyphrases,
            "warnings":   warnings,
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

        results.append(result)
        warn_str = f" | warnings={result['warnings']}" if result["warnings"] else ""
        logger.info(
            f"[{i+1}/{len(samples)}] id={sample_id} "
            f"| gold={result['gold_label']} | pred={result['prediction']}{warn_str}"
        )

    if results:
        correct  = sum(r["gold_label"] == r["prediction"] for r in results)
        accuracy = correct / len(results)
        logger.info("=" * 60)
        logger.info(f"Processed : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy  : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info("=" * 60)

    return results
