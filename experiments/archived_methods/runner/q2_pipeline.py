"""
Q2 Pipeline Runner — Query-Based Factual Verification for NLI
=============================================================
Two-stage pipeline:
  Stage 1 (Question Generator): H → anchors + 2-3 verification questions   (LLM)
  Stage 2 (Evidence + Classify): locate_and_answer per question, then one
                                  classify_evidence call over the surviving
                                  (question, answer) pairs.

Evidence-finding (Locator + Answer Extractor) and final classification are
shared with p_question, h_question, and h_multihop — see
utils/locator_extractor.py and prompts/shared_classifier.py.
"""

from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from config.config import (
    DEFAULT_PARAMS,
    Q2QuestionOutput,
    get_structured_llm,
    logger,
)
from experiments.archived_methods.prompts.q2_pipeline import (
    Q2_QUESTION_SYSTEM_PROMPT,
    Q2_QUESTION_USER_PROMPT,
)
from prompts.shared_classifier import classify_evidence
from utils.locator_extractor import locate_and_answer
from utils.premise_indexer import number_sentences
from utils.retry import call_with_retry

METHOD = "q2_pipeline"
FALLBACK_LABEL = "Neutral"


class Q2Pipeline:
    """Two-stage NLI verifier: question generation → shared evidence + classification."""

    def __init__(self, model: str, params: dict):
        self.model = model
        self.params = params

    def stage1_generate_questions(self, hypothesis: str) -> Q2QuestionOutput | None:
        messages = [
            SystemMessage(content=Q2_QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=Q2_QUESTION_USER_PROMPT.format(hypothesis=hypothesis)),
        ]
        llm = get_structured_llm(self.model, Q2QuestionOutput, self.params)
        return call_with_retry(llm.invoke, messages)

    def run_sample(self, sample: dict) -> dict:
        premise    = sample["premise"]
        hypothesis = sample["hypothesis"]
        warnings: list[str] = []

        # Stage 1 — generate verification questions from H
        q_output = None
        try:
            q_output = self.stage1_generate_questions(hypothesis)
        except Exception as exc:
            logger.warning(f"Stage 1 failed: {exc}")

        anchors   = q_output.anchors if q_output else []
        questions = q_output.questions if q_output else []
        if not questions:
            warnings.append("Stage 1 returned no questions; defaulting to Neutral.")
            return self._make_result(sample, anchors, [], FALLBACK_LABEL, warnings)

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
            return self._make_result(sample, anchors, qa_pairs, FALLBACK_LABEL, warnings)

        prediction = classify_evidence(self.model, self.params, qa_pairs, hypothesis)
        return self._make_result(sample, anchors, qa_pairs, prediction, warnings)

    @staticmethod
    def _make_result(
        sample:      dict,
        anchors:     list[str],
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
            "stage1_anchors": anchors,
            "warnings":       warnings,
        }


# ── Public entry point (matches the runner contract used by main.py) ──────────

def run(samples: list[dict], model: str, params: dict = DEFAULT_PARAMS) -> list[dict]:
    pipeline = Q2Pipeline(model, params)

    logger.info("=" * 60)
    logger.info("Experiment : q2_pipeline")
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
        correct = sum(r["gold_label"] == r["prediction"] for r in results)
        accuracy = correct / len(results)
        logger.info("=" * 60)
        logger.info(f"Processed : {len(results)} samples  |  Skipped: {skipped}")
        logger.info(f"Accuracy  : {correct}/{len(results)} = {accuracy:.4f}")
        logger.info("=" * 60)

    return results
